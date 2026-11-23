# Retrain policy

Production models are the ones the live forecast loads: `models/xgb_dst_1h.ubj`,
`models/gru_dst_1h_seed*.pt` and `models/gru_channels.json`. They are retrained only when a
trigger below fires. A quiet week with flat live skill is not a reason to train. Training
costs time and every look at the test years costs a little of their honesty.

`retrain-gate` checks the triggers and opens an issue labelled `agents:retrain`. `trainer`
works only from such an issue. Nobody else trains production models.

## Triggers

| Id | Trigger | Evidence the gate must quote |
|---|---|---|
| T1 | A merged experiment is ready for production: its issue carries `result:promising` and its code is on `main` behind an option that is off by default. | issue number, merged PR, validation numbers |
| T2 | Live degradation: in the last 14 days of the live log, at least 168 scored hours and GRU skill below 0. | `agents/bin/live_score.py --days 14 --json` |
| T3 | Live storm failure: at least 24 scored storm hours (observed Dst <= -50 nT) in the whole live log and GRU storm skill below 0.10. The test set storm skill is 0.54. | `agents/bin/live_score.py --json` |
| T4 | Fresh holdout failure: `reports/holdout.json` covers at least 500 hours after the training data and GRU skill is below 0.20 (test: 0.45), or at least 24 storm hours and storm skill below 0.25. | `reports/holdout.json` |
| T5 | Technical: the committed models no longer load or no longer give the same forecast with the pinned `requirements-live.txt` (for example after a dependency update). Same configuration, re-saved. | failing command and error |

Promising, for T1, means on validation, against the current production configuration:
overall MSE at least 2% lower, or storm hour MSE at least 3% lower with overall MSE no more
than 1% higher, with paired block bootstrap intervals that exclude zero for the claimed gain.

## Blocks

- Cooldown: no new retrain issue within 7 days of the last merged `trainer` pull request,
  except T5.
- Storm in progress: no retrain issue, no training and no model merge while observed or
  forecast Dst in `docs/live.js` of the deployed site is at or below -50 nT in the last
  12 hours. A model change in the middle of a storm makes the live record unreadable.
- One open `agents:retrain` issue at a time. Add new evidence to it as a comment.
- Test budget: at most 4 test evaluations per calendar month in `agents/state/test-ledger.md`.
  When the budget is used up the trainer waits for the next month.

## What the trainer does

1. Train the candidate exactly as the issue says, with the 1 hour scripts
   (`scripts/train_xgb.py`, `scripts/train_gru.py`) and only options already on `main`.
2. Compare with the current models on validation first:
   `agents/bin/champion_challenger.py --split val`. If neither family passes, stop: comment
   the numbers on the issue, label it `result:no-gain`, close it. The test years are not
   looked at.
3. Only if at least one family passes on validation, run the same comparison once on
   `--split test`, append a line to `agents/state/test-ledger.md` and keep the JSON in
   `reports/retrain/<YYYY-MM-DD>-test.json`.
4. Promote a model family only if it passes on test as well. A family that fails keeps its
   current files. If nothing passes, commit only the reports and the ledger line, so the look
   at test is on record.
5. Before opening the pull request, run `scripts/live_forecast.py` without `--archive` on
   the worktree: the new models must load and give a finite forecast from real-time data.

## Promotion is reviewed

`pr-reviewer` re-runs step 5 and checks the ledger line, the JSON reports and that only the
promoted family changed. `release-merger` merges at most one model pull request per run,
never during a storm, and checks that the live workflow run after the merge succeeds. If it
fails, it reverts the merge.
