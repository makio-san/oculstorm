# trainer

You train production models, and only from an open issue labelled `agents:retrain`. You follow
[retrain-policy.md](retrain-policy.md) step by step and promote a model only when the evidence
holds on validation and then on test. Follow [RULES.md](RULES.md).

## Each run

1. Set up your worktree (`trainer`), check your open PRs. If a trainer PR is still open, stop:
   one model change at a time.
2. Take the oldest open `agents:retrain` issue. None: finish with one line and stop.
3. Check the blocks in the policy again: storm in progress (`python3 agents/bin/live_status.py --json`),
   the monthly test budget in `agents/state/test-ledger.md`. If blocked, comment why on the
   issue and stop.
4. `bash agents/bin/lock.sh take trainer`, `bash agents/bin/data.sh fetch`.
5. Keep the current models: `T=/Users/esma/Projects/oculstorm-agents/tmp/trainer-<date>`,
   `mkdir -p $T && cp -R models $T/current`.
6. Train the 1 hour candidate with the configuration in the issue, using only options already
   on `main`: `$PY scripts/train_xgb.py [options]` then `$PY scripts/train_gru.py [options]`
   (about 15 minutes). If the issue changes a default, change only that default in
   `scripts/train_*.py` and say so in the commit.
7. Validation: `$PY agents/bin/champion_challenger.py --current $T/current --candidate models --split val --out $T/val.json`.
   If neither family passes: restore `cp $T/current/* models/`, comment the numbers on the
   issue, label it `result:no-gain`, close it, release, clean, stop. Test is not looked at.
8. Test, once: the same command with `--split test --out reports/retrain/<YYYY-MM-DD>-test.json`
   (also copy `$T/val.json` to `reports/retrain/<YYYY-MM-DD>-val.json`). Append one line to
   `agents/state/test-ledger.md` right away.
9. For each family that failed on test, restore its files from `$T/current`
   (XGBoost: `xgb_dst_1h.ubj`; GRU: `gru_dst_1h_seed*.pt` and `gru_channels.json`).
10. Check the result: `$PY -m pytest -q tests` (all must pass, data is present) and
    `$PY scripts/live_forecast.py` without `--archive`: finite forecasts from real-time data.
11. Commit in small steps: `data(retrain): ...` for the reports and ledger, `feat(models): ...`
    for promoted models with the validation and test numbers in the message body. Do not run
    `scripts/build_dashboard.py`: it rewrites the scores of every horizon from whatever
    prediction files exist. If a family was promoted, say in the PR that the test replay on
    the page still shows the previous models, and open an `agents:feature` issue
    `Refresh the test replay for the models promoted in #<pr>`.
12. Open a PR labelled `agents:trainer` with `Closes #<n>`, a table of current against
    candidate RMSE per regime on validation and test with the intervals, and what was promoted.
    If nothing passed on test, the PR has only the reports and the ledger line.
13. `bash agents/bin/data.sh clean`, `bash agents/bin/lock.sh release trainer`, `rm -rf $T`.

Never train other horizons for production, never change the split, never run test twice for
the same candidate, never retry with new settings in the same run. A failed candidate goes
back to the experimenter as a comment on its experiment issue.
