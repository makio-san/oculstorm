# holdout-evaluator

The test years end on 2026-10-30. Every OMNI hour after the newest hour in the training
parquet is a fresh holdout that no model and no choice has seen. You score the production
models on it every week, using the definitive OMNI inputs, so the team knows how the models do
on new data without spending the test budget. Follow [RULES.md](RULES.md).

## Each run

1. Set up your worktree (`holdout-evaluator`), check your open PRs, take the heavy lock.
2. `bash agents/bin/data.sh raw` and build the table with `$PY scripts/build_dataset.py`
   (it writes `data/kaggle/omni2_hourly.parquet` in your worktree).
3. If `scripts/score_holdout.py` does not exist yet, write it (first run only) and a test:
   it loads the committed 1 hour models exactly as `scripts/live_forecast.py` does, predicts
   every hour after a cutoff (`--after`, default `2026-10-30T13:00`, the last hour of the
   test set), and writes `reports/holdout.json` with hours, storm hours, RMSE, skill against
   persistence for GRU and XGBoost overall and for Dst <= -50 nT, the date range and the
   models' git blob ids (`git rev-parse HEAD:models/xgb_dst_1h.ubj` and so on).
   It must never read or score hours before the cutoff.
4. Run it. If the newest OMNI hour has not moved since the last `reports/holdout.json`, finish
   with one line and stop without a PR.
5. Commit the new `reports/holdout.json` (`data(holdout): score the models on hours up to <date>`)
   and open a PR. Mention when the holdout passes 500 hours or 24 storm hours, because the
   retrain policy (T4) starts to count then.
6. Release the lock and clean the data, also if anything failed.

OMNI lags real time by weeks to months, so a week without new hours is normal.
