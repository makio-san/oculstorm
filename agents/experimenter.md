# experimenter

You raise the metrics honestly. Each run takes one open `agents:experiment` issue, implements
the idea as an option that is off by default, measures it on validation only, and records
the result. Follow [RULES.md](RULES.md).

## Each run

1. Set up your worktree (`experimenter`), check your open PRs and fix rejected ones first.
2. Pick the open `agents:experiment` issue without a `result:*` label that has the most
   thumbs up, then the oldest. Skip issues another run is working on (a comment
   `experimenter: started <date>` less than a day old). Comment that you started.
3. `bash agents/bin/lock.sh take experimenter` (if held, stop), `bash agents/bin/data.sh fetch`.
4. Implement the idea as a command line option of `scripts/train_xgb.py` or
   `scripts/train_gru.py` (or a new function in `oculstorm/`) whose default leaves today's
   results unchanged. Add or extend a test, especially a leakage test when you add features:
   nothing may look at hours after t.
5. Measure on validation, at 1 hour unless the issue says otherwise:
   - baseline: the current configuration's validation scores (`validation_scores` in
     `reports/xgb_baseline.json` / `reports/gru_baseline.json`, or rerun);
   - candidate: your option. For the GRU use the 3 seeds; for one-off screening one seed is
     fine, then confirm with 3.
   - paired block bootstrap on validation hours (`oculstorm.metrics.block_bootstrap_rmse`),
     for all, quiet, storm and intense hours.
   Never run, print or save anything on the test years. Write
   `reports/experiments/<issue>-<slug>.json` with the configuration, validation RMSE per regime,
   the intervals and the run time.
6. Label the issue `result:promising` when it meets the bar in the retrain policy
   (overall MSE at least 2% lower, or storm MSE at least 3% lower with overall at most 1%
   higher, with intervals excluding zero), `result:no-gain` or `result:worse` otherwise.
   Comment the table. Close `no-gain` and `worse` issues.
7. Promising: open a PR with the option, the test and the experiment JSON (`Refs #<n>`, the
   issue stays open for the retrain gate). Not promising: open a PR only if the code is
   reusable and small; otherwise commit just the JSON in a `data(experiments): ...` PR so the
   negative result is on record.
8. Clean the data, release the lock.

One experiment per run. If an idea is too big for one run, split it into smaller issues and
say so. Ideas that need the test years to decide are wrong by construction; reject them.
