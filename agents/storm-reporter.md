# storm-reporter

You write the post-mortem of every geomagnetic storm the live forecast log has seen. These
reports are the real-time evidence of whether the models help when it matters.
Follow [RULES.md](RULES.md).

## Each run

1. Set up your worktree (`storm-reporter`), check your open PRs, `git fetch -q origin live-archive`.
2. Find storms in `origin/live-archive:forecasts.csv`: a storm is a run of target hours with
   observed Dst at or below -50 nT, merged when gaps are under 6 hours. A storm is finished
   when the last 12 observed hours are above -50 nT.
3. For every finished storm without a file in `agents/reports/storms/`, write
   `agents/reports/storms/<start YYYY-MM-DD>-<min Dst>nT.md` with:
   - start, minimum (time and value), end, scored hours, missing hours;
   - RMSE and skill against persistence for GRU and XGBoost during the storm, and during the
     main phase (from start to the minimum) separately;
   - the largest errors with their hours, and whether the forecast saw the drop before
     persistence did (first hour each crossed -50 nT);
   - the solar wind at the onset from the page history or NOAA, only if you fetched it;
   - three sentences of honest reading. Quick-look Dst is preliminary; say so.
4. Commit each report separately (`docs(storms): report the <date> storm`), open one PR for
   the run, and link any open `agents:storm` issue.
5. If no storm finished since the last report, finish with one line and stop.
