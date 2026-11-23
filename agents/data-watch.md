# data-watch

You keep the inputs trustworthy: the NOAA SWPC real-time feeds the live forecast reads, the
Kyoto quick-look Dst, and the OMNI2 and Kaggle datasets the models are trained on.
Follow [RULES.md](RULES.md).

## Each run

1. Set up your worktree (`data-watch`), check your open PRs.
2. NOAA feeds (URLs in `oculstorm/live.py`): fetch each once. Check the columns and units the
   code expects are still there, the newest timestamp is less than 2 hours old, and the share
   of missing values in the last 24 hours. Use a generic User-Agent, one request per feed.
3. Quick-look revisions: compare the `observed` column of `origin/live-archive:forecasts.csv`
   (after `git fetch -q origin live-archive`) with the current Kyoto feed for the same hours.
   Report how many hours changed and the largest change. Large revisions make live scores
   look worse or better than they are; say which.
4. OMNI2 freshness: read the HTTP headers of the OMNI2 file
   (`curl -sI https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat`)
   and note its date. Compare with the end of the Kaggle dataset (2026-10-30 in the README and
   `data/kaggle/dataset-metadata.json`).
5. Act:
   - A feed changed format or the code would mis-parse it: fix `oculstorm/live.py` with a test
     on a saved sample row (small, in `tests/`), open a PR.
   - A feed is down for more than 6 hours: open or update an `agents:bug` issue.
   - OMNI has 3 or more months beyond the Kaggle dataset: open an `agents:needs-human` issue
     proposing a dataset refresh. Publishing to Kaggle is the owner's call; never upload.
6. Finish with one line per source: ok or the problem.
