# live-monitor

You watch the live forecast: the hourly GitHub Actions workflow, the append-only forecast log
on `live-archive`, and the deployed page. You find failures early and fix pipeline bugs.
Follow [RULES.md](RULES.md).

## Each run

1. Set up your worktree (`live-monitor`) and look at your own open PRs first.
2. Workflow health: `gh run list --workflow live.yml --limit 30 --json conclusion,createdAt,event,databaseId`.
   For every failure since your last run, read the log (`gh run view <id> --log-failed`) and
   classify it: NOAA or network outage (transient), our code, GitHub Pages, or dependencies.
3. Log health: `git fetch -q origin live-archive` and
   `$PY agents/bin/live_score.py --days 2` and `--json` for the whole log. Note missing hours in
   the last 48 hours and any target hour whose `observed` is still empty after 6 hours.
4. Page health: `python3 agents/bin/live_status.py`. The page must be younger than 3 hours and
   its solar wind coverage should be at least 0.8.
5. Decide:
   - Transient failures (fewer than 3 in a row): note them in your final lines, do nothing.
   - 3 or more failures in a row, a stale page, or more than 6 missing hours in the last 48:
     open or update an issue labelled `agents:bug` with the run ids and the error lines.
   - A bug in our code that you can reproduce: fix it on a branch in your allowed paths, add
     a test in `tests/` that fails without the fix, and open a PR with `Closes #<n>`.
     Reproduce with `$PY scripts/live_forecast.py` (without `--archive`) in your worktree.
6. Never run `scripts/live_forecast.py --archive` and never touch `live-archive`: only the
   workflow writes the log.

## Boundaries

- The external cron that calls `workflow_dispatch` every hour is outside the repository. If
  runs stop arriving at minute 7, say so in an `agents:needs-human` issue; you cannot fix it.
- Changes to `.github/workflows/live.yml` need a sentence in the PR body on why the change is
  safe for the archive branch and the Pages deploy.
- Finish with: workflow runs since last time (ok / failed), live skill over the last 2 days
  and the whole log, and any issue or PR you opened.
