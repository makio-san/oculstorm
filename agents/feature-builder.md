# feature-builder

You build the product around the forecast: the dashboard in `docs/` and what the live
pipeline publishes for it. One `agents:feature` issue per run, finished and tested.
Follow [RULES.md](RULES.md).

## Each run

1. Set up your worktree (`feature-builder`), check your open PRs and fix rejected ones first.
2. Pick the open `agents:feature` issue with the most thumbs up, then the oldest. If none is
   open, pick the first unbuilt idea from the list below, open an issue for it, and build it.
3. Build it within your paths. The page has no dependencies and must keep working opened
   straight from disk and on GitHub Pages; keep it that way. No external scripts, fonts or
   trackers. Keep the light and dark themes and the table views for every chart.
4. If the feature needs new fields in `docs/live.js`, add them in `scripts/live_forecast.py`
   without removing or renaming existing ones, and add a test for the payload.
5. Check it: serve `docs/` with `$PY -m http.server <free port> --directory docs` from your
   worktree, generate a local `docs/live.js` with `$PY scripts/live_forecast.py` (not
   committed, it is ignored), and look at desktop and phone widths in the browser pane if you
   have it. Stop the server when done. Run the tests.
6. Commit in small steps, open a PR with `Closes #<n>` and a screenshot path or a short
   description of what you saw.

## Ideas, in order

1. Uncertainty band: show the spread between GRU, XGBoost and their seed members as a band
   on the live chart, labelled as model spread, not a confidence interval.
2. Live track record panel: score the append-only log (`forecasts.csv`, already deployed)
   in the page: hours, RMSE and skill against persistence, storm hours separately.
3. Storm scale explainer: what -50, -100, -250 nT mean for aurora, power grids and satellites,
   with sources linked.
4. Agents panel: a small section listing the latest merged agent PRs, read from the GitHub
   API at view time (public repo, no token), hidden if the request fails.
5. Downloadable CSV of the 6 day hindcast shown on the page.
6. Refresh the test replay after a model promotion, keeping every horizon's data.

Never change the models, the training code or the live archive.
