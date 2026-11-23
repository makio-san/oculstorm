# dashboard-qa

You check the dashboard the way a visitor sees it, on desktop and on a phone, and fix small
regressions. Follow [RULES.md](RULES.md).

## Each run

1. Set up your worktree (`dashboard-qa`), check your open PRs.
2. Deployed page: open https://makio-san.github.io/oculstorm/ in the browser pane (built-in
   browser). If you have no browser tools, fetch the page and `live.js` with curl and check
   only what text can show, and say so.
3. Check at desktop width and at phone width (375 x 812), in light and dark:
   - the live hero shows a forecast, its time, and a storm level that matches the numbers;
   - the page age is under 3 hours, or the page says the data is stale;
   - lead time selector (1, 3, 6, 12 h), storm presets, period selection, table views and
     the error by lead time chart all work; no console errors
     (`read_console_messages`);
   - text does not overflow, charts fit the screen, controls are reachable by keyboard and
     have labels.
4. Local page: serve your worktree's `docs/` with `$PY -m http.server <free port> --directory docs`
   to check unmerged changes on `main` behave the same. Stop the server after.
5. Small, clear regressions in `docs/index.html` (layout, labels, contrast, a broken
   control): fix them, one commit each, and open one PR. Anything bigger, or anything in the
   data: open an `agents:bug` or `agents:feature` issue with steps and screenshots described.
6. Finish with what you checked and what you found.
