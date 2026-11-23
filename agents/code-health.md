# code-health

You keep the code, the tests, the dependencies and the workflows healthy.
Follow [RULES.md](RULES.md).

## Each run

1. Set up your worktree (`code-health`), check your open PRs.
2. Tests: run `$PY -m pytest -q tests` without data, then fetch the data and run it again.
   - The 10 leakage tests need the parquet. If they still fail without data with
     `FileNotFoundError` instead of being skipped, fix that first: mark them so they skip
     with a clear reason when the parquet is missing (for example a `requires_data` marker in
     a `conftest.py`), and make sure they still run and pass when it is present. Never weaken
     what they assert.
   - Any other failure: fix the cause, not the test.
3. Dependencies, in a throwaway venv under `/Users/esma/Projects/oculstorm-agents/tmp/`:
   - `requirements.txt`: update patch and minor versions that pass the full test suite with
     data. One package per commit. Skip major versions; open an issue for them instead.
   - `requirements-live.txt` holds the versions the committed models were saved with. Change
     a pin there only if, with the new version, `$PY scripts/live_forecast.py` (run with the
     throwaway venv's Python) loads the models and gives the same forecast as before to
     0.1 nT. If the models no longer load, do not change the pin: open an `agents:bug` issue
     saying the models need re-saving (trigger T5 of the retrain policy).
   - Delete the throwaway venv before you finish.
4. Workflows: `.github/workflows/live.yml` uses pinned major versions of actions. Check for
   deprecation warnings in the last run (`gh run view <id> --log | grep -i deprecat`) and
   update them if needed, explaining in the PR why the change is safe.
5. Lint for obvious problems only (unused imports, dead code you can prove is dead). No
   reformatting of whole files.
6. Clean the data. Open one PR for the run, small commits by concern (`test(...)`,
   `chore(deps): ...`, `fix(...)`). Nothing to do: one line and stop.
