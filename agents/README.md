# oculstorm agents

Seventeen scheduled agents watch the live forecast, improve the models, retrain only when the
evidence says so, build features, keep the code healthy, and ship their work through pull
requests that a reviewer checks and a merger lands on `main`. Each agent is one file here;
the rules they all share are in [RULES.md](RULES.md), retraining is governed by
[retrain-policy.md](retrain-policy.md).

## The agents

| Agent | Schedule (Istanbul time) | Job | Output |
|---|---|---|---|
| [live-monitor](live-monitor.md) | daily 08:20, 20:20 | Health of the hourly live workflow, the forecast log and the deployed page; fixes pipeline bugs | issues `agents:bug`, fix PRs |
| [storm-alert](storm-alert.md) | every 2 hours at :25 | Watches the deployed forecast for storm levels and notifies the owner | notification, issue `agents:storm` |
| [storm-reporter](storm-reporter.md) | daily 09:40 | Writes a post-mortem for every storm the live log has seen | `agents/reports/storms/*.md` PR |
| [holdout-evaluator](holdout-evaluator.md) | Wednesday 11:30 | Scores the production models on OMNI hours newer than the training data | `reports/holdout.json` PR |
| [retrain-gate](retrain-gate.md) | daily 10:30 | Checks the triggers in the retrain policy; decides, never trains | issue `agents:retrain` or nothing |
| [trainer](trainer.md) | daily 12:10 | Works an open retrain issue: train, validate, one look at test, promote or not | model PR |
| [experimenter](experimenter.md) | daily 13:30, 18:30 | Runs one experiment issue per run, on validation only | issue result, code PR behind an option |
| [tuner](tuner.md) | Saturday 12:30 | Small hyperparameter searches on validation | experiment issues with evidence |
| [research-scout](research-scout.md) | Tuesday, Friday 09:10 | Reads recent Dst and solar wind forecasting work and files ideas | issues `agents:experiment` |
| [data-watch](data-watch.md) | Thursday 10:10 | NOAA feed formats, Kyoto quick-look revisions, OMNI and Kaggle freshness | issues, fix PRs |
| [feature-builder](feature-builder.md) | Monday, Thursday 15:30 | Builds one dashboard or product feature issue per run | feature PR |
| [dashboard-qa](dashboard-qa.md) | Tuesday 16:10 | Desktop and phone checks of the deployed and the local page | fix PRs, issues |
| [code-health](code-health.md) | Sunday 11:10 | Tests, test hygiene, safe dependency updates, workflow health | `chore`, `test`, `fix` PRs |
| [docs-keeper](docs-keeper.md) | Saturday 17:10 | Keeps the README numbers equal to the tracked reports | `docs` PRs |
| [pr-reviewer](pr-reviewer.md) | daily 11:15, 16:15, 21:15 | Reviews every open agent PR and approves or rejects its exact head commit | PR comments and labels |
| [release-merger](release-merger.md) | daily 11:50, 16:50, 21:50 | Merges approved PRs into `main` after its own checks, verifies the deploy | `main`, `agents/reports/merge-log.md` |
| [weekly-digest](weekly-digest.md) | Monday 09:00 | One page on the week: live skill, merges, experiments, open questions | issue `agents:digest` |

## Flow

1. `research-scout` and `tuner` file ideas as `agents:experiment` issues.
2. `experimenter` runs them on validation, labels the result and opens a code PR for
   promising ones, with the new option off by default.
3. `live-monitor`, `holdout-evaluator` and `storm-reporter` measure how the production models
   really do. `retrain-gate` reads their evidence and the merged experiments, and opens an
   `agents:retrain` issue only when a trigger in the policy fires.
4. `trainer` trains, looks at test once, and opens a model PR only for what passes.
5. `feature-builder`, `dashboard-qa`, `data-watch`, `code-health` and `docs-keeper` improve
   the product and keep it working.
6. `pr-reviewer` approves or rejects each PR for its exact head commit.
7. `release-merger` merges approved PRs, pushes `main`, and the live workflow deploys.

## Paths each agent may change

The reviewer rejects and the merger skips any PR that touches a path outside its agent's list.
No agent may change `agents/*.md`, `agents/bin/**`, `.gitignore`, or the `live-archive` branch.

| Agent | Paths |
|---|---|
| live-monitor | `oculstorm/live.py`, `oculstorm/archive.py`, `scripts/live_forecast.py`, `.github/workflows/live.yml`, `tests/**`, `agents/reports/**` |
| storm-reporter | `agents/reports/storms/**` |
| holdout-evaluator | `reports/holdout.json`, `scripts/score_holdout.py`, `tests/**`, `agents/reports/**` |
| trainer | `models/**`, `reports/**`, `agents/state/test-ledger.md`, `agents/reports/**`, and default values in `scripts/train_*.py` named in its retrain issue |
| experimenter, tuner | `oculstorm/**` except `live.py` and `archive.py`, `scripts/train_*.py`, `scripts/compare_models.py`, `tests/**`, `reports/experiments/**`, `agents/reports/**` |
| data-watch | `oculstorm/live.py`, `scripts/build_dataset.py`, `data/kaggle/*.txt`, `data/kaggle/*.json`, `tests/**`, `agents/reports/**` |
| feature-builder | `docs/**`, `scripts/build_dashboard.py`, `scripts/live_forecast.py`, `tests/**`, `agents/reports/**` |
| dashboard-qa | `docs/index.html`, `agents/reports/**` |
| code-health | `requirements.txt`, `requirements-live.txt`, `oculstorm/**`, `scripts/**`, `tests/**`, `conftest.py`, `pytest.ini`, `pyproject.toml`, `.github/workflows/**`, `agents/reports/**` |
| docs-keeper | `README.md`, `data/kaggle/*.md`, `agents/reports/**` |
| release-merger | `agents/reports/merge-log.md`, revert commits |
| retrain-gate, research-scout, pr-reviewer, weekly-digest, storm-alert | no files; issues and comments only |

## Gates before anything reaches `main`

A PR is merged only when `pr-reviewer` has approved its current head commit, all commits are
authored by `svarts <killbillbeatrixx@gmail.com>` with Conventional Commit subjects and no AI
attribution, the paths are allowed for its agent, it merges cleanly, and the tests pass on the
merged result. Model PRs also need the checks in the retrain policy. After every merge that
changes the deployed site or the models, the merger watches the live workflow run and reverts
the merge if it fails.

## Workspace

Agents never work in the owner's checkout. `/Users/esma/Projects/oculstorm-agents/` holds a
clone (`base/`) and one git worktree per agent (`work/<agent>/`), plus the heavy job lock.
Downloaded data lives only for the length of one run.

## Turning things off

Scheduled tasks run in the Claude desktop app while it is open, and catch up on next launch.
Disable `oculstorm-release-merger` to stop anything reaching `main`: the other agents then
only open PRs. Disable `oculstorm-trainer` to freeze the production models.
