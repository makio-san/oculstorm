# Merge log

One line per pull request the release-merger considered. Appended by the merger only.

| Time (UTC) | PR | Branch | Head | Decision | Reason |
|---|---|---|---|---|---|
| 2026-11-23T17:40Z | #14 | agents/tuner-2026-11-23 | cd98c64 | merged | all gates hold; validation only report, tests pass (data not needed) |
| 2026-11-23T17:41Z | #9 | agents/feature-builder-2026-11-23 | a89d34c | merged | all gates hold; full tests pass, live forecast finite, live run 45230397675 succeeded |
| 2026-11-23T17:43Z | #16 | agents/live-monitor-2026-11-23 | b5eb780 | merged | all gates hold; full tests pass, live run 45230735582 succeeded |
| 2026-11-23T17:45Z | #3 | agents/holdout-evaluator-2026-11-23 | 7748b01 | merged | all gates hold; full tests pass |
| 2026-11-23T17:45Z | #18 | agents/experimenter-2026-11-23 | 49f0c85 | merged | all gates hold; option off by default, full tests pass, live forecast unchanged and finite, live run 45231123934 succeeded |
| 2026-11-23T17:47Z | #20 | agents/experimenter-2026-11-23-2 | 27381ab | skipped | limit of 5 merges per run reached |
| 2026-11-23T17:47Z | #19 | agents/experimenter-2026-11-23-3 | 82c24a1 | skipped | limit of 5 merges per run reached |
| 2026-11-23T17:47Z | #17 | agents/feature-builder-2026-11-23-2 | 11470b8 | skipped | limit of 5 merges per run reached |
| 2026-11-24T08:28Z | #20 | agents/experimenter-2026-11-23-2 | 27381ab | skipped | does not merge cleanly into main 8ef7605 (conflicts in features, sequences, train scripts, tests); commented |
| 2026-11-24T08:29Z | #19 | agents/experimenter-2026-11-23-3 | 82c24a1 | merged | all gates hold; validation only, full tests pass, live run 45356493095 succeeded |
| 2026-11-24T08:30Z | #17 | agents/feature-builder-2026-11-23-2 | 11470b8 | skipped | does not merge cleanly into main 2f9b07a (conflicts in docs/index.html, scripts/live_forecast.py); commented |
| 2026-11-24T08:31Z | #26 | agents/experimenter-2026-11-24 | 1056a86 | merged | all gates hold; validation only, full tests pass, live forecast unchanged and finite, live run 45356857560 succeeded |
| 2026-11-24T08:33Z | #22 | agents/dashboard-qa-2026-11-24 | f0e356b | merged | all gates hold; tests pass, live run 45357198194 succeeded |
| 2026-11-24T08:36Z | #21 | agents/data-watch-2026-11-24 | 1d7d03c | merged | all gates hold; report only, tests pass |
| 2026-11-24T17:50Z | #20 | agents/experimenter-2026-11-23-2 | 93f01ff | merged | all gates hold; validation only, full tests pass, live forecast finite, live run 45452273427 succeeded |
| 2026-11-24T17:50Z | #17 | agents/feature-builder-2026-11-23-2 | b8ed2ef | merged | all gates hold; full tests pass, live forecast finite, live run 45452729895 succeeded |
| 2026-11-24T17:50Z | #28 | agents/experimenter-2026-11-24 | 7a717f7 | skipped | conflicts with main in scripts/train_xgb.py after #20; commented |
| 2026-11-24T17:50Z | #29 | agents/experimenter-2026-11-24-2 | 257a235 | skipped | conflicts with main in scripts/train_xgb.py after #20; commented |
| 2026-11-25T17:27Z | #29 | agents/experimenter-2026-11-24-2 | efd078a | merged | all gates hold; validation only, full tests pass, live forecast finite, live run 45664931625 succeeded |
| 2026-11-25T17:32Z | #28 | agents/experimenter-2026-11-24 | 2e6ec37 | skipped | conflicts with main in scripts/train_xgb.py after #29; commented |
| 2026-11-25T17:32Z | #33 | agents/experimenter-2026-11-25 | 08e2c01 | skipped | conflicts with main in oculstorm/augment.py, scripts/train_xgb.py and tests/test_augment.py after #29; commented |
| 2026-11-26T08:15Z | #33 | agents/experimenter-2026-11-25 | 08e2c01 | skipped | still conflicts with main in oculstorm/augment.py, scripts/train_xgb.py and tests/test_augment.py; head unchanged since the comment of 2026-11-25 |
| 2026-11-26T08:16Z | #28 | agents/experimenter-2026-11-24 | ea572ff | merged | all gates hold; validation only, full tests pass, live forecast finite |
| 2026-11-26T08:17Z | #40 | agents/experimenter-2026-11-26 | e53c631 | skipped | conflicts with main in scripts/train_xgb.py after #28; commented |
| 2026-11-26T08:17Z | #37 | agents/experimenter-2026-11-26-2 | a3a280d | merged | all gates hold; validation only, full tests pass, live forecast finite, live run 45781924087 succeeded |
| 2026-11-26T08:19Z | #38 | agents/experimenter-2026-11-26-3 | d8a7ffa | merged | all gates hold; validation only, full tests pass, live forecast finite, live run 45782228261 succeeded |
| 2026-11-26T08:21Z | #39 | agents/holdout-evaluator-2026-11-26 | 1aef939 | merged | all gates hold; full tests pass |
| 2026-11-26T08:21Z | #35 | agents/data-watch-2026-11-26 | 313e247 | merged | all gates hold; report only, full tests pass |
| 2026-11-26T08:22Z | #36 | agents/feature-builder-2026-11-26 | 2eb4a53 | skipped | limit of 5 merges per run reached |
| 2026-11-26T16:56Z | #33 | agents/experimenter-2026-11-25 | 3730893 | merged | all gates hold; validation only, full tests pass, live forecast finite, live run 45862644393 succeeded |
| 2026-11-26T16:56Z | #36 | agents/feature-builder-2026-11-26 | 2eb4a53 | merged | all gates hold; full tests pass, live forecast finite, live run 45862954586 succeeded |
| 2026-11-26T16:56Z | #40 | agents/experimenter-2026-11-26 | e53c631 | skipped | conflicts with main in oculstorm/augment.py and scripts/train_xgb.py after #33; commented |
