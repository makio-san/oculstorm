# oculstorm: real-time forecast log

`forecasts.csv` is written by the hourly workflow on `main` (`.github/workflows/live.yml`).

- One row per target hour (UTC, start of the hour).
- `issued`: when the forecast was made. A row is written once and never changed.
- `gru`, `xgb`, `persistence`: forecast Dst for the target hour, in nT.
- `observed`: WDC Kyoto quick-look Dst, filled in by a later run (and updated if Kyoto revises it).

Hours when the workflow did not run are missing; they are not backfilled.
