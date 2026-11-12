# oculstorm

Forecasting geomagnetic storm intensity (the Dst index) one hour ahead from recent solar wind measurements.

## Plan

1. **Data:** NASA OMNI2 hourly solar wind, IMF and geomagnetic indices (1963-2026), cleaned into a single Parquet table and hosted on Kaggle.
2. **Models:** XGBoost baseline first, then a small GRU.
3. **Evaluation:** every model must beat the persistence baseline ("next hour = current hour").
4. **Product:** a space-weather dashboard showing observed vs. predicted storm intensity.

## Data

Source: [NASA SPDF OMNIWeb](https://omniweb.gsfc.nasa.gov/), low-resolution OMNI2 file
`omni2_all_years.dat`. Raw files and the Parquet output are not tracked in git; the
Parquet table is published as a Kaggle dataset:
[makiosann/omni2-hourly-dst](https://www.kaggle.com/datasets/makiosann/omni2-hourly-dst)
(559,527 hourly rows × 56 columns, 1963-01-01 to 2026-10-30).

```bash
.venv/bin/kaggle datasets download -d makiosann/omni2-hourly-dst -p data/kaggle --unzip
```

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
curl -o data/raw/omni2_all_years.dat https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat
```

```bash
.venv/bin/python scripts/build_dataset.py
```

The script names the 55 columns, replaces OMNI fill values with nulls, adds a UTC
`time` column and drops the trailing fill rows after the last available Dst value.

## Baseline results (1 hour ahead)

```bash
.venv/bin/python scripts/train_xgb.py
```

Train 1995-2016, validation 2017-2020 (early stopping), test 2021-01 to 2026-10
(51,086 hours, never seen during training, includes the May 2024 superstorm).
The model predicts the change Dst(t+1) - Dst(t) from the last 6 hours of solar wind,
IMF, coupling functions and Dst. Skill is the MSE skill score against persistence
(0 = same as persistence, 1 = perfect).

| Test subset | Hours | Persistence RMSE | XGBoost RMSE | Skill |
|---|---|---|---|---|
| All | 51,086 | 4.81 nT | 3.67 nT | 0.42 |
| Quiet (Dst > -50) | 48,962 | 4.12 nT | 3.24 nT | 0.38 |
| Storm (Dst <= -50) | 2,124 | 12.86 nT | 9.12 nT | 0.50 |
| Intense (Dst <= -100) | 322 | 23.75 nT | 17.97 nT | 0.43 |

- The 95% interval of the overall skill (monthly block bootstrap) is 0.38 to 0.45, and
  skill is positive in every test year (0.32 to 0.45).
- An XGBoost model given only the Dst history reaches a skill of 0.14, so most of the
  gain comes from the solar wind inputs.
- Linear extrapolation of the last hourly change is worse than persistence (skill -0.42).

Caveats: OMNI is a cleaned, time-shifted science product and the Dst values are
final/provisional Kyoto values, so real-time performance on raw L1 data will be lower.
The intense-storm subset is small (322 hours), so its interval is wide (0.25 to 0.59).

## GRU results (1 hour ahead)

```bash
.venv/bin/python scripts/train_gru.py
```

A single layer GRU (hidden size 64, 18,241 parameters) reads the last 24 hours of
18 channels (Dst, solar wind, IMF, coupling functions, observed/missing masks, calendar)
and predicts the same Dst change as XGBoost. Same split and same 51,086 test hours.
Three seeds were trained; the headline GRU is their average.

| Test subset | Persistence RMSE | XGBoost RMSE | GRU RMSE | GRU skill |
|---|---|---|---|---|
| All | 4.81 nT | 3.67 nT | 3.58 nT | 0.45 |
| Quiet (Dst > -50) | 4.12 nT | 3.24 nT | 3.18 nT | 0.41 |
| Storm (Dst <= -50) | 12.86 nT | 9.12 nT | 8.73 nT | 0.54 |
| Intense (Dst <= -100) | 23.75 nT | 17.97 nT | 17.18 nT | 0.48 |

- The GRU beats persistence in every regime and every test year (skill 0.36 to 0.49).
- Against XGBoost the gain is small but consistent: 4.8% lower MSE overall
  (95% interval 3.0% to 7.4%, paired monthly block bootstrap) and better in all six test years.
- For intense storms the GRU vs XGBoost difference is not statistically clear
  (interval -6.5% to +23%), so the two models should be treated as equal there.
- Single seeds score 3.61 to 3.64 nT, still ahead of XGBoost; averaging the three seeds
  gives 3.58 nT, and averaging GRU with XGBoost gives 3.57 nT.

## Dashboard

Live: https://makio-san.github.io/oculstorm/

A static page in `docs/` shows observed Dst against the GRU, XGBoost and persistence
forecasts for any period of the test years, with the largest storms as presets, error
tiles for the selected period, a table view of every chart and the full scorecard.

```bash
.venv/bin/python scripts/build_dashboard.py
```

```bash
.venv/bin/python -m http.server 8765 --directory docs
```

Then open http://localhost:8765. The page has no dependencies, so `docs/index.html`
also opens straight from disk and the folder can be served by GitHub Pages.
It replays the held out test set; it is not connected to a live data feed.

Full numbers are in `reports/xgb_baseline.json` and `reports/gru_baseline.json`.
Run the leakage tests with:

```bash
.venv/bin/python -m pytest tests
```

## Layout

```
oculstorm/data.py          load the hourly table, chronological split
oculstorm/features.py      backward-looking lag and coupling features
oculstorm/metrics.py       persistence baselines, RMSE / MAE / skill by regime
oculstorm/sequences.py     24 hour input windows for the GRU
oculstorm/gru.py           GRU model definition
scripts/train_xgb.py       train XGBoost, score against persistence
scripts/train_gru.py       train the GRU, score against persistence and XGBoost
scripts/build_dashboard.py export test predictions to docs/data.js
docs/                      static dashboard (index.html + generated data.js)
tests/                     look-ahead leakage tests
reports/                   metrics (tracked) and test predictions (ignored)
scripts/build_dataset.py   raw OMNI2 -> data/kaggle/omni2_hourly.parquet
data/raw/                  downloaded NASA files (ignored)
data/kaggle/               Kaggle upload folder: parquet (ignored), metadata, column docs
```
