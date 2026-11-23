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

## Longer horizons (3, 6 and 12 hours ahead)

```bash
.venv/bin/python scripts/train_xgb.py --horizon 6
```

```bash
.venv/bin/python scripts/train_gru.py --horizon 6
```

Same inputs, same split and same models; only the target moves to Dst(t+h). Persistence
("Dst stays where it is now") stays the reference. RMSE on the test years, in nT, with the
MSE skill score against persistence in parentheses:

| Lead time | Hours | Persistence | XGBoost | GRU | Storm hours: persistence | XGBoost | GRU |
|---|---|---|---|---|---|---|---|
| 1 h | 51,086 | 4.81 | 3.67 (0.42) | 3.58 (0.45) | 12.86 | 9.12 (0.50) | 8.73 (0.54) |
| 3 h | 51,084 | 9.69 | 7.94 (0.33) | 7.78 (0.36) | 28.39 | 20.19 (0.49) | 20.24 (0.49) |
| 6 h | 51,081 | 13.85 | 11.39 (0.32) | 11.10 (0.36) | 44.93 | 32.49 (0.48) | 34.53 (0.41) |
| 12 h | 51,075 | 18.69 | 15.75 (0.29) | 15.02 (0.35) | 66.03 | 50.79 (0.41) | 52.45 (0.37) |

- Both models beat persistence at every lead time, overall and in storms.
- Errors grow quickly with lead time: 12 hours ahead the typical storm-time error is about
  50 nT, so a 12 hour forecast tells you a storm is likely, not how deep it will get.
- Overall the GRU stays ahead. From 6 hours on, XGBoost is better during storms
  (and intense storms: 65 vs 71 nT at 6 h), while the GRU is better in quiet time.
  Both differences are significant, see the bootstrap intervals below.
- Skill in storms holds up better than overall skill, because persistence fails worst
  exactly when Dst changes fast.
- The solar wind measured at L1 reaches Earth only 30 to 60 minutes later, so beyond about
  one hour the models are extrapolating from the current state of the solar wind and the
  ring current, not reading the future driver. That is the physical ceiling on these numbers.

Only the 1 hour models are committed (the live forecast uses them); the others are
reproducible with the commands above. Reports are in `reports/*_{3,6,12}h.json`.

## Storm weighting and significance

```bash
.venv/bin/python scripts/train_xgb.py --horizon 6 --storm-weight 10
```

```bash
.venv/bin/python scripts/train_gru.py --horizon 6 --storm-weight 10
```

```bash
.venv/bin/python scripts/compare_models.py --horizon 6 --storm-weight 10
```

Storm hours are about 4% of the record, so a plain loss is dominated by quiet time.
`--storm-weight K` gives every training hour a weight that rises linearly from 1 at
Dst >= 0 nT to K at Dst <= -100 nT, in the loss and in the validation score used for early
stopping. K = 10 was picked on the validation years (XGBoost at 1 h, K = 1, 3, 10:
validation storm RMSE 7.17, 7.13, 7.02 nT), never on test.

`compare_models.py` puts a 95% interval on every RMSE and on every difference between two
models. Test hours are resampled in one week blocks (errors an hour apart are not
independent) and all models share the same resamples, so a difference is significant when
its interval excludes zero. Test RMSE in nT, weighted runs after the arrow; **bold** marks a
significant change:

| Lead time | Storm hours: XGBoost | GRU | Quiet hours: XGBoost | GRU | All hours: XGBoost | GRU |
|---|---|---|---|---|---|---|
| 1 h | 9.12 → 8.99 | 8.73 → **8.49** | 3.24 → **3.27** | 3.17 → **3.27** | 3.67 → 3.69 | 3.58 → **3.64** |
| 3 h | 20.19 → **19.59** | 20.24 → **19.45** | 6.94 → **7.42** | 6.73 → **7.48** | 7.94 → **8.29** | 7.78 → **8.33** |
| 6 h | 32.49 → **31.14** | 34.53 → **32.41** | 9.46 → **10.55** | 8.77 → **10.22** | 11.39 → **12.12** | 11.10 → **11.99** |
| 12 h | 50.79 → **47.96** | 52.45 → **50.32** | 12.11 → **14.65** | 10.77 → **12.61** | 15.75 → **17.36** | 15.02 → **16.05** |

- Weighting buys storm accuracy with quiet-time accuracy, and the trade gets steeper with
  lead time: at 12 h XGBoost gains 2.8 nT in storms and loses 2.5 nT in quiet time.
  It is a choice of what the forecast is for, not a free improvement.
- At 1 h the gain is small: 0.24 nT for the GRU in storms, at a cost of 0.10 nT in quiet time.
- GRU against XGBoost (unweighted): the GRU is significantly better overall at 1, 3 and 12 h
  and in quiet time at every lead time; XGBoost is significantly better in storms at 6 and
  12 h (by 2.0 and 1.7 nT) and in intense storms at 6 and 12 h (by 5.6 and 4.2 nT).
- Intervals on intense storms (322 hours, a handful of events) are wide; most differences
  there are not significant.

The live forecast keeps the unweighted 1 hour models. Reports are in
`reports/comparison{,_3h,_6h,_12h}.json` and `reports/*_sw10.json`.

## Dashboard

Live: https://makio-san.github.io/oculstorm/

A static page in `docs/` shows observed Dst against the GRU, XGBoost and persistence
forecasts for any period of the test years at a lead time of 1, 3, 6 or 12 hours, a chart of
error against lead time, the largest storms as presets, error
tiles for the selected period, a table view of every chart and the full scorecard.
The hero shows the live forecast on a storm scale (quiet, moderate, intense, severe), the
largest storms are one click replays with sparklines, and a light/dark switch sits in the top bar.

```bash
.venv/bin/python scripts/build_dashboard.py
```

```bash
.venv/bin/python -m http.server 8765 --directory docs
```

Then open http://localhost:8765. The page has no dependencies, so `docs/index.html`
also opens straight from disk and the folder can be served by GitHub Pages.
The lower part replays the held out test set.

### Live forecast

The top of the page is live. Every hour a GitHub Actions workflow
(`.github/workflows/live.yml`) runs `scripts/live_forecast.py`, which

1. fetches NOAA SWPC real-time solar wind (propagated from L1 to Earth), the WDC Kyoto
   quick-look Dst and the F10.7 flux,
2. averages them into the same hourly table layout as OMNI, dropping the hour in progress,
3. runs the committed XGBoost and GRU models to forecast the hour in progress,
4. hindcasts the last 6 days with the same real-time inputs and reports the RMSE against
   persistence, so the page always shows how the models do on operational data,
5. appends the new forecast to `forecasts.csv` on the `live-archive` branch,
6. deploys the page with a fresh `docs/live.js`. Nothing is committed to `main`.

The hindcast in step 4 is recomputed every hour from whatever data is available by then.
The log in step 5 is the stricter record: each forecast is written once, at the moment it
is issued, and never changed; only its `observed` column is filled in when the Dst value
arrives. The dashboard scores that log separately (all hours and storm hours), and the file
is downloadable from https://makio-san.github.io/oculstorm/forecasts.csv. Hours when the
workflow did not run are simply missing; they are not backfilled.

GitHub's own `schedule` trigger proved unreliable for this repo (about 4 runs a day instead
of 24), so an external cron service calls the `workflow_dispatch` API every hour at minute 7
UTC; the `schedule` trigger stays as a backup. Extra runs are harmless: a target hour that is
already in the log is never rewritten. The service needs a fine-grained GitHub token limited to
this repository with only the "Actions: read and write" permission; the token lives in the cron
service, never in this repository.

```
POST https://api.github.com/repos/makio-san/oculstorm/actions/workflows/live.yml/dispatches
Accept: application/vnd.github+json
Authorization: Bearer <token>
X-GitHub-Api-Version: 2022-11-28

{"ref": "main"}
```

```bash
.venv/bin/python scripts/live_forecast.py
```

On the first run (2026-11-12, 142 quiet hours, deepest Dst -38 nT) the real-time RMSE was
2.63 nT for the GRU, 2.67 nT for XGBoost and 3.66 nT for persistence. One quiet week is
not evidence for storm-time skill on real-time data; that needs a storm to pass through.
Quick-look Dst is preliminary and gets revised, and the forecast lead time is at most one hour.

Full numbers are in `reports/xgb_baseline.json` and `reports/gru_baseline.json`.
Run the leakage tests with:

```bash
.venv/bin/python -m pytest tests
```

## Layout

```
oculstorm/data.py          load the hourly table, chronological split
oculstorm/features.py      backward-looking lag and coupling features
oculstorm/metrics.py       persistence baselines, RMSE / MAE / skill by regime, storm weights, block bootstrap
oculstorm/sequences.py     24 hour input windows for the GRU
oculstorm/gru.py           GRU model definition
scripts/train_xgb.py       train XGBoost, score against persistence
scripts/train_gru.py       train the GRU, score against persistence and XGBoost
scripts/compare_models.py  bootstrap intervals for RMSE differences between models
scripts/build_dashboard.py export test predictions to docs/data.js
oculstorm/live.py          NOAA real-time inputs as an OMNI-style hourly table
oculstorm/archive.py       append-only log of forecasts issued in real time
scripts/live_forecast.py   live forecast and 6 day hindcast -> docs/live.js
models/                    trained models used by the live forecast
docs/                      static dashboard (index.html + generated data.js)
tests/                     look-ahead leakage, live table, archive and metric tests
reports/                   metrics (tracked) and test predictions (ignored)
scripts/build_dataset.py   raw OMNI2 -> data/kaggle/omni2_hourly.parquet
data/raw/                  downloaded NASA files (ignored)
data/kaggle/               Kaggle upload folder: parquet (ignored), metadata, column docs
agents/                    scheduled agents: rules, retrain policy, helpers (see agents/README.md)
```
