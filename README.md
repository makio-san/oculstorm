# oculstorm

Forecasting geomagnetic storm intensity (the Dst index) one hour ahead from recent solar wind measurements.

## Plan

1. **Data** — NASA OMNI2 hourly solar wind, IMF and geomagnetic indices (1963–2026), cleaned into a single Parquet table and hosted on Kaggle.
2. **Models** — XGBoost baseline first, then a small GRU.
3. **Evaluation** — every model must beat the persistence baseline ("next hour = current hour").
4. **Product** — a space-weather dashboard showing observed vs. predicted storm intensity.

## Data

Source: [NASA SPDF OMNIWeb](https://omniweb.gsfc.nasa.gov/), low-resolution OMNI2 file
`omni2_all_years.dat`. Raw files and the Parquet output are not tracked in git; the
Parquet table is published as a Kaggle dataset.

```bash
python3 -m venv .venv && .venv/bin/pip install pandas pyarrow kaggle
```

```bash
curl -o data/raw/omni2_all_years.dat https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat
```

```bash
.venv/bin/python scripts/build_dataset.py
```

The script names the 55 columns, replaces OMNI fill values with nulls, adds a UTC
`time` column and drops the trailing fill rows after the last available Dst value.

## Layout

```
scripts/build_dataset.py   raw OMNI2 -> data/kaggle/omni2_hourly.parquet
data/raw/                  downloaded NASA files (ignored)
data/kaggle/               Kaggle upload folder: parquet (ignored), metadata, column docs
```
