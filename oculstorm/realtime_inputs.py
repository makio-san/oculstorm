"""Score the production 1 hour models on validation with archived real-time ACE solar wind.

Usage: python -m oculstorm.realtime_inputs [--out reports/experiments/31-realtime-ace-inputs.json] [--keep]
Reads : data/kaggle/omni2_hourly.parquet, models/xgb_dst_1h.ubj, models/gru_dst_1h_seed*.pt,
        models/gru_channels.json, and the SWPC monthly files YYYYMM_ace_mag_1h.txt and
        YYYYMM_ace_swepam_1h.txt for 2017 to 2020 (downloaded into data/ace_rt, deleted at the end)
Writes: the JSON report

Validation only, no training. The table is cut to 2017-01-01 .. 2020-12-31 before a model sees it,
and file names outside those months are refused, so no test hour is downloaded, read or scored.
Dst and F10.7 stay from the parquet; only the eight solar wind columns are replaced.

The ACE files are hourly averages measured at L1 and not time shifted. They are delayed by
DELAYS hours (1.5 million km takes 0.5 to 1.5 hours), never advanced. Rows whose status is not 0
or that hold a missing sentinel become NaN, and the models cope with them as they do live.
"""
import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, TEST_START, VAL_START, load_hourly
from .features import HORIZON, N_LAGS, SOLAR_WIND
from .metrics import REGIMES, block_bootstrap_rmse, rmse, skill
from .sequences import WINDOW

BASE_URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly"
USER_AGENT = "python-urllib"  # generic on purpose: nothing about the project or its owner
CACHE = ROOT / "data" / "ace_rt"
OUT = ROOT / "reports" / "experiments" / "31-realtime-ace-inputs.json"
DELAYS = (1, 0)  # hours the L1 measurement is moved later; the first one is the main variant
KINDS = {
    "mag": ["bx", "by_gsm", "bz_gsm", "b_mag_avg", "lat", "lon"],
    "swepam": ["proton_density", "flow_speed", "proton_temp"],
}
SENTINEL = -999.0  # every missing value in the files is at or below this (-999.9, -9999.9, -1.00e+05)
BZ_LARGE = 4.0  # nT, the StormFocus threshold for a large real-time against OMNI Bz difference


def months(start: str = VAL_START, stop: str = TEST_START) -> list[str]:
    """YYYYMM of every validation month, 2017-01 .. 2020-12."""
    return [m.strftime("%Y%m") for m in pd.date_range(start, stop, freq="MS", inclusive="left")]


def file_name(month: str, kind: str) -> str:
    """Name of one monthly file; months outside the validation years are refused."""
    if month not in months():
        raise ValueError(f"{month} is outside the validation years; nothing else may be downloaded or read")
    if kind not in KINDS:
        raise ValueError(f"unknown file kind {kind}")
    return f"{month}_ace_{kind}_1h.txt"


def download(cache: Path = CACHE) -> list[Path]:
    """Fetch the 96 monthly files that are not in `cache` yet."""
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for month in months():
        for kind in KINDS:
            path = cache / file_name(month, kind)
            if not path.exists():
                request = urllib.request.Request(f"{BASE_URL}/{path.name}", headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=60) as response:
                    path.write_bytes(response.read())
            paths.append(path)
    return paths


def parse_month(text: str, kind: str) -> pd.DataFrame:
    """One monthly file as a frame indexed by the hour; bad status or sentinel values are NaN."""
    names = KINDS[kind]
    times, rows = [], []
    for line in text.splitlines():
        if not line.strip() or line[0] in ":#":
            continue
        parts = line.split()
        if len(parts) != 7 + len(names):
            raise ValueError(f"unexpected row in an ACE {kind} file: {line!r}")
        times.append(pd.Timestamp(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3][:2]), int(parts[3][2:])))
        values = np.array(parts[7:], dtype=float)
        if int(parts[6]) != 0:
            values[:] = np.nan
        values[values <= SENTINEL] = np.nan
        rows.append(values)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(times), columns=names, dtype=float)


def load_ace(cache: Path = CACHE) -> pd.DataFrame:
    """Hourly real-time ACE table on the validation hours, named like the OMNI columns, not delayed."""
    index = pd.date_range(VAL_START, TEST_START, freq="h", inclusive="left")
    parts = []
    for kind in KINDS:
        frames = []
        for month in months():
            frame = parse_month((cache / file_name(month, kind)).read_text(), kind)
            frames.append(frame[frame.index.strftime("%Y%m") == month])  # a file only speaks for its own month
        joined = pd.concat(frames)
        parts.append(joined[~joined.index.duplicated(keep="last")].reindex(index))
    ace = pd.concat(parts, axis=1)
    out = ace[["bz_gsm", "by_gsm", "b_mag_avg", "flow_speed", "proton_density", "proton_temp"]].copy()
    # Same formulas as oculstorm/live.py (5% helium for the pressure).
    out["flow_pressure"] = 2e-6 * out["proton_density"] * out["flow_speed"] ** 2
    out["electric_field"] = -out["flow_speed"] * out["bz_gsm"] * 1e-3
    return out[SOLAR_WIND]


def delayed(ace: pd.DataFrame, hours: int) -> pd.DataFrame:
    """The table with every value moved `hours` later. A negative delay would look ahead and is refused."""
    if hours < 0:
        raise ValueError("the L1 measurements may be delayed, never advanced")
    return ace.shift(hours)


def validation_table(table: pd.DataFrame) -> pd.DataFrame:
    """Only the validation years: no train hour and no test hour reaches a model or a statistic."""
    return table[(table.index >= VAL_START) & (table.index < TEST_START)]


def with_solar_wind(omni: pd.DataFrame, wind: pd.DataFrame) -> pd.DataFrame:
    """The OMNI table with its solar wind columns replaced by `wind`; Dst and F10.7 are kept."""
    out = omni.copy()
    out[SOLAR_WIND] = wind[SOLAR_WIND].reindex(omni.index)
    return out


def input_statistics(omni: pd.DataFrame, wind: pd.DataFrame) -> dict:
    """Per solar wind variable: bias and spread of real-time minus OMNI, and the missing shares."""
    stats = {}
    for col in SOLAR_WIND:
        a, o = wind[col].reindex(omni.index), omni[col]
        both = a.notna() & o.notna()
        diff = (a - o)[both]
        stats[col] = {
            "bias": round(float(diff.mean()), 4),
            "std_difference": round(float(diff.std()), 4),
            "correlation": round(float(np.corrcoef(a[both], o[both])[0, 1]), 4),
            "omni_mean": round(float(o[both].mean()), 4),
            "omni_std": round(float(o[both].std()), 4),
            "missing_share_realtime": round(float(a.isna().mean()), 4),
            "missing_share_omni": round(float(o.isna().mean()), 4),
            "hours_compared": int(both.sum()),
        }
    bz = (wind["bz_gsm"].reindex(omni.index) - omni["bz_gsm"]).dropna()
    density = (wind["proton_density"].reindex(omni.index) - omni["proton_density"]).dropna()
    if "b_vector_mag" in omni:
        # The ACE hourly Bt is the magnitude of the averaged vector, OMNI b_mag_avg the averaged magnitude.
        vector = (wind["b_mag_avg"].reindex(omni.index) - omni["b_vector_mag"]).dropna()
        stats["b_mag_avg"]["bias_to_omni_b_vector_mag"] = round(float(vector.mean()), 4)
        stats["b_mag_avg"]["std_difference_to_omni_b_vector_mag"] = round(float(vector.std()), 4)
    stats["bz_gsm"][f"share_abs_difference_above_{BZ_LARGE:g}nT"] = round(float((bz.abs() > BZ_LARGE).mean()), 4)
    stats["proton_density"]["share_realtime_below_omni"] = round(float((density < 0).mean()), 4)
    return stats


def coverage(ace: pd.DataFrame) -> dict:
    """Where the real-time record is missing: per year, and how much of it is in gaps a 3 hour fill bridges."""
    out = {}
    for label, col in {"imf": "bz_gsm", "plasma": "flow_speed"}.items():
        missing = ace[col].isna()
        lengths = missing.groupby((missing != missing.shift()).cumsum()[missing]).sum()
        out[label] = {
            "missing_share_by_year": {str(y): round(float(v), 4) for y, v in missing.groupby(ace.index.year).mean().items()},
            "gaps": int(len(lengths)),
            "longest_gap_hours": int(lengths.max()) if len(lengths) else 0,
            "share_of_missing_hours_in_gaps_up_to_3h": round(float(lengths[lengths <= 3].sum() / max(missing.sum(), 1)), 4),
        }
    return out


def alignment(omni: pd.DataFrame, ace: pd.DataFrame, delays=(0, 1, 2, 3)) -> dict:
    """Correlation with OMNI for each causal delay, a check of the time stamps and of the L1 travel time."""
    return {f"delay{d}h": {col: round(float(delayed(ace, d)[col].corr(omni[col])), 4) for col in ("bz_gsm", "by_gsm", "flow_speed")}
            for d in delays}


def issue_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Issue hours with WINDOW hours of validation history behind them, also after the longest delay."""
    return np.asarray(index >= pd.Timestamp(VAL_START) + pd.Timedelta(hours=WINDOW - 1 + max(DELAYS)))


def inputs_observed(table: pd.DataFrame) -> pd.Series:
    """True at hour t when every solar wind input was observed at each of the hours t-N_LAGS+1 .. t."""
    observed = table[SOLAR_WIND].notna().all(axis=1).astype(float)
    return observed.rolling(N_LAGS, min_periods=N_LAGS).min().fillna(0.0).astype(bool)


def interval(values: np.ndarray, digits: int = 3) -> list:
    """95% interval over the bootstrap resamples that hold at least one hour of the regime."""
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None]
    return [round(float(v), digits) for v in np.percentile(values, [2.5, 97.5])]


def score(frame: pd.DataFrame, reference: str, candidates: list[str], rows: np.ndarray | None = None) -> dict:
    """RMSE and skill per regime, and the paired block bootstrap of each candidate against `reference`.

    `frame` holds the observed Dst, persistence and one forecast column per input variant, in
    time order. `rows` limits the scored hours (the bootstrap still draws blocks of the full axis).
    """
    y = frame["observed"].to_numpy()
    rows = np.ones(len(y), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    names = [reference, *candidates]
    out = {}
    for regime, mask_fn in REGIMES.items():
        mask = mask_fn(y) & rows
        result = {"n_hours": int(mask.sum())}
        if mask.any():
            base = frame["persistence"].to_numpy()[mask]
            result["rmse_persistence"] = round(rmse(y[mask], base), 4)
            value = {m: rmse(y[mask], frame[m].to_numpy()[mask]) for m in names}
            boot = block_bootstrap_rmse(y, {m: frame[m].to_numpy() for m in names}, mask)
            result["inputs"] = {
                m: {"rmse": round(value[m], 4), "rmse_ci95": interval(boot[:, j]),
                    "skill": round(skill(y[mask], frame[m].to_numpy()[mask], base), 4)}
                for j, m in enumerate(names)
            }
            result["differences"] = {}
            for j, m in enumerate(names[1:], start=1):
                low, high = interval(boot[:, j] - boot[:, 0])
                result["differences"][f"{m} minus {reference}"] = {
                    "rmse": round(value[m] - value[reference], 4),
                    "rmse_ci95": [low, high],
                    "mse_change_percent": round(100 * (value[m] ** 2 / value[reference] ** 2 - 1), 2),
                    "mse_change_percent_ci95": interval(100 * (boot[:, j] ** 2 / boot[:, 0] ** 2 - 1), 2),
                    "skill_change": round(result["inputs"][m]["skill"] - result["inputs"][reference]["skill"], 4),
                    "significant": bool(low is not None and (low > 0 or high < 0)),
                    "resamples_with_hours": int(np.isfinite(boot[:, 0]).sum()),
                }
        out[regime] = result
    return out


def forecast_frame(tables: dict[str, pd.DataFrame], predict) -> pd.DataFrame:
    """Observed Dst, persistence and `predict` on every table, indexed by the issue hour."""
    first = next(iter(tables.values()))
    frame = pd.DataFrame({"observed": first["dst"].shift(-HORIZON), "persistence": first["dst"]})
    for name, table in tables.items():
        frame[name] = predict(table).reindex(frame.index)
    return frame[issue_mask(frame.index)].dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=CACHE, help="folder for the monthly files")
    parser.add_argument("--keep", action="store_true", help="do not delete the downloaded files at the end")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_forecast import predict_gru, predict_xgb  # the models are loaded exactly as the live forecast does

    started = time.time()
    try:
        files = download(args.cache)
        megabytes = sum(p.stat().st_size for p in files) / 1e6
        ace = load_ace(args.cache)
    finally:
        if not args.keep:
            shutil.rmtree(args.cache, ignore_errors=True)

    omni = validation_table(load_hourly())
    name = lambda d: f"ace_delay{d}h"  # noqa: E731
    tables = {"omni": omni, **{name(d): with_solar_wind(omni, delayed(ace, d)) for d in DELAYS}}
    candidates = [name(d) for d in DELAYS]

    # XGBoost first, then torch with one thread: see predict_gru in scripts/live_forecast.py.
    frames = {"xgb": forecast_frame(tables, predict_xgb),
              "gru": forecast_frame(tables, lambda t: predict_gru(t).mean(axis=1))}
    hours = frames["xgb"].index
    complete = inputs_observed(tables["omni"])
    for d in DELAYS:
        complete &= inputs_observed(tables[name(d)])
    complete = complete.reindex(hours).to_numpy()

    stamp = lambda t: t.strftime("%Y-%m-%dT%H:%M")  # noqa: E731
    report = {
        "issue": 31,
        "title": "Production 1 hour models on validation with archived real-time ACE solar wind in place of OMNI",
        "split": f"validation only, {VAL_START} .. 2020-12-31; no training; the table is cut to these years before any model call",
        "horizon_hours": HORIZON,
        "source": {"url": BASE_URL, "files": len(files), "megabytes": round(megabytes, 2),
                   "pattern": "YYYYMM_ace_mag_1h.txt, YYYYMM_ace_swepam_1h.txt", "user_agent": USER_AGENT,
                   "kept": bool(args.keep)},
        "configuration": {
            "models": "models/xgb_dst_1h.ubj and the mean of models/gru_dst_1h_seed*.pt, loaded by scripts/live_forecast.py",
            "replaced_columns": SOLAR_WIND,
            "kept_from_parquet": ["dst", "f107"],
            "missing": "status other than 0 or a value at or below -999 becomes NaN; XGBoost reads NaN, the GRU forward fills 3 hours and masks",
            "derived": "flow_pressure = 2e-6 * n * V^2, electric_field = -V * Bz * 1e-3, as oculstorm/live.py",
            "delays_hours": list(DELAYS),
            "scored_hours": f"issue hours from {stamp(hours.min())} to {stamp(hours.max())}, the same for every input; "
                            f"the first {WINDOW - 1 + max(DELAYS)} hours only serve as input history",
        },
        "bootstrap": {"method": "oculstorm.metrics.block_bootstrap_rmse, paired", "block_hours": 168, "resamples": 2000, "seed": 0,
                      "intervals": "95%; differences are ACE inputs minus OMNI inputs for the same model"},
        "input_statistics": {name(d): input_statistics(omni, delayed(ace, d)) for d in DELAYS},
        "coverage": coverage(ace),
        "alignment_correlation_with_omni": alignment(omni, ace),
        "issue_hours": {"all": len(hours), "all_inputs_observed_last_6h_in_every_variant": int(complete.sum())},
        "scores": {
            "all_hours": {m: score(f, "omni", candidates) for m, f in frames.items()},
            "inputs_observed": {m: score(f, "omni", candidates, complete) for m, f in frames.items()},
        },
    }
    report["seconds"] = round(time.time() - started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")

    for variant, stats in report["input_statistics"].items():
        print(f"\n{variant}: real-time minus OMNI")
        for col, s in stats.items():
            print(f"  {col:15s} bias {s['bias']:9.3f}  std {s['std_difference']:9.3f}  r {s['correlation']:.3f}  missing {100 * s['missing_share_realtime']:.1f}%")
    for subset, by_model in report["scores"].items():
        for model, regimes in by_model.items():
            print(f"\n{subset} / {model}")
            for regime, r in regimes.items():
                if not r["n_hours"]:
                    print(f"  {regime}: no hours")
                    continue
                line = "  ".join(f"{m} {v['rmse']:.3f} ({v['skill']:.3f})" for m, v in r["inputs"].items())
                print(f"  {regime} ({r['n_hours']:,} h): {line}")
                for d, v in r["differences"].items():
                    print(f"    {d}: MSE {v['mse_change_percent']:+.1f}% {v['mse_change_percent_ci95']}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
