"""Score the production 1 hour models on OMNI hours after the end of the test set.

Usage: python scripts/score_holdout.py [--after 2026-10-30T13:00] [--out reports/holdout.json]
Reads : data/kaggle/omni2_hourly.parquet (built by scripts/build_dataset.py from the newest OMNI2 file),
        models/xgb_dst_1h.ubj, models/gru_dst_1h_seed*.pt, models/gru_channels.json
Writes: reports/holdout.json

Every hour after the cutoff is a fresh holdout that no model and no choice has seen. The table is
cut to the hours after the cutoff before anything else, so no hour at or before it is ever read:
the first WINDOW hours of the holdout only serve as input history, and forecasts are scored from
the first issue hour that has a full window of holdout inputs behind it. The models are loaded
exactly as scripts/live_forecast.py loads them.

The newest OMNI2 file often has Dst for weeks before the IMF and plasma values are filled in.
Forecasts made on such hours measure how the models cope with missing inputs, not how they do
on fresh data, so an issue hour is only scored when every solar wind input was observed in each
of the last N_LAGS hours (the history the XGBoost lags read). The report states the input
coverage of the issue hours and how many were left out for missing inputs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from oculstorm.features import HORIZON, N_LAGS, SOLAR_WIND  # noqa: E402
from oculstorm.metrics import rmse, skill  # noqa: E402
from oculstorm.sequences import WINDOW  # noqa: E402

CUTOFF = "2026-10-30T13:00"  # last hour of the test set
OUT = ROOT / "reports" / "holdout.json"
MODEL_FILES = ["models/xgb_dst_1h.ubj", "models/gru_channels.json"]
STORM = -50.0  # nT, observed Dst at the target hour


def after_cutoff(table: pd.DataFrame, after: pd.Timestamp) -> pd.DataFrame:
    """Only the hours strictly after the cutoff; nothing earlier reaches a model."""
    return table[table.index > after]


def issue_hours(holdout: pd.DataFrame) -> pd.DataFrame:
    """Holdout hours with at least WINDOW hours of holdout history, the hours a forecast is issued at."""
    if holdout.empty:
        return holdout
    return holdout[holdout.index >= holdout.index.min() + pd.Timedelta(hours=WINDOW - 1)]


def inputs_observed(holdout: pd.DataFrame) -> pd.Series:
    """True at hour t when every solar wind input was observed at each of the hours t-N_LAGS+1 .. t."""
    observed = holdout[SOLAR_WIND].notna().all(axis=1).astype(float)
    return observed.rolling(N_LAGS, min_periods=N_LAGS).min().fillna(0.0).astype(bool)


def coverage(holdout: pd.DataFrame) -> dict:
    """Share of issue hours with each solar wind input observed, and with all of them over the lags."""
    issued = issue_hours(holdout)
    shares = {col: round(float(issued[col].notna().mean()), 3) if len(issued) else None for col in SOLAR_WIND}
    complete = inputs_observed(holdout).reindex(issued.index)
    return {"issue_hours": len(issued), "observed_share": shares,
            "issue_hours_with_inputs": int(complete.sum()), "issue_hours_without_inputs": int((~complete).sum())}


def forecasts(holdout: pd.DataFrame, predictors: dict[str, Callable[[pd.DataFrame], pd.Series]]) -> pd.DataFrame:
    """Forecasts of every model and persistence, the observation, indexed by the target hour.

    Issue hours with less than WINDOW hours of holdout history are dropped, so the GRU window
    and the XGBoost 24 hour rolling features are complete and both models score the same hours.
    Issue hours without every solar wind input observed over the last N_LAGS hours are dropped too.
    """
    issued = issue_hours(holdout)
    if issued.empty:
        return pd.DataFrame(columns=[*predictors, "persistence", "observed"], dtype=float)
    frame = pd.DataFrame({name: predict(holdout) for name, predict in predictors.items()})
    frame = frame.reindex(holdout.index)
    frame["persistence"] = holdout["dst"]
    step = pd.Timedelta(hours=HORIZON)
    frame["observed"] = holdout["dst"].shift(-HORIZON)
    frame = frame[(frame.index >= issued.index.min()) & inputs_observed(holdout)]
    frame.index = frame.index + step
    return frame.dropna()


def scores(frame: pd.DataFrame, models: list[str]) -> dict:
    """RMSE and skill against persistence, overall and for storm hours."""
    out = {}
    for regime, mask in {"all": np.ones(len(frame), dtype=bool), "storm": (frame["observed"] <= STORM).to_numpy()}.items():
        part = frame[mask]
        truth = part["observed"].to_numpy()
        entry = {"hours": int(mask.sum())}
        if len(part):
            reference = part["persistence"].to_numpy()
            entry["persistence"] = {"rmse": round(rmse(truth, reference), 3)}
            for name in models:
                pred = part[name].to_numpy()
                entry[name] = {"rmse": round(rmse(truth, pred), 3), "skill": round(skill(truth, pred, reference), 4)}
        out[regime] = entry
    return out


def blob_ids() -> dict[str, str]:
    files = MODEL_FILES + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "models").glob("gru_dst_1h_seed*.pt"))
    ids = {}
    for name in files:
        result = subprocess.run(["git", "rev-parse", f"HEAD:{name}"], cwd=ROOT, capture_output=True, text=True)
        ids[name] = result.stdout.strip() if result.returncode == 0 else None
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--after", default=CUTOFF, help="score only hours after this one (default: last test hour)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from live_forecast import predict_gru, predict_xgb  # noqa: E402

    from oculstorm.data import load_hourly

    after = pd.Timestamp(args.after)
    table = load_hourly()
    newest = table.index.max()
    holdout = after_cutoff(table, after)
    del table
    # XGBoost first, then torch with one thread: see predict_gru in scripts/live_forecast.py.
    inputs = coverage(holdout)
    if not inputs["issue_hours"]:
        raise SystemExit(f"no issue hours after {after} plus {WINDOW - 1} warmup hours (newest OMNI hour {newest})")
    frame = forecasts(holdout, {"xgb": predict_xgb, "gru": predict_gru})

    result = scores(frame, ["gru", "xgb"])
    stamp = lambda t: t.strftime("%Y-%m-%dT%H:%M")  # noqa: E731
    payload = {
        "cutoff": stamp(after),
        "newest_omni_hour": stamp(newest),
        "first_target_hour": stamp(frame.index.min()) if len(frame) else None,
        "last_target_hour": stamp(frame.index.max()) if len(frame) else None,
        "horizon_hours": HORIZON,
        "warmup_hours": WINDOW - 1,
        "input_rule": f"issue hour scored only if every solar wind input is observed in each of the last {N_LAGS} hours",
        "input_coverage": inputs,
        "hours": result["all"]["hours"],
        "storm_hours": result["storm"]["hours"],
        "storm_threshold_nt": STORM,
        "skill": "1 - MSE_model / MSE_persistence",
        "all": result["all"],
        "storm": result["storm"],
        "models": blob_ids(),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{inputs['issue_hours']} issue hours, {inputs['issue_hours_without_inputs']} left out for missing solar wind inputs, "
          f"coverage {inputs['observed_share']}")
    print(f"{payload['hours']} hours ({payload['storm_hours']} storm) from {payload['first_target_hour']} "
          f"to {payload['last_target_hour']} -> {args.out}")
    for regime in ("all", "storm"):
        entry = result[regime]
        if entry["hours"]:
            print(f"{regime}: " + ", ".join(f"{m} rmse {entry[m]['rmse']} skill {entry[m].get('skill', '-')}" for m in ("gru", "xgb", "persistence")))


if __name__ == "__main__":
    main()
