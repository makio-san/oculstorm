"""Fetch real-time data, forecast the hour in progress and hindcast the last week.

Usage: python scripts/live_forecast.py [--archive forecasts.csv]
Reads : models/xgb_dst_1h.ubj, models/gru_dst_1h_seed*.pt, models/gru_channels.json
Writes: docs/live.js (loaded by the dashboard)

The GRU forecast is the mean of its seed members. The spread field is the lowest and highest
of the GRU seed members and XGBoost for each hour: how much the models disagree, not a
confidence interval.

With --archive the newest forecast is also appended to a log that is never rewritten,
and the log's scores are added to live.js. That log is the honest real-time track record.

The hindcast replays the same pipeline over the last ~6 days of real-time inputs,
so the dashboard can show how the models actually do on raw operational data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oculstorm import archive  # noqa: E402
from oculstorm.features import HORIZON, build_features  # noqa: E402
from oculstorm.live import fetch_live_table  # noqa: E402
from oculstorm.metrics import rmse  # noqa: E402
from oculstorm.sequences import build_windows  # noqa: E402

MODELS = ROOT / "models"
OUT = ROOT / "docs" / "live.js"
INPUTS = {"bz_gsm": "IMF Bz (GSM)", "flow_speed": "Solar wind speed", "proton_density": "Proton density", "flow_pressure": "Dynamic pressure"}


def predict_xgb(table: pd.DataFrame) -> pd.Series:
    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(MODELS / "xgb_dst_1h.ubj")
    X, _ = build_features(table, require_target=False)
    X = X[model.get_booster().feature_names]
    return pd.Series(X["dst_lag0"].to_numpy() + model.predict(X), index=X.index)


def predict_gru(table: pd.DataFrame) -> pd.DataFrame:
    """One column per GRU seed member, each a full Dst forecast; their mean is the GRU forecast."""
    import torch

    from oculstorm.gru import DstGRU

    # torch and xgboost each bundle their own OpenMP runtime; running both multi-threaded
    # in one process crashes on macOS, and one thread is plenty for a few hundred rows.
    torch.set_num_threads(1)
    stats = json.loads((MODELS / "gru_channels.json").read_text())
    windows, meta = build_windows(table, stats, require_target=False)
    x = torch.from_numpy(np.ascontiguousarray(windows[meta["row"].to_numpy()]))
    members = {}
    for path in sorted(MODELS.glob("gru_dst_1h_seed*.pt")):
        checkpoint = torch.load(path, weights_only=True)
        model = DstGRU(windows.shape[2], checkpoint["hidden"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            delta = model(x).numpy() * checkpoint["scale"]
        members["gru_" + path.stem.rsplit("_", 1)[-1]] = meta["dst_lag0"].to_numpy() + delta
    return pd.DataFrame(members, index=meta.index)


def model_spread(members: pd.DataFrame) -> pd.DataFrame:
    """Lowest and highest member forecast per hour; missing if any member is missing that hour."""
    return pd.DataFrame({"low": members.min(axis=1, skipna=False), "high": members.max(axis=1, skipna=False)})


def as_list(series: pd.Series, digits: int = 1) -> list:
    return [None if pd.isna(v) else round(float(v), digits) for v in series]


def align(table: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Forecasts, observed Dst and the model spread, indexed by the hour they are for.

    members has one column per GRU seed member (gru_*) and an xgb column, indexed by issue hour.
    Forecasts are issued at hour t for hour t + HORIZON.
    """
    gru = members.filter(like="gru_").mean(axis=1, skipna=False)
    frame = pd.concat([gru.rename("gru"), members["xgb"], table["dst"].rename("persistence"), model_spread(members)], axis=1, sort=True)
    frame.index = frame.index + pd.Timedelta(hours=HORIZON)
    frame = frame.dropna(subset=["gru", "xgb"])
    if frame.empty:
        raise SystemExit("not enough real-time history to make a forecast")
    frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="h"))
    frame["observed"] = table["dst"].reindex(frame.index)  # the newest target hour is not observed yet
    return frame


def build_payload(now: pd.Timestamp, table: pd.DataFrame, frame: pd.DataFrame, members: list, record: dict | None = None) -> dict:
    """The live.js payload; frame comes from align, members names the spread's member forecasts."""
    hours = frame.index
    scored = frame[["gru", "xgb", "persistence", "observed"]].dropna()
    truth = scored["observed"].to_numpy()
    scores = {name: round(rmse(truth, scored[name].to_numpy()), 2) for name in ["gru", "xgb", "persistence"]}
    latest = table.index.max()
    return {
        "generated": now.strftime("%Y-%m-%dT%H:%MZ"),
        "start": hours[0].strftime("%Y-%m-%dT%H:%M"),
        "hours": len(hours),
        "target": hours[-1].strftime("%Y-%m-%dT%H:%M"),
        "target_is_future": bool(pd.isna(frame["observed"].iloc[-1])),
        "observed": as_list(frame["observed"], 0),
        "gru": as_list(frame["gru"]),
        "xgb": as_list(frame["xgb"]),
        "persistence": as_list(frame["persistence"]),
        "spread": {"members": list(members), "low": as_list(frame["low"]), "high": as_list(frame["high"])},
        "hindcast": {"hours": len(scored), "rmse": scores},
        "archive": record,
        "inputs_hour": latest.strftime("%Y-%m-%dT%H:%M"),
        "inputs": {name: None if pd.isna(table.loc[latest, col]) else round(float(table.loc[latest, col]), 2) for col, name in INPUTS.items()},
        "solar_wind_coverage": round(float(table["flow_speed"].notna().mean()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, help="CSV log of forecasts issued in real time; created if missing")
    args = parser.parse_args()

    now = pd.Timestamp.now("UTC").tz_localize(None)
    table = fetch_live_table(now)

    xgb_forecast = predict_xgb(table)  # before torch is imported, see predict_gru
    members = predict_gru(table).join(xgb_forecast.rename("xgb"), how="outer")
    frame = align(table, members)
    hours = frame.index

    record = None
    if args.archive:
        log = archive.update(archive.load(args.archive), now, hours[-1], frame.iloc[-1].to_dict(), table["dst"])
        archive.save(log, args.archive)
        record = archive.summary(log)

    payload = build_payload(now, table, frame, members.columns, record)
    OUT.write_text("window.OCULSTORM_LIVE = " + json.dumps(payload, separators=(",", ":")) + ";\n")

    newest = frame.iloc[-1]
    print(f"generated {payload['generated']}  inputs through {table.index.max()}  solar wind coverage {payload['solar_wind_coverage']:.0%}")
    print(f"forecast for {hours[-1]}: GRU {newest['gru']:.1f} nT, XGBoost {newest['xgb']:.1f} nT, "
          f"persistence {newest['persistence']:.0f} nT, model spread {newest['low']:.1f} to {newest['high']:.1f} nT")
    print(f"hindcast over {payload['hindcast']['hours']} hours, RMSE: {payload['hindcast']['rmse']}")
    if record:
        print(f"archive: {record['issued']} forecasts logged since {record['since']}, {record['all']['hours']} scored: {record['all']['rmse']}")


if __name__ == "__main__":
    main()
