"""Fetch real-time data, forecast the hour in progress and hindcast the last week.

Usage: python scripts/live_forecast.py
Reads : models/xgb_dst_1h.ubj, models/gru_dst_1h_seed*.pt, models/gru_channels.json
Writes: docs/live.js (loaded by the dashboard)

The hindcast replays the same pipeline over the last ~6 days of real-time inputs,
so the dashboard can show how the models actually do on raw operational data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


def predict_gru(table: pd.DataFrame) -> pd.Series:
    import torch

    from oculstorm.gru import DstGRU

    # torch and xgboost each bundle their own OpenMP runtime; running both multi-threaded
    # in one process crashes on macOS, and one thread is plenty for a few hundred rows.
    torch.set_num_threads(1)
    stats = json.loads((MODELS / "gru_channels.json").read_text())
    windows, meta = build_windows(table, stats, require_target=False)
    x = torch.from_numpy(np.ascontiguousarray(windows[meta["row"].to_numpy()]))
    deltas = []
    for path in sorted(MODELS.glob("gru_dst_1h_seed*.pt")):
        checkpoint = torch.load(path, weights_only=True)
        model = DstGRU(windows.shape[2], checkpoint["hidden"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            deltas.append(model(x).numpy() * checkpoint["scale"])
    return pd.Series(meta["dst_lag0"].to_numpy() + np.mean(deltas, axis=0), index=meta.index)


def as_list(series: pd.Series, digits: int = 1) -> list:
    return [None if pd.isna(v) else round(float(v), digits) for v in series]


def main() -> None:
    now = pd.Timestamp.now("UTC").tz_localize(None)
    table = fetch_live_table(now)
    step = pd.Timedelta(hours=HORIZON)

    # Forecasts are issued at hour t for hour t + HORIZON; index everything by the target hour.
    xgb_forecast = predict_xgb(table)  # before torch is imported, see predict_gru
    forecasts = pd.DataFrame({"gru": predict_gru(table), "xgb": xgb_forecast, "persistence": table["dst"]})
    forecasts.index = forecasts.index + step
    forecasts = forecasts.dropna(subset=["gru", "xgb"])
    if forecasts.empty:
        raise SystemExit("not enough real-time history to make a forecast")
    hours = pd.date_range(forecasts.index.min(), forecasts.index.max(), freq="h")
    forecasts = forecasts.reindex(hours)
    observed = table["dst"].reindex(hours)  # the newest target hour is not observed yet

    scored = forecasts.assign(observed=observed).dropna()
    truth = scored["observed"].to_numpy()
    scores = {name: round(rmse(truth, scored[name].to_numpy()), 2) for name in ["gru", "xgb", "persistence"]}

    latest = table.index.max()
    payload = {
        "generated": now.strftime("%Y-%m-%dT%H:%MZ"),
        "start": hours[0].strftime("%Y-%m-%dT%H:%M"),
        "hours": len(hours),
        "target": hours[-1].strftime("%Y-%m-%dT%H:%M"),
        "target_is_future": bool(pd.isna(observed.iloc[-1])),
        "observed": as_list(observed, 0),
        "gru": as_list(forecasts["gru"]),
        "xgb": as_list(forecasts["xgb"]),
        "persistence": as_list(forecasts["persistence"]),
        "hindcast": {"hours": len(scored), "rmse": scores},
        "inputs_hour": latest.strftime("%Y-%m-%dT%H:%M"),
        "inputs": {name: None if pd.isna(table.loc[latest, col]) else round(float(table.loc[latest, col]), 2) for col, name in INPUTS.items()},
        "solar_wind_coverage": round(float(table["flow_speed"].notna().mean()), 3),
    }
    OUT.write_text("window.OCULSTORM_LIVE = " + json.dumps(payload, separators=(",", ":")) + ";\n")

    print(f"generated {payload['generated']}  inputs through {latest}  solar wind coverage {payload['solar_wind_coverage']:.0%}")
    print(f"forecast for {hours[-1]}: GRU {forecasts['gru'].iloc[-1]:.1f} nT, XGBoost {forecasts['xgb'].iloc[-1]:.1f} nT, "
          f"persistence {forecasts['persistence'].iloc[-1]:.0f} nT")
    print(f"hindcast over {len(scored)} hours, RMSE: {scores}")


if __name__ == "__main__":
    main()
