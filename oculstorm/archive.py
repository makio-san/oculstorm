"""Append-only log of the forecasts that were issued in real time.

The hindcast on the dashboard is recomputed every hour from whatever data is available by
then. This log is different: a forecast is written once, when it is issued, and never
changed. Only its `observed` column is filled in later, when the Dst value arrives.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from oculstorm.metrics import rmse

MODELS = ["gru", "xgb", "persistence"]
COLUMNS = ["issued", *MODELS, "observed"]
STORM = -50  # nT, same threshold as the "storm" regime of the test scorecard


def load(path: Path) -> pd.DataFrame:
    """Read the log, indexed by target hour. A missing file is an empty log."""
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="target"))
    return pd.read_csv(path, parse_dates=["target"], index_col="target", dtype={"issued": str})[COLUMNS]


def update(log: pd.DataFrame, issued: pd.Timestamp, target: pd.Timestamp, forecast: dict, observed: pd.Series) -> pd.DataFrame:
    """Log the forecast for `target` unless one is already there, then fill in observed values.

    `observed` is hourly Dst indexed by hour; it may revise values that were filled earlier.
    """
    log = log.copy()
    if target not in log.index and all(pd.notna(forecast[m]) for m in ["gru", "xgb"]):
        log.loc[target] = [issued.strftime("%Y-%m-%dT%H:%MZ"), *(round(float(forecast[m]), 1) for m in MODELS), np.nan]
    seen = observed.dropna()
    seen = seen[seen.index.isin(log.index)]
    log.loc[seen.index, "observed"] = seen.to_numpy()
    log.index.name = "target"
    return log.sort_index()


def save(log: pd.DataFrame, path: Path) -> None:
    out = log.copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M")
    out.to_csv(path, index_label="target", float_format="%.1f")


def _scores(rows: pd.DataFrame) -> dict:
    truth = rows["observed"].to_numpy(dtype=float)
    return {"hours": len(rows), "rmse": {m: round(rmse(truth, rows[m].to_numpy(dtype=float)), 2) for m in MODELS} if len(rows) else None}


def summary(log: pd.DataFrame) -> dict | None:
    """Scores of the logged forecasts, overall and for storm hours (observed Dst <= -50 nT)."""
    if log.empty:
        return None
    scored = log.dropna(subset=["observed", *MODELS])
    return {
        "since": log.index.min().strftime("%Y-%m-%dT%H:%M"),
        "issued": len(log),
        "deepest": None if scored.empty else float(scored["observed"].min()),
        "all": _scores(scored),
        "storm": _scores(scored[scored["observed"] <= STORM]),
    }
