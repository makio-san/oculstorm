"""Weighted average of the GRU and XGBoost forecasts, with weights fitted on validation hours.

pred = w * gru + (1 - w) * xgb, with w taken from a coarse grid, either one global weight or
one weight per Dst band. The band of an hour comes from dst_lag0, the Dst known at the time
the forecast is issued, never from the observed target.
"""
import numpy as np

GRID = np.round(np.linspace(0.0, 1.0, 21), 2)  # 0.00, 0.05, ..., 1.00
MIN_BAND_HOURS = 24  # a band with fewer fitting hours falls back to the global weight

# Bands by the current Dst (nT), checked in order: the first that matches wins.
BANDS = {
    "intense (dst_lag0 <= -100)": lambda d: d <= -100,
    "storm (-100 < dst_lag0 <= -50)": lambda d: d <= -50,
    "quiet (dst_lag0 > -50)": lambda d: np.ones(len(d), dtype=bool),
}


def band_of(dst_lag0) -> np.ndarray:
    """Name of the band of every hour, from the Dst at the issue time."""
    d = np.asarray(dst_lag0, dtype=float)
    out = np.full(len(d), "", dtype=object)
    for name, rule in BANDS.items():
        free = (out == "") & rule(d)
        out[free] = name
    return out


def fit_weight(y, gru, xgb, grid: np.ndarray = GRID) -> float:
    """Grid weight of the GRU that gives the lowest MSE on (y, gru, xgb)."""
    y, gru, xgb = (np.asarray(a, dtype=float) for a in (y, gru, xgb))
    mse = [np.mean((y - (w * gru + (1 - w) * xgb)) ** 2) for w in grid]
    return float(grid[int(np.argmin(mse))])


def fit_band_weights(y, gru, xgb, dst_lag0, min_hours: int = MIN_BAND_HOURS) -> dict[str, float]:
    """One grid weight per band; bands with fewer than `min_hours` hours get the global weight."""
    y, gru, xgb = (np.asarray(a, dtype=float) for a in (y, gru, xgb))
    bands = band_of(dst_lag0)
    overall = fit_weight(y, gru, xgb)
    weights = {}
    for name in BANDS:
        mask = bands == name
        weights[name] = fit_weight(y[mask], gru[mask], xgb[mask]) if mask.sum() >= min_hours else overall
    return weights


def combine(gru, xgb, weight: float | dict[str, float], dst_lag0=None) -> np.ndarray:
    """Blend the two forecasts with one weight, or with per band weights (then pass dst_lag0)."""
    gru, xgb = np.asarray(gru, dtype=float), np.asarray(xgb, dtype=float)
    if isinstance(weight, dict):
        w = np.array([weight[b] for b in band_of(dst_lag0)], dtype=float)
    else:
        w = np.full(len(gru), float(weight))
    return w * gru + (1 - w) * xgb
