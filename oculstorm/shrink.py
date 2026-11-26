"""Shrink the predicted Dst change in quiet conditions (issue 32).

forecast = Dst(t) + alpha * change, with alpha fitted by least squares on the hours of a
causally defined quiet condition and alpha = 1 everywhere else, so storm hours keep the full
predicted change. With clean inputs the least squares alpha is about 1. With real-time like
input errors it falls below 1, because part of the predicted change answers the input error.

Everything here uses values known at the issue time t only: the quiet condition comes from
dst_lag0, dst_min24 and vbs_mean3 as the model saw them, never from the observed target.
"""
import numpy as np

QUIET_DST = -30.0  # nT, dst_lag0 must be above
QUIET_MIN24 = -50.0  # nT, dst_min24 must be above
QUIET_VBS = 0.5  # mV/m, vbs_mean3 must be below
CHANGE_EDGE = 2.0  # nT, edge of the two bins of the absolute predicted change
BINS = ("abs change < 2 nT", "abs change >= 2 nT")
MIN_FIT_HOURS = 24  # with fewer fitting hours alpha stays 1

# Real-time like error of the newest Dst hour: AR(1) around a negative mean (issue 32).
ERROR_MEAN, ERROR_SD, ERROR_COEFFICIENT = -3.0, 4.0, 0.6


def quiet_condition(dst_lag0, dst_min24, vbs_mean3) -> np.ndarray:
    """Hours that are quiet as far as hour t can tell. A missing value is not quiet."""
    lag0, min24, vbs = (np.asarray(a, dtype=float) for a in (dst_lag0, dst_min24, vbs_mean3))
    with np.errstate(invalid="ignore"):
        return (lag0 > QUIET_DST) & (min24 > QUIET_MIN24) & (vbs < QUIET_VBS)


def least_squares_alpha(y, base, change, mask=None, min_hours: int = MIN_FIT_HOURS) -> float:
    """Least squares alpha of y = base + alpha * change on the hours in `mask`, not clipped.

    Returns 1 (no shrink) when there are fewer than `min_hours` hours or no predicted change.
    """
    y, base, change = (np.asarray(a, dtype=float) for a in (y, base, change))
    mask = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    c = change[mask]
    power = float(np.sum(c * c))
    if mask.sum() < min_hours or power == 0:
        return 1.0
    return float(np.sum(c * (y[mask] - base[mask])) / power)


def fit_alpha(y, base, change, mask=None, min_hours: int = MIN_FIT_HOURS) -> float:
    """The least squares alpha clipped to [0, 1]: a shrink never amplifies or reverses the change."""
    return float(np.clip(least_squares_alpha(y, base, change, mask, min_hours), 0.0, 1.0))


def change_bin(change) -> np.ndarray:
    """Bin name of every hour by the absolute predicted change."""
    big = np.abs(np.asarray(change, dtype=float)) >= CHANGE_EDGE
    return np.where(big, BINS[1], BINS[0])


def fit_bin_alphas(y, base, change, mask, min_hours: int = MIN_FIT_HOURS) -> dict[str, float]:
    """One alpha per bin of the absolute predicted change, on the hours in `mask`."""
    bins = change_bin(change)
    mask = np.asarray(mask, dtype=bool)
    return {name: fit_alpha(y, base, change, mask & (bins == name), min_hours) for name in BINS}


def shrink(base, change, alpha: float | dict[str, float], quiet) -> np.ndarray:
    """base + alpha * change in the `quiet` hours and base + change elsewhere."""
    base, change = np.asarray(base, dtype=float), np.asarray(change, dtype=float)
    if isinstance(alpha, dict):
        factor = np.array([alpha[b] for b in change_bin(change)], dtype=float)
    else:
        factor = np.full(len(base), float(alpha))
    factor = np.where(np.asarray(quiet, dtype=bool), factor, 1.0)
    return base + factor * change


def ar1_error(n: int, seed: int, mean: float = ERROR_MEAN, sd: float = ERROR_SD, coefficient: float = ERROR_COEFFICIENT) -> np.ndarray:
    """Stationary AR(1) series of length n with the given mean, standard deviation and coefficient."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(scale=sd * np.sqrt(1 - coefficient**2), size=n)
    out = np.empty(n)
    previous = rng.normal(scale=sd)  # start in the stationary distribution
    for i in range(n):
        previous = coefficient * previous + shocks[i]
        out[i] = previous
    return mean + out
