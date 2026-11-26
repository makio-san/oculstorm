"""Dst baseline offset augmentation (issue 23).

Quick-look Dst has a quiet time baseline that drifts by tens of nT, while the models learn a
recovery toward the final Dst baseline. Persistence is invariant to a constant offset, the
models are not. The augmentation adds one random offset per 168 hour block to the Dst column
of a copy of the table, so inputs and target of a row move together and every delta target
stays what it was.

Unsettled newest hour (issue 30). Live, the newest quick-look Dst hour is not settled when the
forecast is issued. newest_hour_error draws an AR(1) error for that hour; build_features adds
it to Dst(t) as seen from row t only (newest_dst_error), the target stays clean.
"""
import numpy as np
import pandas as pd

BLOCK = 168  # hours that share one offset
OFFSET_RANGE = (-25.0, 10.0)  # nT, from the quick-look bias reported by Nair et al. (2023)
WARMUP = 24  # hours of a block a row needs behind it: the 24 hour Dst window and the lags
# Error of the newest quick-look hour at issue time against the settled value (issue 30),
# from 160 live input hours of November 2026: mean and standard deviation in nT, lag 1 coefficient.
NEWEST_ERROR = (-3.0, 4.0, 0.6)


def block_offsets(index: pd.DatetimeIndex, seed: int, block: int = BLOCK, low: float = OFFSET_RANGE[0], high: float = OFFSET_RANGE[1]) -> np.ndarray:
    """One offset in [low, high] per block of `block` hours, repeated for every hour of the block.

    Blocks are counted from the first row of `index`, so a longer table gives the same offsets
    for the same hours.
    """
    n_blocks = len(index) // block + 1
    draws = np.random.default_rng(seed).uniform(low, high, size=n_blocks)
    return np.repeat(draws, block)[: len(index)]


def offset_table(df: pd.DataFrame, seed: int, **kwargs) -> pd.DataFrame:
    """Copy of the hourly table with the block offsets added to the Dst column."""
    out = df.copy()
    out["dst"] = df["dst"] + block_offsets(df.index, seed, **kwargs)
    return out


def consistent_rows(index: pd.DatetimeIndex, horizon: int, block: int = BLOCK, warmup: int = WARMUP) -> np.ndarray:
    """Rows whose whole history window and target lie inside one offset block.

    Only these rows have inputs and target shifted by the same offset. The others see a jump
    in the baseline in the middle of their window and are left out.
    """
    position = np.arange(len(index)) % block
    return (position >= warmup - 1) & (position + horizon < block)


def newest_hour_error(index: pd.DatetimeIndex, seed: int, level: float = 1.0, mean: float = NEWEST_ERROR[0], std: float = NEWEST_ERROR[1], coef: float = NEWEST_ERROR[2]) -> pd.Series:
    """Stationary AR(1) error in nT for every hour of `index`: mean `level * mean`, standard deviation `level * std`.

    e(t) - m = coef * (e(t-1) - m) + innovation, with the innovation variance chosen so the
    series has the requested standard deviation from the first hour on. The draw depends only
    on the seed and the length, never on the data.
    """
    noise = np.random.default_rng(seed).normal(size=len(index))
    x = np.empty(len(index))
    innovation = np.sqrt(1 - coef**2)
    previous = 0.0
    for i, z in enumerate(noise):
        previous = z if i == 0 else coef * previous + innovation * z
        x[i] = previous
    return pd.Series(level * (mean + std * x), index=index, name="newest_dst_error")
