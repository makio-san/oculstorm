"""Real-time like perturbation of the solar wind columns of the hourly table (experiment, issue 24).

Real-time L1 solar wind is noisier than the OMNI science data the models are trained on:
density and temperature are less certain, speed and field a little, there are short gaps and
the propagation to the bow shock can be off by about an hour. `perturb_solar_wind` imitates
that on a copy of the table, so the training years can be seen a second time the way a live
feed would show them.

Nothing here looks ahead: the perturbed solar wind at hour t is made from the clean values at
hour t or at hour t-1 (a delay, never an advance) and from random numbers. Dst, F10.7 and
every other column are left alone.
"""
import numpy as np
import pandas as pd

from .data import VAL_START

SIGMA_LOG_PLASMA = 0.2  # proton_density and proton_temp are multiplied by exp(N(0, sigma))
SIGMA_SPEED = 0.03  # flow_speed is multiplied by 1 + N(0, sigma)
SIGMA_FIELD = 0.5  # nT, added to bz_gsm, by_gsm and b_mag_avg
DROPOUT_FRACTION = 0.02  # share of hours whose solar wind is removed, in runs of 1 to 3 hours
DELAY_PROBABILITY = 0.5  # chance that a 24 hour segment shows the solar wind one hour late
SEGMENT_HOURS = 24

FIELD = ["bz_gsm", "by_gsm", "b_mag_avg"]
PLASMA = ["proton_density", "proton_temp"]
DERIVED = ["flow_pressure", "electric_field"]
BLOCK = FIELD + ["flow_speed"] + PLASMA + DERIVED  # the same columns as features.SOLAR_WIND


def delay_hours(index: pd.DatetimeIndex, rng: np.random.Generator, probability: float = DELAY_PROBABILITY) -> np.ndarray:
    """Delay (0 or 1 hour) of every row, constant inside each SEGMENT_HOURS segment of the table."""
    segment = np.arange(len(index)) // SEGMENT_HOURS
    late = rng.random(int(segment[-1]) + 1 if len(index) else 0) < probability
    return late[segment].astype(int)


def dropout_mask(n: int, rng_start: np.random.Generator, rng_length: np.random.Generator, fraction: float = DROPOUT_FRACTION) -> np.ndarray:
    """True for the hours to remove: runs of 1 to 3 hours that cover about `fraction` of the rows."""
    starts = rng_start.random(n) < fraction / 2  # a run is 2 hours long on average
    lengths = rng_length.integers(1, 4, size=n)
    mask = np.zeros(n, dtype=bool)
    for offset in range(3):
        covered = starts & (lengths > offset)
        mask[offset:] |= covered[: n - offset]
    return mask


def perturb_solar_wind(df: pd.DataFrame, level: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Return a copy of the hourly table whose solar wind looks like a real-time feed.

    `level` scales the whole recipe: the noise widths, the dropout share and the chance of a
    delayed segment (1.0 is the recipe of issue 24, 0.5 half of it, 0 returns the clean values).
    In this order:

    1. per 24 hour segment the whole solar wind block is shown 0 or 1 hour late (the value at
       t becomes the clean value at t-1; an advance would be leakage and is never made);
    2. per hour, density and temperature are multiplied by exp(N(0, 0.2)), speed by
       1 + N(0, 0.03) and N(0, 0.5 nT) is added to Bz, By and |B|;
    3. flow_pressure and electric_field are recomputed from the perturbed values (pressure
       scales with n v^2, so the helium share of the hour is kept; E = -v Bz 1e-3 as in OMNI);
    4. the block is set to NaN in runs of 1 to 3 hours covering about 2% of the hours.

    Every random stream is drawn once in time order, so the result up to hour t is the same
    whatever comes after t in `df`. The index must be the regular hourly grid of `load_hourly`.
    """
    if level < 0:
        raise ValueError("level must not be negative")
    out = df.copy()
    if level == 0 or not len(df):
        return out
    n = len(df)
    r_delay, r_plasma, r_speed, r_field, r_start, r_length = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)
    )

    clean = df[BLOCK].astype("float64")
    late = delay_hours(df.index, r_delay, DELAY_PROBABILITY * level) == 1
    block = clean.copy()
    block[late] = clean.shift(1)[late]  # shift(1): the row of hour t-1, never t+1

    plasma_noise = np.exp(r_plasma.normal(0.0, SIGMA_LOG_PLASMA * level, size=(n, len(PLASMA))))
    speed_noise = 1 + r_speed.normal(0.0, SIGMA_SPEED * level, size=n)
    field_noise = r_field.normal(0.0, SIGMA_FIELD * level, size=(n, len(FIELD)))

    new = block.copy()
    new[PLASMA] = block[PLASMA] * plasma_noise
    new["flow_speed"] = block["flow_speed"] * speed_noise
    new[FIELD] = block[FIELD] + field_noise
    new["b_mag_avg"] = new["b_mag_avg"].clip(lower=0)
    new["flow_pressure"] = block["flow_pressure"] * plasma_noise[:, 0] * speed_noise**2
    new["electric_field"] = -new["flow_speed"] * new["bz_gsm"] * 1e-3

    new[dropout_mask(n, r_start, r_length, DROPOUT_FRACTION * level)] = np.nan
    out[BLOCK] = new.astype(df[BLOCK].dtypes.to_dict())
    return out


def train_copies(df: pd.DataFrame, level: float, copies: int = 1, seed: int = 0) -> list[pd.DataFrame]:
    """`copies` perturbed versions of the training years of `df` (seeds seed, seed+1, ...).

    The table is cut at VAL_START before it is perturbed, so the copies hold no validation or
    test hour at all. With level 0 or copies 0 the list is empty: the option is off.
    """
    if level <= 0 or copies <= 0:
        return []
    train = df[df.index < VAL_START]
    return [perturb_solar_wind(train, level, seed + k) for k in range(copies)]
