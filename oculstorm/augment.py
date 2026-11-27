"""Training set augmentations of the hourly table. All are off by default.

Dst baseline offset augmentation (issue 23). Quick-look Dst has a quiet time baseline that
drifts by tens of nT, while the models learn a recovery toward the final Dst baseline.
Persistence is invariant to a constant offset, the models are not. The augmentation adds one
random offset per 168 hour block to the Dst column of a copy of the table, so inputs and target
of a row move together and every delta target stays what it was.

Real-time like solar wind perturbation (experiment, issue 24). Real-time L1 solar wind is
noisier than the OMNI science data the models are trained on: density and temperature are less
certain, speed and field a little, there are short gaps and the propagation to the bow shock
can be off by about an hour. `perturb_solar_wind` imitates that on a copy of the table, so the
training years can be seen a second time the way a live feed would show them. Nothing there
looks ahead: the perturbed solar wind at hour t is made from the clean values at hour t or at
hour t-1 (a delay, never an advance) and from random numbers. Dst, F10.7 and every other column
are left alone.

Real-time like IMF and plasma gaps (experiment, issue 41). The real-time feed loses the field
and the plasma instrument separately, often for hours at a time, which OMNI almost never does.
`blank_gaps` removes spans of each block from a copy of the table at the rates and lengths
measured in issue 31, without any noise, so XGBoost learns its missing value branches on
gaps that look like the live ones.

Unsettled newest hour (issue 30). Live, the newest quick-look Dst hour is not settled when the
forecast is issued. newest_hour_error draws an AR(1) error for that hour; build_features adds
it to Dst(t) as seen from row t only (newest_dst_error), the target stays clean.
"""
import numpy as np
import pandas as pd

from .data import VAL_START

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


# Real-time like solar wind perturbation (issue 24)

SIGMA_LOG_PLASMA = 0.2  # proton_density and proton_temp are multiplied by exp(N(0, sigma))
SIGMA_SPEED = 0.03  # flow_speed is multiplied by 1 + N(0, sigma)
SIGMA_FIELD = 0.5  # nT, added to bz_gsm, by_gsm and b_mag_avg
DROPOUT_FRACTION = 0.02  # share of hours whose solar wind is removed, in runs of 1 to 3 hours
DELAY_PROBABILITY = 0.5  # chance that a 24 hour segment shows the solar wind one hour late
SEGMENT_HOURS = 24

FIELD = ["bz_gsm", "by_gsm", "b_mag_avg"]
PLASMA = ["proton_density", "proton_temp"]
DERIVED = ["flow_pressure", "electric_field"]
SW_COLUMNS = FIELD + ["flow_speed"] + PLASMA + DERIVED  # the same columns as features.SOLAR_WIND


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

    clean = df[SW_COLUMNS].astype("float64")
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
    out[SW_COLUMNS] = new.astype(df[SW_COLUMNS].dtypes.to_dict())
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


# Real-time like IMF and plasma gaps (issue 41)
#
# Issue 31 (reports/experiments/31-realtime-ace-inputs.json, "coverage") measured the real-time
# ACE gaps on the validation years 2017 to 2020 (35,064 hours): IMF 18.62% of hours missing in
# 1,629 gaps, 22.12% of the missing hours in gaps of at most 3 hours, longest 145 hours; plasma
# 26.95% in 3,641 gaps, 40.11% in gaps of at most 3 hours, longest 23 hours. Only these summary
# numbers are used here, never the validation mask itself.

IMF_COLUMNS = ["bz_gsm", "by_gsm", "b_mag_avg", "electric_field"]
PLASMA_COLUMNS = ["flow_speed", "proton_density", "proton_temp", "flow_pressure", "electric_field"]
SHORT_DECAY = 0.5  # gaps of 1, 2 and 3 hours have weights 1, 0.5, 0.25 (an assumption, not measured)
VALIDATION_HOURS = 35064  # hours of 2017 to 2020, the base of the issue 31 counts


def gap_model(missing_share: float, gaps: int, short_share: float, longest: int, hours: int = VALIDATION_HOURS) -> dict:
    """Gap length model of one block, matched to the measured mean length and short gap share.

    A gap is 1 to 3 hours long with probability `p_short` (weights SHORT_DECAY**k), otherwise
    4 hours plus a geometric tail, capped at `longest`. `p_short` and the mean long length are
    solved so that the mean gap length (missing hours / gaps) and the share of missing hours
    in gaps of at most 3 hours equal the measured ones.
    """
    mean_gap = missing_share * hours / gaps
    w = SHORT_DECAY ** np.arange(3)
    short_mean = float(np.dot(np.arange(1, 4), w) / w.sum())
    p_short = short_share * mean_gap / short_mean
    long_mean = (mean_gap - p_short * short_mean) / (1 - p_short)
    if not (0 < p_short < 1 and long_mean > 4):
        raise ValueError("gap statistics cannot be matched by the mixture")
    return {
        "missing_share": missing_share, "mean_gap": mean_gap, "p_short": p_short,
        "short_weights": w / w.sum(), "long_mean": long_mean, "longest": longest,
    }


GAP_MODELS = {
    "imf": gap_model(0.1862, 1629, 0.2212, 145),
    "plasma": gap_model(0.2695, 3641, 0.4011, 23),
}


def gap_mask(n: int, model: dict, rng: np.random.Generator, rate: float = 1.0) -> np.ndarray:
    """True for missing hours: observed runs and gaps alternate over `n` hours, in time order.

    Observed runs are geometric with the mean that gives the missing share of `model` times
    `rate` (1 is the measured share, 0.5 half of it); gap lengths stay as measured. The draws
    are made in time order, so the first m hours of the mask are the same for any n >= m.
    """
    if rate <= 0 or n == 0:
        return np.zeros(n, dtype=bool)
    share = model["missing_share"] * rate
    observed_p = share / (model["mean_gap"] * (1 - share))  # 1 / mean observed run length
    long_p = 1 / (model["long_mean"] - 3)  # 4 + Geometric(long_p) - 1 has the mean long_mean
    mask = np.zeros(n, dtype=bool)
    pos = int(rng.geometric(observed_p)) - 1  # the table starts inside an observed run
    while pos < n:
        if rng.random() < model["p_short"]:
            length = int(rng.choice(np.arange(1, 4), p=model["short_weights"]))
        else:
            length = 3 + int(rng.geometric(long_p))
        length = min(length, model["longest"])
        mask[pos : pos + length] = True
        pos += length + int(rng.geometric(observed_p))
    return mask


def blank_gaps(df: pd.DataFrame, rate: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Copy of the hourly table with IMF and plasma spans set to NaN like a real-time feed.

    The IMF block (bz_gsm, by_gsm, b_mag_avg, electric_field) and the plasma block (flow_speed,
    proton_density, proton_temp, flow_pressure, electric_field) get independent masks from
    GAP_MODELS; electric_field needs both, so it is missing when either is. Values are only
    removed, never changed or moved in time; Dst and every other column stay as they are.
    The coupling terms follow when build_features is run on the copy.
    """
    if rate < 0:
        raise ValueError("rate must not be negative")
    out = df.copy()
    if rate == 0 or not len(df):
        return out
    r_imf, r_plasma = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    imf = gap_mask(len(df), GAP_MODELS["imf"], r_imf, rate)
    plasma = gap_mask(len(df), GAP_MODELS["plasma"], r_plasma, rate)
    out.loc[imf, IMF_COLUMNS] = np.nan
    out.loc[plasma, PLASMA_COLUMNS] = np.nan
    return out


def gap_copies(df: pd.DataFrame, copies: int, rate: float = 1.0, seed: int = 0) -> list[pd.DataFrame]:
    """`copies` gap blanked versions of the training years of `df` (seeds seed, seed+1, ...).

    The table is cut at VAL_START first, so no validation or test hour is in a copy. Empty
    (the option is off) for copies 0 or rate 0.
    """
    if copies <= 0 or rate <= 0:
        return []
    train = df[df.index < VAL_START]
    return [blank_gaps(train, rate, seed + k) for k in range(copies)]


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
