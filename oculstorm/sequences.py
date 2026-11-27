"""Sliding window inputs for the GRU: the last WINDOW hours ending at hour t."""
import numpy as np
import pandas as pd

from .data import VAL_START
from .features import HORIZON, MEMORY_DRIVERS, SOLAR_WIND, _coupling, _ring_current, decaying_memory

WINDOW = 24  # hours of history per sample
MAX_FFILL = 3  # hours a missing solar wind value may be carried forward
LOG_COLUMNS = ["proton_temp", "proton_density", "flow_pressure", "vbs", "newell"]
RING_CURRENT_CHANNELS = ["dst_star", "om_step"]  # appended last, so channel 0 stays Dst


def _raw_channels(df: pd.DataFrame, ring_current: bool = False, memory: tuple[float, ...] = ()) -> pd.DataFrame:
    parts = [df[["dst"]], df[SOLAR_WIND], _coupling(df)]
    if ring_current:
        parts.append(_ring_current(df)[RING_CURRENT_CHANNELS])
    raw = pd.concat(parts, axis=1).astype("float64")
    # Optional decaying driver memory, computed causally on the full hourly index.
    names = []
    for name in MEMORY_DRIVERS:
        for tau in memory:
            raw[f"{name}_ewm{tau:g}"] = decaying_memory(raw[name], tau)
            names.append(f"{name}_ewm{tau:g}")
    log_columns = LOG_COLUMNS + names
    raw[log_columns] = np.log1p(raw[log_columns].clip(lower=0))
    return raw


def _memory_in(stats: dict) -> tuple[float, ...]:
    """Time constants of the driver memory channels that saved `stats` were made with."""
    prefix = f"{MEMORY_DRIVERS[0]}_ewm"
    return tuple(float(k[len(prefix):]) for k in stats["mean"] if k.startswith(prefix))


def channel_stats(
    df: pd.DataFrame, ring_current: bool = False, memory: tuple[float, ...] = (), end=VAL_START,
) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of every scaled channel over the training years only.

    `end` (exclusive) defaults to VAL_START; a backtest that fits on fewer years passes the
    end of its fit years, so its early stopping years do not shape the scaling either.
    """
    train = _raw_channels(df, ring_current, memory)
    train = train[train.index < end]
    return {"mean": train.mean().to_dict(), "std": train.std().to_dict()}


def build_channels(
    df: pd.DataFrame, stats: dict | None = None, ring_current: bool = False, memory: tuple[float, ...] = ()
) -> tuple[pd.DataFrame, pd.Series]:
    """Return per-hour input channels (standardised, no NaN) and the Dst series.

    Scaling statistics come from the training years only; pass the saved `stats`
    when `df` does not contain them (live data). Gaps are forward filled for at
    most MAX_FFILL hours (never backward), then set to the training mean, and two
    mask channels tell the network which values were actually observed.
    With ring_current=True the O'Brien and McPherron (2000) Dst* and Burton step are added.
    `memory` adds decaying means of VBs and Newell (off by default); with saved `stats`
    the time constants they were made with are used.
    """
    memory = _memory_in(stats) if stats else memory
    raw = _raw_channels(df, ring_current, memory)
    stats = stats or channel_stats(df, ring_current, memory)
    scaled = (raw - pd.Series(stats["mean"])[raw.columns]) / pd.Series(stats["std"])[raw.columns]
    filled = scaled.ffill(limit=MAX_FFILL).fillna(0.0)

    filled["imf_observed"] = df["bz_gsm"].notna().astype("float64")
    filled["plasma_observed"] = df["flow_speed"].notna().astype("float64")
    hour, doy = df.index.hour, df.index.dayofyear
    filled["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    filled["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    filled["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    filled["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return filled.astype("float32"), df["dst"]


def build_windows(
    df: pd.DataFrame, stats: dict | None = None, require_target: bool = True, horizon: int = HORIZON,
    ring_current: bool = False, memory: tuple[float, ...] = (),
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (windows, meta).

    windows[i] has shape (WINDOW, n_channels) and covers hours t-WINDOW+1 .. t.
    meta is indexed by t and holds dst_lag0 = Dst(t) and dst_target = Dst(t+horizon).
    windows is a zero-copy view, so index it per batch instead of copying it whole.
    With require_target=False the newest hours, whose target is not known yet, are kept.
    """
    channels, dst = build_channels(df, stats, ring_current, memory)
    view = np.lib.stride_tricks.sliding_window_view(channels.to_numpy(), WINDOW, axis=0)
    windows = view.transpose(0, 2, 1)  # (n, WINDOW, channels)

    meta = pd.DataFrame({"dst_lag0": dst, "dst_target": dst.shift(-horizon)}).iloc[WINDOW - 1 :]
    keep = meta["dst_lag0"].notna()
    if require_target:
        keep &= meta["dst_target"].notna()
    keep = keep.to_numpy()
    meta = meta[keep]
    meta["row"] = np.flatnonzero(keep)  # position of each sample inside `windows`
    return windows, meta


def concat_windows(pieces: list[tuple[np.ndarray, pd.DataFrame]]) -> tuple[np.ndarray, pd.DataFrame]:
    """Join several (windows, meta) pairs of `build_windows` into one, still without copying windows.

    Used to add perturbed copies of the training years to the clean table. The channels of the
    pieces are laid end to end and `row` of every meta is moved to its place in the joined
    view, so no sample of one piece can reach into another: windows that straddle a seam are
    never pointed at by a row.
    """
    channels, metas, offset = [], [], 0
    for windows, meta in pieces:
        channels.append(np.concatenate([windows[0], windows[1:, -1, :]]))
        metas.append(meta.assign(row=meta["row"] + offset))
        offset += len(channels[-1])
    view = np.lib.stride_tricks.sliding_window_view(np.concatenate(channels), WINDOW, axis=0)
    return view.transpose(0, 2, 1), pd.concat(metas)
