"""Sliding window inputs for the GRU: the last WINDOW hours ending at hour t."""
import numpy as np
import pandas as pd

from .data import VAL_START
from .features import HORIZON, SOLAR_WIND, _coupling

WINDOW = 24  # hours of history per sample
MAX_FFILL = 3  # hours a missing solar wind value may be carried forward
LOG_COLUMNS = ["proton_temp", "proton_density", "flow_pressure", "vbs", "newell"]


def build_channels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return per-hour input channels (standardised, no NaN) and the Dst series.

    Scaling statistics come from the training years only. Gaps are forward filled
    for at most MAX_FFILL hours (never backward), then set to the training mean,
    and two mask channels tell the network which values were actually observed.
    """
    raw = pd.concat([df[["dst"]], df[SOLAR_WIND], _coupling(df)], axis=1).astype("float64")
    raw[LOG_COLUMNS] = np.log1p(raw[LOG_COLUMNS].clip(lower=0))

    train = raw[raw.index < VAL_START]
    scaled = (raw - train.mean()) / train.std()
    filled = scaled.ffill(limit=MAX_FFILL).fillna(0.0)

    filled["imf_observed"] = df["bz_gsm"].notna().astype("float64")
    filled["plasma_observed"] = df["flow_speed"].notna().astype("float64")
    hour, doy = df.index.hour, df.index.dayofyear
    filled["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    filled["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    filled["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    filled["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return filled.astype("float32"), df["dst"]


def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (windows, meta).

    windows[i] has shape (WINDOW, n_channels) and covers hours t-WINDOW+1 .. t.
    meta is indexed by t and holds dst_lag0 = Dst(t) and dst_target = Dst(t+HORIZON).
    windows is a zero-copy view, so index it per batch instead of copying it whole.
    """
    channels, dst = build_channels(df)
    view = np.lib.stride_tricks.sliding_window_view(channels.to_numpy(), WINDOW, axis=0)
    windows = view.transpose(0, 2, 1)  # (n, WINDOW, channels)

    meta = pd.DataFrame({"dst_lag0": dst, "dst_target": dst.shift(-HORIZON)}).iloc[WINDOW - 1 :]
    keep = meta.notna().all(axis=1).to_numpy()
    meta = meta[keep]
    meta["row"] = np.flatnonzero(keep)  # position of each sample inside `windows`
    return windows, meta
