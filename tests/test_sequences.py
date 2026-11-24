"""Guards against look-ahead leakage in the GRU window pipeline."""
import numpy as np
import pandas as pd

from oculstorm.data import load_hourly
import pytest

from oculstorm.features import HORIZONS, MEMORY_TAUS
from oculstorm.sequences import WINDOW, build_channels, build_windows, channel_stats

CUT = pd.Timestamp("2024-05-10 12:00")


def test_windows_do_not_depend_on_the_future():
    df = load_hourly()
    windows, meta = build_windows(df)

    corrupted = df.copy()
    floats = corrupted.select_dtypes("float").columns
    corrupted.loc[corrupted.index > CUT, floats] = -1234.0
    windows_corrupted, _ = build_windows(corrupted)

    past = meta.loc[meta.index <= CUT, "row"].to_numpy()[-500:]
    future = meta.loc[meta.index > CUT, "row"].to_numpy()[:50]
    assert np.array_equal(windows[past], windows_corrupted[past])
    assert not np.array_equal(windows[future], windows_corrupted[future])


@pytest.mark.parametrize("horizon", HORIZONS)
def test_window_ends_at_t_and_target_is_one_horizon_ahead(horizon):
    df = load_hourly()
    windows, meta = build_windows(df, horizon=horizon)
    assert not np.isnan(windows[meta["row"].to_numpy()[-1000:]]).any()

    t = CUT
    row = int(meta.loc[t, "row"])
    assert windows[row].shape[0] == WINDOW
    assert meta.loc[t, "dst_lag0"] == df.loc[t, "dst"]
    assert meta.loc[t, "dst_target"] == df.loc[t + pd.Timedelta(hours=horizon), "dst"]
    # Channel 0 is standardised Dst: the last step of the window must be hour t,
    # so it has to move in step with the raw Dst over the window.
    raw = df["dst"].loc[t - pd.Timedelta(hours=WINDOW - 1) : t].to_numpy()
    assert np.corrcoef(windows[row][:, 0], raw)[0, 1] > 0.9999


def test_ring_current_windows_do_not_depend_on_the_future():
    df = load_hourly()
    windows, meta = build_windows(df, ring_current=True)
    base, _ = build_windows(df)
    assert windows.shape[2] == base.shape[2] + 2

    corrupted = df.copy()
    floats = corrupted.select_dtypes("float").columns
    corrupted.loc[corrupted.index > CUT, floats] = -1234.0
    windows_corrupted, _ = build_windows(corrupted, ring_current=True)

    past = meta.loc[meta.index <= CUT, "row"].to_numpy()[-500:]
    assert np.array_equal(windows[past], windows_corrupted[past])
    assert not np.isnan(windows[meta["row"].to_numpy()[-1000:]]).any()


def test_driver_memory_windows_do_not_depend_on_the_future():
    df = load_hourly()
    windows, meta = build_windows(df, memory=MEMORY_TAUS)
    base, _ = build_windows(df)
    assert windows.shape[2] == base.shape[2] + 2 * len(MEMORY_TAUS)

    corrupted = df.copy()
    floats = corrupted.select_dtypes("float").columns
    corrupted.loc[corrupted.index > CUT, floats] = -1234.0
    windows_corrupted, _ = build_windows(corrupted, memory=MEMORY_TAUS)

    past = meta.loc[meta.index <= CUT, "row"].to_numpy()[-500:]
    future = meta.loc[meta.index > CUT, "row"].to_numpy()[:50]
    assert np.array_equal(windows[past], windows_corrupted[past])
    assert not np.array_equal(windows[future], windows_corrupted[future])


def test_saved_stats_bring_their_driver_memory():
    df = load_hourly()
    stats = channel_stats(df, memory=MEMORY_TAUS)
    channels, _ = build_channels(df, stats)
    assert channels.equals(build_channels(df, memory=MEMORY_TAUS)[0])
    # Without memory the channels are exactly today's.
    assert set(channel_stats(df)["mean"]) == set(stats["mean"]) - {c for c in stats["mean"] if "_ewm" in c}


def test_ring_current_and_driver_memory_combine():
    df = load_hourly()
    both, _ = build_channels(df, ring_current=True, memory=MEMORY_TAUS)
    rc, _ = build_channels(df, ring_current=True)
    assert both.shape[1] == rc.shape[1] + 2 * len(MEMORY_TAUS)
    stats = channel_stats(df, ring_current=True, memory=MEMORY_TAUS)
    assert build_channels(df, stats, ring_current=True)[0].equals(both)
