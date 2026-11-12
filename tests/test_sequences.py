"""Guards against look-ahead leakage in the GRU window pipeline."""
import numpy as np
import pandas as pd

from oculstorm.data import load_hourly
from oculstorm.features import HORIZON
from oculstorm.sequences import WINDOW, build_windows

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


def test_window_ends_at_t_and_target_is_one_horizon_ahead():
    df = load_hourly()
    windows, meta = build_windows(df)
    assert not np.isnan(windows[meta["row"].to_numpy()[-1000:]]).any()

    t = CUT
    row = int(meta.loc[t, "row"])
    assert windows[row].shape[0] == WINDOW
    assert meta.loc[t, "dst_lag0"] == df.loc[t, "dst"]
    assert meta.loc[t, "dst_target"] == df.loc[t + pd.Timedelta(hours=HORIZON), "dst"]
    # Channel 0 is standardised Dst: the last step of the window must be hour t,
    # so it has to move in step with the raw Dst over the window.
    raw = df["dst"].loc[t - pd.Timedelta(hours=WINDOW - 1) : t].to_numpy()
    assert np.corrcoef(windows[row][:, 0], raw)[0, 1] > 0.9999
