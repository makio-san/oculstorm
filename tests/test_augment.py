"""The Dst offset augmentation moves inputs and target together and never reaches validation."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from oculstorm.augment import BLOCK, OFFSET_RANGE, block_offsets, consistent_rows, offset_table
from oculstorm.features import build_features

SPEC = importlib.util.spec_from_file_location("train_xgb_aug", Path(__file__).resolve().parent.parent / "scripts" / "train_xgb.py")
train_xgb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_xgb)


def _table(start="2016-06-01", end="2017-03-01") -> pd.DataFrame:
    hours = pd.date_range(start, end, freq="h")
    rng = np.random.default_rng(1)
    n = len(hours)
    df = pd.DataFrame({c: rng.normal(size=n) for c in ["bz_gsm", "by_gsm", "b_mag_avg", "proton_density", "proton_temp", "electric_field"]}, index=hours)
    df["flow_speed"] = 400 + 50 * rng.random(n)
    df["flow_pressure"] = 2 + rng.random(n)
    df["f107"] = 100.0
    df["dst"] = -20 + 15 * np.sin(np.arange(n) / 40) + rng.normal(0, 2, n)
    return df


def test_offsets_are_constant_per_block_and_in_range():
    off = block_offsets(pd.date_range("2000-01-01", periods=5 * BLOCK + 10, freq="h"), seed=3)
    assert len(off) == 5 * BLOCK + 10
    assert all(len(set(off[i * BLOCK:(i + 1) * BLOCK])) == 1 for i in range(5))
    assert len(set(off[::BLOCK])) == 6
    assert OFFSET_RANGE[0] <= off.min() and off.max() <= OFFSET_RANGE[1]


def test_consistent_rows_keep_delta_target_and_diffs_unchanged():
    df = _table()
    shifted = offset_table(df, seed=5)
    X0, y0 = build_features(df, horizon=1)
    X1, y1 = build_features(shifted, horizon=1)
    rows = pd.Series(consistent_rows(df.index, 1), index=df.index).reindex(X0.index).to_numpy()
    assert rows.mean() < 0.9
    for name in ["dst_diff1", "dst_diff3"]:
        np.testing.assert_allclose(X1[name][rows], X0[name][rows], atol=1e-3)
    np.testing.assert_allclose((y1 - X1["dst_lag0"])[rows], (y0 - X0["dst_lag0"])[rows], atol=1e-3)
    for name in ["dst_lag0", "dst_lag5", "dst_min24", "dst_mean24"]:
        shift = (X1[name] - X0[name])[rows]
        np.testing.assert_allclose((y1 - y0)[rows], shift, atol=1e-3)
    assert not np.allclose((y1 - y0)[rows], 0)
    np.testing.assert_array_equal(X1["bz_gsm_lag0"], X0["bz_gsm_lag0"])


def test_ring_current_terms_follow_the_shifted_dst():
    df = _table()
    X0, _ = build_features(df, ring_current=True)
    X1, _ = build_features(offset_table(df, seed=5), ring_current=True)
    rows = pd.Series(consistent_rows(df.index, 1), index=df.index).reindex(X0.index).to_numpy()
    shift = (X1["dst_lag0"] - X0["dst_lag0"])[rows]
    np.testing.assert_allclose((X1["dst_star_lag3"] - X0["dst_star_lag3"])[rows], shift, atol=1e-3)
    assert not np.allclose((X1["om_step_lag0"] - X0["om_step_lag0"])[rows], 0)


def test_copies_stay_in_the_train_years():
    df = _table()
    X, y = train_xgb.offset_copies(df, copies=2, horizon=1)
    assert X.index.max() < pd.Timestamp("2017-01-01") and len(X) == len(y)
    assert X.index.is_unique is False  # two copies of the same hours
    X_clean, _ = build_features(df)
    assert list(X.columns) == list(X_clean.columns)


def test_off_by_default():
    assert train_xgb.build_parser().parse_args([]).dst_offset_aug == 0
