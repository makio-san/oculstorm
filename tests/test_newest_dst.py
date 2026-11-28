"""Issue 30: a Dst history that ends one hour earlier, and an erroneous newest Dst hour.

Both options are off by default, never look past hour t and never touch the target.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oculstorm.augment import NEWEST_ERROR, newest_hour_error
from oculstorm.features import build_features

SPEC = importlib.util.spec_from_file_location("train_xgb_newest", Path(__file__).resolve().parent.parent / "scripts" / "train_xgb.py")
train_xgb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_xgb)

DST_COLUMNS = [f"dst_lag{lag}" for lag in range(6)] + ["dst_diff1", "dst_diff3", "dst_min24", "dst_mean24"]


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


def _corrupt_after(df: pd.DataFrame, cut: pd.Timestamp, columns=None) -> pd.DataFrame:
    out = df.copy()
    out.loc[out.index > cut, list(df.columns) if columns is None else columns] = -1234.0
    return out


def test_both_options_are_off_by_default():
    args = train_xgb.build_parser().parse_args([])
    assert args.dst_skip == 0 and args.dst_newest_noise == 0
    df = _table()
    X, y = build_features(df)
    X0, y0 = build_features(df, dst_skip=0, newest_dst_error=None)
    assert X.equals(X0) and y.equals(y0)


def test_dst_skip_moves_every_dst_feature_back_and_nothing_else():
    df = _table()
    X, y = build_features(df)
    X1, y1 = build_features(df, dst_skip=1)
    assert list(X1.columns) == list(X.columns)
    # Every Dst feature of row t is the default feature of row t-1.
    previous = X[DST_COLUMNS].shift(1).loc[X1.index]
    pd.testing.assert_frame_equal(X1[DST_COLUMNS], previous)
    assert (X1["dst_lag0"] == df["dst"].shift(1).astype("float32").loc[X1.index]).all()
    # Solar wind, calendar and the target are where they were.
    others = [c for c in X.columns if c not in DST_COLUMNS]
    assert X1[others].equals(X.loc[X1.index, others])
    assert y1.equals(df["dst"].shift(-1).loc[y1.index].rename("dst_target"))
    assert y1.equals(y.loc[y1.index])


def test_dst_skip_features_ignore_dst_from_hour_t_on():
    df = _table()
    cut = pd.Timestamp("2016-11-10 12:00")
    X1, _ = build_features(df, dst_skip=1, require_target=False)
    # Corrupt Dst at the cut hour itself and later, and everything else after it.
    corrupted = _corrupt_after(df, cut)
    corrupted.loc[cut, "dst"] = -999.0
    X1_corrupted, _ = build_features(corrupted, dst_skip=1, require_target=False)
    past = X1[X1.index <= cut]
    assert past.equals(X1_corrupted.loc[past.index])
    assert not X1.loc[X1.index > cut, "dst_lag0"].head(5).equals(X1_corrupted.loc[X1_corrupted.index > cut, "dst_lag0"].head(5))
    # The default layout does see Dst(t).
    X0, _ = build_features(df, require_target=False)
    X0_corrupted, _ = build_features(corrupted, require_target=False)
    assert X0.loc[cut, "dst_lag0"] != X0_corrupted.loc[cut, "dst_lag0"]


def test_newest_hour_error_is_a_stationary_ar1_series():
    index = pd.date_range("2000-01-01", periods=200_000, freq="h")
    e = newest_hour_error(index, seed=3)
    mean, std, coef = NEWEST_ERROR
    assert e.mean() == pytest.approx(mean, abs=0.1)
    assert e.std() == pytest.approx(std, abs=0.1)
    assert np.corrcoef(e[1:], e[:-1])[0, 1] == pytest.approx(coef, abs=0.02)
    assert e.iloc[:5000].std() == pytest.approx(std, abs=0.3)  # no warm-up
    half = newest_hour_error(index, seed=3, level=0.5)
    np.testing.assert_allclose(half, 0.5 * e)
    assert newest_hour_error(index, seed=3).equals(e) and not newest_hour_error(index, seed=4).equals(e)
    # The first hours do not depend on how long the series is.
    np.testing.assert_array_equal(newest_hour_error(index[:100], seed=3), e.iloc[:100])


def test_error_reaches_only_the_newest_hour_of_each_row():
    df = _table()
    error = newest_hour_error(df.index, seed=7)
    X, y = build_features(df)
    Xe, ye = build_features(df, newest_dst_error=error)
    assert Xe.index.equals(X.index) and ye.equals(y)  # target clean, same rows
    e = error.loc[X.index].to_numpy()
    full = slice(30, None)  # rows with a full 24 hour window behind them
    np.testing.assert_allclose(Xe["dst_lag0"] - X["dst_lag0"], e, atol=1e-3)
    for name in ["dst_diff1", "dst_diff3"]:
        np.testing.assert_allclose((Xe[name] - X[name])[full], e[full], atol=1e-3)
    unchanged = [c for c in X.columns if c not in ("dst_lag0", "dst_diff1", "dst_diff3", "dst_min24", "dst_mean24")]
    assert Xe[unchanged].equals(X[unchanged])  # older lags and the solar wind stay clean
    np.testing.assert_allclose((Xe["dst_mean24"] - X["dst_mean24"])[full], e[full] / 24, atol=1e-3)
    dst = df["dst"]
    expected_min = np.minimum(dst + error, dst.shift(1).rolling(23, min_periods=1).min()).loc[X.index]
    np.testing.assert_allclose(Xe["dst_min24"][full], expected_min[full], atol=1e-3)
    assert not np.allclose(Xe["dst_min24"], X["dst_min24"])


def test_noisy_features_do_not_depend_on_the_future():
    df = _table()
    cut = pd.Timestamp("2016-11-10 12:00")
    error = newest_hour_error(df.index, seed=7)
    Xe, _ = build_features(df, newest_dst_error=error, require_target=False)
    corrupted_error = error.copy()
    corrupted_error[corrupted_error.index > cut] = 500.0
    Xc, _ = build_features(_corrupt_after(df, cut), newest_dst_error=corrupted_error, require_target=False)
    past = Xe[Xe.index <= cut]
    assert past.equals(Xc.loc[past.index])


def test_options_that_do_not_combine_are_refused():
    df = _table()
    error = newest_hour_error(df.index, seed=7)
    with pytest.raises(ValueError):
        build_features(df, dst_skip=1, newest_dst_error=error)
    with pytest.raises(ValueError):
        build_features(df, dst_skip=1, ring_current=True)
    with pytest.raises(ValueError):
        build_features(df, newest_dst_error=error, ring_current=True)
    with pytest.raises(ValueError):
        build_features(df, dst_skip=-1)
    with pytest.raises(ValueError):
        build_features(df, newest_dst_error=error.iloc[:-5])


def test_noisy_copies_stay_in_the_train_years_with_a_clean_target():
    df = _table()
    X, y = train_xgb.noisy_copies(df, copies=2, horizon=1)
    X_clean, y_clean = build_features(df)
    n = len(train_xgb.split(X_clean)[0])
    assert X.index.max() < pd.Timestamp("2017-01-01") and len(X) == len(y) == 2 * n
    assert list(X.columns) == list(X_clean.columns)
    first, second = X.iloc[:n], X.iloc[n:]
    assert y.iloc[:n].equals(y_clean.iloc[:n]) and y.iloc[n:].equals(y_clean.iloc[:n])
    spread = [(part["dst_lag0"] - X_clean["dst_lag0"].iloc[:n]).std() for part in (first, second)]
    assert spread[0] == pytest.approx(NEWEST_ERROR[1], rel=0.15)
    assert spread[1] == pytest.approx(NEWEST_ERROR[1] / 2, rel=0.15)
    assert first["dst_lag1"].equals(X_clean["dst_lag1"].iloc[:n])
    with pytest.raises(ValueError):
        train_xgb.noisy_copies(df, copies=3, horizon=1)


def test_newest_flag_is_off_by_default_and_adds_only_a_constant_column():
    args = train_xgb.build_parser().parse_args([])
    assert args.dst_newest_flag == 0
    df = _table()
    X, y = build_features(df)
    assert "dst_newest_unsettled" not in X.columns
    Xf, yf = build_features(df, newest_flag=0.0)
    assert yf.equals(y) and Xf.drop(columns="dst_newest_unsettled").equals(X)
    assert (Xf["dst_newest_unsettled"] == 0).all()
    X1, _ = build_features(df, newest_flag=1.0, newest_dst_error=newest_hour_error(df.index, seed=7))
    assert (X1["dst_newest_unsettled"] == 1).all()


def test_newest_flag_features_do_not_depend_on_the_future():
    df = _table()
    cut = pd.Timestamp("2016-11-10 12:00")
    error = newest_hour_error(df.index, seed=7)
    Xe, _ = build_features(df, newest_dst_error=error, newest_flag=1.0, require_target=False)
    corrupted_error = error.copy()
    corrupted_error[corrupted_error.index > cut] = 500.0
    Xc, _ = build_features(_corrupt_after(df, cut), newest_dst_error=corrupted_error, newest_flag=1.0, require_target=False)
    past = Xe[Xe.index <= cut]
    assert past.equals(Xc.loc[past.index])


def test_flagged_copies_carry_their_error_level_and_match_the_plain_copies():
    df = _table()
    X, y = train_xgb.noisy_copies(df, copies=2, horizon=1, flag=True)
    X_plain, y_plain = train_xgb.noisy_copies(df, copies=2, horizon=1)
    assert y.equals(y_plain) and X.drop(columns="dst_newest_unsettled").equals(X_plain)
    n = len(X) // 2
    assert (X["dst_newest_unsettled"].iloc[:n] == 1.0).all() and (X["dst_newest_unsettled"].iloc[n:] == 0.5).all()
    assert X.index.max() < pd.Timestamp("2017-01-01")
    X_clean, _ = build_features(df, newest_flag=0.0)
    assert list(X.columns) == list(X_clean.columns)


@pytest.mark.parametrize("extra", [["--dst-newest-noise", "1"], ["--dst-skip", "1"], ["--ring-current"], ["--dst-offset-aug", "1"]])
def test_newest_flag_refuses_options_it_does_not_combine_with(extra, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_xgb.py", "--dst-newest-flag", "1", *extra])
    with pytest.raises(SystemExit):
        train_xgb.main()
