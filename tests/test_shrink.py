"""The shrink factor: causal quiet condition, least squares fit, and the newest hour error."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from oculstorm.features import build_features
from oculstorm.shrink import BINS, ar1_error, change_bin, fit_alpha, fit_bin_alphas, least_squares_alpha, quiet_condition, shrink

SPEC = importlib.util.spec_from_file_location("compare_models_shrink", Path(__file__).resolve().parent.parent / "scripts" / "compare_models.py")
compare_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_models)


def _table(n=400) -> pd.DataFrame:
    hours = pd.date_range("2018-01-01", periods=n, freq="h")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({c: rng.normal(size=n) for c in ["bz_gsm", "by_gsm", "b_mag_avg", "proton_density", "proton_temp", "electric_field"]}, index=hours)
    df["flow_speed"] = 400 + 50 * rng.random(n)
    df["flow_pressure"] = 2 + rng.random(n)
    df["f107"] = 100.0
    df["dst"] = -20 + 15 * np.sin(np.arange(n) / 40) + rng.normal(0, 2, n)
    return df


def test_quiet_condition_edges_and_missing_values():
    lag0 = [-10, -30, -10, -10, -10]
    min24 = [-20, -20, -50, -20, -20]
    vbs = [0.1, 0.1, 0.1, 0.5, np.nan]
    assert list(quiet_condition(lag0, min24, vbs)) == [True, False, False, False, False]


def test_fit_alpha_recovers_the_factor_and_stays_in_range():
    rng = np.random.default_rng(0)
    base, change = rng.normal(size=5000), rng.normal(size=5000)
    y = base + 0.6 * change + rng.normal(scale=0.1, size=5000)
    assert abs(fit_alpha(y, base, change) - 0.6) < 0.01
    assert fit_alpha(base + 1.5 * change, base, change) == 1.0  # never amplifies
    assert fit_alpha(base - change, base, change) == 0.0
    assert abs(least_squares_alpha(base - change, base, change) + 1) < 1e-9  # the fit itself is reported too
    assert fit_alpha(y[:10], base[:10], change[:10]) == 1.0  # too few hours: no shrink


def test_fit_alpha_uses_only_the_masked_hours():
    rng = np.random.default_rng(1)
    base, change = np.zeros(2000), rng.normal(size=2000)
    mask = np.arange(2000) < 1000
    y = np.where(mask, 0.3 * change, change)
    assert abs(fit_alpha(y, base, change, mask) - 0.3) < 1e-9


def test_bin_alphas_follow_the_absolute_predicted_change():
    assert list(change_bin([0.0, -1.9, 2.0, -3.0])) == [BINS[0], BINS[0], BINS[1], BINS[1]]
    rng = np.random.default_rng(2)
    change = rng.normal(scale=3, size=4000)
    y = np.where(np.abs(change) >= 2, 0.9 * change, 0.2 * change)
    alphas = fit_bin_alphas(y, np.zeros(4000), change, np.ones(4000, dtype=bool))
    assert abs(alphas[BINS[0]] - 0.2) < 1e-9 and abs(alphas[BINS[1]] - 0.9) < 1e-9
    assert np.allclose(shrink(np.zeros(4000), change, alphas, np.ones(4000, dtype=bool)), y)


def test_shrink_leaves_the_other_hours_alone():
    base, change = np.array([1.0, 1.0]), np.array([4.0, 4.0])
    assert list(shrink(base, change, 0.5, [True, False])) == [3.0, 5.0]
    assert list(shrink(base, change, 1.0, [True, True])) == [5.0, 5.0]


def test_ar1_error_has_the_asked_moments():
    e = ar1_error(200_000, seed=0)
    assert abs(e.mean() + 3) < 0.1 and abs(e.std() - 4) < 0.1
    assert abs(np.corrcoef(e[:-1], e[1:])[0, 1] - 0.6) < 0.02
    assert np.array_equal(e, ar1_error(200_000, seed=0))


def test_newest_hour_features_equal_a_table_with_only_that_hour_changed():
    df = _table()
    X, y = build_features(df)
    error = pd.Series(ar1_error(len(df), seed=3), index=df.index)
    got = compare_models.newest_hour_features(X, df["dst"], error)
    dst_columns = [c for c in X if "dst" in c]
    for t in [df.index[0], df.index[5], df.index[30], df.index[200], df.index[-2]]:
        one = df.copy()
        one.loc[t, "dst"] += error[t]  # only hour t is unsettled
        want, _ = build_features(one, require_target=False)
        assert np.allclose(got.loc[t, dst_columns], want.loc[t, dst_columns], atol=1e-4, equal_nan=True), t
    others = [c for c in X if c not in ("dst_lag0", "dst_diff1", "dst_diff3", "dst_min24", "dst_mean24")]
    assert got[others].equals(X[others])  # older lags and the drivers are clean
    assert y.equals(build_features(df)[1])  # the target is clean


def test_shrink_is_off_by_default():
    parser_source = (Path(compare_models.__file__)).read_text()
    assert 'add_argument("--shrink", action="store_true"' in parser_source
