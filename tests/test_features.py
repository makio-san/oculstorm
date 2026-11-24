"""Guards against look-ahead leakage in the feature pipeline."""
import numpy as np
import pandas as pd

from oculstorm.data import load_hourly
import pytest

from oculstorm.features import HORIZONS, MEMORY_TAUS, N_LAGS, build_features, decaying_memory

CUT = pd.Timestamp("2024-05-10 12:00")


def test_features_do_not_depend_on_the_future():
    df = load_hourly("2024-01-01")
    X, _ = build_features(df)

    corrupted = df.copy()
    floats = corrupted.select_dtypes("float").columns
    corrupted.loc[corrupted.index > CUT, floats] = -1234.0
    X_corrupted, _ = build_features(corrupted)

    past = X[X.index <= CUT]
    assert past.equals(X_corrupted.loc[past.index])
    assert not X[X.index > CUT].head(50).equals(X_corrupted[X_corrupted.index > CUT].head(50))


@pytest.mark.parametrize("horizon", HORIZONS)
def test_target_is_dst_one_horizon_ahead(horizon):
    df = load_hourly("2024-01-01")
    X, y = build_features(df, horizon=horizon)
    expected = df["dst"].shift(-horizon).loc[y.index]
    assert (y == expected).all()
    assert y.index.max() == df.index.max() - pd.Timedelta(hours=horizon)
    # The inputs are the same whatever the lead time; only the target moves.
    X1, _ = build_features(df, horizon=1)
    assert X.equals(X1.loc[X.index])


def test_ring_current_features_do_not_depend_on_the_future():
    df = load_hourly("2024-01-01")
    X, _ = build_features(df, ring_current=True)
    assert {"dst_star_lag0", "om_step_lag5", "q_om", "tau_om", "dpress"} <= set(X.columns)

    corrupted = df.copy()
    floats = corrupted.select_dtypes("float").columns
    corrupted.loc[corrupted.index > CUT, floats] = -1234.0
    X_corrupted, _ = build_features(corrupted, ring_current=True)

    past = X[X.index <= CUT]
    assert past.equals(X_corrupted.loc[past.index])


def test_ring_current_is_off_by_default_and_matches_the_published_formulas():
    df = load_hourly("2024-01-01")
    X, _ = build_features(df)
    X_om, _ = build_features(df, ring_current=True)
    assert not any(c.startswith(("dst_star", "om_step")) for c in X.columns)
    assert X.equals(X_om[X.columns])

    t = CUT
    p, dst = df.loc[t, "flow_pressure"], df.loc[t, "dst"]
    vbs = df.loc[t, "flow_speed"] * max(-df.loc[t, "bz_gsm"], 0) * 1e-3
    dst_star = dst - 7.26 * np.sqrt(p) + 11
    q = -4.4 * max(vbs - 0.49, 0)
    tau = 2.40 * np.exp(9.74 / (4.69 + vbs))
    prev_p = df.loc[t - pd.Timedelta(hours=1), "flow_pressure"]
    row = X_om.loc[t]
    assert row["dst_star_lag0"] == pytest.approx(dst_star, rel=1e-5)
    assert row["q_om"] == pytest.approx(q, rel=1e-5, abs=1e-6)
    assert row["tau_om"] == pytest.approx(tau, rel=1e-5)
    assert row["om_step_lag0"] == pytest.approx(q - dst_star / tau, rel=1e-5, abs=1e-5)
    assert row["dpress"] == pytest.approx(7.26 * (np.sqrt(p) - np.sqrt(prev_p)), rel=1e-4, abs=1e-5)
    assert row["om_step_lag1"] == X_om.loc[t - pd.Timedelta(hours=1), "om_step_lag0"]


def test_n_lags_default_is_unchanged_and_longer_history_does_not_look_ahead():
    df = load_hourly("2024-01-01")
    X, _ = build_features(df)
    assert X.equals(build_features(df, n_lags=N_LAGS)[0])

    X10, _ = build_features(df, n_lags=10)
    assert {"dst_lag9", "bz_gsm_lag9", "newell_lag9"} <= set(X10.columns)
    assert "dst_lag10" not in X10.columns
    # Rows are the same (dst_lag0 decides them); the extra lags only add columns.
    assert X.equals(X10[X.columns])
    lagged = df["bz_gsm"].shift(9).astype("float32").loc[X10.index]
    assert X10["bz_gsm_lag9"].equals(lagged.rename("bz_gsm_lag9"))

    corrupted = df.copy()
    floats = corrupted.select_dtypes("float").columns
    corrupted.loc[corrupted.index > CUT, floats] = -1234.0
    X10_corrupted, _ = build_features(corrupted, n_lags=10)
    past = X10[X10.index <= CUT]
    assert past.equals(X10_corrupted.loc[past.index])

    with pytest.raises(ValueError):
        build_features(df, n_lags=0)


def test_driver_memory_does_not_depend_on_the_future():
    df = load_hourly("2024-01-01")
    X, _ = build_features(df, memory=MEMORY_TAUS)
    memory_cols = [c for c in X if "_ewm" in c]
    assert len(memory_cols) == 2 * len(MEMORY_TAUS)

    corrupted = df.copy()
    floats = corrupted.select_dtypes("float").columns
    corrupted.loc[corrupted.index > CUT, floats] = -1234.0
    X_corrupted, _ = build_features(corrupted, memory=MEMORY_TAUS)

    past = X[X.index <= CUT]
    assert past.equals(X_corrupted.loc[past.index])
    future = X.index[X.index > CUT][:50]
    assert not X.loc[future, memory_cols].equals(X_corrupted.loc[future, memory_cols])
    # Off by default: the baseline columns are unchanged by the option.
    X_default, _ = build_features(df)
    assert X_default.equals(X[X_default.columns])
    assert not any("_ewm" in c for c in X_default)


def test_decaying_memory_is_a_causal_exponential_filter():
    index = pd.date_range("2020-01-01", periods=48, freq="h")
    x = pd.Series(0.0, index=index)
    x.iloc[10] = 1.0
    m = decaying_memory(x, 8)
    a = 1 - np.exp(-1 / 8)
    assert (m.iloc[:10] == 0).all()
    assert np.allclose(m.iloc[10:].to_numpy(), a * (1 - a) ** np.arange(38))
