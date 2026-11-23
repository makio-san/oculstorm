"""Guards against look-ahead leakage in the feature pipeline."""
import numpy as np
import pandas as pd

from oculstorm.data import load_hourly
import pytest

from oculstorm.features import HORIZONS, build_features

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
