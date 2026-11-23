"""Guards against look-ahead leakage in the feature pipeline."""
import numpy as np
import pandas as pd

from oculstorm.data import load_hourly
import pytest

from oculstorm.features import HORIZONS, MEMORY_TAUS, build_features, decaying_memory

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
