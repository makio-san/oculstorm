"""Guards against look-ahead leakage in the feature pipeline."""
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
