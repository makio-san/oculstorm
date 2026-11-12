"""Guards against look-ahead leakage in the feature pipeline."""
import pandas as pd

from oculstorm.data import load_hourly
from oculstorm.features import HORIZON, build_features

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


def test_target_is_dst_one_horizon_ahead():
    df = load_hourly("2024-01-01")
    _, y = build_features(df)
    expected = df["dst"].shift(-HORIZON).loc[y.index]
    assert (y == expected).all()
    assert y.index.max() == df.index.max() - pd.Timedelta(hours=HORIZON)
