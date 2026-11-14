"""The real-time log must keep the forecast that was issued first and only fill in observations."""
import numpy as np
import pandas as pd

from oculstorm import archive

T1, T2 = pd.Timestamp("2026-01-01 13:00"), pd.Timestamp("2026-01-01 14:00")


def _dst(values: dict) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_issued_forecast_is_never_rewritten(tmp_path):
    log = archive.load(tmp_path / "missing.csv")
    log = archive.update(log, pd.Timestamp("2026-01-01 13:07"), T1, {"gru": -60.0, "xgb": -58.0, "persistence": -40.0}, _dst({T1: np.nan}))
    # A later run sees the same target with different numbers: the first issue must stand.
    log = archive.update(log, pd.Timestamp("2026-01-01 13:40"), T1, {"gru": -99.0, "xgb": -99.0, "persistence": -99.0}, _dst({T1: np.nan}))
    assert len(log) == 1
    assert log.loc[T1, "gru"] == -60.0
    assert log.loc[T1, "issued"] == "2026-01-01T13:07Z"
    assert pd.isna(log.loc[T1, "observed"])


def test_observations_arrive_later_and_survive_a_round_trip(tmp_path):
    path = tmp_path / "forecasts.csv"
    log = archive.update(archive.load(path), pd.Timestamp("2026-01-01 13:07"), T1, {"gru": -60.0, "xgb": -58.0, "persistence": -40.0}, _dst({}))
    archive.save(log, path)
    log = archive.update(archive.load(path), pd.Timestamp("2026-01-01 14:07"), T2, {"gru": -10.0, "xgb": -12.0, "persistence": -64.0}, _dst({T1: -64.0, T2: np.nan}))
    archive.save(log, path)
    log = archive.load(path)
    assert list(log.index) == [T1, T2]
    assert log.loc[T1, "observed"] == -64.0 and log.loc[T1, "gru"] == -60.0
    assert pd.isna(log.loc[T2, "observed"])


def test_summary_scores_only_observed_hours_and_splits_storms():
    log = archive.load("does-not-exist.csv")
    assert archive.summary(log) is None
    log = archive.update(log, T1, T1, {"gru": -60.0, "xgb": -58.0, "persistence": -40.0}, _dst({}))
    log = archive.update(log, T2, T2, {"gru": -10.0, "xgb": -12.0, "persistence": -64.0}, _dst({T1: -64.0}))
    s = archive.summary(log)
    assert s["issued"] == 2 and s["all"]["hours"] == 1 and s["storm"]["hours"] == 1
    assert s["all"]["rmse"] == {"gru": 4.0, "xgb": 6.0, "persistence": 24.0}
    log = archive.update(log, T2, T2, {}, _dst({T2: -9.0}))
    s = archive.summary(log)
    assert s["all"]["hours"] == 2 and s["storm"]["hours"] == 1 and s["deepest"] == -64.0


def test_a_run_without_a_forecast_logs_nothing():
    log = archive.update(archive.load("does-not-exist.csv"), T1, T1, {"gru": np.nan, "xgb": -5.0, "persistence": -4.0}, _dst({}))
    assert log.empty
