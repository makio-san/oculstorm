"""The holdout score must never read an hour at or before the cutoff."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from oculstorm.sequences import WINDOW

SPEC = importlib.util.spec_from_file_location("score_holdout", Path(__file__).resolve().parent.parent / "scripts" / "score_holdout.py")
score_holdout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_holdout)

CUTOFF = pd.Timestamp("2026-10-30 13:00")


def _table() -> pd.DataFrame:
    hours = pd.date_range("2026-10-20", "2026-11-10", freq="h")
    dst = -20 + 40 * np.sin(np.arange(len(hours)) / 10)
    return pd.DataFrame({"dst": dst}, index=hours)


def _rolling_model(seen: list):
    def predict(table: pd.DataFrame) -> pd.Series:
        seen.append(table.index.min())
        return table["dst"].rolling(24, min_periods=1).mean()
    return predict


def test_no_hour_before_the_cutoff_reaches_a_model():
    seen = []
    holdout = score_holdout.after_cutoff(_table(), CUTOFF)
    frame = score_holdout.forecasts(holdout, {"model": _rolling_model(seen)})
    assert seen == [CUTOFF + pd.Timedelta(hours=1)]
    assert frame.index.min() == CUTOFF + pd.Timedelta(hours=WINDOW + 1)


def test_hours_before_the_cutoff_do_not_change_the_scores():
    clean = _table()
    poisoned = clean.copy()
    poisoned.loc[poisoned.index <= CUTOFF, "dst"] = -999.0
    a = score_holdout.forecasts(score_holdout.after_cutoff(clean, CUTOFF), {"model": _rolling_model([])})
    b = score_holdout.forecasts(score_holdout.after_cutoff(poisoned, CUTOFF), {"model": _rolling_model([])})
    pd.testing.assert_frame_equal(a, b)


def test_scores_count_storm_hours_and_use_persistence_as_reference():
    holdout = score_holdout.after_cutoff(_table(), CUTOFF)
    frame = score_holdout.forecasts(holdout, {"perfect": lambda t: t["dst"].shift(-1), "persist": lambda t: t["dst"]})
    result = score_holdout.scores(frame, ["perfect", "persist"])
    assert result["all"]["hours"] == len(frame)
    assert result["storm"]["hours"] == int((frame["observed"] <= -50).sum()) > 0
    assert result["all"]["perfect"]["skill"] == 1.0
    assert result["all"]["persist"]["skill"] == 0.0


def test_no_hours_after_the_cutoff_gives_an_empty_frame():
    holdout = score_holdout.after_cutoff(_table(), pd.Timestamp("2026-12-01"))
    assert score_holdout.forecasts(holdout, {"model": _rolling_model([])}).empty
