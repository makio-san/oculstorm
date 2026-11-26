"""The holdout score must never read an hour at or before the cutoff."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from oculstorm.features import N_LAGS, SOLAR_WIND
from oculstorm.sequences import WINDOW

SPEC = importlib.util.spec_from_file_location("score_holdout", Path(__file__).resolve().parent.parent / "scripts" / "score_holdout.py")
score_holdout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_holdout)

CUTOFF = pd.Timestamp("2026-10-30 13:00")


def _table() -> pd.DataFrame:
    hours = pd.date_range("2026-10-20", "2026-11-10", freq="h")
    dst = -20 + 40 * np.sin(np.arange(len(hours)) / 10)
    table = pd.DataFrame({"dst": dst}, index=hours)
    for col in SOLAR_WIND:
        table[col] = 1.0
    return table


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


def test_hours_without_solar_wind_inputs_are_not_scored():
    table = _table()
    gap_start, gap_end = pd.Timestamp("2026-11-03 00:00"), pd.Timestamp("2026-11-04 23:00")
    table.loc[gap_start:gap_end, "bz_gsm"] = np.nan
    table.loc[table.index >= pd.Timestamp("2026-11-08 00:00"), "flow_speed"] = np.nan
    holdout = score_holdout.after_cutoff(table, CUTOFF)
    frame = score_holdout.forecasts(holdout, {"model": _rolling_model([])})
    issued = frame.index - pd.Timedelta(hours=1)
    lags = pd.Timedelta(hours=N_LAGS - 1)
    assert not ((issued >= gap_start) & (issued <= gap_end + lags)).any()
    assert issued.max() < pd.Timestamp("2026-11-08 00:00")
    assert ((issued > gap_end + lags) & (issued < pd.Timestamp("2026-11-08 00:00"))).sum() > 0

    inputs = score_holdout.coverage(holdout)
    assert inputs["issue_hours"] == len(score_holdout.issue_hours(holdout))
    assert inputs["issue_hours_with_inputs"] == len(frame)  # every issue hour with inputs has its target observed here
    assert inputs["issue_hours_with_inputs"] + inputs["issue_hours_without_inputs"] == inputs["issue_hours"]
    assert inputs["observed_share"]["by_gsm"] == 1.0
    assert inputs["observed_share"]["bz_gsm"] < 1.0


def test_no_solar_wind_inputs_gives_no_scored_hours():
    table = _table()
    table[["bz_gsm", "by_gsm", "b_mag_avg", "electric_field"]] = np.nan
    holdout = score_holdout.after_cutoff(table, CUTOFF)
    frame = score_holdout.forecasts(holdout, {"model": _rolling_model([])})
    assert frame.empty
    result = score_holdout.scores(frame, ["model"])
    assert result["all"]["hours"] == result["storm"]["hours"] == 0
    assert score_holdout.coverage(holdout)["issue_hours_with_inputs"] == 0


def test_a_model_with_seed_members_is_scored_on_their_mean():
    holdout = score_holdout.after_cutoff(_table(), CUTOFF)
    members = lambda t: pd.DataFrame({"gru_seed0": t["dst"] - 2.0, "gru_seed1": t["dst"] + 2.0})  # noqa: E731
    a = score_holdout.forecasts(holdout, {"model": members})
    b = score_holdout.forecasts(holdout, {"model": lambda t: t["dst"]})
    pd.testing.assert_frame_equal(a, b)


def test_an_hour_with_a_missing_seed_member_is_not_scored():
    holdout = score_holdout.after_cutoff(_table(), CUTOFF)
    gap = holdout.index[WINDOW + 5]

    def members(t: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({"gru_seed0": t["dst"], "gru_seed1": t["dst"]})
        out.loc[gap, "gru_seed1"] = np.nan
        return out

    frame = score_holdout.forecasts(holdout, {"model": members})
    assert gap + pd.Timedelta(hours=1) not in frame.index
