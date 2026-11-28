"""The issue 50 scoring sees only validation hours and lines every model up on the same hours."""
import numpy as np
import pandas as pd

from oculstorm import realtime_inputs as rt
from oculstorm import realtime_training as rtt
from oculstorm.data import TEST_START, VAL_START


def test_candidate_frames_share_issue_hours_and_target_is_one_hour_ahead():
    hours = pd.date_range(VAL_START, TEST_START, freq="h", inclusive="left")
    table = pd.DataFrame({"dst": np.arange(len(hours), dtype=float)}, index=hours)
    predictors = {"zero": lambda t: t["dst"] * 0.0, "same": lambda t: t["dst"].iloc[3:]}
    frame = rtt.candidate_frames({"omni": table}, predictors)["omni"]
    assert frame.index.min() >= hours[0] and frame.index.max() < pd.Timestamp(TEST_START)
    assert rt.issue_mask(frame.index).all()
    assert (frame["observed"] == frame["persistence"] + 1).all()
    assert list(frame.columns) == ["observed", "persistence", "zero", "same"]
