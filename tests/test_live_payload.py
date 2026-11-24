"""The live.js payload: every field the dashboard reads, plus the model spread band."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "live_forecast.py"
spec = importlib.util.spec_from_file_location("live_forecast", SCRIPT)
live_forecast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_forecast)

FIELDS = ["generated", "start", "hours", "target", "target_is_future", "observed", "gru", "xgb", "persistence",
          "hindcast", "archive", "inputs_hour", "inputs", "solar_wind_coverage"]


def _inputs():
    hours = pd.date_range("2026-01-01 00:00", periods=6, freq="h")
    table = pd.DataFrame({"dst": [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0], "bz_gsm": -5.0, "flow_speed": 500.0,
                          "proton_density": 5.0, "flow_pressure": 2.5}, index=hours)
    # Issued at hours 1 .. 5; hour 3 has no XGBoost forecast, so the band has a gap there.
    issued = hours[1:]
    members = pd.DataFrame({"gru_seed0": [-21.0, -31.0, -41.0, -51.0, -61.0],
                            "gru_seed1": [-23.0, -33.0, -43.0, -53.0, -63.0],
                            "gru_seed2": [-22.0, -32.0, -42.0, -52.0, -62.0],
                            "xgb": [-25.0, -28.0, np.nan, -55.0, -58.0]}, index=issued)
    return table, members


def _payload():
    table, members = _inputs()
    frame = live_forecast.align(table, members)
    return live_forecast.build_payload(pd.Timestamp("2026-01-01 05:20"), table, frame, members.columns)


def test_existing_fields_are_kept():
    payload = _payload()
    for field in FIELDS:
        assert field in payload
    assert payload["start"] == "2026-01-01T02:00"
    assert payload["target"] == "2026-01-01T06:00"
    assert payload["hours"] == 5
    assert payload["target_is_future"] is True
    assert payload["observed"] == [-30.0, -40.0, -50.0, -60.0, None]
    assert payload["gru"] == [-22.0, -32.0, None, -52.0, -62.0]
    assert payload["xgb"] == [-25.0, -28.0, None, -55.0, -58.0]
    assert payload["persistence"] == [-20.0, -30.0, None, -50.0, -60.0]


def test_spread_is_the_range_of_every_member():
    spread = _payload()["spread"]
    assert spread["members"] == ["gru_seed0", "gru_seed1", "gru_seed2", "xgb"]
    assert spread["low"] == [-25.0, -33.0, None, -55.0, -63.0]
    assert spread["high"] == [-21.0, -28.0, None, -51.0, -58.0]
    payload = _payload()
    assert len(spread["low"]) == len(spread["high"]) == payload["hours"]
    for low, high, gru, xgb in zip(spread["low"], spread["high"], payload["gru"], payload["xgb"]):
        if low is not None:
            assert low <= min(gru, xgb) and max(gru, xgb) <= high


def test_spread_is_missing_when_any_member_is():
    members = pd.DataFrame({"gru_seed0": [-1.0, np.nan], "xgb": [-3.0, -4.0]})
    band = live_forecast.model_spread(members)
    assert band["low"].tolist()[0] == -3.0 and band["high"].tolist()[0] == -1.0
    assert band.iloc[1].isna().all()
