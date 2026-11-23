"""The calibrated error band: validation quantiles per Dst regime and the live payload fields."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


live_forecast = _script("live_forecast")
build_dashboard = _script("build_dashboard")


def test_committed_bands_cover_every_regime_from_validation_years():
    bands = json.loads((ROOT / "docs" / "error_bands.json").read_text())
    assert [r["name"] for r in bands["regimes"]] == ["quiet", "storm", "intense"]
    assert bands["quantiles"] == [0.1, 0.9]
    start, end = bands["period"].split(" .. ")
    assert start >= "2017-01-01" and end < "2021-01-01"  # never the test years
    for r in bands["regimes"]:
        assert r["hours"] > 0 and r["q10"] < 0 < r["q90"]


def test_regime_boundaries():
    regimes = build_dashboard.REGIMES
    pick = lambda v: live_forecast.regime_of(v, regimes)  # noqa: E731
    assert pick(-49)["name"] == "quiet"
    assert pick(-50)["name"] == "storm"
    assert pick(-99)["name"] == "storm"
    assert pick(-100)["name"] == "intense"
    assert pick(np.nan) is None
    for v in [-49, -50, -100, -300]:
        assert build_dashboard.regime_of(v) == pick(v)["name"]


def test_error_quantiles_per_regime():
    index = pd.date_range("2018-01-01", periods=300, freq="h")
    dst_lag0 = pd.Series([0.0] * 100 + [-70.0] * 100 + [-150.0] * 100, index=index)
    forecast = pd.Series(0.0, index=index)
    error = np.concatenate([np.linspace(-1, 1, 100), np.linspace(-5, 5, 100), np.linspace(-20, 20, 100)])
    out = build_dashboard.error_quantiles(pd.Series(error, index=index), forecast, dst_lag0)
    assert [r["hours"] for r in out] == [100, 100, 100]
    assert out[0]["q10"] == -0.8 and out[0]["q90"] == 0.8
    assert out[2]["q10"] == -16.0 and out[2]["q90"] == 16.0


def test_band_follows_the_regime_of_each_hour():
    bands = {"regimes": [
        {"name": "quiet", "above": -50, "q10": -2.0, "q90": 3.0},
        {"name": "storm", "above": -100, "q10": -6.0, "q90": 7.0},
        {"name": "intense", "above": None, "q10": -15.0, "q90": 15.0},
    ]}
    index = pd.date_range("2026-01-01", periods=4, freq="h")
    gru = pd.Series([-10.0, -60.0, -120.0, np.nan], index=index)
    dst_lag0 = pd.Series([-5.0, -55.0, -110.0, np.nan], index=index)
    band = live_forecast.error_band(gru, dst_lag0, bands)
    assert list(band["regime"].iloc[:3]) == ["quiet", "storm", "intense"] and pd.isna(band["regime"].iloc[3])
    assert live_forecast.as_list(band["low"]) == [-12.0, -66.0, -135.0, None]
    assert live_forecast.as_list(band["high"]) == [-7.0, -53.0, -105.0, None]


def test_missing_bands_file_is_tolerated(tmp_path):
    assert live_forecast.load_error_bands(tmp_path / "missing.json") is None
