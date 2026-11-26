"""The real-time ACE score must stay on the validation years and never look ahead."""
import numpy as np
import pandas as pd
import pytest

from oculstorm import realtime_inputs as rt
from oculstorm.data import TEST_START, VAL_START
from oculstorm.features import SOLAR_WIND
from oculstorm.sequences import WINDOW

MAG = """:Product: 201701_ace_mag_1h.txt
# Status(S): 0 = nominal data, 1 to 8 = bad data record, 9 = no data
# YR MO DA  HHMM    Day      Day    S     Bx      By      Bz      Bt     Lat.   Long.
2017 01 01  0000   57754       0    9  -999.9  -999.9  -999.9  -999.9  -999.9  -999.9
2017 01 01  0100   57754    3600    0     2.0    -2.1    -0.6     2.9   -10.8   314.0
2017 01 01  0200   57754    7200    3    -0.7     2.1    -2.8     3.6   -51.2   109.2
2017 01 01  0300   57754   10800    0    -2.9  -999.9     0.1     2.9     1.7   185.0
"""
SWEPAM = """:Product: 201701_ace_swepam_1h.txt
# YR MO DA  HHMM    Day     Day     S    Density     Speed     Temperature
2017 01 01  0000   57754       0    9    -9999.9    -9999.9    -1.00e+05
2017 01 01  0100   57754    3600    0        2.7      542.8     1.73e+05
2017 01 01  0200   57754    7200    0        3.3    -9999.9    -1.00e+05
"""


def _omni(start="2016-12-25", stop="2021-01-10") -> pd.DataFrame:
    hours = pd.date_range(start, stop, freq="h")
    table = pd.DataFrame({"dst": -20 + 30 * np.sin(np.arange(len(hours)) / 9), "f107": 70.0}, index=hours)
    for i, col in enumerate(SOLAR_WIND):
        table[col] = i + 1 + np.cos(np.arange(len(hours)) / 7)
    return table


def test_only_validation_months_are_named():
    names = [rt.file_name(m, kind) for m in rt.months() for kind in rt.KINDS]
    assert len(names) == 96
    assert names[0] == "201701_ace_mag_1h.txt" and names[-1] == "202012_ace_swepam_1h.txt"
    for month in ("201612", "202101", "202405"):
        with pytest.raises(ValueError):
            rt.file_name(month, "mag")


def test_bad_status_and_sentinels_become_missing():
    mag = rt.parse_month(MAG, "mag")
    assert list(mag.index) == list(pd.date_range("2017-01-01", periods=4, freq="h"))
    assert mag.iloc[0].isna().all() and mag.iloc[2].isna().all()
    assert mag.loc["2017-01-01 01:00", "bz_gsm"] == -0.6 and mag.loc["2017-01-01 01:00", "b_mag_avg"] == 2.9
    assert np.isnan(mag.loc["2017-01-01 03:00", "by_gsm"]) and mag.loc["2017-01-01 03:00", "bz_gsm"] == 0.1
    plasma = rt.parse_month(SWEPAM, "swepam")
    assert plasma.iloc[0].isna().all()
    assert plasma.loc["2017-01-01 01:00"].tolist() == [2.7, 542.8, 1.73e5]
    assert plasma.loc["2017-01-01 02:00", "proton_density"] == 3.3
    assert plasma.loc["2017-01-01 02:00", ["flow_speed", "proton_temp"]].isna().all()


def test_loaded_table_covers_only_validation_hours_and_derives_like_live(tmp_path):
    for month in rt.months():
        stray = f"{month[:4]} {month[4:]} 01  0000   57754       0    0"
        (tmp_path / rt.file_name(month, "mag")).write_text(MAG if month == "201701" else stray + "     1.0     1.0    -5.0     5.2     0.0     0.0\n")
        # A row of another month inside a file must be ignored.
        other = "2021 01 01  0000   59215       0    0        9.0      900.0     9.00e+05\n"
        (tmp_path / rt.file_name(month, "swepam")).write_text(SWEPAM if month == "201701" else stray + "        4.0      400.0     1.00e+05\n" + other)
    ace = rt.load_ace(tmp_path)
    assert ace.index.min() == pd.Timestamp(VAL_START) and ace.index.max() == pd.Timestamp(TEST_START) - pd.Timedelta("1h")
    assert list(ace.columns) == SOLAR_WIND
    row = ace.loc["2017-02-01 00:00"]
    assert row["flow_pressure"] == pytest.approx(2e-6 * 4.0 * 400.0**2)
    assert row["electric_field"] == pytest.approx(400.0 * 5.0 * 1e-3)
    assert ace["flow_speed"].max() == 542.8  # the stray 2021 row never arrives


def test_delay_moves_values_later_and_never_earlier():
    ace = pd.DataFrame({"bz_gsm": np.arange(5.0)}, index=pd.date_range("2017-01-01", periods=5, freq="h"))
    late = rt.delayed(ace, 1)
    assert np.isnan(late["bz_gsm"].iloc[0]) and late["bz_gsm"].iloc[1:].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert rt.delayed(ace, 0).equals(ace)
    with pytest.raises(ValueError):
        rt.delayed(ace, -1)


def test_no_hour_outside_validation_reaches_a_model():
    omni = rt.validation_table(_omni())
    wind = pd.DataFrame(0.5, index=omni.index, columns=SOLAR_WIND)
    tables = {"omni": omni, "ace_delay1h": rt.with_solar_wind(omni, rt.delayed(wind, 1))}
    seen = []

    def predict(table: pd.DataFrame) -> pd.Series:
        seen.append((table.index.min(), table.index.max()))
        return table["dst"] + table["bz_gsm"].fillna(0.0)

    frame = rt.forecast_frame(tables, predict)
    last = pd.Timestamp(TEST_START) - pd.Timedelta("1h")
    assert seen == [(pd.Timestamp(VAL_START), last)] * 2
    assert frame.index.min() == pd.Timestamp(VAL_START) + pd.Timedelta(hours=WINDOW - 1 + max(rt.DELAYS))
    assert frame.index.max() == last - pd.Timedelta("1h")  # the target of the last issue hour is still 2020
    # Dst and F10.7 are untouched, only the solar wind columns are replaced.
    assert tables["ace_delay1h"]["dst"].equals(omni["dst"]) and tables["ace_delay1h"]["f107"].equals(omni["f107"])
    assert (tables["ace_delay1h"][SOLAR_WIND].iloc[1:] == 0.5).all().all()
    assert tables["ace_delay1h"][SOLAR_WIND].iloc[0].isna().all()


def test_input_statistics_report_bias_spread_and_missing_share():
    omni = rt.validation_table(_omni()).iloc[:1000]
    wind = omni[SOLAR_WIND] + 2.0
    wind.iloc[:100] = np.nan
    stats = rt.input_statistics(omni, wind)
    for col in SOLAR_WIND:
        assert stats[col]["bias"] == pytest.approx(2.0)
        assert stats[col]["std_difference"] == pytest.approx(0.0)
        assert stats[col]["missing_share_realtime"] == pytest.approx(0.1)
        assert stats[col]["hours_compared"] == 900


def test_score_is_paired_and_respects_the_row_filter():
    rng = np.random.default_rng(0)
    hours = pd.date_range("2017-01-02", periods=3000, freq="h")
    observed = -20 + 60 * np.sin(np.arange(3000) / 40)
    frame = pd.DataFrame({"observed": observed, "persistence": observed + rng.normal(0, 3, 3000),
                          "omni": observed + rng.normal(0, 1, 3000)}, index=hours)
    frame["ace"] = frame["omni"] + rng.normal(0, 2, 3000)
    result = rt.score(frame, "omni", ["ace"])
    diff = result["all"]["differences"]["ace minus omni"]
    assert result["all"]["n_hours"] == 3000 and diff["mse_change_percent"] > 100 and diff["significant"]
    assert diff["mse_change_percent_ci95"][0] > 0
    assert result["all"]["inputs"]["omni"]["skill"] > result["all"]["inputs"]["ace"]["skill"]
    rows = np.arange(3000) < 500
    part = rt.score(frame, "omni", ["ace"], rows)
    assert part["all"]["n_hours"] == 500
    assert part["storm (Dst <= -50)"]["n_hours"] == int(((observed <= -50) & rows).sum())
