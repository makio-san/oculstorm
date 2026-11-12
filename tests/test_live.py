"""The live table must follow the OMNI hourly convention and never include an unfinished hour."""
import pandas as pd

from oculstorm.live import build_live_table

HEADER = ["time_tag", "speed", "density", "temperature", "bx", "by", "bz", "bt", "vx", "vy", "vz", "propagated_time_tag"]


def _minute_rows(start: str, minutes: int, delay_min: int, speed: float, bz: float) -> list:
    rows = []
    for t in pd.date_range(start, periods=minutes, freq="min"):
        arrival = t + pd.Timedelta(minutes=delay_min)
        rows.append([t.strftime("%Y-%m-%dT%H:%M:%SZ"), speed, 5.0, 1e5, 0.0, 2.0, bz, 6.0, -speed, 0, 0, arrival.strftime("%Y-%m-%dT%H:%M:%SZ")])
    return rows


def _table(now: str) -> pd.DataFrame:
    # Measured 09:30 .. 12:29 at L1, arriving 60 minutes later: 10:30 .. 13:29 at Earth.
    wind = [HEADER] + _minute_rows("2026-01-01T09:30:00", 90, 60, 400.0, -5.0) + _minute_rows("2026-01-01T11:00:00", 90, 60, 600.0, 5.0)
    dst = [{"time_tag": f"2026-01-01T{h:02d}:00:00", "dst": -10 * h} for h in range(8, 14)]
    f107 = [{"time_tag": "2025-12-31T20:00:00", "flux": 150.0, "reporting_schedule": "Noon"}]
    return build_live_table(wind, dst, f107, pd.Timestamp(now))


def test_hour_in_progress_is_dropped():
    table = _table("2026-01-01 13:20")
    assert table.index.max() == pd.Timestamp("2026-01-01 12:00")
    assert (table.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert table.loc["2026-01-01 12:00", "dst"] == -120


def test_solar_wind_is_binned_by_arrival_time():
    table = _table("2026-01-01 13:20")
    # Arrivals 11:00 .. 11:59 were all measured before 11:00 at L1: the slow, southward stream.
    assert table.loc["2026-01-01 11:00", "flow_speed"] == 400.0
    assert table.loc["2026-01-01 11:00", "bz_gsm"] == -5.0
    assert table.loc["2026-01-01 11:00", "electric_field"] == 2.0  # -V * Bz * 1e-3
    assert abs(table.loc["2026-01-01 11:00", "flow_pressure"] - 2e-6 * 5.0 * 400.0**2) < 1e-9
    assert table.loc["2026-01-01 12:00", "flow_speed"] == 600.0
    # Hours before any solar wind arrived are kept as gaps, not invented.
    assert pd.isna(table.loc["2026-01-01 08:00", "flow_speed"]) if pd.Timestamp("2026-01-01 08:00") in table.index else True
    assert table["f107"].iloc[-1] == 150.0
