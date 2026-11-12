"""Real-time inputs from NOAA SWPC, shaped like the hourly OMNI table the models were trained on.

Sources (all public JSON, no key needed):
  propagated-solar-wind.json  1 minute L1 solar wind, time-shifted to Earth by SWPC
  kyoto-dst.json              hourly quick-look Dst from WDC Kyoto
  f107_cm_flux.json           daily 10.7 cm solar radio flux

OMNI convention kept here: the record stamped hour h is the average over h:00 .. h:59.
"""
import json
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://services.swpc.noaa.gov"
SOLAR_WIND_URL = f"{BASE}/products/geospace/propagated-solar-wind.json"
DST_URL = f"{BASE}/products/kyoto-dst.json"
F107_URL = f"{BASE}/json/f107_cm_flux.json"

MIN_MINUTES = 10  # an hourly average needs at least this many 1 minute samples


def _get(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "dst-forecast-dashboard"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _naive_utc(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="ISO8601")).tz_localize(None)


def solar_wind_hourly(rows: list) -> pd.DataFrame:
    """Hourly means of the propagated solar wind, named like the OMNI columns.

    Samples are binned by their arrival time at Earth (propagated_time_tag), which is
    what OMNI's bow shock time shift does. Field components are GSM.
    """
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    frame.index = _naive_utc(frame["propagated_time_tag"]).floor("h")
    numeric = frame[["speed", "density", "temperature", "by", "bz", "bt"]].apply(pd.to_numeric, errors="coerce")
    grouped = numeric.groupby(level=0)
    hourly = grouped.mean().where(grouped.count() >= MIN_MINUTES)

    out = pd.DataFrame(index=hourly.index)
    out["bz_gsm"] = hourly["bz"]
    out["by_gsm"] = hourly["by"]
    out["b_mag_avg"] = hourly["bt"]
    out["flow_speed"] = hourly["speed"]
    out["proton_density"] = hourly["density"]
    out["proton_temp"] = hourly["temperature"]
    # OMNI: P = 1.67e-6 * Np * V^2 * (1 + 4 Na/Np). No real-time alpha ratio, so the
    # usual 5% helium assumption is used, which gives the factor 2e-6.
    out["flow_pressure"] = 2e-6 * out["proton_density"] * out["flow_speed"] ** 2
    out["electric_field"] = -out["flow_speed"] * out["bz_gsm"] * 1e-3
    return out


def dst_hourly(rows: list) -> pd.Series:
    """Quick-look Dst indexed by the start of each hour (checked against the Kyoto table)."""
    frame = pd.DataFrame(rows)
    series = pd.Series(pd.to_numeric(frame["dst"], errors="coerce").to_numpy(), index=_naive_utc(frame["time_tag"]))
    return series[~series.index.duplicated(keep="last")].sort_index()


def f107_daily(rows: list) -> pd.Series:
    """Local-noon F10.7 flux per day, the value OMNI carries for every hour of that day."""
    frame = pd.DataFrame(rows)
    frame = frame[frame["reporting_schedule"] == "Noon"]
    series = pd.Series(pd.to_numeric(frame["flux"], errors="coerce").to_numpy(), index=_naive_utc(frame["time_tag"]).floor("D"))
    return series[~series.index.duplicated(keep="last")].sort_index()


def build_live_table(solar_wind: list, dst: list, f107: list, now: pd.Timestamp) -> pd.DataFrame:
    """Gap-free hourly table ending at the last hour that is already complete at `now`.

    Kyoto publishes a running value for the hour in progress; it is dropped so the
    models only ever see finished hours, exactly as in training.
    """
    wind, dst_series = solar_wind_hourly(solar_wind), dst_hourly(dst)
    last_complete = now.floor("h") - pd.Timedelta("1h")
    start = max(wind.index.min(), dst_series.index.min())
    index = pd.date_range(start, last_complete, freq="h")

    table = wind.reindex(index)
    table["dst"] = dst_series.reindex(index)
    try:
        flux = f107_daily(f107)
        table["f107"] = flux.reindex(index.floor("D"), method="ffill").to_numpy()
    except (KeyError, ValueError):
        table["f107"] = np.nan  # a minor input; the tree model handles it missing
    return table


def fetch_live_table(now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now("UTC").tz_localize(None)
    return build_live_table(_get(SOLAR_WIND_URL), _get(DST_URL), _get(F107_URL), now)
