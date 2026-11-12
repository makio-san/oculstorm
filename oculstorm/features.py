"""Feature engineering for one hour ahead Dst forecasting.

Every feature for row t uses only records at hour t or earlier. The target is
Dst at hour t+1. Kp, ap, AE, AL, AU and PC are deliberately excluded: Kp and ap
are 3 hour values that cover hours after t (leakage), and the others are
geomagnetic responses rather than solar wind drivers.
"""
import numpy as np
import pandas as pd

HORIZON = 1  # hours ahead
N_LAGS = 6  # hours of history given to the model (lag 0 .. N_LAGS-1)

SOLAR_WIND = [
    "bz_gsm", "by_gsm", "b_mag_avg", "flow_speed", "proton_density",
    "flow_pressure", "electric_field", "proton_temp",
]


def _coupling(df: pd.DataFrame) -> pd.DataFrame:
    """Physics based solar wind / magnetosphere coupling terms."""
    out = pd.DataFrame(index=df.index)
    v, by, bz = df["flow_speed"], df["by_gsm"], df["bz_gsm"]
    # Rectified dawn-dusk electric field: only southward IMF drives the ring current.
    out["vbs"] = v * (-bz).clip(lower=0) * 1e-3
    # Newell et al. (2007) coupling function.
    bt = np.sqrt(by**2 + bz**2)
    clock = np.arctan2(by, bz)
    out["newell"] = v ** (4 / 3) * bt ** (2 / 3) * np.abs(np.sin(clock / 2)) ** (8 / 3) * 1e-3
    # Magnetopause current contribution to Dst scales with sqrt(dynamic pressure).
    out["sqrt_pressure"] = np.sqrt(df["flow_pressure"])
    return out


def build_features(
    df: pd.DataFrame, use_solar_wind: bool = True, require_target: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y is Dst at t+HORIZON and X only looks backwards from t.

    With use_solar_wind=False only the Dst history and calendar terms are kept,
    which shows how much of the skill actually comes from the solar wind.
    With require_target=False the newest hours, whose target is not known yet, are kept.
    """
    cols: dict[str, pd.Series] = {}
    dst = df["dst"]

    for lag in range(N_LAGS):
        cols[f"dst_lag{lag}"] = dst.shift(lag)
    cols["dst_diff1"] = dst - dst.shift(1)
    cols["dst_diff3"] = dst - dst.shift(3)
    cols["dst_min24"] = dst.rolling(24, min_periods=1).min()
    cols["dst_mean24"] = dst.rolling(24, min_periods=1).mean()

    if use_solar_wind:
        drivers = pd.concat([df[SOLAR_WIND], _coupling(df)], axis=1)
        for name in drivers:
            for lag in range(N_LAGS):
                cols[f"{name}_lag{lag}"] = drivers[name].shift(lag)
        for name in ["vbs", "newell", "bz_gsm", "flow_speed"]:
            for window in (3, 12):
                cols[f"{name}_mean{window}"] = drivers[name].rolling(window, min_periods=1).mean()
        cols["f107"] = df["f107"]

    # Dst has a diurnal and a semiannual (Russell-McPherron) modulation.
    hour = df.index.hour + 0.0
    doy = df.index.dayofyear + 0.0
    cols["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=df.index)
    cols["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=df.index)
    cols["doy_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=df.index)
    cols["doy_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=df.index)

    X = pd.DataFrame(cols).astype("float32")
    y = dst.shift(-HORIZON).rename("dst_target")

    keep = X["dst_lag0"].notna()
    if require_target:
        keep &= y.notna()
    return X[keep], y[keep]
