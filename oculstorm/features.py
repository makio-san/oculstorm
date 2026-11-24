"""Feature engineering for Dst forecasting a few hours ahead.

Every feature for row t uses only records at hour t or earlier. The target is
Dst at hour t+horizon (1 hour by default). Kp, ap, AE, AL, AU and PC are deliberately excluded: Kp and ap
are 3 hour values that cover hours after t (leakage), and the others are
geomagnetic responses rather than solar wind drivers.
"""
import numpy as np
import pandas as pd

HORIZON = 1  # default hours ahead
HORIZONS = (1, 3, 6, 12)  # lead times the models are trained for
N_LAGS = 6  # hours of history given to the model (lag 0 .. N_LAGS-1)

def suffix(horizon: int) -> str:
    """File name suffix for reports of a horizon; the 1 hour files keep their original names."""
    return "" if horizon == 1 else f"_{horizon}h"


def weight_tag(storm_weight: float) -> str:
    """File name tag of a storm-weighted run; the unweighted baseline has none."""
    return "" if storm_weight == 1 else f"_sw{storm_weight:g}"


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


# O'Brien and McPherron (2000) ring current model, Burton form dDst*/dt = Q - Dst*/tau.
OM_PRESSURE = 7.26  # nT / sqrt(nPa)
OM_QUIET = 11.0  # nT
OM_VBS_THRESHOLD = 0.49  # mV/m
OM_INJECTION = -4.4  # nT/h per mV/m


def _ring_current(df: pd.DataFrame) -> pd.DataFrame:
    """O'Brien and McPherron (2000) terms at hour t, from Dst, VBs and pressure at t (and P at t-1).

    dst_star is the pressure corrected Dst, q_om the injection, tau_om the decay time in hours,
    om_step the predicted Dst* change over the next hour and dpress the pressure correction
    change over the last hour. Missing pressure or VBs gives NaN, like the other drivers.
    """
    vbs = _coupling(df)["vbs"]
    sqrt_p = np.sqrt(df["flow_pressure"])
    out = pd.DataFrame(index=df.index)
    out["dst_star"] = df["dst"] - OM_PRESSURE * sqrt_p + OM_QUIET
    out["q_om"] = OM_INJECTION * (vbs - OM_VBS_THRESHOLD).clip(lower=0)
    out["tau_om"] = 2.40 * np.exp(9.74 / (4.69 + vbs))
    out["om_step"] = out["q_om"] - out["dst_star"] / out["tau_om"]
    out["dpress"] = OM_PRESSURE * (sqrt_p - sqrt_p.shift(1))
    return out


def build_features(
    df: pd.DataFrame, use_solar_wind: bool = True, require_target: bool = True, horizon: int = HORIZON,
    ring_current: bool = False, n_lags: int = N_LAGS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y is Dst at t+horizon and X only looks backwards from t.

    X does not depend on the horizon; only the target moves.

    With use_solar_wind=False only the Dst history and calendar terms are kept,
    which shows how much of the skill actually comes from the solar wind.
    With require_target=False the newest hours, whose target is not known yet, are kept.
    With ring_current=True the O'Brien and McPherron (2000) terms are added (needs the solar wind).
    n_lags sets how many hours of history (lag 0 .. n_lags-1) the lag columns cover; the
    default N_LAGS is what the production models and the live forecast use.
    """
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    cols: dict[str, pd.Series] = {}
    dst = df["dst"]

    for lag in range(n_lags):
        cols[f"dst_lag{lag}"] = dst.shift(lag)
    cols["dst_diff1"] = dst - dst.shift(1)
    cols["dst_diff3"] = dst - dst.shift(3)
    cols["dst_min24"] = dst.rolling(24, min_periods=1).min()
    cols["dst_mean24"] = dst.rolling(24, min_periods=1).mean()

    if use_solar_wind:
        drivers = pd.concat([df[SOLAR_WIND], _coupling(df)], axis=1)
        for name in drivers:
            for lag in range(n_lags):
                cols[f"{name}_lag{lag}"] = drivers[name].shift(lag)
        for name in ["vbs", "newell", "bz_gsm", "flow_speed"]:
            for window in (3, 12):
                cols[f"{name}_mean{window}"] = drivers[name].rolling(window, min_periods=1).mean()
        cols["f107"] = df["f107"]
        if ring_current:
            om = _ring_current(df)
            for name in ["dst_star", "om_step"]:
                for lag in range(n_lags):
                    cols[f"{name}_lag{lag}"] = om[name].shift(lag)
            for name in ["q_om", "tau_om", "dpress"]:
                cols[name] = om[name]

    # Dst has a diurnal and a semiannual (Russell-McPherron) modulation.
    hour = df.index.hour + 0.0
    doy = df.index.dayofyear + 0.0
    cols["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=df.index)
    cols["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=df.index)
    cols["doy_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=df.index)
    cols["doy_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=df.index)

    X = pd.DataFrame(cols).astype("float32")
    y = dst.shift(-horizon).rename("dst_target")

    keep = X["dst_lag0"].notna()
    if require_target:
        keep &= y.notna()
    return X[keep], y[keep]
