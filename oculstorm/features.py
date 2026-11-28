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
MEMORY_TAUS = (3, 8, 24)  # e-folding times (hours) of the optional decaying driver memory
MEMORY_DRIVERS = ("vbs", "newell")

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


AGE_CAP = 48  # hours; older gaps all look the same to the gap age features
AGE_SOURCES = {"imf_age": "bz_gsm", "plasma_age": "flow_speed"}  # block -> column that marks it observed


def observation_age(observed: pd.Series, cap: int = AGE_CAP) -> pd.Series:
    """Hours since the last observed hour at or before t (0 if t itself is observed), capped at `cap`.

    Counts backwards from t only: the value at t depends on `observed` up to t. Hours before
    the first observation get `cap`.
    """
    step = np.arange(len(observed), dtype="float64")
    last = pd.Series(np.where(observed.to_numpy(dtype=bool), step, np.nan), index=observed.index).ffill()
    return (step - last).fillna(cap).clip(upper=cap)


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


def decaying_memory(x: pd.Series, tau: float) -> pd.Series:
    """Causal exponentially weighted mean of `x` with e-folding time `tau` hours.

    m(t) = a * x(t) + (1 - a) * m(t-1) with a = 1 - exp(-1/tau), so it uses hour t and earlier
    only. Missing hours keep the last value and the weights of older hours decay through the gap.
    """
    return x.ewm(alpha=1 - np.exp(-1 / tau), adjust=False).mean()
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
    memory: tuple[float, ...] = (),
    ring_current: bool = False, n_lags: int = N_LAGS, gap_age: bool = False,
    dst_skip: int = 0, newest_dst_error: pd.Series | None = None, newest_flag: float | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y is Dst at t+horizon and X only looks backwards from t.

    X does not depend on the horizon; only the target moves.

    With use_solar_wind=False only the Dst history and calendar terms are kept,
    which shows how much of the skill actually comes from the solar wind.
    With require_target=False the newest hours, whose target is not known yet, are kept.
    `memory` lists time constants (hours) of decaying means of VBs and the Newell function
    (see MEMORY_TAUS); empty, the default, adds none. They are computed on the full hourly
    index before any row is dropped.
    With ring_current=True the O'Brien and McPherron (2000) terms are added (needs the solar wind).
    n_lags sets how many hours of history (lag 0 .. n_lags-1) the lag columns cover; the
    default N_LAGS is what the production models and the live forecast use.
    With gap_age=True imf_age and plasma_age are added: hours since bz_gsm and flow_speed were
    last observed at or before t, capped at AGE_CAP (needs the solar wind).
    dst_skip=S (issue 30) builds every Dst feature from Dst shifted by S hours, so the Dst
    history of row t ends at t-S: dst_lag0 is Dst(t-S), and persistence and the learned change
    start from that hour. Solar wind features and the target stay where they are. The default
    0 is the production layout. Not available together with ring_current.
    newest_dst_error (issue 30) is an error in nT, indexed like df, added to Dst(t) as seen
    from row t only: dst_lag0, both differences and the 24 hour minimum and mean of row t see
    the erroneous newest hour, while the older lags, the same hour seen from later rows and
    the target stay clean. It imitates a quick-look value that is not settled when the
    forecast is issued. None, the default, adds nothing.
    newest_flag (issue 49) adds the constant column dst_newest_unsettled with this value: 0
    on clean rows, the error level on rows built with newest_dst_error, so the model can be
    told whether the newest Dst hour is settled. It is known at issue time and depends on no
    data. None, the default, adds no column.
    """
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    if dst_skip < 0:
        raise ValueError("dst_skip must be 0 or more")
    if ring_current and (dst_skip or newest_dst_error is not None):
        raise ValueError("ring_current cannot be combined with dst_skip or newest_dst_error")
    if dst_skip and newest_dst_error is not None:
        raise ValueError("dst_skip and newest_dst_error cannot be combined")
    cols: dict[str, pd.Series] = {}
    target_source = df["dst"]
    dst = target_source.shift(dst_skip) if dst_skip else target_source

    for lag in range(n_lags):
        cols[f"dst_lag{lag}"] = dst.shift(lag)
    cols["dst_diff1"] = dst - dst.shift(1)
    cols["dst_diff3"] = dst - dst.shift(3)
    cols["dst_min24"] = dst.rolling(24, min_periods=1).min()
    cols["dst_mean24"] = dst.rolling(24, min_periods=1).mean()
    if newest_dst_error is not None:
        error = newest_dst_error.reindex(df.index).astype("float64")
        if error.isna().any():
            raise ValueError("newest_dst_error must cover every hour of the table")
        for name in ["dst_lag0", "dst_diff1", "dst_diff3"]:
            cols[name] = cols[name] + error
        # The window of row t is the clean hours t-23 .. t-1 and the erroneous hour t.
        cols["dst_min24"] = np.fmin(dst + error, dst.shift(1).rolling(23, min_periods=1).min())
        cols["dst_mean24"] = cols["dst_mean24"] + error / dst.rolling(24, min_periods=1).count()
    if newest_flag is not None:
        cols["dst_newest_unsettled"] = pd.Series(float(newest_flag), index=df.index)

    if use_solar_wind:
        drivers = pd.concat([df[SOLAR_WIND], _coupling(df)], axis=1)
        for name in drivers:
            for lag in range(n_lags):
                cols[f"{name}_lag{lag}"] = drivers[name].shift(lag)
        for name in ["vbs", "newell", "bz_gsm", "flow_speed"]:
            for window in (3, 12):
                cols[f"{name}_mean{window}"] = drivers[name].rolling(window, min_periods=1).mean()
        for name in MEMORY_DRIVERS:
            for tau in memory:
                cols[f"{name}_ewm{tau:g}"] = decaying_memory(drivers[name], tau)
        cols["f107"] = df["f107"]
        if gap_age:
            for name, column in AGE_SOURCES.items():
                cols[name] = observation_age(df[column].notna())
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
    y = target_source.shift(-horizon).rename("dst_target")

    keep = X["dst_lag0"].notna()
    if require_target:
        keep &= y.notna()
    return X[keep], y[keep]
