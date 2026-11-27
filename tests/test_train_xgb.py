"""The tuning options of scripts/train_xgb.py leave the production run unchanged."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("train_xgb", Path(__file__).resolve().parent.parent / "scripts" / "train_xgb.py")
train_xgb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_xgb)

from oculstorm.features import SOLAR_WIND  # noqa: E402


def _frame() -> tuple[pd.DataFrame, pd.Series]:
    hours = pd.date_range("2016-12-01", "2021-01-10", freq="h")
    rng = np.random.default_rng(0)
    dst = pd.Series(-20 + 30 * np.sin(np.arange(len(hours)) / 30) + rng.normal(0, 2, len(hours)), index=hours)
    X = pd.DataFrame({"dst_lag0": dst, "dst_lag1": dst.shift(1)}, index=hours).astype("float32")
    return X, dst.shift(-1).rename("dst_target")


def test_defaults_are_the_production_params():
    args = vars(train_xgb.build_parser().parse_args([]))
    assert {name: args[name] for name in train_xgb.TUNABLE} == {name: train_xgb.PARAMS[name] for name in train_xgb.TUNABLE}
    assert args["n_lags"] == train_xgb.N_LAGS
    assert not args["validation_only"]


def test_fit_without_test_never_predicts_the_test_years():
    X, y = _frame()
    keep = y.notna()
    X, y = X[keep], y[keep]
    params = {**train_xgb.PARAMS, "n_estimators": 5, "early_stopping_rounds": 2}
    model, val_pred, test_pred = train_xgb.fit(X, y, params=params, test=False)
    assert test_pred is None
    assert len(val_pred) == len(train_xgb.split(X)[1])
    _, _, test_pred = train_xgb.fit(X, y, params=params)
    assert len(test_pred) == len(train_xgb.split(X)[2])


def test_default_objective_leaves_params_unchanged():
    args = train_xgb.build_parser().parse_args([])
    assert args.objective == "squarederror"
    assert train_xgb.objective_params(args.objective, args.huber_slope) == {}


def test_robust_objectives_set_loss_and_matching_early_stopping_metric():
    assert train_xgb.objective_params("absolute") == {"objective": "reg:absoluteerror", "eval_metric": "mae"}
    assert train_xgb.objective_params("pseudohuber", 2.0) == {"objective": "reg:pseudohubererror", "eval_metric": "mphe", "huber_slope": 2.0}


def test_robust_objective_fits():
    X, y = _frame()
    keep = y.notna()
    X, y = X[keep], y[keep]
    for objective in ("pseudohuber", "absolute"):
        params = {**train_xgb.PARAMS, "n_estimators": 5, "early_stopping_rounds": 2, **train_xgb.objective_params(objective)}
        _, val_pred, _ = train_xgb.fit(X, y, params=params, test=False)
        assert np.isfinite(val_pred).all()


def test_main_validation_only_runs_with_a_robust_objective(monkeypatch, capsys):
    X, y = _frame()
    keep = y.notna()
    X, y = X[keep], y[keep]
    monkeypatch.setitem(train_xgb.PARAMS, "n_estimators", 5)
    monkeypatch.setitem(train_xgb.PARAMS, "early_stopping_rounds", 2)
    monkeypatch.setattr(train_xgb, "load_hourly", lambda: None)
    monkeypatch.setattr(train_xgb, "build_features", lambda *a, **k: (X, y))
    for objective in ("absolute", "pseudohuber"):
        monkeypatch.setattr(train_xgb.sys, "argv", ["train_xgb.py", "--validation-only", "--objective", objective])
        train_xgb.main()
        out = capsys.readouterr().out
        assert "reg:" in out and "Validation:" in out


def _hourly(n: int = 200) -> pd.DataFrame:
    """A small hourly table with every column build_features reads."""
    hours = pd.date_range("2016-12-01", periods=n, freq="h")
    rng = np.random.default_rng(1)
    columns = ["dst", "f107", *SOLAR_WIND]
    df = pd.DataFrame(rng.normal(0, 3, (n, len(columns))), index=hours, columns=columns)
    df["flow_speed"] = 400 + df["flow_speed"]
    df["flow_pressure"] = 2 + df["flow_pressure"].abs()
    return df


def test_default_feature_set_is_unchanged_by_the_no_density_option():
    assert not train_xgb.build_parser().parse_args([]).no_density
    df = _hourly()
    X_default, y_default = train_xgb.build_features(df)
    X_off, y_off = train_xgb.build_features(df, drop_plasma=False)
    pd.testing.assert_frame_equal(X_default, X_off)
    pd.testing.assert_series_equal(y_default, y_off)
    # The production 1 hour model has 89 inputs (reports/xgb_baseline.json), plasma moments included.
    assert X_default.shape[1] == 89
    for name in train_xgb.PLASMA_MOMENTS:
        assert all(f"{name}_lag{lag}" in X_default for lag in range(train_xgb.N_LAGS))


def test_no_density_drops_only_the_plasma_moments():
    df = _hourly()
    X_default, _ = train_xgb.build_features(df)
    X_nd, _ = train_xgb.build_features(df, drop_plasma=True)
    dropped = {f"{name}_lag{lag}" for name in train_xgb.PLASMA_MOMENTS for lag in range(train_xgb.N_LAGS)}
    assert set(X_default.columns) - set(X_nd.columns) == dropped
    assert set(X_nd.columns) <= set(X_default.columns)
    pd.testing.assert_frame_equal(X_nd, X_default[X_nd.columns])
    assert {"electric_field_lag0", "bz_gsm_lag0", "flow_speed_lag0", "vbs_lag0", "newell_lag0"} <= set(X_nd.columns)


def test_no_density_cannot_be_combined_with_the_ring_current_terms():
    with pytest.raises(ValueError):
        train_xgb.build_features(_hourly(), drop_plasma=True, ring_current=True)
