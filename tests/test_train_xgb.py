"""The tuning options of scripts/train_xgb.py leave the production run unchanged."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("train_xgb", Path(__file__).resolve().parent.parent / "scripts" / "train_xgb.py")
train_xgb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_xgb)


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


def test_quantiles_are_off_by_default():
    args = train_xgb.build_parser().parse_args([])
    assert not args.quantiles


def test_quantile_model_gives_one_column_per_quantile():
    X, y = _frame()
    keep = y.notna()
    X, y = X[keep], y[keep]
    params = {**train_xgb.PARAMS, "n_estimators": 20, "early_stopping_rounds": 5, **train_xgb.quantile_params()}
    _, val_pred, test_pred = train_xgb.fit(X, y, params=params, test=False)
    assert test_pred is None
    assert val_pred.shape == (len(train_xgb.split(X)[1]), len(train_xgb.QUANTILES))
    scores = train_xgb.interval_scores(train_xgb.split(y)[1], val_pred)
    scores = scores[scores["n_hours"] > 0]  # an empty regime gives NaN
    assert (scores["mean_width"] >= 0).all() and scores["coverage"].between(0, 1).all()


def test_interval_scores_count_hours_inside_the_band():
    y = pd.Series([-10.0, -60.0, -120.0, 0.0])
    q = np.array([[-11, -10, -9], [-50, -55, -40], [-130, -120, -110], [5, 3, 1]], dtype=float)  # last row crosses
    scores = train_xgb.interval_scores(y, q).set_index("regime")
    assert scores.loc["all", "coverage"] == 0.5 and scores.loc["all", "n_hours"] == 4
    assert scores.loc["intense (Dst <= -100)", "coverage"] == 1.0
    assert scores.loc["quiet (Dst > -50)", "mean_width"] == 3.0


def test_main_quantiles_needs_validation_only(monkeypatch):
    monkeypatch.setattr(train_xgb.sys, "argv", ["train_xgb.py", "--quantiles"])
    try:
        train_xgb.main()
    except SystemExit as exit_:
        assert exit_.code == 2
    else:
        raise AssertionError("--quantiles without --validation-only must fail")


def test_main_validation_only_runs_with_quantiles(monkeypatch, capsys):
    X, y = _frame()
    keep = y.notna()
    X, y = X[keep], y[keep]
    monkeypatch.setitem(train_xgb.PARAMS, "n_estimators", 5)
    monkeypatch.setitem(train_xgb.PARAMS, "early_stopping_rounds", 2)
    monkeypatch.setattr(train_xgb, "load_hourly", lambda: None)
    monkeypatch.setattr(train_xgb, "build_features", lambda *a, **k: (X, y))
    monkeypatch.setattr(train_xgb.sys, "argv", ["train_xgb.py", "--validation-only", "--quantiles"])
    train_xgb.main()
    out = capsys.readouterr().out
    assert "quantile interval" in out and "coverage" in out and "Validation:" in out
