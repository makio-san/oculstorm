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


def _long_frame() -> tuple[pd.DataFrame, pd.Series]:
    hours = pd.date_range("2011-06-01", "2021-01-10", freq="6h")
    rng = np.random.default_rng(1)
    dst = pd.Series(-20 + 30 * np.sin(np.arange(len(hours)) / 7) + rng.normal(0, 2, len(hours)), index=hours)
    X = pd.DataFrame({"dst_lag0": dst, "dst_lag1": dst.shift(1)}, index=hours).astype("float32")
    y = dst.shift(-1).rename("dst_target")
    keep = y.notna()
    return X[keep], y[keep]


def test_backtest_split_keeps_validation_and_test_out_of_fit_and_early_stopping():
    X, _ = _long_frame()
    fit_rows, early_rows, scored = train_xgb.backtest_split(X, 4)
    assert fit_rows.index.max() < pd.Timestamp("2013-01-01") <= early_rows.index.min()
    assert early_rows.index.max() < pd.Timestamp(train_xgb.VAL_START) <= scored.index.min()
    assert scored.index.max() < pd.Timestamp(train_xgb.TEST_START)


def test_backtest_defaults_are_off():
    args = train_xgb.build_parser().parse_args([])
    assert args.backtest_shift == 0 and args.fixed_trees == 0


def test_backtest_options_never_see_the_scored_or_test_targets():
    X, y = _long_frame()
    params = {**train_xgb.PARAMS, "n_estimators": 20, "early_stopping_rounds": 3}
    hidden = y.copy()
    hidden[hidden.index >= train_xgb.VAL_START] = 500.0  # validation and test targets changed
    for options in ({"backtest_shift": 4}, {"fixed_trees": 7}):
        model, val_pred, test_pred = train_xgb.fit(X, y, params=params, **options)
        _, val_hidden, _ = train_xgb.fit(X, hidden, params=params, **options)
        assert test_pred is None
        assert len(val_pred) == len(train_xgb.split(X)[1])
        np.testing.assert_array_equal(val_pred, val_hidden)
    assert train_xgb.n_trees(model) == 7


def test_backtest_early_stopping_ignores_the_years_it_scores():
    X, y = _long_frame()
    params = {**train_xgb.PARAMS, "n_estimators": 20, "early_stopping_rounds": 3}
    model, _, _ = train_xgb.fit(X, y, params=params, backtest_shift=4)
    early = train_xgb.backtest_split(X, 4)[1]
    assert len(model.evals_result()["validation_0"]["rmse"]) >= train_xgb.n_trees(model)
    hidden = y.copy()
    hidden[(hidden.index >= train_xgb.VAL_START)] = 500.0
    model_hidden, _, _ = train_xgb.fit(X, hidden, params=params, backtest_shift=4)
    assert train_xgb.n_trees(model_hidden) == train_xgb.n_trees(model)
    assert len(early) > 0
