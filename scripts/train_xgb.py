"""Train the XGBoost baseline and compare it with persistence on the held out test years.

Usage: python scripts/train_xgb.py [--horizon H] [--storm-weight K] [--ring-current]
       [--n-lags N] [--max-depth D] [--min-child-weight W] [--learning-rate R] [--subsample S]
       [--colsample-bytree C] [--n-estimators T] [--seed S] [--validation-only]
Writes: models/xgb_dst_{H}h{_swK}.ubj, reports/xgb_baseline{_Hh}{_swK}.json,
reports/xgb_test_predictions{_Hh}{_swK}.parquet
(the 1 hour reports keep their names without the suffix; _swK only when K is not 1;
--ring-current adds the O'Brien and McPherron terms and the tag _rc to every file name;
a lag count other than N_LAGS adds _lN and any other model option that differs from PARAMS
adds _hp, so a tuned run never overwrites the production files.
--validation-only fits, prints the validation scores and writes nothing; the test years are
not predicted)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oculstorm.data import TEST_START, VAL_START, load_hourly, split  # noqa: E402
from oculstorm.features import HORIZON, HORIZONS, N_LAGS, build_features, suffix, weight_tag  # noqa: E402
from oculstorm.metrics import evaluate, persistence, storm_weights, trend_persistence  # noqa: E402

PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    early_stopping_rounds=100,
    random_state=0,
)


# Model options that can be changed from the command line; the defaults are PARAMS.
TUNABLE = {
    "max_depth": int, "min_child_weight": float, "learning_rate": float, "subsample": float,
    "colsample_bytree": float, "n_estimators": int, "random_state": int,
}


def fit(
    X: pd.DataFrame, y: pd.Series, peak: float = 1.0, params: dict | None = None, test: bool = True,
) -> tuple[xgb.XGBRegressor, np.ndarray, np.ndarray | None]:
    """Fit on train, early stop on validation, return the model and validation and test predictions.

    The model learns the change Dst(t+h) - Dst(t), so an output of zero is exactly
    persistence and anything it learns is skill on top of it. With `peak` > 1 storm hours
    weigh more in the loss, on validation as well, so early stopping follows the same goal.
    `params` defaults to PARAMS. With test=False the test years are not predicted (None).
    """
    delta = y - X["dst_lag0"]
    (X_tr, X_va, X_te), (d_tr, d_va, _), (y_tr, y_va, _) = split(X), split(delta), split(y)
    model = xgb.XGBRegressor(**(PARAMS if params is None else params))
    model.fit(X_tr, d_tr, sample_weight=storm_weights(y_tr, peak), eval_set=[(X_va, d_va)],
              sample_weight_eval_set=[storm_weights(y_va, peak)], verbose=False)
    val_pred = X_va["dst_lag0"].to_numpy() + model.predict(X_va)
    test_pred = X_te["dst_lag0"].to_numpy() + model.predict(X_te) if test else None
    return model, val_pred, test_pred


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, default=HORIZON, choices=HORIZONS, help="hours ahead")
    parser.add_argument("--storm-weight", type=float, default=1.0, help="loss weight at Dst <= -100 nT relative to quiet time")
    parser.add_argument("--ring-current", action="store_true", help="add the O'Brien and McPherron (2000) ring current terms")
    parser.add_argument("--n-lags", type=int, default=N_LAGS, help="hours of history in the lag features (lag 0 .. N-1)")
    for name, kind in TUNABLE.items():
        flag = "--seed" if name == "random_state" else "--" + name.replace("_", "-")
        parser.add_argument(flag, dest=name, type=kind, default=PARAMS[name], help=f"XGBoost {name} (default {PARAMS[name]})")
    parser.add_argument("--validation-only", action="store_true", help="print validation scores only, write nothing, never predict the test years")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    h, peak, n_lags = args.horizon, args.storm_weight, args.n_lags
    params = {**PARAMS, **{name: getattr(args, name) for name in TUNABLE}}
    rc = "_rc" if args.ring_current else ""
    variant = rc + (f"_l{n_lags}" if n_lags != N_LAGS else "") + ("_hp" if params != PARAMS else "")
    tag = suffix(h) + weight_tag(peak) + variant
    df = load_hourly()
    X, y = build_features(df, horizon=h, ring_current=args.ring_current, n_lags=n_lags)
    (_, X_va, X_te), (_, y_va, y_te) = split(X), split(y)
    print(f"horizon: {h}h  storm weight: {peak:g}  ring current: {args.ring_current}  lags: {n_lags}  features: {X.shape[1]}  train: {len(split(X)[0]):,}  val: {len(split(X)[1]):,}")
    if params != PARAMS:
        print("params:", {k: v for k, v in params.items() if v != PARAMS[k]})

    if args.validation_only:
        model, val_pred, _ = fit(X, y, peak, params, test=False)
        print(f"trees: {model.best_iteration + 1} of {params['n_estimators']}")
        val_scores = evaluate(y_va, {"persistence": persistence(X_va), "xgboost (solar wind + Dst)": val_pred})
        print("\nValidation:")
        print(val_scores.round(4).to_string(index=False))
        return

    X_dst, _ = build_features(df, use_solar_wind=False, horizon=h, n_lags=n_lags)
    model, val_pred, pred = fit(X, y, peak, params)
    model_dst, _, pred_dst = fit(X_dst, y, peak, params)
    print(f"test: {len(X_te):,}  trees: full={model.best_iteration + 1}, dst-only={model_dst.best_iteration + 1}")

    predictions = {
        "persistence": persistence(X_te),
        "trend persistence": trend_persistence(X_te, h),
        "xgboost (Dst history only)": pred_dst,
        "xgboost (solar wind + Dst)": pred,
    }
    scores = evaluate(y_te, predictions)
    pd.set_option("display.width", 200)
    print(scores.round(3).to_string(index=False))
    # Choices such as the storm weight are made on these validation scores, never on test.
    val_scores = evaluate(y_va, {"persistence": persistence(X_va), "xgboost (solar wind + Dst)": val_pred})
    print("\nValidation:")
    print(val_scores.round(3).to_string(index=False))

    gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
    top = (gain / gain.sum()).sort_values(ascending=False).head(15)
    print("\nTop features by share of total gain:")
    print(top.round(4).to_string())

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    model.save_model(ROOT / "models" / f"xgb_dst_{h}h{weight_tag(peak)}{variant}.ubj")
    pd.DataFrame({"dst_observed": y_te, **predictions}).to_parquet(ROOT / "reports" / f"xgb_test_predictions{tag}.parquet")
    report = {
        "horizon_hours": h,
        "storm_weight": peak,
        "ring_current": args.ring_current,
        "n_lags": n_lags,
        "train": f"1995-01-01 .. {VAL_START} (exclusive)",
        "validation": f"{VAL_START} .. {TEST_START} (exclusive)",
        "test": f"{TEST_START} .. {y_te.index.max()}",
        "n_features": int(X.shape[1]),
        "best_iteration": int(model.best_iteration + 1),
        "params": params,
        "scores": json.loads(scores.round(4).to_json(orient="records")),
        "validation_scores": json.loads(val_scores.round(4).to_json(orient="records")),
        "top_features_gain_share": top.round(4).to_dict(),
    }
    (ROOT / "reports" / f"xgb_baseline{tag}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
