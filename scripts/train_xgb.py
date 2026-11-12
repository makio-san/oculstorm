"""Train the XGBoost baseline and compare it with persistence on the held out test years.

Usage: python scripts/train_xgb.py
Writes: models/xgb_dst_1h.json, reports/xgb_baseline.json, reports/xgb_test_predictions.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oculstorm.data import TEST_START, VAL_START, load_hourly, split  # noqa: E402
from oculstorm.features import HORIZON, build_features  # noqa: E402
from oculstorm.metrics import evaluate, persistence, trend_persistence  # noqa: E402

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


def fit(X: pd.DataFrame, y: pd.Series) -> tuple[xgb.XGBRegressor, np.ndarray]:
    """Fit on train, early stop on validation, return the model and test predictions.

    The model learns the change Dst(t+1) - Dst(t), so an output of zero is exactly
    persistence and anything it learns is skill on top of it.
    """
    delta = y - X["dst_lag0"]
    (X_tr, X_va, X_te), (d_tr, d_va, _) = split(X), split(delta)
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X_tr, d_tr, eval_set=[(X_va, d_va)], verbose=False)
    return model, X_te["dst_lag0"].to_numpy() + model.predict(X_te)


def main() -> None:
    df = load_hourly()
    X, y = build_features(df)
    X_dst, _ = build_features(df, use_solar_wind=False)
    X_te, y_te = split(X)[2], split(y)[2]
    print(f"features: {X.shape[1]}  train: {len(split(X)[0]):,}  val: {len(split(X)[1]):,}  test: {len(X_te):,}")

    model, pred = fit(X, y)
    model_dst, pred_dst = fit(X_dst, y)
    print(f"trees: full={model.best_iteration + 1}, dst-only={model_dst.best_iteration + 1}")

    predictions = {
        "persistence": persistence(X_te),
        "trend persistence": trend_persistence(X_te),
        "xgboost (Dst history only)": pred_dst,
        "xgboost (solar wind + Dst)": pred,
    }
    scores = evaluate(y_te, predictions)
    pd.set_option("display.width", 200)
    print(scores.round(3).to_string(index=False))

    gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
    top = (gain / gain.sum()).sort_values(ascending=False).head(15)
    print("\nTop features by share of total gain:")
    print(top.round(4).to_string())

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    model.save_model(ROOT / "models" / "xgb_dst_1h.json")
    pd.DataFrame({"dst_observed": y_te, **predictions}).to_parquet(ROOT / "reports" / "xgb_test_predictions.parquet")
    report = {
        "horizon_hours": HORIZON,
        "train": f"1995-01-01 .. {VAL_START} (exclusive)",
        "validation": f"{VAL_START} .. {TEST_START} (exclusive)",
        "test": f"{TEST_START} .. {y_te.index.max()}",
        "n_features": int(X.shape[1]),
        "best_iteration": int(model.best_iteration + 1),
        "params": PARAMS,
        "scores": json.loads(scores.round(4).to_json(orient="records")),
        "top_features_gain_share": top.round(4).to_dict(),
    }
    (ROOT / "reports" / "xgb_baseline.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
