"""Train the XGBoost baseline and compare it with persistence on the held out test years.

Usage: python scripts/train_xgb.py [--horizon H] [--storm-weight K] [--ring-current] [--driver-memory]
       [--n-lags N] [--max-depth D] [--min-child-weight W] [--learning-rate R] [--subsample S]
       [--colsample-bytree C] [--n-estimators T] [--seed S] [--validation-only]
       [--dst-offset-aug K] [--dst-skip S] [--dst-newest-noise K]
Writes: models/xgb_dst_{H}h{_swK}{variant}.ubj, reports/xgb_baseline{_Hh}{_swK}{variant}.json,
reports/xgb_test_predictions{_Hh}{_swK}{variant}.parquet
(the 1 hour reports keep their names without the suffix; _swK only when K is not 1;
--ring-current adds the O'Brien and McPherron terms and the tag _rc to every file name;
--driver-memory adds decaying means of VBs and Newell and the tag _mem;
a lag count other than N_LAGS adds _lN and any other model option that differs from PARAMS
adds _hp, and --dst-offset-aug K adds _oaK, so a tuned run never overwrites the production files.
--dst-skip S ends the Dst history at t-S and adds _dsS: persistence and the learned change then
start from Dst(t-S), so the persistence rows are S+H hour persistence;
--dst-newest-noise K trains on K extra copies of the train years whose newest Dst hour carries
an AR(1) error and adds _nnK.
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

from oculstorm.augment import consistent_rows, newest_hour_error, offset_table  # noqa: E402
from oculstorm.data import TEST_START, VAL_START, load_hourly, split  # noqa: E402
from oculstorm.features import HORIZON, HORIZONS, MEMORY_TAUS, N_LAGS, build_features, suffix, weight_tag  # noqa: E402
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


def offset_copies(df: pd.DataFrame, copies: int, horizon: int, ring_current: bool = False, n_lags: int = N_LAGS, seed: int = 1, memory: tuple[float, ...] = ()) -> tuple[pd.DataFrame, pd.Series]:
    """Train years of `copies` Dst offset copies of the table (oculstorm.augment), features and target.

    Only rows before the validation years are returned, so no augmented row reaches validation or test.
    """
    parts = []
    for k in range(copies):
        shifted = offset_table(df, seed + k)
        X_k, y_k = build_features(shifted, horizon=horizon, ring_current=ring_current, n_lags=n_lags, memory=memory)
        rows = pd.Series(consistent_rows(df.index, horizon), index=df.index).reindex(X_k.index)
        X_k, y_k = X_k[rows.to_numpy()], y_k[rows.to_numpy()]
        X_k, y_k = split(X_k)[0], split(y_k)[0]
        parts.append((X_k, y_k))
    return pd.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts])


NOISE_LEVELS = (1.0, 0.5)  # share of the newest hour error in the first and the second noisy copy


def noisy_copies(df: pd.DataFrame, copies: int, horizon: int, n_lags: int = N_LAGS, seed: int = 1, memory: tuple[float, ...] = ()) -> tuple[pd.DataFrame, pd.Series]:
    """Train years of `copies` copies whose newest Dst hour carries an AR(1) error (issue 30), features and target.

    Copy k uses NOISE_LEVELS[k] of the error. Only the Dst(t) seen from row t is wrong; the
    older lags and the target are clean. Only rows before the validation years are returned.
    """
    if not 0 <= copies <= len(NOISE_LEVELS):
        raise ValueError(f"at most {len(NOISE_LEVELS)} noisy copies")
    parts = []
    for k in range(copies):
        error = newest_hour_error(df.index, seed + k, NOISE_LEVELS[k])
        X_k, y_k = build_features(df, horizon=horizon, n_lags=n_lags, memory=memory, newest_dst_error=error)
        parts.append((split(X_k)[0], split(y_k)[0]))
    return pd.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts])


def fit(
    X: pd.DataFrame, y: pd.Series, peak: float = 1.0, params: dict | None = None, test: bool = True,
    extra: tuple[pd.DataFrame, pd.Series] | None = None,
) -> tuple[xgb.XGBRegressor, np.ndarray, np.ndarray | None]:
    """Fit on train, early stop on validation, return the model and validation and test predictions.

    The model learns the change Dst(t+h) - Dst(t), so an output of zero is exactly
    persistence and anything it learns is skill on top of it. With `peak` > 1 storm hours
    weigh more in the loss, on validation as well, so early stopping follows the same goal.
    `params` defaults to PARAMS. With test=False the test years are not predicted (None).
    `extra` is more training rows (offset_copies, noisy_copies); validation never gets any.
    """
    delta = y - X["dst_lag0"]
    (X_tr, X_va, X_te), (d_tr, d_va, _), (y_tr, y_va, _) = split(X), split(delta), split(y)
    if extra is not None:
        X_ex, y_ex = extra
        X_tr, y_tr = pd.concat([X_tr, X_ex]), pd.concat([y_tr, y_ex])
        d_tr = y_tr - X_tr["dst_lag0"]
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
    parser.add_argument("--driver-memory", action="store_true",
                        help=f"add decaying means of VBs and Newell with time constants {MEMORY_TAUS} h")
    parser.add_argument("--n-lags", type=int, default=N_LAGS, help="hours of history in the lag features (lag 0 .. N-1)")
    for name, kind in TUNABLE.items():
        flag = "--seed" if name == "random_state" else "--" + name.replace("_", "-")
        parser.add_argument(flag, dest=name, type=kind, default=PARAMS[name], help=f"XGBoost {name} (default {PARAMS[name]})")
    parser.add_argument("--dst-offset-aug", type=int, default=0, metavar="K", help="train on K extra copies of the train years with a random Dst offset per 168 hours (0 = off)")
    parser.add_argument("--dst-skip", type=int, default=0, metavar="S", help="end the Dst history at t-S instead of t (0 = off)")
    parser.add_argument("--dst-newest-noise", type=int, default=0, metavar="K", choices=range(len(NOISE_LEVELS) + 1),
                        help="train on K extra copies of the train years with an AR(1) error on the newest Dst hour (0 = off)")
    parser.add_argument("--validation-only", action="store_true", help="print validation scores only, write nothing, never predict the test years")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.ring_current and (args.dst_skip or args.dst_newest_noise):
        parser.error("--ring-current cannot be combined with --dst-skip or --dst-newest-noise")
    if args.dst_skip and (args.dst_newest_noise or args.dst_offset_aug):
        parser.error("--dst-skip cannot be combined with --dst-newest-noise or --dst-offset-aug")
    skip, noise = args.dst_skip, args.dst_newest_noise
    h, peak, n_lags = args.horizon, args.storm_weight, args.n_lags
    params = {**PARAMS, **{name: getattr(args, name) for name in TUNABLE}}
    rc = "_rc" if args.ring_current else ""
    memory = MEMORY_TAUS if args.driver_memory else ()
    variant = rc + ("_mem" if memory else "") + (f"_l{n_lags}" if n_lags != N_LAGS else "") + ("_hp" if params != PARAMS else "")
    variant += f"_oa{args.dst_offset_aug}" if args.dst_offset_aug else ""
    variant += (f"_ds{skip}" if skip else "") + (f"_nn{noise}" if noise else "")
    tag = suffix(h) + weight_tag(peak) + variant
    df = load_hourly()
    X, y = build_features(df, horizon=h, ring_current=args.ring_current, n_lags=n_lags, memory=memory, dst_skip=skip)
    (_, X_va, X_te), (_, y_va, y_te) = split(X), split(y)
    print(f"horizon: {h}h  storm weight: {peak:g}  ring current: {args.ring_current}  lags: {n_lags}  features: {X.shape[1]}  train: {len(split(X)[0]):,}  val: {len(split(X)[1]):,}")
    if params != PARAMS:
        print("params:", {k: v for k, v in params.items() if v != PARAMS[k]})

    extra = offset_copies(df, args.dst_offset_aug, h, args.ring_current, n_lags, memory=memory) if args.dst_offset_aug else None
    if extra is not None:
        print(f"offset augmentation: {args.dst_offset_aug} copies, {len(extra[0]):,} extra train rows")
    if noise:
        noisy = noisy_copies(df, noise, h, n_lags, memory=memory)
        extra = noisy if extra is None else (pd.concat([extra[0], noisy[0]]), pd.concat([extra[1], noisy[1]]))
        print(f"newest hour noise: {noise} copies, {len(noisy[0]):,} extra train rows")
    if skip:
        print(f"Dst history ends at t-{skip}: persistence below is Dst(t-{skip}), {skip + h} hours before the target")

    if args.validation_only:
        model, val_pred, _ = fit(X, y, peak, params, test=False, extra=extra)
        print(f"trees: {model.best_iteration + 1} of {params['n_estimators']}")
        val_scores = evaluate(y_va, {"persistence": persistence(X_va), "xgboost (solar wind + Dst)": val_pred})
        print("\nValidation:")
        print(val_scores.round(4).to_string(index=False))
        return

    X_dst, _ = build_features(df, use_solar_wind=False, horizon=h, n_lags=n_lags, dst_skip=skip)
    model, val_pred, pred = fit(X, y, peak, params, extra=extra)
    extra_dst = offset_copies(df, args.dst_offset_aug, h, False, n_lags) if args.dst_offset_aug else None
    if extra_dst is not None:
        extra_dst = (extra_dst[0][[c for c in X_dst.columns]], extra_dst[1])
    if noise:
        noisy_dst = (noisy[0][list(X_dst.columns)], noisy[1])
        extra_dst = noisy_dst if extra_dst is None else (pd.concat([extra_dst[0], noisy_dst[0]]), pd.concat([extra_dst[1], noisy_dst[1]]))
    model_dst, _, pred_dst = fit(X_dst, y, peak, params, extra=extra_dst)
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
        "driver_memory_taus": list(memory),
        "n_lags": n_lags,
        "dst_offset_aug": args.dst_offset_aug,
        "dst_skip": skip,
        "dst_newest_noise": noise,
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
