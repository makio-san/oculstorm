"""Bootstrap confidence intervals for the test RMSE of each model and the differences between them.

Usage: python scripts/compare_models.py [--horizon H] [--storm-weight K]
       python scripts/compare_models.py --ensemble [--out PATH]
Reads : reports/xgb_test_predictions{_Hh}.parquet, reports/gru_test_predictions{_Hh}.parquet
        and, with --storm-weight, the {_swK} versions of both
Writes: reports/comparison{_Hh}.json

With --ensemble (experiment, validation only) it instead loads the production 1 hour models,
predicts the validation years 2017 to 2020 from a table cut before 2021, fits GRU plus XGBoost
weights on one half, scores them on the other (both ways round) and writes --out.

Test hours are resampled in one week blocks, because errors an hour apart are not independent.
Every model is scored on the same resamples, so a difference is significant when its 95%
interval does not contain zero.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oculstorm.features import HORIZON, HORIZONS, suffix, weight_tag  # noqa: E402
from oculstorm.metrics import REGIMES, block_bootstrap_rmse, rmse  # noqa: E402

REPORTS = ROOT / "reports"
XGB, GRU = "xgboost (solar wind + Dst)", "gru (mean of 3 seeds)"


def load(h: int, peak: float) -> pd.DataFrame:
    """Observed Dst and every forecast, on the test hours all of them cover."""
    tag = suffix(h) + weight_tag(peak)
    xgb = pd.read_parquet(REPORTS / f"xgb_test_predictions{tag}.parquet")
    gru = pd.read_parquet(REPORTS / f"gru_test_predictions{tag}.parquet")
    name = "" if peak == 1 else " weighted"
    frame = pd.DataFrame({"observed": gru["dst_observed"], "persistence": gru["persistence"],
                          f"xgboost{name}": xgb[XGB], f"gru{name}": gru[GRU]})
    return frame.dropna()


def interval(values: np.ndarray) -> list[float]:
    return [round(float(v), 3) for v in np.nanpercentile(values, [2.5, 97.5])]


def validation_predictions() -> pd.DataFrame:
    """Production 1 hour GRU (mean of the seeds) and XGBoost forecasts on the validation hours.

    The table is cut before TEST_START first, so no test hour is used, not even as a target.
    """
    import xgboost as xgb

    from oculstorm.data import TEST_START, load_hourly, split
    from oculstorm.features import build_features
    from oculstorm.gru import DstGRU
    from oculstorm.sequences import build_windows

    models = ROOT / "models"
    df = load_hourly()
    df = df[df.index < TEST_START]

    booster = xgb.XGBRegressor(n_jobs=1)
    booster.load_model(models / "xgb_dst_1h.ubj")
    X, _ = build_features(df)
    X = split(X[booster.get_booster().feature_names])[1]
    xgb_pred = pd.Series(X["dst_lag0"].to_numpy() + booster.predict(X), index=X.index)

    # torch and xgboost each bundle their own OpenMP runtime: as in scripts/live_forecast.py,
    # xgboost runs first and torch is imported after it, with one thread, or macOS crashes.
    import torch

    torch.set_num_threads(1)
    stats = json.loads((models / "gru_channels.json").read_text())
    windows, meta = build_windows(df, stats)
    meta = split(meta)[1]
    rows = meta["row"].to_numpy()
    deltas = []
    for path in sorted(models.glob("gru_dst_1h_seed*.pt")):
        checkpoint = torch.load(path, weights_only=True)
        model = DstGRU(windows.shape[2], checkpoint["hidden"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            parts = [model(torch.from_numpy(np.ascontiguousarray(windows[rows[i : i + 4096]]))).numpy()
                     for i in range(0, len(rows), 4096)]
        deltas.append(np.concatenate(parts) * checkpoint["scale"])
    gru_pred = pd.Series(meta["dst_lag0"].to_numpy() + np.mean(deltas, axis=0), index=meta.index)

    frame = pd.DataFrame({"observed": meta["dst_target"], "dst_lag0": meta["dst_lag0"],
                          "persistence": meta["dst_lag0"], "gru": gru_pred, "xgboost": xgb_pred}).dropna()
    assert frame.index.max() < pd.Timestamp(TEST_START)
    return frame


def score(frame: pd.DataFrame, models: list[str], reference: str = "gru") -> dict:
    """RMSE per regime with block bootstrap intervals, and each model's paired difference to `reference`."""
    y = frame["observed"].to_numpy()
    out = {}
    for regime, mask_fn in REGIMES.items():
        mask = mask_fn(y)
        if not mask.any():
            out[regime] = {"n_hours": 0}  # e.g. no intense hours in 2019 to 2020
            continue
        boot = block_bootstrap_rmse(y, {m: frame[m] for m in models}, mask)
        col = {m: j for j, m in enumerate(models)}
        value ={m: rmse(y[mask], frame[m].to_numpy()[mask]) for m in models}
        result = {"n_hours": int(mask.sum()), "rmse": {}, "differences": {}}
        for m in models:
            result["rmse"][m] = {"value": round(value[m], 4), "ci95": interval(boot[:, col[m]])}
        for m in models:
            if m in (reference, "persistence"):
                continue
            diff = boot[:, col[m]] - boot[:, col[reference]]
            ratio = 100 * (boot[:, col[m]] ** 2 / boot[:, col[reference]] ** 2 - 1)
            low, high = interval(diff)
            result["differences"][f"{m} minus {reference}"] = {
                "rmse_value": round(value[m] - value[reference], 4),
                "rmse_ci95": [low, high],
                "mse_change_percent": round(100 * (value[m] ** 2 / value[reference] ** 2 - 1), 2),
                "mse_change_percent_ci95": [round(float(v), 2) for v in np.nanpercentile(ratio, [2.5, 97.5])],
                "significant": bool(low > 0 or high < 0),
                "share_first_better": round(float(np.nanmean(diff < 0)), 3),
            }
        out[regime] = result
    return out


def ensemble_experiment(out_path: Path) -> None:
    """Validation only: fit GRU plus XGBoost weights on one half of 2017 to 2020, score on the other."""
    from oculstorm.ensemble import BANDS, GRID, MIN_BAND_HOURS, band_of, combine, fit_band_weights, fit_weight

    started = time.time()
    frame = validation_predictions()
    halves = {"2017-2018": frame[frame.index < "2019-01-01"], "2019-2020": frame[frame.index >= "2019-01-01"]}
    models = ["persistence", "gru", "xgboost", "mean 50/50", "global weight", "band weights"]
    folds, held_out = {}, []
    for fit_name, score_name in [("2017-2018", "2019-2020"), ("2019-2020", "2017-2018")]:
        fit, held = halves[fit_name], halves[score_name].copy()
        w = fit_weight(fit["observed"], fit["gru"], fit["xgboost"])
        bands = fit_band_weights(fit["observed"], fit["gru"], fit["xgboost"], fit["dst_lag0"])
        held["mean 50/50"] = combine(held["gru"], held["xgboost"], 0.5)
        held["global weight"] = combine(held["gru"], held["xgboost"], w)
        held["band weights"] = combine(held["gru"], held["xgboost"], bands, held["dst_lag0"])
        held_out.append(held)
        band_hours = {b: int(n) for b, n in pd.Series(band_of(fit["dst_lag0"])).value_counts().items()}
        folds[f"fit {fit_name}, score {score_name}"] = {
            "gru_weight_global": w, "gru_weight_by_band": bands, "fit_hours_by_band": band_hours,
            "n_hours_scored": len(held), "scores": score(held, models),
        }
    pooled = pd.concat(held_out).sort_index()

    report = {
        "issue": 7,
        "experiment": "GRU plus XGBoost ensemble, pred = w * gru + (1 - w) * xgb",
        "horizon_hours": 1,
        "models": "production models/xgb_dst_1h.ubj and the mean of models/gru_dst_1h_seed*.pt, not retrained",
        "data": f"validation {frame.index.min()} .. {frame.index.max()}, table cut before 2021-01-01, test years not used",
        "weight_grid": [float(g) for g in GRID],
        "bands": list(BANDS),
        "min_band_hours": MIN_BAND_HOURS,
        "scoring_regimes": "by the observed target Dst, as in oculstorm.metrics.REGIMES",
        "bootstrap": {"block_hours": 168, "resamples": 2000, "seed": 0},
        "folds": folds,
        "pooled_out_of_sample": {"n_hours": len(pooled), "scores": score(pooled, models)},
    }
    report["promising_bar"] = ("retrain policy on the pooled out of sample hours: overall MSE at least 2% lower, or storm MSE "
                               "at least 3% lower with overall at most 1% higher, intervals excluding zero")
    report["promising"] = {}
    pooled_scores = report["pooled_out_of_sample"]["scores"]
    for m in ["mean 50/50", "global weight", "band weights"]:
        d_all = pooled_scores["all"]["differences"][f"{m} minus gru"]
        d_storm = pooled_scores["storm (Dst <= -50)"]["differences"][f"{m} minus gru"]
        overall = d_all["mse_change_percent"] <= -2 and d_all["rmse_ci95"][1] < 0
        storm = d_storm["mse_change_percent"] <= -3 and d_storm["rmse_ci95"][1] < 0 and d_all["mse_change_percent"] <= 1
        report["promising"][m] = bool(overall or storm)
    report["seconds"] = round(time.time() - started)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    for name, block in [*folds.items(), ("pooled out of sample", report["pooled_out_of_sample"])]:
        print(f"\n{name}  global {block.get('gru_weight_global', '-')}  bands {block.get('gru_weight_by_band', '-')}")
        for regime, result in block["scores"].items():
            if not result["n_hours"]:
                print(f"  {regime}: no hours")
                continue
            print(f"  {regime} ({result['n_hours']:,} h): " + "  ".join(f"{m} {r['value']:.3f}" for m, r in result["rmse"].items()))
            for d, v in result["differences"].items():
                print(f"    {d:<28} MSE {v['mse_change_percent']:+6.2f}% {v['mse_change_percent_ci95']}  "
                      f"RMSE {v['rmse_value']:+.3f} {v['rmse_ci95']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, default=HORIZON, choices=HORIZONS, help="hours ahead")
    parser.add_argument("--storm-weight", type=float, help="also compare the storm-weighted runs with this weight")
    parser.add_argument("--ensemble", action="store_true", help="validation only: fit and score GRU plus XGBoost weights at 1 hour")
    parser.add_argument("--out", type=Path, default=REPORTS / "experiments" / "7-gru-xgb-ensemble.json", help="report of --ensemble")
    args = parser.parse_args()
    if args.ensemble:
        ensemble_experiment(args.out)
        return
    h = args.horizon

    frame = load(h, 1.0)
    pairs = [("gru", "xgboost")]
    if args.storm_weight:
        weighted = load(h, args.storm_weight).drop(columns=["observed", "persistence"])
        frame = frame.join(weighted, how="inner")
        pairs += [("xgboost weighted", "xgboost"), ("gru weighted", "gru"), ("gru weighted", "xgboost weighted")]
    models = [c for c in frame.columns if c != "observed"]
    y = frame["observed"].to_numpy()

    out = {"horizon_hours": h, "storm_weight": args.storm_weight, "test_hours": len(frame),
           "block_hours": 168, "resamples": 2000, "regimes": {}}
    for regime, mask_fn in REGIMES.items():
        mask = mask_fn(y)
        boot = block_bootstrap_rmse(y, {m: frame[m] for m in models}, mask)
        col = {m: j for j, m in enumerate(models)}
        result = {"n_hours": int(mask.sum()), "rmse": {}, "differences": {}}
        for m in models:
            result["rmse"][m] = {"value": round(rmse(y[mask], frame[m].to_numpy()[mask]), 3), "ci95": interval(boot[:, col[m]])}
        for a, b in pairs:
            diff = boot[:, col[a]] - boot[:, col[b]]
            low, high = interval(diff)
            result["differences"][f"{a} minus {b}"] = {
                "value": round(result["rmse"][a]["value"] - result["rmse"][b]["value"], 3),
                "ci95": [low, high],
                "significant": bool(low > 0 or high < 0),
                "share_first_better": round(float(np.nanmean(diff < 0)), 3),
            }
        out["regimes"][regime] = result

    print(f"horizon {h}h, {len(frame):,} test hours, RMSE in nT with 95% block bootstrap intervals")
    for regime, result in out["regimes"].items():
        print(f"\n{regime}  ({result['n_hours']:,} hours)")
        for m, r in result["rmse"].items():
            print(f"  {m:<18} {r['value']:7.2f}  [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]")
        for name, d in result["differences"].items():
            flag = "significant" if d["significant"] else "not significant"
            print(f"  {name:<36} {d['value']:+6.2f}  [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]  {flag}")
    (REPORTS / f"comparison{suffix(h)}.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
