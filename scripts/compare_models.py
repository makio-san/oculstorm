"""Bootstrap confidence intervals for the test RMSE of each model and the differences between them.

Usage: python scripts/compare_models.py [--horizon H] [--storm-weight K]
Reads : reports/xgb_test_predictions{_Hh}.parquet, reports/gru_test_predictions{_Hh}.parquet
        and, with --storm-weight, the {_swK} versions of both
Writes: reports/comparison{_Hh}.json

Test hours are resampled in one week blocks, because errors an hour apart are not independent.
Every model is scored on the same resamples, so a difference is significant when its 95%
interval does not contain zero.
"""
import argparse
import json
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, default=HORIZON, choices=HORIZONS, help="hours ahead")
    parser.add_argument("--storm-weight", type=float, help="also compare the storm-weighted runs with this weight")
    args = parser.parse_args()
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
