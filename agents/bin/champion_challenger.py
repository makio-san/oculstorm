"""Compare candidate 1 hour models with the current ones on the same hours, with paired intervals.

Usage (from inside an agent worktree, with the parquet downloaded):
  /Users/esma/Projects/oculstorm/.venv/bin/python agents/bin/champion_challenger.py \
      --current /path/to/current/models --candidate models --split val|test [--out report.json]

Each directory holds xgb_dst_1h.ubj, gru_dst_1h_seed*.pt and gru_channels.json, the files the
live forecast loads. Every model is scored on the same hours and the same one week block
resamples, so `candidate minus current` intervals are paired. The verdict per model family
follows agents/retrain-policy.md:

  pass  the candidate is significantly better overall, or significantly better in storm hours
        while not significantly worse overall, and it is significantly worse in no regime
  fail  anything else

`--split val` is for choosing. `--split test` is for the one confirmation a candidate gets;
log every test run in agents/state/test-ledger.md.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from oculstorm.data import load_hourly, split  # noqa: E402
from oculstorm.features import build_features  # noqa: E402
from oculstorm.metrics import REGIMES, block_bootstrap_rmse, rmse  # noqa: E402


def xgb_predict(models: Path, df: pd.DataFrame) -> pd.Series:
    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(models / "xgb_dst_1h.ubj")
    X, _ = build_features(df)
    X = X[model.get_booster().feature_names]
    return pd.Series(X["dst_lag0"].to_numpy() + model.predict(X), index=X.index)


def gru_predict(models: Path, df: pd.DataFrame) -> pd.Series:
    import torch

    from oculstorm.gru import DstGRU
    from oculstorm.sequences import build_windows

    torch.set_num_threads(1)  # torch and xgboost OpenMP runtimes clash when both run threaded
    stats = json.loads((models / "gru_channels.json").read_text())
    windows, meta = build_windows(df, stats)
    rows = meta["row"].to_numpy()
    deltas = []
    for path in sorted(models.glob("gru_dst_1h_seed*.pt")):
        checkpoint = torch.load(path, weights_only=True)
        model = DstGRU(windows.shape[2], checkpoint["hidden"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(rows), 4096):
                out.append(model(torch.from_numpy(np.ascontiguousarray(windows[rows[start : start + 4096]]))).numpy())
        deltas.append(np.concatenate(out) * checkpoint["scale"])
    if not deltas:
        raise SystemExit(f"no gru_dst_1h_seed*.pt in {models}")
    return pd.Series(meta["dst_lag0"].to_numpy() + np.mean(deltas, axis=0), index=meta.index)


def interval(values: np.ndarray) -> list[float]:
    return [round(float(v), 3) for v in np.nanpercentile(values, [2.5, 97.5])]


def verdict(regimes: dict, family: str) -> dict:
    diffs = {r: regimes[r]["differences"][f"{family} candidate minus current"] for r in regimes}
    better = lambda d: d["ci95"][1] < 0  # noqa: E731
    worse = lambda d: d["ci95"][0] > 0  # noqa: E731
    worse_in = [r for r, d in diffs.items() if worse(d)]
    storm = "storm (Dst <= -50)"
    gain = better(diffs["all"]) or (better(diffs[storm]) and not worse(diffs["all"]))
    passed = gain and not worse_in
    reason = ("significantly worse in: " + ", ".join(worse_in)) if worse_in else (
        "significant gain" if gain else "no significant gain overall or in storm hours")
    return {"verdict": "pass" if passed else "fail", "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", type=Path, required=True, help="directory with the current models")
    parser.add_argument("--candidate", type=Path, required=True, help="directory with the candidate models")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args()

    df = load_hourly()
    # xgboost first, torch second (see live_forecast.py).
    preds = {"xgb current": xgb_predict(args.current, df), "xgb candidate": xgb_predict(args.candidate, df)}
    preds |= {"gru current": gru_predict(args.current, df), "gru candidate": gru_predict(args.candidate, df)}
    _, target = build_features(df)
    frame = pd.DataFrame(preds)
    frame["persistence"] = df["dst"].reindex(frame.index)
    frame["observed"] = target.reindex(frame.index)
    frame = frame.dropna()
    part = dict(zip(["train", "val", "test"], split(frame)))[args.split]
    y = part["observed"].to_numpy()
    models = [c for c in part.columns if c != "observed"]

    out = {"split": args.split, "hours": len(part), "from": str(part.index.min()), "to": str(part.index.max()),
           "current": str(args.current), "candidate": str(args.candidate), "regimes": {}}
    for regime, mask_fn in REGIMES.items():
        mask = mask_fn(y)
        boot = block_bootstrap_rmse(y, {m: part[m] for m in models}, mask)
        col = {m: j for j, m in enumerate(models)}
        res = {"n_hours": int(mask.sum()), "rmse": {}, "differences": {}}
        for m in models:
            res["rmse"][m] = {"value": round(rmse(y[mask], part[m].to_numpy()[mask]), 3), "ci95": interval(boot[:, col[m]])}
        for family in ["xgb", "gru"]:
            diff = boot[:, col[f"{family} candidate"]] - boot[:, col[f"{family} current"]]
            low, high = interval(diff)
            res["differences"][f"{family} candidate minus current"] = {
                "value": round(res["rmse"][f"{family} candidate"]["value"] - res["rmse"][f"{family} current"]["value"], 3),
                "ci95": [low, high],
                "significant": bool(low > 0 or high < 0),
            }
        out["regimes"][regime] = res
    out["verdict"] = {family: verdict(out["regimes"], family) for family in ["xgb", "gru"]}

    print(f"{args.split}: {len(part):,} hours, {out['from']} to {out['to']}")
    for regime, res in out["regimes"].items():
        print(f"\n{regime} ({res['n_hours']:,} hours)")
        for m, r in res["rmse"].items():
            print(f"  {m:<16} {r['value']:7.3f}  [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
        for name, d in res["differences"].items():
            print(f"  {name:<32} {d['value']:+7.3f}  [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
                  f"  {'significant' if d['significant'] else 'not significant'}")
    print("\nverdict:", json.dumps(out["verdict"]))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
