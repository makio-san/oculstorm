"""Train the 1 hour XGBoost on archived real-time ACE solar wind and score it on validation (issue 50).

Usage: OMP_NUM_THREADS=3 python -m oculstorm.realtime_training [--seeds 0 1 2] [--out reports/experiments/50-xgb-trained-on-realtime-ace.json] [--keep]
Reads : data/kaggle/omni2_hourly.parquet, models/xgb_dst_1h.ubj, and the SWPC monthly ACE files
        for 2001 to 2020 (downloaded into data/ace_rt, deleted at the end unless --keep)
Writes: the JSON report, nothing else (no model file)

Validation only. Candidates are fitted with scripts/train_xgb.py `fit`: variant A (ace_rt) on the
2001 to 2016 rows with real-time ACE solar wind delayed 1 hour, variant B (omni+ace_rt) on the OMNI
train years plus that copy, both early stopped on validation with OMNI inputs as production.
A retrained OMNI model per seed shows the seed noise. Every model is scored on the validation
years with OMNI inputs and with the real-time ACE inputs of issue 31, against the production
model, by regime, with paired block bootstrap intervals. The test years are never predicted.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import realtime_inputs as rt
from .data import ROOT, VAL_START, load_hourly
from .features import HORIZON, build_features

OUT = ROOT / "reports" / "experiments" / "50-xgb-trained-on-realtime-ace.json"
DELAY = rt.DELAYS[0]
VARIANTS = ("omni", "ace_rt", "omni+ace_rt")


def predictor(model):
    """A function that forecasts Dst from a table with `model`, as predict_xgb does in the live forecast."""
    def predict(table: pd.DataFrame) -> pd.Series:
        X, _ = build_features(table, require_target=False)
        X = X[model.get_booster().feature_names]
        return pd.Series(X["dst_lag0"].to_numpy() + model.predict(X), index=X.index)
    return predict


def candidate_frames(tables: dict[str, pd.DataFrame], predictors: dict) -> dict[str, pd.DataFrame]:
    """Per input table: observed, persistence and one column per model, on the issue hours."""
    out = {}
    for inputs, table in tables.items():
        frame = pd.DataFrame({"observed": table["dst"].shift(-HORIZON), "persistence": table["dst"]})
        for name, predict in predictors.items():
            frame[name] = predict(table).reindex(frame.index)
        out[inputs] = frame[rt.issue_mask(frame.index)].dropna()
    return out


def summary(scores: dict, reference: str) -> dict:
    """Short table: MSE change of every model against `reference`, per regime."""
    out = {}
    for regime, r in scores.items():
        if not r["n_hours"]:
            continue
        out[regime] = {"n_hours": r["n_hours"], f"{reference}_rmse": r["inputs"][reference]["rmse"]}
        for key, d in r["differences"].items():
            out[regime][key.split(" minus ")[0]] = {"mse_change_percent": d["mse_change_percent"],
                                                    "ci95": d["mse_change_percent_ci95"], "significant": d["significant"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=rt.CACHE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--keep", action="store_true", help="do not delete the downloaded files at the end")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    import train_xgb
    from live_forecast import predict_xgb

    started = time.time()
    try:
        files = rt.download(args.cache, "train") + rt.download(args.cache, "validation")
        megabytes = sum(p.stat().st_size for p in files) / 1e6
        ace_train = rt.load_ace(args.cache, "train")
        ace_val = rt.load_ace(args.cache, "validation")
    finally:
        if not args.keep:
            shutil.rmtree(args.cache, ignore_errors=True)
    print(f"{len(files)} files, {megabytes:.1f} MB", flush=True)

    df = load_hourly()
    X, y = build_features(df)
    X_ace, y_ace = build_features(rt.train_table(df, ace_train, DELAY))
    X_ace, y_ace = train_xgb.split(X_ace)[0], train_xgb.split(y_ace)[0]
    rows = {"omni": int(len(train_xgb.split(X)[0])), "ace_rt": int(len(X_ace))}
    rows["omni+ace_rt"] = rows["omni"] + rows["ace_rt"]

    predictors, fits = {"production": predict_xgb}, {}
    for variant in VARIANTS:
        for seed in args.seeds:
            name = f"{variant} seed{seed}"
            t0 = time.time()
            params = {**train_xgb.PARAMS, "random_state": seed}
            kwargs = {"train": (X_ace, y_ace)} if variant == "ace_rt" else {"extra": [(X_ace, y_ace)]} if variant == "omni+ace_rt" else {}
            model, _, _ = train_xgb.fit(X, y, params=params, test=False, **kwargs)
            fits[name] = {"trees": int(model.best_iteration + 1), "seconds": round(time.time() - t0)}
            predictors[name] = predictor(model)
            print(f"{name}: {fits[name]}", flush=True)

    omni_val = rt.validation_table(df)
    tables = {"omni": omni_val, f"ace_delay{DELAY}h": rt.with_solar_wind(omni_val, rt.delayed(ace_val, DELAY))}
    frames = candidate_frames(tables, predictors)
    hours = frames["omni"].index.intersection(frames[f"ace_delay{DELAY}h"].index)
    frames = {k: v.loc[hours] for k, v in frames.items()}
    complete = (rt.inputs_observed(tables["omni"]) & rt.inputs_observed(tables[f"ace_delay{DELAY}h"])).reindex(hours).to_numpy()
    for k, v in frames.items():  # the seed mean of each variant, a 3 member ensemble
        for variant in VARIANTS:
            v[f"{variant} seed mean"] = v[[f"{variant} seed{s}" for s in args.seeds]].mean(axis=1)

    names = [c for c in frames["omni"].columns if c not in ("observed", "persistence", "production")]
    scores = {subset: {inputs: rt.score(f, "production", names, None if subset == "all_hours" else complete)
                       for inputs, f in frames.items()}
              for subset in ("all_hours", "inputs_observed")}
    stamp = lambda t: t.strftime("%Y-%m-%dT%H:%M")  # noqa: E731
    report = {
        "issue": 50,
        "title": "1 hour XGBoost trained on archived real-time ACE solar wind for 2001 to 2016, scored on validation",
        "split": f"validation only, {VAL_START} .. 2020-12-31; train rows end before {VAL_START}; the test years are never predicted",
        "source": {"url": rt.BASE_URL, "files": len(files), "megabytes": round(megabytes, 2), "kept": bool(args.keep),
                   "train_months": f"{rt.year_months('train')[0]} .. {rt.year_months('train')[-1]}",
                   "validation_months": f"{rt.year_months()[0]} .. {rt.year_months()[-1]}"},
        "configuration": {
            "reference": "production: models/xgb_dst_1h.ubj loaded by scripts/live_forecast.py predict_xgb",
            "variants": {
                "omni": "scripts/train_xgb.py defaults retrained with each seed (seed 0 is the production configuration): seed noise",
                "ace_rt": "train rows 2001-01-01 .. 2016-12-31 with real-time ACE solar wind delayed 1 hour, Dst and F10.7 from OMNI; train_xgb.py --train-inputs ace_rt",
                "omni+ace_rt": "OMNI train rows 1995 .. 2016 plus the ACE copy (K = 1); train_xgb.py --train-inputs omni+ace_rt",
                "seed mean": "mean forecast of the seeds of one variant",
            },
            "train_rows": rows,
            "early_stopping": "validation years with OMNI inputs, as production",
            "seeds": args.seeds,
            "params": {k: v for k, v in train_xgb.PARAMS.items()},
            "validation_inputs": {"omni": "parquet", f"ace_delay{DELAY}h": "issue 31 real-time ACE inputs delayed 1 hour"},
            "scored_hours": f"{len(hours):,} issue hours from {stamp(hours.min())} to {stamp(hours.max())}, the same for every model and input",
        },
        "bootstrap": {"method": "oculstorm.metrics.block_bootstrap_rmse, paired", "block_hours": 168, "resamples": 2000, "seed": 0,
                      "intervals": "95%; differences are model minus production on the same inputs"},
        "fits": fits,
        "train_years_input_statistics": rt.input_statistics(df[(df.index >= rt.TRAIN_ACE_START) & (df.index < VAL_START)], rt.delayed(ace_train, DELAY)),
        "train_years_coverage": rt.coverage(ace_train),
        "issue_hours": {"all": len(hours), "all_inputs_observed_last_6h_in_both_inputs": int(complete.sum())},
        "summary": {subset: {inputs: summary(s, "production") for inputs, s in by_inputs.items()} for subset, by_inputs in scores.items()},
        "scores": scores,
    }
    report["seconds"] = round(time.time() - started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")

    for subset, by_inputs in report["summary"].items():
        for inputs, regimes in by_inputs.items():
            print(f"\n{subset} / {inputs} inputs")
            for regime, r in regimes.items():
                cells = "  ".join(f"{m} {v['mse_change_percent']:+.1f}% {v['ci95']}" for m, v in r.items() if isinstance(v, dict) and "seed mean" in m)
                print(f"  {regime} ({r['n_hours']:,} h) production {r['production_rmse']:.3f}: {cells}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
