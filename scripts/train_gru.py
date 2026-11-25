"""Train the small GRU and compare it with persistence and the XGBoost baseline.

Usage: python scripts/train_gru.py [--horizon H] [--storm-weight K] [--ring-current] [--driver-memory]
Writes: models/gru_dst_{H}h{_swK}{_rc}{_mem}_seed*.pt, models/gru_channels{_rc}{_mem}.json,
reports/gru_baseline{_Hh}{_swK}{_rc}{_mem}.json, reports/gru_test_predictions{_Hh}{_swK}{_rc}{_mem}.parquet
(the 1 hour reports keep their names without the suffix; _swK only when K is not 1;
--ring-current adds the Dst* and Burton step channels and the tag _rc to every file name;
_mem only with --driver-memory, which adds decaying means of VBs and Newell)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oculstorm.data import TEST_START, VAL_START, load_hourly, split  # noqa: E402
from oculstorm.features import HORIZON, HORIZONS, MEMORY_TAUS, suffix, weight_tag  # noqa: E402
from oculstorm.gru import DstGRU  # noqa: E402
from oculstorm.metrics import evaluate, rmse, storm_weights  # noqa: E402
from oculstorm.sequences import WINDOW, build_windows, channel_stats  # noqa: E402

SEEDS = [0, 1, 2]
HIDDEN = 64
BATCH = 512
LR = 2e-3
MAX_EPOCHS = 40
PATIENCE = 6


def predict(model: nn.Module, windows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), 4096):
            out.append(model(torch.from_numpy(windows[rows[start : start + 4096]])).numpy())
    return np.concatenate(out)


def train_one(seed: int, windows: np.ndarray, parts: dict[str, pd.DataFrame], scale: float, peak: float = 1.0) -> tuple[nn.Module, dict]:
    """Train one seed with early stopping on validation RMSE. Targets are delta / scale.

    With `peak` > 1 storm hours weigh more, in the loss and in the validation RMSE used for
    early stopping alike.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DstGRU(windows.shape[2], HIDDEN)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=2)
    loss_fn = nn.HuberLoss(delta=3.0, reduction="none")  # robust to the rare, very large storm-time jumps

    tr, va = parts["train"], parts["val"]
    tr_rows = tr["row"].to_numpy()
    tr_target = ((tr["dst_target"] - tr["dst_lag0"]) / scale).to_numpy("float32")
    tr_weight = storm_weights(tr["dst_target"], peak).astype("float32")
    va_delta = (va["dst_target"] - va["dst_lag0"]).to_numpy()
    va_weight = storm_weights(va["dst_target"], peak)

    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(tr_rows))
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            x = torch.from_numpy(windows[tr_rows[idx]])
            w = torch.from_numpy(tr_weight[idx])
            loss = (loss_fn(model(x), torch.from_numpy(tr_target[idx])) * w).sum() / w.sum()
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        val_error = va_delta - predict(model, windows, va["row"].to_numpy()) * scale
        val_rmse = float(np.sqrt(np.average(val_error**2, weights=va_weight)))
        scheduler.step(val_rmse)
        history.append(round(val_rmse, 4))
        if val_rmse < best - 1e-4:
            best, bad = val_rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {"seed": seed, "epochs": epoch, "best_val_rmse": round(best, 4), "val_rmse_history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, default=HORIZON, choices=HORIZONS, help="hours ahead")
    parser.add_argument("--storm-weight", type=float, default=1.0, help="loss weight at Dst <= -100 nT relative to quiet time")
    parser.add_argument("--ring-current", action="store_true", help="add the O'Brien and McPherron (2000) Dst* and Burton step channels")
    parser.add_argument("--driver-memory", action="store_true",
                        help=f"add decaying means of VBs and Newell with time constants {MEMORY_TAUS} h as channels")
    args = parser.parse_args()
    h, peak = args.horizon, args.storm_weight
    rc = "_rc" if args.ring_current else ""
    memory = MEMORY_TAUS if args.driver_memory else ()
    mem_tag = "_mem" if memory else ""
    tag = suffix(h) + weight_tag(peak) + rc + mem_tag
    torch.set_num_threads(4)
    df = load_hourly()
    windows, meta = build_windows(df, horizon=h, ring_current=args.ring_current, memory=memory)
    parts = dict(zip(["train", "val", "test"], split(meta)))
    scale = float((parts["train"]["dst_target"] - parts["train"]["dst_lag0"]).std())
    n_params = sum(p.numel() for p in DstGRU(windows.shape[2], HIDDEN).parameters())
    print(f"horizon: {h}h  storm weight: {peak:g}  channels: {windows.shape[2]}  window: {WINDOW}h  params: {n_params:,}  "
          + "  ".join(f"{k}: {len(v):,}" for k, v in parts.items()))

    te = parts["test"]
    te_rows, lag0 = te["row"].to_numpy(), te["dst_lag0"].to_numpy()
    va = parts["val"]
    va_rows, va_lag0 = va["row"].to_numpy(), va["dst_lag0"].to_numpy()
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    # Scaling statistics the live pipeline needs to prepare inputs without the training data.
    (ROOT / "models" / f"gru_channels{rc}{mem_tag}.json").write_text(json.dumps(channel_stats(df, args.ring_current, memory), indent=2) + "\n")

    runs, seed_preds, val_preds = [], {}, []
    for seed in SEEDS:
        started = time.time()
        model, info = train_one(seed, windows, parts, scale, peak)
        seed_preds[seed] = lag0 + predict(model, windows, te_rows) * scale
        val_preds.append(va_lag0 + predict(model, windows, va_rows) * scale)
        info["test_rmse"] = round(rmse(te["dst_target"].to_numpy(), seed_preds[seed]), 4)
        info["seconds"] = round(time.time() - started)
        runs.append(info)
        torch.save({"state_dict": model.state_dict(), "scale": scale, "hidden": HIDDEN, "window": WINDOW},
                   ROOT / "models" / f"gru_dst_{h}h{weight_tag(peak)}{rc}{mem_tag}_seed{seed}.pt")
        print({k: v for k, v in info.items() if k != "val_rmse_history"})

    predictions = {"persistence": lag0}
    xgb_path = ROOT / "reports" / f"xgb_test_predictions{tag}.parquet"
    if xgb_path.exists():
        xgb = pd.read_parquet(xgb_path)["xgboost (solar wind + Dst)"]
        if xgb.index.equals(te.index):
            predictions["xgboost (solar wind + Dst)"] = xgb.to_numpy()
    # Single-seed numbers are in `runs`; the headline GRU is the mean of the seeds.
    predictions["gru (single seed, median)"] = seed_preds[sorted(runs, key=lambda r: r["test_rmse"])[len(runs) // 2]["seed"]]
    predictions[f"gru (mean of {len(SEEDS)} seeds)"] = np.mean(list(seed_preds.values()), axis=0)
    if "xgboost (solar wind + Dst)" in predictions:
        predictions["xgboost + gru average"] = (
            predictions["xgboost (solar wind + Dst)"] + predictions[f"gru (mean of {len(SEEDS)} seeds)"]
        ) / 2

    scores = evaluate(te["dst_target"], predictions)
    pd.set_option("display.width", 200)
    print(scores.round(3).to_string(index=False))
    # Choices such as the storm weight are made on these validation scores, never on test.
    val_scores = evaluate(va["dst_target"], {"persistence": va_lag0, f"gru (mean of {len(SEEDS)} seeds)": np.mean(val_preds, axis=0)})
    print("\nValidation:")
    print(val_scores.round(3).to_string(index=False))

    pd.DataFrame({"dst_observed": te["dst_target"], **predictions}).to_parquet(ROOT / "reports" / f"gru_test_predictions{tag}.parquet")
    report = {
        "horizon_hours": h,
        "storm_weight": peak,
        "ring_current": args.ring_current,
        "driver_memory_taus": list(memory),
        "window_hours": WINDOW,
        "train": f"1995-01-01 .. {VAL_START} (exclusive)",
        "validation": f"{VAL_START} .. {TEST_START} (exclusive)",
        "test": f"{TEST_START} .. {te.index.max()}",
        "n_channels": int(windows.shape[2]),
        "n_parameters": int(n_params),
        "hyperparameters": {"hidden": HIDDEN, "batch": BATCH, "lr": LR, "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
        "runs": runs,
        "scores": json.loads(scores.round(4).to_json(orient="records")),
        "validation_scores": json.loads(val_scores.round(4).to_json(orient="records")),
    }
    (ROOT / "reports" / f"gru_baseline{tag}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
