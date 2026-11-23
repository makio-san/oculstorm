"""Score the append-only live forecast log, with skill against persistence.

Usage (from inside an agent worktree):
  /Users/esma/Projects/oculstorm/.venv/bin/python agents/bin/live_score.py [--days N] [--json]

Reads forecasts.csv from origin/live-archive (run `git fetch origin live-archive` first).
Skill is the MSE skill score: 1 - (RMSE_model / RMSE_persistence)^2.
"""
import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from oculstorm import archive  # noqa: E402

STORM = archive.STORM


def load_log() -> pd.DataFrame:
    text = subprocess.run(["git", "-C", str(ROOT), "show", "origin/live-archive:forecasts.csv"],
                          check=True, capture_output=True, text=True).stdout
    tmp = io.StringIO(text)
    return pd.read_csv(tmp, parse_dates=["target"], index_col="target", dtype={"issued": str})


def score(rows: pd.DataFrame) -> dict:
    rows = rows.dropna(subset=["observed", *archive.MODELS])
    if rows.empty:
        return {"hours": 0}
    truth = rows["observed"].to_numpy(dtype=float)
    rmse = {m: float(((rows[m].to_numpy(dtype=float) - truth) ** 2).mean() ** 0.5) for m in archive.MODELS}
    p = rmse["persistence"]
    skill = {m: (None if p == 0 else round(1 - (rmse[m] / p) ** 2, 3)) for m in ["gru", "xgb"]}
    return {"hours": len(rows), "rmse": {m: round(v, 2) for m, v in rmse.items()}, "skill": skill}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, help="only the last N days of target hours")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args()

    log = load_log()
    if args.days:
        log = log[log.index >= log.index.max() - pd.Timedelta(days=args.days)]
    expected = pd.date_range(log.index.min(), log.index.max(), freq="h")
    out = {
        "since": log.index.min().strftime("%Y-%m-%dT%H:%M"),
        "until": log.index.max().strftime("%Y-%m-%dT%H:%M"),
        "issued": len(log),
        "missing_hours": int(len(expected) - len(log)),
        "deepest_observed": None if log["observed"].dropna().empty else float(log["observed"].min()),
        "all": score(log),
        "storm": score(log[log["observed"] <= STORM]),
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"log {out['since']} to {out['until']}: {out['issued']} forecasts, {out['missing_hours']} hours missing, "
          f"deepest observed Dst {out['deepest_observed']} nT")
    for name in ["all", "storm"]:
        s = out[name]
        if s["hours"]:
            print(f"{name}: {s['hours']} scored hours, RMSE {s['rmse']}, skill {s['skill']}")
        else:
            print(f"{name}: no scored hours")


if __name__ == "__main__":
    main()
