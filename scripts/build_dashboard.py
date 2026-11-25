"""Export the held out test predictions for the static dashboard.

Usage: python scripts/build_dashboard.py   (after train_xgb.py and train_gru.py for each horizon)
       python scripts/build_dashboard.py --error-bands   (only the live error band; needs the parquet)
Reads : reports/gru_test_predictions{_Hh}.parquet
Writes: docs/data.js (1 hour forecasts, plus the scores of every horizon) and
        docs/data_{H}h.js for the longer horizons, loaded only when selected.
        Plain script files, so the dashboard also works when opened straight from disk.
        With --error-bands: docs/error_bands.json, the 1 hour GRU error quantiles per Dst
        regime on the validation years, read by scripts/live_forecast.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oculstorm.features import HORIZONS, suffix  # noqa: E402
from oculstorm.metrics import evaluate  # noqa: E402

DOCS = ROOT / "docs"
ERROR_BANDS = DOCS / "error_bands.json"
QUANTILES = (0.1, 0.9)
# Regimes of the Dst at the hour the forecast is issued (dst_lag0), quietest first.
# A Dst belongs to the first regime whose `above` it exceeds; the last one takes the rest.
REGIMES = [
    {"name": "quiet", "label": "Dst > -50 nT", "above": -50},
    {"name": "storm", "label": "-100 < Dst <= -50 nT", "above": -100},
    {"name": "intense", "label": "Dst <= -100 nT", "above": None},
]

# dashboard key -> column in the predictions file
MODELS = {
    "persistence": "persistence",
    "xgb": "xgboost (solar wind + Dst)",
    "gru": "gru (mean of 3 seeds)",
}
STORM_LEVEL = -100  # nT, an event needs at least one hour at or below this
EVENT_GAP = pd.Timedelta("48h")  # quieter gaps longer than this separate two events
N_EVENTS = 12


def find_storms(observed: pd.Series) -> list[dict]:
    """Deepest storms of the period: (time of minimum, minimum Dst), strongest first."""
    hours = observed[observed <= STORM_LEVEL]
    event_id = (hours.index.to_series().diff() > EVENT_GAP).cumsum()
    storms = [
        {"time": group.idxmin().strftime("%Y-%m-%dT%H:%M"), "min_dst": int(group.min())}
        for _, group in hours.groupby(event_id)
    ]
    return sorted(storms, key=lambda s: s["min_dst"])[:N_EVENTS]


def export(horizon: int) -> dict:
    frame = pd.read_parquet(ROOT / "reports" / f"gru_test_predictions{suffix(horizon)}.parquet")
    # Rows are indexed by the hour the forecast was issued; shift to the hour it is for.
    frame.index = frame.index + pd.Timedelta(hours=horizon)
    if not (frame.index.to_series().diff().dropna() == pd.Timedelta("1h")).all():
        raise ValueError("test predictions are not a regular hourly series")

    observed = frame["dst_observed"]
    scores = evaluate(observed, {key: frame[col].to_numpy() for key, col in MODELS.items()})
    return {
        "start": frame.index[0].strftime("%Y-%m-%dT%H:%M"),
        "hours": len(frame),
        "horizon_hours": horizon,
        "observed": [int(v) for v in observed],
        **{key: [round(float(v), 1) for v in frame[col]] for key, col in MODELS.items()},
        "scores": json.loads(scores.round(4).to_json(orient="records")),
        "storms": find_storms(observed),
    }


def regime_of(dst_lag0: float) -> str | None:
    """Name of the regime a current Dst falls in, None when it is missing."""
    if dst_lag0 is None or pd.isna(dst_lag0):
        return None
    return next(r["name"] for r in REGIMES if r["above"] is None or dst_lag0 > r["above"])


def error_quantiles(observed: pd.Series, forecast: pd.Series, dst_lag0: pd.Series) -> list[dict]:
    """Quantiles of observed minus forecast per regime of dst_lag0, with the hours behind them."""
    error = observed - forecast
    regime = dst_lag0.map(regime_of)
    out = []
    for spec in REGIMES:
        e = error[regime == spec["name"]].dropna().to_numpy()
        if not len(e):
            raise ValueError(f"no validation hours in the {spec['name']} regime")
        low, high = np.quantile(e, QUANTILES)
        out.append({**spec, "hours": int(len(e)), "q10": round(float(low), 2), "q90": round(float(high), 2)})
    return out


def export_error_bands() -> dict:
    """Score the committed 1 hour GRU (mean of seeds) on the validation years only.

    The table is cut at the start of the test years before anything is built, so no test hour
    is read; the last validation hour, whose target would fall in the test years, is dropped.
    """
    import torch

    from oculstorm.data import TEST_START, VAL_START, load_hourly
    from oculstorm.gru import DstGRU
    from oculstorm.sequences import build_windows

    models = ROOT / "models"
    df = load_hourly()
    df = df[df.index < TEST_START]
    stats = json.loads((models / "gru_channels.json").read_text())  # the same scaling as live
    windows, meta = build_windows(df, stats)
    meta = meta[meta.index >= VAL_START]
    rows = meta["row"].to_numpy()

    torch.set_num_threads(3)
    deltas = []
    for path in sorted(models.glob("gru_dst_1h_seed*.pt")):
        checkpoint = torch.load(path, weights_only=True)
        model = DstGRU(windows.shape[2], checkpoint["hidden"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(rows), 4096):
                x = torch.from_numpy(np.ascontiguousarray(windows[rows[start : start + 4096]]))
                out.append(model(x).numpy() * checkpoint["scale"])
        deltas.append(np.concatenate(out))
    forecast = pd.Series(meta["dst_lag0"].to_numpy() + np.mean(deltas, axis=0), index=meta.index)

    return {
        "model": f"gru (mean of {len(deltas)} seeds), 1 hour",
        "period": f"{meta.index.min():%Y-%m-%d} .. {meta.index.max():%Y-%m-%d}",
        "error": "observed minus forecast, nT",
        "quantiles": list(QUANTILES),
        "regimes": error_quantiles(meta["dst_target"], forecast, meta["dst_lag0"]),
    }


def write(path: Path, variable: str, payload: dict) -> None:
    path.write_text(f"{variable} = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"{payload['hours']:,} hours from {payload['start']} -> {path.name} ({path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--error-bands", action="store_true", help="only write docs/error_bands.json from the validation years")
    args = parser.parse_args()
    DOCS.mkdir(exist_ok=True)
    if args.error_bands:
        bands = export_error_bands()
        ERROR_BANDS.write_text(json.dumps(bands, indent=2) + "\n")
        print(f"{bands['model']} errors over {bands['period']} -> {ERROR_BANDS.name}")
        for r in bands["regimes"]:
            print(f"  {r['name']:8s} {r['label']:22s} {r['hours']:6,} hours  q10 {r['q10']:+.2f}  q90 {r['q90']:+.2f}")
        return
    available = [h for h in HORIZONS if (ROOT / "reports" / f"gru_test_predictions{suffix(h)}.parquet").exists()]
    payloads = {h: export(h) for h in available}

    # Scores of every horizon go in the main file, so the lead time chart needs no extra download.
    main_payload = {**payloads[1], "horizons": {str(h): payloads[h]["scores"] for h in available}}
    write(DOCS / "data.js", "window.OCULSTORM", main_payload)
    for h in available[1:]:
        write(DOCS / f"data_{h}h.js", f"(window.OCULSTORM_H = window.OCULSTORM_H || {{}})[{h}]", payloads[h])
    for storm in main_payload["storms"]:
        print(f"  {storm['time']}  {storm['min_dst']} nT")


if __name__ == "__main__":
    main()
