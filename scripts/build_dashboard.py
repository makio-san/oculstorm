"""Export the held out test predictions for the static dashboard.

Usage: python scripts/build_dashboard.py   (after train_xgb.py and train_gru.py for each horizon)
Reads : reports/gru_test_predictions{_Hh}.parquet
Writes: docs/data.js (1 hour forecasts, plus the scores of every horizon) and
        docs/data_{H}h.js for the longer horizons, loaded only when selected.
        Plain script files, so the dashboard also works when opened straight from disk.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oculstorm.features import HORIZONS, suffix  # noqa: E402
from oculstorm.metrics import evaluate  # noqa: E402

DOCS = ROOT / "docs"

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


def write(path: Path, variable: str, payload: dict) -> None:
    path.write_text(f"{variable} = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"{payload['hours']:,} hours from {payload['start']} -> {path.name} ({path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
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
