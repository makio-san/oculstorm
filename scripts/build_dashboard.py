"""Export the held out test predictions as docs/data.js for the static dashboard.

Usage: python scripts/build_dashboard.py   (after train_xgb.py and train_gru.py)
Reads : reports/gru_test_predictions.parquet
Writes: docs/data.js  (loaded by docs/index.html through a script tag, so the
        dashboard also works when opened straight from disk)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oculstorm.features import HORIZON  # noqa: E402
from oculstorm.metrics import evaluate  # noqa: E402

SOURCE = ROOT / "reports" / "gru_test_predictions.parquet"
OUT = ROOT / "docs" / "data.js"

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


def main() -> None:
    frame = pd.read_parquet(SOURCE)
    # Rows are indexed by the hour the forecast was issued; shift to the hour it is for.
    frame.index = frame.index + pd.Timedelta(hours=HORIZON)
    if not (frame.index.to_series().diff().dropna() == pd.Timedelta("1h")).all():
        raise ValueError("test predictions are not a regular hourly series")

    observed = frame["dst_observed"]
    scores = evaluate(observed, {key: frame[col].to_numpy() for key, col in MODELS.items()})

    payload = {
        "start": frame.index[0].strftime("%Y-%m-%dT%H:%M"),
        "hours": len(frame),
        "horizon_hours": HORIZON,
        "observed": [int(v) for v in observed],
        **{key: [round(float(v), 1) for v in frame[col]] for key, col in MODELS.items()},
        "scores": json.loads(scores.round(4).to_json(orient="records")),
        "storms": find_storms(observed),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.OCULSTORM = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"{len(frame):,} hours, {frame.index[0]} .. {frame.index[-1]} -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    for storm in payload["storms"]:
        print(f"  {storm['time']}  {storm['min_dst']} nT")


if __name__ == "__main__":
    main()
