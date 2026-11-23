"""Read the deployed live forecast and say whether a storm is in progress.

Usage: python3 agents/bin/live_status.py [--json]
Needs only the standard library, so it runs without a worktree.

storm_in_progress is true when observed or forecast Dst (GRU or XGBoost) is at or below
-50 nT in the last 12 hours of the page, the block used by the retrain policy.
Exit code 0 always; read the output.
"""
import argparse
import json
import urllib.request
from datetime import datetime, timezone

URL = "https://makio-san.github.io/oculstorm/live.js"
STORM = -50.0
WINDOW = 12


def fetch() -> dict:
    request = urllib.request.Request(URL, headers={"User-Agent": "oculstorm-agents", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode()
    return json.loads(text[text.index("{") : text.rindex("}") + 1])


def last(values: list):
    return next((v for v in reversed(values) if v is not None), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    live = fetch()
    generated = datetime.strptime(live["generated"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - generated).total_seconds() / 3600
    recent = {k: [v for v in live[k][-WINDOW:] if v is not None] for k in ["observed", "gru", "xgb"]}
    lows = {k: (min(v) if v else None) for k, v in recent.items()}
    out = {
        "generated": live["generated"],
        "age_hours": round(age_hours, 1),
        "stale": age_hours > 3,
        "target": live["target"],
        "forecast": {"gru": last(live["gru"]), "xgb": last(live["xgb"]), "persistence": last(live["persistence"])},
        "latest_observed": last(live["observed"]),
        "lowest_last_12h": lows,
        "storm_in_progress": any(v is not None and v <= STORM for v in lows.values()),
        "solar_wind_coverage": live.get("solar_wind_coverage"),
        "inputs": live.get("inputs"),
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"page generated {out['generated']} ({out['age_hours']} h ago{', STALE' if out['stale'] else ''})")
    print(f"forecast for {out['target']}: {out['forecast']}, latest observed {out['latest_observed']} nT")
    print(f"lowest in last {WINDOW} h: {lows}, storm in progress: {out['storm_in_progress']}")


if __name__ == "__main__":
    main()
