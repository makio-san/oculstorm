"""Loading the OMNI2 hourly table and splitting it chronologically."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "kaggle" / "omni2_hourly.parquet"

# Solar wind coverage is only 50-70% before 1995 (pre Wind/ACE), so earlier years are dropped.
START = "1995-01-01"

# Chronological split, never shuffled: a random split would leak neighbouring hours
# of the same storm into both train and test.
VAL_START = "2017-01-01"
TEST_START = "2021-01-01"  # solar cycle 25, includes the May 2024 superstorm


def load_hourly(start: str = START) -> pd.DataFrame:
    """Return the hourly table from `start` onwards, indexed by UTC time with no gaps."""
    df = pd.read_parquet(PARQUET)
    df = df[df["time"] >= start].set_index("time").sort_index()
    if not (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all():
        raise ValueError("time axis is not a regular 1 hour grid; lag features would be wrong")
    return df


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split any time-indexed frame into train / validation / test by date."""
    train = frame[frame.index < VAL_START]
    val = frame[(frame.index >= VAL_START) & (frame.index < TEST_START)]
    test = frame[frame.index >= TEST_START]
    return train, val, test
