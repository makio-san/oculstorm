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


def backtest_start(shift: int) -> pd.Timestamp:
    """Start of the early stopping window of a backtest shifted `shift` years back (issue 44).

    The fit ends here and early stopping uses [start, VAL_START), so the validation years stay
    the scored years and nothing of them is seen while fitting or stopping.
    """
    if shift < 1:
        raise ValueError("the backtest shift is at least one year")
    start = pd.Timestamp(VAL_START) - pd.DateOffset(years=shift)
    if start <= pd.Timestamp(START):
        raise ValueError(f"a {shift} year shift leaves no fit years after {START}")
    return start


def backtest_split(frame: pd.DataFrame, shift: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into fit / early stopping / scored years for a backtest shifted `shift` years back.

    With shift 4: fit 1995 to 2012, early stopping 2013 to 2016, scored 2017 to 2020 (the
    validation years). The test years are never returned.
    """
    start = backtest_start(shift)
    train, val, _ = split(frame)
    return train[train.index < start], train[train.index >= start], val
