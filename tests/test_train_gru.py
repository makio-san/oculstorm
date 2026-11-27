"""The refit backtest options of scripts/train_gru.py (issue 44) never read the years they score or the test years."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from oculstorm.data import TEST_START, VAL_START
from oculstorm.sequences import build_windows, channel_stats

# One thread: torch and xgboost each bring an OpenMP runtime on macOS, and a multi-threaded
# torch step after an xgboost fit in the same pytest process can deadlock.
torch.set_num_threads(1)

SPEC = importlib.util.spec_from_file_location("train_gru", Path(__file__).resolve().parent.parent / "scripts" / "train_gru.py")
train_gru = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_gru)


def _table(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """A synthetic hourly table with the columns the pipeline reads, positive plasma values."""
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    rng = np.random.default_rng(seed)
    n = len(hours)
    df = pd.DataFrame(index=hours)
    df["bz_gsm"] = rng.normal(0, 4, n)
    df["by_gsm"] = rng.normal(0, 4, n)
    df["b_mag_avg"] = 6 + np.abs(rng.normal(0, 2, n))
    df["flow_speed"] = 420 + rng.normal(0, 60, n)
    df["proton_density"] = np.exp(rng.normal(1.5, 0.5, n))
    df["proton_temp"] = np.exp(rng.normal(11, 0.6, n))
    df["flow_pressure"] = 2e-6 * df["proton_density"] * df["flow_speed"] ** 2
    df["electric_field"] = -df["flow_speed"] * df["bz_gsm"] * 1e-3
    df["dst"] = -15 + 20 * np.sin(np.arange(n) / 40) + rng.normal(0, 2, n)
    df["f107"] = 100.0
    return df.astype("float32")


def test_backtest_parts_split_by_year_and_never_reach_the_test_years():
    df = _table("2015-10-01", "2021-02-01")
    windows, parts = train_gru.backtest_parts(df, shift=1)
    assert parts["train"].index.max() < pd.Timestamp("2016-01-01") <= parts["val"].index.min()
    assert parts["val"].index.max() < pd.Timestamp(VAL_START) <= parts["score"].index.min()
    assert parts["score"].index.max() < pd.Timestamp(TEST_START)
    assert all(p["row"].max() < len(windows) for p in parts.values())
    # later hours (early stopping, scored and test years) change neither scaling nor fit windows
    changed = df.copy()
    changed.loc[changed.index >= "2016-01-01", ["dst", "flow_speed"]] += 50.0
    windows_c, parts_c = train_gru.backtest_parts(changed, shift=1)
    rows = parts["train"]["row"].to_numpy()
    np.testing.assert_array_equal(windows[rows[:-1]], windows_c[rows[:-1]])
    assert channel_stats(df, end="2016-01-01") == channel_stats(changed, end="2016-01-01")


def test_validation_only_parts_are_the_production_split_without_test_rows():
    df = _table("2016-10-01", "2021-02-01")
    windows, parts = train_gru.backtest_parts(df)
    stats = channel_stats(df)
    full, meta = build_windows(df, stats)
    assert parts["val"].equals(parts["score"])
    assert parts["score"].index.max() < pd.Timestamp(TEST_START)
    rows = parts["score"]["row"].to_numpy()
    full_rows = meta.loc[parts["score"].index, "row"].to_numpy()
    np.testing.assert_array_equal(windows[rows], full[full_rows])


def _small_parts():
    df = _table("2016-11-01", "2017-02-01")
    windows, meta = build_windows(df, channel_stats(df))
    parts = {"train": meta[meta.index < VAL_START], "val": meta[meta.index >= VAL_START]}
    scale = float((parts["train"]["dst_target"] - parts["train"]["dst_lag0"]).std())
    return windows, parts, scale


def test_early_stopped_run_records_best_epoch_and_learning_rate_drops(monkeypatch):
    monkeypatch.setattr(train_gru, "MAX_EPOCHS", 4)
    windows, parts, scale = _small_parts()
    _, info = train_gru.train_one(0, windows, parts, scale)
    assert 1 <= info["best_epoch"] <= info["epochs"] <= 4
    assert all(1 <= e <= info["epochs"] for e in info["lr_drops"])


def test_fixed_epochs_ignore_validation_and_replay_the_schedule():
    windows, parts, scale = _small_parts()
    model, info = train_gru.train_one(0, windows, {"train": parts["train"]}, scale, fixed_epochs=2, lr_drops=(1,))
    assert info["epochs"] == 2 and info["lr_drops"] == [1] and "best_epoch" not in info
    garbage = parts["val"].assign(dst_target=500.0)
    same, _ = train_gru.train_one(0, windows, {"train": parts["train"], "val": garbage}, scale, fixed_epochs=2, lr_drops=(1,))
    for a, b in zip(model.state_dict().values(), same.state_dict().values()):
        assert torch.equal(a, b)
    other, _ = train_gru.train_one(0, windows, {"train": parts["train"]}, scale, fixed_epochs=2)
    assert any(not torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))
