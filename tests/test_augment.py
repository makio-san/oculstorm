"""The real-time like solar wind perturbation never looks ahead and is off by default."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oculstorm import augment
from oculstorm.augment import BLOCK, perturb_solar_wind, train_copies
from oculstorm.data import TEST_START, VAL_START, split
from oculstorm.features import SOLAR_WIND, build_features
from oculstorm.sequences import WINDOW, build_windows, channel_stats, concat_windows

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(start: str = "2016-10-01", end: str = "2017-03-01", seed: int = 0) -> pd.DataFrame:
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


def test_the_perturbed_block_is_the_solar_wind_block():
    assert sorted(BLOCK) == sorted(SOLAR_WIND)


def test_level_zero_changes_nothing_and_other_columns_are_never_touched():
    df = _table()
    pd.testing.assert_frame_equal(perturb_solar_wind(df, level=0.0, seed=3), df)
    out = perturb_solar_wind(df, level=1.0, seed=3)
    pd.testing.assert_frame_equal(out[["dst", "f107"]], df[["dst", "f107"]])
    assert out.index.equals(df.index) and out.dtypes.equals(df.dtypes)
    assert not out[BLOCK].equals(df[BLOCK])
    with pytest.raises(ValueError):
        perturb_solar_wind(df, level=-1.0)


def test_the_time_shift_only_delays_and_never_advances(monkeypatch):
    """Without noise and gaps every value is the clean one of hour t or of hour t-1, never t+1."""
    for name in ("SIGMA_LOG_PLASMA", "SIGMA_SPEED", "SIGMA_FIELD", "DROPOUT_FRACTION"):
        monkeypatch.setattr(augment, name, 0.0)
    df = _table()
    ramp = np.arange(len(df), dtype="float32")  # the value is the hour number, so a shift is visible
    for name in ("bz_gsm", "by_gsm", "b_mag_avg", "flow_speed", "proton_density", "proton_temp"):
        df[name] = ramp + 1
    out = perturb_solar_wind(df, level=1.0, seed=5)
    for name in ("bz_gsm", "by_gsm", "b_mag_avg", "flow_speed", "proton_density", "proton_temp"):
        lag = (df[name] - out[name]).to_numpy()[1:]  # hours the shown value is old
        assert set(np.unique(lag)) == {0.0, 1.0}, name
    # One delay per 24 hour segment, the same for the whole block.
    lag = (df["bz_gsm"] - out["bz_gsm"]).to_numpy()
    days = lag[: len(lag) // 24 * 24].reshape(-1, 24)[1:]
    assert (days == days[:, :1]).all()
    assert 0.2 < days[:, 0].mean() < 0.8
    assert np.array_equal(lag[1:], (df["flow_speed"] - out["flow_speed"]).to_numpy()[1:])


@pytest.mark.parametrize("level", [0.5, 1.0])
def test_the_perturbed_past_does_not_depend_on_the_future(level):
    df = _table()
    cut = df.index[len(df) // 2]
    out = perturb_solar_wind(df, level=level, seed=7)

    # Same table cut at `cut`: the hours up to it come out identical, bit for bit.
    short = perturb_solar_wind(df[df.index <= cut], level=level, seed=7)
    pd.testing.assert_frame_equal(short, out[out.index <= cut])

    # Same hours with a corrupted future: nothing at or before `cut` moves, the future does.
    corrupted = df.copy()
    corrupted.loc[corrupted.index > cut, BLOCK] = corrupted.loc[corrupted.index > cut, BLOCK] * 3 + 50
    out_corrupted = perturb_solar_wind(corrupted, level=level, seed=7)
    pd.testing.assert_frame_equal(out_corrupted[out.index <= cut], out[out.index <= cut])
    assert not out_corrupted[out.index > cut][BLOCK].equals(out[out.index > cut][BLOCK])

    # And so do the features and the GRU windows built from the perturbed table.
    X, _ = build_features(out)
    X_corrupted, _ = build_features(out_corrupted)
    past = X.index[X.index <= cut - pd.Timedelta(hours=1)]
    pd.testing.assert_frame_equal(X.loc[past], X_corrupted.loc[past])
    stats = channel_stats(df)
    windows, meta = build_windows(out, stats)
    windows_corrupted, _ = build_windows(out_corrupted, stats)
    rows = meta.loc[meta.index <= cut - pd.Timedelta(hours=1), "row"].to_numpy()
    assert np.array_equal(windows[rows], windows_corrupted[rows])


def test_noise_widths_gaps_and_recomputed_columns():
    df = _table("2010-01-01", "2012-01-01")
    out = perturb_solar_wind(df, level=1.0, seed=1)
    seen = out["flow_speed"].notna().to_numpy()
    gap_share = 1 - seen.mean()
    assert 0.012 < gap_share < 0.03
    assert out.loc[~seen, BLOCK].isna().all().all()  # a gap removes the whole block
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], (~seen).astype(int), [0]]))))[::2]
    assert runs.min() >= 1 and np.mean(runs <= 3) > 0.95  # runs of 1 to 3 hours, rarely two that touch
    # The derived columns follow the perturbed values, not the clean ones.
    e = -out["flow_speed"] * out["bz_gsm"] * 1e-3
    assert np.allclose(out["electric_field"][seen], e[seen], rtol=1e-4, atol=1e-4)
    ratio = (out["flow_pressure"] / (2e-6 * out["proton_density"] * out["flow_speed"] ** 2))[seen]
    assert np.allclose(ratio, 1.0, rtol=1e-3)
    assert (out["b_mag_avg"][seen] >= 0).all() and (out["proton_density"][seen] > 0).all()
    # Half the level, half the noise: compare the field noise on the hours that were not delayed.
    half = perturb_solar_wind(df, level=0.5, seed=1)
    spread = lambda frame: np.nanmedian(np.abs(frame["b_mag_avg"] - df["b_mag_avg"]))  # noqa: E731
    assert spread(half) < spread(out)


def test_train_copies_hold_training_hours_only_and_are_off_by_default():
    df = _table("2016-10-01", "2021-03-01")
    assert train_copies(df, level=0.0) == []
    assert train_copies(df, level=1.0, copies=0) == []
    copies = train_copies(df, level=1.0, copies=2, seed=4)
    assert len(copies) == 2
    for copy in copies:
        assert copy.index.max() < pd.Timestamp(VAL_START)
        assert copy.index.equals(df.index[df.index < VAL_START])
    assert not copies[0][BLOCK].equals(copies[1][BLOCK])
    pd.testing.assert_frame_equal(copies[0], perturb_solar_wind(df[df.index < VAL_START], 1.0, 4))


def test_xgb_option_is_off_by_default_and_extra_rows_only_reach_train():
    train_xgb = _script("train_xgb")
    args = train_xgb.build_parser().parse_args([])
    assert args.sw_noise_aug == 0.0 and args.sw_noise_copies == 1

    df = _table("2016-10-01", "2021-03-01")
    X, y = build_features(df)
    params = {**train_xgb.PARAMS, "n_estimators": 8, "early_stopping_rounds": 3}
    _, val_plain, _ = train_xgb.fit(X, y, params=params, test=False)
    _, val_none, _ = train_xgb.fit(X, y, params=params, test=False, extra=[])
    assert np.array_equal(val_plain, val_none)

    # Extra rows dated after the training years are ignored, so they cannot leak into the fit.
    late = (X[X.index >= VAL_START] * 0 + 999, y[y.index >= VAL_START] * 0 - 999)
    _, val_late, _ = train_xgb.fit(X, y, params=params, test=False, extra=[late])
    assert np.array_equal(val_plain, val_late)

    extra = [build_features(copy) for copy in train_copies(df, level=1.0, seed=2)]
    _, val_aug, test_pred = train_xgb.fit(X, y, params=params, test=False, extra=extra)
    assert test_pred is None and len(val_aug) == len(val_plain)
    assert not np.array_equal(val_plain, val_aug)


def test_gru_train_copies_join_without_touching_validation_or_test():
    train_gru = _script("train_gru")
    df = _table("2016-10-01", "2021-03-01")
    stats = channel_stats(df)
    windows, meta = build_windows(df)
    same, parts = train_gru.add_train_copies(windows, meta, [], stats)
    assert same is windows
    assert [len(p) for p in parts.values()] == [len(p) for p in split(meta)]

    copies = train_copies(df, level=1.0, copies=2, seed=3)
    joined, aug = train_gru.add_train_copies(windows, meta, copies, stats)
    n_train = len(parts["train"])
    assert len(aug["train"]) > 2.9 * n_train
    assert aug["train"].index.max() < pd.Timestamp(VAL_START)
    assert aug["train"]["row"].is_unique
    # Validation and test samples are the clean ones, value for value.
    for name in ("val", "test"):
        assert aug[name].index.equals(parts[name].index)
        assert np.array_equal(joined[aug[name]["row"].to_numpy()], windows[parts[name]["row"].to_numpy()])
    assert aug["test"].index.min() >= pd.Timestamp(TEST_START)
    # The clean training samples come first, then each copy with its own perturbed windows.
    assert np.array_equal(joined[aug["train"]["row"].to_numpy()[:n_train]], windows[parts["train"]["row"].to_numpy()])
    copy_windows, copy_meta = build_windows(copies[0], stats)
    first = aug["train"].iloc[n_train : n_train + len(copy_meta)]
    assert first.index.equals(copy_meta.index)
    assert np.array_equal(joined[first["row"].to_numpy()], copy_windows[copy_meta["row"].to_numpy()])
    assert not np.array_equal(joined[first["row"].to_numpy()[:200]], windows[parts["train"]["row"].to_numpy()[:200]])


def test_concat_windows_keeps_every_sample():
    a, b = _table("2016-10-01", "2016-11-01"), _table("2016-10-01", "2016-10-20", seed=1)
    stats = channel_stats(a)
    pieces = [build_windows(a, stats), build_windows(b, stats)]
    joined, meta = concat_windows(pieces)
    assert joined.shape[1:] == (WINDOW, pieces[0][0].shape[2])
    start = 0
    for windows, piece in pieces:
        rows = meta["row"].to_numpy()[start : start + len(piece)]
        assert np.array_equal(joined[rows], windows[piece["row"].to_numpy()])
        start += len(piece)
