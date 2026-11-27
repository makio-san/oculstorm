"""The augmentations of oculstorm.augment never reach validation, never look ahead and are off by default."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oculstorm import augment
from oculstorm.augment import BLOCK, OFFSET_RANGE, SW_COLUMNS, block_offsets, consistent_rows, offset_table, perturb_solar_wind, train_copies
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
    assert sorted(SW_COLUMNS) == sorted(SOLAR_WIND)


def test_level_zero_changes_nothing_and_other_columns_are_never_touched():
    df = _table()
    pd.testing.assert_frame_equal(perturb_solar_wind(df, level=0.0, seed=3), df)
    out = perturb_solar_wind(df, level=1.0, seed=3)
    pd.testing.assert_frame_equal(out[["dst", "f107"]], df[["dst", "f107"]])
    assert out.index.equals(df.index) and out.dtypes.equals(df.dtypes)
    assert not out[SW_COLUMNS].equals(df[SW_COLUMNS])
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
    corrupted.loc[corrupted.index > cut, SW_COLUMNS] = corrupted.loc[corrupted.index > cut, SW_COLUMNS] * 3 + 50
    out_corrupted = perturb_solar_wind(corrupted, level=level, seed=7)
    pd.testing.assert_frame_equal(out_corrupted[out.index <= cut], out[out.index <= cut])
    assert not out_corrupted[out.index > cut][SW_COLUMNS].equals(out[out.index > cut][SW_COLUMNS])

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
    assert out.loc[~seen, SW_COLUMNS].isna().all().all()  # a gap removes the whole block
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
    assert not copies[0][SW_COLUMNS].equals(copies[1][SW_COLUMNS])
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


# Dst baseline offset augmentation (issue 23)


def _offset_table(start="2016-06-01", end="2017-03-01") -> pd.DataFrame:
    hours = pd.date_range(start, end, freq="h")
    rng = np.random.default_rng(1)
    n = len(hours)
    df = pd.DataFrame({c: rng.normal(size=n) for c in ["bz_gsm", "by_gsm", "b_mag_avg", "proton_density", "proton_temp", "electric_field"]}, index=hours)
    df["flow_speed"] = 400 + 50 * rng.random(n)
    df["flow_pressure"] = 2 + rng.random(n)
    df["f107"] = 100.0
    df["dst"] = -20 + 15 * np.sin(np.arange(n) / 40) + rng.normal(0, 2, n)
    return df


def test_offsets_are_constant_per_block_and_in_range():
    off = block_offsets(pd.date_range("2000-01-01", periods=5 * BLOCK + 10, freq="h"), seed=3)
    assert len(off) == 5 * BLOCK + 10
    assert all(len(set(off[i * BLOCK:(i + 1) * BLOCK])) == 1 for i in range(5))
    assert len(set(off[::BLOCK])) == 6
    assert OFFSET_RANGE[0] <= off.min() and off.max() <= OFFSET_RANGE[1]


def test_consistent_rows_keep_delta_target_and_diffs_unchanged():
    df = _offset_table()
    shifted = offset_table(df, seed=5)
    X0, y0 = build_features(df, horizon=1)
    X1, y1 = build_features(shifted, horizon=1)
    rows = pd.Series(consistent_rows(df.index, 1), index=df.index).reindex(X0.index).to_numpy()
    assert rows.mean() < 0.9
    for name in ["dst_diff1", "dst_diff3"]:
        np.testing.assert_allclose(X1[name][rows], X0[name][rows], atol=1e-3)
    np.testing.assert_allclose((y1 - X1["dst_lag0"])[rows], (y0 - X0["dst_lag0"])[rows], atol=1e-3)
    for name in ["dst_lag0", "dst_lag5", "dst_min24", "dst_mean24"]:
        shift = (X1[name] - X0[name])[rows]
        np.testing.assert_allclose((y1 - y0)[rows], shift, atol=1e-3)
    assert not np.allclose((y1 - y0)[rows], 0)
    np.testing.assert_array_equal(X1["bz_gsm_lag0"], X0["bz_gsm_lag0"])


def test_ring_current_terms_follow_the_shifted_dst():
    df = _offset_table()
    X0, _ = build_features(df, ring_current=True)
    X1, _ = build_features(offset_table(df, seed=5), ring_current=True)
    rows = pd.Series(consistent_rows(df.index, 1), index=df.index).reindex(X0.index).to_numpy()
    shift = (X1["dst_lag0"] - X0["dst_lag0"])[rows]
    np.testing.assert_allclose((X1["dst_star_lag3"] - X0["dst_star_lag3"])[rows], shift, atol=1e-3)
    assert not np.allclose((X1["om_step_lag0"] - X0["om_step_lag0"])[rows], 0)


def test_copies_stay_in_the_train_years():
    train_xgb = _script("train_xgb")
    df = _offset_table()
    X, y = train_xgb.offset_copies(df, copies=2, horizon=1)
    assert X.index.max() < pd.Timestamp("2017-01-01") and len(X) == len(y)
    assert X.index.is_unique is False  # two copies of the same hours
    X_clean, _ = build_features(df)
    assert list(X.columns) == list(X_clean.columns)


def test_off_by_default():
    assert _script("train_xgb").build_parser().parse_args([]).dst_offset_aug == 0


# Real-time like IMF and plasma gaps (issue 41)

from oculstorm.augment import GAP_MODELS, IMF_COLUMNS, PLASMA_COLUMNS, blank_gaps, gap_copies, gap_mask  # noqa: E402
from oculstorm.features import AGE_CAP, observation_age  # noqa: E402


def _runs(mask: np.ndarray) -> np.ndarray:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


@pytest.mark.parametrize("block, share, short", [("imf", 0.1862, 0.2212), ("plasma", 0.2695, 0.4011)])
def test_gap_masks_match_the_issue_31_rates_and_lengths(block, share, short):
    model = GAP_MODELS[block]
    mask = gap_mask(400_000, model, np.random.default_rng(1))
    runs = _runs(mask)
    assert abs(mask.mean() - share) < 0.01
    assert abs(runs[runs <= 3].sum() / runs.sum() - short) < 0.02
    assert abs(runs.mean() - model["mean_gap"]) < 0.1
    assert runs.max() <= model["longest"]
    half = gap_mask(400_000, model, np.random.default_rng(1), rate=0.5)
    assert abs(half.mean() - share / 2) < 0.01
    assert not gap_mask(1000, model, np.random.default_rng(1), rate=0.0).any()


def test_blank_gaps_only_removes_values_and_keeps_the_blocks_apart():
    df = _table("2010-01-01", "2012-01-01")
    pd.testing.assert_frame_equal(blank_gaps(df, rate=0.0, seed=2), df)
    out = blank_gaps(df, seed=2)
    assert out.index.equals(df.index) and out.dtypes.equals(df.dtypes)
    pd.testing.assert_frame_equal(out[["dst", "f107"]], df[["dst", "f107"]])
    seen = out.notna()
    pd.testing.assert_frame_equal(out[seen], df[seen])  # every value left is the clean one
    imf, plasma = out["bz_gsm"].isna(), out["flow_speed"].isna()
    assert out.loc[imf, IMF_COLUMNS].isna().all().all() and out.loc[plasma, PLASMA_COLUMNS].isna().all().all()
    assert out.loc[~imf, ["bz_gsm", "by_gsm", "b_mag_avg"]].notna().all().all()
    assert out.loc[~plasma, ["flow_speed", "proton_density", "proton_temp", "flow_pressure"]].notna().all().all()
    assert (out["electric_field"].isna() == (imf | plasma)).all()
    assert 0.15 < imf.mean() < 0.22 and 0.23 < plasma.mean() < 0.31
    assert 0.02 < (imf & ~plasma).mean() and 0.02 < (plasma & ~imf).mean()  # independent masks
    with pytest.raises(ValueError):
        blank_gaps(df, rate=-1.0)


def test_blank_gaps_up_to_t_does_not_depend_on_the_future():
    df = _table()
    cut = df.index[len(df) // 2]
    out = blank_gaps(df, seed=7)
    pd.testing.assert_frame_equal(blank_gaps(df[df.index <= cut], seed=7), out[out.index <= cut])
    corrupted = df.copy()
    corrupted.loc[corrupted.index > cut, SW_COLUMNS] = np.nan
    pd.testing.assert_frame_equal(blank_gaps(corrupted, seed=7)[out.index <= cut], out[out.index <= cut])


def test_observation_age_counts_back_from_t_only():
    hours = pd.date_range("2016-01-01", periods=10, freq="h")
    observed = pd.Series([False, True, True, False, False, False, True, False, False, False], index=hours)
    age = observation_age(observed, cap=2)
    assert age.tolist() == [2, 0, 0, 1, 2, 2, 0, 1, 2, 2]
    # Changing hours after t never changes the age at t.
    for t in range(len(observed)):
        flipped = observed.copy()
        flipped.iloc[t + 1 :] = ~flipped.iloc[t + 1 :]
        assert observation_age(flipped, cap=2).iloc[: t + 1].equals(age.iloc[: t + 1])
    df = blank_gaps(_table(), seed=3)
    X, _ = build_features(df, gap_age=True)
    assert X["imf_age"].max() <= AGE_CAP and (X["imf_age"] > 0).mean() > 0.1
    assert ((X["plasma_age"] == 0) == df["flow_speed"].notna().reindex(X.index)).all()


def test_gap_copies_hold_training_hours_only_and_are_off_by_default():
    df = _table("2016-10-01", "2021-03-01")
    assert gap_copies(df, copies=0) == [] and gap_copies(df, copies=2, rate=0.0) == []
    copies = gap_copies(df, copies=2, seed=4)
    for copy in copies:
        assert copy.index.equals(df.index[df.index < VAL_START])
    assert not copies[0]["bz_gsm"].isna().equals(copies[1]["bz_gsm"].isna())
    pd.testing.assert_frame_equal(copies[1], blank_gaps(df[df.index < VAL_START], 1.0, 5))
