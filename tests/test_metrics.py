"""Storm weights and the block bootstrap behind the model comparison."""
import numpy as np

from oculstorm.metrics import block_bootstrap_rmse, rmse, storm_weights


def test_storm_weights_ramp_from_one_to_peak():
    w = storm_weights([20, 0, -50, -100, -400], peak=5)
    assert np.allclose(w, [1, 1, 3, 5, 5])
    assert np.allclose(storm_weights([-300, 10]), 1)  # peak 1 is the unweighted baseline


def test_bootstrap_is_centred_on_the_rmse_and_paired():
    rng = np.random.default_rng(1)
    y = rng.normal(size=5000)
    good, bad = y + rng.normal(scale=1.0, size=5000), y + rng.normal(scale=2.0, size=5000)
    mask = np.ones(5000, dtype=bool)
    boot = block_bootstrap_rmse(y, {"good": good, "bad": bad}, mask, n_boot=500, block=50)
    assert boot.shape == (500, 2)
    assert abs(np.median(boot[:, 0]) - rmse(y, good)) < 0.05
    assert (boot[:, 0] < boot[:, 1]).all()  # same resamples, so the clear winner wins every time


def test_bootstrap_only_scores_masked_hours():
    y = np.zeros(1000)
    pred = np.where(np.arange(1000) < 500, 0.0, 10.0)
    mask = np.arange(1000) < 500
    boot = block_bootstrap_rmse(y, {"p": pred}, mask, n_boot=200, block=20)
    assert np.allclose(boot[~np.isnan(boot)], 0.0)
