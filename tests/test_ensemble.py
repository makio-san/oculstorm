"""GRU plus XGBoost weights: grid fit, bands from the issue time Dst, and the blend."""
import numpy as np

from oculstorm.ensemble import BANDS, GRID, band_of, combine, fit_band_weights, fit_weight

QUIET, STORM, INTENSE = list(BANDS)[2], list(BANDS)[1], list(BANDS)[0]


def test_grid_is_coarse_and_closed():
    assert GRID[0] == 0 and GRID[-1] == 1 and len(GRID) == 21


def test_fit_weight_finds_the_better_model_and_the_mix():
    rng = np.random.default_rng(0)
    y = rng.normal(size=4000)
    a, b = y + rng.normal(scale=1.0, size=4000), y + rng.normal(scale=1.0, size=4000)
    assert fit_weight(y, y, b) == 1.0  # a perfect GRU takes all the weight
    assert fit_weight(y, a, y) == 0.0
    assert abs(fit_weight(y, a, b) - 0.5) <= 0.1  # equal, independent errors: about half and half


def test_bands_come_from_dst_lag0_at_the_band_edges():
    assert list(band_of([0, -49.9, -50, -99.9, -100, -300])) == [QUIET, QUIET, STORM, STORM, INTENSE, INTENSE]


def test_band_weights_use_the_issue_time_band_and_fall_back_when_thin():
    n = 2000
    lag0 = np.where(np.arange(n) < 1000, -10.0, -70.0)  # 1000 quiet hours, 1000 storm hours, no intense
    y = np.zeros(n)
    gru = np.where(lag0 > -50, 0.0, 5.0)  # GRU perfect in quiet time, XGBoost perfect in storms
    xgb = np.where(lag0 > -50, 5.0, 0.0)
    w = fit_band_weights(y, gru, xgb, lag0)
    assert w[QUIET] == 1.0 and w[STORM] == 0.0
    assert w[INTENSE] == fit_weight(y, gru, xgb)  # no intense hours: the global weight
    assert np.allclose(combine(gru, xgb, w, lag0), 0.0)


def test_combine_with_one_weight():
    assert np.allclose(combine([2.0, 4.0], [0.0, 0.0], 0.25), [0.5, 1.0])
