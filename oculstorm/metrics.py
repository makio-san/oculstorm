"""Naive baselines and the metrics used to judge every model against them."""
import numpy as np
import pandas as pd

# Evaluation subsets by the observed Dst at the target hour (nT).
REGIMES = {
    "all": lambda y: np.ones(len(y), dtype=bool),
    "quiet (Dst > -50)": lambda y: y > -50,
    "storm (Dst <= -50)": lambda y: y <= -50,
    "intense (Dst <= -100)": lambda y: y <= -100,
}


def persistence(X: pd.DataFrame) -> np.ndarray:
    """The target hour equals the current hour, whatever the lead time."""
    return X["dst_lag0"].to_numpy()


def trend_persistence(X: pd.DataFrame, horizon: int = 1) -> np.ndarray:
    """The last hourly change continues for `horizon` hours (linear extrapolation)."""
    return (X["dst_lag0"] + horizon * X["dst_diff1"].fillna(0)).to_numpy()


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def skill(y: np.ndarray, pred: np.ndarray, reference: np.ndarray) -> float:
    """MSE skill score: 1 is perfect, 0 equals the reference, negative is worse."""
    return float(1 - np.mean((y - pred) ** 2) / np.mean((y - reference) ** 2))


def evaluate(y: pd.Series, predictions: dict[str, np.ndarray], reference: str = "persistence") -> pd.DataFrame:
    """Score every prediction in every regime, with skill measured against `reference`."""
    truth = y.to_numpy()
    rows = []
    for regime, mask_fn in REGIMES.items():
        mask = mask_fn(truth)
        for name, pred in predictions.items():
            rows.append({
                "regime": regime,
                "n_hours": int(mask.sum()),
                "model": name,
                "rmse": rmse(truth[mask], pred[mask]),
                "mae": mae(truth[mask], pred[mask]),
                "skill_vs_persistence": skill(truth[mask], pred[mask], predictions[reference][mask]),
            })
    return pd.DataFrame(rows)


def storm_weights(dst, peak: float = 1.0, depth: float = 100.0) -> np.ndarray:
    """Training weights that rise linearly from 1 at Dst >= 0 nT to `peak` at Dst <= -depth nT.

    Storm hours are a few percent of the record, so an unweighted loss is dominated by quiet
    time. `peak` = 1 gives equal weights, the plain baseline.
    """
    return 1 + (peak - 1) * np.clip(-np.asarray(dst, dtype=float) / depth, 0, 1)


def block_bootstrap_rmse(y, predictions: dict, mask, n_boot: int = 2000, block: int = 168, seed: int = 0) -> np.ndarray:
    """RMSE of every prediction on `n_boot` moving block resamples of the hours in `mask`.

    Errors an hour apart are strongly correlated (a storm lasts days), so hours are drawn in
    blocks of `block` consecutive rows rather than one by one. All predictions share the same
    resamples, which makes the differences between them paired. Returns shape (n_boot, n_models).
    """
    y, mask = np.asarray(y, dtype=float), np.asarray(mask, dtype=bool)
    n = len(y)
    block = min(block, n)
    starts = np.random.default_rng(seed).integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))

    def block_sums(values: np.ndarray) -> np.ndarray:
        total = np.concatenate([[0.0], np.cumsum(values)])
        return total[block:] - total[:-block]  # sum over rows [s, s + block) for every start s

    counts = block_sums(mask.astype(float))[starts].sum(axis=1)
    out = np.empty((n_boot, len(predictions)))
    for j, pred in enumerate(predictions.values()):
        squared = np.where(mask, (y - np.asarray(pred, dtype=float)) ** 2, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = np.sqrt(block_sums(squared)[starts].sum(axis=1) / counts)
    return out
