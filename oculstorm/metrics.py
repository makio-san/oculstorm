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
    """Next hour equals the current hour."""
    return X["dst_lag0"].to_numpy()


def trend_persistence(X: pd.DataFrame) -> np.ndarray:
    """Next hour continues the last hourly change (linear extrapolation)."""
    return (X["dst_lag0"] + X["dst_diff1"].fillna(0)).to_numpy()


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
