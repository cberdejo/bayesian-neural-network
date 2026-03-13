"""
Computes all evaluation metrics for probabilistic predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Computes all evaluation metrics for probabilistic predictions.
    Args:
        y_true: The true target values.
        y_pred: The predicted target values.
        y_std: The standard deviation of the predicted target values.
        alpha: The alpha level for the Winkler score.
    Returns:
        A dictionary containing the following metrics:
        - RMSE: Root-mean-squared error (point accuracy)
        - MAE: Mean absolute error (point accuracy)
        - PICP: Prediction interval coverage probability (reliability)
        - MPIW: Mean prediction interval width (sharpness)
        - NLL: Gaussian negative log-likelihood (calibration trade-off)
        - Winkler: Winkler score at level alpha (sharpness + coverage)
    """
    sigma = np.clip(y_std, 1e-8, None)
    z = _z_from_alpha(alpha)

    lower = y_pred - z * sigma
    upper = y_pred + z * sigma
    covered = (y_true >= lower) & (y_true <= upper)

    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae     = float(mean_absolute_error(y_true, y_pred))
    picp    = float(np.mean(covered))
    mpiw    = float(np.mean(upper - lower))
    nll     = float(np.mean(
        0.5 * np.log(2 * np.pi * sigma**2) + (y_true - y_pred) ** 2 / (2 * sigma**2)
    ))
    penalty = np.where(
        covered, 0.0,
        np.where(y_true < lower, 2 / alpha * (lower - y_true), 2 / alpha * (y_true - upper)),
    )
    winkler = float(np.mean((upper - lower) + penalty))
    
    return {
        "RMSE": rmse, "MAE": mae,
        "PICP": picp, "MPIW": mpiw,
        "NLL": nll,  "Winkler": winkler,
    }


def _z_from_alpha(alpha: float) -> float:
    """Computes the z-score from the alpha level for a normal distribution."""
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / 2))
