import numpy as np


def fit_least_squares(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta_hat, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta_hat


def fit_pooled_least_squares(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
) -> np.ndarray:
    x = np.vstack([x_source, x_target])
    y = np.concatenate([y_source, y_target])
    return fit_least_squares(x, y)
