import numpy as np


def target_excess_risk(
    beta_hat: np.ndarray,
    beta_true: np.ndarray,
    sigma_target: np.ndarray,
) -> float:
    diff = beta_hat - beta_true
    return float(diff.T @ sigma_target @ diff)
