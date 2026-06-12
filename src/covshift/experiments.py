from __future__ import annotations

import numpy as np
import pandas as pd

from covshift.covariances import diagonal_power_law_covariance
from covshift.data import generate_linear_data
from covshift.estimators import fit_least_squares, fit_pooled_least_squares
from covshift.metrics import target_excess_risk


def make_beta(p: int) -> np.ndarray:
    """Dense unit-norm regression vector."""
    return np.ones(p) / np.sqrt(p)


def make_target_covariance(p: int) -> np.ndarray:
    """Target covariance with T_j = 1 / p."""
    return np.eye(p)


def make_source_covariance(p: int, alpha: float) -> np.ndarray:
    """
    Source covariance with normalized eigenvalues

        S_j = j^{-alpha} / Z_alpha.

    The trace is normalized to p, so the average eigenvalue is 1.
    """
    return diagonal_power_law_covariance(p=p, alpha=alpha)


def theoretical_T(
    n_source: int,
    m_target: int,
    alpha: float,
    p: int,
) -> float:
    """
    Computes your theoretical quantity

        T(n,m,alpha,p)
        = d / m + (1/n) sum_{j in J^c} T_j / S_j,

    in the special case tr(Sigma_T) = tr(Sigma_S).

    Here:
        T_j = 1 / p,
        S_j = j^{-alpha} / Z_alpha,

    and

        J^c = {j : (m / n) * (T_j / S_j) <= 1}.

    Since d = |J|, we have d = p - |J^c|.
    """
    j = np.arange(1, p + 1, dtype=float)

    z_alpha = np.sum(j ** (-alpha))
    s_j = j ** (-alpha) / z_alpha
    t_j = np.ones(p) / p

    ratio = t_j / s_j

    j_complement_mask = (m_target / n_source) * ratio <= 1.0

    d = int((~j_complement_mask).sum())

    return float(
        d / m_target
        + (1.0 / n_source) * np.sum(ratio[j_complement_mask])
    )


def population_pooled_proxy(
    n_source: int,
    m_target: int,
    alpha: float,
    p: int,
    noise_std: float = 1.0,
) -> float:
    """
    Population approximation for the pooled excess risk:

        sigma^2 tr(Sigma_T (n Sigma_S + m Sigma_T)^{-1}).

    This is not the same as the finite-sample Monte Carlo value,
    but it is a useful deterministic comparison curve.
    """
    sigma_source = make_source_covariance(p=p, alpha=alpha)
    sigma_target = make_target_covariance(p=p)

    information = n_source * sigma_source + m_target * sigma_target

    return float(
        noise_std**2
        * np.trace(sigma_target @ np.linalg.inv(information))
    )


def target_only_wishart_proxy(
    m_target: int,
    p: int,
    noise_std: float = 1.0,
) -> float:
    """
    Exact expectation for target-only OLS when Sigma_T = I_p
    and m_target > p + 1:

        E tr((X_T^T X_T)^{-1}) = p / (m_target - p - 1).

    Multiplied by sigma^2.
    """
    if m_target <= p + 1:
        return np.nan

    return float(noise_std**2 * p / (m_target - p - 1))


def run_single_alpha_experiment(
    *,
    alpha: float,
    n_source: int,
    m_target: int,
    p: int,
    noise_std: float,
    seed: int,
) -> list[dict[str, float | int | str]]:
    sigma_source = make_source_covariance(p=p, alpha=alpha)
    sigma_target = make_target_covariance(p=p)
    beta = make_beta(p)

    x_source, y_source = generate_linear_data(
        n=n_source,
        covariance=sigma_source,
        beta=beta,
        noise_std=noise_std,
        seed=seed,
    )

    x_target, y_target = generate_linear_data(
        n=m_target,
        covariance=sigma_target,
        beta=beta,
        noise_std=noise_std,
        seed=seed + 10_000_000,
    )

    beta_target = fit_least_squares(x_target, y_target)
    beta_pooled = fit_pooled_least_squares(
        x_source=x_source,
        y_source=y_source,
        x_target=x_target,
        y_target=y_target,
    )

    risk_target = target_excess_risk(
        beta_hat=beta_target,
        beta_true=beta,
        sigma_target=sigma_target,
    )

    risk_pooled = target_excess_risk(
        beta_hat=beta_pooled,
        beta_true=beta,
        sigma_target=sigma_target,
    )

    theory_t = theoretical_T(
        n_source=n_source,
        m_target=m_target,
        alpha=alpha,
        p=p,
    )

    pooled_proxy = population_pooled_proxy(
        n_source=n_source,
        m_target=m_target,
        alpha=alpha,
        p=p,
        noise_std=noise_std,
    )

    target_proxy = target_only_wishart_proxy(
        m_target=m_target,
        p=p,
        noise_std=noise_std,
    )

    common = {
        "alpha": alpha,
        "n_source": n_source,
        "m_target": m_target,
        "p": p,
        "seed": seed,
        "theoretical_T": theory_t,
        "population_pooled_proxy": pooled_proxy,
        "target_only_proxy": target_proxy,
    }

    return [
        {
            **common,
            "estimator": "target_only",
            "excess_risk": risk_target,
        },
        {
            **common,
            "estimator": "pooled",
            "excess_risk": risk_pooled,
        },
    ]


def run_alpha_sweep(
    *,
    alphas: np.ndarray,
    n_source: int,
    m_target: int,
    p: int,
    repetitions: int,
    noise_std: float = 1.0,
    seed: int = 123,
) -> pd.DataFrame:
    rows = []

    for alpha in alphas:
        for repetition in range(repetitions):
            rows.extend(
                run_single_alpha_experiment(
                    alpha=float(alpha),
                    n_source=n_source,
                    m_target=m_target,
                    p=p,
                    noise_std=noise_std,
                    seed=seed + repetition,
                )
            )

    return pd.DataFrame(rows)