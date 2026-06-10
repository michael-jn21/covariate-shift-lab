import numpy as np


def generate_linear_data(
    n: int,
    covariance: np.ndarray,
    beta: np.ndarray,
    noise_std: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = covariance.shape[0]

    x = rng.multivariate_normal(
        mean=np.zeros(p),
        cov=covariance,
        size=n,
    )
    noise = rng.normal(loc=0.0, scale=noise_std, size=n)
    y = x @ beta + noise

    return x, y
