import numpy as np

from covshift.metrics import target_excess_risk


def test_target_excess_risk_is_zero_for_correct_beta() -> None:
    beta = np.array([1.0, 2.0])
    sigma = np.eye(2)

    risk = target_excess_risk(beta, beta, sigma)

    assert risk == 0.0
