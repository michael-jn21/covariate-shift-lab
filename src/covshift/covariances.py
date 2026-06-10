import numpy as np


def diagonal_power_law_covariance(p: int, alpha: float) -> np.ndarray:
    eigenvalues = np.array([j ** (-alpha) for j in range(1, p + 1)])
    eigenvalues = eigenvalues / eigenvalues.mean()
    return np.diag(eigenvalues)


def rotation_matrix_2d(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s], [-s, c]])


def rotated_covariance_2d(eigenvalues: tuple[float, float], theta: float) -> np.ndarray:
    diagonal = np.diag(eigenvalues)
    rotation = rotation_matrix_2d(theta)
    return rotation.T @ diagonal @ rotation
