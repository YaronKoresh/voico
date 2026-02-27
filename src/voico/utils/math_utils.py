import numpy as np


def safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(a, b)
        result[~np.isfinite(result)] = fill
    return result
