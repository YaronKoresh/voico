import numpy as np


def safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """
    Performs division ignoring divide-by-zero errors, replacing them with a fill value.
    Essential for spectral processing where silence/noise floors create zeros.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(a, b)
        result[~np.isfinite(result)] = fill
    return result


def next_power_of_2(x: int) -> int:
    """Returns the next power of 2 greater than or equal to x."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()
