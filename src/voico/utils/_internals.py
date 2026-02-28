import logging
import time
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np

_logger = logging.getLogger(__name__)


def safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(a, b)
        result[~np.isfinite(result)] = fill
    return result


@contextmanager
def timer(name: str, timing_dict: Optional[Dict[str, float]] = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if timing_dict is not None:
            timing_dict[name] = elapsed
        _logger.debug(f"{name} took {elapsed:.4f}s")
