import logging
import time
from contextlib import contextmanager
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str, timing_dict: Optional[Dict[str, float]] = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if timing_dict is not None:
            timing_dict[name] = elapsed
        logger.debug(f"{name} took {elapsed:.4f}s")
