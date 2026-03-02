import asyncio
import functools
import logging
import time
from typing import Any, Callable, Tuple, Type, TypeVar

from voico.utils._internals import timer

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerError(Exception):
    __slots__ = ()


class CircuitBreaker:
    def __init__(
        self, failure_threshold: int = 3, recovery_timeout: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: float = 0.0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.error("CircuitBreaker OPEN: Failure threshold exceeded.")

    def record_success(self) -> None:
        self.failures = 0
        self.state = "CLOSED"

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
        return False


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    logger.warning(
                        f"Retry {attempts}/{max_attempts} for {func.__name__} after {type(e).__name__}: {e}"
                    )
                    if attempts >= max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts."
                        )
                        raise
                    time.sleep(delay)
            raise CircuitBreakerError(
                f"Retry loop ended unexpectedly for {func.__name__}"
            )

        return wrapper

    return decorator


def with_async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    logger.warning(
                        f"Async Retry {attempts}/{max_attempts} for {func.__name__} after {type(e).__name__}: {e}"
                    )
                    if attempts >= max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts."
                        )
                        raise
                    await asyncio.sleep(delay)
            raise CircuitBreakerError(
                f"Async retry loop ended unexpectedly for {func.__name__}"
            )

        return wrapper

    return decorator


def error_boundary(
    fallback: Any = None,
    handled_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    reraise_unhandled: bool = True,
) -> Callable:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except handled_exceptions as e:
                logger.error(
                    f"ErrorBoundary caught {type(e).__name__} in {func.__name__}: {e}"
                )
                return fallback
            except BaseException as e:
                if reraise_unhandled:
                    raise
                logger.error(
                    f"ErrorBoundary unhandled {type(e).__name__} in {func.__name__}: {e}"
                )
                return fallback

        return wrapper

    return decorator


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerError",
    "error_boundary",
    "timer",
    "with_async_retry",
    "with_retry",
]
