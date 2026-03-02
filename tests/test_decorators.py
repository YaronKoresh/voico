import asyncio

import pytest

from voico.utils.decorators import error_boundary, with_async_retry, with_retry


def test_with_retry_retries_then_succeeds() -> None:
    state = {"count": 0}

    @with_retry(max_attempts=3, delay=0.0, exceptions=(ValueError,))
    def flaky() -> int:
        state["count"] += 1
        if state["count"] < 3:
            raise ValueError("transient")
        return 42

    assert flaky() == 42
    assert state["count"] == 3


def test_with_async_retry_retries_then_succeeds() -> None:
    state = {"count": 0}

    @with_async_retry(max_attempts=3, delay=0.0, exceptions=(ValueError,))
    async def flaky() -> int:
        state["count"] += 1
        if state["count"] < 2:
            raise ValueError("transient")
        return 7

    assert asyncio.run(flaky()) == 7
    assert state["count"] == 2


def test_error_boundary_handles_expected_exception() -> None:
    @error_boundary(fallback="fallback", handled_exceptions=(ValueError,))
    def fail() -> str:
        raise ValueError("expected")

    assert fail() == "fallback"


def test_error_boundary_reraises_unhandled() -> None:
    @error_boundary(fallback="fallback", handled_exceptions=(ValueError,))
    def fail() -> str:
        raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError):
        fail()
