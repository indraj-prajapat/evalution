"""Retry utilities with exponential back-off and jitter."""

from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable, TypeVar, Sequence, Type

from Evalution.planner.config import RetryConfig
from Evalution.planner.logging_utils import get_logger

log = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, attempts: int, last_exception: BaseException | None) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(message)


def _compute_delay(config: RetryConfig, attempt: int) -> float:
    """Calculate the delay for a given attempt number using exponential back-off.

    Parameters:
        config: Retry configuration.
        attempt: Zero-based attempt index (0 = first retry).

    Returns:
        Delay in seconds, capped at ``config.max_delay``.  Jitter is applied
        when ``config.jitter`` is True.
    """
    delay = min(config.base_delay * (config.exponential_base ** attempt), config.max_delay)
    if config.jitter:
        delay = delay * (0.5 + random.random())
    return delay


def with_retry(
    config: RetryConfig,
    retryable_exceptions: Sequence[Type[BaseException]] = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries the wrapped function on specified exceptions.

    Parameters:
        config: :class:`RetryConfig` instance controlling back-off behaviour.
        retryable_exceptions: Tuple of exception types that trigger a retry.

    Returns:
        A decorator that wraps the target function.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(config.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == config.max_retries:
                        break
                    delay = _compute_delay(config, attempt)
                    log.warning(
                        "retry_attempt",
                        function=fn.__qualname__,
                        attempt=attempt + 1,
                        max_retries=config.max_retries,
                        delay=round(delay, 2),
                        error=str(exc),
                    )
                    time.sleep(delay)
            raise RetryError(
                f"All {config.max_retries + 1} attempts failed for {fn.__qualname__}",
                attempts=config.max_retries + 1,
                last_exception=last_exc,
            )

        return wrapper  # type: ignore[return-value]

    return decorator