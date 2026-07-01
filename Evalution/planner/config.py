"""Configuration dataclasses for the tender_information_planner package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from Evalution.client import DEFAULT_OPENROUTER_BASE_URL


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Configuration for the LLM client.

    Attributes:
        api_key: API key for the LLM provider. Can be empty string for local models.
        base_url: Base URL of the OpenAI-compatible API endpoint.
        model: Model identifier string (e.g. "gpt-4o", "grok-3"). Defaults to OPENROUTER_MODEL env var.
        temperature: Sampling temperature. Low values produce more deterministic output.
        max_tokens: Maximum tokens in the LLM response.
        timeout: Request timeout in seconds.
    """

    api_key: str = ""
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Configuration for retry behaviour with exponential back-off.

    Attributes:
        max_retries: Maximum number of retry attempts before raising.
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Upper-bound cap on the back-off delay in seconds.
        exponential_base: Base for the exponential calculation (delay = base^attempt).
        jitter: When True, add a small random jitter to each delay to avoid
                thundering-herd effects.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Top-level configuration for :class:`TenderPlanner`.

    Attributes:
        llm: LLM connection settings.
        retry: Retry and back-off settings used by both the HTTP client and
               the validation-repair loop.
        validation_retries: How many times the planner should ask the LLM to
            repair its output when Pydantic validation fails.
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    validation_retries: int = 3