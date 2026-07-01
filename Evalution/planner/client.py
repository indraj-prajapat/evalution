"""OpenAI-compatible LLM client with retry, rate-limiting, and timeout support.

The client is model-agnostic: it works with any provider that exposes an
OpenAI-compatible chat completions endpoint (OpenAI, Grok/xAI, Azure OpenAI,
vLLM, Ollama, etc.).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import APIConnectionError, APITimeoutError, RateLimitError
from openai.types.chat import ChatCompletion

from Evalution.client import OpenRouterClient, get_openrouter_client
from Evalution.planner.config import LLMConfig, RetryConfig
from Evalution.planner.logging_utils import get_logger
from Evalution.planner.retry import with_retry, RetryError

log = get_logger(__name__)


@dataclass(slots=True)
class LLMResponse:
    """Thin wrapper around an LLM chat-completion response.

    Attributes:
        content: The assistant's reply text.
        model: The model that generated the response.
        usage: Token-usage statistics (prompt_tokens, completion_tokens, total_tokens).
        raw: The raw :class:`ChatCompletion` object for advanced inspection.
    """

    content: str
    model: str
    usage: dict[str, int]
    raw: ChatCompletion | None = None


class LLMClient:
    """OpenAI-compatible LLM client with built-in retry and logging.

    Parameters:
        config: :class:`LLMConfig` for model / endpoint settings.
        retry_config: :class:`RetryConfig` for retry behaviour.

    Example::

        client = LLMClient(
            config=LLMConfig(
                api_key="sk-...",
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
            )
        )
        response = client.chat(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello!",
        )
        print(response.content)
    """

    def __init__(
        self,
        config: LLMConfig,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        self._config = config
        self._retry_config = retry_config or RetryConfig()
        self._client = get_openrouter_client(
            api_key=config.api_key or None,
            model=config.model,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat-completion request with retry on transient errors.

        Parameters:
            system_prompt: System message that sets the LLM's behaviour.
            user_prompt: The user message containing the actual request.
            temperature: Override the default temperature for this call.
            max_tokens: Override the default max_tokens for this call.

        Returns:
            :class:`LLMResponse` with the assistant's reply and metadata.

        Raises:
            RetryError: If all retry attempts are exhausted.
        """
        retryable = (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        )

        decorated = with_retry(self._retry_config, retryable_exceptions=retryable)(
            self._single_call
        )
        return decorated(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _single_call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Make a single (non-retried) chat-completion call."""
        temp = temperature if temperature is not None else self._config.temperature
        tokens = max_tokens if max_tokens is not None else self._config.max_tokens

        log.debug(
            "llm_request",
            model=self._config.model,
            temperature=temp,
            max_tokens=tokens,
            system_prompt_length=len(system_prompt),
            user_prompt_length=len(user_prompt),
        )

        start = time.monotonic()
        completion = self._client.create_chat_completion(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            max_tokens=tokens,
        )
        elapsed = time.monotonic() - start

        content = completion.choices[0].message.content or ""
        usage = (
            {
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "total_tokens": completion.usage.total_tokens if completion.usage else 0,
            }
        )

        log.info(
            "llm_response",
            model=completion.model,
            elapsed_seconds=round(elapsed, 3),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            finish_reason=completion.choices[0].finish_reason,
        )

        return LLMResponse(
            content=content,
            model=completion.model,
            usage=usage,
            raw=completion,
        )