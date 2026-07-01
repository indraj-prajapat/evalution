"""Shared OpenRouter-compatible client helpers for the evaluation package."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import httpx

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_shared_client: Optional["OpenRouterClient"] = None


def _normalize_model_name(model: Optional[str]) -> str:
    """Normalize OpenRouter-compatible model names."""
    if not model:
        return get_llm_model()
    model = model.strip()
    if not model:
        return get_llm_model()
    if "/" in model:
        return model
    return f"openai/{model}"


def _load_env() -> None:
    """Load environment variables from the workspace .env file when present."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        for candidate in [
            Path(__file__).resolve().parent.parent / ".env",
            Path.home() / ".env",
        ]:
            if candidate.exists():
                env_path = candidate
                break
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    if key not in os.environ:
                        os.environ[key] = value
        except Exception:
            pass


def get_llm_api_key() -> str:
    """Return the configured API key for the OpenRouter provider."""
    _load_env()
    key = os.getenv("OPENROUTER_API_KEY", "").strip()

    print("Using key:", key[:12] + "..." if key else "NOT FOUND")
    return os.getenv("OPENROUTER_API_KEY", "").strip() 


def get_llm_model() -> str:
    """Return the configured model identifier for the OpenRouter provider."""
    _load_env()
    model = (
        os.getenv("OPENROUTER_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or os.getenv("MODEL", "").strip()
    )
    if not model:
        raise ValueError(
            "No model configured. Please set OPENROUTER_MODEL in your .env file."
        )
    return model


class OpenRouterClient:
    """Small wrapper around the OpenAI-compatible OpenRouter endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: int = 120,
    ) -> None:
        self.api_key = (api_key or get_llm_api_key()).strip()
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found. Set it in .env or as an env var.")
        self.model = _normalize_model_name(model or get_llm_model())
        self.base_url = (base_url or DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
        self.timeout = timeout

    def create_chat_completion(
        self,
        *,
        model: Optional[str] = None,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": _normalize_model_name(model or self.model),
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        request_kwargs.update(kwargs)

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/",
                "X-Title": "TenderEvaluation",
            },
            json=request_kwargs,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return self._wrap_payload(payload)

    def _wrap_payload(self, payload: dict[str, Any]) -> Any:
        choices = payload.get("choices", [])
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {})
        usage = payload.get("usage") or {}
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=message.get("content") or ""),
                    finish_reason=first_choice.get("finish_reason"),
                )
            ],
            model=payload.get("model") or self.model,
            usage=SimpleNamespace(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )


def get_openrouter_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    timeout: int = 120,
) -> OpenRouterClient:
    """Return a shared OpenRouter client instance."""
    global _shared_client
    if _shared_client is None:
        _shared_client = OpenRouterClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
    return _shared_client
