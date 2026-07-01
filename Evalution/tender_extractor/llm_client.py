"""
LLM Client — lightweight OpenAI wrapper.

Reads OPENROUTER_API_KEY and OPENROUTER_MODEL from .env (or environment).
Model: Configured via OPENROUTER_MODEL environment variable.
Temperature: 0 for deterministic extraction.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from Evalution.client import get_openrouter_client, get_llm_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: Optional[Any] = None
_model: str = ""


def _load_env() -> None:
    """Load .env file if present (no external dependency needed)."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Also check parent dirs
        for p in [Path(__file__).resolve().parent.parent / ".env",
                  Path.home() / ".env"]:
            if p.exists():
                env_path = p
                break
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    # Only set if not already in env
                    if key not in os.environ:
                        os.environ[key] = value
        except Exception:
            pass  # Silently ignore .env parse errors


def get_client() -> Any:
    """Return the shared OpenRouter-backed client singleton."""
    global _client, _model
    if _client is None:
        _load_env()
        _model = get_llm_model()
        _client = get_openrouter_client(model=_model)
        logger.debug("OpenRouter client initialised (model=%s)", _model)
    return _client


def get_model() -> str:
    global _model
    if not _model:
        _model = get_llm_model()
    return _model


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------

def chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    response_format: Optional[Dict[str, str]] = None,
) -> str:
    """Send a chat completion request and return the content string."""
    client = get_client()
    kwargs: Dict[str, Any] = {
        "model": _model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if response_format:
        kwargs["response_format"] = response_format

    resp = client.create_chat_completion(**kwargs)
    content = resp.choices[0].message.content or ""
    return content


def chat_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Send a chat completion with JSON mode and return parsed dict."""
    raw = chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON, attempting repair")
        # Try to extract JSON from markdown code block
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return {"error": "invalid_json", "raw": raw}