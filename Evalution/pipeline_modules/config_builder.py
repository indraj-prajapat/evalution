"""Pipeline configuration module.

Handles building and managing configuration for the evaluation pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..client import DEFAULT_OPENROUTER_BASE_URL, get_llm_api_key, get_llm_model
from ..planner.config import LLMConfig, PlannerConfig, RetryConfig


def build_planner_config(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 120,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    validation_retries: int = 3,
) -> PlannerConfig:
    """
    Build a PlannerConfig with the specified or default settings.
    
    All parameters are optional. If not provided, values are read from
    environment variables or use sensible defaults.
    
    Args:
        api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
        model: Model name. Defaults to OPENROUTER_MODEL env var.
        base_url: Base URL for API. Defaults to OPENROUTER_BASE_URL env var.
        temperature: LLM temperature.
        max_tokens: Maximum tokens for response.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        base_delay: Base delay between retries.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff.
        jitter: Whether to add random jitter to delays.
        validation_retries: Number of validation retry attempts.
    
    Returns:
        Configured PlannerConfig instance.
    """
    return PlannerConfig(
        llm=LLMConfig(
            api_key=api_key or get_llm_api_key(),
            base_url=base_url or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            model=model or get_llm_model(),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        ),
        retry=RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
        ),
        validation_retries=validation_retries,
    )


def build_planner_tender(tender_output: dict[str, Any]) -> dict[str, Any]:
    """
    Build the tender structure expected by the planner.
    
    Args:
        tender_output: The tender output dictionary containing key_points.
    
    Returns:
        Dictionary formatted for the planner with criteria.
    """
    criteria = []
    for idx, key_point in enumerate(tender_output.get("key_points", []), start=1):
        kp = key_point if isinstance(key_point, dict) else {}
        criteria.append(
            {
                "criterion_id": _get_key_point_id(kp, idx),
                "text": kp.get("point") or kp.get("text") or "",
            }
        )
    
    metadata = tender_output.get("metadata", {})
    return {
        "tender_id": (
            metadata.get("gem_bid_id")
            or metadata.get("tender_reference_number")
            or metadata.get("tender_title")
            or "TENDER_OUTPUT"
        ),
        "title": metadata.get("tender_title", ""),
        "key_points": criteria,
    }


def _get_key_point_id(key_point: dict[str, Any], idx: int) -> str:
    """Extract or generate a key point ID."""
    return (
        key_point.get("key_id")
        or key_point.get("criterion_id")
        or key_point.get("id")
        or f"CRIT_{idx:03d}"
    )


def resolve_worker_count(max_workers: Optional[int], item_count: int) -> int:
    """
    Resolve the number of workers to use for parallel processing.
    
    Args:
        max_workers: Configured maximum workers.
        item_count: Number of items to process.
    
    Returns:
        Actual number of workers to use.
    """
    if item_count <= 0:
        return 1
    
    default_max_workers = int(os.environ.get("EVALUATION_MAX_WORKERS", "4"))
    configured = max_workers if max_workers is not None else default_max_workers
    return max(1, min(configured, item_count))
