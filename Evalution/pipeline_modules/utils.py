"""Pipeline utilities module.

Provides utility functions for JSON handling and output formatting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_json_dict(obj: Any) -> Any:
    """Recursively convert dataclasses / Pydantic models to plain JSON values."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if is_dataclass(obj):
        return {k: to_json_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_json_dict(v) for k, v in obj.items()}
    return obj


def write_json(path: str | Path, data: Any) -> None:
    """Write data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_dict(data), f, indent=4, ensure_ascii=False)


def print_planner_outcomes(results: list[Any]) -> None:
    """Print planning results to console."""
    for idx, result in enumerate(results):
        sep = "=" * 60
        if result is None:
            # Quarantined key point (failed input validation)
            print(f"\n{sep}\n   key_points[{idx}] - SKIPPED (malformed input, not planned)\n{sep}")
            continue
        if result.success and result.plan is not None:
            print(f"\n{sep}\n   {result.criterion_id} - SUCCESS  (attempts: {result.attempts})\n{sep}")
            print(json.dumps(result.plan.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False))
        else:
            print(f"\n{sep}\n   {result.criterion_id} - FAILED  (attempts: {result.attempts})\n{sep}")
            print(f"Error:\n{result.error}")
    
    succeeded = sum(1 for r in results if r is not None and r.success)
    print(f"\n{'=' * 60}\n   Summary: {succeeded}/{len(results)} criteria planned successfully\n{'=' * 60}")
