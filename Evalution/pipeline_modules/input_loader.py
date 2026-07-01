"""Pipeline input loader module.

Handles loading and validation of JSON inputs for the evaluation pipeline.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_json_input(data: dict[str, Any] | str | Path, label: str) -> dict[str, Any]:
    """
    Load JSON from various input types.
    
    Args:
        data: Can be a dict (already loaded), a JSON string, or a file path.
        label: Label for error messages.
    
    Returns:
        A deep copy of the loaded JSON dict.
    
    Raises:
        TypeError: If the input type is not supported.
        FileNotFoundError: If a file path is provided but doesn't exist.
        json.JSONDecodeError: If the JSON string is invalid.
    """
    if isinstance(data, dict):
        return deepcopy(data)
    
    if isinstance(data, str) and data.lstrip().startswith(("{", "[")):
        return json.loads(data)
    
    if isinstance(data, Path) or isinstance(data, str):
        path = Path(data)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"{label}: File not found at {path}")
    
    raise TypeError(f"{label}: unsupported type {type(data)}")


def validate_key_points(key_points: list[Any]) -> dict[int, str]:
    """
    Validate the minimal required shape of each key point.
    
    Previously, `key_point.get("point") or key_point.get("text") or ""`
    silently turned a missing/malformed key point into an empty-string
    criterion, which the planner and verification engine would then process
    as if it were a real (if odd) requirement -- producing a degenerate
    verdict with no indication anything was actually wrong with the input.
    
    Returns a mapping of {key_point index -> human-readable reason} for
    every entry that fails validation. Entries not in the mapping are valid.
    Nothing is dropped from `key_points` here -- callers are expected to
    quarantine invalid indices (skip planning/evaluation for them) and
    surface `errors[idx]` as an explicit failure for that one key point,
    not abort the whole batch.
    
    Args:
        key_points: List of key points to validate.
    
    Returns:
        Dictionary mapping invalid indices to error messages.
    """
    errors: dict[int, str] = {}
    for idx, kp in enumerate(key_points):
        if not isinstance(kp, dict):
            errors[idx] = f"key_points[{idx}] is not a JSON object (got {type(kp).__name__})"
            continue
        
        text = kp.get("point") or kp.get("text")
        if not isinstance(text, str) or not text.strip():
            kp_id = _get_key_point_id(kp, idx)
            errors[idx] = (
                f"key_points[{idx}] (id={kp_id!r}) is missing a non-empty "
                f"'point' or 'text' field"
            )
    
    return errors


def _get_key_point_id(key_point: dict[str, Any], idx: int) -> str:
    """Extract or generate a key point ID."""
    return (
        key_point.get("key_id")
        or key_point.get("criterion_id")
        or key_point.get("id")
        or f"CRIT_{idx:03d}"
    )
