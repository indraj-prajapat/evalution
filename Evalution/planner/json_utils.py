"""JSON parsing utilities with robust extraction from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any

from Evalution.planner.logging_utils import get_logger

log = get_logger(__name__)


class JSONParseError(Exception):
    """Raised when JSON cannot be extracted or parsed from a text string."""


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON string into a Python dictionary.

    Parameters:
        text: A valid JSON string.

    Returns:
        Parsed dictionary.

    Raises:
        JSONParseError: If the string is not valid JSON.
    """
    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            raise JSONParseError(f"Expected a JSON object, got {type(result).__name__}")
        return result
    except json.JSONDecodeError as exc:
        raise JSONParseError(f"Invalid JSON: {exc}") from exc


def extract_json_from_response(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response string.

    LLMs frequently wrap their JSON in markdown code fences or prepend /
    append explanatory text.  This function handles the following patterns:

    1. `````json ... ```  `` markdown fences
    2. ````` ... ```  `` plain fences
    3. ``{ ... }``  bare JSON anywhere in the string

    Parameters:
        text: Raw LLM response text.

    Returns:
        The first parsed JSON dictionary found.

    Raises:
        JSONParseError: If no valid JSON object can be extracted.
    """
    # Strategy 1: Try to extract from markdown code fences (json or bare)
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    matches = fence_pattern.findall(text)
    for match in matches:
        match_stripped = match.strip()
        if match_stripped:
            try:
                return parse_json(match_stripped)
            except JSONParseError:
                continue

    # Strategy 2: Find the first top-level { ... } block
    brace_depth = 0
    start_idx: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                candidate = text[start_idx : i + 1]
                try:
                    return parse_json(candidate)
                except JSONParseError:
                    start_idx = None

    raise JSONParseError("No valid JSON object found in LLM response text")


def to_json_string(data: dict[str, Any] | list[Any]) -> str:
    """Serialise *data* to a compact JSON string.

    Parameters:
        data: A JSON-serialisable Python object.

    Returns:
        Minified JSON string.
    """
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))