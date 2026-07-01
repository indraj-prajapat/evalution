"""
Normalisation utilities.

Converts raw OCR text into canonical forms:
  - Dates  -> YYYY-MM-DD
  - Currency -> integer (paise dropped)
  - Booleans -> true / false
  - Whitespace -> trimmed
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Date normalisation
# ---------------------------------------------------------------------------

_INDIAN_DATE_PATTERNS = [
    # DD/MM/YYYY  or  DD-MM-YYYY  or  DD.MM.YYYY
    (re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})"), "%d/%m/%Y"),
    # DD Month YYYY  (e.g. "15 March 2023")
    (re.compile(r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"[a-z]*\s+(\d{4})", re.IGNORECASE), None),
    # Month DD, YYYY  (e.g. "March 15, 2023")
    (re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"[a-z]*\s+(\d{1,2}),?\s+(\d{4})", re.IGNORECASE), None),
    # YYYY-MM-DD (already normalised)
    (re.compile(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})"), "%Y/%m/%d"),
]

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_month_abbr(text: str) -> Optional[int]:
    return _MONTH_ABBR.get(text.lower()[:3])


def normalise_date(raw: str) -> Optional[str]:
    """Attempt to parse *raw* and return YYYY-MM-DD, or None."""
    raw = raw.strip()
    if not raw:
        return None

    # 1. Numeric separators  (DD/MM/YYYY or YYYY-MM-DD)
    for pattern, fmt in _INDIAN_DATE_PATTERNS[:1] + _INDIAN_DATE_PATTERNS[3:]:
        m = pattern.search(raw)
        if m:
            try:
                parts = [int(g) for g in m.groups()]
                if fmt.startswith("%Y"):
                    dt = datetime(parts[0], parts[1], parts[2])
                else:
                    dt = datetime(parts[2], parts[1], parts[0])
                return dt.strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                continue

    # 2. DD Month YYYY
    m = _INDIAN_DATE_PATTERNS[1][0].search(raw)
    if m:
        day = int(m.group(1))
        month = _parse_month_abbr(m.group(2))
        year = int(m.group(3))
        if month:
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                pass

    # 3. Month DD, YYYY
    m = _INDIAN_DATE_PATTERNS[2][0].search(raw)
    if m:
        month = _parse_month_abbr(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if month:
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                pass

    # 4. Already ISO-like
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw

    return None


# ---------------------------------------------------------------------------
# Currency normalisation
# ---------------------------------------------------------------------------

_CURRENCY_RE = re.compile(
    r"[₹$€£]?\s*"
    r"([\d,]+(?:\.\d{2})?)"
    r"\s*(?:crore|lac|lakh|million|billion)?",
    re.IGNORECASE,
)

_LAC_WORDS = re.compile(r"lakh|lac", re.IGNORECASE)
_CRORE_WORDS = re.compile(r"crore", re.IGNORECASE)
_MILLION_WORDS = re.compile(r"million", re.IGNORECASE)
_BILLION_WORDS = re.compile(r"billion", re.IGNORECASE)


def normalise_currency(raw: str) -> Optional[int]:
    """Convert Indian / Western currency strings to integer (base unit).

    Examples
    --------
    "₹4,56,78,900"       -> 45678900
    "Rs. 1,23,456.78"    -> 123456
    "2.5 Crore"          -> 25000000
    "15 Lakh"            -> 1500000
    """
    raw = raw.strip()
    if not raw:
        return None

    # Extract numeric part
    m_num = re.search(r"([\d,]+(?:\.\d+)?)", raw)
    if not m_num:
        return None

    num_str = m_num.group(1).replace(",", "")
    try:
        value = float(num_str)
    except ValueError:
        return None

    # Check for multipliers
    if _CRORE_WORDS.search(raw):
        value *= 10_000_000
    elif _LAC_WORDS.search(raw):
        value *= 100_000
    elif _BILLION_WORDS.search(raw):
        value *= 1_000_000_000
    elif _MILLION_WORDS.search(raw):
        value *= 1_000_000

    # Return integer (truncate paise / cents — do NOT round up)
    return int(value)


# ---------------------------------------------------------------------------
# Boolean normalisation
# ---------------------------------------------------------------------------

_TRUE_WORDS = {"yes", "true", "y", "1", "affirmative", "available",
               "present", "complied", "compliance", "submitted"}
_FALSE_WORDS = {"no", "false", "n", "0", "negative", "not available",
                "absent", "not complied", "non-compliance", "not submitted"}


def normalise_boolean(raw: str) -> Optional[bool]:
    """Map common true/false words to bool, or None."""
    raw = raw.strip().lower()
    if raw in _TRUE_WORDS:
        return True
    if raw in _FALSE_WORDS:
        return False
    # Heuristic: if the string starts with a known word
    for w in _TRUE_WORDS:
        if raw.startswith(w):
            return True
    for w in _FALSE_WORDS:
        if raw.startswith(w):
            return False
    return None


# ---------------------------------------------------------------------------
# Generic normalisation dispatcher
# ---------------------------------------------------------------------------

def normalise_value(raw: str, datatype: str) -> Tuple[Any, Optional[str]]:
    """Normalise *raw* according to *datatype*.

    Returns
    -------
    (normalised_value, normalised_date_str_or_none)
    """
    raw = raw.strip()
    if not raw:
        return ("NOT_FOUND", None)

    if datatype == "currency":
        result = normalise_currency(raw)
        if result is not None:
            return (result, None)
        return ("NOT_FOUND", None)

    if datatype == "date":
        result = normalise_date(raw)
        if result is not None:
            return (result, None)
        return ("NOT_FOUND", None)

    if datatype == "boolean":
        result = normalise_boolean(raw)
        if result is not None:
            return (result, None)
        return ("NOT_FOUND", None)

    if datatype == "integer":
        m = re.search(r"(\d+)", raw)
        if m:
            return (int(m.group(1)), None)
        return ("NOT_FOUND", None)

    if datatype == "float":
        m = re.search(r"([\d,]+(?:\.\d+)?)", raw)
        if m:
            return (float(m.group(1).replace(",", "")), None)
        return ("NOT_FOUND", None)

    # Default: string — trim whitespace
    return (raw, None)