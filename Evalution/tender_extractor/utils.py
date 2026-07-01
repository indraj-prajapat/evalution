"""
Utility helpers used across the extraction pipeline.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def truncate_snippet(text: str, max_length: int = 300) -> str:
    """Return a trimmed snippet for evidence, preserving word boundaries."""
    text = clean_text(text)
    if len(text) <= max_length:
        return text
    # Trim at last space before limit
    cut = text[:max_length].rfind(" ")
    if cut == -1:
        cut = max_length
    return text[:cut] + "..."


# ---------------------------------------------------------------------------
# Fuzzy matching (lightweight, no external dependency)
# ---------------------------------------------------------------------------

def tokenise(text: str) -> set[str]:
    """Lower-case alpha-numeric token set."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_overlap(a: str, b: str) -> float:
    """Jaccard-like overlap between two strings (0-1)."""
    ta, tb = tokenise(a), tokenise(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def contains_any(text: str, keywords: list[str]) -> bool:
    """Return True if *text* (lowered) contains any keyword (lowered)."""
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def contains_all(text: str, keywords: list[str]) -> bool:
    """Return True if *text* (lowered) contains ALL keywords (lowered)."""
    lower = text.lower()
    return all(kw.lower() in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------

def extract_pan(text: str) -> Optional[str]:
    """Find a PAN pattern (AAAAA9999A)."""
    m = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text.upper())
    return m.group(0) if m else None


def extract_gstin(text: str) -> Optional[str]:
    """Find a GSTIN pattern (22AAAAA0000A1Z5)."""
    m = re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]\b", text.upper())
    return m.group(0) if m else None


def extract_cin(text: str) -> Optional[str]:
    """Find a CIN pattern (L12345MH2020PTC123456)."""
    m = re.search(r"\b[A-Z][0-9]{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b", text.upper())
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Document name matching
# ---------------------------------------------------------------------------

def doc_name_matches(
    doc_name: str,
    target_names: list[str],
    threshold: float = 0.5,
) -> bool:
    """Check whether *doc_name* is similar enough to any *target_names*.

    Uses exact match first, then token overlap.
    Substring check only applies when the shorter string is >= 4 chars
    to avoid matching generic words like "Certificate" inside longer names.
    """
    dn_lower = doc_name.lower().strip()
    for target in target_names:
        t_lower = target.lower().strip()
        if t_lower == dn_lower:
            return True
        # Substring check: only if the shorter string is >= 4 chars
        # and the longer string STARTS with it (prefix match)
        # This avoids "certificate" matching inside "gst registration certificate"
        shorter, longer = (t_lower, dn_lower) if len(t_lower) <= len(dn_lower) else (dn_lower, t_lower)
        if len(shorter) >= 4 and longer.startswith(shorter):
            return True
        if token_overlap(doc_name, target) >= threshold:
            return True
    return False


def doc_category_matches(
    doc_category: str,
    target_categories: list[str],
    threshold: float = 0.4,
) -> bool:
    """Check whether *doc_category* falls under any *target_categories*."""
    return doc_name_matches(doc_category, target_categories, threshold)