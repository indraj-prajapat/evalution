"""
Criterion Parser - Generic criterion analysis without hardcoded domains.

Extracts structured requirement checks from ANY criterion text using
regex and simple NLP, with an optional LLM fallback for clauses the
regex tier can't confidently parse. No domain-specific keywords or
business logic beyond generic requirement patterns.

v2 changes (multi-clause support):
  - A criterion is first split into logical CLAUSES on connectives
    (AND / OR / NOT / UNLESS / EXCEPT). Each clause is then parsed
    independently through the same regex extractors that previously
    ran once over the whole string. This fixes silent loss of every
    requirement after the first when a criterion contains more than
    one condition (e.g. "turnover of Rs. 25 Lakhs AND 2 completed
    projects").
  - RequirementCheck gained three new, backward-compatible fields
    (connective, logic_group, is_exception) so callers that only read
    the original fields (description/check_type/target/...) are
    completely unaffected. ParsedCriterion.requirements is still a
    FLAT list - not a tree - so numeric_verifier.py and engine.py do
    not need to change how they iterate it.
  - Currency/amount parsing is no longer Indian-only: crore/lakh/lac
    are still supported, but $, EUR, GBP, and plain million/billion
    (without Indian suffixes) are now recognized too.
  - An optional `llm_fallback` callable can be passed into
    parse_criterion() for clauses where the regex tier can't find any
    specific requirement signal (multi-clause soup, non-English text,
    oddly worded conditions). The regex tier remains the fast default
    path and always runs first; the LLM tier is opt-in and only
    invoked per-clause when regex extraction comes up empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RequirementCheck:
    """A single checkable requirement extracted from the criterion."""
    description: str          # Human-readable description of what's needed
    check_type: str           # "existence", "threshold", "count", "comparison", "date"
    target: str = ""          # What to look for (e.g., "CISA certificate", "turnover")
    operator: str = ""        # ">=", ">", "<=", "<", "=="
    threshold_value: Optional[float] = None  # Numeric threshold if applicable
    min_count: int = 1        # Minimum count required (default 1)
    unit: str = ""            # Unit like "years", "crores", "lakhs", "%"
    currency: str = ""        # Currency code if the amount is monetary (e.g. "INR", "USD")
    # Ownership of the check:
    #   True  -> Python verifies this deterministically (numbers, comparisons,
    #            calculations, counts). Its result is AUTHORITATIVE and may drive
    #            a FAIL/REVIEW verdict.
    #   False -> Document presence / keyword / textual existence. This is the
    #            LLM's domain. Python only records it as an INFORMATIONAL fact and
    #            NEVER blocks the verdict.
    is_numeric_check: bool = False

    # --- Clause-level logic metadata (v2) -----------------------------
    # These describe how this requirement relates to the OTHER
    # requirements extracted from the same criterion. They are additive
    # and default to "no special logic" so existing flat-list consumers
    # (numeric_verifier.verify_requirement, engine.py) keep working
    # unmodified if they simply ignore the new fields.
    #
    #   connective: "" (first/only clause), "AND", "OR", "NOT",
    #               "UNLESS", "EXCEPT"
    #   logic_group: requirements sharing the same logic_group and
    #               joined by "OR" are ALTERNATIVES of each other
    #               (satisfying any one of the group should count,
    #               rather than requiring all of them). Requirements
    #               in different groups are independent AND-ed
    #               conditions. Group ids are 1-indexed per criterion.
    #   is_exception: True when the clause was introduced by
    #               "unless"/"except" - i.e. it describes a carve-out
    #               rather than a positive requirement. Downstream
    #               verifiers should treat these specially (e.g. do
    #               not silently AND them into the main requirement
    #               set) rather than assuming they are ordinary must-
    #               satisfy checks.
    connective: str = ""
    logic_group: int = 1
    is_exception: bool = False
    negated: bool = False


@dataclass
class ParsedCriterion:
    """Complete parsed result of a criterion."""
    raw_text: str = ""
    requirements: list[RequirementCheck] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)
    required_document_types: list[str] = field(default_factory=list)
    company_name_in_doc_required: bool = False
    # Cross-criterion dependency signals (v3, additive): phrases like
    # "not applicable to joint ventures" or "relaxed for MSME" describe a
    # condition that CANNOT be resolved from this criterion's own evidence
    # -- it depends on a fact established by a DIFFERENT key point
    # elsewhere in tender_output["key_points"] (e.g. a separate "bidder
    # type" or "MSME status" criterion). This engine has no visibility
    # into other key points, so these are captured here as unresolved
    # HINTS only -- see verification_engine/dependency.py for the
    # resolver that a caller with cross-key-point context can invoke.
    # Each hint: {"phrase": str, "category": str, "effect": "exempt"|"relaxed"}
    cross_reference_hints: list[dict] = field(default_factory=list)


# LLM fallback hook type: given clause text, return zero or more
# RequirementCheck objects, or None if it can't help either. The
# parser fills in connective/logic_group/is_exception on whatever the
# callback returns, so implementations only need to worry about
# check_type/target/operator/threshold_value/etc.
LLMFallback = Callable[[str], Optional[list[RequirementCheck]]]


# ---------------------------------------------------------------------------
# Number / Currency Parsing Helpers
# ---------------------------------------------------------------------------

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty-five": 25, "thirty": 30,
    "fifty": 50, "hundred": 100,
}

# Scale-word multipliers. Indian numbering (crore/lakh/lac) sits
# alongside generic/international scale words (thousand/million/
# billion) - neither is treated as the "default"; whichever appears in
# the text wins. This is what makes the parser locale-agnostic instead
# of assuming Indian tenders only.
_UNIT_MULTIPLIERS = {
    # Indian numbering system
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5,
    # Generic / international scale words
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    # Duration
    "year": 1, "years": 1,
    "month": 1 / 12, "months": 1 / 12,
}

# Currency symbol / code -> ISO-ish code used purely for labeling the
# extracted requirement (does not affect the numeric multiplier, which
# comes from the scale word above).
_CURRENCY_TOKEN_TO_CODE = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
    "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "aed": "AED", "dirham": "AED", "dirhams": "AED",
}

# Matches a currency marker either as a symbol or as a word token.
# Ordered longest-first within alternation so "Rs." doesn't get cut at "Rs".
_CURRENCY_PATTERN = r"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP|AED|[Rr]upees?|[Dd]ollars?|[Ee]uros?|[Pp]ounds?)"

# Matches a scale word/abbreviation immediately after a number.
_SCALE_UNIT_PATTERN = (
    r"(?:[Cc]rores?|[Cc]r|[Ll]akh?s?|[Ll]acs?|"
    r"[Tt]housand|[Kk]|[Mm]illion|[Mm]n|[Mm]|"
    r"[Bb]illion|[Bb]n|[Bb])"
)

# Generic financial keywords used to decide whether a bare number
# (no currency symbol, no scale word) should still be treated as a
# monetary amount rather than, say, a document count.
_FINANCIAL_KEYWORDS = (
    "turnover", "revenue", "value", "worth", "amount", "cost", "price",
    "networth", "net worth", "capital", "emd", "fee", "fees", "budget",
)

_STOP_WORDS = {
    "the", "and", "for", "should", "have", "has", "had", "with",
    "from", "been", "being", "shall", "will", "must", "may",
    "this", "that", "which", "whose", "where", "when", "what",
    "not", "but", "are", "was", "were", "is", "am", "be",
    "any", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "than", "too", "very",
    "can", "just", "into", "over", "also", "then", "once",
    "copy", "submitted", "bidder", "firm", "company", "case",
    "per", "or", "of", "in", "to", "by", "as", "an", "a",
    "its", "their", "our", "your", "my", "his", "her",
    "if", "no", "do", "does", "did", "at", "on", "so",
    "up", "out", "about", "who", "whom", "these", "those",
}

_DOC_TYPE_SUFFIX = (
    r"(?:certificate|agreement|deed|statement|sheet|ITR|form|report|return|"
    r"license|licence|registration)"
)


def _parse_count_value(text: str) -> Optional[int]:
    """Parse 'at least N' or word numbers from text."""
    text_lower = text.lower().strip()

    m = re.search(r"at least\s+(\d+)", text_lower)
    if m:
        return int(m.group(1))

    m = re.search(r"minimum\s+(?:of\s+)?(\d+)", text_lower)
    if m:
        return int(m.group(1))

    m = re.search(r"at least\s+(\w+)", text_lower)
    if m:
        word = m.group(1)
        if word in _WORD_TO_NUM:
            return _WORD_TO_NUM[word]

    m = re.search(r"(\d+)\s+(?:or more|and above)", text_lower)
    if m:
        return int(m.group(1))

    if "at least one" in text_lower:
        return 1

    return None


def _detect_currency(text: str) -> str:
    """Best-effort currency code detection for labeling purposes."""
    tl = text.lower()
    # Symbols first (unambiguous)
    for sym, code in _CURRENCY_TOKEN_TO_CODE.items():
        if len(sym) == 1 and sym in text:
            return code
    for token, code in _CURRENCY_TOKEN_TO_CODE.items():
        if len(token) > 1 and re.search(rf"\b{re.escape(token)}\b", tl):
            return code
    return ""


def _looks_monetary(text: str) -> bool:
    """Whether a clause plausibly describes a monetary amount at all."""
    if re.search(_CURRENCY_PATTERN, text):
        return True
    if re.search(_SCALE_UNIT_PATTERN + r"\b", text):
        return True
    tl = text.lower()
    return any(kw in tl for kw in _FINANCIAL_KEYWORDS)


def _parse_amount(text: str) -> Optional[tuple[float, str, str]]:
    """
    Parse a monetary/numeric amount with unit from text.
    Returns (numeric_value, unit_string, currency_code) or None.

    Locale-agnostic: recognizes Indian scale words (crore/lakh/lac)
    AND international ones (thousand/million/billion), plus currency
    symbols/codes for $, EUR, GBP, INR, AED.
    """
    if not _looks_monetary(text):
        return None

    currency = _detect_currency(text)

    patterns = [
        # Currency marker BEFORE the number: "₹25 Lakhs" / "$5 Million" / "Rs. 25 Lakhs"
        rf"{_CURRENCY_PATTERN}\s*([\d,]+(?:\.\d+)?)\s*({_SCALE_UNIT_PATTERN})?",
        # Number with scale word, currency marker AFTER (or absent): "25 Lakhs" / "5 Million USD"
        rf"([\d,]+(?:\.\d+)?)\s*({_SCALE_UNIT_PATTERN})\b",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            num_str = m.group(1).replace(",", "")
            unit = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
            try:
                amount = float(num_str)
            except (ValueError, TypeError):
                continue

            if unit:
                multiplier = _UNIT_MULTIPLIERS.get(unit.lower(), 1)
                amount *= multiplier

            if not currency:
                currency = _detect_currency(text)

            return amount, unit, currency

    # Last resort: a currency-marked number with no scale word at all,
    # e.g. "$500,000" or "Rs. 50,000".
    m = re.search(rf"{_CURRENCY_PATTERN}\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        num_str = m.group(1).replace(",", "")
        try:
            amount = float(num_str)
        except (ValueError, TypeError):
            return None
        return amount, "", currency or _detect_currency(text)

    return None


def _parse_duration(text: str) -> Optional[tuple[float, str]]:
    """Parse a duration like '5 years', '3 months'."""
    m = re.search(r"(\d+)\s+(years?|months?|days?)", text, re.IGNORECASE)
    if m:
        value = int(m.group(1))
        unit = m.group(2).lower()
        multiplier = _UNIT_MULTIPLIERS.get(unit, 1)
        return value * multiplier, unit
    return None


# ---------------------------------------------------------------------------
# Key Term Extraction
# ---------------------------------------------------------------------------

def _extract_key_terms(criterion: str) -> list[str]:
    """Extract meaningful terms from criterion for evidence filtering."""
    terms = []

    slash_groups = re.findall(r"\b([A-Z]{2,}(?:/[A-Z]{2,})+)\b", criterion)
    for group in slash_groups:
        for term in group.split("/"):
            term = term.strip()
            if len(term) >= 2:
                terms.append(term)

    for pat in [r"(?:copy of|copy|submit|enclose)\s+([^.]+?)(?:\s+shall|\s+and|\s+is|\s*$)"]:
        matches = re.findall(pat, criterion, re.IGNORECASE)
        for match in matches:
            match = match.strip().rstrip(".")
            if len(match) > 3:
                terms.append(match)

    words = re.findall(r"\b[A-Za-z][A-Za-z'-]+\b", criterion)
    for w in words:
        wl = w.lower()
        if wl not in _STOP_WORDS and len(wl) >= 3 and wl not in [t.lower() for t in terms]:
            terms.append(w)

    seen = set()
    result = []
    for t in terms:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            result.append(t)

    return result


# ---------------------------------------------------------------------------
# Document Type Extraction
# ---------------------------------------------------------------------------

def extract_document_types(criterion: str) -> list[str]:
    """
    Extract required document types from criterion text.

    Public/shared so other modules (e.g. engine.py) don't need to keep
    their own near-duplicate regex for the same job. engine.py's
    `_extract_required_doc_types` should call this instead of
    maintaining a second copy.
    """
    doc_types = []
    seen = set()

    for pat in [
        r"copy of\s+([A-Za-z][\w\s/']*?)(?:\s+shall|\s+and\s|\s+is\s|\.$)",
        rf"([A-Za-z][\w\s]*?{_DOC_TYPE_SUFFIX})\b",
    ]:
        for m in re.finditer(pat, criterion, re.IGNORECASE):
            dtype = m.group(1).strip()
            dl = dtype.lower()
            if dl not in seen and len(dl) >= 4:
                seen.add(dl)
                doc_types.append(dtype)

    return doc_types


# Backward-compatible private alias (old name some callers may still import).
_extract_document_types = extract_document_types


# ---------------------------------------------------------------------------
# Clause Splitting (AND / OR / NOT / UNLESS / EXCEPT)
# ---------------------------------------------------------------------------

@dataclass
class _Clause:
    text: str
    connective: str   # "" | "AND" | "OR" | "NOT" | "UNLESS" | "EXCEPT"
    group_id: int


_CONNECTIVE_RE = re.compile(
    r"\b(and\s*/\s*or|and|or|unless|except(?:\s+for)?|but\s+not|provided\s+that)\b",
    re.IGNORECASE,
)


def _normalize_connective(raw: str) -> str:
    r = re.sub(r"\s+", "", raw.lower())
    if r == "and/or":
        # Ambiguous by nature; treat permissively as OR so a satisfied
        # alternative isn't wrongly required alongside the other side.
        return "OR"
    if r == "and":
        return "AND"
    if r == "or":
        return "OR"
    if r == "unless":
        return "UNLESS"
    if r.startswith("except"):
        return "EXCEPT"
    if r == "butnot":
        return "NOT"
    if r == "providedthat":
        return "UNLESS"
    return "AND"


def _looks_like_requirement_fragment(fragment: str) -> bool:
    """
    Heuristic: does this fragment plausibly stand alone as its own
    requirement, or is it just a dangling word/phrase that happened to
    sit next to a connective (e.g. "duly filled AND signed")?
    """
    f = fragment.strip()
    if not f:
        return False
    if len(f) < 4:
        return False
    if re.search(r"\d", f):
        return True
    if re.search(_DOC_TYPE_SUFFIX, f, re.IGNORECASE):
        return True
    if re.search(r"\bcopy of\b", f, re.IGNORECASE):
        return True
    if re.search(r"\bat least\b", f, re.IGNORECASE):
        return True
    # A fragment with a decent number of words is probably its own
    # clause even without a number (e.g. "registered with GeM").
    return len(f.split()) >= 4


def _split_clauses(criterion: str) -> list[_Clause]:
    """
    Split a criterion into logical clauses on top-level connectives.

    This is intentionally a lightweight heuristic splitter (no
    parenthesis-aware masking, no dependency parsing) - it trades some
    precision for staying fast, dependency-free, and predictable. It
    is designed to catch the common tender-drafting pattern of two or
    three conditions strung together with "and"/"or", not to be a full
    logical-form parser. Clauses it can't confidently make sense of
    fall through to the regular per-clause extractors, and from there
    to the optional LLM fallback tier.
    """
    tokens = _CONNECTIVE_RE.split(criterion)
    if len(tokens) == 1:
        return [_Clause(text=criterion.strip(), connective="", group_id=1)]

    raw: list[tuple[str, str]] = []
    connective = ""
    buf = tokens[0]
    i = 1
    while i < len(tokens):
        conn_norm = _normalize_connective(tokens[i])
        frag = tokens[i + 1] if i + 1 < len(tokens) else ""
        raw.append((buf, connective))
        connective = conn_norm
        buf = frag
        i += 2
    raw.append((buf, connective))

    merged: list[_Clause] = []
    group_id = 0
    for text, conn in raw:
        text = text.strip(" ,.;")
        if not text:
            continue

        if merged and conn in ("AND", "OR") and not _looks_like_requirement_fragment(text):
            # Glue dangling fragments back onto the previous clause
            # instead of manufacturing a bogus standalone requirement
            # out of e.g. "signed" from "duly filled and signed".
            prev = merged[-1]
            prev.text = f"{prev.text} {conn.lower()} {text}".strip()
            continue

        if conn == "OR" and merged:
            gid = merged[-1].group_id
        else:
            group_id += 1
            gid = group_id

        merged.append(_Clause(text=text, connective=conn, group_id=gid))

    return merged if merged else [_Clause(text=criterion.strip(), connective="", group_id=1)]


# ---------------------------------------------------------------------------
# Per-Clause Requirement Extraction
# ---------------------------------------------------------------------------

_THRESHOLD_INVERSE = {">=": "<", ">": "<=", "<=": ">", "<": ">="}


def _extract_requirements_from_clause(clause_text: str) -> list[RequirementCheck]:
    """Run the regex extractors (count/amount/duration/doc-existence) on ONE clause."""
    requirements: list[RequirementCheck] = []
    clause_lower = clause_text.lower()

    # --- Count requirement ---
    count_val = _parse_count_value(clause_text)
    amount_looks_monetary = _looks_monetary(clause_text)
    if count_val is not None and not amount_looks_monetary:
        count_target_match = re.search(
            r"at least\s+(?:\w+\s+)?(?:Qualified\s+)?(.+?)(?:\s*\.|\s*\(|\s*$)",
            clause_text, re.IGNORECASE
        )
        target = count_target_match.group(1).strip() if count_target_match else "items"

        requirements.append(RequirementCheck(
            description=f"At least {count_val} {target}",
            check_type="count",
            target=target,
            min_count=count_val,
            # A count of documents/certificates/items is fundamentally a
            # "how many were found" question -> the LLM is far better at
            # this than Python's brittle unique-value counting.
            is_numeric_check=False,
        ))

    # --- Amount / threshold requirement ---
    amount_result = _parse_amount(clause_text)
    if amount_result:
        amount, unit, currency = amount_result

        # NOTE: "not exceeding"/"no more than" MUST be checked before the
        # bare "exceeding"/"more than" patterns, since the negated phrase
        # contains the positive one as a substring - checking the positive
        # pattern first would silently flip the operator to ">" on a "<"
        # requirement (this was a real bug in the original single-pass
        # version of this function).
        if "not exceeding" in clause_lower or "no more than" in clause_lower or "less than" in clause_lower:
            operator = "<"
        elif "at least" in clause_lower or "minimum" in clause_lower:
            operator = ">="
        elif "exceeding" in clause_lower or "more than" in clause_lower:
            operator = ">"
        else:
            operator = ">="

        currency_prefix = f"{currency} " if currency else ""
        target_desc = (
            f"threshold of {currency_prefix}{amount:,.2f} ({unit})"
            if unit else f"threshold of {currency_prefix}{amount:,.2f}"
        )

        past_years_match = re.search(
            r"past\s+(\d+)\s+financial\s+years", clause_lower, re.IGNORECASE
        )
        past_years_token = ""
        if past_years_match:
            past_years_token = f" [PAST_N_FINANCIAL_YEARS={int(past_years_match.group(1))}]"

        requirements.append(RequirementCheck(
            description=(
                f"Value {operator} {target_desc}. "
                f"CriterionContext={clause_text.strip()}{past_years_token}"
            ),
            check_type="threshold",
            target="financial_value",
            operator=operator,
            threshold_value=amount,
            unit=unit,
            currency=currency,
            is_numeric_check=True,
        ))

    # --- Duration requirement ---
    duration_result = _parse_duration(clause_text)
    if duration_result and not amount_result:
        # Skip if this clause was already consumed as a monetary
        # threshold containing e.g. "past 3 financial years" - that "3
        # years" is context for the amount, not a separate duration
        # requirement.
        duration, dur_unit = duration_result
        requirements.append(RequirementCheck(
            description=f"Duration of at least {duration:.0f} {dur_unit}",
            check_type="threshold",
            target="duration",
            operator=">=",
            threshold_value=duration,
            unit=dur_unit,
            is_numeric_check=True,
        ))

    # --- Document existence requirements ---
    for dtype in extract_document_types(clause_text):
        requirements.append(RequirementCheck(
            description=f"Document '{dtype}' must be submitted/available",
            check_type="existence",
            target=dtype,
        ))

    return requirements


# ---------------------------------------------------------------------------
# Cross-Reference Hint Detection (v3, additive)
# ---------------------------------------------------------------------------
# Distinct from the UNLESS/EXCEPT clause splitting above: those describe
# carve-outs checkable from THIS criterion's own evidence (e.g. "unless a
# waiver certificate is submitted"). A cross-reference hint instead names a
# BIDDER CATEGORY (joint venture, MSME, startup, ...) whose applicability is
# a fact that lives on some OTHER key point, not in this criterion's text or
# evidence at all. Detection here is deliberately generic (trigger phrase +
# free-text category), not a fixed list of category keywords, so it isn't
# tied to any one tender's domain vocabulary.

_CROSS_REF_TRIGGER_PATTERN = re.compile(
    r"(not applicable to|does not apply to|shall not apply to|"
    r"not applicable for|exempt(?:ed)?\s+(?:if|for|from|when)|"
    r"relax(?:ed|ation)\s+(?:for|is available for|applicable for)|"
    r"waived?\s+for|not required for|not mandatory for|"
    r"is not applicable in case of)\s+((?:\S+\s+){0,4}\S+)",
    re.IGNORECASE,
)

_RELAXATION_TRIGGERS = ("relax", "waive")


def _detect_cross_reference_hints(criterion_text: str) -> list[dict]:
    """
    Find phrases that make this criterion's applicability conditional on a
    fact belonging to a DIFFERENT key point (bidder category / status),
    e.g. "not applicable to joint ventures" or "relaxed for MSME bidders".

    Returns a list of hint dicts (possibly empty). This function only
    DETECTS the signal -- it never tries to resolve whether the condition
    is actually true, since that requires data (other key points' results)
    this module has no access to. See dependency.py for resolution.
    """
    hints: list[dict] = []
    for match in _CROSS_REF_TRIGGER_PATTERN.finditer(criterion_text):
        trigger_phrase = match.group(1).strip()
        category = match.group(2).strip().rstrip(".,;:")
        if not category:
            continue
        effect = "relaxed" if any(t in trigger_phrase.lower() for t in _RELAXATION_TRIGGERS) else "exempt"
        hints.append({
            "phrase": trigger_phrase,
            "category": category,
            "effect": effect,
        })
    return hints


def _clause_has_specific_signal(requirements: list[RequirementCheck]) -> bool:
    """True if regex extraction found anything more specific than a generic catch-all."""
    return len(requirements) > 0


# ---------------------------------------------------------------------------
# Main Parser
# ---------------------------------------------------------------------------

def parse_criterion(
    criterion_text: str,
    llm_fallback: Optional[LLMFallback] = None,
) -> ParsedCriterion:
    """
    Parse a tender criterion into structured, checkable requirements.

    This is fully generic - no domain-specific keywords or logic. It
    works for ANY criterion text, in any currency/locale, and now
    handles criteria that bundle multiple conditions together with
    AND / OR / NOT / UNLESS / EXCEPT.

    Args:
        criterion_text: The raw criterion string.
        llm_fallback: Optional callable(clause_text) -> list[RequirementCheck]
            invoked ONLY for individual clauses where the fast regex
            tier found nothing specific to extract (i.e. it would
            otherwise fall back to a generic whole-clause existence
            check). Lets a caller wire in an LLM-based extractor for
            multi-clause soup, non-English text, or oddly worded
            conditions without slowing down the common case. If it
            returns None or an empty list, the regex tier's generic
            fallback is used as before.
    """
    result = ParsedCriterion(raw_text=criterion_text)
    criterion = criterion_text.strip()
    criterion_lower = criterion.lower()

    # Key terms / doc types / company-name flag are computed over the
    # whole criterion - these are evidence-filtering aids, not
    # requirement checks, so clause-splitting them buys nothing.
    result.key_terms = _extract_key_terms(criterion)
    result.required_document_types = extract_document_types(criterion)
    result.cross_reference_hints = _detect_cross_reference_hints(criterion)

    if any(phrase in criterion_lower for phrase in [
        "copy of", "shall be submitted", "submitted by the bidder",
        "enclose", "enclosed", "attached",
    ]):
        result.company_name_in_doc_required = True

    clauses = _split_clauses(criterion)

    for clause in clauses:
        clause_requirements = _extract_requirements_from_clause(clause.text)

        if not _clause_has_specific_signal(clause_requirements):
            # Nothing specific found via regex. Try the optional LLM
            # fallback tier before giving up and emitting a generic
            # existence check for the whole clause.
            llm_requirements: Optional[list[RequirementCheck]] = None
            if llm_fallback is not None:
                try:
                    llm_requirements = llm_fallback(clause.text)
                except Exception:
                    # Fallback tier is best-effort; never let it break
                    # the deterministic regex path.
                    llm_requirements = None

            if llm_requirements:
                clause_requirements = llm_requirements
            else:
                clause_requirements = [RequirementCheck(
                    description=clause.text,
                    check_type="existence",
                    target=clause.text[:100],
                )]

        # Tag every requirement produced by this clause with the
        # clause's logic metadata.
        is_exception = clause.connective in ("UNLESS", "EXCEPT")
        is_negated = clause.connective == "NOT"
        for req in clause_requirements:
            req.connective = clause.connective
            req.logic_group = clause.group_id
            req.is_exception = is_exception
            if is_negated:
                req.negated = True
                if req.check_type == "threshold" and req.operator in _THRESHOLD_INVERSE:
                    req.operator = _THRESHOLD_INVERSE[req.operator]

        result.requirements.extend(clause_requirements)

    # --- If nothing at all was parsed (shouldn't normally happen once
    # clauses always fall back to a generic existence check, but kept
    # as a final safety net), create one whole-criterion requirement. ---
    if not result.requirements:
        result.requirements.append(RequirementCheck(
            description=criterion,
            check_type="existence",
            target=criterion[:100],
        ))

    return result