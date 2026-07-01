"""
Data models for the Verification Engine.

All input/output structures are defined here using plain Python dataclasses
(no external dependencies required).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class FactType(str, Enum):
    NUMERIC = "NUMERIC"
    COMPARISON = "COMPARISON"
    COUNT = "COUNT"
    EXISTENCE = "EXISTENCE"
    DATE = "DATE"
    THRESHOLD = "THRESHOLD"
    DUPLICATE = "DUPLICATE"
    ENTITY_MATCH = "ENTITY_MATCH"
    LOGICAL = "LOGICAL"
    FINANCIAL_YEAR = "FINANCIAL_YEAR"
    MISSING_INFO = "MISSING_INFO"
    CONFLICT = "CONFLICT"


class FactStatus(str, Enum):
    VERIFIED = "VERIFIED"
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNVERIFIED = "UNVERIFIED"
    CONFLICTING = "CONFLICTING"
    NOT_FOUND = "NOT_FOUND"


class DocumentStatus(str, Enum):
    FOUND = "FOUND"
    PARTIAL = "PARTIAL"
    NOT_FOUND = "NOT_FOUND"


# ---------------------------------------------------------------------------
# Input Models (Module 2 Output)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedField:
    """A single field extracted from a document by Module 2."""
    name: str
    value: str
    datatype: str
    page: int
    snippet: str
    confidence: float
    raw_value: str


@dataclass
class ExtractedRecord:
    """A record (entity instance) from a matched document."""
    entity_id: str
    group: Optional[str]
    group_value: Optional[str]
    fields: list[ExtractedField] = field(default_factory=list)

    def get_field(self, name: str) -> Optional[ExtractedField]:
        """Get first field matching name."""
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def get_all_fields(self, name: str) -> list[ExtractedField]:
        """Get all fields matching name."""
        return [f for f in self.fields if f.name == name]

    def get_field_value(self, name: str) -> Optional[str]:
        """Get value of first field matching name."""
        f = self.get_field(name)
        return f.value if f else None


@dataclass
class MatchedDocument:
    """A document matched by Module 2 for a requirement."""
    document_id: str
    document_name: str
    status: str  # FOUND, PARTIAL, NOT_FOUND
    pages: list[int] = field(default_factory=list)
    summary: str = ""
    records: list[ExtractedRecord] = field(default_factory=list)

    @property
    def is_found(self) -> bool:
        return self.status.upper() == "FOUND"

    @property
    def is_partial(self) -> bool:
        return self.status.upper() == "PARTIAL"

    def get_all_values_for_field(self, field_name: str) -> list[str]:
        """Collect all values for a given field name across all records."""
        values = []
        for rec in self.records:
            for f in rec.fields:
                if f.name == field_name:
                    values.append(f.value)
        return values


@dataclass
class DocumentRequirement:
    """A single document requirement from the tender."""
    requirement_document: str
    mode: str  # EXPLICIT, IMPLICIT
    entity_type: str
    matched_documents: list[MatchedDocument] = field(default_factory=list)


@dataclass
class Module2Output:
    """Complete Module 2 extraction output."""
    documents: list[DocumentRequirement] = field(default_factory=list)
    criterion_id: str = ""
    criterion: str = ""


@dataclass
class CompanyDocument:
    """A document from the Company JSON (ground truth)."""
    doc_id: str
    doc_type: str
    document_name: str
    pages: list[int] = field(default_factory=list)
    page_span: dict = field(default_factory=dict)
    entities: dict = field(default_factory=dict)
    summary: str = ""
    confidence: dict = field(default_factory=dict)


@dataclass
class CompanyPageText:
    """Raw page text from Company JSON."""
    page_key: str
    text: str


@dataclass
class CompanyJSON:
    """Complete Company JSON structure."""
    file_name: str = ""
    total_pages: int = 0
    documents: list[CompanyDocument] = field(default_factory=list)
    page_texts: list[CompanyPageText] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output Models (Verified Facts + Verification Report)
# ---------------------------------------------------------------------------

@dataclass
class VerifiedFact:
    """A single verified fact produced by Python deterministic computation."""
    id: str
    statement: str
    type: str  # FactType value
    status: str  # FactStatus value
    computed_by: str = "python"
    source_fields: list[str] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class Evidence:
    """A piece of supporting evidence with page-level verification."""
    document_name: str
    page: int
    text: str                          # Exact text from the document page
    document_id: str = ""
    llm_finding: str = ""              # LLM's assessment of what this shows
    summary: str = ""                  # Brief summary: what was used and why
    page_text_verified: bool = False   # Whether we verified against actual page text
    page_text_match: str = ""          # Exact matching text from Company JSON page


@dataclass
class GroundTruthVerification:
    """Result of ground truth search in Company JSON."""
    completed: bool = False
    additional_documents_checked: list[str] = field(default_factory=list)
    contradictions_found: bool = False
    contradiction_details: list[str] = field(default_factory=list)
    # NEW: split contradictions into relevant vs irrelevant for verdict
    relevant_contradiction_details: list[str] = field(default_factory=list)
    irrelevant_contradiction_details: list[str] = field(default_factory=list)
    relevant_contradictions_found: bool = False
    irrelevant_contradictions_found: bool = False
    supporting_evidence_found: bool = False
    missing_evidence: list[str] = field(default_factory=list)
    linked_entities: list[str] = field(default_factory=list)


@dataclass
class LLMEvaluation:
    """GPT-4o-mini's evaluation of the verification report."""
    verdict: str = "REVIEW"
    confidence: float = 0.0
    reasoning: str = ""
    key_findings: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    requirement_analysis: dict = field(default_factory=dict)
    python_verdict_agreement: str = ""
    model_used: str = "gpt-4o-mini"
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_findings": self.key_findings,
            "risks": self.risks,
            "recommendations": self.recommendations,
            "requirement_analysis": self.requirement_analysis,
            "python_verdict_agreement": self.python_verdict_agreement,
            "model_used": self.model_used,
        }


@dataclass
class VerificationReport:
    """Complete verification report - the ONLY output of this engine."""
    criterion_id: str = ""
    criterion: str = ""
    company_name: str = ""
    verdict: str = Verdict.REVIEW.value
    python_verdict: str = ""  # Python's preliminary verdict before LLM
    # Records how `verdict` was decided: "python_authoritative" when the
    # final verdict is Python's own determination (LLM agreed, was rejected,
    # or wasn't used), or "llm_adjusted" when the LLM's document/text
    # judgment changed the outcome (e.g. a FAIL->REVIEW caveat, or a
    # PASS->REVIEW/FAIL adjustment). Lets downstream consumers tell the
    # difference without re-deriving the transition policy themselves.
    verdict_source: str = ""
    summary: str = ""
    reason: list[str] = field(default_factory=list)
    verified_facts: list[VerifiedFact] = field(default_factory=list)
    ground_truth_verification: GroundTruthVerification = field(default_factory=GroundTruthVerification)
    evidence: list[Evidence] = field(default_factory=list)  # ALL extracted evidence
    verdict_evidence: list[Evidence] = field(default_factory=list)  # ONLY evidence used in verdict
    missing_information: list[str] = field(default_factory=list)
    # NEW: point-to-point criterion breakdown
    criterion_requirements: list[dict] = field(default_factory=list)
    # NEW: informational notes (things observed but not affecting verdict)
    informational_notes: list[str] = field(default_factory=list)
    # NEW (v3): notes from cross-criterion dependency resolution -- see
    # dependency.py. Populated only when the criterion contained a
    # conditional exempt/relaxed clause referencing another key point.
    # UNLIKE informational_notes, these CAN affect the verdict (forced to
    # REVIEW when triggered or unresolved -- see engine.py::run()).
    dependency_notes: list[str] = field(default_factory=list)
    llm_evaluation: Optional[LLMEvaluation] = None

    def to_dict(self) -> dict:
        result = {
            "criterion_id": self.criterion_id,
            "criterion": self.criterion,
            "company_name": self.company_name,
            "verdict": self.verdict,
            "python_verdict": self.python_verdict or self.verdict,
            "verdict_source": self.verdict_source or "python_authoritative",
            "summary": self.summary,
            "reason": self.reason,
            "verified_facts": [
                {
                    "id": f.id,
                    "statement": f.statement,
                    "type": f.type,
                    "status": f.status,
                    "computed_by": f.computed_by,
                    "source_fields": f.source_fields,
                    "source_documents": f.source_documents,
                    "details": f.details,
                }
                for f in self.verified_facts
            ],
            "ground_truth_verification": {
                "completed": self.ground_truth_verification.completed,
                "additional_documents_checked": self.ground_truth_verification.additional_documents_checked,
                "contradictions_found": self.ground_truth_verification.contradictions_found,
                "contradiction_details": self.ground_truth_verification.contradiction_details,
                "relevant_contradictions_found": self.ground_truth_verification.relevant_contradictions_found,
                "relevant_contradiction_details": self.ground_truth_verification.relevant_contradiction_details,
                "irrelevant_contradictions_found": self.ground_truth_verification.irrelevant_contradictions_found,
                "irrelevant_contradiction_details": self.ground_truth_verification.irrelevant_contradiction_details,
                "supporting_evidence_found": self.ground_truth_verification.supporting_evidence_found,
                "missing_evidence": self.ground_truth_verification.missing_evidence,
                "linked_entities": self.ground_truth_verification.linked_entities,
            },
            "evidence": [
                {
                    "document_name": e.document_name,
                    "document_id": e.document_id,
                    "page": e.page,
                    "text": e.text,
                    "llm_finding": e.llm_finding,
                    "summary": e.summary,
                    "page_text_verified": e.page_text_verified,
                    "page_text_match": e.page_text_match,
                }
                for e in self.evidence
            ],
            "verdict_evidence": [
                {
                    "document_name": e.document_name,
                    "document_id": e.document_id,
                    "page": e.page,
                    "text": e.text,
                    "llm_finding": e.llm_finding,
                    "summary": e.summary,
                    "page_text_verified": e.page_text_verified,
                    "page_text_match": e.page_text_match,
                }
                for e in self.verdict_evidence
            ],
            "missing_information": self.missing_information,
            "criterion_requirements": self.criterion_requirements,
            "informational_notes": self.informational_notes,
            "dependency_notes": self.dependency_notes,
        }
        if self.llm_evaluation:
            result["llm_evaluation"] = self.llm_evaluation.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Parsing Utilities
# ---------------------------------------------------------------------------

def parse_module2_output(data: dict) -> Module2Output:
    """Parse raw dict into Module2Output structure."""
    docs = []
    for d in data.get("documents", []):
        matched = []
        for md in d.get("matched_documents", []):
            records = []
            for r in md.get("records", []):
                fields = [
                    ExtractedField(
                        name=f["name"],
                        value=f["value"],
                        datatype=f.get("datatype", "string"),
                        page=f.get("page", 0),
                        snippet=f.get("snippet", ""),
                        confidence=f.get("confidence", 0.0),
                        raw_value=f.get("raw_value", f["value"]),
                    )
                    for f in r.get("fields", [])
                ]
                records.append(
                    ExtractedRecord(
                        entity_id=r.get("entity_id", ""),
                        group=r.get("group"),
                        group_value=r.get("group_value"),
                        fields=fields,
                    )
                )
            matched.append(
                MatchedDocument(
                    document_id=md.get("document_id", ""),
                    document_name=md.get("document_name", ""),
                    status=md.get("status", "NOT_FOUND"),
                    pages=md.get("pages", []),
                    summary=md.get("summary", ""),
                    records=records,
                )
            )
        docs.append(
            DocumentRequirement(
                requirement_document=d.get("requirement_document", ""),
                mode=d.get("mode", "EXPLICIT"),
                entity_type=d.get("entity_type", ""),
                matched_documents=matched,
            )
        )
    return Module2Output(
        documents=docs,
        criterion_id=data.get("criterion_id", ""),
        criterion=data.get("criterion", ""),
    )


def parse_company_json(data: dict) -> CompanyJSON:
    """Parse raw Company JSON dict into CompanyJSON structure."""
    # Company JSON can have multiple top-level keys (one per source file)
    documents = []
    page_texts = []
    file_name = ""
    total_pages = 0

    for source_key, source_val in data.items():
        if not isinstance(source_val, dict):
            continue

        file_name = source_val.get("file_name", source_key)
        total_pages = source_val.get("summary", {}).get("total_pages", 0)

        # Parse structured documents
        docs_section = source_val.get("documents", {})
        if isinstance(docs_section, dict):
            total = docs_section.get("total_pages", 0)
            if total:
                total_pages = total
            for doc in docs_section.get("documents", []):
                documents.append(
                    CompanyDocument(
                        doc_id=doc.get("doc_id", ""),
                        doc_type=doc.get("doc_type", ""),
                        document_name=doc.get("document_name", ""),
                        pages=doc.get("pages", []),
                        page_span=doc.get("page_span", {}),
                        entities=doc.get("entities", {}),
                        summary=doc.get("summary", ""),
                        confidence=doc.get("confidence", {}),
                    )
                )

        # Parse page-level text
        for key, val in source_val.items():
            if key.startswith("page_") and isinstance(val, str):
                page_texts.append(
                    CompanyPageText(
                        page_key=key,
                        text=val,
                    )
                )

    return CompanyJSON(
        file_name=file_name,
        total_pages=total_pages,
        documents=documents,
        page_texts=page_texts,
        raw=data,
    )


# ---------------------------------------------------------------------------
# Numeric / Date Parsing Utilities
# ---------------------------------------------------------------------------

# Indian number system: ₹1,23,456.78  or  123456.78 or 1.5 Crore
#
# BUGFIX (mid-word unit matching): the unit alternation is ordered longest-first
# (multi-character words before single letters), and a trailing negative
# lookahead `(?![a-zA-Z])` ensures a single-letter unit (L/M/B/K) is only
# accepted when it is NOT immediately followed by more letters. This prevents
# incorrect matches such as "1234 Litres" -> "L" (=Lakh) or
# "500 Metric tons" -> "M" (=Million).
_CURRENCY_RE = re.compile(
    r"[₹Rs.\s]*"
    r"([\d,]+(?:\.\d+)?)"                                   # main number
    r"\s*(Crore|Lakh|Lac|Thousand|Million|Billion|Cr|[LMBK])?"  # optional unit
    r"(?![a-zA-Z])",                                        # not followed by more letters
    re.IGNORECASE,
)

_CURRENCY_MULTIPLIERS = {
    "crore": 1e7, "cr": 1e7,
    "lakh": 1e5, "lac": 1e5, "l": 1e5,
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6,
    "billion": 1e9, "b": 1e9,
}

# Formats that are unambiguous regardless of locale convention: the year
# is either 4-digit-and-first (ISO order) or the month is spelled out, so
# there is only one possible reading. These are always tried first and
# never require a day-first/month-first guess.
_UNAMBIGUOUS_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d",
    "%d %b %Y", "%d %B %Y", "%B %d, %Y",
    "%d-%b-%Y", "%d-%B-%Y",
]

# Purely numeric "NN<sep>NN<sep>YYYY" formats where, taken in isolation,
# either component could plausibly be the day or the month. Resolved via
# _resolve_numeric_date() rather than by trying these in a fixed order,
# since trying them in order silently picks whichever convention happens
# to come first in the list.
_AMBIGUOUS_NUMERIC_DATE_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$")


def parse_currency(value: str) -> Optional[float]:
    """
    Parse a currency string into a float.
    Handles: ₹1,23,456.78, 1.5 Crore, 12.34 Lakh, 1234567, etc.
    Returns None if the value is missing/unparseable, and 0.0 for an
    explicit zero value (e.g. "0", "0.0", "₹0").
    """
    if not value or not isinstance(value, str):
        return None
    value = value.strip()

    # BUGFIX: "0" must NOT be treated as a missing/NA value - it is a valid
    # amount and should be returned as 0.0, not None. Only genuinely
    # "no value" markers are excluded here.
    if value in ("", "NA", "N/A", "nil", "NIL", "-"):
        return None

    # Explicit zero handling (covers "0", "0.0", "₹0", "Rs. 0.00", etc.)
    stripped_numeric = value.replace("₹", "").replace("Rs.", "").replace("Rs", "").strip()
    try:
        if float(stripped_numeric.replace(",", "")) == 0.0:
            return 0.0
    except (ValueError, TypeError):
        pass

    match = _CURRENCY_RE.search(value)
    if not match:
        # Try plain float
        try:
            return float(value.replace(",", ""))
        except (ValueError, TypeError):
            return None

    num_str = match.group(1).replace(",", "")
    unit = match.group(2)

    try:
        amount = float(num_str)
    except (ValueError, TypeError):
        return None

    if unit:
        multiplier = _CURRENCY_MULTIPLIERS.get(unit.lower())
        if multiplier:
            amount *= multiplier

    return amount


def parse_number(value: str) -> Optional[float]:
    """Parse a plain number string (no currency unit)."""
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if value in ("", "NA", "N/A", "nil", "NIL"):
        return None
    try:
        return float(value.replace(",", ""))
    except (ValueError, TypeError):
        return None


def _resolve_numeric_date(
    comp1: str, comp2: str, year_str: str, day_first: Optional[bool]
) -> Optional[str]:
    """
    Resolve a numeric date from two leading components (comp1, comp2, in
    the order they appeared in the source text) plus a year, where it is
    not known a priori whether comp1 or comp2 is the day.

    - If only one of the two readings (comp1=day/comp2=month vs.
      comp2=day/comp1=month) is a valid calendar date, that reading is
      used regardless of `day_first` - it isn't actually ambiguous, only
      the raw format is (e.g. "25/03/2023" can only be 25 March, since
      month 25 doesn't exist).
    - If both readings are valid calendar dates, the date is genuinely
      ambiguous:
        * day_first=True  -> comp1 is treated as the day
        * day_first=False -> comp1 is treated as the month
        * day_first=None  -> refuse to guess; returns None so the caller
          can surface this as a low-confidence/unverified result instead
          of silently assuming a convention.
    """
    try:
        n1, n2, year_n = int(comp1), int(comp2), int(year_str)
    except (TypeError, ValueError):
        return None
    if year_n < 100:
        year_n += 2000

    def build(day: int, month: int) -> Optional[str]:
        try:
            return datetime(year_n, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    reading_day_first = build(n1, n2)      # comp1 = day, comp2 = month
    reading_month_first = build(n2, n1)    # comp2 = day, comp1 = month

    if reading_day_first and not reading_month_first:
        return reading_day_first
    if reading_month_first and not reading_day_first:
        return reading_month_first
    if reading_day_first and reading_month_first:
        if day_first is True:
            return reading_day_first
        if day_first is False:
            return reading_month_first
        return None  # genuinely ambiguous, day_first=None -> don't guess
    return None


def parse_date(value: str, day_first: Optional[bool] = True) -> Optional[str]:
    """
    Parse a date string into ISO format (YYYY-MM-DD).
    Returns None if parsing fails, OR if the date is a purely numeric
    "NN/NN/YYYY"-style string that is genuinely ambiguous (both leading
    components are <=12, so either day-first or month-first is a valid
    reading) and `day_first` is None.

    `day_first` controls how such genuinely ambiguous numeric dates are
    resolved:
      - True (default): assume day-first (DD/MM), matching the
        convention used in most non-US documents, including Indian
        tender documents - the primary source this parser was built for.
      - False: assume month-first (MM/DD), e.g. for US-style documents.
      - None: never guess on genuinely ambiguous input; return None
        instead so the caller can treat the result as unverified rather
        than silently picking a convention that may be wrong.

    Unambiguous dates (a component >12, a spelled-out month, or ISO
    year-first order) are always parsed correctly and are unaffected by
    `day_first`.
    """
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if value in ("", "NA", "N/A", "nil", "NIL"):
        return None

    # Indian documents overwhelmingly write dates with ordinal suffixes
    # ("31st March 2023", "1st April 2022") which none of the strptime
    # formats below can parse as-is. Strip them up front so both the exact
    # -format pass and the textual-date regex fallback benefit, instead of
    # every caller needing to know to strip ordinals first.
    cleaned = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", value, flags=re.IGNORECASE)

    for candidate in (value, cleaned):
        # Unambiguous formats (ISO order or spelled-out month) first -
        # these never require a day-first/month-first guess.
        for fmt in _UNAMBIGUOUS_DATE_FORMATS:
            try:
                dt = datetime.strptime(candidate, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Purely numeric "NN<sep>NN<sep>YYYY" occupying the whole string -
        # resolved via the ambiguity-aware helper rather than by trying
        # "%d-%m-%Y" / "%m/%d/%Y" in a fixed order (which would silently
        # prefer whichever came first in the list).
        m = _AMBIGUOUS_NUMERIC_DATE_RE.match(candidate)
        if m:
            resolved = _resolve_numeric_date(m.group(1), m.group(2), m.group(3), day_first)
            if resolved:
                return resolved

    # Try extracting date-like patterns embedded in a larger string (e.g.
    # "Balance Sheet as at 31 March 2023" rather than a bare date string).
    # ISO-order (YYYY first) is checked first since it is unambiguous;
    # the DD/MM-or-MM/DD pattern is genuinely ambiguous and goes through
    # the same resolver as above.
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", cleaned)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", cleaned)
    if m:
        resolved = _resolve_numeric_date(m.group(1), m.group(2), m.group(3), day_first)
        if resolved:
            return resolved

    # Textual "31 March 2023" / "March 31, 2023" embedded anywhere in a
    # sentence. Without this, any date not sitting alone as the entire
    # string (the common case in document names/snippets) is unparseable.
    textual_patterns = [
        r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b",   # 31 March 2023
        r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b",  # March 31, 2023
    ]
    for pat in textual_patterns:
        m = re.search(pat, cleaned)
        if not m:
            continue
        g = m.groups()
        day_str, month_str, year_str = (g[0], g[1], g[2]) if g[0].isdigit() else (g[1], g[0], g[2])
        for month_fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                dt = datetime.strptime(f"{day_str} {month_str} {year_str}", month_fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

    return None


def financial_year_from_date(
    value: str,
    fiscal_year_start_month: int = 4,
    day_first: Optional[bool] = True,
) -> Optional[str]:
    """
    Derive a financial year label from ANY parseable calendar date
    string, e.g. with the default April-March (Indian) convention:
    '31st March 2023' -> 'FY2022-23', '25 September 2023' -> 'FY2023-24'.

    This is a FALLBACK for evidence that carries a real date (balance
    sheet "as at" date, certificate/report date, etc.) but no explicit
    "FY2022-23" style string anywhere in the text. Without it, a document
    named e.g. "Balance Sheet as at 31st March 2023" can never be mapped
    to a financial year at all, even though the date fully determines it.

    `fiscal_year_start_month` is the calendar month (1-12) the fiscal
    year begins on. Defaults to 4 (April, the Indian convention) but is
    overridable per-caller/per-tender rather than being a fixed literal,
    since not every jurisdiction or company uses an April-March year.

    `day_first` is forwarded to parse_date() to control how an ambiguous
    numeric date embedded in `value` is interpreted; see parse_date()'s
    docstring. Passing day_first=None avoids guessing on truly ambiguous
    dates, at the cost of returning None more often.

    Returns None if no date can be confidently parsed from `value`.
    """
    if not (1 <= fiscal_year_start_month <= 12):
        raise ValueError("fiscal_year_start_month must be between 1 and 12")

    iso = parse_date(value, day_first=day_first)
    if not iso:
        return None
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d")
    except ValueError:
        return None
    start_year = dt.year if dt.month >= fiscal_year_start_month else dt.year - 1
    return f"FY{start_year}-{str(start_year + 1)[-2:]}"


def resolve_financial_year(
    *texts: Optional[str],
    fiscal_year_start_month: int = 4,
    day_first: Optional[bool] = True,
) -> Optional[str]:
    """
    Best-effort financial year resolution across several candidate texts
    (e.g. field name, value, source document name, snippet, explicit
    period tag).

    DESIGN: financial-year evidence is frequently split across several
    pieces of context that live on different objects (the number is in
    `value`, the year is in the document name, the exact date is in a
    snippet) rather than all being present in one string. Any single
    caller that only inspects one of those strings will systematically
    fail to map perfectly good evidence to a year. This function is the
    single place that combines all available context, so every caller
    gets the same (correct) answer instead of each reimplementing a
    partial version of this search.

    `fiscal_year_start_month` and `day_first` are forwarded to
    financial_year_from_date() (see its docstring) so callers can adapt
    this to non-Indian fiscal-year conventions or explicit US-style
    dates without duplicating this search logic. Both default to the
    settings that match this module's original Indian-tender use case,
    so existing callers keep their current behavior unless they opt in
    to overriding it.

    Priority order:
      1. An explicit "FY2022-23" / "2022-23" pattern in any text (most
         reliable - it's literally stated).
      2. A calendar date found in any text, converted to the financial
         year it falls in (less reliable but still concrete).
    """
    candidates = [t for t in texts if t]
    for t in candidates:
        fy = parse_financial_year(t)
        if fy:
            return fy
    for t in candidates:
        fy = financial_year_from_date(
            t,
            fiscal_year_start_month=fiscal_year_start_month,
            day_first=day_first,
        )
        if fy:
            return fy
    return None


def parse_financial_year(value: str) -> Optional[str]:
    """
    Parse a financial year string like '2021-22', 'FY2021-22', '2021-2022',
    or the same embedded in a larger string (e.g. 'Turnover FY2021-22')
    into normalized 'FY2021-22' format.
    """
    if not value:
        return None
    value = value.strip()

    # BUGFIX: use re.search (not re.match/anchored) so embedded values like
    # "Turnover FY2021-22" are found, not just full-string matches.
    # BUGFIX: added re.IGNORECASE so lowercase "fy2021-22" is also recognized
    # and correctly normalized to uppercase "FY2021-22" (previously an
    # already-lowercase input would fail the "already normalized" check and
    # then be re-parsed and returned inconsistently).

    # Case 1: "FY2021-22" / "fy2021-22" (already in, or close to, normalized form)
    m = re.search(r"FY\s*(\d{4})-(\d{2})\b", value, re.IGNORECASE)
    if m:
        return f"FY{m.group(1)}-{m.group(2)}"

    # Case 2: "FY 2021-2022" / "fy2021-2022" (FY prefix, 4-digit end year)
    m = re.search(r"FY\s*(\d{4})-(\d{4})\b", value, re.IGNORECASE)
    if m:
        start = m.group(1)
        end = m.group(2)[-2:]
        return f"FY{start}-{end}"

    # Case 3: bare "2021-22" or "2021-2022" (no FY prefix), anywhere in the string
    m = re.search(r"\b(\d{4})-(\d{2,4})\b", value)
    if m:
        start = m.group(1)
        end_raw = m.group(2)
        end = end_raw if len(end_raw) == 2 else end_raw[-2:]
        return f"FY{start}-{end}"

    return None