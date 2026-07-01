"""Internal models, constants, and lookup tables used across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Supported document categories (canonical list)
# ---------------------------------------------------------------------------

VALID_CATEGORIES: list[str] = [
    "Financial Documents",
    "Experience Documents",
    "Registration Documents",
    "Technical Documents",
    "Personnel Documents",
    "Legal Documents",
    "Declarations",
    "Certificates",
    "Tax Documents",
    "Compliance Documents",
    "Bid Security Documents",
    "Performance Security Documents",
    "Others",
]

# ---------------------------------------------------------------------------
# Derived-field detection patterns
# ---------------------------------------------------------------------------

DERIVED_FIELD_PATTERNS: list[str] = [
    # Aggregations / statistics
    "average",
    "avg",
    "mean",
    "median",
    "total",
    "sum",
    "aggregate",
    "highest",
    "lowest",
    "maximum",
    "minimum",
    "max",
    "min",
    # Counts / derived numbers
    "number of",
    "count of",
    "count",
    "how many",
    "no. of",
    "no of",
    "# of",
    # Temporal derivations
    "years of experience",
    "experience years",
    "total experience",
    "years in business",
    # Financial derivations
    "net worth",
    "networth",
    "net worth",
    "average turnover",
    "avg turnover",
    "total turnover",
    "cumulative turnover",
    "average annual turnover",
    # Rankings / comparisons
    "highest order",
    "lowest bid",
    "largest project",
    "smallest project",
    "biggest contract",
    "top project",
]

# ---------------------------------------------------------------------------
# Category → likely document mapping (used for CATEGORY-mode inference)
# ---------------------------------------------------------------------------

CATEGORY_DOCUMENT_MAP: dict[str, list[str]] = {
    "Financial Documents": [
        "CA Certificate",
        "Balance Sheet",
        "Profit and Loss Account",
        "Audited Financial Statement",
        "Financial Statement",
        "Turnover Certificate",
        "Net Worth Certificate",
    ],
    "Experience Documents": [
        "Work Order",
        "Completion Certificate",
        "Performance Certificate",
        "Experience Certificate",
        "Contract Agreement",
        "Purchase Order",
    ],
    "Registration Documents": [
        "Registration Certificate",
        "Trade License",
        "Incorporation Certificate",
        "Partnership Deed",
        "MEMORANDUM OF ASSOCIATION",
        "ARTICLES OF ASSOCIATION",
    ],
    "Technical Documents": [
        "Technical Proposal",
        "Methodology Document",
        "Technical Specification",
        "Drawings",
        "BOQ (Bill of Quantities)",
    ],
    "Personnel Documents": [
        "Resume",
        "CV",
        "Experience Certificate of Key Personnel",
        "Qualification Certificate",
        "Salary Slip",
    ],
    "Legal Documents": [
        "Affidavit",
        "Legal Undertaking",
        "Power of Attorney",
        "Court Order",
        "Legal Opinion",
    ],
    "Declarations": [
        "Self Declaration",
        "Affidavit of Compliance",
        "Declaration of No Litigation",
        "Integrity Pact",
    ],
    "Certificates": [
        "ISO Certificate",
        "Quality Certificate",
        "Safety Certificate",
        "Environmental Clearance",
        "Industry Certificate",
    ],
    "Tax Documents": [
        "GST Registration Certificate",
        "PAN Card",
        "TAN Certificate",
        "Tax Return",
        "Tax Clearance Certificate",
        "Challan",
    ],
    "Compliance Documents": [
        "Compliance Certificate",
        "Statutory Compliance Report",
        "Labour Compliance Certificate",
        "Environmental Compliance Certificate",
    ],
    "Bid Security Documents": [
        "Earnest Money Deposit Receipt",
        "Bid Bond",
        "Bank Guarantee for EMD",
        "Demand Draft",
    ],
    "Performance Security Documents": [
        "Performance Bank Guarantee",
        "Performance Bond",
        "Security Deposit Receipt",
    ],
    "Others": [],
}

# ---------------------------------------------------------------------------
# Criterion input model (internal representation)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CriterionInput:
    """Internal representation of a single criterion to be planned.

    Attributes:
        criterion_id: Unique identifier.
        text: The full criterion text.
        index: Positional index in the original criteria list (0-based).
    """

    criterion_id: str
    text: str
    index: int = 0


@dataclass(slots=True)
class PlanningResult:
    """Result of planning a single criterion.

    Attributes:
        criterion_id: The criterion that was planned.
        plan: The validated :class:`CriterionPlan` (``None`` if planning failed).
        raw_response: The raw LLM response text before validation.
        attempts: Number of LLM calls made (including repair retries).
        success: Whether planning produced a valid plan.
        error: Error message if ``success`` is False.
    """

    criterion_id: str
    plan: Optional["CriterionPlan"] = None  # noqa: UP037 – forward ref kept explicit
    raw_response: str = ""
    attempts: int = 0
    success: bool = False
    error: str = ""