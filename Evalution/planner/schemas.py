"""Pydantic v2 output schemas for the tender information planner.

Every piece of data that the planner returns is validated against these models
before being handed back to the caller.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentMode(str, Enum):
    """How the document was identified.

    EXPLICIT  – the tender text explicitly names a document (e.g. "Submit Work Order").
    CATEGORY  – no document was named; the category was inferred from the criterion.
    """

    EXPLICIT = "EXPLICIT"
    CATEGORY = "CATEGORY"


class FieldType(str, Enum):
    """Supported atomic field data-types."""

    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    CURRENCY = "currency"
    BOOLEAN = "boolean"
    ENUM = "enum"
    PERCENTAGE = "percentage"
    LIST_STRING = "list[string]"


# ---------------------------------------------------------------------------
# Document-level models
# ---------------------------------------------------------------------------

class DocumentField(BaseModel):
    """A single atomic data field extracted from (or expected in) a document.

    Attributes:
        name: Snake_case identifier for the field (e.g. ``annual_turnover``).
        datatype: One of the :class:`FieldType` values.
        description: Human-readable description of what this field represents.
        repeatable: When ``True`` the document may contain multiple values
            (e.g. one per financial year).
        group_by: Optional grouping key used when ``repeatable`` is True
            (e.g. ``financial_year``).
        required: Whether this field must be present for evaluation.
        examples: Example values to guide data-entry or extraction.
    """

    name: str
    datatype: FieldType
    description: str
    repeatable: bool = False
    group_by: Optional[str] = None
    required: bool = True
    examples: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def name_must_be_snake_case(cls, v: str) -> str:
        if v != v.lower() or " " in v or "-" in v:
            raise ValueError(
                f"Field name '{v}' must be lowercase snake_case without spaces or hyphens"
            )
        return v


class DocumentSpec(BaseModel):
    """Specification of a single document required to evaluate a criterion.

    Attributes:
        mode: :class:`DocumentMode` – how the document was identified.
        document: Human-readable document name (e.g. "Work Order", "CA Certificate").
        category: One of the supported :data:`VALID_CATEGORIES`.
        expected_documents: List of concrete document names the bidder is
            expected to submit under this category.
        priority: Search priority – lower numbers are checked first.
        required: Whether this document is mandatory.
        fields: List of :class:`DocumentField` objects to extract from the document.
    """

    mode: DocumentMode
    document: str
    category: str
    expected_documents: list[str] = Field(default_factory=list)
    priority: int = Field(ge=1)
    required: bool = True
    fields: list[DocumentField] = Field(default_factory=list)

    @field_validator("category")
    @classmethod
    def category_must_be_valid(cls, v: str) -> str:
        from Evalution.planner.models import VALID_CATEGORIES

        if v not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{v}'. Must be one of: {VALID_CATEGORIES}"
            )
        return v


class CriterionPlan(BaseModel):
    """The information requirement plan for a single tender criterion.

    Attributes:
        criterion_id: Unique identifier for the criterion (e.g. ``CRIT001``).
        criterion: The original criterion text from the tender.
        required_documents: Ordered list of :class:`DocumentSpec` objects.
    """

    criterion_id: str
    criterion: str
    required_documents: list[DocumentSpec] = Field(default_factory=list)

    @field_validator("criterion_id")
    @classmethod
    def criterion_id_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("criterion_id must not be empty or whitespace")
        return v.strip()