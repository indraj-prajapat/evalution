"""
Data models for the Tender Information Extraction Agent.

Uses Pydantic v2 for strict validation of the extraction schema,
extracted records, evidence, and final output.

Supports both formats:
  - Old: ``required_fields`` key
  - New: ``fields`` key (user's actual format with category/priority/required)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input / Schema models
# ---------------------------------------------------------------------------

class DocumentMode(str, Enum):
    EXPLICIT = "EXPLICIT"
    CATEGORY = "CATEGORY"


class FieldSpec(BaseModel):
    """Describes a single field the tender wants extracted."""
    name: str
    datatype: str = "string"  # currency, date, string, boolean, integer, float
    repeatable: bool = False
    description: str = ""
    required: bool = True
    examples: list[str] = Field(default_factory=list)
    group_by: Optional[str] = None  # per-field grouping override


class RequirementDocument(BaseModel):
    """One row in the tender's information plan."""
    document: str
    mode: DocumentMode = DocumentMode.EXPLICIT
    category: str = ""
    expected_documents: list[str] = Field(default_factory=list)
    required_fields: list[FieldSpec] = Field(default_factory=list)
    entity_type: str = ""  # auto-inferred if empty
    group_by: Optional[str] = None
    priority: int = 1
    required: bool = True
    description: str = ""


class TenderInfoPlan(BaseModel):
    """Top-level container for the extraction schema."""
    criterion_id: str = ""
    criterion: str = ""
    required_documents: list[RequirementDocument] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Bid document JSON models
# ---------------------------------------------------------------------------

class PageSummary(BaseModel):
    page_number: int
    text: str = ""
    summary: str = ""


class DetectedDocument(BaseModel):
    """A logical document detected inside a PDF."""
    document_id: str
    document_name: str
    source_file: str = ""
    document_category: str = ""
    document_type: str = ""
    summary: str = ""
    pages: list[PageSummary] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BidDocument(BaseModel):
    """One uploaded PDF's full parsed representation."""
    pdf_id: str = ""
    overall_summary: str = ""
    detected_documents: list[DetectedDocument] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring models (NEW)
# ---------------------------------------------------------------------------

class ScoredDocument(BaseModel):
    """A document with its relevance scores from RAG + LLM."""
    document_id: str
    document_name: str
    rag_score: float = 0.0
    llm_score: float = 0.0
    combined_score: float = 0.0
    llm_reason: str = ""
    excluded: bool = False
    exclusion_reason: str = ""


class EntityMention(BaseModel):
    """A mention of an entity found in a document."""
    document_id: str
    document_name: str
    context_snippet: str = ""


class ExcludedDocument(BaseModel):
    """A document that was scored as not relevant for a requirement."""
    document_id: str
    document_name: str
    selection_score: float
    rag_score: float = 0.0
    llm_score: float = 0.0
    exclusion_reason: str = ""


# ---------------------------------------------------------------------------
# Extraction output models
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    page: int
    document_id: str
    document_name: str
    snippet: str
    confidence: float = Field(ge=0.5, le=1.0)


class ExtractedField(BaseModel):
    name: str
    value: Any
    datatype: str
    page: int
    snippet: str
    confidence: float = Field(ge=0.5, le=1.0)
    raw_value: Optional[str] = None


class ExtractedRecord(BaseModel):
    entity_id: str
    group: Optional[str] = None
    group_value: Optional[str] = None
    fields: list[ExtractedField] = Field(default_factory=list)
    entity_mentions: list[dict[str, str]] = Field(default_factory=list)


class MatchedDocument(BaseModel):
    document_id: str
    document_name: str
    status: str = "FOUND"
    pages: list[int] = Field(default_factory=list)
    summary: str = ""
    records: list[ExtractedRecord] = Field(default_factory=list)
    selection_score: float = 0.0
    rag_score: float = 0.0
    llm_score: float = 0.0


class RequirementOutput(BaseModel):
    requirement_document: str
    mode: str
    entity_type: str
    matched_documents: list[MatchedDocument] = Field(default_factory=list)
    excluded_documents: list[ExcludedDocument] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    documents: list[RequirementOutput] = Field(default_factory=list)

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to strict JSON (no markdown, no comments)."""
        return self.model_dump_json(**kwargs)
