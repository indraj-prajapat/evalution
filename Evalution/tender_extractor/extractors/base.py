"""
Base extractor class.

All entity-specific extractors inherit from this and implement
`extract_from_page()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import (
    DetectedDocument,
    ExtractedField,
    ExtractedRecord,
    FieldSpec,
    PageSummary,
)
from ..evidence import EvidenceTracker


class BaseExtractor(ABC):
    """Abstract base for per-entity-type extraction logic."""

    entity_type: str = "Other"

    def __init__(self, evidence_tracker: EvidenceTracker):
        self.tracker = evidence_tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        doc: DetectedDocument,
        required_fields: List[FieldSpec],
        group_by: Optional[str] = None,
    ) -> List[ExtractedRecord]:
        """Extract all requested fields from every relevant page of *doc*.

        Parameters
        ----------
        doc : the detected document
        required_fields : fields requested by the tender
        group_by : optional grouping key (e.g. "financial_year", "project")
        """
        records: List[ExtractedRecord] = []
        field_names = [f.name for f in required_fields]

        # Determine which pages to inspect
        relevant_pages = self._select_pages(doc, field_names)
        if not relevant_pages:
            # No records could be extracted
            return records

        # Extract entities from each page
        for page in relevant_pages:
            page_records = self.extract_from_page(
                page=page,
                doc=doc,
                required_fields=required_fields,
                group_by=group_by,
            )
            records.extend(page_records)

        # Apply repeatable grouping
        if group_by:
            records = self._apply_grouping(records, group_by)

        return records

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_from_page(
        self,
        page: PageSummary,
        doc: DetectedDocument,
        required_fields: List[FieldSpec],
        group_by: Optional[str] = None,
    ) -> List[ExtractedRecord]:
        """Extract records from a single page.

        Must be implemented by each entity-specific extractor.
        """
        ...

    # ------------------------------------------------------------------
    # Page selection
    # ------------------------------------------------------------------

    def _select_pages(
        self,
        doc: DetectedDocument,
        field_names: List[str],
    ) -> List[PageSummary]:
        """Pick pages likely to contain the requested fields."""
        selected: List[PageSummary] = []

        # First, check document summary for relevance hints
        doc_summary_lower = doc.summary.lower() if doc.summary else ""

        for page in doc.pages:
            page_text = (page.text or "").lower()
            page_summary = (page.summary or "").lower()

            # A page is relevant if it mentions keywords from the entity type
            if self._page_is_relevant(page_text, page_summary, doc_summary_lower, field_names):
                selected.append(page)

        # If no pages selected via keywords, use all pages (fallback)
        if not selected and doc.pages:
            selected = doc.pages

        return selected

    def _page_is_relevant(
        self,
        page_text: str,
        page_summary: str,
        doc_summary: str,
        field_names: List[str],
    ) -> bool:
        """Override in subclass for entity-specific relevance."""
        # Default: check if any field name (with underscores replaced) appears
        for fn in field_names:
            keywords = fn.replace("_", " ").split()
            if all(kw in page_text for kw in keywords):
                return True
            if all(kw in page_summary for kw in keywords):
                return True
        return False

    # ------------------------------------------------------------------
    # Grouping helper
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_grouping(
        records: List[ExtractedRecord],
        group_by: str,
    ) -> List[ExtractedRecord]:
        """Merge records sharing the same group_value (kept separate by default)."""
        # By default, records are already separate — no merging.
        # The hard rules say "never merge", so we just tag them.
        for rec in records:
            rec.group = group_by
        return records

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def _build_field(
        self,
        name: str,
        raw_value: str,
        datatype: str,
        page: int,
        document_id: str,
        document_name: str,
        full_page_text: str = "",
    ) -> Optional[ExtractedField]:
        """Convenience wrapper around EvidenceTracker.build_field."""
        return self.tracker.build_field(
            name=name,
            raw_value=raw_value,
            datatype=datatype,
            page=page,
            document_id=document_id,
            document_name=document_name,
            full_page_text=full_page_text,
        )

    def _find_field_spec(
        self, field_name: str, required_fields: List[FieldSpec]
    ) -> Optional[FieldSpec]:
        for fs in required_fields:
            if fs.name == field_name:
                return fs
        return None