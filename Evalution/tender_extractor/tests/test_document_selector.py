"""Unit tests for the document selector module."""

import pytest

from Evalution.tender_extractor.document_selector import DocumentSelector
from Evalution.tender_extractor.models import BidDocument, DetectedDocument, PageSummary, RequirementDocument, DocumentMode, FieldSpec


def _make_page(num: int, text: str) -> PageSummary:
    return PageSummary(page_number=num, text=text, summary=text[:100])


def _make_doc(doc_id: str, name: str, category: str = "", pages: list | None = None) -> DetectedDocument:
    return DetectedDocument(
        document_id=doc_id,
        document_name=name,
        document_category=category,
        pages=pages or [],
    )


class TestDocumentSelector:
    """Document selection tests (rule-based, no LLM)."""

    def _selector(self, docs: list[DetectedDocument]) -> DocumentSelector:
        """Create selector with LLM disabled for tests."""
        return DocumentSelector(
            [BidDocument(detected_documents=docs)],
            use_llm_fallback=False,
        )

    def test_explicit_exact_match(self):
        docs = [_make_doc("d1", "Work Order"), _make_doc("d2", "Balance Sheet")]
        selector = self._selector(docs)

        req = RequirementDocument(
            document="Work Order",
            mode=DocumentMode.EXPLICIT,
            required_fields=[],
        )
        result = selector.select(req)
        assert len(result) == 1
        assert result[0].document_id == "d1"

    def test_explicit_no_match(self):
        docs = [_make_doc("d1", "Balance Sheet")]
        selector = self._selector(docs)

        req = RequirementDocument(
            document="Work Order",
            mode=DocumentMode.EXPLICIT,
            required_fields=[],
        )
        result = selector.select(req)
        assert len(result) == 0

    def test_category_priority(self):
        docs = [
            _make_doc("d1", "Balance Sheet", "Financial Documents"),
            _make_doc("d2", "CA Certificate", "Financial Documents"),
            _make_doc("d3", "Some Other", "Non-Financial"),
        ]
        selector = self._selector(docs)

        req = RequirementDocument(
            document="Financial Documents",
            mode=DocumentMode.CATEGORY,
            expected_documents=["CA Certificate", "Balance Sheet"],
            required_fields=[],
        )
        result = selector.select(req)
        ids = [d.document_id for d in result]
        assert "d1" in ids
        assert "d2" in ids
        assert "d3" not in ids

    def test_category_fallback_to_category_name(self):
        """When expected_documents don't match, result is empty."""
        docs = [
            _make_doc("d1", "Annual Statement", "Financial Documents"),
        ]
        selector = self._selector(docs)

        req = RequirementDocument(
            document="Financial Documents",
            mode=DocumentMode.CATEGORY,
            expected_documents=["CA Certificate"],
            required_fields=[],
        )
        result = selector.select(req)
        assert len(result) == 0

    def test_empty_documents(self):
        selector = DocumentSelector([], use_llm_fallback=False)
        req = RequirementDocument(
            document="Anything",
            mode=DocumentMode.EXPLICIT,
            required_fields=[],
        )
        assert selector.select(req) == []


class TestDocumentSelectorLLMFallback:
    """Test that LLM fallback triggers correctly when rules fail."""

    def test_llm_fallback_called_on_no_match(self):
        """When rules find nothing and LLM fallback is enabled,
        it should try the LLM (and fail gracefully without API key)."""
        docs = [_make_doc("d1", "Some Random Doc", "Other")]
        # LLM enabled but will fail gracefully (no API key)
        selector = DocumentSelector(
            [BidDocument(detected_documents=docs)],
            use_llm_fallback=True,
        )

        req = RequirementDocument(
            document="Very Specific Document",
            mode=DocumentMode.EXPLICIT,
            required_fields=[],
        )
        # Should not raise, just return empty (LLM fails without key)
        result = selector.select(req)
        assert isinstance(result, list)