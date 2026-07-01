"""
Document selector.

Implements the two document-matching modes:

  EXPLICIT  – the tender names an exact document type; only search for that.
  CATEGORY  – the tender names a category; search in priority order of
              expected_documents.

Falls back to LLM-based selection when rule-based matching finds nothing.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .models import (
    BidDocument,
    DetectedDocument,
    DocumentMode,
    RequirementDocument,
)
from .utils import doc_name_matches, doc_category_matches

logger = logging.getLogger(__name__)


class DocumentSelector:
    """Selects detected documents that match a requirement."""

    def __init__(
        self,
        bid_documents: List[BidDocument],
        use_llm_fallback: bool = True,
    ):
        # Flatten all detected docs across all PDFs
        self._all_docs: List[DetectedDocument] = []
        for bid_doc in bid_documents:
            self._all_docs.extend(bid_doc.detected_documents)
        self._use_llm_fallback = use_llm_fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        requirement: RequirementDocument,
    ) -> List[DetectedDocument]:
        """Return detected documents that satisfy *requirement*.

        Strategy: try rule-based first, fall back to LLM if nothing matched.
        """
        if requirement.mode == DocumentMode.EXPLICIT:
            results = self._select_explicit(requirement)
        else:
            results = self._select_category(requirement)

        # LLM fallback: if rule-based found nothing, ask GPT-4o-mini
        if not results and self._use_llm_fallback and self._all_docs:
            logger.info(
                "Rule-based selection found nothing for '%s', trying LLM fallback",
                requirement.document,
            )
            results = self._llm_fallback(requirement)

        return results

    # ------------------------------------------------------------------
    # EXPLICIT mode
    # ------------------------------------------------------------------

    def _select_explicit(
        self,
        requirement: RequirementDocument,
    ) -> List[DetectedDocument]:
        """Only match documents whose name or category exactly matches
        the requirement document name."""
        target = requirement.document
        results: List[DetectedDocument] = []

        for doc in self._all_docs:
            # Primary: match on document name
            if doc_name_matches(doc.document_name, [target], threshold=0.6):
                results.append(doc)
                continue
            # Secondary: match on document category
            if doc.document_category and doc_category_matches(
                doc.document_category, [target], threshold=0.6
            ):
                results.append(doc)
                continue
            # Tertiary: match on document type
            if doc.document_type and doc_name_matches(
                doc.document_type, [target], threshold=0.5
            ):
                results.append(doc)
                continue

        return results

    # ------------------------------------------------------------------
    # CATEGORY mode
    # ------------------------------------------------------------------

    def _select_category(
        self,
        requirement: RequirementDocument,
    ) -> List[DetectedDocument]:
        """Search in priority order of expected_documents."""
        expected = requirement.expected_documents or []
        if not expected:
            # Fallback: search by category name
            return self._select_explicit(requirement)

        assigned: set[str] = set()
        results: List[DetectedDocument] = []

        for exp_name in expected:
            for doc in self._all_docs:
                if doc.document_id in assigned:
                    continue

                matched = False
                if doc_name_matches(doc.document_name, [exp_name], threshold=0.5):
                    matched = True
                elif (doc.document_category
                      and len(doc.document_category.split()) >= 2
                      and doc_category_matches(
                          doc.document_category, [exp_name], threshold=0.6
                      )):
                    matched = True

                if matched:
                    results.append(doc)
                    assigned.add(doc.document_id)

        return results

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _llm_fallback(
        self,
        requirement: RequirementDocument,
    ) -> List[DetectedDocument]:
        """Use GPT-4o-mini to match documents when rules fail.

        Sends both document metadata AND page text excerpts so the LLM
        can match based on actual content (not just names).
        """
        try:
            from .llm_extractor import select_documents_with_llm

            # First try the summary-based LLM selection
            results = select_documents_with_llm(
                requirement_name=requirement.document,
                requirement_category=requirement.category,
                expected_documents=requirement.expected_documents,
                available_docs=self._all_docs,
                mode=requirement.mode.value,
            )
            if results:
                return results

            # If still nothing, try content-based selection using page text
            logger.info(
                "Summary-based LLM fallback found nothing for '%s', "
                "trying content-based LLM fallback",
                requirement.document,
            )
            return self._llm_content_fallback(requirement)

        except Exception as e:
            logger.warning("LLM doc selection failed: %s", e)
            return []

    def _llm_content_fallback(
        self,
        requirement: RequirementDocument,
    ) -> List[DetectedDocument]:
        """Use GPT-4o-mini to match documents by their page text content."""
        try:
            from .llm_client import chat_json
        except ImportError:
            return []

        # Build content-based document summaries
        doc_entries = []
        doc_index_map = []
        for i, doc in enumerate(self._all_docs):
            # Get first 300 chars from first page as content preview
            content_preview = ""
            for page in doc.pages:
                if page.text and page.text.strip():
                    content_preview = page.text[:300].replace("\n", " ").strip()
                    break

            doc_entries.append(
                f"[{i}] Name: {doc.document_name} | "
                f"Category: {doc.document_category} | "
                f"Content: {content_preview}"
            )
            doc_index_map.append(i)

        if not doc_entries:
            return []

        user_prompt = (
            f"Requirement: '{requirement.document}'\n"
            f"Category: '{requirement.category}'\n"
        )
        if requirement.expected_documents:
            user_prompt += f"Expected: {', '.join(requirement.expected_documents)}\n"

        user_prompt += (
            "\nAvailable documents (with content preview):\n"
            + "\n".join(doc_entries)
            + "\n\nWhich documents contain content matching the requirement? "
            "Match based on ACTUAL CONTENT, not just the document name. "
            "Return JSON: {\"matched_indices\": [0, 1, ...]}"
        )

        system_prompt = (
            "You are a document content analyst for Indian tender evaluation. "
            "Match documents to requirements based on their content. "
            "A document containing 'branch registration' text matches "
            "'Branch/Office Registered Certificate' even if named differently. "
            "Return JSON only."
        )

        try:
            result = chat_json(system_prompt, user_prompt, max_tokens=256)
        except Exception as e:
            logger.warning("LLM content fallback failed: %s", e)
            return []

        matched_indices = result.get("matched_indices", [])
        if not isinstance(matched_indices, list):
            return []

        selected = []
        for idx in matched_indices:
            if isinstance(idx, (int, float)) and 0 <= int(idx) < len(self._all_docs):
                selected.append(self._all_docs[int(idx)])

        if selected:
            logger.info(
                "LLM content fallback matched %d doc(s) for '%s'",
                len(selected), requirement.document,
            )
        return selected

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_all_docs(self) -> List[DetectedDocument]:
        return list(self._all_docs)

    def find_by_id(self, document_id: str) -> Optional[DetectedDocument]:
        for doc in self._all_docs:
            if doc.document_id == document_id:
                return doc
        return None