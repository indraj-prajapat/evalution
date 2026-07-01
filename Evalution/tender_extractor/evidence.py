"""
Evidence tracker.

Builds evidence objects for every extracted value, capturing the
source page, document, snippet, and confidence score.

NEW: Cross-document entity mention scanning.
After extraction, scans ALL documents (including excluded ones) to find
mentions of extracted entity names. This ensures no evidence is missed
even if it spans multiple documents.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from .models import Evidence, ExtractedField, ExtractedRecord, DetectedDocument
from .normalizer import normalise_value
from .utils import truncate_snippet, clean_text

logger = logging.getLogger(__name__)

# Fields that typically contain entity names worth cross-referencing
_ENTITY_NAME_FIELDS = frozenset({
    "company_name", "partner_name", "personnel_name", "employee_name",
    "certificate_holder", "taxpayer_name", "client_name", "employer_name",
    "declarant_name", "director_name", "proprietor_name", "member_name",
    "name_of_firm", "applicant_name", "bidder_name", "contractor_name",
})


class EvidenceTracker:
    """Tracks provenance of every extracted atomic fact."""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_field(
        self,
        name: str,
        raw_value: str,
        datatype: str,
        page: int,
        document_id: str,
        document_name: str,
        full_page_text: str = "",
    ) -> Optional[ExtractedField]:
        """Normalise *raw_value* and wrap it in an ExtractedField.

        Returns None if confidence < min_confidence or value is NOT_FOUND.
        """
        raw_value = clean_text(raw_value)
        if not raw_value:
            return None

        normalised, _ = normalise_value(raw_value, datatype)
        if normalised == "NOT_FOUND":
            return None

        confidence = self._assess_confidence(
            raw_value, normalised, full_page_text
        )
        if confidence < self.min_confidence:
            return None

        snippet = self._build_snippet(raw_value, full_page_text)

        return ExtractedField(
            name=name,
            value=normalised,
            datatype=datatype,
            page=page,
            snippet=snippet,
            confidence=round(confidence, 2),
            raw_value=raw_value,
        )

    def build_evidence(
        self,
        page: int,
        document_id: str,
        document_name: str,
        snippet: str,
        confidence: float,
    ) -> Evidence:
        return Evidence(
            page=page,
            document_id=document_id,
            document_name=document_name,
            snippet=truncate_snippet(snippet),
            confidence=round(confidence, 2),
        )

    # ------------------------------------------------------------------
    # Cross-document entity mention scanning (NEW)
    # ------------------------------------------------------------------

    def scan_cross_document_mentions(
        self,
        matched_outputs: list,
        all_docs: List[DetectedDocument],
    ) -> None:
        """Scan ALL documents for mentions of extracted entity names.

        For each extracted record, finds which OTHER documents mention
        the same entity name. This enables cross-document verification
        in the next step (e.g., checking if a CISA certificate holder
        is listed as a partner in the partner list document).

        Modifies matched_outputs in-place by setting entity_mentions.

        Parameters
        ----------
        matched_outputs : list of MatchedDocument (or dicts with 'records')
        all_docs : ALL detected documents (including excluded ones)
        """
        # Step 1: Collect all entity names from extracted records
        entity_names: Set[str] = set()
        for mo in matched_outputs:
            for rec in mo.records:
                for field in rec.fields:
                    if field.name in _ENTITY_NAME_FIELDS and isinstance(field.value, str):
                        # Add both full name and individual name parts
                        name = field.value.strip()
                        if name and len(name) > 2:
                            entity_names.add(name.lower())
                            # Also add significant name parts (for partial matching)
                            parts = self._extract_name_parts(name)
                            entity_names.update(parts)

        if not entity_names:
            return

        logger.debug(
            "Cross-doc scan: looking for %d entity name variants across %d docs",
            len(entity_names), len(all_docs),
        )

        # Step 2: Build document text index (doc_id -> full text)
        doc_texts: Dict[str, str] = {}
        for doc in all_docs:
            full_text = "\n".join(p.text for p in doc.pages if p.text).lower()
            doc_texts[doc.document_id] = full_text

        # Step 3: For each record, find which documents mention the entity
        for mo in matched_outputs:
            for rec in mo.records:
                mentions: List[dict] = []

                # Get the primary entity name from this record
                record_entity_names: Set[str] = set()
                for field in rec.fields:
                    if field.name in _ENTITY_NAME_FIELDS and isinstance(field.value, str):
                        name = field.value.strip()
                        if name and len(name) > 2:
                            record_entity_names.add(name.lower())
                            parts = self._extract_name_parts(name)
                            record_entity_names.update(parts)

                if not record_entity_names:
                    continue

                # Get the source document ID for this record's fields
                source_doc_id = mo.document_id

                # Check each document for mentions
                for doc in all_docs:
                    doc_text = doc_texts.get(doc.document_id, "")
                    if not doc_text:
                        continue

                    # Check if any entity name variant appears in this doc
                    matched_entity = None
                    for ename in record_entity_names:
                        if ename in doc_text:
                            matched_entity = ename
                            break

                    if matched_entity:
                        # Extract a context snippet
                        snippet = self._find_mention_context(
                            doc_text, matched_entity, doc
                        )

                        mentions.append({
                            "document_id": doc.document_id,
                            "document_name": doc.document_name,
                            "context_snippet": snippet,
                        })

                # Deduplicate mentions (same doc_id should appear only once)
                seen_doc_ids: set = set()
                unique_mentions = []
                for m in mentions:
                    if m["document_id"] not in seen_doc_ids:
                        seen_doc_ids.add(m["document_id"])
                        unique_mentions.append(m)

                if unique_mentions:
                    rec.entity_mentions = unique_mentions

        total_mentions = sum(
            len(mo_entity)
            for mo in matched_outputs
            for rec in mo.records
            for mo_entity in [rec.entity_mentions]
        )
        if total_mentions > 0:
            logger.info(
                "Cross-doc scan found %d entity mentions across documents",
                total_mentions,
            )

    # ------------------------------------------------------------------
    # Confidence assessment
    # ------------------------------------------------------------------

    def _assess_confidence(
        self,
        raw_value: str,
        normalised_value,
        full_page_text: str,
    ) -> float:
        """Assign a confidence score (0.5 – 1.0)."""
        if full_page_text:
            # Check if raw_value appears exactly on the page
            escaped = re.escape(raw_value[:80])
            if re.search(escaped, full_page_text):
                return 0.95

            # Partial match (maybe OCR introduced spaces)
            partial = re.escape(raw_value[:40])
            if re.search(partial, re.sub(r"\s+", " ", full_page_text)):
                return 0.8

            # Check if normalised form appears
            if isinstance(normalised_value, str):
                if normalised_value in full_page_text:
                    return 0.85
            elif isinstance(normalised_value, int):
                indian = self._to_indian_number(normalised_value)
                if indian and indian in full_page_text:
                    return 0.9

        # No page text provided — moderate confidence
        return 0.7

    @staticmethod
    def _build_snippet(raw_value: str, full_page_text: str) -> str:
        """Extract surrounding context as a snippet."""
        if not full_page_text:
            return truncate_snippet(raw_value)

        escaped = re.escape(raw_value[:60])
        m = re.search(escaped, full_page_text)
        if m:
            start = max(0, m.start() - 80)
            end = min(len(full_page_text), m.end() + 80)
            return truncate_snippet(full_page_text[start:end])
        return truncate_snippet(raw_value)

    @staticmethod
    def _to_indian_number(value: int) -> str:
        """Convert integer to Indian number format string."""
        s = str(abs(value))
        if len(s) <= 3:
            return s
        last_three = s[-3:]
        rest = s[:-3]
        groups = []
        while rest:
            groups.append(rest[-2:])
            rest = rest[:-2]
        groups.reverse()
        return ",".join(groups) + "," + last_three

    @staticmethod
    def _extract_name_parts(name: str) -> List[str]:
        """Extract significant name parts for partial matching.

        E.g., "M/s ABC Pvt. Ltd." -> ["abc pvt", "pvt ltd", "abc"]
              "Rahul Sharma" -> ["rahul sharma", "rahul", "sharma"]
        """
        import re as _re
        # Remove common prefixes
        cleaned = _re.sub(r"^M/[sS]\.?\s*", "", name, flags=_re.IGNORECASE).strip()

        parts = []
        # Full name (lowered)
        parts.append(cleaned.lower())

        # Individual words (2+ chars)
        words = _re.findall(r"[a-z]{2,}", cleaned.lower())
        for w in words:
            if w not in ("pvt", "ltd", "llp", "private", "limited", "the", "and",
                          "ms", "inc", "corp", "co"):
                parts.append(w)

        # Bigrams of significant words
        significant = [w for w in words if w not in ("pvt", "ltd", "llp", "private",
                                                       "limited", "the", "and", "ms")]
        for i in range(len(significant) - 1):
            parts.append(f"{significant[i]} {significant[i+1]}")

        return parts

    @staticmethod
    def _find_mention_context(
        doc_text_lower: str,
        entity_name: str,
        doc: DetectedDocument,
    ) -> str:
        """Find a snippet of text surrounding the entity mention in a document."""
        # Find the entity in the original (non-lowered) text
        original_text = "\n".join(p.text for p in doc.pages if p.text)

        # Search for the entity name (case-insensitive)
        idx = original_text.lower().find(entity_name)
        if idx >= 0:
            start = max(0, idx - 60)
            end = min(len(original_text), idx + len(entity_name) + 60)
            snippet = original_text[start:end].replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            return snippet

        return ""