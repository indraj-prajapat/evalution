"""
Document Scorer — RAG + LLM scoring for all documents.

Key design principles:
  1. SCORE every document, don't FILTER. Only exclude when 100% sure unrelated.
  2. Two independent scoring signals: RAG (BM25+semantic+RRF) and LLM (filename+hint).
  3. Combined score = weighted average. Low threshold for exclusion (0.05).
  4. Token-efficient LLM scoring: only filename + 200 char content hint per doc.
  5. Batched LLM calls to handle large document sets.

Scoring guide for LLM:
  1.0  — Document name directly matches the requirement
  0.8  — Document category/type clearly matches
  0.6  — Content hint suggests relevance even if name doesn't match
  0.3  — Partially relevant, might contain some useful info
  0.1  — Unlikely to be relevant
  0.0  — Completely unrelated (only when 100% certain)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .config import ExtractionConfig
from .models import (
    DetectedDocument,
    RequirementDocument,
    ScoredDocument,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM scoring prompt
# ---------------------------------------------------------------------------

_SCORING_SYSTEM = """\
You are a document relevance scorer for Indian tender document evaluation.
Your job is to score each document's relevance to a specific tender requirement.

SCORING GUIDE:
- 1.0: Document name DIRECTLY matches the requirement name (e.g., both say "GST Registration Certificate")
- 0.8: Document type/category clearly matches (e.g., requirement asks for "Financial Documents", doc is a "Balance Sheet")
- 0.6: Content preview suggests relevance even if filename is different (e.g., file named "Annexure G" but content shows "Branch Registration Certificate")
- 0.3: Partially relevant — might contain some useful information
- 0.1: Unlikely to be relevant but cannot be 100% sure
- 0.0: COMPLETELY UNRELATED — only use this when you are ABSOLUTELY CERTAIN the document has zero relevance to the requirement

CRITICAL: When in doubt, give a HIGHER score. It is far better to include a marginally relevant document
than to miss critical evidence. Only score 0.0 for documents that are clearly about a completely
different topic (e.g., a "Medical Certificate" for a "Financial Turnover" requirement).

Return JSON only: {"scores": [{"index": 0, "score": 0.8, "reason": "brief reason"}, ...]}
"""


# ---------------------------------------------------------------------------
# Document Scorer
# ---------------------------------------------------------------------------

class DocumentScorer:
    """Scores ALL documents for relevance to a requirement.

    Uses two independent signals:
      1. RAG score: BM25 + semantic search + RRF over page-level content
      2. LLM score: GPT-4o-mini evaluates filename + content hint

    Documents are only excluded when combined_score < doc_exclusion_threshold
    (default 0.05 — i.e., 100% sure unrelated).
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_documents(
        self,
        requirement: RequirementDocument,
        all_docs: List[DetectedDocument],
        rag_doc_scores: Optional[Dict[str, float]] = None,
    ) -> List[ScoredDocument]:
        """Score all documents for the given requirement.

        Parameters
        ----------
        requirement : the tender plan requirement
        all_docs : ALL detected documents (no pre-filtering)
        rag_doc_scores : optional pre-computed RAG scores {doc_id: score}
                        If None, RAG scoring is skipped (e.g., if RAG not available)

        Returns
        -------
        List of ScoredDocument, sorted by combined_score descending.
        Documents with excluded=True should be skipped for extraction
        (but still scanned by regex safety net).
        """
        if not all_docs:
            return []

        rag_scores = rag_doc_scores or {}

        # Phase 1: LLM scoring (batched, token-efficient)
        llm_scores = {}
        llm_reasons = {}
        if self.config.use_llm_doc_scoring:
            try:
                llm_scores, llm_reasons = self._llm_score_all(requirement, all_docs)
            except Exception as e:
                logger.warning("LLM scoring failed: %s. Using default scores.", e)
                # Default: give all docs benefit of doubt (0.3)
                for doc in all_docs:
                    llm_scores[doc.document_id] = 0.3
                    llm_reasons[doc.document_id] = "LLM scoring failed, default score"
        else:
            # LLM disabled: give all docs benefit of doubt
            for doc in all_docs:
                llm_scores[doc.document_id] = 0.3
                llm_reasons[doc.document_id] = "LLM scoring disabled, default score"

        # Phase 2: Combine scores
        w_rag = self.config.rag_score_weight
        w_llm = self.config.llm_score_weight
        threshold = self.config.doc_exclusion_threshold

        scored: List[ScoredDocument] = []
        for doc in all_docs:
            doc_id = doc.document_id
            rag = rag_scores.get(doc_id, 0.0)
            llm = llm_scores.get(doc_id, 0.3)
            combined = round(w_rag * rag + w_llm * llm, 4)

            excluded = combined < threshold
            exclusion_reason = ""
            if excluded:
                exclusion_reason = (
                    f"Combined score {combined:.3f} below threshold {threshold}. "
                    f"RAG={rag:.3f}, LLM={llm:.3f}. "
                    f"LLM reason: {llm_reasons.get(doc_id, 'N/A')}"
                )
                logger.debug(
                    "EXCLUDED doc '%s' (score=%.4f): %s",
                    doc.document_name, combined, exclusion_reason,
                )
            else:
                logger.debug(
                    "INCLUDED doc '%s' (score=%.4f): RAG=%.3f, LLM=%.3f",
                    doc.document_name, combined, rag, llm,
                )

            scored.append(ScoredDocument(
                document_id=doc_id,
                document_name=doc.document_name,
                rag_score=round(rag, 4),
                llm_score=round(llm, 4),
                combined_score=combined,
                llm_reason=llm_reasons.get(doc_id, ""),
                excluded=excluded,
                exclusion_reason=exclusion_reason,
            ))

        # Sort by combined score descending
        scored.sort(key=lambda s: s.combined_score, reverse=True)

        included = sum(1 for s in scored if not s.excluded)
        logger.info(
            "Document scoring for '%s': %d/%d included, %d excluded",
            requirement.document, included, len(scored),
            len(scored) - included,
        )

        return scored

    # ------------------------------------------------------------------
    # LLM scoring (batched)
    # ------------------------------------------------------------------

    def _llm_score_all(
        self,
        requirement: RequirementDocument,
        all_docs: List[DetectedDocument],
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Score all documents using LLM in batches.

        Returns (doc_id -> score, doc_id -> reason) mappings.
        Token-efficient: only sends filename + short content hint per document.
        """
        from .llm_client import chat_json

        scores: Dict[str, float] = {}
        reasons: Dict[str, str] = {}

        batch_size = self.config.llm_scoring_batch_size
        hint_chars = self.config.llm_scoring_content_hint_chars

        # Build requirement description
        req_desc = f"Requirement: '{requirement.document}'"
        if requirement.category:
            req_desc += f"\nCategory: '{requirement.category}'"
        if requirement.expected_documents:
            req_desc += f"\nExpected documents: {', '.join(requirement.expected_documents)}"
        if requirement.description:
            req_desc += f"\nDescription: {requirement.description}"
        field_names = [f.name for f in requirement.required_fields]
        if field_names:
            req_desc += f"\nFields needed: {', '.join(field_names)}"

        # Process in batches
        for batch_start in range(0, len(all_docs), batch_size):
            batch = all_docs[batch_start:batch_start + batch_size]

            # Build document entries (filename + content hint only)
            doc_entries = []
            for i, doc in enumerate(batch):
                # Get content hint: first non-empty page text, truncated
                content_hint = ""
                for page in doc.pages:
                    if page.text and page.text.strip():
                        content_hint = page.text[:hint_chars].replace("\n", " ").strip()
                        break
                if not content_hint:
                    content_hint = (doc.summary or "")[:hint_chars]

                entry = f"[{i}] File: {doc.document_name}"
                if doc.document_category:
                    entry += f" | Category: {doc.document_category}"
                if doc.document_type:
                    entry += f" | Type: {doc.document_type}"
                if content_hint:
                    entry += f" | Content: {content_hint}"
                doc_entries.append(entry)

            user_prompt = (
                f"{req_desc}\n\n"
                f"Documents to score (index | filename | category | content hint):\n"
                + "\n".join(doc_entries)
                + "\n\nScore each document's relevance to the requirement. "
                "Remember: when in doubt, score HIGHER. Only score 0.0 if completely unrelated.\n"
                "Return JSON: {\"scores\": [{\"index\": 0, \"score\": 0.8, \"reason\": \"brief reason\"}, ...]}"
            )

            try:
                result = chat_json(
                    system_prompt=_SCORING_SYSTEM,
                    user_prompt=user_prompt,
                    max_tokens=1024,
                    temperature=0.0,
                )
            except Exception as e:
                logger.warning("LLM scoring batch failed: %s", e)
                # Give benefit of doubt for this batch
                for doc in batch:
                    scores[doc.document_id] = 0.3
                    reasons[doc.document_id] = "LLM scoring failed, default score"
                continue

            llm_score_list = result.get("scores", [])
            if not isinstance(llm_score_list, list):
                for doc in batch:
                    scores[doc.document_id] = 0.3
                    reasons[doc.document_id] = "LLM returned invalid format"
                continue

            # Map LLM results back to documents
            for item in llm_score_list:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                score = item.get("score", 0.3)
                reason = item.get("reason", "")

                if isinstance(idx, (int, float)) and 0 <= int(idx) < len(batch):
                    doc = batch[int(idx)]
                    # Clamp score to [0, 1]
                    clamped = max(0.0, min(1.0, float(score)))
                    scores[doc.document_id] = clamped
                    reasons[doc.document_id] = reason

            # For any docs in this batch not scored by LLM, give default
            for doc in batch:
                if doc.document_id not in scores:
                    scores[doc.document_id] = 0.3
                    reasons[doc.document_id] = "Not scored by LLM, default"

        return scores, reasons