"""
Main extraction orchestrator with scoring-based pipeline.

TenderExtractor is the primary entry point. It:
  1. Builds a RAG index (BM25 + semantic embeddings) over all pages
  2. For each requirement:
     a. SCORES all documents (RAG + LLM) — never filters aggressively
     b. Only excludes documents with combined_score < 0.05 (100% sure unrelated)
     c. Extracts from all non-excluded documents (LLM primary, regex fallback)
     d. Runs regex safety net on ALL documents including excluded ones
     e. Scans cross-document entity mentions across ALL documents
  3. Returns an ExtractionResult as strict JSON.

KEY CHANGES from v3.1:
  - Document selection is now SCORING-based, not filtering-based
  - Company ownership is ANNOTATION only (not a gate)
  - Cross-document entity mentions are tracked
  - Regex safety net runs on ALL documents
  - LLM scoring uses only filename + 200 char content hint (token efficient)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .models import (
    BidDocument,
    DetectedDocument,
    ExtractionResult,
    ExcludedDocument,
    MatchedDocument,
    RequirementDocument,
    RequirementOutput,
    ScoredDocument,
    TenderInfoPlan,
)
from .config import ExtractionConfig
from .company_matcher import CompanyMatcher
from .evidence import EvidenceTracker
from .entity_extractor import extract_entities
from .retriever import RetrievalIndex
from .document_scorer import DocumentScorer

logger = logging.getLogger(__name__)


class TenderExtractor:
    """Orchestrates the full extraction pipeline with scoring + evidence safety nets.

    Pipeline flow per requirement:
      1. RAG scores all documents (BM25 + semantic + RRF → document-level scores)
      2. LLM scores all documents (filename + content hint → relevance score)
      3. Combined score = weighted average; exclude only if < 0.05
      4. LLM extracts fields from non-excluded documents
      5. Regex extraction supplements LLM (and runs on ALL docs as safety net)
      6. Cross-document entity mention scanning across ALL documents
    """

    def __init__(
        self,
        bidder_name: str,
        tender_plan: Union[Dict[str, Any], TenderInfoPlan],
        bid_documents: Union[List[Dict[str, Any]], List[BidDocument]],
        config: Optional[ExtractionConfig] = None,
    ):
        self.bidder_name = bidder_name
        self.config = config or ExtractionConfig()
        self.config.setup_logging()

        # Parse inputs
        self.tender_plan = self._parse_tender_plan(tender_plan)
        self.bid_documents = self._parse_bid_documents(bid_documents)

        # Flatten all detected documents from all PDFs
        self._all_detected_docs: List[DetectedDocument] = []
        for bd in self.bid_documents:
            self._all_detected_docs.extend(bd.detected_documents)

        # Initialize components
        self.tracker = EvidenceTracker(min_confidence=self.config.min_confidence)
        self.company_matcher = CompanyMatcher(bidder_name=bidder_name)
        self.doc_scorer = DocumentScorer(self.config)

        # Build RAG index
        self.index: Optional[RetrievalIndex] = None
        if self.config.use_rag:
            self._build_rag_index()

        logger.info(
            "TenderExtractor initialised: bidder='%s', %d requirements, %d PDF(s), "
            "%d total docs, rag=%s, llm=%s",
            bidder_name,
            len(self.tender_plan.required_documents),
            len(self.bid_documents),
            len(self._all_detected_docs),
            self.config.use_rag,
            self.config.use_llm_extraction,
        )

    def _build_rag_index(self) -> None:
        """Build the RAG retrieval index."""
        try:
            from .embedder import init_embedder
            init_embedder(self.config.embedding_model)

            self.index = RetrievalIndex(self.bid_documents)
            self.index.build()

            logger.info(
                "RAG index ready: %d pages indexed",
                self.index.total_pages(),
            )
        except Exception as e:
            logger.error("Failed to build RAG index: %s. RAG scoring disabled.", e)
            self.index = None
            self.config.use_rag = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> ExtractionResult:
        """Execute the full extraction pipeline."""
        outputs: List[RequirementOutput] = []

        for req in self.tender_plan.required_documents:
            logger.info("--- Processing requirement: %s (mode=%s) ---", req.document, req.mode)
            req_output = self._process_requirement(req)
            outputs.append(req_output)

        result = ExtractionResult(documents=outputs)
        logger.info(
            "Extraction complete: %d requirement(s) processed", len(outputs)
        )
        return result

    # ------------------------------------------------------------------
    # Per-requirement processing
    # ------------------------------------------------------------------

    def _process_requirement(self, req: RequirementDocument) -> RequirementOutput:
        """Process a single requirement using the scoring-based pipeline.

        Phase 1: Score all documents (RAG + LLM)
        Phase 2: Extract from non-excluded documents (LLM + regex)
        Phase 3: Regex safety net on excluded documents
        Phase 4: Cross-document entity mention scanning
        """
        # ================================================================
        # Phase 1: SCORE ALL DOCUMENTS
        # ================================================================
        rag_scores: Dict[str, float] = {}

        # 1a. RAG document-level scoring
        if self.index and self.config.use_rag:
            try:
                rag_scores = self.index.score_documents(
                    req,
                    top_k=self.config.rag_top_k * 3,
                    rrf_k=self.config.rag_rrf_k,
                )
            except Exception as e:
                logger.warning("RAG scoring failed for '%s': %s", req.document, e)

        # 1b. Full scoring (RAG + LLM)
        scored_docs = self.doc_scorer.score_documents(
            requirement=req,
            all_docs=self._all_detected_docs,
            rag_doc_scores=rag_scores,
        )

        # Split into included and excluded
        included_docs = [
            (sd, self._find_doc(sd.document_id))
            for sd in scored_docs if not sd.excluded
        ]
        excluded_docs = [sd for sd in scored_docs if sd.excluded]

        if not self._all_detected_docs:
            logger.info("No documents available for '%s'", req.document)
            return RequirementOutput(
                requirement_document=req.document,
                mode=req.mode.value,
                entity_type=req.entity_type,
                matched_documents=[],
                excluded_documents=[],
            )

        # ================================================================
        # Phase 2: EXTRACT FROM NON-EXCLUDED DOCUMENTS
        # ================================================================
        matched_outputs: List[MatchedDocument] = []
        seen_doc_ids: set = set()

        for sd, doc in included_docs:
            if doc is None:
                continue

            # Deduplicate by document_id (shouldn't happen, but safety)
            if doc.document_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc.document_id)

            # Extract using LLM (primary) + regex (fallback)
            records = extract_entities(
                doc=doc,
                requirement=req,
                tracker=self.tracker,
                config=self.config,
            )

            # Annotate company ownership (informational, not gating)
            self._annotate_company_ownership(records, doc)

            matched_outputs.append(
                MatchedDocument(
                    document_id=self._public_document_id(doc),
                    document_name=doc.document_name,
                    status="FOUND" if records else "PARTIAL",
                    pages=[p.page_number for p in doc.pages],
                    summary=doc.summary,
                    records=records,
                    selection_score=sd.combined_score,
                    rag_score=sd.rag_score,
                    llm_score=sd.llm_score,
                )
            )

        # ================================================================
        # Phase 3: REGEX SAFETY NET ON EXCLUDED DOCUMENTS
        # ================================================================
        if self.config.regex_safety_net and excluded_docs:
            safety_records = self._regex_safety_net(excluded_docs, req)

            if safety_records:
                logger.info(
                    "Safety net found %d additional record(s) from %d excluded docs for '%s'",
                    sum(len(r) for r in safety_records.values()),
                    len(safety_records),
                    req.document,
                )
                # Add safety net findings as additional matched documents
                for sd, doc in [
                    (sd, self._find_doc(sd.document_id))
                    for sd in excluded_docs
                ]:
                    if doc is None or doc.document_id not in safety_records:
                        continue
                    if doc.document_id in seen_doc_ids:
                        continue
                    seen_doc_ids.add(doc.document_id)

                    matched_outputs.append(
                        MatchedDocument(
                            document_id=self._public_document_id(doc),
                            document_name=doc.document_name,
                            status="FOUND",
                            pages=[p.page_number for p in doc.pages],
                            summary=doc.summary,
                            records=safety_records[doc.document_id],
                            selection_score=sd.combined_score,
                            rag_score=sd.rag_score,
                            llm_score=sd.llm_score,
                        )
                    )

        # ================================================================
        # Phase 4: CROSS-DOCUMENT ENTITY MENTION SCANNING
        # ================================================================
        if self.config.cross_document_mentions and matched_outputs:
            try:
                self.tracker.scan_cross_document_mentions(
                    matched_outputs=matched_outputs,
                    all_docs=self._all_detected_docs,
                )
            except Exception as e:
                logger.warning("Cross-document mention scanning failed: %s", e)

        # ================================================================
        # Build excluded documents list for output
        # ================================================================
        excluded_output = [
            ExcludedDocument(
                document_id=sd.document_id,
                document_name=sd.document_name,
                selection_score=sd.combined_score,
                rag_score=sd.rag_score,
                llm_score=sd.llm_score,
                exclusion_reason=sd.exclusion_reason,
            )
            for sd in excluded_docs
        ]

        logger.info(
            "Requirement '%s': %d matched, %d excluded, %d total records extracted",
            req.document,
            len(matched_outputs),
            len(excluded_docs),
            sum(len(mo.records) for mo in matched_outputs),
        )

        return RequirementOutput(
            requirement_document=req.document,
            mode=req.mode.value,
            entity_type=req.entity_type,
            matched_documents=matched_outputs,
            excluded_documents=excluded_output,
        )

    # ------------------------------------------------------------------
    # Regex safety net
    # ------------------------------------------------------------------

    def _regex_safety_net(
        self,
        excluded_scored: List[ScoredDocument],
        req: RequirementDocument,
    ) -> Dict[str, list]:
        """Run regex extraction on excluded documents as a safety net.

        If regex finds valid records in an excluded document, it means
        the document actually contains relevant evidence and should be
        included despite its low score.

        Returns {doc_id: [ExtractedRecord, ...]} for docs with findings.
        """
        findings: Dict[str, list] = {}

        for sd in excluded_scored:
            doc = self._find_doc(sd.document_id)
            if doc is None:
                continue

            # Only use regex (no LLM) to save tokens
            try:
                from .extractors import get_extractor

                entity_type = req.entity_type or "Other"
                extractor = get_extractor(entity_type, self.tracker)

                records = extractor.extract(
                    doc=doc,
                    required_fields=req.required_fields,
                    group_by=req.group_by,
                )

                if records:
                    findings[doc.document_id] = records
                    logger.debug(
                        "Safety net: regex found %d record(s) in excluded doc '%s'",
                        len(records), doc.document_name,
                    )

            except Exception as e:
                logger.debug(
                    "Safety net regex failed for '%s': %s",
                    sd.document_name, e,
                )

        return findings

    # ------------------------------------------------------------------
    # Company ownership annotation (NOT a gate)
    # ------------------------------------------------------------------

    def _annotate_company_ownership(
        self,
        records: list,
        doc: DetectedDocument,
    ) -> None:
        """Annotate records with company ownership info (informational only).

        Does NOT filter or reject any records. Just adds metadata
        so the next step can decide whether to verify.
        """
        if self.config.skip_company_match:
            return

        full_text = "\n".join(p.text for p in doc.pages)

        # Check if this is an experience document (different entity expected)
        is_experience = self.company_matcher.is_experience_document(
            doc.document_category, doc.document_name
        )

        is_owner, reason = self.company_matcher.is_owned_by_bidder(
            doc_text=full_text,
            doc_name=doc.document_name,
            doc_category=doc.document_category,
            doc_metadata=doc.metadata,
        )

        # Store as extra context on each record (via entity_id prefix or metadata)
        # This is informational — the verification step will use it
        for rec in records:
            # We annotate by adding a marker to entity_id or a field
            # But to keep output clean, we just log it
            logger.debug(
                "Doc '%s': owner=%s, reason='%s', experience=%s",
                doc.document_name, is_owner, reason, is_experience,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_doc(self, document_id: str) -> Optional[DetectedDocument]:
        """Find a detected document by ID from the flattened list."""
        for doc in self._all_detected_docs:
            if doc.document_id == document_id:
                return doc
        return None

    @staticmethod
    def _public_document_id(doc: DetectedDocument) -> str:
        """Return output-facing document id with source file provenance."""
        if doc.source_file:
            return f"{doc.source_file}::{doc.document_id}"
        return doc.document_id

    # ------------------------------------------------------------------
    # Input parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tender_plan(data: Union[Dict[str, Any], TenderInfoPlan]) -> TenderInfoPlan:
        if isinstance(data, TenderInfoPlan):
            return data
        return TenderInfoPlan.model_validate(data)

    @staticmethod
    def _parse_bid_documents(data: Union[List[Dict[str, Any]], List[BidDocument]]) -> List[BidDocument]:
        if isinstance(data, list) and data and isinstance(data[0], BidDocument):
            return data
        return [BidDocument.model_validate(d) for d in data]
