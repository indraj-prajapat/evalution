"""
RAG Retriever — BM25 + Semantic Search + Reciprocal Rank Fusion.

Pipeline:
  1. Build query from requirement (name + category + field descriptions)
  2. BM25 search over document names, summaries, and page text
  3. Semantic search using sentence-transformer embeddings
  4. Reciprocal Rank Fusion (RRF) to combine rankings
  5. Return top-k pages (grouped by document)

This solves the problem where document names don't match but content
does (e.g., a certificate with an unexpected filename).

NEW: score_documents() method returns document-level RAG scores
for the scoring pipeline (used by DocumentScorer).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import BidDocument, DetectedDocument, PageSummary, RequirementDocument, FieldSpec
from .embedder import embed_texts, embed_query, cosine_similarity, init_embedder, get_embedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageHit:
    """A page retrieved by the RAG pipeline."""
    document_id: str
    document_name: str
    document_category: str
    page_number: int
    text: str
    score: float
    source: str  # "bm25", "semantic", "rule", "rrf"


@dataclass
class RetrievalResult:
    """Result of the RAG retrieval for one requirement."""
    query: str
    hits: List[PageHit] = field(default_factory=list)

    @property
    def top_pages(self) -> List[PageHit]:
        return self.hits

    def get_unique_documents(self) -> List[Tuple[DetectedDocument, List[PageHit]]]:
        """Group hits by document, preserving page ordering."""
        from collections import OrderedDict
        doc_pages: Dict[str, Tuple[DetectedDocument, List[PageHit]]] = OrderedDict()
        for hit in self.hits:
            if hit.document_id not in doc_pages:
                doc_pages[hit.document_id] = (
                    DetectedDocument(
                        document_id=hit.document_id,
                        document_name=hit.document_name,
                        document_category=hit.document_category,
                        pages=[],
                    ),
                    [],
                )
            doc, pages = doc_pages[hit.document_id]
            pages.append(hit)

        # Build page objects from hits
        result = []
        for doc_id, (doc, page_hits) in doc_pages.items():
            # Deduplicate pages and sort by page_number
            seen = set()
            unique_pages = []
            for ph in sorted(page_hits, key=lambda h: h.page_number):
                if ph.page_number not in seen:
                    seen.add(ph.page_number)
                    unique_pages.append(
                        PageSummary(
                            page_number=ph.page_number,
                            text=ph.text,
                        )
                    )
            doc.pages = unique_pages
            result.append((doc, page_hits))

        return result


# ---------------------------------------------------------------------------
# Tokenizer for BM25
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple word tokenization for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Index — built once, queried many times
# ---------------------------------------------------------------------------

class RetrievalIndex:
    """Pre-computes embeddings and BM25 index for all pages."""

    def __init__(self, bid_documents: List[BidDocument]):
        self._all_docs: List[DetectedDocument] = []
        self._all_pages: List[PageHit] = []  # flattened pages
        self._bm25_corpus: List[List[str]] = []
        self._bm25 = None
        self._page_embeddings: Optional[np.ndarray] = None
        self._page_texts: List[str] = []

        self._flatten_pages(bid_documents)

    def _flatten_pages(self, bid_documents: List[BidDocument]) -> None:
        """Flatten all documents into page-level entries."""
        for bid_doc in bid_documents:
            for doc in bid_doc.detected_documents:
                self._all_docs.append(doc)
                for page in doc.pages:
                    text = page.text or ""
                    if not text.strip():
                        continue
                    self._all_pages.append(
                        PageHit(
                            document_id=doc.document_id,
                            document_name=doc.document_name,
                            document_category=doc.document_category,
                            page_number=page.page_number,
                            text=text,
                            score=0.0,
                            source="",
                        )
                    )

    def build(self) -> None:
        """Build BM25 index and compute embeddings. Call once after init."""
        if not self._all_pages:
            logger.info("No pages to index")
            return

        logger.info("Indexing %d pages from %d documents", len(self._all_pages), len(self._all_docs))

        # --- BM25 index ---
        self._build_bm25()

        # --- Semantic embeddings ---
        self._build_embeddings()

        logger.info("Retrieval index built successfully")

    def _build_bm25(self) -> None:
        """Build BM25 index over page text + document metadata."""
        try:
            from rank_bm25 import BM25Okapi

            self._bm25_corpus = []
            for ph in self._all_pages:
                # Combine document name, category, and page text for BM25
                combined = f"{ph.document_name} {ph.document_category} {ph.text}"
                self._bm25_corpus.append(_tokenize(combined))

            self._bm25 = BM25Okapi(self._bm25_corpus)
            logger.debug("BM25 index built with %d documents", len(self._bm25_corpus))
        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 search disabled")
            self._bm25 = None

    def _build_embeddings(self) -> None:
        """Compute embeddings for all pages."""
        if not self._all_pages:
            return

        try:
            # Build text representations for embedding
            self._page_texts = []
            for ph in self._all_pages:
                # For embedding, use document name + summary of page (not full text)
                # Full text can be too long for the model
                summary_text = ph.text[:500] if len(ph.text) > 500 else ph.text
                self._page_texts.append(
                    f"{ph.document_name} {ph.document_category} {summary_text}"
                )

            self._page_embeddings = embed_texts(self._page_texts)
            logger.debug(
                "Computed %d page embeddings (dim=%d)",
                len(self._page_texts),
                self._page_embeddings.shape[1] if self._page_embeddings.ndim == 2 else 0,
            )
        except Exception as e:
            logger.warning("Failed to compute embeddings: %s", e)
            self._page_embeddings = None

    # ------------------------------------------------------------------
    # Document-level scoring (NEW)
    # ------------------------------------------------------------------

    def score_documents(
        self,
        requirement: RequirementDocument,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> Dict[str, float]:
        """Return document-level RAG scores.

        Runs the full RAG pipeline (rule + BM25 + semantic + RRF),
        then aggregates page-level hits into document-level scores.

        Returns {doc_id: normalized_score} for ALL documents that got any hit.
        Documents with zero hits get score 0.0 (caller handles default).
        """
        if not self._all_pages:
            return {}

        query = self._build_query(requirement)

        # Run all search methods
        rule_hits = self._rule_search(requirement)
        bm25_hits = self._bm25_search(query, top_k=top_k * 3)
        semantic_hits = self._semantic_search(query, top_k=top_k * 3)

        # RRF fusion
        fused = self._reciprocal_rank_fusion(
            rankings=[rule_hits, bm25_hits, semantic_hits],
            k=rrf_k,
        )

        # Aggregate by document: take max RRF score across all pages
        doc_scores: Dict[str, float] = {}
        for hit in fused:
            doc_id = hit.document_id
            if doc_id not in doc_scores or hit.score > doc_scores[doc_id]:
                doc_scores[doc_id] = hit.score

        # Normalize to 0-1 range
        if doc_scores:
            max_score = max(doc_scores.values())
            if max_score > 0:
                doc_scores = {k: round(v / max_score, 4) for k, v in doc_scores.items()}

        logger.debug(
            "RAG document scoring for '%s': %d docs scored, max=%.4f",
            requirement.document, len(doc_scores),
            max(doc_scores.values()) if doc_scores else 0,
        )

        return doc_scores

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(
        self,
        requirement: RequirementDocument,
        top_k: int = 5,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
        rrf_k: int = 60,
    ) -> RetrievalResult:
        """Retrieve relevant pages for a requirement using RAG.

        Parameters
        ----------
        requirement : the tender plan requirement
        top_k : max pages to return
        bm25_weight : weight for BM25 score (unused in RRF, kept for API compat)
        semantic_weight : weight for semantic score (unused in RRF)
        rrf_k : RRF constant (higher = less emphasis on rank)

        Returns
        -------
        RetrievalResult with ranked page hits.
        """
        if not self._all_pages:
            return RetrievalResult(query="")

        query = self._build_query(requirement)
        logger.debug("RAG query: '%s'", query[:200])

        # --- Rule-based pre-filter (cheap, exact matches) ---
        rule_hits = self._rule_search(requirement)

        # --- BM25 search ---
        bm25_hits = self._bm25_search(query, top_k=top_k * 2)

        # --- Semantic search ---
        semantic_hits = self._semantic_search(query, top_k=top_k * 2)

        # --- Reciprocal Rank Fusion ---
        fused = self._reciprocal_rank_fusion(
            rankings=[rule_hits, bm25_hits, semantic_hits],
            k=rrf_k,
        )

        # Take top-k
        result = RetrievalResult(query=query, hits=fused[:top_k])

        # --- LLM content-based fallback ---
        # If RAG found nothing, use GPT-4o-mini to read page content
        # and find relevant documents (handles completely mismatched filenames)
        if not result.hits and self._all_pages:
            logger.info(
                "RAG found 0 pages for '%s', trying LLM content fallback",
                requirement.document,
            )
            llm_hits = self._llm_content_search(requirement, top_k=top_k)
            if llm_hits:
                result = RetrievalResult(query=query, hits=llm_hits)
                logger.info(
                    "LLM content fallback found %d page(s) for '%s'",
                    len(llm_hits), requirement.document,
                )

        logger.info(
            "RAG retrieved %d pages for '%s' (rule=%d, bm25=%d, semantic=%d)",
            len(result.hits),
            requirement.document,
            len(rule_hits),
            len(bm25_hits),
            len(semantic_hits),
        )

        return result

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(requirement: RequirementDocument) -> str:
        """Build a search query from the requirement.

        Includes document name, category, expected names, field descriptions,
        and the criterion description for maximum retrieval coverage.
        """
        parts = [requirement.document]

        if requirement.category:
            parts.append(requirement.category)

        # Add expected document names
        for ed in requirement.expected_documents:
            parts.append(ed)

        # Add field names and descriptions for better semantic matching
        for fs in requirement.required_fields:
            parts.append(fs.name)
            if fs.description:
                parts.append(fs.description)

        # Add requirement description if present (often contains key context)
        if requirement.description:
            parts.append(requirement.description)

        # Extract and add key content terms for BM25 matching.
        combined = " ".join(parts).lower()
        keywords = re.findall(r"[a-z]{3,}", combined)
        # Deduplicate while preserving order
        seen = set()
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                parts.append(kw)

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def _rule_search(self, requirement: RequirementDocument) -> List[PageHit]:
        """Name/category/content-keyword matching — gives highest confidence hits.

        Three matching strategies:
        1. Document name match (exact, substring, or token overlap >= 0.5)
        2. Document category match
        3. **Content-keyword match** — checks if key terms from the requirement
           name appear in the page text (handles mismatched filenames)
        """
        import re as _re

        hits: List[PageHit] = []
        target_names = [requirement.document] + requirement.expected_documents

        # Extract content keywords from requirement name for page-text matching.
        name_keywords = set(_re.findall(r"[a-z]{4,}", requirement.document.lower()))
        for ed in requirement.expected_documents:
            name_keywords.update(_re.findall(r"[a-z]{4,}", ed.lower()))
        # Remove overly generic terms that would cause false positives
        name_keywords -= {"document", "certificate", "copy", "submitted", "shall", "should"}

        for i, ph in enumerate(self._all_pages):
            name_lower = ph.document_name.lower()
            cat_lower = ph.document_category.lower()
            text_lower = ph.text.lower()

            # --- Strategy 1: Document name match ---
            name_matched = False
            for target in target_names:
                t_lower = target.lower()
                if t_lower == name_lower or t_lower in name_lower or name_lower in t_lower:
                    hits.append(PageHit(
                        document_id=ph.document_id,
                        document_name=ph.document_name,
                        document_category=ph.document_category,
                        page_number=ph.page_number,
                        text=ph.text,
                        score=1.0,
                        source="rule",
                    ))
                    name_matched = True
                    break

            # --- Token-overlap name match (catches partial name matches) ---
            if not name_matched:
                for target in target_names:
                    t_tokens = set(_re.findall(r"[a-z0-9]+", target.lower()))
                    n_tokens = set(_re.findall(r"[a-z0-9]+", name_lower))
                    if t_tokens and n_tokens:
                        overlap = len(t_tokens & n_tokens)
                        jaccard = overlap / len(t_tokens | n_tokens)
                        if jaccard >= 0.4:
                            hits.append(PageHit(
                                document_id=ph.document_id,
                                document_name=ph.document_name,
                                document_category=ph.document_category,
                                page_number=ph.page_number,
                                text=ph.text,
                                score=0.85,
                                source="rule",
                            ))
                            name_matched = True
                            break

            if name_matched:
                continue

            # --- Strategy 2: Category match ---
            if requirement.category and requirement.category.lower() in cat_lower:
                hits.append(PageHit(
                    document_id=ph.document_id,
                    document_name=ph.document_name,
                    document_category=ph.document_category,
                    page_number=ph.page_number,
                    text=ph.text,
                    score=0.8,
                    source="rule",
                ))
                continue

            # --- Strategy 3: Content-keyword match in page text ---
            if name_keywords:
                matched_keywords = name_keywords & set(_re.findall(r"[a-z]{4,}", text_lower))
                if len(matched_keywords) >= 2:
                    hits.append(PageHit(
                        document_id=ph.document_id,
                        document_name=ph.document_name,
                        document_category=ph.document_category,
                        page_number=ph.page_number,
                        text=ph.text,
                        score=0.7,
                        source="rule",
                    ))

        return hits

    def _bm25_search(self, query: str, top_k: int) -> List[PageHit]:
        """BM25 search over page texts.

        Returns pages with BM25 score > 0 (any keyword match). The RRF fusion
        will down-rank weak matches naturally.
        """
        if self._bm25 is None:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get all indices with score > 0, sorted by score descending
        positive_mask = scores > 0
        if not positive_mask.any():
            return []

        # Sort positive scores and take top-k
        top_indices = np.argsort(scores[positive_mask])[::-1][:top_k]
        # Map back to original indices
        original_indices = np.where(positive_mask)[0][top_indices]

        hits = []
        for idx in original_indices:
            ph = self._all_pages[int(idx)]
            hits.append(PageHit(
                document_id=ph.document_id,
                document_name=ph.document_name,
                document_category=ph.document_category,
                page_number=ph.page_number,
                text=ph.text,
                score=float(scores[int(idx)]),
                source="bm25",
            ))

        return hits

    def _semantic_search(self, query: str, top_k: int) -> List[PageHit]:
        """Semantic search using embeddings + cosine similarity.

        Lowered threshold to 0.05 to avoid discarding potentially relevant pages.
        RRF fusion handles final ranking.
        """
        if self._page_embeddings is None:
            return []

        try:
            query_vec = embed_query(query)
            sims = cosine_similarity(query_vec, self._page_embeddings)
            if sims.ndim > 1:
                sims = sims.flatten()

            top_indices = np.argsort(sims)[::-1][:top_k]

            hits = []
            for idx in top_indices:
                if sims[idx] <= 0.05:  # lowered threshold
                    break
                ph = self._all_pages[int(idx)]
                hits.append(PageHit(
                    document_id=ph.document_id,
                    document_name=ph.document_name,
                    document_category=ph.document_category,
                    page_number=ph.page_number,
                    text=ph.text,
                    score=float(sims[int(idx)]),
                    source="semantic",
                ))

            return hits
        except Exception as e:
            logger.warning("Semantic search failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # LLM content-based fallback
    # ------------------------------------------------------------------

    def _llm_content_search(
        self,
        requirement: RequirementDocument,
        top_k: int = 5,
    ) -> List[PageHit]:
        """Use GPT-4o-mini to find pages with relevant content.

        This is the last resort when rule + BM25 + semantic all fail.
        """
        try:
            from .llm_client import chat_json
        except ImportError:
            logger.debug("LLM client not available, skipping content search")
            return []

        # Build a summary of each unique document (first 200 chars per page, max 3 pages)
        doc_summaries = []
        page_index_map: List[tuple] = []  # (doc_idx, page_idx) for each summary entry
        seen_docs: Dict[str, int] = {}  # document_id -> count of pages added

        for i, ph in enumerate(self._all_pages):
            doc_id = ph.document_id
            page_count = seen_docs.get(doc_id, 0)
            # Limit to 3 pages per document to keep prompt manageable
            if page_count >= 3:
                continue
            seen_docs[doc_id] = page_count + 1

            text_preview = ph.text[:300].replace("\n", " ").strip()
            doc_summaries.append(
                f"[{len(doc_summaries)}] Doc: {ph.document_name} | "
                f"Page {ph.page_number} | Text: {text_preview}"
            )
            page_index_map.append(i)

        if not doc_summaries:
            return []

        # Truncate summaries if too many (keep prompt under ~4000 chars)
        max_summaries = 30
        if len(doc_summaries) > max_summaries:
            doc_summaries = doc_summaries[:max_summaries]
            page_index_map = page_index_map[:max_summaries]

        user_prompt = (
            f"Requirement: '{requirement.document}'\n"
            f"Category: '{requirement.category}'\n"
        )
        if requirement.expected_documents:
            user_prompt += f"Expected: {', '.join(requirement.expected_documents)}\n"
        if requirement.required_fields:
            field_descs = [f.name for f in requirement.required_fields]
            user_prompt += f"Fields needed: {', '.join(field_descs)}\n"

        user_prompt += (
            f"\nPage excerpts from uploaded documents:\n"
            + "\n".join(doc_summaries)
            + "\n\nWhich page excerpts contain content related to the requirement? "
            "Consider the ACTUAL CONTENT of each page, not just the document name. "
            "A page about 'branch registration' or 'registered office' matches "
            "'Branch/Office Registered Certificate' even if the filename is different.\n"
            "Return JSON: {\"matched_indices\": [0, 2, ...]} (list of indices above)."
        )

        system_prompt = (
            "You are a document content analyst for Indian tender evaluation. "
            "Match pages to requirements based on their ACTUAL TEXT CONTENT. "
            "A registration certificate page mentioning 'branch', 'office', "
            "'registered' should match 'Branch/Office Registered Certificate'. "
            "Return JSON only."
        )

        try:
            result = chat_json(system_prompt, user_prompt, max_tokens=256)
        except Exception as e:
            logger.warning("LLM content search failed: %s", e)
            return []

        matched_indices = result.get("matched_indices", [])
        if not isinstance(matched_indices, list):
            return []

        hits = []
        for idx in matched_indices:
            if isinstance(idx, (int, float)) and 0 <= int(idx) < len(page_index_map):
                original_idx = page_index_map[int(idx)]
                ph = self._all_pages[original_idx]
                hits.append(PageHit(
                    document_id=ph.document_id,
                    document_name=ph.document_name,
                    document_category=ph.document_category,
                    page_number=ph.page_number,
                    text=ph.text,
                    score=0.6,
                    source="llm_content",
                ))

        return hits[:top_k]

    # ------------------------------------------------------------------
    # Rank fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        rankings: List[List[PageHit]],
        k: int = 60,
    ) -> List[PageHit]:
        """Combine multiple rankings using Reciprocal Rank Fusion.

        RRF(d) = sum_i  1 / (k + rank_i(d))

        Each PageHit is identified by (document_id, page_number).
        """
        from collections import defaultdict

        rrf_scores: Dict[tuple, float] = defaultdict(float)
        best_hit: Dict[tuple, PageHit] = {}

        for ranking in rankings:
            for rank, hit in enumerate(ranking):
                key = (hit.document_id, hit.page_number)
                rrf_scores[key] += 1.0 / (k + rank + 1)
                if key not in best_hit or hit.score > best_hit[key].score:
                    best_hit[key] = hit

        # Sort by RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        result = []
        for key in sorted_keys:
            hit = best_hit[key]
            result.append(PageHit(
                document_id=hit.document_id,
                document_name=hit.document_name,
                document_category=hit.document_category,
                page_number=hit.page_number,
                text=hit.text,
                score=round(rrf_scores[key], 4),
                source="rrf",
            ))

        return result

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_all_docs(self) -> List[DetectedDocument]:
        return list(self._all_docs)

    def total_pages(self) -> int:
        return len(self._all_pages)