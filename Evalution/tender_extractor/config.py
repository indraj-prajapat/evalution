"""
Configuration for the extraction pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractionConfig:
    """Tunable parameters for the extraction pipeline."""

    # Minimum confidence to include an extracted value
    min_confidence: float = 0.5

    # Skip company ownership verification (now annotation-only by default)
    skip_company_match: bool = True

    # Logging level
    log_level: str = "INFO"

    # Output path (None = stdout only)
    output_path: Optional[str] = None

    # Debug mode
    debug: bool = False

    # Maximum snippet length for evidence
    max_snippet_length: int = 300

    # Token overlap threshold for document name matching
    doc_name_match_threshold: float = 0.5

    # Token overlap threshold for company name matching
    company_name_match_threshold: float = 0.7

    # --- LLM settings ---

    # Use LLM for field extraction
    use_llm_extraction: bool = True

    # Use LLM for document scoring
    use_llm_doc_scoring: bool = True

    # Maximum characters of document text per LLM extraction call
    llm_max_doc_chars: int = 6000

    # Retry count for LLM extraction calls
    llm_max_retries: int = 1

    # --- RAG settings ---

    # Enable RAG pipeline (BM25 + semantic + RRF)
    use_rag: bool = True

    # Sentence-transformer model name
    embedding_model: str = "all-MiniLM-L6-v2"

    # Top-k pages to retrieve per requirement (used for RAG scoring, NOT gating)
    rag_top_k: int = 5

    # RRF constant (higher = less emphasis on top rank)
    rag_rrf_k: int = 60

    # --- Document scoring settings (NEW) ---

    # Weight for RAG score in combined scoring (0-1)
    rag_score_weight: float = 0.4

    # Weight for LLM score in combined scoring (0-1)
    llm_score_weight: float = 0.6

    # Minimum combined score below which a document is excluded.
    # Set very low (0.05) to only exclude documents that are 100% sure unrelated.
    doc_exclusion_threshold: float = 0.05

    # Maximum chars of content hint per document for LLM scoring
    # (keeps token usage low — only filename + short content preview)
    llm_scoring_content_hint_chars: int = 200

    # Maximum documents per LLM scoring batch (to stay within token limits)
    llm_scoring_batch_size: int = 40

    # --- Safety net settings ---

    # Always run regex extraction on ALL documents (even excluded) as safety net
    regex_safety_net: bool = True

    # Run cross-document entity mention scanning
    cross_document_mentions: bool = True

    # Internal fields
    _logging_configured: bool = field(default=False, repr=False)

    def setup_logging(self) -> None:
        if self._logging_configured:
            return
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        self._logging_configured = True