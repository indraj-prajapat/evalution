"""
Tender Information Extraction Agent v4.0.

Scoring-based pipeline (RAG score + LLM score per document) with
GPT-4o-mini extraction. Never misses evidence.

Key improvements over v3.1:
  - Document SCORING instead of FILTERING (only exclude when 100% sure)
  - LLM scores each document by filename + 200 char content hint
  - Regex safety net on ALL documents including excluded
  - Cross-document entity mention tracking
  - Company matching is annotation-only (not a gate)
  - Token-efficient LLM scoring (batched, minimal content)

Usage::

    from tender_extractor import extract_from_bid

    result = extract_from_bid(
        company_json=company_data,
        tender_plan=tender_plan_data,
    )
"""

from .api import extract_from_bid
from .extractor import TenderExtractor
from .models import ExtractionResult
from .config import ExtractionConfig

__version__ = "4.0.0"
__all__ = [
    "extract_from_bid",
    "TenderExtractor",
    "ExtractionResult",
    "ExtractionConfig",
]