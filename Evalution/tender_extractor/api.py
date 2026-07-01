"""
Public API — the single function you call from your code.

Usage
-----
    from tender_extractor import extract_from_bid

    result = extract_from_bid(
        company_json=company_data,
        tender_plan=tender_plan_data,
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

from .adapter import adapt_company_json, adapt_tender_plan
from .config import ExtractionConfig
from .extractor import TenderExtractor
from .models import ExtractionResult

logger = logging.getLogger(__name__)


def extract_from_bid(
    company_json: Union[Dict[str, Any], str, Path],
    tender_plan: Union[Dict[str, Any], str, Path],
    file_path: Union[str, Path, None] = None,
    bidder_name: str = "",
    min_confidence: float = 0.5,
    debug: bool = False,
    use_llm: bool = True,
    use_rag: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2",
    rag_top_k: int = 15,
    llm_max_doc_chars: int = 6000,
    llm_max_retries: int = 1,
    skip_company_match: bool = True,
) -> Dict[str, Any]:
    """Extract structured atomic information from bidder documents.

    Uses a scoring pipeline (RAG score + LLM score per document) to
    find relevant documents, then GPT-4o-mini for extraction.

    Parameters
    ----------
    company_json :
        The company document JSON (dict, JSON string, or file path).
    tender_plan :
        The tender information plan (dict, JSON string, or file path).
        Format: ``criterion_id``, ``criterion``, ``required_documents[]``
        with ``fields`` key per document.
    bidder_name :
        Bidder company name. Auto-detected if empty.
    use_llm :
        Use GPT-4o-mini for extraction and scoring. Default: True.
    use_rag :
        Use RAG pipeline (BM25 + semantic + RRF) for document scoring.
        Default: True. Set False for LLM-only scoring.
    embedding_model :
        Sentence-transformer model name. Default: "all-MiniLM-L6-v2".
    rag_top_k :
        Max pages to retrieve per requirement for RAG scoring. Default: 5.
    llm_max_doc_chars :
        Max chars of page text per LLM extraction call. Default: 6000.
    llm_max_retries :
        Retry LLM extraction calls on failure. Default: 1.
    skip_company_match :
        Skip company ownership verification (now annotation-only by default).

    Returns
    -------
    dict
        Extraction result with ``criterion_id``, ``criterion``, ``documents[]``.
        Each document entry includes ``matched_documents`` (with extraction results)
        and ``excluded_documents`` (with scores and reasons for exclusion).

    Example
    -------
    ::

        from tender_extractor import extract_from_bid

        result = extract_from_bid(
            company_json=company_data,
            tender_plan=tender_plan_data,
        )
        print(json.dumps(result))
    """
    # 1. Load inputs
    plan_dict = _load_input(tender_plan, "tender_plan")
    company_dict = _load_input(company_json, "company_json")

    # 2. Auto-detect bidder name
    if not bidder_name:
        bidder_name = _auto_detect_bidder_name(company_dict)
    logger.info("Bidder: '%s'", bidder_name)

    # 3. Adapt to internal models
    internal_plan = adapt_tender_plan(plan_dict)
    internal_docs = adapt_company_json(company_dict)

    if not internal_docs:
        logger.warning("No bid documents could be parsed from company JSON")
        return _empty_result(plan_dict)

    # 4. Run extraction
    config = ExtractionConfig(
        min_confidence=min_confidence,
        debug=debug,
        skip_company_match=skip_company_match,
        log_level="DEBUG" if debug else "INFO",
        use_llm_extraction=use_llm,
        use_llm_doc_scoring=use_llm,
        llm_max_doc_chars=llm_max_doc_chars,
        llm_max_retries=llm_max_retries,
        use_rag=use_rag,
        embedding_model=embedding_model,
        rag_top_k=rag_top_k,
    )

    extractor = TenderExtractor(
        bidder_name=bidder_name,
        tender_plan=internal_plan,
        bid_documents=internal_docs,
        config=config,
    )
    result = extractor.run()

    # 5. Build output dict
    output = json.loads(result.to_json())
    output["criterion_id"] = plan_dict.get("criterion_id", "")
    output["criterion"] = plan_dict.get("criterion", "")

    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_input(data: Union[Dict[str, Any], str, Path], label: str) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    if isinstance(data, Path) or (isinstance(data, str) and _looks_like_path(data)):
        path = Path(data)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"{label}: file not found: {data}")
    if isinstance(data, str):
        return json.loads(data)
    raise TypeError(f"{label}: unsupported type {type(data)}")


def _looks_like_path(s: str) -> bool:
    return "." in s and len(s) < 500 and ("\\" in s or "/" in s)


def _auto_detect_bidder_name(company_dict: Dict[str, Any]) -> str:
    for pdf_key, pdf_data in company_dict.items():
        if not isinstance(pdf_data, dict):
            continue
        docs_container = pdf_data.get("documents", {})
        doc_list = (
            docs_container.get("documents", [])
            if isinstance(docs_container, dict)
            else docs_container
        )
        for doc in doc_list:
            entities = doc.get("entities", {})
            name = entities.get("company_name", "")
            if name and name.lower() not in ("", "disclaimer", "unknown"):
                return str(name)
    return "Unknown Company"


def _empty_result(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "criterion_id": plan_dict.get("criterion_id", ""),
        "criterion": plan_dict.get("criterion", ""),
        "documents": [],
    }