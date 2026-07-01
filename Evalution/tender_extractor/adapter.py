"""
Adapter — converts real-world company JSON + tender plan into internal models.

The actual company JSON from the document pipeline has this structure::

    {
      "filename.pdf": {
        "file_name": "...",
        "documents": {
          "documents": [
            {
              "doc_id": "doc_1",
              "doc_type": "certificate",
              "pages": [1],
              "document_name": "Balance Sheet ...",
              "entities": { "company_name": "...", "pan": "...", ... },
              "summary": "...",
            }
          ]
        },
        "pages": {
          "page_1": "full OCR text...",
          "page_2": "...",
        }
      }
    }

This module transforms it into the internal ``BidDocument`` / ``DetectedDocument``
models that the extraction pipeline expects.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .models import (
    BidDocument,
    DetectedDocument,
    DocumentMode,
    FieldSpec,
    PageSummary,
    RequirementDocument,
    TenderInfoPlan,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Company JSON → internal models
# ---------------------------------------------------------------------------

def adapt_company_json(raw: Dict[str, Any]) -> List[BidDocument]:
    """Convert the company JSON into ``BidDocument`` list."""
    if isinstance(raw, list):
        items = raw
    elif _is_company_json_format(raw):
        items = [raw]
    else:
        items = [raw]

    bid_docs: List[BidDocument] = []
    for item in items:
        pdf_data = _unwrap_pdf_entry(item)
        if pdf_data is None:
            continue
        bid_doc = _convert_single_pdf(pdf_data)
        if bid_doc:
            bid_docs.append(bid_doc)

    logger.info(
        "Adapted %d PDF(s) with %d total detected documents",
        len(bid_docs),
        sum(len(bd.detected_documents) for bd in bid_docs),
    )
    return bid_docs


def _is_company_json_format(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    for key in data:
        val = data[key]
        if isinstance(val, dict) and "documents" in val and "pages" in val:
            return True
    return False


def _unwrap_pdf_entry(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key, val in item.items():
        if isinstance(val, dict) and ("documents" in val or "pages" in val):
            return val
    if "documents" in item or "pages" in item:
        return item
    return None


def _convert_single_pdf(pdf_data: Dict[str, Any]) -> Optional[BidDocument]:
    file_name = pdf_data.get("file_name", pdf_data.get("relative_path", "unknown.pdf"))
    pages_map: Dict[str, str] = pdf_data.get("pages", {})

    docs_container = pdf_data.get("documents", {})
    if isinstance(docs_container, dict):
        doc_list = docs_container.get("documents", [])
    elif isinstance(docs_container, list):
        doc_list = docs_container
    else:
        doc_list = []

    if not doc_list:
        logger.warning("No detected documents in '%s'", file_name)
        return None

    summary_obj = pdf_data.get("summary", {})
    if isinstance(summary_obj, dict):
        overall_summary = (
            f"Total pages: {summary_obj.get('total_pages', '?')}, "
            f"Documents detected: {summary_obj.get('documents_detected', '?')}"
        )
    else:
        overall_summary = str(summary_obj)

    detected: List[DetectedDocument] = []
    for doc in doc_list:
        dd = _convert_detected_document(doc, pages_map, file_name)
        if dd:
            detected.append(dd)

    return BidDocument(
        pdf_id=file_name,
        overall_summary=overall_summary,
        detected_documents=detected,
    )


def _convert_detected_document(
    doc: Dict[str, Any],
    pages_map: Dict[str, str],
    source_file: str = "",
) -> Optional[DetectedDocument]:
    doc_id = doc.get("doc_id", "")
    if not doc_id:
        return None

    doc_name = doc.get("document_name", "")
    doc_type = doc.get("doc_type", "")
    doc_category = doc.get("category", "")
    summary = doc.get("summary", "")
    entities = doc.get("entities", {})
    page_numbers = doc.get("pages", [])

    page_summaries: List[PageSummary] = []
    for pn in page_numbers:
        page_key = f"page_{pn}"
        ocr_text = pages_map.get(page_key, "")
        page_summaries.append(
            PageSummary(
                page_number=int(pn),
                text=ocr_text,
                summary="",
            )
        )

    metadata: Dict[str, Any] = {}
    if isinstance(entities, dict):
        for key, val in entities.items():
            if val and val != "":
                metadata[key] = val

    category = _map_doc_type_to_category(doc_type, doc_name)

    return DetectedDocument(
        document_id=doc_id,
        document_name=doc_name,
        source_file=source_file,
        document_category=category,
        document_type=doc_type,
        summary=summary,
        pages=page_summaries,
        metadata=metadata,
    )


def _map_doc_type_to_category(doc_type: str, doc_name: str) -> str:
    type_lower = doc_type.lower()
    name_lower = doc_name.lower()

    if any(kw in name_lower for kw in [
        "balance sheet", "financial statement", "p&l",
        "profit and loss", "income statement", "turnover",
        "ca certificate", "audited",
    ]):
        return "Financial Documents"

    if any(kw in name_lower for kw in [
        "work order", "completion certificate", "performance certificate",
        "experience certificate", "contract agreement",
    ]):
        return "Experience Documents"

    if any(kw in name_lower for kw in ["gst", "tax", "tin", "pan card"]):
        return "Tax Documents"

    if any(kw in name_lower for kw in [
        "registration", "certificate of incorporation",
        "udyam", "msme",
    ]):
        return "Registration Documents"

    type_map = {
        "report": "Financial Documents",
        "certificate": "Certificate",
        "letter": "Letter",
        "contract": "Experience Documents",
        "agreement": "Agreement",
        "form": "Form",
    }
    return type_map.get(type_lower, type_lower.capitalize() if type_lower else "")


# ---------------------------------------------------------------------------
# Tender plan → internal models
# ---------------------------------------------------------------------------

def adapt_tender_plan(raw: Dict[str, Any]) -> TenderInfoPlan:
    """Convert the real tender plan format into ``TenderInfoPlan``.

    Handles both formats:
      - User's format: ``fields`` key, ``category``, ``priority``, ``required``
      - Old format: ``required_fields`` key, ``entity_type``

    Auto-detects ``entity_type`` from document/category names when absent.
    """
    criterion_id = raw.get("criterion_id", "")
    criterion = raw.get("criterion", "")

    req_docs_raw = raw.get("required_documents", [])
    req_docs: List[RequirementDocument] = []

    for rd in req_docs_raw:
        # Support both "fields" and "required_fields" keys
        fields_raw = rd.get("fields", rd.get("required_fields", []))
        field_specs = []
        effective_group_by = rd.get("group_by")

        for f in fields_raw:
            field_group = f.get("group_by", effective_group_by)
            fs = FieldSpec(
                name=f["name"],
                datatype=f.get("datatype", "string"),
                repeatable=f.get("repeatable", False),
                description=f.get("description", ""),
                required=f.get("required", True),
                examples=f.get("examples", []),
                group_by=field_group,
            )
            field_specs.append(fs)

        mode_str = rd.get("mode", "EXPLICIT")
        try:
            mode = DocumentMode(mode_str)
        except ValueError:
            mode = DocumentMode.EXPLICIT

        entity_type = rd.get("entity_type", "")
        if not entity_type:
            entity_type = _infer_entity_type(
                rd.get("document", ""),
                rd.get("category", ""),
                rd.get("expected_documents", []),
            )

        req_doc = RequirementDocument(
            document=rd.get("document", ""),
            mode=mode,
            category=rd.get("category", ""),
            expected_documents=rd.get("expected_documents", []),
            required_fields=field_specs,
            entity_type=entity_type,
            group_by=effective_group_by,
            priority=rd.get("priority", 1),
            required=rd.get("required", True),
            description=rd.get("description", ""),
        )
        req_docs.append(req_doc)

    return TenderInfoPlan(
        criterion_id=criterion_id,
        criterion=criterion,
        required_documents=req_docs,
    )


def _infer_entity_type(
    document: str,
    category: str,
    expected_documents: List[str],
) -> str:
    # Category is the strongest signal — check it first
    cat_lower = category.lower()
    doc_lower = document.lower()
    expected_lower = " ".join(expected_documents).lower()
    combined = (doc_lower + " " + cat_lower + " " + expected_lower)

    # Category-level matches (highest priority)
    if any(kw in cat_lower for kw in [
        "experience", "work order",
    ]):
        return "Experience"

    if any(kw in cat_lower for kw in [
        "financial", "turnover", "balance sheet",
    ]):
        return "FinancialYear"

    if any(kw in cat_lower for kw in [
        "tax", "gst",
    ]):
        return "TaxRecord"

    if any(kw in cat_lower for kw in [
        "registration", "incorporation",
    ]):
        return "Company"

    # Document/expected name matches
    if any(kw in combined for kw in [
        "work order", "completion certificate", "experience",
        "performance certificate", "project",
    ]):
        return "Experience"

    if any(kw in combined for kw in [
        "balance sheet", "financial", "turnover", "p&l",
        "profit", "audited", "ca certificate",
    ]):
        return "FinancialYear"

    if any(kw in combined for kw in [
        "gst", "tax", "tds", "income tax", "gstin",
    ]):
        return "TaxRecord"

    if any(kw in combined for kw in [
        "company", "incorporation", "cin", "memorandum",
    ]):
        return "Company"

    if any(kw in combined for kw in [
        "certificate", "registration", "license",
    ]):
        return "Certificate"

    if any(kw in combined for kw in [
        "equipment", "plant", "machinery", "vehicle",
    ]):
        return "Equipment"

    if any(kw in combined for kw in [
        "personnel", "staff", "employee", "key personnel",
    ]):
        return "Personnel"

    if any(kw in combined for kw in [
        "compliance", "statutory", "pf", "esi",
    ]):
        return "ComplianceRecord"

    if any(kw in combined for kw in [
        "declaration", "undertaking", "affidavit",
    ]):
        return "Declaration"

    return "Other"
