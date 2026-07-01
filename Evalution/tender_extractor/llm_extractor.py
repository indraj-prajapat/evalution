"""
LLM-based field extractor with RAG integration.

Receives pre-filtered pages from the RAG retriever (or falls back to
full document pages). Sends only relevant page text to GPT-4o-mini.

Anti-hallucination strategy:
  1. Prompt instructs model to return EXACT raw text from document
  2. Post-processing validates each value through the normalizer
  3. Normalizer rejects values that don't parse (currency, date, etc.)
  4. Evidence snippet must appear in source text — verified at output time
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .llm_client import chat_json
from .models import (
    DetectedDocument,
    ExtractedField,
    ExtractedRecord,
    FieldSpec,
    PageSummary,
)
from .evidence import EvidenceTracker
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are a precise document information extractor for Indian tender documents.
Your job is to extract EXACT values from the given document text.

CRITICAL RULES:
1. Return ONLY values that are EXPLICITLY stated in the document text.
2. For EVERY field, return the exact raw text as it appears in the document — do NOT reformat, calculate, or interpret.
3. For currency fields: return the number string exactly as written (e.g. "4,56,78,900" or "Rs. 2.5 Crore").
4. For date fields: return the date exactly as written (e.g. "15 March 2020" or "01/04/2023").
5. For boolean fields: return the exact word found (e.g. "Yes", "Complied", "Available").
6. For string fields: return the exact text found, trimmed of leading/trailing whitespace.
7. If a field value is NOT found in the document, return null for that field.
8. Do NOT guess, infer, calculate, or derive any value.
9. Do NOT combine information from multiple places.
10. Return the "page" number where you found each value.
11. Return the exact "snippet" (15-50 chars surrounding the value) as it appears in the text.
12. For person/entity names (certificate_holder, partner_name, etc.), return the EXACT name as written — do not verify or cross-check against other documents. Just extract what you see.
"""

# Max chars of document text per LLM call.
_MAX_DOC_TEXT_CHARS = 6000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_fields_with_llm(
    doc: DetectedDocument,
    required_fields: List[FieldSpec],
    group_by: Optional[str] = None,
    tracker: Optional[EvidenceTracker] = None,
    max_retries: int = 1,
    rag_result: Optional[RetrievalResult] = None,
) -> List[ExtractedRecord]:
    """Extract all requested fields from *doc* using GPT-4o-mini.

    Parameters
    ----------
    doc : the matched detected document (may have filtered pages from RAG)
    required_fields : fields requested by the tender plan
    group_by : optional grouping key
    tracker : evidence tracker for building fields
    max_retries : retry on empty/invalid response
    rag_result : optional RAG retrieval result with page-level scores

    Returns
    -------
    List of ExtractedRecord (one per group, or one if no grouping).
    """
    if tracker is None:
        tracker = EvidenceTracker()

    # Build the user prompt with document text and field specs
    user_prompt = _build_extraction_prompt(doc, required_fields)

    # Call LLM
    result = _call_extraction_llm(user_prompt, required_fields, max_retries)

    if not result or "fields" not in result:
        logger.warning("LLM returned no extractable fields for '%s'", doc.document_name)
        return []

    # Convert LLM response to ExtractedRecords
    records = _convert_llm_response(
        llm_fields=result["fields"],
        doc=doc,
        required_fields=required_fields,
        group_by=group_by,
        tracker=tracker,
    )

    logger.info(
        "LLM extracted %d record(s), %d field(s) from '%s'",
        len(records),
        sum(len(r.fields) for r in records),
        doc.document_name,
    )
    return records


def select_documents_with_llm(
    requirement_name: str,
    requirement_category: str,
    expected_documents: List[str],
    available_docs: List[DetectedDocument],
    mode: str = "EXPLICIT",
) -> List[DetectedDocument]:
    """Use LLM to select matching documents when regex fails.

    This is a FALLBACK — RAG + rule-based is tried first.
    """
    if not available_docs:
        return []

    doc_summaries = []
    for i, d in enumerate(available_docs):
        doc_summaries.append(
            f"{i}. ID={d.document_id} | Name={d.document_name} | "
            f"Category={d.document_category} | Type={d.document_type} | "
            f"Summary={d.summary[:100]}"
        )

    user_prompt = (
        f"Requirement: '{requirement_name}'\n"
        f"Category: '{requirement_category}'\n"
        f"Mode: {mode}\n"
    )
    if expected_documents:
        user_prompt += f"Expected documents: {', '.join(expected_documents)}\n"

    user_prompt += (
        "\nAvailable documents:\n"
        + "\n".join(doc_summaries)
        + "\n\nWhich of these documents match the requirement? "
        "Return a JSON object with key 'matched_indices' containing a list "
        "of integer indices (0-based) of matching documents."
    )

    system_prompt = (
        "You are a document matching assistant for Indian tender evaluation. "
        "Match documents to requirements based on name, category, and type similarity. "
        "Be strict — only match when there is a clear relationship. "
        "Return JSON only."
    )

    result = chat_json(system_prompt, user_prompt, max_tokens=512)

    matched_indices = result.get("matched_indices", [])
    if not isinstance(matched_indices, list):
        return []

    selected = []
    for idx in matched_indices:
        if isinstance(idx, (int, float)) and 0 <= int(idx) < len(available_docs):
            selected.append(available_docs[int(idx)])

    if selected:
        logger.info(
            "LLM doc selection: matched %d doc(s) for '%s'",
            len(selected), requirement_name,
        )

    return selected


# ---------------------------------------------------------------------------
# Internal: build prompts
# ---------------------------------------------------------------------------

def _build_extraction_prompt(
    doc: DetectedDocument,
    required_fields: List[FieldSpec],
) -> str:
    """Build the user prompt for extraction."""
    # Gather document text from all pages (RAG may have filtered them)
    doc_text = _collect_doc_text(doc)

    # Build field specs as compact JSON
    fields_desc = []
    for f in required_fields:
        desc: Dict[str, Any] = {"name": f.name, "datatype": f.datatype}
        if f.description:
            desc["description"] = f.description
        if f.examples:
            desc["examples"] = f.examples
        if f.repeatable:
            desc["repeatable"] = True
        fields_desc.append(desc)

    fields_json = _compact_json(fields_desc)

    prompt = (
        f"## Document\n"
        f"Name: {doc.document_name}\n"
        f"Category: {doc.document_category}\n\n"
        f"## Document Text\n{doc_text}\n\n"
        f"## Fields to Extract\n{fields_json}\n\n"
        f"Extract each field from the document text above. "
        f"For repeatable fields, extract ALL occurrences.\n"
        f"Return a JSON object with key 'fields' containing an array of objects.\n"
        f"Each object must have:\n"
        f'- "name": field name\n'
        f'- "raw_value": exact text from document (or null if not found)\n'
        f'- "page": page number where found (integer)\n'
        f'- "snippet": 15-50 chars of surrounding text from the document\n'
    )

    return prompt


def _collect_doc_text(doc: DetectedDocument, max_chars: int = _MAX_DOC_TEXT_CHARS) -> str:
    """Collect page texts into a single string, respecting budget."""
    parts = []
    total = 0
    for page in doc.pages:
        text = page.text or ""
        if not text.strip():
            continue
        part = f"--- Page {page.page_number} ---\n{text}"
        if total + len(part) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(part[:remaining] + "\n...[truncated]")
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Internal: call LLM with retry
# ---------------------------------------------------------------------------

def _call_extraction_llm(
    user_prompt: str,
    required_fields: List[FieldSpec],
    max_retries: int,
) -> Optional[Dict[str, Any]]:
    """Call the LLM and validate the response."""
    for attempt in range(max_retries + 1):
        result = chat_json(
            system_prompt=_EXTRACTION_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=2048,
        )

        if result.get("error"):
            logger.warning("LLM error (attempt %d): %s", attempt + 1, result.get("error"))
            continue

        fields = result.get("fields", [])
        if not fields and required_fields:
            logger.debug("LLM returned empty fields (attempt %d)", attempt + 1)
            if attempt < max_retries:
                continue

        return result

    return None


# ---------------------------------------------------------------------------
# Internal: convert LLM response to ExtractedRecords
# ---------------------------------------------------------------------------

def _convert_llm_response(
    llm_fields: List[Dict[str, Any]],
    doc: DetectedDocument,
    required_fields: List[FieldSpec],
    group_by: Optional[str],
    tracker: EvidenceTracker,
) -> List[ExtractedRecord]:
    """Convert the LLM's JSON response into ExtractedRecord objects."""
    field_spec_map = {f.name: f for f in required_fields}
    all_fields: List[ExtractedField] = []

    for item in llm_fields:
        name = item.get("name", "")
        raw_value = item.get("raw_value")
        page = item.get("page", 0)
        snippet = item.get("snippet", "")

        if not name or raw_value is None:
            continue

        fs = field_spec_map.get(name)
        if fs is None:
            logger.debug("LLM returned unknown field '%s', skipping", name)
            continue

        full_page_text = ""
        for p in doc.pages:
            if p.page_number == page:
                full_page_text = p.text or ""
                break

        field = tracker.build_field(
            name=name,
            raw_value=str(raw_value),
            datatype=fs.datatype,
            page=int(page),
            document_id=doc.document_id,
            document_name=doc.document_name,
            full_page_text=full_page_text,
        )

        if field is not None:
            if snippet and len(snippet) > 10:
                from .utils import truncate_snippet
                field.snippet = truncate_snippet(snippet)
            all_fields.append(field)

    if not all_fields:
        return []

    if group_by:
        return _group_into_records(all_fields, group_by, doc)

    entity_id = doc.document_id
    for f in all_fields:
        if f.name in ("project_name", "company_name", "work_order_number",
                       "financial_year", "registration_number"):
            entity_id = str(f.value)[:80]
            break

    return [
        ExtractedRecord(
            entity_id=entity_id,
            group=group_by,
            group_value=None,
            fields=all_fields,
        )
    ]


def _group_into_records(
    fields: List[ExtractedField],
    group_by: str,
    doc: DetectedDocument,
) -> List[ExtractedRecord]:
    """Group fields by their group_by value."""
    group_field_value = None
    for f in fields:
        if f.name == group_by:
            group_field_value = str(f.value)
            break

    if group_field_value:
        return [
            ExtractedRecord(
                entity_id=group_field_value,
                group=group_by,
                group_value=group_field_value,
                fields=fields,
            )
        ]

    return [
        ExtractedRecord(
            entity_id=doc.document_id,
            group=group_by,
            group_value=None,
            fields=fields,
        )
    ]


def _compact_json(obj: Any) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"))