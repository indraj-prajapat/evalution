"""
Evidence Filter - LLM-based evidence relevance filtering.

Pipeline step: Given a criterion and ALL extracted evidence, use LLM to
determine which evidence items are actually related to the criterion.
Then verify related evidence against Company JSON page-level text.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from .models import Evidence, ExtractedField, MatchedDocument, CompanyJSON, Module2Output

load_dotenv()


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FilteredEvidence:
    """An evidence item with LLM relevance assessment."""
    original_index: int
    document_id: str
    document_name: str
    page: int
    field_name: str
    field_value: str
    snippet: str
    confidence: float
    requirement_document: str = ""
    document_status: str = ""
    is_document_level: bool = False
    is_related: bool = False
    relevance_reason: str = ""
    what_it_shows: str = ""
    extracted_data_points: list[dict] = field(default_factory=list)
    # Page-level verification
    page_text_verified: bool = False
    page_text_match: Optional[str] = None  # Exact text found on the page
    page_text_summary: str = ""  # Brief summary of what the page text confirms


@dataclass
class EvidenceFilterResult:
    """Result of evidence filtering."""
    total_evidence: int = 0
    related_count: int = 0
    unrelated_count: int = 0
    related_items: list[FilteredEvidence] = field(default_factory=list)
    unrelated_items: list[FilteredEvidence] = field(default_factory=list)
    criterion_requirements: list[dict] = field(default_factory=list)
    filter_method: str = ""  # "llm" or "keyword_fallback"


# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

_FILTER_SYSTEM_PROMPT = """You are an expert document analyst for tender compliance verification.
Your job is to analyze a tender criterion and a list of extracted evidence items,
then determine which evidence items are RELATED to verifying the criterion.

## CRITICAL RULES:
1. An evidence item is RELATED only if it directly helps verify whether the criterion is met
2. An evidence item is UNRELATED if it's about a different requirement entirely
3. For each related item, explain WHAT it shows about the criterion
4. If an item contains numeric data (amounts, dates, counts), extract those as data_points
5. NEVER do mathematical calculations — just identify the data points
6. If a data point is a financial figure (turnover, revenue, etc.), ALSO record
   the financial year or as-at date it belongs to in a "period" field, using
   whatever the document states (a "FY2022-23" label, an "as at 31 March 2023"
   date, a document title year, etc.). If no period is stated anywhere for
   that figure, leave "period" as an empty string — do not guess.
6. Be strict: only mark items as related if they genuinely help verify the criterion
7. Do NOT hallucinate — only reference what is explicitly in the evidence text
8. NEVER mark a numeric evidence item as UNRELATED because you think its value
   FAILS the criterion. If an item carries a number relevant to the criterion's
   subject (turnover, amount, count, duration, year, etc.), it is ALWAYS RELATED
   regardless of whether the value seems too low/high. Python decides pass/fail
   on numbers — your job is only to surface the number, never to judge it.
"""

_FILTER_USER_PROMPT = """## TENDER CRITERION
{criterion}

## EVIDENCE ITEMS
{evidence_section}

## YOUR TASK
For EACH evidence item above, determine:
1. Is it RELATED to verifying the criterion? (true/false)
2. If related, what does it show? (1 sentence)
3. If related, what specific data points does it contain? (e.g., amounts, dates, names, counts)

Also, break down the criterion into specific checkable requirements.

Respond in JSON format:
{{
  "criterion_requirements": [
    {{"requirement": "description of what needs to be checked", "check_type": "existence|threshold|count|comparison", "expected": "what value/type is expected"}}
  ],
  "evidence_assessment": [
    {{
      "index": 0,
      "is_related": true/false,
      "relevance_reason": "why it is or isn't related to the criterion",
      "what_it_shows": "if related: what specific information this evidence provides",
      "data_points": [{{"field": "field name", "value": "extracted value", "is_numeric": true/false, "period": "e.g. FY2022-23, or as-at date, or empty string if not stated"}}]
    }}
  ]
}}"""

_FILTER_SYSTEM_PROMPT += """

## REQUIREMENT LOGIC ADDENDUM
Before judging evidence relevance, break the criterion into:
- mandatory AND requirements
- OR / either alternatives
- exception, exemption, waiver, and substitute paths

Evidence for an exception or exemption is RELATED evidence. For example, if a
criterion requires EMD unless an exemption certificate is present, an exemption
certificate is directly related because it may satisfy the exception path.

Document-level items with no extracted fields can still be RELATED when the
document name, status, page, or summary helps prove a required document,
alternative branch, exception, exemption, waiver, or substitution.
"""

_FILTER_USER_PROMPT += """

When returning criterion_requirements, include these fields whenever possible:
- condition_type: mandatory|alternative|exception|waiver|substitute
- logic_group: AND group / OR group / exception group label
- expected: expected document, value, text, or condition
"""


# ---------------------------------------------------------------------------
# Keyword Fallback (when LLM is not available)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "and", "for", "should", "have", "has", "had", "with",
    "from", "been", "being", "shall", "will", "must", "may",
    "this", "that", "which", "whose", "where", "when", "what",
    "not", "but", "are", "was", "were", "is", "am", "be",
    "any", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "than", "too", "very",
    "can", "just", "into", "over", "also", "then", "once",
    "copy", "submitted", "bidder", "firm", "company", "case",
    "per", "or", "of", "in", "to", "by", "as", "an", "a",
}


def _extract_criterion_keywords(criterion: str) -> set[str]:
    """Extract meaningful keywords from criterion for fallback filtering."""
    keywords = set()

    # Extract capitalized terms (like CISA, DISA, ITR)
    caps = re.findall(r"\b([A-Z]{2,}(?:/[A-Z]{2,})*)\b", criterion)
    for c in caps:
        for part in c.split("/"):
            keywords.add(part.lower())

    # Extract terms after "copy of" or "shall"
    for pat in [r"copy of\s+([A-Za-z][\w\s]*?)(?:\s+shall|\s+and|$)"]:
        for m in re.finditer(pat, criterion, re.IGNORECASE):
            words = m.group(1).split()
            for w in words:
                if w.lower() not in _STOP_WORDS and len(w) >= 3:
                    keywords.add(w.lower())

    # Generic words
    words = re.findall(r"\b[A-Za-z]{3,}\b", criterion)
    for w in words:
        if w.lower() not in _STOP_WORDS:
            keywords.add(w.lower())

    return keywords


# ---------------------------------------------------------------------------
# Evidence Filter
# ---------------------------------------------------------------------------

class EvidenceFilter:
    """
    Filters evidence items based on relevance to the criterion.

    Primary method: LLM-based analysis
    Fallback: Keyword-based matching (when LLM unavailable)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                return None
            from Evalution.client import get_openrouter_client
            self._client = get_openrouter_client(api_key=self.api_key, model=self.model)
        return self._client

    def _collect_all_evidence(self, module2: Module2Output) -> list[FilteredEvidence]:
        """Collect ALL evidence items from Module 2 output."""
        items = []
        idx = 0
        for req in module2.documents:
            for doc in req.matched_documents:
                first_page = doc.pages[0] if doc.pages else 0
                pages_text = ", ".join(str(p) for p in doc.pages) if doc.pages else "unknown"
                items.append(FilteredEvidence(
                    original_index=idx,
                    document_id=doc.document_id,
                    document_name=doc.document_name,
                    page=first_page,
                    field_name="document_presence",
                    field_value=doc.document_name,
                    snippet=(
                        f"Requirement document: {req.requirement_document}. "
                        f"Matched document status: {doc.status}. "
                        f"Pages: {pages_text}. "
                        f"Summary: {doc.summary or doc.document_name}"
                    ),
                    confidence=1.0 if doc.is_found else 0.7,
                    requirement_document=req.requirement_document,
                    document_status=doc.status,
                    is_document_level=True,
                ))
                idx += 1
                for rec in doc.records:
                    for f in rec.fields:
                        items.append(FilteredEvidence(
                            original_index=idx,
                            document_id=doc.document_id,
                            document_name=doc.document_name,
                            page=f.page,
                            field_name=f.name,
                            field_value=f.value,
                            snippet=f.snippet or "",
                            confidence=f.confidence,
                            requirement_document=req.requirement_document,
                            document_status=doc.status,
                            is_document_level=False,
                        ))
                        idx += 1
        return items

    def _build_evidence_section(self, items: list[FilteredEvidence]) -> str:
        """Build the evidence section for the LLM prompt."""
        lines = []
        for i, item in enumerate(items):
            lines.append(
                f"[{i}] Document: {item.document_name}, ID: {item.document_id}, Page: {item.page}\n"
                f"    Requirement: {item.requirement_document}, Status: {item.document_status}\n"
                f"    Field: {item.field_name} = {item.field_value}\n"
                f"    Snippet: {item.snippet[:300]}{'...' if len(item.snippet) > 300 else ''}"
            )
        return "\n\n".join(lines)

    def _llm_filter(
        self,
        criterion: str,
        items: list[FilteredEvidence],
    ) -> EvidenceFilterResult:
        """Use LLM to filter evidence by relevance."""
        result = EvidenceFilterResult(
            total_evidence=len(items),
            filter_method="llm",
        )

        # Truncate evidence list if too large for LLM context.
        # Use 200 to avoid dropping critical evidence when many PARTIAL docs exist.
        max_items = 200
        truncated_items = items[:max_items]
        truncation_note = ""
        if len(items) > max_items:
            truncation_note = (
                f"\n\nNOTE: Only showing first {max_items} of {len(items)} "
                f"evidence items. Remaining items will use keyword fallback."
            )

        evidence_section = self._build_evidence_section(truncated_items) + truncation_note
        user_prompt = _FILTER_USER_PROMPT.format(
            criterion=criterion,
            evidence_section=evidence_section,
        )

        try:
            client = self._get_client()
            if not client:
                return self._keyword_filter(criterion, items, "keyword_fallback (no API key)")

            response = client.create_chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": _FILTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content or "{}"
            result = self._parse_llm_response(raw, criterion, truncated_items, result)

            # If we truncated, apply keyword filtering to remaining items
            # so they don't all default to "unrelated"
            if len(items) > max_items:
                remaining_items = items[max_items:]
                keywords = _extract_criterion_keywords(criterion)
                for item in remaining_items:
                    searchable_text = (
                        f"{item.field_name} {item.field_value} {item.snippet} "
                        f"{item.document_name}"
                    ).lower()
                    is_related = any(
                        kw in searchable_text for kw in keywords
                    )
                    item.is_related = is_related
                    if is_related:
                        item.relevance_reason = (
                            "Keyword match (LLM truncation fallback)"
                        )
                        result.related_items.append(item)
                    else:
                        result.unrelated_items.append(item)

                result.related_count = len(result.related_items)
                result.unrelated_count = len(result.unrelated_items)
                result.filter_method = f"llm (first {max_items}) + keyword (remaining)"

            return result

        except Exception as e:
            # Fallback to keyword filtering
            return self._keyword_filter(criterion, items, f"keyword_fallback (LLM failed: {str(e)[:100]})")

    def _parse_llm_response(
        self,
        raw: str,
        criterion: str,
        items: list[FilteredEvidence],
        result: EvidenceFilterResult,
    ) -> EvidenceFilterResult:
        """Parse the LLM's JSON response."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self._keyword_filter(
                        criterion,
                        items,
                        "keyword_fallback (invalid LLM JSON)",
                    )
            else:
                return self._keyword_filter(
                    criterion,
                    items,
                    "keyword_fallback (no JSON in LLM response)",
                )

        # Extract criterion requirements
        result.criterion_requirements = data.get("criterion_requirements", [])

        # Process evidence assessment
        assessments = data.get("evidence_assessment", [])
        assessment_map = {}
        for a in assessments:
            idx = a.get("index", -1)
            assessment_map[idx] = a

        for i, item in enumerate(items):
            assessment = assessment_map.get(i, {})
            item.is_related = bool(assessment.get("is_related", False))
            item.relevance_reason = assessment.get("relevance_reason", "")
            item.what_it_shows = assessment.get("what_it_shows", "")
            item.extracted_data_points = assessment.get("data_points", [])

            if item.is_related:
                result.related_items.append(item)
            else:
                result.unrelated_items.append(item)

        result.related_count = len(result.related_items)
        result.unrelated_count = len(result.unrelated_items)

        return result

    def _keyword_filter(
        self,
        criterion: str,
        items: list[FilteredEvidence],
        filter_method_override: str = "",
    ) -> EvidenceFilterResult:
        """Keyword-based fallback when LLM is unavailable."""
        result = EvidenceFilterResult(
            total_evidence=len(items),
            filter_method=filter_method_override or "keyword_fallback",
        )

        keywords = _extract_criterion_keywords(criterion)

        for item in items:
            # Check if any keyword appears in the evidence
            searchable_text = (
                f"{item.field_name} {item.field_value} {item.snippet} {item.document_name}"
            ).lower()

            is_related = any(
                kw in searchable_text
                for kw in keywords
            )

            item.is_related = is_related
            if is_related:
                item.relevance_reason = "Keyword match against criterion terms"
                result.related_items.append(item)
            else:
                result.unrelated_items.append(item)

        result.related_count = len(result.related_items)
        result.unrelated_count = len(result.unrelated_items)

        return result

    def verify_against_page_text(
        self,
        filtered_items: list[FilteredEvidence],
        company: Optional[CompanyJSON],
    ) -> list[FilteredEvidence]:
        """
        For each related evidence item, go to the actual Company JSON page text
        and verify that the evidence is actually present on that page.

        This is the critical verification step — don't trust extractions blindly.
        """
        if not company:
            return filtered_items

        # Build page text map
        page_text_map: dict[int, str] = {}
        for pt in company.page_texts:
            m = re.search(r"page[_\s](\d+)", pt.page_key, re.IGNORECASE)
            if m:
                page_text_map[int(m.group(1))] = pt.text

        for item in filtered_items:
            page_text = page_text_map.get(item.page)
            if not page_text:
                continue

            item.page_text_verified = True

            if item.is_document_level:
                search_terms = [
                    term
                    for term in re.findall(
                        r"[A-Za-z0-9]{3,}",
                        f"{item.document_name} {item.requirement_document}",
                    )
                    if term.lower() not in {"the", "and", "for", "with", "document"}
                ]
                match_idx = -1
                matched_term = ""
                for term in search_terms:
                    match_idx = page_text.lower().find(term.lower())
                    if match_idx >= 0:
                        matched_term = term
                        break
                if match_idx >= 0:
                    start = max(0, match_idx - 120)
                    end = min(len(page_text), match_idx + 380)
                    item.page_text_match = page_text[start:end].strip()
                    item.page_text_summary = (
                        f"Document-level evidence verified on page {item.page} "
                        f"near term '{matched_term}'"
                    )
                else:
                    item.page_text_match = page_text[:500].strip()
                    item.page_text_summary = (
                        f"Page {item.page} text is available for matched document "
                        f"'{item.document_name}'"
                    )
                continue

            # Search for the extracted value or snippet in the actual page text
            value = str(item.field_value or "").strip()
            snippet = str(item.snippet or "").strip()

            # Try exact match of value first
            if value and len(value) > 2 and value.lower() in page_text.lower():
                # Find the exact context around the match
                idx = page_text.lower().find(value.lower())
                start = max(0, idx - 80)
                end = min(len(page_text), idx + len(value) + 120)
                item.page_text_match = page_text[start:end].strip()
                item.page_text_summary = (
                    f"Confirmed: '{value}' found on page {item.page} "
                    f"in document '{item.document_name}'"
                )
            elif snippet and len(snippet) > 5:
                # Try matching a portion of the snippet
                snippet_words = snippet.split()[:5]
                for word in snippet_words:
                    if len(word) > 3 and word.lower() in page_text.lower():
                        idx = page_text.lower().find(word.lower())
                        start = max(0, idx - 80)
                        end = min(len(page_text), idx + len(word) + 120)
                        item.page_text_match = page_text[start:end].strip()
                        item.page_text_summary = (
                            f"Partial match on page {item.page}: "
                            f"related text found near '{word}'"
                        )
                        break
            else:
                item.page_text_summary = (
                    f"Could not verify extracted value '{value}' on page {item.page} "
                    f"of '{item.document_name}'. The exact text was not found."
                )

        return filtered_items

    def filter(
        self,
        module2: Module2Output,
        criterion: str,
        company: Optional[CompanyJSON] = None,
    ) -> EvidenceFilterResult:
        """
        Main entry point: collect, filter, and verify evidence.

        Returns EvidenceFilterResult with related/unrelated items.
        """
        # Step 1: Collect all evidence
        items = self._collect_all_evidence(module2)

        if not items:
            return EvidenceFilterResult(total_evidence=0, filter_method="none")

        # Step 2: Filter by relevance (LLM or keyword fallback)
        result = self._llm_filter(criterion, items)

        # Step 3: Verify related evidence against Company JSON page text
        result.related_items = self.verify_against_page_text(
            result.related_items, company
        )

        return result
