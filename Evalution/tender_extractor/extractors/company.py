"""Company entity extractor."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from ..utils import extract_pan, extract_gstin, extract_cin, truncate_snippet
from .base import BaseExtractor


class CompanyExtractor(BaseExtractor):
    entity_type = "Company"

    # Patterns for company fields
    _PATTERNS = {
        "company_name": [
            re.compile(
                r"(?:name\s+of\s+(?:the\s+)?(?:firm|company|bidder|contractor|proprietor|organisation|applicant))\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
            re.compile(r"M/[sS]\.?\s+(.+?)(?:\n|,|\.|$)"),
        ],
        "date_of_incorporation": [
            re.compile(
                r"(?:date\s+of\s+incorporation|incorporation\s+date|cin\s*date|registration\s+date)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "pan": [
            re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"),
        ],
        "gstin": [
            re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d])\b"),
        ],
        "cin": [
            re.compile(r"\b([A-Z][0-9]{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})\b"),
        ],
        "registered_address": [
            re.compile(
                r"(?:registered\s+office|office\s+address|address)\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|\n\d)",
                re.IGNORECASE | re.DOTALL,
            ),
        ],
        "constitution": [
            re.compile(
                r"(?:constitution|nature\s+of\s+entity|type\s+of\s+organisation)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "authorized_capital": [
            re.compile(
                r"(?:authorised\s+capital|authorized\s+capital)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "paid_up_capital": [
            re.compile(
                r"(?:paid.?up\s+capital|subscribed\s+capital)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
    }

    def extract_from_page(
        self,
        page: PageSummary,
        doc: DetectedDocument,
        required_fields: List[FieldSpec],
        group_by: Optional[str] = None,
    ) -> List[ExtractedRecord]:
        text = page.text or ""
        page_num = page.page_number
        fields: List[ExtractedField] = []

        for fs in required_fields:
            if fs.name in self._PATTERNS:
                for pattern in self._PATTERNS[fs.name]:
                    m = pattern.search(text)
                    if m:
                        raw = m.group(1).strip()
                        # For identifier fields, use the whole match
                        if fs.name in ("pan", "gstin", "cin"):
                            raw = m.group(0).strip()
                        field = self._build_field(
                            name=fs.name,
                            raw_value=raw,
                            datatype=fs.datatype,
                            page=page_num,
                            document_id=doc.document_id,
                            document_name=doc.document_name,
                            full_page_text=text,
                        )
                        if field:
                            fields.append(field)
                        if not fs.repeatable:
                            break

        if not fields:
            return []

        return [
            ExtractedRecord(
                entity_id=doc.document_id,
                group=group_by,
                group_value=None,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = page_text + " " + page_summary
        company_keywords = [
            "company", "firm", "bidder", "proprietor", "organisation",
            "incorporation", "pan", "gstin", "cin", "registered office",
            "constitution", "memorandum", "articles of association",
        ]
        return any(kw in combined for kw in company_keywords)