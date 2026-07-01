"""Tax Record entity extractor (GST, Income Tax, etc.)."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from ..utils import extract_gstin, extract_pan
from .base import BaseExtractor


class TaxRecordExtractor(BaseExtractor):
    entity_type = "TaxRecord"

    _PATTERNS = {
        "registration_number": [
            re.compile(
                r"(?:GSTIN|GST\s*(?:No|Number|Registration))\s*[:\-]?\s*(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d])",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:registration\s*(?:no|number|No\.?)|reg\.?\s*no\.?)\s*[:\-]?\s*([A-Za-z0-9/\-]+)",
                re.IGNORECASE,
            ),
        ],
        "tax_type": [
            re.compile(
                r"(?:tax\s+type|type\s+of\s+(?:tax|registration))\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "registration_date": [
            re.compile(
                r"(?:date\s+of\s+registration|registration\s+date)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "taxpayer_name": [
            re.compile(
                r"(?:taxpayer\s+name|name\s+of\s+(?:taxpayer|assessee))\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "state": [
            re.compile(
                r"(?:state|state\s+code)\s*[:\-]?\s*([A-Za-z ]+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "pan": [
            re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"),
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
                    for m in pattern.finditer(text):
                        raw = m.group(1).strip()
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
                    if not fs.repeatable and fields:
                        break

        if not fields:
            return []

        return [
            ExtractedRecord(
                entity_id=doc.document_id,
                group=group_by or "tax_record",
                group_value=None,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = (page_text + " " + page_summary).lower()
        keywords = ["gst", "gstin", "tax", "registration", "income tax", "vat", "tds"]
        return any(kw in combined for kw in keywords)