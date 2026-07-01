"""Declaration entity extractor."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from .base import BaseExtractor


class DeclarationExtractor(BaseExtractor):
    entity_type = "Declaration"

    _PATTERNS = {
        "declaration_type": [
            re.compile(
                r"(?:declaration|undertaking|affidavit)\s*(?:of|for)?\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "declarant_name": [
            re.compile(
                r"(?:declared\s+by|signed\s+by|undertaken\s+by|affirmed\s+by|deponent)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:name|declarant)\s*[:\-]?\s*([A-Z][A-Za-z. ]+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "declaration_date": [
            re.compile(
                r"(?:date|date\s+of\s+declaration|dated)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "place": [
            re.compile(
                r"(?:place|signed\s+at)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "designation": [
            re.compile(
                r"(?:designation|capacity|in\s+the\s+capacity\s+of)\s*[:\-]?\s*([A-Za-z ]+?)(?:\n|$)",
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
                group=group_by or "declaration",
                group_value=None,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = (page_text + " " + page_summary).lower()
        keywords = ["declaration", "undertaking", "affidavit", "declared", "solemnly affirm"]
        return any(kw in combined for kw in keywords)