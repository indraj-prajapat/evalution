"""Partner entity extractor."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from .base import BaseExtractor


class PartnerExtractor(BaseExtractor):
    entity_type = "Partner"

    _PATTERNS = {
        "partner_name": [
            re.compile(
                r"(?:partner|member|director|proprietor)\s*(?:\d+|name|s\.?no\.?)?\s*[:\-]?\s*([A-Z][A-Za-z. ]+?)(?:\n|$|\d)",
                re.IGNORECASE,
            ),
        ],
        "partner_pan": [
            re.compile(
                r"(?:partner|member|director)\s*\d*\s*[:\-]?\s*[A-Za-z. ]+?\s+(?:PAN)\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])",
                re.IGNORECASE,
            ),
        ],
        "partner_shareholding": [
            re.compile(
                r"(?:share\s*holding|share|stake|interest)\s*[:\-]?\s*([\d.]+\s*%?)",
                re.IGNORECASE,
            ),
        ],
        "partner_designation": [
            re.compile(
                r"(?:designation|capacity|role)\s*[:\-]?\s*([A-Za-z ]+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "partner_din": [
            re.compile(
                r"(?:DIN|Director\s*ID)\s*[:\-]?\s*(\d{8})",
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

        entity_id = f"partner_page_{page_num}"
        for f in fields:
            if f.name == "partner_name":
                entity_id = str(f.value)[:80]
                break

        return [
            ExtractedRecord(
                entity_id=entity_id,
                group=group_by or "partner",
                group_value=entity_id,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = (page_text + " " + page_summary).lower()
        keywords = ["partner", "director", "member", "proprietor", "shareholder", "din", "shareholding"]
        return any(kw in combined for kw in keywords)
