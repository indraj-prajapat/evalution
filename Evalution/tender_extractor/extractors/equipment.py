"""Equipment entity extractor."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from .base import BaseExtractor


class EquipmentExtractor(BaseExtractor):
    entity_type = "Equipment"

    _PATTERNS = {
        "equipment_name": [
            re.compile(
                r"(?:equipment|plant|machinery|vehicle)\s*(?:name|description|type)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9 ]+?)(?:\n|$|\d)",
                re.IGNORECASE,
            ),
        ],
        "equipment_type": [
            re.compile(
                r"(?:type|category|class)\s*[:\-]?\s*([A-Za-z0-9 ]+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "ownership": [
            re.compile(
                r"(?:ownership|owned|leased|hired|rented)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "quantity": [
            re.compile(
                r"(?:quantity|no\.?\s*of\s+(?:equipment|units?)|count)\s*[:\-]?\s*(\d+)",
                re.IGNORECASE,
            ),
        ],
        "purchase_date": [
            re.compile(
                r"(?:purchase\s+date|date\s+of\s+purchase|acquisition\s+date)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "purchase_value": [
            re.compile(
                r"(?:purchase\s+(?:value|price|cost)|cost|value)\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?)",
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

        entity_id = f"equip_page_{page_num}"
        for f in fields:
            if f.name == "equipment_name":
                entity_id = str(f.value)[:80]
                break

        return [
            ExtractedRecord(
                entity_id=entity_id,
                group=group_by or "equipment",
                group_value=entity_id,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = (page_text + " " + page_summary).lower()
        keywords = ["equipment", "plant", "machinery", "vehicle", "owned", "leased", "purchase"]
        return any(kw in combined for kw in keywords)