"""Personnel entity extractor."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from .base import BaseExtractor


class PersonnelExtractor(BaseExtractor):
    entity_type = "Personnel"

    _PATTERNS = {
        "personnel_name": [
            re.compile(
                r"(?:name|employee\s+name|staff\s+name|personnel\s+name)\s*[:\-]?\s*([A-Z][A-Za-z. ]+?)(?:\n|$|\d)",
                re.IGNORECASE,
            ),
        ],
        "designation": [
            re.compile(
                r"(?:designation|designation|title|position|role|post)\s*[:\-]?\s*([A-Za-z ]+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "qualification": [
            re.compile(
                r"(?:qualification|education|degree|qualification)\s*[:\-]?\s*([A-Za-z. ]+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "experience_years": [
            re.compile(
                r"(?:experience|years?\s+of\s+experience|total\s+experience)\s*[:\-]?\s*([\d.]+\s*(?:years?|yrs?)?)",
                re.IGNORECASE,
            ),
        ],
        "date_of_joining": [
            re.compile(
                r"(?:date\s+of\s+joining|joining\s+date)\s*[:\-]?\s*(.+?)(?:\n|$)",
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

        entity_id = f"personnel_page_{page_num}"
        for f in fields:
            if f.name == "personnel_name":
                entity_id = str(f.value)[:80]
                break

        return [
            ExtractedRecord(
                entity_id=entity_id,
                group=group_by or "personnel",
                group_value=entity_id,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = (page_text + " " + page_summary).lower()
        keywords = ["personnel", "staff", "employee", "key personnel", "manpower", "team"]
        return any(kw in combined for kw in keywords)