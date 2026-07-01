"""Registration entity extractor."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from .base import BaseExtractor


class RegistrationExtractor(BaseExtractor):
    entity_type = "Registration"

    _PATTERNS = {
        "registration_number": [
            re.compile(
                r"(?:registration\s*(?:no|number|No\.?)|reg\.?\s*no\.?|registration\s*id)\s*[:\-]?\s*([A-Za-z0-9/\-]+)",
                re.IGNORECASE,
            ),
        ],
        "registration_date": [
            re.compile(
                r"(?:date\s+of\s+registration|registration\s+date)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "registering_authority": [
            re.compile(
                r"(?:registering\s+authority|registered\s+with|registrar)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "registration_type": [
            re.compile(
                r"(?:type\s+of\s+registration|registration\s+under)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "validity_period": [
            re.compile(
                r"(?:valid(?:ity)?\s*(?:period|from|till|until))\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "registered_address": [
            re.compile(
                r"(?:registered\s+(?:office\s+)?address|principal\s+place)\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|\n\d)",
                re.IGNORECASE | re.DOTALL,
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
                group=group_by or "registration",
                group_value=None,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = (page_text + " " + page_summary).lower()
        keywords = ["registration", "registered", "registrar", "registration number"]
        return any(kw in combined for kw in keywords)