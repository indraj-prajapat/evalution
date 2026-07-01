"""Compliance Record entity extractor."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from .base import BaseExtractor


class ComplianceRecordExtractor(BaseExtractor):
    entity_type = "ComplianceRecord"

    _PATTERNS = {
        "compliance_type": [
            re.compile(
                r"(?:compliance|statutory\s+compliance|regulatory)\s*(?:type|category|requirement)?\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "compliance_status": [
            re.compile(
                r"(?:status|compliance\s+status)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "authority": [
            re.compile(
                r"(?:authority|regulating\s+body|compliance\s+authority)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "valid_from": [
            re.compile(
                r"(?:valid\s+from|effective\s+from|period\s+from)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "valid_till": [
            re.compile(
                r"(?:valid\s+till|valid\s+until|validity)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "certificate_number": [
            re.compile(
                r"(?:certificate\s*(?:no|number|No\.?)|cert\.?\s*no\.?)\s*[:\-]?\s*([A-Za-z0-9/\-]+)",
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
                group=group_by or "compliance",
                group_value=None,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = (page_text + " " + page_summary).lower()
        keywords = ["compliance", "statutory", "regulatory", "pf", "esi", "epfo", "professional tax"]
        return any(kw in combined for kw in keywords)