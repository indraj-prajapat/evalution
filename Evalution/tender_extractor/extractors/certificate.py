"""Certificate entity extractor."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from .base import BaseExtractor


class CertificateExtractor(BaseExtractor):
    entity_type = "Certificate"

    _PATTERNS = {
        "certificate_number": [
            re.compile(
                r"(?:certificate\s*(?:no|number|No\.?)|cert\.?\s*no\.?)\s*[:\-]?\s*([A-Za-z0-9/\-]+)",
                re.IGNORECASE,
            ),
        ],
        "certificate_type": [
            re.compile(
                r"(?:certificate|cert)\s*(?:of)?\s*([A-Za-z ]+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "issuing_authority": [
            re.compile(
                r"(?:issued\s+by|issuing\s+authority|certified\s+by|signed\s+by)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "date_of_issue": [
            re.compile(
                r"(?:date\s+of\s+(?:issue|issuance)|issued\s+(?:on|date))\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "valid_from": [
            re.compile(
                r"(?:valid\s+from|validity\s+from|period\s+from)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "valid_till": [
            re.compile(
                r"(?:valid\s+till|valid\s+until|validity\s+(?:till|until|to)|period\s+to)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "certificate_holder": [
            re.compile(
                r"(?:this\s+is\s+to\s+certify\s+that|certified\s+that)\s+(.+?)(?:\n|has|have)",
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
                group=group_by or "certificate",
                group_value=None,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = (page_text + " " + page_summary).lower()
        keywords = ["certificate", "certified", "certify", "issuing authority", "signed by"]
        return any(kw in combined for kw in keywords)