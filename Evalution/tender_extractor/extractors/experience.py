"""Experience entity extractor.

Reuses Project patterns but specifically targets experience/work-order/
completion-certificate documents. The client name will differ from
the bidder name (expected).
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from .base import BaseExtractor


class ExperienceExtractor(BaseExtractor):
    entity_type = "Experience"

    _PATTERNS = {
        "project_name": [
            re.compile(
                r"(?:project\s+name|name\s+of\s+(?:the\s+)?project|description\s+of\s+work|work\s+description)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:title|subject)\s*[:\-]\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "client_name": [
            re.compile(
                r"(?:client|employer|authority|department|organisation|customer|awarded\s+by)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "nature_of_work": [
            re.compile(
                r"(?:nature\s+of\s+work|scope\s+of\s+work|type\s+of\s+work|work\s+type|description)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "completion_date": [
            re.compile(
                r"(?:date\s+of\s+completion|completion\s+date|completed\s+on|date\s+of\s+final\s+payment)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "start_date": [
            re.compile(
                r"(?:date\s+of\s+(?:start|commencement|award)|start\s+date|commencement\s+date|agreement\s+date)\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "contract_value": [
            re.compile(
                r"(?:contract\s+value|work\s+order\s+value|order\s+value|project\s+value|agreement\s+value|value\s+of\s+(?:the\s+)?work|tender\s+value)\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?)",
                re.IGNORECASE,
            ),
        ],
        "work_order_number": [
            re.compile(
                r"(?:work\s+order\s+(?:no|number|No\.?)|wo\s+no\.?|order\s+no\.?)\s*[:\-]?\s*([A-Za-z0-9/\-]+)",
                re.IGNORECASE,
            ),
        ],
        "project_location": [
            re.compile(
                r"(?:location|site|place)\s+(?:of\s+(?:the\s+)?work|address)?\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "duration": [
            re.compile(
                r"(?:duration|period)\s*(?:of\s+(?:the\s+)?work|of\s+contract)?\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ],
        "experience_certificate_number": [
            re.compile(
                r"(?:experience\s+certificate\s*(?:no|number|No\.?)|exp\.?\s*cert\.?\s*no\.?)\s*[:\-]?\s*([A-Za-z0-9/\-]+)",
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

        entity_id = doc.document_id
        for f in fields:
            if f.name == "project_name":
                entity_id = str(f.value)[:80]
                break
            elif f.name == "work_order_number":
                entity_id = str(f.value)
                break

        return [
            ExtractedRecord(
                entity_id=entity_id,
                group=group_by or "experience",
                group_value=entity_id,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = (page_text + " " + page_summary).lower()
        keywords = [
            "experience", "work order", "completion certificate",
            "performance certificate", "contract", "client", "employer",
        ]
        return any(kw in combined for kw in keywords)