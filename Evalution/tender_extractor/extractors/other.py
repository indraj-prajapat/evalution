"""Other / fallback entity extractor.

Used when the entity type is not one of the known types.
Falls back to keyword-based extraction from the OCR text.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from .base import BaseExtractor


class OtherExtractor(BaseExtractor):
    entity_type = "Other"

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
            # Build a generic pattern from the field name
            # e.g. "annual_turnover" -> "annual.turnover"
            pattern_str = fs.name.replace("_", r"[\s_\-]+")
            pattern = re.compile(
                rf"(?:{pattern_str})\s*[:\-]?\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            )

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
        # For "Other" type, consider all pages relevant
        return True