"""Financial Year entity extractor."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import DetectedDocument, ExtractedField, ExtractedRecord, FieldSpec, PageSummary
from .base import BaseExtractor


class FinancialYearExtractor(BaseExtractor):
    entity_type = "FinancialYear"

    _FY_PATTERN = re.compile(
        r"(?:FY|financial\s+year)\s*[:\-]?\s*(\d{4})\s*[-–/]\s*(\d{2,4})",
        re.IGNORECASE,
    )

    _PATTERNS = {
        "annual_turnover": [
            re.compile(
                r"(?:annual\s+)?turnover\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?(?:\.\d{2})?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:total\s+)?(?:revenue|income|turnover)\s*(?:from\s+operations)?\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?)",
                re.IGNORECASE,
            ),
        ],
        "net_profit": [
            re.compile(
                r"(?:net\s+)?profit\s*(?:after\s+tax)?\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"profit\s+for\s+the\s+year\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?)",
                re.IGNORECASE,
            ),
        ],
        "net_worth": [
            re.compile(
                r"net\s+worth\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?)",
                re.IGNORECASE,
            ),
        ],
        "total_assets": [
            re.compile(
                r"(?:total\s+)?assets?\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?)",
                re.IGNORECASE,
            ),
        ],
        "total_liabilities": [
            re.compile(
                r"(?:total\s+)?liabilit(?:y|ies)\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?)",
                re.IGNORECASE,
            ),
        ],
        "reserves_and_surplus": [
            re.compile(
                r"reserves?\s*(?:and|&)\s*surplus\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?)",
                re.IGNORECASE,
            ),
        ],
        "borrowings": [
            re.compile(
                r"(?:total\s+)?borrowings?\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,.\s]+(?:crore|lakh|lac|million)?)",
                re.IGNORECASE,
            ),
        ],
        "financial_year": [
            re.compile(
                r"(?:FY|financial\s+year)\s*[:\-]?\s*(\d{4}\s*[-–/]\s*\d{2,4})",
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

        # Try to identify the financial year on this page
        fy_match = self._FY_PATTERN.search(text)
        fy_value = None
        if fy_match:
            y1 = fy_match.group(1)
            y2 = fy_match.group(2)
            if len(y2) == 2:
                y2 = str(int(y1[:2]) + 1) + y2 if int(y1[-2:]) > int(y2) else y1[:2] + y2
            fy_value = f"{y1}-{y2[-2:]}"

        fields: List[ExtractedField] = []
        for fs in required_fields:
            if fs.name == "financial_year" and fy_value:
                field = self._build_field(
                    name="financial_year",
                    raw_value=fy_value,
                    datatype=fs.datatype,
                    page=page_num,
                    document_id=doc.document_id,
                    document_name=doc.document_name,
                    full_page_text=text,
                )
                if field:
                    fields.append(field)
                continue

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
                        break_field = False
                        for f in fields:
                            if f.name == fs.name:
                                break_field = True
                                break
                        if break_field:
                            break

        if not fields:
            return []

        entity_id = fy_value or f"fy_page_{page_num}"
        return [
            ExtractedRecord(
                entity_id=entity_id,
                group=group_by or "financial_year",
                group_value=fy_value,
                fields=fields,
            )
        ]

    def _page_is_relevant(
        self, page_text: str, page_summary: str, doc_summary: str, field_names: List[str]
    ) -> bool:
        combined = page_text + " " + page_summary
        keywords = [
            "turnover", "balance sheet", "profit", "loss", "revenue",
            "financial year", "net worth", "assets", "liabilities",
            "audited", "balance sheet", "income statement",
        ]
        return any(kw in combined.lower() for kw in keywords)