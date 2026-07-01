"""
Base Verifier - abstract class all verifiers must extend.

v5.0 — Fully generic. No domain-specific logic.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional
from ..models import (
    Module2Output,
    CompanyJSON,
    DocumentRequirement,
    VerifiedFact,
    FactType,
    FactStatus,
    ExtractedField,
)


class BaseVerifier(ABC):
    """Abstract base class for all deterministic verifiers."""

    def __init__(self, module2: Module2Output, company: Optional[CompanyJSON] = None):
        self.module2 = module2
        self.company = company
        self._fact_counter = 0

    def _next_fact_id(self, prefix: str = "FACT") -> str:
        self._fact_counter += 1
        return f"{prefix}{self._fact_counter:03d}"

    def _make_fact(
        self,
        statement: str,
        fact_type: str,
        status: str,
        source_fields: Optional[list[str]] = None,
        source_documents: Optional[list[str]] = None,
        details: Optional[dict] = None,
        prefix: str = "FACT",
    ) -> VerifiedFact:
        return VerifiedFact(
            id=self._next_fact_id(prefix),
            statement=statement,
            type=fact_type,
            status=status,
            computed_by="python",
            source_fields=source_fields or [],
            source_documents=source_documents or [],
            details=details or {},
        )

    def _get_company_json_page_text(self, page_num: int) -> Optional[str]:
        """Get the full page text from Company JSON for a given page number."""
        if not self.company:
            return None
        for pt in self.company.page_texts:
            m = re.search(r"page[_\s](\d+)", pt.page_key, re.IGNORECASE)
            if m and int(m.group(1)) == page_num:
                return pt.text
        return None

    def _is_garbage_extraction(self, value: str, field_name: str = "") -> bool:
        """Detect if an extracted value is garbage/meaningless."""
        if not value or not value.strip():
            return True
        v = value.strip()
        if re.match(r"^\s*$", v):
            return True
        if v.lower() in ("yes", "no", "na", "n/a", "nil", "none"):
            if field_name and any(kw in field_name.lower() for kw in ["type", "name"]):
                return True
        return False

    @abstractmethod
    def verify(self, requirement: DocumentRequirement) -> list[VerifiedFact]:
        """Run deterministic verification. Returns list of VerifiedFact objects."""
        ...