"""
Company matching module.

Verifies whether a detected document belongs to the bidder company
using PAN, GSTIN, CIN, name similarity, and address overlap.

Exception: experience/work-order documents are accepted even when the
client name differs (the bidder is the service provider, not the client).
"""

from __future__ import annotations

import re
from typing import Optional

from .utils import token_overlap, extract_pan, extract_gstin, extract_cin


class CompanyMatcher:
    """Determines whether a document belongs to the bidder."""

    def __init__(
        self,
        bidder_name: str,
        bidder_pan: Optional[str] = None,
        bidder_gstin: Optional[str] = None,
        bidder_cin: Optional[str] = None,
        bidder_address: Optional[str] = None,
    ):
        self.bidder_name = bidder_name.strip()
        self.bidder_pan = (bidder_pan or "").strip().upper()
        self.bidder_gstin = (bidder_gstin or "").strip().upper()
        self.bidder_cin = (bidder_cin or "").strip().upper()
        self.bidder_address = (bidder_address or "").strip()
        self._bidder_tokens = self._name_tokens(self.bidder_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_owned_by_bidder(
        self,
        doc_text: str,
        doc_name: str = "",
        doc_category: str = "",
        doc_metadata: Optional[dict] = None,
    ) -> tuple[bool, str]:
        """Return (is_match, reason).

        Parameters
        ----------
        doc_text : full page or document OCR text
        doc_name : detected document name
        doc_category : detected document category
        doc_metadata : any structured metadata (e.g. extracted PAN/GSTIN)
        """
        if not doc_metadata:
            doc_metadata = {}

        # 1. Hard identifier match (highest confidence)
        if self.bidder_pan:
            doc_pan = (doc_metadata.get("pan") or "").upper() or extract_pan(doc_text)
            if doc_pan and doc_pan == self.bidder_pan:
                return True, "PAN match"

        if self.bidder_gstin:
            doc_gstin = (doc_metadata.get("gstin") or "").upper() or extract_gstin(doc_text)
            if doc_gstin and doc_gstin == self.bidder_gstin:
                return True, "GSTIN match"

        if self.bidder_cin:
            doc_cin = (doc_metadata.get("cin") or "").upper() or extract_cin(doc_text)
            if doc_cin and doc_cin == self.bidder_cin:
                return True, "CIN match"

        # 2. Name similarity
        doc_company = (doc_metadata.get("company_name") or "").strip()
        if not doc_company:
            doc_company = self._guess_company_name(doc_text)

        if doc_company:
            score = token_overlap(self.bidder_name, doc_company)
            if score >= 0.7:
                return True, f"Company name match ({score:.2f})"

            # Also check bidirectional substring
            b_lower = self.bidder_name.lower()
            d_lower = doc_company.lower()
            if b_lower in d_lower or d_lower in b_lower:
                return True, "Company name substring match"

        # 3. Check if bidder name appears anywhere in the document text
        bidder_core = self.bidder_name.lower()
        # Strip common prefixes like "M/s ", "Ms ", "M/s. "
        for prefix in ["m/s ", "ms ", "m/s. ", "m/s", "ms"]:
            if bidder_core.startswith(prefix):
                bidder_core = bidder_core[len(prefix):].strip()
                break
        if bidder_core and bidder_core in doc_text[:3000].lower():
            return True, "Bidder name found in document text"

        # 4. Address overlap (weak signal, only if name also partially matches)
        if self.bidder_address:
            score = token_overlap(self.bidder_address, doc_text[:2000])
            if score >= 0.6:
                doc_name_score = token_overlap(self.bidder_name, doc_text[:2000])
                if doc_name_score >= 0.3:
                    return True, f"Address + name match ({score:.2f})"

        return False, "No matching identifier found"

    def is_experience_document(self, doc_category: str, doc_name: str) -> bool:
        """Heuristic: is this an experience / work-order type document?

        These documents are allowed to have a *different* company (the client).
        """
        text = (doc_category + " " + doc_name).lower()
        keywords = [
            "work order", "work-order", "completion certificate",
            "experience certificate", "performance certificate",
            "project completion", "contract agreement",
            "experience", "work order",
        ]
        return any(kw in text for kw in keywords)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _name_tokens(name: str) -> set[str]:
        import re as _re
        return set(_re.findall(r"[a-z0-9]+", name.lower()))

    @staticmethod
    def _guess_company_name(text: str) -> str:
        """Naive heuristic: find 'M/s ...' or 'Name:' patterns."""
        # M/s ABC Pvt. Ltd.
        m = re.search(r"M/[sS]\.?\s+(.+?)(?:\n|,|\.)", text[:2000])
        if m:
            return m.group(1).strip()
        # "Name of the Firm / Company : XYZ"
        m = re.search(
            r"(?:name\s+of\s+(?:the\s+)?(?:firm|company|contractor|bidder|proprietor))\s*[:\-]?\s*(.+?)(?:\n|,|\.)",
            text[:2000],
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return ""