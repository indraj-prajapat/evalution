"""
Entity Checker - Verifies company name and entity names across documents.

Checks if the company/entity name in documents matches the expected company.
Flags mismatches that should trigger REVIEW.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import CompanyJSON, CompanyDocument, Evidence


@dataclass
class EntityCheckResult:
    """Result of company/entity name verification."""
    company_name_from_input: str = ""
    company_names_found_in_docs: list[str] = field(default_factory=list)
    entity_names_in_company_json: list[str] = field(default_factory=list)
    mismatches: list[dict] = field(default_factory=list)
    has_mismatch: bool = False
    verification_summary: str = ""


class EntityChecker:
    """
    Verifies that company/entity names are consistent across documents.

    Production-grade checks:
    - Extract company name from Company JSON (file_name, entities, document names)
    - Compare against the company_name provided by the user
    - Check each evidence document's page text for entity names
    - Flag if different company names appear in documents
    """

    def __init__(self, company: Optional[CompanyJSON] = None):
        self.company = company

    def _normalize_name(self, name: str) -> str:
        """Normalize a company name for comparison."""
        if not name:
            return ""
        # Remove common prefixes/suffixes
        name = re.sub(r"^(M/s\.?|M/s)\s*", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^(The\s+)", "", name, flags=re.IGNORECASE)
        # Remove punctuation except &, ., -
        name = re.sub(r"[,;:()'\"\/]", " ", name)
        # Collapse whitespace
        name = re.sub(r"\s+", " ", name).strip()
        return name.lower()

    def _names_match(self, name1: str, name2: str, threshold: float = 0.7) -> bool:
        """
        Check if two company names refer to the same entity.
        Uses normalized comparison with Jaccard-like word overlap.
        """
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)

        if not n1 or not n2:
            return False

        if n1 == n2:
            return True

        # Check if one contains the other
        if n1 in n2 or n2 in n1:
            return True

        # Word-level Jaccard similarity
        words1 = set(n1.split())
        words2 = set(n2.split())

        if not words1 or not words2:
            return False

        # Check for significant word overlap
        common = words1 & words2
        # Remove common generic words
        generic = {"company", "co", "firm", "llp", "pvt", "ltd", "limited",
                    "private", "and", "&", "the", "chartered", "accountants",
                    "ca", "associates"}
        common -= generic
        words1 -= generic
        words2 -= generic

        if not words1 or not words2:
            # After removing generic words, if nothing left, use original
            words1 = set(n1.split()) - {"and", "&"}
            words2 = set(n2.split()) - {"and", "&"}
            common = words1 & words2

        if not words1 or not words2:
            return False

        jaccard = len(common) / len(words1 | words2)
        return jaccard >= threshold

    def _extract_company_names_from_json(self) -> list[str]:
        """Extract all company name variants from Company JSON."""
        names = []
        if not self.company:
            return names

        # From file_name
        if self.company.file_name:
            # Remove extension
            name = re.sub(r"\.(pdf|PDF|json|JSON)$", "", self.company.file_name)
            # Remove common suffixes like "_merged_with_Page_Numbers"
            name = re.sub(r"[_\s]+(merged|with|page|numbers).*", "", name, flags=re.IGNORECASE)
            name = name.replace("_", " ").replace("-", " ")
            names.append(name)

        # From document entities
        for doc in self.company.documents:
            if doc.entities:
                company_name = doc.entities.get("company_name", "")
                if company_name and company_name not in names:
                    names.append(company_name)

            # From document names (first part before " – " or " - ")
            if doc.document_name:
                parts = re.split(r"\s*[–\-]\s*", doc.document_name, maxsplit=1)
                if parts:
                    doc_name_part = parts[0].strip()
                    if doc_name_part and doc_name_part not in names:
                        names.append(doc_name_part)

        return names

    def check(
        self,
        company_name: str,
        evidence_items: Optional[list[Evidence]] = None,
        page_texts: Optional[dict[int, str]] = None,
    ) -> EntityCheckResult:
        """
        Check company/entity name consistency.

        Args:
            company_name: The expected company name (from user input).
            evidence_items: Related evidence items to check.
            page_texts: Optional page_num -> text mapping for verification.

        Returns:
            EntityCheckResult with all findings.
        """
        result = EntityCheckResult(company_name_from_input=company_name)

        # Extract names from Company JSON
        json_names = self._extract_company_names_from_json()
        result.entity_names_in_company_json = json_names

        # Find the primary company name from Company JSON
        primary_json_name = ""
        for name in json_names:
            if company_name and self._names_match(name, company_name):
                primary_json_name = name
                break
            if not primary_json_name and len(name) > 3:
                primary_json_name = name

        if primary_json_name:
            result.company_names_found_in_docs.append(primary_json_name)

        # Check each evidence document's page text for entity names
        if evidence_items and page_texts:
            for ev in evidence_items:
                page_text = page_texts.get(ev.page)
                if not page_text:
                    continue

                # Look for company name patterns in the page text.
                # Fully generic — no hardcoded company names.
                # Common patterns: "M/s. COMPANY NAME", "Name of the firm : COMPANY"
                name_patterns = [
                    r"(?:M/s\.?|Name of the (?:firm|company)\s*:\s*|"
                    r"Firm\s*:\s*|Name\s+of\s+(?:the\s+)?(?:bidder|applicant)\s*:\s*)"
                    r"\s*(.+?)(?:\n|$)",
                ]

                for pat in name_patterns:
                    m = re.search(pat, page_text, re.IGNORECASE)
                    if m:
                        found_name = m.group(1).strip()
                        if found_name and len(found_name) > 3:
                            if found_name not in result.company_names_found_in_docs:
                                result.company_names_found_in_docs.append(found_name)

        # Check for mismatches
        if company_name and result.company_names_found_in_docs:
            for found_name in result.company_names_found_in_docs:
                if not self._names_match(found_name, company_name):
                    result.mismatches.append({
                        "expected": company_name,
                        "found": found_name,
                        "document": "Company JSON",
                    })
                    result.has_mismatch = True

        # Also check within found names for inconsistencies
        unique_normalized = {}
        for name in result.company_names_found_in_docs:
            norm = self._normalize_name(name)
            if norm and norm not in unique_normalized:
                unique_normalized[norm] = name

        # If there are multiple distinct normalized names, check if they all match
        norm_names = list(unique_normalized.keys())
        for i in range(len(norm_names)):
            for j in range(i + 1, len(norm_names)):
                if not self._names_match(norm_names[i], norm_names[j]):
                    result.mismatches.append({
                        "expected": unique_normalized[norm_names[i]],
                        "found": unique_normalized[norm_names[j]],
                        "document": "Cross-document inconsistency",
                    })
                    result.has_mismatch = True

        # Build summary
        if result.has_mismatch:
            mismatch_desc = "; ".join(
                f"found '{m['found']}' (expected '{m['expected']}')" 
                for m in result.mismatches[:3]
            )
            result.verification_summary = (
                f"Company name verification found inconsistencies: {mismatch_desc}. "
                f"This requires manual review."
            )
        elif result.company_names_found_in_docs:
            result.verification_summary = (
                f"Company name '{company_name or primary_json_name}' "
                f"is consistently referenced across the checked documents."
            )
        else:
            result.verification_summary = (
                "No company name could be extracted from the documents for verification."
            )

        return result