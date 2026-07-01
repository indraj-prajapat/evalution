"""
Ground Truth Searcher - searches Company JSON for additional/contradicting evidence.

v3.0 additions:
  - Document-level re-verification: for each matched doc, re-read the full
    Company JSON page text to validate extraction quality
  - Certificate type re-detection from full page text
  - Contradiction detection between extraction and ground truth
  - Returns re-verification results that verifiers can use
"""

from __future__ import annotations

import re
from typing import Optional
from .models import (
    CompanyJSON,
    CompanyDocument,
    CompanyPageText,
    GroundTruthVerification,
    Evidence,
)


class GroundTruthSearcher:
    """
    Searches the Company JSON for supporting/contradicting evidence
    related to a criterion.
    """

    def __init__(self, company_json: CompanyJSON):
        self.company = company_json
        self._page_text_map: dict[int, str] = {}
        self._doc_map: dict[str, CompanyDocument] = {}
        self._index_company()

    def _index_company(self) -> None:
        """Build lookup indices for fast searching."""
        # Index page texts
        for pt in self.company.page_texts:
            m = re.search(r"page[_\s](\d+)", pt.page_key, re.IGNORECASE)
            if m:
                page_num = int(m.group(1))
                self._page_text_map[page_num] = pt.text

        # Index documents by doc_id and document_name
        for doc in self.company.documents:
            self._doc_map[doc.doc_id.lower()] = doc
            self._doc_map[doc.document_name.lower()] = doc

    def search_pages(
        self,
        keywords: list[str],
        require_all: bool = False,
        case_sensitive: bool = False,
    ) -> list[tuple[int, str, list[str]]]:
        """
        Search all page texts for keywords.

        Returns:
            List of (page_num, page_text, matched_keywords) tuples.
        """
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE

        for page_num, text in sorted(self._page_text_map.items()):
            matched = []
            for kw in keywords:
                pattern = re.escape(kw)
                if re.search(pattern, text, flags):
                    matched.append(kw)

            if require_all and len(matched) < len(keywords):
                continue
            if not require_all and not matched:
                continue

            results.append((page_num, text, matched))

        return results

    def search_documents(
        self,
        keywords: list[str],
        search_fields: Optional[list[str]] = None,
        doc_type_filter: Optional[str] = None,
    ) -> list[tuple[CompanyDocument, dict[str, list[str]]]]:
        """
        Search structured document entities for keywords.

        Returns:
            List of (CompanyDocument, {field_name: [matched_values]}).
        """
        results = []
        flags = re.IGNORECASE

        for doc in self.company.documents:
            if doc_type_filter and doc.doc_type.lower() != doc_type_filter.lower():
                continue

            field_matches: dict[str, list[str]] = {}
            for field_name, field_value in doc.entities.items():
                if search_fields and field_name not in search_fields:
                    continue
                if not isinstance(field_value, str):
                    field_value = str(field_value)
                for kw in keywords:
                    if re.search(re.escape(kw), field_value, flags):
                        field_matches.setdefault(field_name, []).append(field_value)
                        break

            if field_matches:
                results.append((doc, field_matches))

        return results

    def search_summaries(self, keywords: list[str]) -> list[tuple[CompanyDocument, list[str]]]:
        """Search document summaries for keywords."""
        results = []
        for doc in self.company.documents:
            matched = []
            for kw in keywords:
                if re.search(re.escape(kw), doc.summary, re.IGNORECASE):
                    matched.append(kw)
            if matched:
                results.append((doc, matched))
        return results

    def find_document_by_id(self, doc_id: str) -> Optional[CompanyDocument]:
        """Find a document by its doc_id."""
        return self._doc_map.get(doc_id.lower())

    def find_documents_by_type(self, doc_type: str) -> list[CompanyDocument]:
        """Find all documents of a given type."""
        return [
            doc for doc in self.company.documents
            if doc.doc_type.lower() == doc_type.lower()
        ]

    def find_documents_by_name_pattern(self, pattern: str) -> list[CompanyDocument]:
        """Find documents whose name matches a regex pattern."""
        results = []
        try:
            pat = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return results
        for doc in self.company.documents:
            if pat.search(doc.document_name):
                results.append(doc)
        return results

    def get_page_text(self, page_num: int) -> Optional[str]:
        """Get the raw text of a specific page."""
        return self._page_text_map.get(page_num)

    def get_document_pages(self, doc: CompanyDocument) -> list[str]:
        """Get all page texts for a document."""
        texts = []
        for p in doc.pages:
            t = self._page_text_map.get(p)
            if t:
                texts.append(t)
        return texts

    def find_linked_entities(
        self,
        entity_names: list[str],
        context_keywords: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Find entities linked to the given entity names across the Company JSON.
        Returns list of {entity_name, page, doc_name, context_snippet}.
        """
        linked = []
        seen = set()

        for page_num, text in sorted(self._page_text_map.items()):
            for entity in entity_names:
                pattern = re.escape(entity)
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for m in matches:
                    start = max(0, m.start() - 100)
                    end = min(len(text), m.end() + 100)
                    snippet = text[start:end].strip()

                    if context_keywords:
                        has_context = any(
                            kw.lower() in snippet.lower() for kw in context_keywords
                        )
                        if not has_context:
                            continue

                    key = (entity.lower(), page_num)
                    if key not in seen:
                        seen.add(key)
                        doc_name = ""
                        for doc in self.company.documents:
                            if page_num in doc.pages:
                                doc_name = doc.document_name
                                break

                        linked.append({
                            "entity_name": entity,
                            "page": page_num,
                            "doc_name": doc_name,
                            "context_snippet": snippet,
                        })

        return linked

    def re_verify_document(
        self,
        document_name: str,
        page_numbers: list[int],
        search_terms: list[str],
    ) -> dict:
        """
        Re-verify a specific document by re-reading its full Company JSON page text.

        This is the KEY v3.0 addition: when Module 2 extraction is suspicious,
        we go BACK to the source (Company JSON) and read the actual page content.

        Returns:
            {
                "document_name": str,
                "pages_checked": int,
                "pages_found": int,
                "page_results": [
                    {
                        "page": int,
                        "found": bool,
                        "term_matches": {term: [context_snippets]},
                        "full_text_length": int,
                        "relevant_snippet": str or None,
                    }
                ],
                "any_match": bool,
                "all_page_snippets": [str],  # Concatenated relevant snippets
            }
        """
        result = {
            "document_name": document_name,
            "pages_checked": len(page_numbers),
            "pages_found": 0,
            "page_results": [],
            "any_match": False,
            "all_page_snippets": [],
        }

        for page_num in page_numbers:
            page_text = self._page_text_map.get(page_num)
            page_result = {
                "page": page_num,
                "found": page_text is not None,
                "term_matches": {},
                "full_text_length": len(page_text) if page_text else 0,
                "relevant_snippet": None,
            }

            if page_text:
                result["pages_found"] += 1
                for term in search_terms:
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    matches = []
                    for m in pattern.finditer(page_text):
                        start = max(0, m.start() - 150)
                        end = min(len(page_text), m.end() + 150)
                        snippet = page_text[start:end].strip()
                        matches.append(snippet)
                    if matches:
                        page_result["term_matches"][term] = matches

                # Extract a relevant 300-char window
                for term in search_terms:
                    idx = page_text.upper().find(term.upper())
                    if idx >= 0:
                        start = max(0, idx - 100)
                        end = min(len(page_text), idx + 200)
                        page_result["relevant_snippet"] = page_text[start:end].strip()
                        result["all_page_snippets"].append(page_result["relevant_snippet"])
                        break

                if page_result["term_matches"]:
                    result["any_match"] = True

            result["page_results"].append(page_result)

        return result

    def detect_contradictions(
        self,
        module2_doc_ids: list[str],
        keywords: list[str],
    ) -> list[dict]:
        """
        Detect contradictions between Module 2 extractions and Company JSON.

        Looks for cases where the same entity has different values
        in Module 2 vs Company JSON.
        """
        contradictions = []

        # Search for keywords in pages NOT referenced by Module 2
        m2_pages = set()
        for doc_id in module2_doc_ids:
            doc = self.find_document_by_id(doc_id)
            if doc:
                m2_pages.update(doc.pages)

        for page_num, text in sorted(self._page_text_map.items()):
            if page_num in m2_pages:
                continue

            for kw in keywords:
                if re.search(re.escape(kw), text, re.IGNORECASE):
                    contradictions.append({
                        "type": "unreferenced_evidence",
                        "keyword": kw,
                        "page": page_num,
                        "snippet": text[:500],
                        "detail": (
                            f"Keyword '{kw}' found on page {page_num} which was not "
                            f"referenced by Module 2 matched documents"
                        ),
                    })

        return contradictions

    def check_evidence_completeness(
        self,
        required_document_types: list[str],
        module2_docs: list[str],
    ) -> dict:
        """Check if all required document types exist in the Company JSON."""
        company_doc_types = set()
        company_doc_names_lower = set()

        for doc in self.company.documents:
            company_doc_types.add(doc.doc_type.lower())
            company_doc_names_lower.add(doc.document_name.lower())

        found_in_company = []
        missing_from_company = []

        for req_type in required_document_types:
            req_lower = req_type.lower()
            if req_lower in company_doc_types:
                found_in_company.append(req_type)
            elif any(req_lower in name for name in company_doc_names_lower):
                found_in_company.append(req_type)
            else:
                missing_from_company.append(req_type)

        return {
            "found_in_company": found_in_company,
            "missing_from_company": missing_from_company,
            "company_doc_types": sorted(company_doc_types),
        }

    def perform_full_search(
        self,
        criterion: str,
        module2_doc_ids: list[str],
        module2_doc_names: list[str],
        search_keywords: list[str],
        required_doc_types: Optional[list[str]] = None,
        context_keywords: Optional[list[str]] = None,
    ) -> GroundTruthVerification:
        """
        Perform the full ground truth verification search.
        """
        result = GroundTruthVerification()
        result.completed = True

        # 1. Search pages for keywords
        page_results = self.search_pages(search_keywords)
        result.additional_documents_checked = [
            f"Page {p} ({len(matched)} keyword matches)"
            for p, _, matched in page_results
        ]

        # 2. Search structured documents
        doc_results = self.search_documents(search_keywords)
        for doc, field_matches in doc_results:
            if doc.document_name not in result.additional_documents_checked:
                result.additional_documents_checked.append(doc.document_name)

        # 3. Search summaries
        summary_results = self.search_summaries(search_keywords)
        for doc, matched in summary_results:
            if doc.document_name not in result.additional_documents_checked:
                result.additional_documents_checked.append(doc.document_name)

        # 4. Detect contradictions
        contradictions = self.detect_contradictions(module2_doc_ids, search_keywords)
        result.contradictions_found = len(contradictions) > 0
        result.contradiction_details = [
            c["detail"] for c in contradictions
        ]

        # 5. Check supporting evidence
        result.supporting_evidence_found = len(page_results) > 0 or len(doc_results) > 0

        # 6. Check missing evidence
        if required_doc_types:
            completeness = self.check_evidence_completeness(
                required_doc_types, module2_doc_names
            )
            result.missing_evidence = completeness["missing_from_company"]

        # 7. Find linked entities
        if context_keywords:
            linked = self.find_linked_entities(search_keywords, context_keywords)
            result.linked_entities = [
                f"{l['entity_name']} on page {l['page']} ({l['doc_name']})"
                for l in linked
            ]

        return result