"""
Certificate Verifier - verifies certificate-related criteria.

v4.0 — Revised certificate verification:
  1. NEVER trusts the extracted `value` field blindly
  2. ALWAYS cross-checks with the `snippet` field
  3. When extraction looks suspicious, re-reads the full Company JSON page text
  4. Detects wrong cert types (ISA != CISA, Membership != Certification)
  5. Generates facts about WHAT WAS FOUND vs WHAT IS NEEDED
  6. ISA found where CISA required → NOT FAIL, it's REVIEW because:
     - We checked the certificate document, it's ISA not CISA
     - We can't conclusively prove no other partner has CISA
     - Summary explains this in plain language
  7. FAIL only when we can DIRECTLY prove: e.g., threshold for turnover
     is provably not met with all documents checked
"""

from __future__ import annotations

import re
from typing import Optional
from .base import BaseVerifier
from ..models import (
    Module2Output,
    CompanyJSON,
    DocumentRequirement,
    MatchedDocument,
    VerifiedFact,
    FactType,
    FactStatus,
    ExtractedField,
    Evidence,
)


# Certificate type aliases for fuzzy matching
_CISA_ALIASES = {"cisa", "disa", "dissa", "cisa/disa", "cisa/disa/dissa",
                  "certified information systems auditor", "disa certified"}


class CertificateVerifier(BaseVerifier):
    """Verifies certificate-related tender requirements — production-grade."""

    def __init__(self, module2: Module2Output, company: Optional[CompanyJSON] = None):
        super().__init__(module2, company)
        self._target_cert_types: list[str] = []
        self._min_count: int = 1
        self._related_evidence: list[Evidence] = []

    def set_target_certificates(self, cert_types: list[str], min_count: int = 1) -> "CertificateVerifier":
        """Set which certificate types to look for and minimum count."""
        self._target_cert_types = [ct.upper() for ct in cert_types]
        self._min_count = min_count
        return self

    def get_related_evidence(self) -> list[Evidence]:
        """Return all evidence items that were related to the certificate criterion."""
        return self._related_evidence

    def _is_cert_type_match(self, value: str) -> bool:
        """Check if a value matches any of the target certificate types."""
        value_upper = value.upper().strip()
        if value_upper in self._target_cert_types:
            return True
        for target in self._target_cert_types:
            if target in value_upper:
                return True
        for alias in _CISA_ALIASES:
            if alias.upper() in value_upper and any(t in alias.upper() for t in self._target_cert_types):
                return True
        return False

    def _resolve_cert_type(self, field: ExtractedField, doc_name: str) -> dict:
        """
        Determine the ACTUAL certificate type from a field.

        Resolution order:
          1. If `value` is NOT garbage and matches target → use it
          2. Extract from `snippet` text
          3. Re-verify with Company JSON full page text
          4. If nothing found → unknown

        Returns dict with:
          - detected_type: str or None
          - source: "value" | "snippet" | "company_json" | "none"
          - is_garbage: bool
          - is_wrong_type: bool
          - wrong_type_reason: str
          - company_json_page_text: str or None
          - snippet_text: str
        """
        result = {
            "detected_type": None,
            "source": "none",
            "is_garbage": False,
            "is_wrong_type": False,
            "wrong_type_reason": "",
            "company_json_page_text": None,
            "snippet_text": (field.snippet or ""),
        }

        value = (field.value or "").strip()
        snippet = (field.snippet or "").strip()

        # Step 1: Check if value is garbage
        is_garbage = self._is_garbage_extraction(value, field.name)
        result["is_garbage"] = is_garbage

        # Step 2: If value is good, try matching against target
        if not is_garbage and value:
            for target in self._target_cert_types:
                if target in value.upper():
                    result["detected_type"] = target
                    result["source"] = "value"
                    return result
            # Value is present but doesn't match target — note it
            result["detected_type"] = value.upper()

        # Step 3: Always check snippet (even if value matched, for verification)
        snippet_type = self._extract_cert_type_from_text(snippet)
        if snippet_type:
            # If we found something in snippet, it's more reliable
            result["detected_type"] = snippet_type
            result["source"] = "snippet"

        # Step 4: If still uncertain or wrong type, check Company JSON
        if self.company and field.page > 0:
            page_text = self._get_company_json_page_text(field.page)
            if page_text:
                result["company_json_page_text"] = page_text
                cj_type = self._extract_cert_type_from_text(page_text)
                if cj_type:
                    # Company JSON full text is most reliable
                    result["detected_type"] = cj_type
                    result["source"] = "company_json"

        # Step 5: Check if detected type is WRONG for the requirement
        if result["detected_type"]:
            is_wrong, reason = self._is_wrong_cert_type(
                result["detected_type"], self._target_cert_types
            )
            result["is_wrong_type"] = is_wrong
            result["wrong_type_reason"] = reason

        return result

    def verify(self, requirement: DocumentRequirement) -> list[VerifiedFact]:
        """
        Verify certificate-related requirements.

        Pipeline:
          1. Go through EACH matched document's records and fields one by one
          2. For each piece of evidence, check if it's related to certificate criterion
          3. If related, pick it and collect it
          4. For picked evidence, resolve the actual cert type using
             value → snippet → Company JSON page text
          5. Generate facts about WHAT WAS FOUND vs WHAT IS NEEDED
          6. NEVER generate FALSE facts for wrong cert type (that causes FAIL)
             — instead generate facts about what exists
        """
        facts: list[VerifiedFact] = []
        self._related_evidence = []

        target_types_str = ", ".join(self._target_cert_types)

        # ============================================================
        # Step 1: Go through each matched document, each record, each field
        # Check one by one if it's related to our criteria
        # ============================================================

        cert_related_docs = []  # docs that contain certificate-related evidence
        cert_related_fields = []  # (doc, record, field, resolution) for cert-related fields
        all_cert_doc_names = []  # names of all docs matched for cert requirement
        found_cert_types = []  # types actually found
        found_wrong_types = []  # (type, reason) for wrong types
        checked_doc_count = 0

        for doc in requirement.matched_documents:
            all_cert_doc_names.append(doc.document_name)

            for rec in doc.records:
                for field in rec.fields:
                    checked_doc_count += 1

                    # Check if this evidence is related to our certificate criterion
                    is_relevant = self._is_evidence_relevant_to_criterion(
                        field=field,
                        doc_name=doc.document_name,
                        criterion_keywords=set(
                            kw.lower() for kw in self._target_cert_types
                        ) | {"certificate", "certification", "certified", "cert"},
                        criterion_domain="certificate",
                    )

                    if not is_relevant:
                        continue

                    # This evidence IS related — pick it
                    self._related_evidence.append(Evidence(
                        document_name=doc.document_name,
                        page=field.page,
                        text=field.snippet or "",
                        llm_finding=f"{field.name} = {field.value} (confidence: {field.confidence})",
                    ))

                    if doc.document_name not in cert_related_docs:
                        cert_related_docs.append(doc.document_name)

                    # Resolve the actual certificate type
                    resolution = self._resolve_cert_type(field, doc.document_name)
                    cert_related_fields.append((doc, rec, field, resolution))

                    if resolution["detected_type"]:
                        if resolution["is_wrong_type"]:
                            found_wrong_types.append((
                                resolution["detected_type"],
                                resolution["wrong_type_reason"],
                            ))
                        else:
                            found_cert_types.append(resolution["detected_type"])

        # ============================================================
        # Step 2: Generate facts about what was found
        # ============================================================

        # FACT: How many documents were matched and how many were relevant
        facts.append(self._make_fact(
            statement=(
                f"Python checked {len(requirement.matched_documents)} document(s) "
                f"matched by Module 2 for the certificate requirement. "
                f"Out of these, {len(cert_related_docs)} document(s) contained "
                f"evidence related to the certificate criterion."
            ),
            fact_type=FactType.COUNT.value,
            status=FactStatus.VERIFIED.value,
            source_documents=all_cert_doc_names,
            details={
                "total_matched": len(requirement.matched_documents),
                "relevant_to_criterion": len(cert_related_docs),
                "relevant_docs": cert_related_docs,
            },
        ))

        # FACT: What certificate types were actually found
        required_set = set(t.upper() for t in self._target_cert_types)
        found_set = set(t.upper() for t in found_cert_types)
        matching_certs = found_set & required_set

        if matching_certs:
            facts.append(self._make_fact(
                statement=(
                    f"Python found {len(matching_certs)} matching certificate type(s): "
                    f"{', '.join(sorted(matching_certs))}. "
                    f"This matches the required certificate type(s): {target_types_str}."
                ),
                fact_type=FactType.ENTITY_MATCH.value,
                status=FactStatus.TRUE.value,
                source_fields=["certificate_type"],
                source_documents=cert_related_docs,
                details={
                    "found_types": sorted(matching_certs),
                    "required_types": self._target_cert_types,
                    "is_match": True,
                },
            ))
        else:
            # No matching cert found — but this is NOT FALSE, it's NOT_FOUND
            # because we can't prove no CISA exists (other documents might have it)
            if found_wrong_types:
                wrong_type_names = [wt[0] for wt in found_wrong_types]
                facts.append(self._make_fact(
                    statement=(
                        f"Python checked the certificate-related evidence and found "
                        f"{', '.join(wrong_type_names)} — which is NOT the required "
                        f"{target_types_str}. "
                        f"No CISA/DISA/DISSA certificate was found in the checked evidence."
                    ),
                    fact_type=FactType.ENTITY_MATCH.value,
                    status=FactStatus.NOT_FOUND.value,
                    source_fields=["certificate_type"],
                    source_documents=cert_related_docs,
                    details={
                        "found_types": wrong_type_names,
                        "required_types": self._target_cert_types,
                        "is_match": False,
                        "wrong_type_reasons": [wt[1] for wt in found_wrong_types],
                    },
                ))
            elif cert_related_docs:
                facts.append(self._make_fact(
                    statement=(
                        f"Python checked {len(cert_related_docs)} certificate-related "
                        f"document(s) but could not confirm the presence of "
                        f"{target_types_str}. "
                        f"The extracted data did not contain a clear match for the "
                        f"required certificate type."
                    ),
                    fact_type=FactType.EXISTENCE.value,
                    status=FactStatus.NOT_FOUND.value,
                    source_fields=["certificate_type"],
                    source_documents=cert_related_docs,
                    details={
                        "required_types": self._target_cert_types,
                        "docs_checked": cert_related_docs,
                    },
                ))
            else:
                facts.append(self._make_fact(
                    statement=(
                        f"Python checked {len(requirement.matched_documents)} matched "
                        f"document(s) for the certificate requirement but found NO "
                        f"evidence related to {target_types_str}. "
                        f"None of the matched documents contained certificate-type "
                        f"information."
                    ),
                    fact_type=FactType.EXISTENCE.value,
                    status=FactStatus.NOT_FOUND.value,
                    source_documents=all_cert_doc_names,
                    details={
                        "required_types": self._target_cert_types,
                        "docs_checked": all_cert_doc_names,
                    },
                ))

        # FACT: Threshold check (e.g., "at least one")
        if matching_certs:
            count = len(matching_certs)
            meets = count >= self._min_count
            facts.append(self._make_fact(
                statement=(
                    f"Python verified: {count} matching {target_types_str} "
                    f"certificate(s) found. Required: at least {self._min_count}. "
                    f"Requirement {'MET' if meets else 'NOT MET'}."
                ),
                fact_type=FactType.THRESHOLD.value,
                status=FactStatus.TRUE.value if meets else FactStatus.FALSE.value,
                source_documents=cert_related_docs,
                details={
                    "found_count": count,
                    "min_required": self._min_count,
                    "threshold_met": meets,
                },
            ))
        else:
            facts.append(self._make_fact(
                statement=(
                    f"Python verified: 0 matching {target_types_str} "
                    f"certificate(s) found. Required: at least {self._min_count}. "
                    f"Requirement NOT MET."
                ),
                fact_type=FactType.THRESHOLD.value,
                status=FactStatus.FALSE.value,
                source_documents=cert_related_docs or all_cert_doc_names,
                details={
                    "found_count": 0,
                    "min_required": self._min_count,
                    "threshold_met": False,
                },
            ))

        # ============================================================
        # Step 3: Generate per-document detail facts
        # (for evidence used in verdict)
        # ============================================================

        for doc, rec, field, resolution in cert_related_fields:
            source_desc = f"'{doc.document_name}' (page {field.page})"

            if resolution["is_garbage"]:
                facts.append(self._make_fact(
                    statement=(
                        f"Python detected garbage extraction in {source_desc}: "
                        f"field '{field.name}' had value '{field.value}' which is "
                        f"not a valid certificate type. "
                        f"Actual type determined from "
                        f"{resolution['source']}: {resolution['detected_type'] or 'UNKNOWN'}."
                    ),
                    fact_type=FactType.CONFLICT.value,
                    status=FactStatus.VERIFIED.value,
                    source_fields=[field.name],
                    source_documents=[doc.document_name],
                    details={
                        "original_value": field.value,
                        "resolved_type": resolution["detected_type"],
                        "resolution_source": resolution["source"],
                        "page": field.page,
                    },
                ))

            if resolution["is_wrong_type"]:
                facts.append(self._make_fact(
                    statement=(
                        f"Python verified in {source_desc}: the document contains "
                        f"a {resolution['detected_type']} certificate, which is NOT "
                        f"the required {target_types_str}. "
                        f"{resolution['wrong_type_reason']}"
                    ),
                    fact_type=FactType.ENTITY_MATCH.value,
                    status=FactStatus.NOT_FOUND.value,
                    source_fields=[field.name],
                    source_documents=[doc.document_name],
                    details={
                        "found_type": resolution["detected_type"],
                        "required_types": self._target_cert_types,
                        "reason": resolution["wrong_type_reason"],
                        "resolution_source": resolution["source"],
                        "page": field.page,
                    },
                ))

        # ============================================================
        # Step 4: Company JSON ground truth check
        # ============================================================

        if self.company:
            # Search for CISA/DISA/DISSA in ALL company pages
            all_page_results = []
            for pt in self.company.page_texts:
                m = re.search(r"page[_\s](\d+)", pt.page_key, re.IGNORECASE)
                if not m:
                    continue
                page_num = int(m.group(1))
                text = pt.text
                found_in_page = []
                for ct in self._target_cert_types:
                    if re.search(rf"\b{re.escape(ct)}\b", text, re.IGNORECASE):
                        found_in_page.append(ct)
                if found_in_page:
                    all_page_results.append((page_num, found_in_page))

            if all_page_results:
                page_desc = ", ".join(
                    f"page {p} ({', '.join(cts)})"
                    for p, cts in all_page_results
                )
                facts.append(self._make_fact(
                    statement=(
                        f"Python searched the entire Company JSON document text and found "
                        f"{len(all_page_results)} page(s) containing mentions of "
                        f"{target_types_str}: {page_desc}."
                    ),
                    fact_type=FactType.EXISTENCE.value,
                    status=FactStatus.VERIFIED.value,
                    source_documents=[f"Company JSON page {p}" for p, _ in all_page_results],
                    details={
                        "pages_with_matches": [
                            {"page": p, "types": cts}
                            for p, cts in all_page_results
                        ],
                    },
                ))
            else:
                facts.append(self._make_fact(
                    statement=(
                        f"Python searched the entire Company JSON document text "
                        f"and found NO mentions of {target_types_str} on any page."
                    ),
                    fact_type=FactType.EXISTENCE.value,
                    status=FactStatus.NOT_FOUND.value,
                    source_documents=["Company JSON (all pages)"],
                    details={"searched_for": self._target_cert_types},
                ))

        # ============================================================
        # Step 5: If no documents matched at all — negative fact
        # ============================================================

        if not requirement.matched_documents:
            facts.append(self._make_fact(
                statement=(
                    f"Python verified: NO documents were matched by Module 2 "
                    f"for the requirement '{requirement.requirement_document}'. "
                    f"Therefore, 0 {target_types_str} certificates were found."
                ),
                fact_type=FactType.EXISTENCE.value,
                status=FactStatus.NOT_FOUND.value,
                details={"requirement": requirement.requirement_document},
            ))

        return facts