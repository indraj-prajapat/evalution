"""
Fact Generator - orchestrates verification and assembles Verified Facts.

v5.1 — Production-grade fixes:
  - Safe type coercion for all field values (bool/None/int/float/str)
  - Corrected is_numeric logic (removed contradictory alpha check)
  - Robust entity-check page_texts dict construction (no walrus side-effects)
  - Consistent None-guards throughout the pipeline
  - Helper methods _safe_str() and _is_numeric_value() used everywhere
"""

from __future__ import annotations

import re
from typing import Optional
from .models import (
    Module2Output,
    CompanyJSON,
    DocumentRequirement,
    VerifiedFact,
    FactType,
    FactStatus,
    Evidence,
    VerificationReport,
    parse_currency,
    parse_number,
)
from .evidence_filter import EvidenceFilter, FilteredEvidence
from .criterion_parser import parse_criterion, RequirementCheck
from .numeric_verifier import NumericVerifier
from .entity_checker import EntityChecker
from .ground_truth import GroundTruthSearcher
from .data_point_validator import DataPointValidator


class FactGenerator:
    """
    Analyzes criterion + Module 2 output and generates all Verified Facts.

    v5.1 Pipeline (fully generic, no hardcoded domains):
      1. Parse criterion into checkable requirements
      2. Collect ALL evidence from Module 2 matched documents
      3. Use LLM to filter evidence by relevance to criterion
      4. Verify related evidence against Company JSON page text
      5. Check company/entity name consistency
      6. Python handles all numeric checks (threshold, comparison, count)
      7. Generate facts about what was found vs what was needed
      8. Build evidence lists with page-level verification
    """

    def __init__(
        self,
        module2: Module2Output,
        company: Optional[CompanyJSON] = None,
        ground_truth: Optional[GroundTruthSearcher] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        company_name: str = "",
    ):
        self.module2 = module2
        self.company = company
        self.ground_truth = ground_truth
        self.company_name = company_name
        self._fact_counter = 0
        self._all_evidence: list[Evidence] = []
        self._related_evidence: list[Evidence] = []
        self._evidence_filter = EvidenceFilter(api_key=api_key, model=model)
        self._entity_checker = EntityChecker(company)
        self._numeric_verifier = NumericVerifier()
        self._data_point_validator = DataPointValidator(api_key=api_key, model=model)
        self._generated_facts: list[VerifiedFact] = []
        self._criterion_requirements: list[dict] = []
        self._entity_check = None
        self._filter_result = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_str(value: object) -> str:
        """
        Safely coerce any field value to a plain string.

        Handles: None → "", bool → "true"/"false", numeric types → their
        string representation, everything else via str().
        """
        if value is None:
            return ""
        if isinstance(value, bool):
            # Must come before int check — bool is a subclass of int.
            return str(value).lower()
        if isinstance(value, (int, float)):
            return str(value)
        return str(value)

    @staticmethod
    def _is_numeric_value(value: object) -> bool:
        """
        Return True if *value* looks like a pure number (possibly with
        commas/decimal point/leading sign).

        Examples that return True : "1,23,456", "3.14", "-42", "0"
        Examples that return False: True, None, "", "INR 50 lakhs", "abc"
        """
        if value is None or isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        s = str(value).strip()
        if not s:
            return False
        # Strip optional leading sign then formatting characters
        cleaned = s.lstrip("+-").replace(",", "").replace(".", "", 1)
        return cleaned.isdigit()

    @staticmethod
    def _is_parseable_amount(value: object) -> bool:
        """
        Return True if *value* can be parsed into a real number by the same
        numeric engine the verifier uses — including currency/units like
        "25 Lakhs", "1.5 Crore", "₹1,23,456".

        This is the correct test for ROUTING values to numeric (threshold/
        comparison) checks: anything Python can turn into a number is fair game
        for Python's authoritative numeric verification.
        """
        if value is None or isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        s = str(value).strip()
        if not s:
            return False
        return (parse_currency(s) is not None) or (parse_number(s) is not None)

    def _select_requirement_data_points(
        self,
        req: RequirementCheck,
        all_data_points: list[dict],
    ) -> list[dict]:
        """
        Select the evidence data points relevant to a single requirement.

        Extracted from the old inline body of the Step 8c loop so it can be
        reused both for standalone requirements and for each member of an
        OR-alternative group (see generate_all_facts). Behavior is
        unchanged from before the extraction.

        Routing rule (production):
          - NUMERIC requirements (threshold/comparison): Python is
            authoritative. Feed it the numeric data points. We do NOT require
            a keyword match on field names, because financial figures are
            often stored under generic field names (e.g. "value", "amount").
            If keyword-matched numeric points exist we prefer those; otherwise
            we fall back to ALL numeric points so a real number is never missed.
          - INFORMATIONAL requirements (existence/count): scoped by keyword so
            the observation is relevant, but the result never blocks a verdict.
        """
        req_keywords = {
            kw for kw in req.target.lower().split() if len(kw) >= 3
        }
        keyword_points = [
            dp for dp in all_data_points
            if any(
                kw in f"{dp.get('field', '')} {dp.get('value', '')}".lower()
                for kw in req_keywords
            )
        ]

        if req.is_numeric_check:
            # Route any value Python can parse into a number (incl. units
            # like "25 Lakhs", "1.5 Crore", "₹1,23,456"). Prefer keyword-
            # matched amounts; only fall back to ALL parseable amounts when
            # NO keyword match exists, so a real number is never missed
            # because of a generic field name.
            #
            # BUGFIX: the previous version treated "prefer" as "prefer, but
            # also union in everything else" - whenever ANY keyword-matched
            # numeric point existed, it still appended every OTHER numeric
            # value found ANYWHERE in the entire evidence set (e.g. staff
            # counts, EMD amounts, project durations belonging to an
            # unrelated criterion) into this requirement's data points.
            # NumericVerifier then summed/averaged/thresholded across all of
            # those unrelated numbers together with the real value,
            # producing wrong sums, wrong averages, and wrong PASS/FAIL
            # verdicts. Once a keyword match exists, ONLY those matched
            # points are used - the "ALL parseable amounts" fallback is
            # reserved strictly for when keyword matching found nothing.
            numeric_keyword_points = [
                dp for dp in keyword_points
                if self._is_parseable_amount(dp.get("value"))
            ]
            if numeric_keyword_points:
                req_data_points = list(numeric_keyword_points)
            else:
                req_data_points = [
                    dp for dp in all_data_points
                    if self._is_parseable_amount(dp.get("value"))
                ]
            # Deduplicate identical (field, value, source) points so a value
            # that appears both as a structured data point AND a raw field
            # is not double-counted in sums/averages.
            seen_dp: set = set()
            deduped: list[dict] = []
            for dp in req_data_points:
                key = (
                    str(dp.get("field", "")).strip().lower(),
                    str(dp.get("value", "")).strip().lower(),
                    str(dp.get("source_doc", "")).strip().lower(),
                )
                if key not in seen_dp:
                    seen_dp.add(key)
                    deduped.append(dp)
            req_data_points = deduped
        else:
            req_data_points = keyword_points
            # For existence checks with no keyword hits, observe all points.
            if not req_data_points and req.check_type == "existence":
                req_data_points = all_data_points

        return req_data_points

    # ------------------------------------------------------------------
    # Fact construction helpers
    # ------------------------------------------------------------------

    def _next_fact_id(self) -> str:
        self._fact_counter += 1
        return f"FACT{self._fact_counter:03d}"

    def _make_fact(
        self,
        statement: str,
        fact_type: str,
        status: str,
        source_fields: Optional[list[str]] = None,
        source_documents: Optional[list[str]] = None,
        details: Optional[dict] = None,
    ) -> VerifiedFact:
        return VerifiedFact(
            id=self._next_fact_id(),
            statement=statement,
            type=fact_type,
            status=status,
            computed_by="python",
            source_fields=source_fields or [],
            source_documents=source_documents or [],
            details=details or {},
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def generate_all_facts(self) -> list[VerifiedFact]:
        """
        Main entry point: generate ALL verified facts using the generic pipeline.
        """
        all_facts: list[VerifiedFact] = []
        criterion = self.module2.criterion

        # ============================================================
        # Step 1: Parse criterion into checkable requirements
        # ============================================================
        parsed = parse_criterion(criterion)

        self._criterion_requirements = [
            {
                "requirement": req.description,
                "check_type": req.check_type,
                "target": req.target,
                "operator": req.operator,
                "threshold_value": req.threshold_value,
                "min_count": req.min_count,
                "unit": req.unit,
            }
            for req in parsed.requirements
        ]

        # ============================================================
        # Step 2: Document status overview
        # ============================================================
        total_matched = 0
        found_count = 0
        partial_count = 0
        not_found_count = 0
        all_doc_names: list[str] = []

        for req in self.module2.documents:
            for doc in req.matched_documents:
                total_matched += 1
                all_doc_names.append(doc.document_name)
                status = (doc.status or "").upper()
                if status == "FOUND":
                    found_count += 1
                elif status == "PARTIAL":
                    partial_count += 1
                else:
                    not_found_count += 1

        all_facts.append(self._make_fact(
            statement=(
                f"Python found {total_matched} document(s) matched by Module 2: "
                f"{found_count} FOUND, {partial_count} PARTIAL, "
                f"{not_found_count} NOT FOUND."
            ),
            fact_type=FactType.COUNT.value,
            status=FactStatus.VERIFIED.value,
            source_documents=all_doc_names,
            details={
                "total": total_matched,
                "found": found_count,
                "partial": partial_count,
                "not_found": not_found_count,
            },
        ))

        # ============================================================
        # Step 3: LLM-based evidence filtering
        # ============================================================
        filter_result = self._evidence_filter.filter(
            self.module2, criterion, self.company
        )
        self._filter_result = filter_result

        all_facts.append(self._make_fact(
            statement=(
                f"Python filtered {filter_result.total_evidence} evidence item(s) "
                f"using {filter_result.filter_method}: "
                f"{filter_result.related_count} related, "
                f"{filter_result.unrelated_count} unrelated to the criterion."
            ),
            fact_type=FactType.COUNT.value,
            status=FactStatus.VERIFIED.value,
            details={
                "total": filter_result.total_evidence,
                "related": filter_result.related_count,
                "unrelated": filter_result.unrelated_count,
                "filter_method": filter_result.filter_method,
            },
        ))

        # ============================================================
        # Step 4: Build ALL evidence list (for output)
        # ============================================================
        self._all_evidence = self._build_all_evidence()

        # ============================================================
        # Step 5: Build RELATED evidence list with page verification
        # ============================================================
        self._related_evidence = self._build_related_evidence(
            filter_result.related_items
        )

        # ============================================================
        # Step 6: Ground truth search
        # ============================================================
        if self.ground_truth:
            search_keywords = parsed.key_terms
            if not search_keywords:
                words = re.findall(r"\b[A-Za-z]{3,}\b", criterion)
                search_keywords = [w for w in words if len(w) >= 3][:10]

            gt_pages = self.ground_truth.search_pages(search_keywords)
            if gt_pages:
                preview = ", ".join(
                    f"page {p} ({', '.join(kw[:3])})" for p, _, kw in gt_pages[:5]
                )
                overflow = (
                    f" and {len(gt_pages) - 5} more" if len(gt_pages) > 5 else ""
                )
                all_facts.append(self._make_fact(
                    statement=(
                        f"Python searched Company JSON and found "
                        f"{len(gt_pages)} page(s) with criterion-related keywords: "
                        f"{preview}{overflow}."
                    ),
                    fact_type=FactType.EXISTENCE.value,
                    status=FactStatus.VERIFIED.value,
                    source_documents=[
                        f"Company JSON page {p}" for p, _, _ in gt_pages
                    ],
                    details={
                        "pages": [
                            {"page": p, "keywords": kw} for p, _, kw in gt_pages
                        ],
                    },
                ))

        # ============================================================
        # Step 7: Company/entity name verification
        # ============================================================
        page_texts: Optional[dict[int, str]] = None
        if self.company:
            page_texts = {}
            for pt in self.company.page_texts:
                m = re.search(r"page[_\s](\d+)", pt.page_key, re.IGNORECASE)
                if m:
                    try:
                        page_texts[int(m.group(1))] = pt.text
                    except (ValueError, TypeError):
                        pass  # skip malformed page keys

        entity_check = self._entity_checker.check(
            company_name=self.company_name,
            evidence_items=None,
            page_texts=page_texts,
        )
        self._entity_check = entity_check

        if entity_check.has_mismatch:
            for mismatch in entity_check.mismatches:
                all_facts.append(self._make_fact(
                    statement=(
                        f"Company name mismatch detected: found "
                        f"'{mismatch['found']}' where '{mismatch['expected']}' "
                        f"was expected. Source: {mismatch['document']}."
                    ),
                    fact_type=FactType.CONFLICT.value,
                    status=FactStatus.VERIFIED.value,
                    details=mismatch,
                ))

        # ============================================================
        # Step 8: Numeric verification of requirements
        # ============================================================
        all_data_points: list[dict] = []

        # 8a. Structured data points extracted by the filter
        #
        # A data point's own {field, value} pair is frequently NOT enough to
        # tell what period/year it belongs to (e.g. field="annual_turnover").
        # That context usually exists elsewhere on the same evidence item
        # (document_name, snippet, page) but was previously dropped here,
        # which is why year-mapping downstream (NumericVerifier) silently
        # failed even when the year was visible right on the document name.
        # Carry it through instead of discarding it.
        for item in filter_result.related_items:
            for dp in item.extracted_data_points:
                # Defensive copy so we don't mutate the original
                dp_copy = dict(dp)
                dp_copy.setdefault("source_doc", item.document_name)
                dp_copy.setdefault("document_name", item.document_name)
                dp_copy.setdefault("snippet", self._safe_str(item.snippet))
                dp_copy.setdefault("page", item.page)
                all_data_points.append(dp_copy)

        # 8b. Raw field values from related items
        for item in filter_result.related_items:
            fv_str = self._safe_str(item.field_value)
            all_data_points.append({
                "field": self._safe_str(item.field_name),
                "value": fv_str,
                "is_numeric": self._is_numeric_value(item.field_value),
                "source_doc": self._safe_str(item.document_name),
                "document_name": self._safe_str(item.document_name),
                "snippet": self._safe_str(item.snippet),
                "page": item.page,
            })

        # 8c. Verify each parsed requirement.
        #
        # Routing rule (production):
        #   - NUMERIC requirements (threshold/comparison): Python is
        #     authoritative. Feed it the numeric data points. We do NOT require
        #     a keyword match on field names, because financial figures are
        #     often stored under generic field names (e.g. "value", "amount").
        #     If keyword-matched numeric points exist we prefer those; otherwise
        #     we fall back to ALL numeric points so a real number is never missed.
        #   - INFORMATIONAL requirements (existence/count): scoped by keyword so
        #     the observation is relevant, but the result never blocks a verdict.
        #
        # Grouping rule (production):
        #   criterion_parser.py may split one criterion into several
        #   RequirementCheck objects. Most of the time each belongs to
        #   its own `logic_group` (an implicit AND between independent
        #   conditions) and is verified exactly as before. But when two
        #   or more requirements share a `logic_group` (criterion_parser
        #   tags OR-alternatives this way, e.g. "min net worth of $5M OR
        #   a bank guarantee of $500k"), verifying them independently and
        #   dumping every resulting fact into the same flat facts list
        #   would let the FIRST alternative's failure block the verdict
        #   even if the bidder satisfied the second one. Those groups go
        #   through verify_requirement_group() instead, which produces a
        #   single authoritative "at least one alternative met" fact and
        #   demotes each alternative's own fact to informational.
        requirement_groups: dict[int, list[RequirementCheck]] = {}
        for req in parsed.requirements:
            requirement_groups.setdefault(req.logic_group, []).append(req)

        for group_reqs in requirement_groups.values():
            if len(group_reqs) == 1:
                req = group_reqs[0]
                req_data_points = self._select_requirement_data_points(req, all_data_points)
                
                # NEW STEP 8d: Validate data points with LLM before Python verification
                # This ensures only correct, relevant data points reach the numeric verifier
                validated_result = self._data_point_validator.validate_data_points(req, req_data_points)
                
                # Convert validated points back to the format numeric_verifier expects
                validated_data_points = [
                    {
                        "field": vp.field,
                        "value": vp.value,
                        "is_numeric": vp.is_numeric,
                        "period": vp.period,
                        "source_doc": vp.source_doc,
                        "document_name": vp.document_name,
                        "snippet": vp.snippet,
                        "page": vp.page,
                    }
                    for vp in validated_result.validated_points
                ]
                
                # Record validation facts for audit trail
                if validated_result.rejected_points:
                    all_facts.append(self._make_fact(
                        statement=(
                            f"LLM validated data points for '{req.target}': "
                            f"{len(validated_result.validated_points)} accepted, "
                            f"{len(validated_result.rejected_points)} rejected. "
                            f"Reason: {validated_result.selection_logic}"
                        ),
                        fact_type=FactType.COUNT.value,
                        status=FactStatus.VERIFIED.value,
                        details={
                            "validation_method": validated_result.validation_method,
                            "accepted_count": len(validated_result.validated_points),
                            "rejected_count": len(validated_result.rejected_points),
                            "selection_logic": validated_result.selection_logic,
                            "rejected_reasons": [rp.rejection_reason for rp in validated_result.rejected_points[:5]],
                        },
                    ))
                
                # Now pass validated points to numeric verifier
                num_result = self._numeric_verifier.verify_requirement(req, validated_data_points)
            else:
                # For OR-alternative groups, validate each alternative's data points
                validated_group_results = []
                validated_group_data_points = []
                
                for req in group_reqs:
                    req_data_points = self._select_requirement_data_points(req, all_data_points)
                    validated_result = self._data_point_validator.validate_data_points(req, req_data_points)
                    validated_group_results.append(validated_result)
                    
                    validated_data_points = [
                        {
                            "field": vp.field,
                            "value": vp.value,
                            "is_numeric": vp.is_numeric,
                            "period": vp.period,
                            "source_doc": vp.source_doc,
                            "document_name": vp.document_name,
                            "snippet": vp.snippet,
                            "page": vp.page,
                        }
                        for vp in validated_result.validated_points
                    ]
                    validated_group_data_points.append(validated_data_points)
                
                num_result = self._numeric_verifier.verify_requirement_group(
                    group_reqs, validated_group_data_points
                )
            all_facts.extend(num_result.facts)

        # ============================================================
        # Step 9: Page-level text verification facts
        # ============================================================
        verified_count = 0
        unverified_count = 0

        for item in filter_result.related_items:
            if item.page_text_verified:
                verified_count += 1
                if item.page_text_match:
                    all_facts.append(self._make_fact(
                        statement=(
                            f"Python verified evidence on page {item.page} of "
                            f"'{item.document_name}': {item.page_text_summary}"
                        ),
                        fact_type=FactType.EXISTENCE.value,
                        status=FactStatus.VERIFIED.value,
                        source_documents=[item.document_name],
                        details={
                            "page": item.page,
                            "page_text_match": item.page_text_match[:300],
                        },
                    ))
            else:
                unverified_count += 1

        if verified_count or unverified_count:
            all_facts.append(self._make_fact(
                statement=(
                    f"Python performed page-level text verification: "
                    f"{verified_count} evidence item(s) verified against Company JSON, "
                    f"{unverified_count} could not be verified "
                    f"(page text not available)."
                ),
                fact_type=FactType.COUNT.value,
                status=FactStatus.VERIFIED.value,
                details={
                    "verified": verified_count,
                    "unverified": unverified_count,
                },
            ))

        # ============================================================
        # Step 10: Deduplicate and renumber
        # ============================================================
        all_facts = self._deduplicate_facts(all_facts)
        all_facts = self._renumber_facts(all_facts)
        self._generated_facts = all_facts

        return all_facts

    # ------------------------------------------------------------------
    # Evidence builders
    # ------------------------------------------------------------------

    def _build_all_evidence(self) -> list[Evidence]:
        """Build COMPLETE evidence list from all matched documents."""
        evidence: list[Evidence] = []
        for req in self.module2.documents:
            for doc in req.matched_documents:
                if not doc.records:
                    evidence.append(Evidence(
                        document_name=doc.document_name,
                        page=doc.pages[0] if doc.pages else 0,
                        text=doc.summary or doc.document_name,
                        document_id=doc.document_id,
                        llm_finding=(
                            f"Matched document '{doc.document_name}' "
                            f"for requirement '{req.requirement_document}' "
                            f"with status {doc.status}"
                        ),
                        summary=(
                            f"Document-level match for requirement "
                            f"'{req.requirement_document}'"
                        ),
                    ))
                for rec in doc.records:
                    for f in rec.fields:
                        evidence.append(Evidence(
                            document_name=doc.document_name,
                            page=f.page,
                            text=self._safe_str(f.snippet),
                            document_id=doc.document_id,
                            llm_finding=(
                                f"{self._safe_str(f.name)} = "
                                f"{self._safe_str(f.value)} "
                                f"(confidence: {self._safe_str(f.confidence)})"
                            ),
                            summary=(
                                f"Extracted field '{self._safe_str(f.name)}' "
                                f"with value '{self._safe_str(f.value)}'"
                            ),
                        ))
        return evidence

    def _build_related_evidence(
        self, related_items: list[FilteredEvidence]
    ) -> list[Evidence]:
        """Build evidence list from LLM-filtered related items with page verification."""
        evidence: list[Evidence] = []
        for item in related_items:
            # Prefer verified page-text match, fall back to snippet, then field value
            display_text = (
                self._safe_str(item.page_text_match)
                or self._safe_str(item.snippet)
                or self._safe_str(item.field_value)
            )
            field_str = self._safe_str(item.field_name)
            value_str = self._safe_str(item.field_value)
            what_str = self._safe_str(item.what_it_shows) or f"{field_str} = {value_str}"

            evidence.append(Evidence(
                document_name=item.document_name,
                page=item.page,
                text=display_text,
                document_id=item.document_id,
                llm_finding=what_str,
                summary=f"{what_str} (field: {field_str})",
                page_text_verified=item.page_text_verified,
                page_text_match=self._safe_str(item.page_text_match),
            ))
        return evidence

    # ------------------------------------------------------------------
    # Missing information identification
    # ------------------------------------------------------------------

    def identify_missing_information(self) -> tuple[list[str], list[str]]:
        """
        Identify what is missing / noteworthy for verification.

        DIVISION OF LABOUR:
          Document presence ("was Form V submitted?", "is this certificate
          present?") is the LLM's job — it reads the full evidence and decides.
          Python therefore does NOT push document-presence concerns into the
          verdict-blocking ``missing`` list. Instead they go into
          ``informational_notes`` so they remain visible to the LLM and the
          user, but never force a REVIEW on their own.

          Python keeps only genuinely verdict-relevant, deterministic concerns
          in ``missing``:
            - No Module 2 documents at all (nothing to verify)
            - No evidence extracted at all (nothing to verify)
            - Entity/company name mismatch (a string-equality check Python owns)

        Returns:
            (missing, informational_notes)
        """
        missing: list[str] = []
        notes: list[str] = []

        if not self.module2.documents:
            missing.append(
                "No document requirements found in Module 2 output. "
                "Nothing could be verified."
            )
            return missing, notes

        criterion_lower = self.module2.criterion.lower()

        # Common filler words that should not be used as matching keywords
        filler = {
            "copy", "submitted", "enclosed", "attached", "duly",
            "signed", "original", "attested", "notarised", "valid",
        }

        # Document-presence observations are INFORMATIONAL ONLY (LLM decides).
        for req in self.module2.documents:
            found_docs = [
                d for d in req.matched_documents
                if (d.status or "").upper() == "FOUND" and d.records
            ]

            # At least one FOUND doc with data → nothing to note.
            if found_docs:
                continue

            # Check whether this requirement is actually criterion-relevant
            req_keywords = {
                w.lower()
                for w in (req.requirement_document or "").split()
                if len(w) >= 3
            } - filler

            if not any(kw in criterion_lower for kw in req_keywords):
                # Not related to what the criterion asks — skip
                continue

            not_found_docs = [
                d for d in req.matched_documents
                if (d.status or "").upper() == "NOT_FOUND"
            ]
            partial_empty_docs = [
                d for d in req.matched_documents
                if d.is_partial and not d.records
            ]

            if not_found_docs:
                for doc in not_found_docs:
                    notes.append(
                        f"Module 2 marked document '{doc.document_name}' as NOT_FOUND. "
                        f"Confirming whether this document is actually present is left "
                        f"to the LLM, which reviews the full evidence."
                    )
            elif partial_empty_docs:
                for doc in partial_empty_docs:
                    notes.append(
                        f"Document '{doc.document_name}' was PARTIALLY matched with no "
                        f"extractable data. The LLM determines whether the required "
                        f"information is present."
                    )
            elif not req.matched_documents:
                notes.append(
                    f"No documents matched for requirement "
                    f"'{req.requirement_document}'. The LLM determines whether the "
                    f"required information is present elsewhere."
                )

        # No evidence extracted at all → genuinely nothing for anyone to verify.
        if self._filter_result is not None:
            if self._filter_result.related_count == 0:
                if self._filter_result.total_evidence > 0:
                    notes.append(
                        f"Out of {self._filter_result.total_evidence} extracted "
                        f"evidence item(s), the filter marked none as directly related "
                        f"to the criterion. The LLM makes the final relevance call."
                    )
                else:
                    missing.append(
                        "No evidence was extracted from the matched documents. "
                        "Cannot verify the criterion."
                    )

        # Entity name inconsistency — a deterministic name check Python owns.
        if self._entity_check is not None and self._entity_check.has_mismatch:
            missing.append(
                "Company name inconsistency found in documents. "
                "Manual review required to verify correct entity."
            )

        return missing, notes

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_criterion_requirements(self) -> list[dict]:
        """Return parsed criterion requirements."""
        return self._criterion_requirements

    def get_related_evidence(self) -> list[Evidence]:
        """Return related evidence items."""
        return self._related_evidence

    def get_entity_check(self):
        """Return entity check result."""
        return self._entity_check

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_facts(facts: list[VerifiedFact]) -> list[VerifiedFact]:
        seen: set[str] = set()
        unique: list[VerifiedFact] = []
        for f in facts:
            key = f"{f.type}:{f.statement}"
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    @staticmethod
    def _renumber_facts(facts: list[VerifiedFact]) -> list[VerifiedFact]:
        for i, f in enumerate(facts, 1):
            f.details = dict(f.details or {})
            f.details["original_id"] = f.id
            f.id = f"FACT{i:03d}"
        return facts