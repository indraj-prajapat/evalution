"""
Verification Engine - Main orchestrator (v5.0 — Production Grade, Fully Generic).

Pipeline:
  1. Parse Module 2 Output and Company JSON
  2. Parse criterion into checkable requirements (generic, no hardcoded domains)
  3. LLM filters evidence: related vs unrelated to the criterion
  4. Related evidence verified against Company JSON page-level text
  5. Company/entity name checked for consistency
  6. Python handles ALL number comparisons and calculations
  7. Generate verified facts about WHAT WAS FOUND vs WHAT IS NEEDED
  8. Determine verdict:
     - PASS: 100% of requirements satisfied with verified evidence
     - FAIL: Something directly and conclusively fails the requirement
     - REVIEW: Everything else — with user-friendly reason
  9. LLM produces final user-friendly summary (optional)

KEY v5.0 PRINCIPLES:
  - NO hardcoded criteria, domains, or certificate types
  - Works for ANY tender criterion
  - LLM separates related evidence from unrelated
  - Python handles ALL numeric operations (LLM can hallucinate numbers)
  - Evidence verified against actual Company JSON page text
  - Company name checked across documents
  - User-friendly summary references specific evidence
  - FAIL only for directly provable failures
"""

from __future__ import annotations

import json
import re
from typing import Optional, Union
from pathlib import Path

from .models import (
    Module2Output,
    CompanyJSON,
    VerificationReport,
    Verdict,
    VerifiedFact,
    FactStatus,
    FactType,
    Evidence,
    LLMEvaluation,
    parse_module2_output,
    parse_company_json,
)
from .ground_truth import GroundTruthSearcher
from .fact_generator import FactGenerator
from .llm_evaluator import LLMEvaluator
from .criterion_parser import parse_criterion, extract_document_types
from .dependency import resolve_cross_references


class VerificationEngine:
    """
    Production-grade Verification Engine.

    Fully generic — works for ANY tender criterion.
    No hardcoded domains, certificate types, or domain-specific logic.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        use_llm: bool = True,
    ):
        self.module2: Optional[Module2Output] = None
        self.company: Optional[CompanyJSON] = None
        self.ground_truth: Optional[GroundTruthSearcher] = None
        self.fact_generator: Optional[FactGenerator] = None
        self.use_llm = use_llm
        self.llm_evaluator = LLMEvaluator(model=model, api_key=api_key) if use_llm else None

    # -----------------------------------------------------------------------
    # Verdict Determination — Generic, no hardcoded domains
    # -----------------------------------------------------------------------

    def _determine_verdict(self, report: VerificationReport) -> str:
        """
        Determine PASS/FAIL/REVIEW based on verified facts.

        DIVISION OF LABOUR (production policy):
          Python is authoritative ONLY for NUMBERS — threshold checks,
          comparisons and calculations. Those produce facts that can drive a
          FAIL or REVIEW verdict.

          Document presence, keyword/text existence and "how many items were
          found" are the LLM's job. Python records them as INFORMATIONAL facts
          (details["informational"] == True). Such facts NEVER block the
          verdict — a missing document is decided by the LLM downstream, not by
          Python failing to find it in raw numeric data points.

        GENERIC VERDICT POLICY (works for ANY criterion):
          PASS: No authoritative numeric failure, no unverifiable number, no
                entity name mismatch, no genuinely missing required info.
          FAIL: An authoritative numeric THRESHOLD/COMPARISON is provably FALSE
                (and nothing else is uncertain).
          REVIEW: Everything else.
        """
        facts = report.verified_facts
        missing = report.missing_information

        def _is_informational(fact: VerifiedFact) -> bool:
            return bool(fact.details.get("informational", False))

        # Authoritative facts = everything Python is allowed to decide on.
        # Informational (document/keyword/count) facts are excluded entirely.
        authoritative = [f for f in facts if not _is_informational(f)]

        # Numeric pass/fail signals (threshold & comparison only).
        false_numeric_facts = [
            f for f in authoritative
            if f.status == FactStatus.FALSE.value
            and f.type in ("THRESHOLD", "COMPARISON")
        ]
        unverified_numeric_facts = [
            f for f in authoritative
            if f.status == FactStatus.UNVERIFIED.value
            and f.type in ("THRESHOLD", "COMPARISON", "NUMERIC")
        ]
        # COUNT / EXISTENCE are also blocking for PASS (but never for FAIL).
        unverified_count_existence_facts = [
            f for f in authoritative
            if f.status == FactStatus.UNVERIFIED.value
            and f.type in (FactType.COUNT.value, FactType.EXISTENCE.value)
        ]
        conflicting_facts = [
            f for f in authoritative
            if f.status == FactStatus.CONFLICTING.value
        ]

        # Entity name conflicts (always relevant — manual review).
        entity_conflicts = [
            f for f in facts
            if f.type == FactType.CONFLICT.value
            and "mismatch" in f.statement.lower()
        ]
        has_entity_mismatch = len(entity_conflicts) > 0

        # ============================================================
        # PASS: nothing authoritative fails or is uncertain
        # ============================================================
        if (not false_numeric_facts
                and not unverified_numeric_facts
                and not unverified_count_existence_facts
                and not conflicting_facts
                and not has_entity_mismatch
                and not missing):

            report.verdict = Verdict.PASS.value
            report.summary = self._build_user_summary(report, "PASS")
            return Verdict.PASS.value

        # ============================================================
        # FAIL: ONLY when a number is DIRECTLY provably out of bounds
        # AND nothing else is uncertain.
        # ============================================================
        if (false_numeric_facts
                and not missing
                and not unverified_numeric_facts
                and not has_entity_mismatch):
            report.verdict = Verdict.FAIL.value
            report.summary = self._build_user_summary(report, "FAIL")
            report.reason = [f.statement for f in false_numeric_facts]
            return Verdict.FAIL.value

        # ============================================================
        # REVIEW: Everything else
        # ============================================================
        review_reasons = []

        if missing:
            review_reasons.extend(missing)

        if unverified_count_existence_facts:
            for f in unverified_count_existence_facts:
                review_reasons.append(f.statement)

        if unverified_numeric_facts:
            for f in unverified_numeric_facts:
                review_reasons.append(f.statement)

        if false_numeric_facts:
            for f in false_numeric_facts:
                review_reasons.append(f.statement)

        if has_entity_mismatch:
            for f in entity_conflicts:
                review_reasons.append(f.statement)

        if conflicting_facts:
            for f in conflicting_facts:
                if f not in entity_conflicts:
                    review_reasons.append(f.statement)

        if not review_reasons:
            review_reasons.append(
                "Insufficient verified evidence to conclusively determine PASS or FAIL."
            )

        report.verdict = Verdict.REVIEW.value
        report.summary = self._build_user_summary(report, "REVIEW")
        report.reason = review_reasons

        return Verdict.REVIEW.value

    # -----------------------------------------------------------------------
    # User-Friendly Summary Builder (Generic)
    # -----------------------------------------------------------------------

    def _build_user_summary(self, report: VerificationReport, verdict: str) -> str:
        """
        Build a USER-FRIENDLY summary.

        GENERIC: works for any criterion type.
        References specific facts, evidence, and pages.
        """
        criterion = report.criterion
        company = report.company_name
        facts = report.verified_facts
        evidence = report.verdict_evidence or report.evidence

        if verdict == "PASS":
            evidence_refs = self._build_evidence_references(evidence[:3])
            return (
                f"{company} meets the requirement: {criterion}. "
                f"All verified evidence confirms the criteria are satisfied. "
                f"{evidence_refs}"
            ).strip()

        if verdict == "FAIL":
            false_thresholds = [
                f for f in facts
                if f.status == FactStatus.FALSE.value
                and f.type in ("THRESHOLD", "COMPARISON")
            ]
            if false_thresholds:
                reasons = "; ".join(f.statement for f in false_thresholds[:2])
                return (
                    f"{company} does NOT meet the requirement: {criterion}. "
                    f"Reason: {reasons}. "
                    f"This was verified through Python's numerical analysis."
                )
            return (
                f"{company} does NOT meet the requirement: {criterion}. "
                f"The verified evidence conclusively shows the criteria are not satisfied."
            )

        # REVIEW — build detailed, user-friendly explanation
        parts = [f"Requirement: {criterion}."]

        # What was found
        found_facts = [
            f for f in facts
            if f.status in (FactStatus.VERIFIED.value, FactStatus.TRUE.value)
            and f.type in (FactType.EXISTENCE.value, FactType.ENTITY_MATCH.value,
                            FactType.COUNT.value)
        ]

        if found_facts:
            for f in found_facts[:3]:
                parts.append(f"What was found: {f.statement}")

        # What was NOT found or failed — only AUTHORITATIVE numeric issues.
        # Informational document/keyword facts are excluded so they never
        # appear as a blocking "issue" in the summary.
        not_found_facts = [
            f for f in facts
            if not f.details.get("informational", False)
            and f.status in (FactStatus.FALSE.value, FactStatus.UNVERIFIED.value)
            and f.type in ("THRESHOLD", "COMPARISON", "NUMERIC")
        ]
        if not_found_facts:
            for f in not_found_facts[:3]:
                parts.append(f"Issue: {f.statement}")

        # Entity name issues
        entity_issues = [
            f for f in facts
            if f.type == FactType.CONFLICT.value
            and "mismatch" in f.statement.lower()
        ]
        if entity_issues:
            for f in entity_issues[:2]:
                parts.append(f"Name check: {f.statement}")

        # Missing info
        if report.missing_information:
            parts.append(
                f"Additionally, {len(report.missing_information)} piece(s) of "
                f"information could not be verified."
            )

        # Evidence references
        if evidence:
            evidence_refs = self._build_evidence_references(evidence[:3])
            parts.append(evidence_refs)

        return " ".join(parts)

    def _build_evidence_references(self, evidence: list[Evidence]) -> str:
        """Build a user-friendly reference to evidence items."""
        if not evidence:
            return ""
        refs = []
        for e in evidence[:5]:
            verified = " (page text verified)" if e.page_text_verified else ""
            refs.append(
                f"'{e.document_name}' on page {e.page}{verified}"
            )
        if len(evidence) > 5:
            refs.append(f"and {len(evidence) - 5} more document(s)")
        return "Evidence checked: " + "; ".join(refs) + "."

    # -----------------------------------------------------------------------
    # Ground Truth Search (Generic)
    # -----------------------------------------------------------------------

    def _extract_search_keywords(self, criterion: str) -> list[str]:
        """Extract search keywords from criterion text — fully generic."""
        keywords = []

        # Capitalized terms (like CISA, DISA, ITR)
        caps = re.findall(r"\b([A-Z]{2,}(?:/[A-Z]{2,})*)\b", criterion)
        for c in caps:
            keywords.extend(t.strip() for t in c.split("/"))

        # Terms after "copy of"
        for m in re.finditer(
            r"copy of\s+([A-Za-z][\w\s]*?)(?:\s+shall|\s+and|\s+is|\.$)",
            criterion, re.IGNORECASE
        ):
            keywords.append(m.group(1).strip())

        # Generic meaningful words
        if not keywords:
            stop = {
                "the", "and", "for", "should", "have", "has", "had", "with",
                "from", "been", "being", "shall", "will", "must", "may",
                "this", "that", "which", "whose", "where", "when", "what",
                "not", "but", "are", "was", "were", "is", "am", "be",
            }
            words = re.findall(r"\b[A-Za-z]{3,}\b", criterion)
            keywords = [w for w in words if w.lower() not in stop][:10]

        return list(dict.fromkeys(keywords))

    def _extract_required_doc_types(self, criterion: str) -> list[str]:
        """
        Extract required document types — fully generic.

        Delegates to criterion_parser.extract_document_types() so this
        logic lives in exactly one place instead of two near-identical
        regexes drifting apart over time.
        """
        return extract_document_types(criterion)

    # -----------------------------------------------------------------------
    # Main Pipeline
    # -----------------------------------------------------------------------

    def run(
        self,
        module2_data: dict,
        company_json_data: Optional[dict] = None,
        company_name: str = "",
        sibling_context: Optional[dict[str, dict]] = None,
    ) -> VerificationReport:
        """
        Run the full verification pipeline.

        v5.0 Pipeline (fully generic):
          1. Parse inputs
          2. Parse criterion into requirements (generic regex)
          3. Ground truth search
          4. LLM filters evidence (related vs unrelated)
          5. Verify related evidence against Company JSON page text
          6. Check company/entity name consistency
          7. Python handles ALL numeric operations
          8. Generate verified facts
          9. Python preliminary verdict
          10. Cross-criterion dependency check (v3, see dependency.py)
          11. LLM final evaluation (user-friendly summary)

        Args:
            sibling_context: Optional results from OTHER key points already
                evaluated in this same tender run, keyed by criterion_id --
                see dependency.py module docstring for the exact contract.
                Only consulted when this criterion's parsed text contains a
                conditional exempt/relaxed clause (e.g. "not applicable to
                Joint Ventures"). Building this dict and deciding evaluation
                order across key points is the CALLER's responsibility
                (pipeline.py or an upstream planning stage) -- this engine
                only consumes it, one criterion at a time.
        """
        report = VerificationReport()
        report.company_name = company_name

        # ---- Step 1: Parse inputs ----
        self.module2 = parse_module2_output(module2_data)
        report.criterion_id = self.module2.criterion_id
        report.criterion = self.module2.criterion

        if company_json_data:
            self.company = parse_company_json(company_json_data)

        # ---- Step 2: Parse criterion ----
        # NOTE: parse_criterion() accepts an optional `llm_fallback`
        # callable(clause_text) -> list[RequirementCheck] for clauses the
        # regex tier can't confidently parse (multi-clause soup, unusual
        # phrasing, non-English text). Not wired in yet — would need a
        # small adapter around self.llm_evaluator to turn a clause string
        # into RequirementCheck objects. Left as regex-only for now to
        # keep this step synchronous/cheap in the common case.
        parsed = parse_criterion(self.module2.criterion)
        report.criterion_requirements = [
            {
                "requirement": r.description,
                "check_type": r.check_type,
                "target": r.target,
            }
            for r in parsed.requirements
        ]

        # ---- Step 3: Ground Truth Search ----
        criterion = self.module2.criterion
        search_keywords = self._extract_search_keywords(criterion)
        required_doc_types = self._extract_required_doc_types(criterion)

        m2_doc_ids = []
        m2_doc_names = []
        for req in self.module2.documents:
            for doc in req.matched_documents:
                m2_doc_ids.append(doc.document_id)
                m2_doc_names.append(doc.document_name)

        if self.company:
            self.ground_truth = GroundTruthSearcher(self.company)
            report.ground_truth_verification = self.ground_truth.perform_full_search(
                criterion=criterion,
                module2_doc_ids=m2_doc_ids,
                module2_doc_names=m2_doc_names,
                search_keywords=search_keywords,
                required_doc_types=required_doc_types,
                context_keywords=search_keywords,
            )
        else:
            report.ground_truth_verification.completed = False
            report.missing_information.append(
                "Company JSON was not provided. Ground truth verification skipped."
            )

        # ---- Steps 4-7: Fact Generation (LLM filter + page verify + entity check + numeric) ----
        self.fact_generator = FactGenerator(
            self.module2,
            self.company,
            self.ground_truth,
            api_key=self.llm_evaluator.api_key if self.llm_evaluator else None,
            model=self.llm_evaluator.model if self.llm_evaluator else "gpt-4o-mini",
            company_name=company_name,
        )

        report.verified_facts = self.fact_generator.generate_all_facts()

        # ---- Step 8: Build evidence lists ----
        report.evidence = self.fact_generator._build_all_evidence()
        report.verdict_evidence = self.fact_generator.get_related_evidence()

        # ---- Step 9: Criterion requirements from fact generator ----
        fg_requirements = self.fact_generator.get_criterion_requirements()
        if fg_requirements:
            report.criterion_requirements = fg_requirements

        # ---- Step 10: Missing information (verdict-relevant) + notes (LLM-owned) ----
        missing, doc_notes = self.fact_generator.identify_missing_information()
        report.missing_information.extend(missing)
        report.informational_notes.extend(doc_notes)

        # ---- Step 11: Informational notes ----
        entity_check = self.fact_generator.get_entity_check()
        if entity_check and entity_check.has_mismatch:
            report.informational_notes.append(entity_check.verification_summary)

        # Unrelated evidence count as info
        filter_result = getattr(self.fact_generator, '_filter_result', None)
        if filter_result and filter_result.unrelated_count > 0:
            report.informational_notes.append(
                f"{filter_result.unrelated_count} evidence item(s) were determined "
                f"to be unrelated to the criterion and were excluded from verification."
            )

        # ---- Step 12: Python preliminary verdict ----
        python_verdict = self._determine_verdict(report)

        # ---- Step 13: Cross-criterion dependency check (v3) ----
        # See dependency.py module docstring for the full design and the
        # sibling_context contract. This can only OVERRIDE toward REVIEW
        # (never silently turn a REVIEW/FAIL into PASS) -- an exempt/relaxed
        # clause we couldn't confirm is a reason for a human to look, not a
        # reason to auto-pass.
        if parsed.cross_reference_hints:
            dep_resolution = resolve_cross_references(parsed.cross_reference_hints, sibling_context)
            report.dependency_notes.extend(dep_resolution.to_notes())
            if dep_resolution.any_triggered or dep_resolution.any_unresolved:
                python_verdict = Verdict.REVIEW.value
                report.verdict = Verdict.REVIEW.value  # _apply_llm_verdict reads report.verdict, not the local var
                if dep_resolution.any_unresolved and not dep_resolution.any_triggered:
                    report.missing_information.append(
                        "This criterion has a conditional exempt/relaxed clause that "
                        "could not be checked against other key points -- see "
                        "dependency_notes for details."
                    )

        report.python_verdict = python_verdict

        # ---- Step 14: LLM final evaluation ----
        if self.use_llm and self.llm_evaluator:
            llm_eval = self.llm_evaluator.evaluate(report, company_name=company_name)
            self._apply_llm_verdict(report, llm_eval)
        else:
            report.python_verdict = python_verdict
            report.verdict_source = "python_authoritative"
            report.llm_evaluation = LLMEvaluation(
                verdict=python_verdict,
                confidence=0.5,
                reasoning="LLM evaluation was disabled. Verdict based on Python deterministic analysis only.",
                key_findings=[f"Python verdict: {python_verdict}"],
                risks=["LLM evaluation not performed — reduced confidence"],
                recommendations=["Enable LLM evaluation for higher confidence verdicts"],
                python_verdict_agreement="AGREES (LLM not used)",
                model_used="none",
            )

        return report

    def _apply_llm_verdict(self, report: VerificationReport, llm_eval: LLMEvaluation) -> None:
        """
        Apply the LLM's verdict against Python's preliminary verdict using an
        explicit transition policy, instead of unconditionally trusting the
        LLM's raw verdict string.

        _determine_verdict() (above, untouched) is the sole authority on
        NUMBERS: a Python FAIL can only come from a directly-provable numeric
        threshold/comparison failure, and a Python PASS means no such failure
        exists. The LLM owns document/text judgment and REVIEW-bucket
        adjudication, but the numeric determination itself must never be
        silently reversed.

        Transition policy (python_verdict -> what the LLM is allowed to do):
          FAIL   -> stays FAIL, or softens to REVIEW with caveats.
                    A PASS suggestion is REJECTED outright: Python's numeric
                    FAIL stands, and the rejected LLM suggestion is recorded
                    (not applied) so it's visible for audit.
          PASS   -> stays PASS, or softens to REVIEW freely (the LLM found
                    something inconclusive in the evidence).
                    A FAIL suggestion is honored only as an explicit,
                    hard-flagged override (Python found no numeric failure,
                    so this is the LLM asserting a document is conclusively
                    absent) -- never applied silently.
          REVIEW -> the LLM's verdict is fully trusted; this is exactly the
                    "everything else" bucket Python defers to the LLM for.

        Sets report.verdict_source to "python_authoritative" when the final
        verdict is Python's own determination (LLM agreed or was rejected),
        or "llm_adjusted" when the LLM's judgment changed the outcome.
        """
        python_verdict = report.verdict  # Python's preliminary verdict, pre-adjustment
        llm_verdict = llm_eval.verdict if llm_eval.verdict in ("PASS", "FAIL", "REVIEW") else "REVIEW"

        override_note: Optional[str] = None

        if python_verdict == Verdict.FAIL.value:
            if llm_verdict == "PASS":
                # Numbers are Python's territory -- reject the override.
                final_verdict = Verdict.FAIL.value
                verdict_source = "python_authoritative"
                override_note = (
                    "LLM suggested PASS but this was REJECTED: Python's "
                    "numeric FAIL determination is authoritative and cannot "
                    "be overridden by the LLM's document/text judgment. "
                    f"(Unapplied LLM reasoning: {llm_eval.reasoning})"
                )
            elif llm_verdict == "REVIEW":
                final_verdict = Verdict.REVIEW.value
                verdict_source = "llm_adjusted"
            else:  # FAIL, agrees
                final_verdict = Verdict.FAIL.value
                verdict_source = "python_authoritative"

        elif python_verdict == Verdict.PASS.value:
            if llm_verdict == "FAIL":
                # Allowed, but only as an explicit hard-flagged override --
                # never silent, since Python found no numeric failure at all.
                final_verdict = Verdict.FAIL.value
                verdict_source = "llm_adjusted"
                override_note = (
                    "⚠ HARD FLAG: LLM overrode Python's PASS to FAIL based on "
                    "document/text evidence (no numeric failure was found by "
                    "Python). This override was applied but should be "
                    "manually verified."
                )
            elif llm_verdict == "REVIEW":
                final_verdict = Verdict.REVIEW.value
                verdict_source = "llm_adjusted"
            else:  # PASS, agrees
                final_verdict = Verdict.PASS.value
                verdict_source = "python_authoritative"

        else:  # python_verdict == REVIEW -- the LLM fully owns this bucket
            final_verdict = llm_verdict
            verdict_source = "llm_adjusted" if llm_verdict != python_verdict else "python_authoritative"

        report.python_verdict = python_verdict
        report.verdict = final_verdict
        report.verdict_source = verdict_source

        if llm_eval.reasoning and "LLM evaluation failed" not in llm_eval.reasoning:
            report.summary = llm_eval.reasoning

        reason_additions = ["---"] + llm_eval.key_findings
        if override_note:
            reason_additions.append(override_note)
        report.reason = report.reason + reason_additions

        report.llm_evaluation = llm_eval


def run_verification(
    module2_input: Union[str, dict, Path],
    company_json_input: Optional[Union[str, dict, Path]] = None,
    company_name: str = "",
    output_path: Optional[Union[str, Path]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    use_llm: bool = True,
) -> VerificationReport:
    """Convenience function to run verification."""
    if isinstance(module2_input, (str, Path)):
        with open(module2_input, "r", encoding="utf-8") as f:
            module2_data = json.load(f)
    else:
        module2_data = module2_input

    company_data = None
    if company_json_input is not None:
        if isinstance(company_json_input, (str, Path)):
            with open(company_json_input, "r", encoding="utf-8") as f:
                company_data = json.load(f)
        else:
            company_data = company_json_input

    engine = VerificationEngine(api_key=api_key, model=model, use_llm=use_llm)
    report = engine.run(module2_data, company_data, company_name=company_name)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.to_json(indent=2))

    return report