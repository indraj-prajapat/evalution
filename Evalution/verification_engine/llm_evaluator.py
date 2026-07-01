"""
LLM Evaluator - Generic final evaluation layer.

v5.0 — Fully generic. Works for ANY criterion type.
  - Produces user-friendly summary
  - Gives final verdict
  - References specific evidence and facts
  - No domain-specific logic whatsoever
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from .models import LLMEvaluation, Verdict

_VALID_VERDICTS = tuple(v.value for v in Verdict)

_SYSTEM_PROMPT = """You are an expert tender compliance evaluator. Your job is to
evaluate a Python-generated verification report and produce a final PASS/FAIL/REVIEW
verdict for a SINGLE tender criterion.

## WHAT YOU RECEIVE
- The tender criterion text
- Company name
- VERIFIED FACTS (all numerical/deterministic computations done by Python)
- RELATED evidence ONLY (evidence directly related to the criterion, verified against page text)
- Missing information list
- Python's preliminary verdict

## CRITICAL: YOUR ROLE & THE DIVISION OF LABOUR
There is a strict split of responsibility:

PYTHON owns NUMBERS ONLY and its numeric facts are AUTHORITATIVE — never
re-compute or override them:
- Number parsing, threshold checks, comparisons, sums, averages, min/max, durations.
- These appear as THRESHOLD / COMPARISON / NUMERIC facts. Trust them completely.

YOU (the LLM) own DOCUMENT & TEXT understanding:
- Whether a required document was actually submitted (e.g. "Form V", a
  certificate, an agreement) is YOUR decision, made from the evidence.
- Some Python facts are explicitly marked INFORMATIONAL (their statement says
  presence "is assessed by the LLM" or details.informational = true, or
  details.owner = "llm"). These are mere observations — they are NOT failures.
  NEVER treat an informational EXISTENCE or COUNT fact as a reason that
  something is missing or unverified.
- If you can see in the RELATED EVIDENCE that a required document/keyword is
  present, then it IS present — regardless of any informational Python note
  saying Python "observed no related values". Python does not decide document
  presence; you do.

Your job is to:
1. Trust Python's numeric facts as-is.
2. Decide document/text presence yourself from the evidence.
3. Produce a USER-FRIENDLY summary.
4. Give the final verdict.

Do NOT write summaries claiming "Python could not verify the submission of
<document>" — document submission is YOUR call, not Python's.

## VERDICT RULES (STRICT)

### PASS
Only if: ALL requirements from the criterion are 100% met with verified evidence,
no relevant contradictions, no missing information, and no entity name mismatches.

### FAIL
ONLY if: Something DIRECTLY and CONCLUSIVELY proves the requirement is not met.
Examples:
  - A numeric threshold is provably below the required level (Python calculated — trust it)
  - You can clearly see in the evidence that a required document is genuinely
    absent AND there is no ambiguity or missing context that could contain it
Do NOT issue FAIL merely because a Python informational fact says it "observed
no related values" — that is not proof of absence.

### REVIEW
For everything else, including:
  - Wrong type found (e.g., different certificate than required)
  - Missing information that prevents conclusive determination
  - Partial documents with no extractable data
  - Uncertainty in any verification step
  - Company/entity name mismatch found
  - Evidence could not be verified against page text
  - The criterion is only PARTIALLY met (some but not all sub-requirements satisfied)
  - The criterion appears EXEMPT or NOT APPLICABLE to this company/tender (e.g.
    the requirement only applies to a category the company doesn't fall into)

There are only three possible verdict values: PASS, FAIL, REVIEW. There is no
separate "PARTIAL", "EXEMPT", or "NOT_APPLICABLE" verdict — these situations
all resolve to REVIEW. When one of these applies, say so explicitly and
specifically in your summary/reasoning (e.g. "This requirement appears
partially met — X is satisfied but Y is missing" or "This criterion appears
not applicable, as it only applies to joint ventures and this bid is from a
single entity") so a human reviewer immediately understands *why* it's a
REVIEW rather than treating it as a generic uncertain case.

## USER-FRIENDLY SUMMARY
Your summary MUST be written for the USER, not a developer.
It MUST explain:
  - What was found in the evidence (specific documents, page numbers)
  - What was needed (from the criterion)
  - Why this verdict was given
  - What additional information is needed (if REVIEW)

BAD: "FACT003 returned NOT_FOUND for certificate_type field"
GOOD: "The submitted documents include a Form I (page 50) that mentions 'CISA/DISA Certified: 1', 
but the actual certificate document (page 36) shows an FCA membership — not a CISA/DISA/DISSA certification. 
The partnership certificate found is for firm admission, not information security certification."

## ANTI-HALLUCINATION RULES
- You MUST NOT redo any numerical computation — trust Python's verified facts
- You MUST NOT assume facts not in evidence
- Your reasoning must reference SPECIFIC fact IDs (e.g., FACT001)
You must return your output strictly as a JSON object matching the required schema. Do not include any markdown code blocks or wrapping text outside the valid JSON.
"""

_EVALUATION_PROMPT = """## TENDER CRITERION
Criterion ID: {criterion_id}
Criterion Text: {criterion}
Company Name: {company_name}

## VERIFIED FACTS ({fact_count} facts)
{facts_section}

## RELATED EVIDENCE ({evidence_count} items)
(Page-verified evidence directly related to the criterion)
{evidence_section}

## MISSING INFORMATION ({missing_count} items)
{missing_section}

## INFORMATIONAL NOTES ({notes_count} items)
(Observations that do NOT block the verdict — document presence is YOUR call)
{notes_section}

## PYTHON'S PRELIMINARY VERDICT: {python_verdict}

---
Evaluate the above and produce your FINAL verdict (PASS/FAIL/REVIEW — no other
values exist).
Remember:
- FAIL only for DIRECTLY provable violations (not for "wrong type found")
- Wrong/incompatible type found → REVIEW
- Partially met, exempt, or not applicable → REVIEW, but say so explicitly in the summary
- Summary must be user-friendly with specific document/page references
You must return your output strictly as a JSON object matching the required schema. Do not include any markdown code blocks or wrapping text outside the valid JSON.
"""


class LLMEvaluator:
    """
    Sends the Python verification report to LLM for final evaluation.
    Fully generic — works for any criterion type.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
    ):
        from Evalution.client import get_llm_model
        
        self.model = model or get_llm_model()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not found. Set it in .env or pass api_key parameter."
                )
            from Evalution.client import get_openrouter_client
            self._client = get_openrouter_client(api_key=self.api_key, model=self.model)
        return self._client

    def _build_facts_section(self, report) -> str:
        if not report.verified_facts:
            return "No verified facts were generated by Python."
        lines = []
        for f in report.verified_facts:
            lines.append(
                f"[{f.id}] ({f.type}, status: {f.status})\n"
                f"  {f.statement}"
            )
            if f.source_documents:
                lines.append(f"  Sources: {', '.join(f.source_documents[:5])}")
            lines.append("")
        return "\n".join(lines)

    def _build_evidence_section(self, report) -> str:
        evidence_items = report.verdict_evidence or report.evidence
        if not evidence_items:
            return "No related evidence items."
        lines = []
        for i, e in enumerate(evidence_items[:30], 1):
            verified_tag = " [PAGE VERIFIED]" if e.page_text_verified else ""
            page_match = ""
            if e.page_text_match:
                page_match = f"\n   Page Text: {e.page_text_match[:200]}{'...' if len(e.page_text_match) > 200 else ''}"
            lines.append(
                f"{i}. [{e.document_name}, page {e.page}]{verified_tag}\n"
                f"   Text: {e.text[:200]}{'...' if len(e.text) > 200 else ''}\n"
                f"   Finding: {e.llm_finding}"
                f"{page_match}"
            )
        if len(evidence_items) > 30:
            lines.append(f"... and {len(evidence_items) - 30} more evidence items")
        return "\n".join(lines)

    def _build_missing_section(self, report) -> str:
        if not report.missing_information:
            return "No missing information identified."
        return "\n".join(f"- {m}" for m in report.missing_information)

    def _build_notes_section(self, report) -> str:
        if not report.informational_notes:
            return "No informational notes."
        return "\n".join(f"- {n}" for n in report.informational_notes)

    def evaluate(self, report, company_name: str = "") -> LLMEvaluation:
        user_prompt = _EVALUATION_PROMPT.format(
            criterion_id=report.criterion_id,
            criterion=report.criterion,
            company_name=company_name or "Unknown",
            fact_count=len(report.verified_facts),
            facts_section=self._build_facts_section(report),
            evidence_count=len(report.verdict_evidence or report.evidence),
            evidence_section=self._build_evidence_section(report),
            missing_count=len(report.missing_information),
            missing_section=self._build_missing_section(report),
            notes_count=len(report.informational_notes),
            notes_section=self._build_notes_section(report),
            python_verdict=report.verdict,
        )

        try:
            client = self._get_client()
            response = client.create_chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            return self._parse_response(raw, report.verdict)

        except Exception as e:
            return LLMEvaluation(
                verdict=report.verdict,
                confidence=0.0,
                reasoning=f"LLM evaluation failed: {str(e)}. Using Python verdict: {report.verdict}.",
                key_findings=[f"Python verdict: {report.verdict}"],
                risks=["LLM unavailable — verdict from Python analysis only"],
                recommendations=["Set OPENAI_API_KEY to enable LLM evaluation"],
                python_verdict_agreement="N/A (LLM unavailable)",
                model_used=self.model,
                raw_response="",
            )

    def _parse_response(self, raw: str, python_verdict: str) -> LLMEvaluation:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        verdict = str(data.get("verdict", "REVIEW")).upper()
        if verdict not in _VALID_VERDICTS:
            verdict = "REVIEW"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = data.get("reasoning", data.get("summary", ""))
        key_findings = data.get("key_findings", data.get("findings", []))
        if isinstance(key_findings, str):
            key_findings = [key_findings]

        risks = data.get("risks", [])
        if isinstance(risks, str):
            risks = [risks]

        recommendations = data.get("recommendations", [])
        if isinstance(recommendations, str):
            recommendations = [recommendations]

        # ------------------------------------------------------------------
        # Check the LLM's verdict against Python's preliminary verdict
        # before it can be trusted blindly.
        #
        # Python is authoritative ONLY for numeric facts (THRESHOLD /
        # COMPARISON). Per engine._determine_verdict, a Python FAIL can only
        # be produced by a directly-provable numeric violation. The LLM does
        # not own numbers, so it must never be trusted to silently flip that
        # FAIL back to PASS.
        #
        # A Python PASS, by contrast, only means "nothing numeric failed" —
        # the LLM legitimately owns document/text presence and can still
        # downgrade to FAIL/REVIEW based on evidence it sees. That direction
        # is always considered trustworthy.
        #
        # NOTE: this method does not decide "who wins" — that policy call
        # belongs in engine.py. It only flags the conflict and returns
        # enough information (agreement label + adjusted confidence +
        # an explicit recommendation) for the caller to make that call.
        # ------------------------------------------------------------------
        untrusted_override = (python_verdict == "FAIL" and verdict == "PASS")

        if verdict == python_verdict:
            agreement = "AGREES"
        elif python_verdict == "REVIEW":
            agreement = "PARTIAL"
        elif untrusted_override:
            agreement = "DISAGREES (UNTRUSTED - cannot override Python's numeric FAIL)"
        else:
            agreement = "DISAGREES"

        if untrusted_override:
            # Numbers are Python's territory; don't let a confident-looking
            # LLM response mask that this verdict contradicts an
            # authoritative numeric determination.
            confidence = min(confidence, 0.3)
            recommendations = list(recommendations) + [
                "LLM verdict (PASS) conflicts with Python's authoritative "
                "numeric FAIL determination. Python owns all numeric "
                "threshold/comparison facts, so this override should not be "
                "honored without manual review."
            ]

        return LLMEvaluation(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            key_findings=key_findings,
            risks=risks,
            recommendations=recommendations,
            python_verdict_agreement=agreement,
            model_used=self.model,
            raw_response=raw,
        )