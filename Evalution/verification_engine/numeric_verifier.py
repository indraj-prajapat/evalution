"""
Numeric Verifier - Pure Python number operations for verification.

Handles ALL numeric computations: parsing, comparison, threshold checks,
counting, date arithmetic. LLM must NEVER do these.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Union
from dataclasses import dataclass, field

from .models import (
    VerifiedFact,
    FactType,
    FactStatus,
    parse_currency,
    parse_number,
    parse_date,
    parse_financial_year,
    resolve_financial_year,
)
from .criterion_parser import RequirementCheck


@dataclass
class NumericVerificationResult:
    """Result of numeric verification."""
    facts: list[VerifiedFact] = field(default_factory=list)
    all_passed: bool = True
    any_failed: bool = False
    summary: str = ""


@dataclass
class ThresholdStatistics:
    """
    Common intermediate representation for a threshold computation.

    DESIGN NOTE (why this exists):
    `_verify_threshold` used to compute `nums`/`total`/`avg`/`min_val`/
    `max_val`/`values` independently inside two different branches
    (the "past N financial years" branch and the "plain values" branch),
    and then had a block of *shared* code after the branching that read
    those names back. Because one branch never assigned `values`, that
    shared code could throw UnboundLocalError depending on which path
    was taken.

    The fix is structural: every branch that computes threshold
    statistics must build and return one of these objects. Nothing
    downstream of the branch is allowed to read a bare local variable
    that any particular branch might have skipped - it can only read
    fields off this object, which every branch is required to fully
    populate. Branch-specific scratch variables (e.g. `year_amount_pairs`,
    `fy_to_amounts`, `per_year_amounts`) stay local to the helper that
    produces them and never escape it.
    """
    values: list[float]
    raw_values: list[str]
    total: float
    average: float
    min_value: float
    max_value: float
    count: int
    # Facts that are specific to *how* the statistics were derived
    # (e.g. the "mapped financial years" fact). Kept separate from the
    # generic summary fact so each branch can still contribute its own
    # narrative without smuggling extra variables past the branch boundary.
    branch_facts: list[VerifiedFact] = field(default_factory=list)


# Sentinel-free outcome type: either branch returns a finished
# NumericVerificationResult (when it cannot proceed, e.g. no usable
# evidence), or it returns fully-populated ThresholdStatistics. Calling
# code distinguishes the two with isinstance() and never has to guess
# which intermediate variables exist.
ThresholdOutcome = Union[NumericVerificationResult, ThresholdStatistics]


class NumericVerifier:
    """
    Pure Python deterministic number verifier.

    DIVISION OF LABOUR (production rule):
      - Python owns NUMBERS ONLY: parsing numeric values, threshold checks,
        comparisons, sums, averages, min/max, durations. These results are
        AUTHORITATIVE and may drive a FAIL/REVIEW verdict.
      - Document presence / keyword / textual existence and "how many were
        found" questions are the LLM's domain. Python may still record them,
        but ONLY as INFORMATIONAL facts (status VERIFIED, details["informational"]
        = True). Such facts NEVER block the verdict.

    This guarantees that a perfectly fine submission is never knocked down to
    REVIEW just because Python failed to "find" a document in raw numeric data
    points — that job belongs to the LLM evidence filter.
    """

    def __init__(
        self,
        fact_prefix: str = "NUM",
        fiscal_year_start_month: int = 4,
        date_day_first: Optional[bool] = True,
    ):
        self._fact_counter = 0
        self._fact_prefix = fact_prefix
        # Configurable rather than hardcoded to the Indian April-March
        # convention: callers that know the tender's actual fiscal-year
        # convention (e.g. from tender metadata) can override this per
        # instance instead of the verifier silently assuming April.
        self._fiscal_year_start_month = fiscal_year_start_month
        # Controls how ambiguous NN/NN/YYYY dates are read when mapping
        # evidence to financial years; see models.parse_date(). Defaults
        # to day-first (Indian convention) for backward compatibility.
        # Pass None to refuse to guess on genuinely ambiguous dates.
        self._date_day_first = date_day_first

    def _next_id(self) -> str:
        self._fact_counter += 1
        return f"{self._fact_prefix}{self._fact_counter:03d}"

    def _make_fact(
        self,
        statement: str,
        fact_type: str,
        status: str,
        source_fields: Optional[list[str]] = None,
        source_documents: Optional[list[str]] = None,
        details: Optional[dict] = None,
        informational: bool = False,
    ) -> VerifiedFact:
        details = dict(details or {})
        # Mark whether this fact is authoritative (numeric) or informational
        # (document/keyword presence). The verdict logic uses this flag.
        details.setdefault("informational", informational)
        return VerifiedFact(
            id=self._next_id(),
            statement=statement,
            type=fact_type,
            status=status,
            computed_by="python",
            source_fields=source_fields or [],
            source_documents=source_documents or [],
            details=details,
        )

    def verify_requirement(
        self,
        requirement: RequirementCheck,
        evidence_data_points: list[dict],
    ) -> NumericVerificationResult:
        """
        Verify a single requirement against evidence data points.

        Args:
            requirement: The parsed requirement to verify.
            evidence_data_points: List of {"field": str, "value": str, "is_numeric": bool} 
                                  from the LLM evidence filter.

        Returns:
            NumericVerificationResult with facts.
        """
        result = NumericVerificationResult()
        source_docs = list(set(
            dp.get("source_doc", "") for dp in evidence_data_points if dp.get("source_doc")
        ))

        # AUTHORITATIVE numeric checks — Python owns these completely.
        if requirement.check_type == "threshold":
            result = self._verify_threshold(requirement, evidence_data_points, source_docs)
        elif requirement.check_type == "comparison":
            result = self._verify_comparison(requirement, evidence_data_points, source_docs)
        # INFORMATIONAL checks — document/keyword presence and counts of items.
        # These belong to the LLM. Python records them but never blocks a verdict.
        elif requirement.check_type == "count":
            result = self._verify_count(requirement, evidence_data_points, source_docs)
        elif requirement.check_type == "existence":
            result = self._verify_existence(requirement, evidence_data_points, source_docs)
        else:
            # Unknown check type -> treat as informational existence.
            result = self._verify_existence(requirement, evidence_data_points, source_docs)

        return result

    def verify_requirement_group(
        self,
        requirements: list[RequirementCheck],
        data_points_by_requirement: list[list[dict]],
    ) -> NumericVerificationResult:
        """
        Verify a set of requirements that are OR-alternatives of one
        another - i.e. criterion_parser gave them the same
        `logic_group` with connective "OR" (e.g. "minimum net worth of
        $5M OR a bank guarantee of $500k").

        Why this exists: verify_requirement() only ever looks at ONE
        RequirementCheck. If a caller just ran it separately for every
        alternative and dumped all the resulting facts into the same
        flat report.verified_facts list, the engine's verdict logic
        (which treats every authoritative FALSE/UNVERIFIED fact as
        blocking) would have no way to know two facts are alternatives
        rather than both being mandatory - a bidder who satisfies the
        SECOND alternative but not the first would incorrectly get
        FAILed or sent to REVIEW off the first alternative's fact
        alone. This method produces one authoritative group-level fact
        instead, so the OR semantics survive into the verdict step.

        Policy (mirrors this file's overall PASS/FAIL/REVIEW design):
          - PASS the group the moment ANY alternative is confirmed.
          - FAIL the group ONLY when every alternative is a numeric
            (threshold/comparison) check AND every one of them is
            definitively FALSE. If any alternative is an
            existence/count check, or simply unverified, we cannot be
            100% certain no alternative was actually satisfied, so we
            do not FAIL - we fall through to REVIEW instead.
          - REVIEW otherwise.

        Each alternative's own fact(s) are still included (so the
        reasoning stays fully auditable) but are marked informational,
        since only the GROUP summary fact below is authoritative - not
        any single alternative on its own.
        """
        result = NumericVerificationResult()

        if not requirements:
            return result

        member_results = [
            self.verify_requirement(req, dps)
            for req, dps in zip(requirements, data_points_by_requirement)
        ]

        for mr in member_results:
            for f in mr.facts:
                f.details["informational"] = True
            result.facts.extend(mr.facts)

        any_passed = any(mr.all_passed for mr in member_results)

        numeric_pairs = [
            (req, mr) for req, mr in zip(requirements, member_results)
            if req.check_type in ("threshold", "comparison")
        ]
        all_numeric_and_all_definitively_false = (
            len(numeric_pairs) == len(requirements)
            and len(numeric_pairs) > 0
            and all(mr.any_failed for _, mr in numeric_pairs)
        )

        if any_passed:
            status = FactStatus.TRUE.value
            narrative = "at least one alternative was satisfied."
        elif all_numeric_and_all_definitively_false:
            status = FactStatus.FALSE.value
            narrative = "every numeric alternative was definitively not met."
        else:
            status = FactStatus.UNVERIFIED.value
            narrative = (
                "no alternative could be confirmed, and at least one "
                "alternative is not a numeric check Python can definitively "
                "fail, so this cannot be treated as a proven violation."
            )

        joined = " OR ".join(f"({req.description})" for req in requirements)
        source_documents = sorted({
            doc for mr in member_results for f in mr.facts for doc in f.source_documents
        })

        result.facts.append(self._make_fact(
            statement=(
                f"Python evaluated {len(requirements)} OR-alternative "
                f"requirement(s): {joined}. Group result: {narrative}"
            ),
            fact_type=FactType.THRESHOLD.value if numeric_pairs else FactType.EXISTENCE.value,
            status=status,
            source_documents=source_documents,
            details={
                "check_type": "or_group",
                "group_size": len(requirements),
                "any_passed": any_passed,
                "all_numeric_and_all_definitively_false": all_numeric_and_all_definitively_false,
                "member_all_passed": [mr.all_passed for mr in member_results],
                "member_any_failed": [mr.any_failed for mr in member_results],
            },
            informational=False,
        ))

        result.all_passed = any_passed
        result.any_failed = all_numeric_and_all_definitively_false and not any_passed
        return result

    def _parse_values(self, data_points: list[dict]) -> list[tuple[float, str]]:
        """
        Parse all numeric values from data points.
        Returns list of (parsed_value, raw_value) tuples.
        """
        values = []
        for dp in data_points:
            raw = str(dp.get("value", ""))
            parsed = parse_currency(raw) or parse_number(raw)
            if parsed is not None:
                values.append((parsed, raw))
        return values

    def _verify_count(
        self,
        requirement: RequirementCheck,
        data_points: list[dict],
        source_docs: list[str],
    ) -> NumericVerificationResult:
        """
        Verify a count requirement (e.g., 'at least one CISA cert').

        Policy change:
          - COUNT is NOT informational-only anymore.
          - If we cannot confidently satisfy the minimum count from the
            extracted evidence values, emit UNVERIFIED so the overall
            verdict becomes REVIEW (FAIL is reserved for numeric
            THRESHOLD/COMPARISON).
        """
        result = NumericVerificationResult()
        min_count = requirement.min_count or 1

        # Count unique non-empty values purely as an observation.
        unique_values = set()
        for dp in data_points:
            val = str(dp.get("value", "")).strip()
            if val and val.lower() not in ("na", "n/a", "nil", "none"):
                unique_values.add(val)

        observed = len(unique_values)
        meets = observed >= min_count

        result.facts.append(self._make_fact(
            statement=(
                f"Python observed {observed} distinct value(s) related to "
                f"'{requirement.target}' in the extracted data (criterion asks "
                f"for at least {min_count})."
            ),
            fact_type=FactType.COUNT.value,
            status=FactStatus.TRUE.value if meets else FactStatus.UNVERIFIED.value,
            source_fields=[requirement.target],
            source_documents=source_docs,
            details={
                "check_type": "count",
                "observed_distinct_values": observed,
                "min_required": min_count,
                "unique_values": sorted(unique_values)[:50],
            },
            informational=False,
        ))

        result.all_passed = meets
        result.any_failed = not meets
        return result

    # ------------------------------------------------------------------
    # Threshold verification
    #
    # `_verify_threshold` itself contains NO branch-local variables that
    # outlive their branch. It picks one of two statistics-producing
    # helpers, each of which either (a) gives up and returns a finished
    # NumericVerificationResult, or (b) returns a fully-populated
    # ThresholdStatistics. Everything after that point — the summary
    # fact and the threshold comparison — reads exclusively from the
    # returned object.
    # ------------------------------------------------------------------

    def _extract_past_years_n(self, requirement: RequirementCheck) -> Optional[int]:
        """Pull PAST_N_FINANCIAL_YEARS=<n> out of the requirement description, if present."""
        match = re.search(
            r"PAST_N_FINANCIAL_YEARS=(\d+)", str(requirement.description or "")
        )
        return int(match.group(1)) if match else None

    def _compute_mapped_financial_year_statistics(
        self,
        requirement: RequirementCheck,
        data_points: list[dict],
        source_docs: list[str],
        past_years_n: int,
    ) -> ThresholdOutcome:
        """
        Conservative handling for "average annual ... past N financial years"
        to prevent wrong aggregation across unrelated numbers.

        Returns either an early NumericVerificationResult (when evidence
        cannot be reliably mapped to distinct financial years) or a
        ThresholdStatistics object built from one amount per financial year.
        """
        # Branch-local scratch variables. These never need to be seen
        # outside this method.
        year_amount_pairs: list[tuple[str, float]] = []
        distinct_years: set[str] = set()
        unmapped_amounts: list[str] = []

        for dp in data_points:
            raw_value = str(dp.get("value", "") or "")
            amt = parse_currency(raw_value) or parse_number(raw_value)
            if amt is None:
                continue

            # The financial-year signal is frequently NOT inside field/value
            # (e.g. field="annual_turnover", value="21,59,345.02") - it lives
            # in surrounding context instead: an explicit "period" tag (if the
            # extractor supplied one), the source document's name/filename
            # ("Balance Sheet FY 2021-22 ..."), or a snippet/date near the
            # figure ("as at 31st March 2022"). Search ALL of them - looking
            # only at field+value silently drops evidence that is otherwise
            # perfectly usable.
            fy = resolve_financial_year(
                dp.get("period"),
                dp.get("field"),
                raw_value,
                dp.get("source_doc"),
                dp.get("document_name"),
                dp.get("snippet"),
                fiscal_year_start_month=self._fiscal_year_start_month,
                day_first=self._date_day_first,
            )

            if fy:
                distinct_years.add(fy)
                year_amount_pairs.append((fy, amt))
            else:
                unmapped_amounts.append(raw_value)

        # Require at least N distinct mapped years with amounts.
        if len(distinct_years) < past_years_n or len(year_amount_pairs) < past_years_n:
            unverified = NumericVerificationResult()
            found_desc = (
                f"Years actually mapped: {sorted(distinct_years)} "
                f"({len(year_amount_pairs)} amount(s))."
                if distinct_years else
                "No amount could be tied to any financial year."
            )
            unmapped_desc = (
                f" {len(unmapped_amounts)} numeric value(s) were found but had "
                f"no identifiable year in their field, value, source document "
                f"name, or snippet: {unmapped_amounts[:5]}."
                if unmapped_amounts else ""
            )
            unverified.facts.append(self._make_fact(
                statement=(
                    f"Python cannot verify average over past {past_years_n} "
                    f"financial years: only {len(distinct_years)} distinct "
                    f"year(s) could be mapped from the evidence, need "
                    f"{past_years_n}. {found_desc}{unmapped_desc}"
                ),
                fact_type=FactType.THRESHOLD.value,
                status=FactStatus.UNVERIFIED.value,
                source_documents=source_docs,
                details={
                    "check_type": "threshold",
                    "past_years_n": past_years_n,
                    "distinct_financial_years_found": sorted(distinct_years),
                    "year_amount_pairs_count": len(year_amount_pairs),
                    "year_amount_pairs": [
                        {"financial_year": fy, "amount": amt}
                        for fy, amt in year_amount_pairs
                    ],
                    "unmapped_amounts": unmapped_amounts[:20],
                },
            ))
            unverified.all_passed = False
            unverified.any_failed = False
            return unverified

        # Compute average using one amount per financial year (pick max
        # to be conservative when multiple figures exist for same year).
        fy_to_amounts: dict[str, list[float]] = {}
        for fy, amt in year_amount_pairs:
            fy_to_amounts.setdefault(fy, []).append(amt)

        per_year_amounts: list[float] = []
        per_year_raw: list[str] = []
        for fy in sorted(fy_to_amounts.keys()):
            amounts = fy_to_amounts[fy]
            if len(amounts) > 1:
                amt = sum(amounts) / len(amounts)
                per_year_raw.append(f"{fy}: {amt:,.2f} (avg of {len(amounts)} values)")
            else:
                amt = amounts[0]
                per_year_raw.append(f"{fy}: {amt:,.2f}")
            per_year_amounts.append(amt)

        total = sum(per_year_amounts)
        count = len(per_year_amounts)
        avg = total / count
        min_val = min(per_year_amounts)
        max_val = max(per_year_amounts)

        # Branch-specific narrative fact (averages only; no misleading
        # aggregation over all numeric datapoints). Carried home inside the
        # common ThresholdStatistics object rather than appended directly,
        # so the caller controls fact ordering uniformly across branches.
        branch_fact = self._make_fact(
            statement=(
                f"Python computed average across mapped financial years: "
                f"years={count}, average={avg:,.2f}, "
                f"min={min_val:,.2f}, max={max_val:,.2f}."
            ),
            fact_type=FactType.NUMERIC.value,
            status=FactStatus.VERIFIED.value,
            source_documents=source_docs,
            details={
                "check_type": "threshold",
                "mapped_year_amounts_count": count,
                "average": avg,
                "min": min_val,
                "max": max_val,
                "threshold_mode": "mapped_financial_years",
            },
        )

        return ThresholdStatistics(
            values=per_year_amounts,
            raw_values=per_year_raw,
            total=total,
            average=avg,
            min_value=min_val,
            max_value=max_val,
            count=count,
            branch_facts=[branch_fact],
        )

    def _compute_simple_statistics(
        self,
        requirement: RequirementCheck,
        data_points: list[dict],
        source_docs: list[str],
    ) -> ThresholdOutcome:
        """
        Plain-values branch: parse every numeric value out of the evidence
        and compute basic statistics over all of them.

        Returns either an early NumericVerificationResult (no parseable
        values) or a ThresholdStatistics object.
        """
        parsed = self._parse_values(data_points)  # branch-local only

        if not parsed:
            unverified = NumericVerificationResult()
            unverified.facts.append(self._make_fact(
                statement=(
                    f"Python cannot verify threshold for '{requirement.target}': "
                    f"no parseable numeric values found in the evidence."
                ),
                fact_type=FactType.THRESHOLD.value,
                status=FactStatus.UNVERIFIED.value,
                source_documents=source_docs,
                details={
                    "check_type": "threshold",
                    "raw_data_points": [dp.get("value", "") for dp in data_points],
                },
            ))
            unverified.all_passed = False
            unverified.any_failed = False
            return unverified

        nums = [v[0] for v in parsed]
        raw_values = [v[1] for v in parsed]
        total = sum(nums)
        count = len(nums)
        avg = total / count

        return ThresholdStatistics(
            values=nums,
            raw_values=raw_values,
            total=total,
            average=avg,
            min_value=min(nums),
            max_value=max(nums),
            count=count,
            branch_facts=[],
        )

    def _build_threshold_comparison_fact(
        self,
        requirement: RequirementCheck,
        statistics: ThresholdStatistics,
        source_docs: list[str],
    ) -> tuple[VerifiedFact, bool]:
        """
        Build the PASS/FAIL comparison fact for a threshold requirement.
        Reads only from `statistics` - never from branch-local variables.
        """
        threshold = requirement.threshold_value
        operator = requirement.operator or ">="

        comp_fns = {
            ">=": lambda a, b: a >= b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            "<": lambda a, b: a < b,
            "==": lambda a, b: abs(a - b) < 0.01,
        }
        comp_texts = {
            ">=": "greater than or equal to",
            ">": "greater than",
            "<=": "less than or equal to",
            "<": "less than",
            "==": "equal to",
        }

        fn = comp_fns.get(operator, comp_fns[">="])
        text = comp_texts.get(operator, ">=")

        # Choose the aggregate that matches the criterion wording.
        #   - "average"/"per year"/"per annum" -> average of values
        #   - "total"/"aggregate"/"cumulative"/"sum" -> sum of values
        #   - otherwise, when a single value is provided use it directly;
        #     when multiple values are provided default to the MAXIMUM so a
        #     genuinely qualifying single year/value is not unfairly diluted
        #     by an averaging assumption. This is transparent in the fact.
        desc = requirement.description.lower()
        if any(k in desc for k in ("average", "per year", "per annum", "each year")):
            computed = statistics.average
            aggregate_used = "average"
        elif any(k in desc for k in ("total", "aggregate", "cumulative", "combined", "sum")):
            computed = statistics.total
            aggregate_used = "sum"
        elif statistics.count == 1:
            computed = statistics.values[0]
            aggregate_used = "single_value"
        elif any(k in desc for k in ("maximum", "any one", "any single", "each")):
            computed = statistics.max_value
            aggregate_used = "max"
        elif statistics.count == 1:
            computed = statistics.values[0]
            aggregate_used = "single_value"
        else:
            computed = statistics.average
            aggregate_used = "average (default)"

        meets = fn(computed, threshold)

        unit_str = f" {requirement.unit}" if requirement.unit else ""
        fact = self._make_fact(
            statement=(
                f"Python verified: computed value ({computed:,.2f}{unit_str}, "
                f"aggregate={aggregate_used}) is "
                f"{'NOT ' if not meets else ''}{text} "
                f"the required threshold ({threshold:,.2f}{unit_str}). "
                f"Requirement {'MET' if meets else 'NOT MET'}."
            ),
            fact_type=FactType.THRESHOLD.value,
            status=FactStatus.TRUE.value if meets else FactStatus.FALSE.value,
            source_documents=source_docs,
            details={
                "check_type": "threshold",
                "computed_value": computed,
                "aggregate_used": aggregate_used,
                "threshold": threshold,
                "operator": operator,
                "comparison_text": text,
                "threshold_met": meets,
                "unit": requirement.unit,
            },
        )
        return fact, meets

    def _verify_threshold(
        self,
        requirement: RequirementCheck,
        data_points: list[dict],
        source_docs: list[str],
    ) -> NumericVerificationResult:
        """Verify a threshold requirement (e.g., 'turnover >= 25 Lakhs')."""
        past_years_n = self._extract_past_years_n(requirement)

        if past_years_n:
            outcome = self._compute_mapped_financial_year_statistics(
                requirement, data_points, source_docs, past_years_n
            )
        else:
            outcome = self._compute_simple_statistics(requirement, data_points, source_docs)

        # Either branch may bail out early with a finished result (e.g. no
        # usable evidence). When that happens, there is no ThresholdStatistics
        # to read from, so we hand the result straight back.
        if isinstance(outcome, NumericVerificationResult):
            return outcome

        statistics = outcome
        result = NumericVerificationResult()

        # Branch-specific narrative facts (if any), followed by the common
        # summary fact. Both are built purely from `statistics`, so it does
        # not matter which branch produced it.
        result.facts.extend(statistics.branch_facts)
        result.facts.append(self._make_fact(
            statement=(
                f"Python computed financial statistics from {statistics.count} value(s): "
                f"sum = {statistics.total:,.2f}, average = {statistics.average:,.2f}, "
                f"min = {statistics.min_value:,.2f}, max = {statistics.max_value:,.2f}."
            ),
            fact_type=FactType.NUMERIC.value,
            status=FactStatus.VERIFIED.value,
            source_documents=source_docs,
            details={
                "values": statistics.values,
                "raw_values": statistics.raw_values,
                "sum": statistics.total,
                "average": statistics.average,
                "min": statistics.min_value,
                "max": statistics.max_value,
                "count": statistics.count,
            },
        ))

        if requirement.threshold_value is not None:
            comparison_fact, meets = self._build_threshold_comparison_fact(
                requirement, statistics, source_docs
            )
            result.facts.append(comparison_fact)
            result.all_passed = meets
            result.any_failed = not meets

        return result

    def _verify_comparison(
        self,
        requirement: RequirementCheck,
        data_points: list[dict],
        source_docs: list[str],
    ) -> NumericVerificationResult:
        result = NumericVerificationResult()
        values = self._parse_values(data_points)

        if len(values) < 2:
            result.facts.append(self._make_fact(
                statement=(
                    f"Python cannot perform comparison: need at least 2 numeric values, "
                    f"found {len(values)}."
                ),
                fact_type=FactType.COMPARISON.value,
                status=FactStatus.UNVERIFIED.value,
            ))
            result.all_passed = False
            return result

        # FIX #4: Compare ALL consecutive pairs
        comparison_details = []
        for i in range(len(values) - 1):
            v1, r1 = values[i]
            v2, r2 = values[i + 1]
            diff = v1 - v2
            pct = (diff / v2 * 100) if v2 != 0 else float("inf")
            if abs(diff) < 0.01:
                relation = "equal to"
            elif diff > 0:
                relation = "greater than"
            else:
                relation = "less than"
            comparison_details.append({
                "value1": v1, "raw1": r1,
                "value2": v2, "raw2": r2,
                "difference": diff, "percent_difference": pct,
                "relation": relation,
            })

        pair_summaries = [
            f"'{cd['raw1']}' ({cd['value1']:,.2f}) vs '{cd['raw2']}' ({cd['value2']:,.2f}): first is {cd['relation']} second"
            for cd in comparison_details
        ]

        # FIX #5: Evaluate against requirement operator/threshold
        operator = requirement.operator or ">="
        threshold = requirement.threshold_value
        comp_fns = {">=": lambda a,b: a>=b, ">": lambda a,b: a>b,
                    "<=": lambda a,b: a<=b, "<": lambda a,b: a<b,
                    "==": lambda a,b: abs(a-b)<0.01}
        comp_texts = {">=": "greater than or equal to", ">": "greater than",
                    "<=": "less than or equal to", "<": "less than", "==": "equal to"}

        if threshold is not None:
            fn = comp_fns.get(operator, comp_fns[">="])
            text = comp_texts.get(operator, ">=")
            all_met = True
            threshold_results = []
            for v, r in values:
                meets = fn(v, threshold)
                all_met = all_met and meets
                threshold_results.append({"value": v, "raw": r, "meets_threshold": meets})

            result.facts.append(self._make_fact(
                statement=(
                    f"Python compared {len(values)} value(s) against threshold "
                    f"({operator} {threshold:,.2f}): "
                    f"{'ALL' if all_met else 'NOT ALL'} values are {text} the threshold. "
                    f"Pairs: {'; '.join(pair_summaries[:5])}."
                ),
                fact_type=FactType.COMPARISON.value,
                status=FactStatus.TRUE.value if all_met else FactStatus.FALSE.value,  # FIX #5
                source_documents=source_docs,
                details={"check_type": "comparison", "operator": operator,
                        "threshold": threshold, "all_values_met_threshold": all_met,
                        "pair_comparisons": comparison_details,
                        "threshold_results": threshold_results},
            ))
            result.all_passed = all_met
            result.any_failed = not all_met
        else:
            result.facts.append(self._make_fact(
                statement=f"Python compared {len(values)} value(s): {'; '.join(pair_summaries[:5])}.",
                fact_type=FactType.COMPARISON.value,
                status=FactStatus.VERIFIED.value,
                source_documents=source_docs,
                details={"check_type": "comparison", "pair_comparisons": comparison_details},
                informational=True,
            ))
        return result



    def _verify_existence(
        self,
        requirement: RequirementCheck,
        data_points: list[dict],
        source_docs: list[str],
    ) -> NumericVerificationResult:
        """
        Verify an existence requirement.

        Policy change:
          - EXISTENCE is NO LONGER INFORMATIONAL ONLY.
          - If extracted evidence does not provide any usable value for the
            target, emit UNVERIFIED (review). We never emit authoritative
            FALSE for existence to avoid FAIL on non-numeric claims.
        """
        result = NumericVerificationResult()

        found_values = [
            str(dp.get("value", "")).strip()
            for dp in data_points
            if str(dp.get("value", "")).strip()
        ]

        observed_values = [
            v for v in found_values
            if v.lower() not in ("na", "n/a", "nil", "none", "0")
        ]
        observed = bool(observed_values)

        result.facts.append(self._make_fact(
            statement=(
                f"Python observed evidence for existence of '{requirement.target}' "
                f"in extracted data."
                + (
                    f" Related extracted value(s): {', '.join(found_values[:10])}."
                    if found_values else
                    " No related extracted values were observed."
                )
            ),
            fact_type=FactType.EXISTENCE.value,
            status=FactStatus.TRUE.value if observed else FactStatus.UNVERIFIED.value,
            source_fields=[requirement.target],
            source_documents=source_docs,
            details={
                "check_type": "existence",
                "observed_related_values": observed,
                "values": found_values[:50],
                "data_point_count": len(data_points),
            },
            informational=False,
        ))

        result.all_passed = observed
        result.any_failed = not observed
        return result