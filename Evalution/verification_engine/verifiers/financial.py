"""
Financial Verifier - verifies financial-data criteria.

Handles:
  - Turnover, revenue, profit calculations
  - Currency parsing and comparison (INR with Crore/Lakh)
  - Financial year grouping and comparison
  - Average, sum, min, max over financial data
  - Threshold comparison for financial values
"""

from __future__ import annotations

from typing import Optional
from .base import BaseVerifier
from ..models import (
    Module2Output,
    CompanyJSON,
    DocumentRequirement,
    VerifiedFact,
    FactType,
    FactStatus,
    parse_currency,
    parse_financial_year,
    parse_number,
)


class FinancialVerifier(BaseVerifier):
    """Verifies financial-data-related tender requirements."""

    def verify(self, requirement: DocumentRequirement) -> list[VerifiedFact]:
        """
        Verify financial requirements from extracted data.

        Collects all numeric/financial values, groups by FY if possible,
        and computes sum/average/min/max.
        """
        facts: list[VerifiedFact] = []

        # Collect all financial fields across all matched docs
        all_values: list[float] = []
        all_raw: list[str] = []
        fy_groups: dict[str, list[float]] = {}  # FY -> [values]
        source_fields_set = set()
        source_docs_set = set()

        for doc in requirement.matched_documents:
            for rec in doc.records:
                for f in rec.fields:
                    parsed = parse_currency(f.value) or parse_number(f.value)
                    if parsed is not None:
                        all_values.append(parsed)
                        all_raw.append(f.value)
                        source_fields_set.add(f.name)
                        source_docs_set.add(doc.document_name)

                    # Check for financial year
                    fy = parse_financial_year(f.value)
                    if fy:
                        fy_groups.setdefault(fy, [])

                    # If field name contains financial keywords, try to group
                    if any(kw in f.name.lower() for kw in
                           ["turnover", "revenue", "profit", "amount", "total"]):
                        parsed = parse_currency(f.value) or parse_number(f.value)
                        if parsed is not None:
                            # Try to find FY from group_value
                            if rec.group_value:
                                fy = parse_financial_year(rec.group_value)
                                if fy:
                                    fy_groups.setdefault(fy, []).append(parsed)
                                else:
                                    fy_groups.setdefault("UNKNOWN", []).append(parsed)

        # --- FACT: Count of financial values found ---
        facts.append(self._make_fact(
            statement=(
                f"Python found {len(all_values)} parseable financial/numeric value(s) "
                f"across {len(source_docs_set)} document(s)."
            ),
            fact_type=FactType.COUNT.value,
            status=FactStatus.VERIFIED.value if all_values else FactStatus.NOT_FOUND.value,
            source_fields=sorted(source_fields_set),
            source_documents=sorted(source_docs_set),
            details={
                "count": len(all_values),
                "raw_values": all_raw,
            },
        ))

        if not all_values:
            return facts

        # --- FACT: Sum ---
        total = sum(all_values)
        facts.append(self._make_fact(
            statement=f"Python computed the sum of all financial values as {total:,.2f}.",
            fact_type=FactType.NUMERIC.value,
            status=FactStatus.VERIFIED.value,
            source_fields=sorted(source_fields_set),
            details={"sum": total, "value_count": len(all_values)},
        ))

        # --- FACT: Average ---
        avg = total / len(all_values)
        facts.append(self._make_fact(
            statement=f"Python computed the average of all financial values as {avg:,.2f}.",
            fact_type=FactType.NUMERIC.value,
            status=FactStatus.VERIFIED.value,
            source_fields=sorted(source_fields_set),
            details={"average": avg, "value_count": len(all_values)},
        ))

        # --- FACT: Min / Max ---
        facts.append(self._make_fact(
            statement=(
                f"Python computed minimum financial value as {min(all_values):,.2f} "
                f"and maximum as {max(all_values):,.2f}."
            ),
            fact_type=FactType.NUMERIC.value,
            status=FactStatus.VERIFIED.value,
            details={"min": min(all_values), "max": max(all_values)},
        ))

        # --- FACT: Financial year grouping ---
        if len(fy_groups) > 1 or ("UNKNOWN" not in fy_groups and fy_groups):
            for fy, vals in sorted(fy_groups.items()):
                if fy == "UNKNOWN":
                    continue
                fy_total = sum(vals)
                fy_avg = fy_total / len(vals) if vals else 0
                facts.append(self._make_fact(
                    statement=(
                        f"Python grouped {len(vals)} value(s) under {fy}: "
                        f"sum = {fy_total:,.2f}, average = {fy_avg:,.2f}."
                    ),
                    fact_type=FactType.FINANCIAL_YEAR.value,
                    status=FactStatus.VERIFIED.value,
                    details={
                        "financial_year": fy,
                        "values": vals,
                        "sum": fy_total,
                        "average": fy_avg,
                        "count": len(vals),
                    },
                ))

        return facts

    def verify_threshold(
        self,
        requirement: DocumentRequirement,
        threshold: float,
        comparison: str = "gte",  # gte, gt, lte, lt, eq
        label: str = "threshold",
    ) -> list[VerifiedFact]:
        """Verify if a computed value meets a threshold."""
        facts: list[VerifiedFact] = []

        # Collect all values
        all_values = []
        for doc in requirement.matched_documents:
            for rec in doc.records:
                for f in rec.fields:
                    parsed = parse_currency(f.value) or parse_number(f.value)
                    if parsed is not None:
                        all_values.append(parsed)

        if not all_values:
            facts.append(self._make_fact(
                statement=f"Python could not verify {label}: no parseable financial values found.",
                fact_type=FactType.THRESHOLD.value,
                status=FactStatus.UNVERIFIED.value,
            ))
            return facts

        total = sum(all_values)
        avg = total / len(all_values)

        comparisons = {
            "gte": ("greater than or equal to", lambda a, b: a >= b),
            "gt": ("greater than", lambda a, b: a > b),
            "lte": ("less than or equal to", lambda a, b: a <= b),
            "lt": ("less than", lambda a, b: a < b),
            "eq": ("equal to", lambda a, b: abs(a - b) < 0.01),
        }

        comp_text, comp_fn = comparisons.get(comparison, comparisons["gte"])
        result = comp_fn(avg, threshold)

        facts.append(self._make_fact(
            statement=(
                f"Python verified that the average value ({avg:,.2f}) is "
                f"{'NOT ' if not result else ''}{comp_text} the required {label} ({threshold:,.2f})."
            ),
            fact_type=FactType.COMPARISON.value,
            status=FactStatus.TRUE.value if result else FactStatus.FALSE.value,
            details={
                "average": avg,
                "threshold": threshold,
                "comparison": comparison,
                "result": result,
            },
        ))

        return facts