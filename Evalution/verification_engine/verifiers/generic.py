"""
Generic Verifiers - reusable building-block verifiers.
Fully generic, no domain-specific logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from .base import BaseVerifier
from ..models import (
    Module2Output,
    CompanyJSON,
    DocumentRequirement,
    VerifiedFact,
    FactType,
    FactStatus,
    parse_date,
    parse_number,
    parse_currency,
)


class CountVerifier(BaseVerifier):
    """Counts entities and compares against required minimum."""

    def verify(self, requirement: DocumentRequirement) -> list[VerifiedFact]:
        facts = []
        all_values = []
        source_docs = []

        for doc in requirement.matched_documents:
            for rec in doc.records:
                for f in rec.fields:
                    all_values.append(f.value)
                    if doc.document_name not in source_docs:
                        source_docs.append(doc.document_name)

        unique_count = len(set(all_values))

        facts.append(self._make_fact(
            statement=(
                f"Python found {len(all_values)} extracted value(s) across "
                f"{len(source_docs)} document(s), {unique_count} unique."
            ),
            fact_type=FactType.COUNT.value,
            status=FactStatus.VERIFIED.value,
            source_documents=source_docs,
            details={
                "total": len(all_values),
                "unique": unique_count,
                "values": all_values[:20],
            },
        ))
        return facts


class ThresholdVerifier(BaseVerifier):
    """Verifies whether a count or value meets a threshold."""

    def verify(self, requirement: DocumentRequirement) -> list[VerifiedFact]:
        facts = []
        all_values = []
        source_docs = []

        for doc in requirement.matched_documents:
            for rec in doc.records:
                for f in rec.fields:
                    parsed = parse_currency(f.value) or parse_number(f.value)
                    if parsed is not None:
                        all_values.append(parsed)
                    if doc.document_name not in source_docs:
                        source_docs.append(doc.document_name)

        facts.append(self._make_fact(
            statement=(
                f"Python found {len(all_values)} parseable numeric value(s) "
                f"across {len(source_docs)} document(s)."
            ),
            fact_type=FactType.COUNT.value,
            status=FactStatus.VERIFIED.value if all_values else FactStatus.NOT_FOUND.value,
            source_documents=source_docs,
        ))
        return facts


class ExistenceVerifier(BaseVerifier):
    """Checks for existence or non-existence of entities."""

    def verify(self, requirement: DocumentRequirement) -> list[VerifiedFact]:
        facts = []
        field_names = set()
        for doc in requirement.matched_documents:
            for rec in doc.records:
                for f in rec.fields:
                    field_names.add(f.name)

        for fn in field_names:
            found = False
            found_values = []
            source_docs = []

            for doc in requirement.matched_documents:
                for rec in doc.records:
                    for f in rec.fields:
                        if f.name == fn:
                            found = True
                            found_values.append(f.value)
                            if doc.document_name not in source_docs:
                                source_docs.append(doc.document_name)

            facts.append(self._make_fact(
                statement=(
                    f"Python {'found' if found else 'did NOT find'} "
                    f"field '{fn}' in matched documents."
                    + (f" Values: {', '.join(found_values[:10])}." if found_values else "")
                ),
                fact_type=FactType.EXISTENCE.value,
                status=FactStatus.VERIFIED.value if found else FactStatus.NOT_FOUND.value,
                source_fields=[fn],
                source_documents=source_docs,
                details={"field": fn, "found": found, "values": found_values},
            ))

        return facts


class DuplicateDetector(BaseVerifier):
    """Detects duplicate entities across and within documents."""

    def verify(self, requirement: DocumentRequirement) -> list[VerifiedFact]:
        facts = []
        from collections import Counter

        field_names = set()
        for doc in requirement.matched_documents:
            for rec in doc.records:
                for f in rec.fields:
                    field_names.add(f.name)

        for fn in field_names:
            all_values = []
            value_sources = {}

            for doc in requirement.matched_documents:
                for rec in doc.records:
                    for f in rec.fields:
                        if f.name == fn:
                            all_values.append(f.value)
                            value_sources.setdefault(f.value, []).append(doc.document_name)

            counts = Counter(all_values)
            dupes = {v: c for v, c in counts.items() if c > 1}

            if dupes:
                facts.append(self._make_fact(
                    statement=(
                        f"Python detected {len(dupes)} duplicate '{fn}' value(s): "
                        + "; ".join(
                            f"'{v}' appears {c} time(s)"
                            for v, c in dupes.items()
                        ) + "."
                    ),
                    fact_type=FactType.DUPLICATE.value,
                    status=FactStatus.VERIFIED.value,
                    source_fields=[fn],
                ))
            else:
                facts.append(self._make_fact(
                    statement=f"No duplicate '{fn}' values found.",
                    fact_type=FactType.DUPLICATE.value,
                    status=FactStatus.VERIFIED.value,
                ))

        return facts


class DateComparisonVerifier(BaseVerifier):
    """Performs date comparison and arithmetic."""

    def verify(self, requirement: DocumentRequirement) -> list[VerifiedFact]:
        facts = []
        dates = []
        for doc in requirement.matched_documents:
            for rec in doc.records:
                for f in rec.fields:
                    if f.datatype == "date":
                        dates.append((f.value, doc.document_name))

        parsed_dates = []
        for d_str, doc_name in dates:
            d = parse_date(d_str)
            if d:
                parsed_dates.append((d, d_str, doc_name))

        if len(parsed_dates) >= 2:
            d1_iso, d1_raw, doc1 = parsed_dates[0]
            d2_iso, d2_raw, doc2 = parsed_dates[-1]

            dt1 = datetime.strptime(d1_iso, "%Y-%m-%d")
            dt2 = datetime.strptime(d2_iso, "%Y-%m-%d")
            diff = abs((dt1 - dt2).days)

            if dt1 < dt2:
                relation = "before"
            elif dt1 > dt2:
                relation = "after"
            else:
                relation = "the same as"

            facts.append(self._make_fact(
                statement=(
                    f"Python verified: '{d1_raw}' ({d1_iso}) is {relation} "
                    f"'{d2_raw}' ({d2_iso}). Difference: {diff} day(s)."
                ),
                fact_type=FactType.DATE.value,
                status=FactStatus.VERIFIED.value,
                source_documents=[doc1, doc2],
                details={"date1": d1_iso, "date2": d2_iso, "days": diff},
            ))

        return facts