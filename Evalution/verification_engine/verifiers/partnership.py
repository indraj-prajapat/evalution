"""
Partnership Verifier - verifies partnership-related criteria.

Handles:
  - Partner existence and count
  - Partner name extraction and deduplication
  - Partner qualification matching
  - Partnership agreement existence
  - Partner name cross-validation across documents
  - Partner share/profit ratio verification
"""

from __future__ import annotations

import re
from collections import Counter
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
)


class PartnershipVerifier(BaseVerifier):
    """Verifies partnership-related tender requirements."""

    def verify(self, requirement: DocumentRequirement) -> list[VerifiedFact]:
        """
        Verify partnership agreement / partner-related requirements.

        Checks:
        1. Whether partnership agreement documents were found
        2. Number of unique partners identified
        3. Cross-document partner name consistency
        4. Duplicate partner detection
        5. Partner qualification extraction
        """
        facts: list[VerifiedFact] = []
        req_name = requirement.requirement_document
        matched_docs = requirement.matched_documents

        # --- Collect all partner names from all documents ---
        doc_partner_map: dict[str, list[str]] = {}  # doc_name -> [partner_names]
        all_partner_names: list[str] = []
        partner_fields_found: set[str] = set()

        for doc in matched_docs:
            doc_names = []
            for rec in doc.records:
                names = rec.get_all_fields("partner_names")
                for nf in names:
                    partner_fields_found.add("partner_names")
                    # Clean the name
                    name = nf.value.strip()
                    # Remove "The said " prefix
                    name = re.sub(r"^The said\s+", "", name, flags=re.IGNORECASE)
                    # Remove trailing punctuation artifacts
                    name = re.sub(r"[.\s]+$", "", name)
                    if name and len(name) > 1:
                        doc_names.append(name)
                        all_partner_names.append(name)
            if doc_names:
                doc_partner_map[doc.document_name] = doc_names

        # --- FACT: Document count for partnership requirement ---
        found_docs = [d for d in matched_docs if d.is_found]
        facts.append(self._make_fact(
            statement=(
                f"Python found {len(matched_docs)} document(s) matched against "
                f"partnership requirement: {len(found_docs)} with FOUND status, "
                f"{len(matched_docs) - len(found_docs)} with other statuses."
            ),
            fact_type=FactType.COUNT.value,
            status=FactStatus.VERIFIED.value,
            source_documents=[d.document_name for d in matched_docs],
            details={
                "total": len(matched_docs),
                "found": len(found_docs),
            },
        ))

        # --- FACT: Unique partner count ---
        unique_names = list(set(all_partner_names))
        unique_count = len(unique_names)

        facts.append(self._make_fact(
            statement=(
                f"Python extracted {len(all_partner_names)} total partner name(s) across "
                f"{len(doc_partner_map)} document(s). After deduplication, "
                f"{unique_count} unique partner(s) identified: "
                f"{', '.join(sorted(unique_names)) if unique_names else 'NONE'}."
            ),
            fact_type=FactType.COUNT.value,
            status=FactStatus.VERIFIED.value,
            source_fields=["partner_names"],
            source_documents=list(doc_partner_map.keys()),
            details={
                "total_extracted": len(all_partner_names),
                "unique_count": unique_count,
                "unique_names": sorted(unique_names),
                "per_document": doc_partner_map,
            },
        ))

        # --- FACT: Cross-document consistency ---
        if len(doc_partner_map) > 1:
            name_sets = [set(names) for names in doc_partner_map.values()]
            # Check if all documents agree on the same set of partners
            reference_set = name_sets[0]
            all_same = all(s == reference_set for s in name_sets)

            # Find names that appear in some but not all documents
            name_occurrence = Counter(all_partner_names)
            inconsistent_names = [
                name for name, count in name_occurrence.items()
                if count < len(doc_partner_map)
            ]

            if all_same:
                consistency_status = FactStatus.VERIFIED.value
                consistency_msg = (
                    f"Python verified that all {len(doc_partner_map)} documents agree on "
                    f"the same set of {unique_count} partner(s)."
                )
            else:
                consistency_status = FactStatus.CONFLICTING.value
                consistency_msg = (
                    f"Python detected inconsistency: partners "
                    f"{', '.join(inconsistent_names)} appear in some documents but not all. "
                    f"Document-by-document breakdown: "
                    + "; ".join(
                        f"{doc}: {', '.join(names)}"
                        for doc, names in doc_partner_map.items()
                    )
                )

            facts.append(self._make_fact(
                statement=consistency_msg,
                fact_type=FactType.ENTITY_MATCH.value,
                status=consistency_status,
                source_fields=["partner_names"],
                source_documents=list(doc_partner_map.keys()),
                details={
                    "consistent": all_same,
                    "inconsistent_names": inconsistent_names,
                    "per_document": doc_partner_map,
                    "name_occurrence": dict(name_occurrence),
                },
            ))

        # --- FACT: Duplicate detection within documents ---
        for doc_name, names in doc_partner_map.items():
            name_counts = Counter(names)
            dupes = {n: c for n, c in name_counts.items() if c > 1}
            if dupes:
                facts.append(self._make_fact(
                    statement=(
                        f"Python detected duplicate partner name(s) in '{doc_name}': "
                        f"{', '.join(f'{n} (appears {c} times)' for n, c in dupes.items())}."
                    ),
                    fact_type=FactType.DUPLICATE.value,
                    status=FactStatus.VERIFIED.value,
                    source_documents=[doc_name],
                    details={"duplicates": dict(dupes)},
                ))

        # --- FACT: Partner qualification check ---
        for doc in matched_docs:
            for rec in doc.records:
                # Look for qualification-related fields
                for f in rec.fields:
                    if any(q in f.name.lower() for q in
                           ["qualification", "membership_type", "certified", "fca", "aca"]):
                        facts.append(self._make_fact(
                            statement=(
                                f"Python found qualification field '{f.name}' = '{f.value}' "
                                f"in document '{doc.document_name}' (page {f.page})."
                            ),
                            fact_type=FactType.EXISTENCE.value,
                            status=FactStatus.VERIFIED.value,
                            source_fields=[f.name],
                            source_documents=[doc.document_name],
                            details={"value": f.value, "page": f.page},
                        ))

        # --- FACT: Partnership date extraction ---
        for doc in matched_docs:
            for rec in doc.records:
                date_field = rec.get_field("partnership_date")
                if date_field:
                    facts.append(self._make_fact(
                        statement=(
                            f"Python extracted partnership date '{date_field.value}' "
                            f"from document '{doc.document_name}' (page {date_field.page})."
                        ),
                        fact_type=FactType.DATE.value,
                        status=FactStatus.VERIFIED.value,
                        source_fields=["partnership_date"],
                        source_documents=[doc.document_name],
                        details={"date": date_field.value, "page": date_field.page},
                    ))

        return facts