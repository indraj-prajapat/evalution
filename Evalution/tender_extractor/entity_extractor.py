"""
Entity extraction dispatcher.

Routes extraction requests to GPT-4o-mini (primary) or
regex-based extractors (fallback).

Updated: now also supports MERGE mode where both LLM and regex
results are combined (not just fallback).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .models import (
    DetectedDocument,
    ExtractedRecord,
    FieldSpec,
    RequirementDocument,
)
from .evidence import EvidenceTracker
from .config import ExtractionConfig

logger = logging.getLogger(__name__)


def extract_entities(
    doc: DetectedDocument,
    requirement: RequirementDocument,
    tracker: EvidenceTracker,
    config: Optional[ExtractionConfig] = None,
) -> List[ExtractedRecord]:
    """Extract entities from a single detected document.

    Uses LLM as primary, regex as fallback. When both produce results,
    they are merged (deduplicating by field name + value).

    Parameters
    ----------
    doc : the matched detected document
    requirement : the requirement being fulfilled
    tracker : evidence tracker for building fields
    config : extraction configuration

    Returns
    -------
    List of extracted records (grouped if applicable).
    """
    cfg = config or ExtractionConfig()

    fields_to_extract = requirement.required_fields
    group_by = requirement.group_by

    all_records: List[ExtractedRecord] = []
    llm_records: List[ExtractedRecord] = []
    regex_records: List[ExtractedRecord] = []

    # --- LLM extraction (primary) ---
    if cfg.use_llm_extraction:
        try:
            from .llm_extractor import extract_fields_with_llm

            logger.debug(
                "Using LLM extraction for doc '%s' (%d fields)",
                doc.document_name,
                len(fields_to_extract),
            )
            llm_records = extract_fields_with_llm(
                doc=doc,
                required_fields=fields_to_extract,
                group_by=group_by,
                tracker=tracker,
                max_retries=cfg.llm_max_retries,
            )

            if llm_records:
                logger.info(
                    "LLM extracted %d record(s) from '%s'",
                    len(llm_records),
                    doc.document_name,
                )
                all_records.extend(llm_records)

        except Exception as e:
            logger.warning("LLM extraction failed for '%s': %s", doc.document_name, e)

    # --- Regex extraction (always run as supplement) ---
    regex_records = _regex_extract(doc, requirement, tracker, group_by)

    if regex_records:
        logger.info(
            "Regex extracted %d record(s) from '%s'",
            len(regex_records),
            doc.document_name,
        )

    # --- Merge: combine LLM + regex, deduplicating ---
    if llm_records and regex_records:
        all_records = _merge_records(llm_records, regex_records, doc)
        logger.debug(
            "Merged %d LLM + %d regex records -> %d final records for '%s'",
            len(llm_records), len(regex_records), len(all_records), doc.document_name,
        )
    elif regex_records and not llm_records:
        all_records = regex_records

    return all_records


def _regex_extract(
    doc: DetectedDocument,
    requirement: RequirementDocument,
    tracker: EvidenceTracker,
    group_by: Optional[str],
) -> List[ExtractedRecord]:
    """Use regex-based extractors."""
    entity_type = requirement.entity_type or "Other"

    try:
        from .extractors import get_extractor

        extractor = get_extractor(entity_type, tracker)
        records = extractor.extract(
            doc=doc,
            required_fields=requirement.required_fields,
            group_by=group_by,
        )
        return records

    except Exception as e:
        logger.error("Regex extraction failed for '%s': %s", doc.document_name, e)
        return []


def _merge_records(
    llm_records: List[ExtractedRecord],
    regex_records: List[ExtractedRecord],
    doc: DetectedDocument,
) -> List[ExtractedRecord]:
    """Merge LLM and regex records, deduplicating by field name.

    Strategy:
    - LLM records are the base (higher confidence from LLM extraction)
    - Regex records supplement: add any fields not already found by LLM
    - If regex found a record with fields that LLM missed, add those fields
    - Never remove LLM fields in favor of regex (LLM is more accurate)
    """
    if not llm_records:
        return regex_records
    if not regex_records:
        return llm_records

    # Build a set of (entity_id, field_name) from LLM results
    llm_field_keys: set = set()
    for rec in llm_records:
        for field in rec.fields:
            llm_field_keys.add((rec.entity_id, field.name))

    # Check if regex found any new fields not in LLM results
    new_fields_by_entity: dict = {}
    for rec in regex_records:
        for field in rec.fields:
            # Check across all LLM records for this field name
            found_in_llm = any(
                (llm_rec.entity_id, field.name) in llm_field_keys
                for llm_rec in llm_records
            )
            if not found_in_llm:
                entity_key = rec.entity_id
                if entity_key not in new_fields_by_entity:
                    new_fields_by_entity[entity_key] = []
                new_fields_by_entity[entity_key].append(field)

    # If regex found no new fields, just return LLM records
    if not new_fields_by_entity:
        return llm_records

    # Add new fields from regex to existing LLM records
    for rec in llm_records:
        if rec.entity_id in new_fields_by_entity:
            new_fields = new_fields_by_entity[rec.entity_id]
            # Deduplicate: don't add if same field name already exists
            existing_names = {f.name for f in rec.fields}
            for nf in new_fields:
                if nf.name not in existing_names:
                    rec.fields.append(nf)
                    existing_names.add(nf.name)

    # For any regex records with new fields that don't match any LLM entity,
    # add them as separate records
    for rec in regex_records:
        if rec.entity_id in new_fields_by_entity:
            has_new = any(
                nf.name not in {f.name for f in lr.fields}
                for lr in llm_records
                for nf in new_fields_by_entity[rec.entity_id]
            )
            if has_new and not any(
                lr.entity_id == rec.entity_id for lr in llm_records
            ):
                # This is a new entity not found by LLM at all
                llm_records.append(rec)

    return llm_records