"""
Extraction Value Verifier — AI Agent for verifying extracted values before
they enter the Python verification engine (Module 3).

PROBLEM IT SOLVES:
  When the extraction module (Module 2 / tender_extractor) extracts values
  from documents, the LLM sometimes tags WRONG values too — values from a
  different year, a different entity, a different context, or values that
  are simply not what the tender requirement is asking for.

  Previously, ALL extracted values (correct + wrong) were fed into the
  verification engine's NumericVerifier, which computed statistics (sum,
  average, min, max) across ALL of them — producing wrong results.

HOW IT WORKS:
  For each field that has multiple extracted values, this agent:
    1. Collects every value with its full context: snippet, page number,
       document name, document summary, raw_value, confidence, tag, entity_id.
    2. Builds a structured prompt for the LLM containing:
       - The tender criterion text (what the requirement actually asks)
       - The requirement document description
       - A document summary (from the company JSON / extraction metadata)
       - ALL candidate values with their surrounding context
    3. Asks the LLM to determine which value(s) are the CORRECT input
       for this specific field in the context of this specific requirement.
    4. The LLM returns:
       - "correct" values (with reasoning)
       - "wrong" values (with reasoning — why they are wrong)
       - "none" if NO value is correct for this field
    5. The agent keeps ONLY correct values and removes wrong ones.
    6. If the LLM says no correct value exists, the field is marked as
       "verified_missing" — it will not be passed to the verification
       engine's numeric verifier as a data point.

CRITICAL DESIGN PRINCIPLES:
  - NEVER guesses or hallucinates a value not present in the extraction.
  - NEVER fabricates a value.
  - If the LLM is unsure, the value is REMOVED (conservative: missing
    data → REVIEW verdict, which is safer than wrong data → wrong PASS/FAIL).
  - Uses the SAME OpenRouterClient and model defined in Evalution.client.
  - Works field-by-field, so each field's verification is independent.
  - Handles both repeatable fields (multiple values expected) and
    non-repeatable fields (only one value expected).
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from typing import Any, Optional

from .client import (
    DEFAULT_OPENROUTER_BASE_URL,
    OpenRouterClient,
    get_llm_api_key,
    get_llm_model,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise extraction value verifier for a tender evaluation system. \
Your job is to examine extracted values from documents and determine which \
ones are CORRECT for a specific tender requirement field, and which ones are \
WRONG (mismatches, different entity, different year, different context, etc.).

CRITICAL RULES:
1. NEVER guess or hallucinate a value. You can ONLY select from the values \
provided to you — never invent a new one.
2. NEVER select a value if you are not confident it is correct. It is always \
better to say "none correct" than to pick a wrong value.
3. A value is WRONG if it belongs to a different entity (company), a different \
financial year than required, a different document context, or is clearly a \
mis-tagged number (e.g., a page number, a serial number, or a quantity from a \
different table that the LLM mistakenly tagged).
4. A value is CORRECT only if it directly answers what the requirement is \
asking for, from the correct entity and the correct time period.
5. If multiple values seem correct (e.g., turnover for multiple years), ALL \
correct values should be marked — do not arbitrarily pick just one.
6. Consider the surrounding text snippet carefully. A value like "50,00,000" \
could be turnover, share capital, or EMD depending on context.
7. If no value is correct, return an empty "correct_values" list. This means \
the field has no verified input — the system will treat it as missing.
"""

_FIELD_VERIFICATION_PROMPT = """\
## Tender Requirement (Criterion)
{criterion}

## Required Document / Requirement Type
{requirement_document}

## Entity Type
{entity_type}

## Document Summary
{document_summary}

## Field Being Verified
- Field name: "{field_name}"
- Expected data type: {datatype}
- Description: {field_description}
- Is repeatable: {repeatable}

## All Extracted Values for This Field (from document extraction)
{values_block}

## Task
For the field "{field_name}" above, determine which of the extracted values \
are the CORRECT value(s) that directly answer what the tender requirement is \
asking for.

For each value, state whether it is "correct" or "wrong" and give a brief \
reason. Then list the correct value(s) in the "correct_values" array.

If NO value is correct, return an empty "correct_values" array.

## Response Format (JSON only, no markdown)
{{
  "field_name": "{field_name}",
  "analysis": [
    {{
      "value": "<the exact value string>",
      "verdict": "correct" | "wrong",
      "reason": "<brief explanation>"
    }}
  ],
  "correct_values": ["<exact correct value strings>"],
  "no_correct_found": false,
  "confidence_note": "<optional note about confidence level>"
}}
"""


# ---------------------------------------------------------------------------
# Verifier Agent
# ---------------------------------------------------------------------------

class ExtractionValueVerifier:
    """
    AI agent that verifies extracted field values using LLM before they
    enter the Python verification engine.

    Uses the same OpenRouter client configured for the rest of the pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 120,
    ):
        self.client = OpenRouterClient(
            api_key=api_key,
            model=model,
            base_url=base_url or os.environ.get(
                "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
            ),
            timeout=timeout,
        )
        self.model = self.client.model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_extraction(
        self,
        extraction: dict[str, Any],
        criterion_text: str = "",
        document_summary: str = "",
    ) -> dict[str, Any]:
        """
        Verify ALL extracted values in an extraction result.

        This is the main entry point. It iterates over every requirement
        document, every matched document, every record, and every field.
        For fields with multiple values, it calls the LLM to determine
        which values are correct.

        Parameters
        ----------
        extraction : dict
            The raw extraction result from ``extract_from_bid()``.
        criterion_text : str
            The tender criterion text (what the requirement asks).
        document_summary : str
            A summary of the company's documents (optional, helps context).

        Returns
        -------
        dict
            A NEW extraction dict with only the verified correct values.
            Wrong values are removed. The structure is otherwise identical.
        """
        if not extraction or not extraction.get("documents"):
            log.debug("extraction_value_verifier: empty extraction, returning as-is")
            return extraction

        result = deepcopy(extraction)
        total_fields_verified = 0
        total_values_removed = 0
        total_fields_with_changes = 0

        for req_doc in result.get("documents", []):
            requirement_document = req_doc.get("requirement_document", "")
            entity_type = req_doc.get("entity_type", "")
            criterion = criterion_text or extraction.get("criterion", "")

            for matched_doc in req_doc.get("matched_documents", []):
                doc_name = matched_doc.get("document_name", "")
                doc_summary = matched_doc.get("summary", "")
                pages = matched_doc.get("pages", [])

                for record in matched_doc.get("records", []):
                    entity_id = record.get("entity_id", "")
                    group = record.get("group", "")
                    group_value = record.get("group_value", "")

                    # Collect all field values to identify fields with
                    # multiple values that need verification.
                    field_values_map: dict[str, list[dict]] = {}
                    for field in record.get("fields", []):
                        fname = field.get("name", "")
                        if fname not in field_values_map:
                            field_values_map[fname] = []
                        field_values_map[fname].append(field)

                    # Verify fields that have multiple distinct values.
                    # Single-value fields are kept as-is (no ambiguity).
                    for fname, fields in field_values_map.items():
                        # Get distinct values
                        distinct_values = list({
                            str(f.get("value", "")).strip()
                            for f in fields
                            if str(f.get("value", "")).strip()
                        })

                        if len(distinct_values) <= 1:
                            # Only one distinct value — no ambiguity to resolve.
                            continue

                        total_fields_verified += 1
                        log.info(
                            "extraction_value_verifier: field '%s' has %d "
                            "distinct values, verifying with LLM",
                            fname, len(distinct_values),
                        )

                        # Build the values block for the prompt
                        values_block = self._build_values_block(fields, doc_name, pages)

                        # Get field description if available
                        field_description = fields[0].get("description", "")
                        datatype = fields[0].get("datatype", "string")
                        repeatable = fields[0].get("repeatable", False)

                        # Call LLM to verify
                        verification = self._verify_field_with_llm(
                            criterion=criterion,
                            requirement_document=requirement_document,
                            entity_type=entity_type,
                            field_name=fname,
                            datatype=datatype,
                            field_description=field_description,
                            repeatable=repeatable,
                            values_block=values_block,
                            document_summary=doc_summary or document_summary,
                        )

                        if verification is None:
                            log.warning(
                                "extraction_value_verifier: LLM verification "
                                "failed for field '%s', keeping all values",
                                fname,
                            )
                            continue

                        correct_values = set(
                            str(v).strip()
                            for v in verification.get("correct_values", [])
                        )
                        no_correct_found = verification.get("no_correct_found", False)

                        # Apply verification result: remove wrong values
                        original_count = len(fields)
                        verified_fields = []

                        for field in fields:
                            val = str(field.get("value", "")).strip()
                            if no_correct_found:
                                # LLM said no value is correct — remove all
                                total_values_removed += 1
                                log.debug(
                                    "extraction_value_verifier: removed value "
                                    "'%s' from field '%s' (no correct found)",
                                    val[:50], fname,
                                )
                                continue
                            if val in correct_values:
                                # Keep correct values
                                field["value_verified"] = True
                                field["value_verification_reason"] = "correct"
                                verified_fields.append(field)
                            else:
                                # Remove wrong values
                                total_values_removed += 1
                                log.debug(
                                    "extraction_value_verifier: removed wrong "
                                    "value '%s' from field '%s'",
                                    val[:50], fname,
                                )

                        changes = original_count - len(verified_fields)
                        if changes > 0:
                            total_fields_with_changes += 1

                        # Update the record's fields
                        record["fields"] = [
                            f for f in record.get("fields", [])
                            if any(
                                f.get("value") == vf.get("value")
                                and f.get("page") == vf.get("page")
                                for vf in verified_fields
                            )
                        ] if not no_correct_found else []

        log.info(
            "extraction_value_verifier: verified %d multi-value fields, "
            "removed %d wrong values across %d fields",
            total_fields_verified, total_values_removed, total_fields_with_changes,
        )

        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_values_block(
        self,
        fields: list[dict],
        doc_name: str,
        pages: list[int],
    ) -> str:
        """Build a human-readable block of all extracted values for a field."""
        lines = []
        for i, field in enumerate(fields, 1):
            value = field.get("value", "")
            raw_value = field.get("raw_value", "")
            snippet = field.get("snippet", "")
            page = field.get("page", 0)
            confidence = field.get("confidence", 0.0)

            lines.append(f"### Value Candidate #{i}")
            lines.append(f"- Value: {value}")
            if raw_value and raw_value != str(value):
                lines.append(f"- Raw value: {raw_value}")
            lines.append(f"- Page: {page}")
            lines.append(f"- Confidence: {confidence:.2f}")
            if snippet:
                # Truncate very long snippets
                snippet_text = snippet[:500] + "..." if len(snippet) > 500 else snippet
                lines.append(f"- Surrounding text snippet: \"{snippet_text}\"")
            lines.append("")

        return "\n".join(lines)

    def _verify_field_with_llm(
        self,
        criterion: str,
        requirement_document: str,
        entity_type: str,
        field_name: str,
        datatype: str,
        field_description: str,
        repeatable: bool,
        values_block: str,
        document_summary: str,
    ) -> Optional[dict[str, Any]]:
        """
        Call the LLM to verify which values are correct for a field.

        Returns the parsed JSON response, or None on failure.
        """
        prompt = _FIELD_VERIFICATION_PROMPT.format(
            criterion=criterion or "Not provided",
            requirement_document=requirement_document or "Not provided",
            entity_type=entity_type or "Not specified",
            field_name=field_name,
            datatype=datatype,
            field_description=field_description or "Not provided",
            repeatable="Yes" if repeatable else "No",
            values_block=values_block,
            document_summary=document_summary or "Not available",
        )

        try:
            response = self.client.create_chat_completion(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip() if response.choices else ""
            if not content:
                log.warning("extraction_value_verifier: empty LLM response")
                return None

            # Parse JSON from response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            result = json.loads(content)

            # Basic validation
            if not isinstance(result, dict):
                log.warning("extraction_value_verifier: LLM response is not a dict")
                return None

            if "correct_values" not in result:
                log.warning(
                    "extraction_value_verifier: LLM response missing 'correct_values'"
                )
                return None

            log.debug(
                "extraction_value_verifier: field '%s' → %d correct values, "
                "no_correct_found=%s",
                field_name,
                len(result.get("correct_values", [])),
                result.get("no_correct_found", False),
            )

            return result

        except json.JSONDecodeError as e:
            log.error(
                "extraction_value_verifier: failed to parse LLM JSON response: %s",
                e,
            )
            return None
        except Exception as e:
            log.error(
                "extraction_value_verifier: LLM call failed for field '%s': %s",
                field_name, e,
            )
            return None


# ---------------------------------------------------------------------------
# Convenience function (matches the pattern of extract_from_bid, run_verification)
# ---------------------------------------------------------------------------

def verify_extracted_values(
    extraction: dict[str, Any],
    criterion_text: str = "",
    document_summary: str = "",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Verify extracted values using the AI agent.

    This is the convenience function to call from the pipeline.

    Parameters
    ----------
    extraction : dict
        The raw extraction result from ``extract_from_bid()``.
    criterion_text : str
        The tender criterion / key point text being evaluated.
    document_summary : str
        Optional overall document summary for additional context.
    api_key : str, optional
        OpenRouter API key. Uses env var if not provided.
    model : str, optional
        LLM model name. Uses configured default if not provided.
    base_url : str, optional
        OpenRouter base URL. Uses configured default if not provided.

    Returns
    -------
    dict
        Verified extraction with wrong values removed.
    """
    verifier = ExtractionValueVerifier(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    return verifier.verify_extraction(
        extraction=extraction,
        criterion_text=criterion_text,
        document_summary=document_summary,
    )