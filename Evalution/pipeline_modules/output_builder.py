"""Pipeline output builder module.

Handles building the final output by appending evaluations to key points.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional


def append_evaluations_to_key_points(
    tender_output: dict[str, Any],
    planning_results: list[Any],
    evaluation_results: list[dict[str, Any]],
    validation_errors: Optional[dict[int, str]] = None,
) -> dict[str, Any]:
    """
    Append evaluation results to each key point in the tender output.
    
    Args:
        tender_output: Original tender output dictionary.
        planning_results: List of planning results.
        evaluation_results: List of evaluation results.
        validation_errors: Dictionary of validation errors by index.
    
    Returns:
        Modified tender output with verdicts and summaries appended.
    """
    final_output = deepcopy(tender_output)
    key_points = final_output.get("key_points", [])
    validation_errors = validation_errors or {}
    
    for idx, key_point in enumerate(key_points):
        if not isinstance(key_point, dict):
            # Can't attach fields to a non-dict entry -- replace with object.
            key_point = {"raw_value": key_point}
            key_points[idx] = key_point
        
        if idx in validation_errors:
            # Malformed input -- quarantined, not evaluated.
            key_point["verdict"] = "REVIEW"
            key_point["verdict_source"] = "unavailable"
            key_point["summary"] = (
                f"Malformed key point, not evaluated: {validation_errors[idx]}"
            )
            key_point["matched_docuements"] = []
            continue
        
        planning_result = planning_results[idx] if idx < len(planning_results) else None
        evaluation_result = (
            evaluation_results[idx] if idx < len(evaluation_results) else None
        )
        
        if evaluation_result and evaluation_result.get("verification"):
            verification = evaluation_result["verification"]
            extraction = evaluation_result.get("extraction") or {}
            key_point["verdict"] = verification.get("verdict", "REVIEW")
            key_point["verdict_source"] = verification.get("verdict_source", "python_authoritative")
            key_point["summary"] = verification.get("summary", "")
            key_point["matched_docuements"] = _collect_matched_documents(extraction)
            continue
        
        key_point["verdict"] = "REVIEW"
        key_point["verdict_source"] = "unavailable"
        key_point["summary"] = _failure_summary(planning_result, evaluation_result)
        key_point["matched_docuements"] = []
    
    return final_output


def _collect_matched_documents(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract matched documents from extraction result."""
    matched_documents: list[dict[str, Any]] = []
    for requirement in extraction.get("documents", []):
        for document in requirement.get("matched_documents", []):
            item = deepcopy(document)
            item["requirement_document"] = requirement.get("requirement_document", "")
            item["mode"] = requirement.get("mode", "")
            item["entity_type"] = requirement.get("entity_type", "")
            matched_documents.append(item)
    return matched_documents


def _failure_summary(
    planning_result: Any,
    evaluation_result: Optional[dict[str, Any]],
) -> str:
    """Build a summary message for failed evaluations."""
    if planning_result is not None and not planning_result.success:
        return f"Planning failed: {planning_result.error}"
    if evaluation_result and evaluation_result.get("error"):
        return f"Evaluation failed: {evaluation_result['error']}"
    return "Evaluation could not be completed for this key point."
