"""Pipeline evaluator module.

Handles evaluation of planned key points through extraction, verification, 
and the verification engine.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from ..client import DEFAULT_OPENROUTER_BASE_URL, get_llm_api_key, get_llm_model
from ..tender_extractor import extract_from_bid
from ..extraction_value_verifier import verify_extracted_values
from ..verification_engine.engine import run_verification as run_verification_engine


def evaluate_planned_key_points(
    planning_results: list[Any],
    company_data: dict[str, Any],
    company_name: str,
    max_workers: int,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Evaluate all planned key points in parallel or sequentially.
    
    Args:
        planning_results: List of planning results from the planner.
        company_data: Company JSON data.
        company_name: Human-readable company name.
        max_workers: Maximum number of parallel workers.
        api_key: OpenRouter API key. Defaults to env var.
        model: Model name. Defaults to env var.
        base_url: Base URL for API. Defaults to env var.
    
    Returns:
        List of evaluation results, one per key point.
    """
    indexed_results = list(enumerate(planning_results))
    evaluation_results: list[dict[str, Any]] = [
        _empty_evaluation_result(idx, result)
        for idx, result in indexed_results
    ]
    
    succeeded = [
        (idx, result)
        for idx, result in indexed_results
        if result is not None and result.success and result.plan is not None
    ]
    
    if max_workers <= 1:
        for idx, result in succeeded:
            evaluation_results[idx] = _evaluate_one_key_point(
                idx=idx,
                planning_result=result,
                company_data=company_data,
                company_name=company_name,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
        return evaluation_results
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _evaluate_one_key_point,
                idx=idx,
                planning_result=result,
                company_data=company_data,
                company_name=company_name,
                api_key=api_key,
                model=model,
                base_url=base_url,
            ): idx
            for idx, result in succeeded
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            evaluation_results[idx] = future.result()
    
    return evaluation_results


def _evaluate_one_key_point(
    idx: int,
    planning_result: Any,
    company_data: dict[str, Any],
    company_name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Evaluate a single key point through extraction and verification.
    
    Args:
        idx: Index of the key point.
        planning_result: Planning result with plan data.
        company_data: Company JSON data.
        company_name: Human-readable company name.
        api_key: OpenRouter API key.
        model: Model name.
        base_url: Base URL for API.
    
    Returns:
        Dictionary with extraction, verification, and error info.
    """
    plan_data = _to_json_dict(planning_result.plan)
    criterion_id = planning_result.criterion_id
    criterion_text = plan_data.get("criterion", "")
    
    try:
        # Step 1: Extract values from documents (Module 2 / tender_extractor)
        extraction = extract_from_bid(
            company_json=company_data,
            tender_plan=plan_data,
        )
        
        # Step 2: AI Value Verification Agent
        verified_extraction = verify_extracted_values(
            extraction=extraction,
            criterion_text=criterion_text,
            api_key=api_key or get_llm_api_key(),
            model=model or get_llm_model(),
            base_url=base_url or os.environ.get(
                "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
            ),
        )
        
        # Step 3: Run verification engine with VERIFIED values only
        verification_report = run_verification_engine(
            module2_input=verified_extraction,
            company_json_input=company_data,
            company_name=company_name,
        )
        verification = verification_report.to_dict()
        
        return {
            "index": idx,
            "criterion_id": criterion_id,
            "extraction": verified_extraction,
            "verification": verification,
            "error": None,
        }
    except Exception as exc:
        return {
            "index": idx,
            "criterion_id": criterion_id,
            "extraction": None,
            "verification": None,
            "error": str(exc),
        }


def _empty_evaluation_result(idx: int, planning_result: Any) -> dict[str, Any]:
    """Create an empty evaluation result placeholder."""
    return {
        "index": idx,
        "criterion_id": getattr(planning_result, "criterion_id", f"CRIT{idx + 1:03d}"),
        "extraction": None,
        "verification": None,
        "error": None,
    }


def _to_json_dict(obj: Any) -> Any:
    """Recursively convert dataclasses / Pydantic models to plain JSON values."""
    from dataclasses import asdict, is_dataclass
    
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if is_dataclass(obj):
        return {k: _to_json_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_json_dict(v) for k, v in obj.items()}
    return obj
