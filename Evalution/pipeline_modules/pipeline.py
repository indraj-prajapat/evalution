"""Modular Evaluation Pipeline with Debug Support.

This is the main entry point for the evaluation pipeline, refactored into
small, modular components with comprehensive debug logging capabilities.

The pipeline accepts 4 inputs as usual:
    1. tender_output_json: Tender output dict or path
    2. company_json_path: Path to company JSON file
    3. company_name: Human-readable company name
    4. output_path: File path for final JSON report

Plus an optional debug flag to enable detailed logging.

All API keys and model names are read from .env file only - no hardcoded values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from ..client import DEFAULT_OPENROUTER_BASE_URL, get_llm_api_key, get_llm_model
from ..planner.logging_utils import configure_logging, get_logger
from ..debug_utils import DebugLogger, set_debug_logger, log_step
from .input_loader import load_json_input, validate_key_points
from .config_builder import (
    build_planner_config,
    build_planner_tender,
    resolve_worker_count,
)
from .planner_module import plan_key_points
from .evaluator_module import evaluate_planned_key_points
from .output_builder import append_evaluations_to_key_points
from .utils import to_json_dict, write_json, print_planner_outcomes

# Load environment variables from .env
load_dotenv()

# Default cache paths
PLANNER_CACHE: str = "tender_plan_results.json"
VERDICT_CACHE: str = "final_verdict.json"
EVALUATION_CACHE: str = "evaluation_results.json"


def run_pipeline(
    tender_output_json: dict[str, Any] | str | Path,
    company_json_path: str,
    company_name: str,
    output_path: str,
    max_workers: Optional[int] = None,
    debug: bool = False,
    debug_save_dir: Optional[str] = None,
    # Configuration overrides (all optional, read from .env if not provided)
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    End-to-end tender evaluation pipeline for all extracted key points.
    
    This is the main pipeline function with modular structure and debug support.
    
    Parameters
    ----------
    tender_output_json : dict | str | Path
        Tender output dict or path. Its key_points[*].point values are 
        evaluated as criteria.
    company_json_path : str
        Path to the company JSON file.
    company_name : str
        Human-readable company name used in the report.
    output_path : str
        File path where the final tender-shaped JSON report is written.
    max_workers : int, optional
        Optional parallel worker count. Defaults to EVALUATION_MAX_WORKERS 
        env var or 4.
    debug : bool, default False
        Enable debug mode to log all inputs, outputs, and intermediate steps.
    debug_save_dir : str, optional
        Directory to save debug logs. Defaults to ./debug_logs.
    api_key : str, optional
        OpenRouter API key. Reads from OPENROUTER_API_KEY env var if not provided.
    model : str, optional
        Model name. Reads from OPENROUTER_MODEL env var if not provided.
    base_url : str, optional
        API base URL. Reads from OPENROUTER_BASE_URL env var if not provided.
    
    Returns
    -------
    dict
        The original tender output structure with verdict, summary, and
        matched_documents appended to each key point.
    
    Example
    -------
    >>> result = run_pipeline(
    ...     tender_output_json="tender_output.json",
    ...     company_json_path="company.json",
    ...     company_name="My Company",
    ...     output_path="evaluation_result.json",
    ...     debug=True,  # Enable debug logging
    ... )
    """
    # Initialize logging
    configure_logging(level="DEBUG" if debug else "INFO")
    log = get_logger(__name__)
    
    # Initialize debug logger
    debug_logger = DebugLogger(enabled=debug, save_dir=debug_save_dir)
    set_debug_logger(debug_logger)
    
    log.info(
        "pipeline_start",
        debug_enabled=debug,
        tender_input=str(tender_output_json)[:100],
        company_path=company_json_path,
        company_name=company_name,
        output_path=output_path,
    )
    
    try:
        # ================================================================
        # Step 1: Load and validate tender output
        # ================================================================
        log_step("load_tender_output", {"input": tender_output_json}, None)
        tender_output = load_json_input(tender_output_json, "tender_output_json")
        log_step("load_tender_output", tender_output_json, tender_output)
        
        key_points = tender_output.get("key_points", [])
        if not isinstance(key_points, list):
            raise ValueError("Expected tender_output_json['key_points'] to be a list")
        
        # Validate key points
        validation_errors = validate_key_points(key_points)
        if validation_errors:
            for idx, reason in validation_errors.items():
                log.warning("key_point_validation_failed", index=idx, reason=reason)
            log.warning(
                "key_point_validation_summary",
                malformed_count=len(validation_errors),
                total_count=len(key_points),
            )
        
        log_step(
            "validate_key_points",
            {"key_points_count": len(key_points)},
            {"validation_errors": validation_errors},
        )
        
        # ================================================================
        # Step 2: Resolve worker count and load company data
        # ================================================================
        worker_count = resolve_worker_count(max_workers, len(key_points))
        log.info(
            "pipeline_configuration",
            key_point_count=len(key_points),
            max_workers=worker_count,
            debug_enabled=debug,
        )
        
        log_step("load_company_data", {"path": company_json_path}, None)
        with open(company_json_path, "r", encoding="utf-8") as f:
            company_data = json.load(f)
        log_step("load_company_data", company_json_path, company_data)
        
        # ================================================================
        # Step 3: Build planner configuration and tender structure
        # ================================================================
        log_step("build_planner_config", {"api_key_provided": api_key is not None}, None)
        config = build_planner_config(
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        log_step("build_planner_config", {}, config)
        
        log_step("build_planner_tender", {"tender_output_keys": list(tender_output.keys())}, None)
        planner_tender = build_planner_tender(tender_output)
        log_step("build_planner_tender", tender_output, planner_tender)
        
        # ================================================================
        # Step 4: Plan key points
        # ================================================================
        log_step(
            "plan_key_points",
            {
                "criteria_count": len(planner_tender.get("key_points", [])),
                "skip_indices": list(validation_errors.keys()),
            },
            None,
        )
        planning_results = plan_key_points(
            planner_tender=planner_tender,
            config=config,
            max_workers=worker_count,
            skip_indices=set(validation_errors.keys()),
        )
        serialisable_results = to_json_dict(planning_results)
        log_step("plan_key_points", planner_tender, serialisable_results)
        
        # Save planner results cache
        _write_cache(PLANNER_CACHE, serialisable_results)
        log.info("planner_results_saved", path=PLANNER_CACHE)
        
        # Print planner outcomes
        print_planner_outcomes(planning_results)
        
        # ================================================================
        # Step 5: Evaluate planned key points
        # ================================================================
        log_step(
            "evaluate_key_points",
            {
                "planning_results_count": len(planning_results),
                "company_name": company_name,
            },
            None,
        )
        evaluation_results = evaluate_planned_key_points(
            planning_results=planning_results,
            company_data=company_data,
            company_name=company_name,
            max_workers=worker_count,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        log_step("evaluate_key_points", planning_results, evaluation_results)
        
        # Save caches
        extraction_cache = [
            item["extraction"]
            for item in evaluation_results
            if item.get("extraction") is not None
        ]
        verification_cache = [
            item["verification"]
            for item in evaluation_results
            if item.get("verification") is not None
        ]
        _write_cache(VERDICT_CACHE, extraction_cache)
        _write_cache(EVALUATION_CACHE, verification_cache)
        log.info("verdict_saved", path=VERDICT_CACHE)
        log.info("evaluation_results_saved", path=EVALUATION_CACHE)
        
        # ================================================================
        # Step 6: Build final output
        # ================================================================
        log_step(
            "build_final_output",
            {
                "tender_output_keys": list(tender_output.keys()),
                "planning_results_count": len(planning_results),
                "evaluation_results_count": len(evaluation_results),
            },
            None,
        )
        final_tender_output = append_evaluations_to_key_points(
            tender_output=tender_output,
            planning_results=planning_results,
            evaluation_results=evaluation_results,
            validation_errors=validation_errors,
        )
        log_step("build_final_output", tender_output, final_tender_output)
        
        # Write final output
        write_json(output_path, final_tender_output)
        
        log.info("pipeline_complete", output_path=output_path)
        print(f"\n[SUCCESS] Evaluation report saved to '{output_path}'")
        
        # Save debug summary if enabled
        if debug:
            debug_logger.save_summary()
            print(f"[DEBUG] Debug logs saved to '{debug_logger.save_dir}'")
        
        return final_tender_output
        
    except Exception as e:
        log.exception("pipeline_failed", error=str(e))
        log_step("pipeline_error", {}, {"error": str(e)})
        if debug:
            debug_logger.save_summary()
        raise


def _write_cache(path: str, data: Any) -> None:
    """Write cache file to current directory."""
    cache_path = Path.cwd() / path
    write_json(cache_path, data)


# Backward compatibility: expose the same interface as the original pipeline
__all__ = ["run_pipeline"]
