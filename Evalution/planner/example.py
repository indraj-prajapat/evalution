#!/usr/bin/env python3
"""End-to-end example showing how to use the tender_information_planner package.

This script demonstrates:

1. Configuring the planner with an OpenAI-compatible endpoint (works with
   Grok, GPT-4o, or any compatible model).
2. Passing raw tender criteria JSON.
3. Running the planner and printing the results.
4. Handling validation failures gracefully.

Usage:
    python -m planner.example                          # uses env vars
    OPENAI_API_KEY=sk-... python -m planner.example     # explicit key
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the package root is importable when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Evalution.client import get_llm_model
from Evalution.planner.config import LLMConfig, PlannerConfig, RetryConfig
from Evalution.planner.logging_utils import configure_logging, get_logger
from Evalution.planner.planner import TenderPlanner

configure_logging(level="DEBUG")
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sample tender JSON with realistic criteria
# ---------------------------------------------------------------------------

SAMPLE_TENDER: dict = {
    "tender_id": "TND-2024-0042",
    "title": "Construction of Community Health Centre",
    "key_points": [
        {
            "criterion_id": "CRIT001",
            "text": (
                """
                The entity should have an average annual turnover of not less than Rs.20 Lakh per annum during last 3 
financial years ending as on 31st March 2025.
Bidders shall submit Annual Accounts (Profit & Loss Statement and Balance Sheet) for the last three years 
ending as on 31st March 2025 with UDIN) 

                """
            ),
        },
     
    ],
}

import os
from dotenv import load_dotenv

# This looks for a .env file and loads the variables into os.environ
load_dotenv()


def main() -> None:
    """Run the planner against the sample tender and display results."""
    # --- Configuration ---
    # Supports any OpenAI-compatible endpoint.
    # For Grok free model, set:
    #   base_url = "https://api.x.ai/v1"
    #   model = "grok-3"
    #   api_key = os.environ.get("XAI_API_KEY", "")
    config = PlannerConfig(
        llm=LLMConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=get_llm_model(),
            temperature=0.2,
            max_tokens=4096,
            timeout=120,
        ),
        retry=RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
        ),
        validation_retries=3,
    )

    planner = TenderPlanner(config=config)

    log.info("example_start", tender_id=SAMPLE_TENDER["tender_id"])

    # --- Run the planner ---
    results = planner.plan_from_tender_json(
        tender_json=SAMPLE_TENDER,
        key_points_key="key_points",
        id_key="criterion_id",
        text_key="text",
    )
    from dataclasses import asdict, is_dataclass

    def to_json_dict(obj):
        """Recursively converts dataclasses and Pydantic models into serializable dicts."""
        if obj is None: return None
        if hasattr(obj, "model_dump"): return obj.model_dump(mode="json")
        if hasattr(obj, "dict"): return obj.dict()
        if is_dataclass(obj):
            return {k: to_json_dict(v) for k, v in asdict(obj).items()}
        if isinstance(obj, (list, tuple)): return [to_json_dict(item) for item in obj]
        if isinstance(obj, dict): return {k: to_json_dict(v) for k, v in obj.items()}
        return obj

    # Convert the entire list of results and write it out completely
    full_serializable_data = to_json_dict(results)
    with open("tender_plan_results.json", "w", encoding="utf-8") as f:
        json.dump(full_serializable_data, f, indent=4, ensure_ascii=False)
        
    print("\n[SUCCESS] Entire plan saved completely to 'tender_plan_results.json'")

    # ---------------------------------------------------------------------------
    # 2. DISPLAY AND LOG INDIVIDUAL ITEMS TO TERMINAL
    # ---------------------------------------------------------------------------
    for result in results:
        if result.success and result.plan is not None:
            print(f"\n{'='*60}")
            print(f"   {result.criterion_id}  —  SUCCESS  (attempts: {result.attempts})")
            print(f"{'='*60}")
            
            output = result.plan.model_dump(mode="json", exclude_none=True)
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(f"\n{'='*60}")
            print(f"   {result.criterion_id}  —  FAILED  (attempts: {result.attempts})")
            print(f"{'='*60}")
            print(f"Error:\n{result.error}")

    # --- Summary ---
    succeeded = sum(1 for r in results if r.success)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"   Summary: {succeeded}/{total} criteria planned successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()