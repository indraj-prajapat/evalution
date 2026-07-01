"""
run_verification.py - Simple function call runner (NO CLI).

v5.0 — Production-grade, fully generic verification runner.

Usage:
    from run_verification import verify_criterion

    report = verify_criterion(
        module2_json_path="/path/to/module2_output.json",
        company_json_path="/path/to/company.json",
        company_name="Acme Corp",
        output_path="/path/to/output_report.json",  # optional
    )

    print(f"Verdict: {report.verdict}")
    print(f"Confidence: {report.llm_evaluation.confidence}")
    print(f"Reasoning: {report.llm_evaluation.reasoning}")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

# Ensure the package is importable
_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))
if str(_pkg_dir.parent) not in sys.path:
    sys.path.insert(0, str(_pkg_dir.parent))


def verify_criterion(
    module2_json_path: str,
    company_json_path: str,
    company_name: str,
    output_path: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    use_llm: bool = True,
) -> dict:
    """
    Run the full verification pipeline.

    Args:
        module2_json_path: Path to Module 2 extraction output JSON.
        company_json_path: Path to Company JSON.
        company_name: Name of the company being evaluated.
        output_path: If provided, save the report JSON to this path.
        api_key: OpenAI API key (loaded from .env if not provided).
        model: Model name (defaults to OPENROUTER_MODEL from .env).
        use_llm: Whether to use LLM (default: True).

    Returns:
        dict: The verification report as a dictionary.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from Evalution.verification_engine.engine import run_verification as _run_verification

    report = _run_verification(
        module2_input=module2_json_path,
        company_json_input=company_json_path,
        company_name=company_name,
        output_path=output_path,
        api_key=api_key,
        model=model,
        use_llm=use_llm,
    )

    result = report.to_dict()
    _print_summary(result)

    return result


def _print_summary(report: dict) -> None:
    """Print a human-readable summary of the verification result."""
    verdict = report.get("verdict", "REVIEW")
    python_verdict = report.get("python_verdict", "N/A")
    llm_eval = report.get("llm_evaluation", {})

    colors = {"PASS": "\033[92m", "FAIL": "\033[91m", "REVIEW": "\033[93m"}
    reset = "\033[0m"
    color = colors.get(verdict, "")
    py_color = colors.get(python_verdict, "")

    print("\n" + "=" * 70)
    print(f"  VERIFICATION REPORT v5.0 — {report.get('company_name', 'Unknown Company')}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\n  Criterion:   {report.get('criterion', '')}")
    print(f"  Criterion ID: {report.get('criterion_id', '')}")

    print(f"\n  {'=' * 70}")
    print(f"  FINAL VERDICT:  {color}{verdict}{reset}")
    print(f"  Python Verdict: {py_color}{python_verdict}{reset}")
    if llm_eval:
        print(f"  LLM Model:      {llm_eval.get('model_used', 'N/A')}")
        print(f"  LLM Confidence: {llm_eval.get('confidence', 0):.0%}")
        print(f"  LLM Agreement:  {llm_eval.get('python_verdict_agreement', 'N/A')}")
    print(f"  {'=' * 70}")

    if llm_eval and llm_eval.get("reasoning"):
        print(f"\n  LLM REASONING:")
        for line in llm_eval["reasoning"].split("\n"):
            print(f"    {line}")

    if llm_eval and llm_eval.get("key_findings"):
        print(f"\n  KEY FINDINGS:")
        for finding in llm_eval["key_findings"]:
            print(f"    - {finding}")

    facts = report.get("verified_facts", [])
    print(f"\n  {'─' * 70}")
    print(f"  VERIFIED FACTS ({len(facts)})")
    print(f"  {'─' * 70}")
    for f in facts:
        status = f.get("status", "?")
        fact_color = ""
        if status in ("TRUE", "VERIFIED"):
            fact_color = "\033[92m"
        elif status in ("FALSE", "NOT_FOUND"):
            fact_color = "\033[91m"
        else:
            fact_color = "\033[93m"
        print(f"\n  {fact_color}[{status}]{reset} {f['id']} ({f['type']})")
        print(f"  {f['statement']}")

    missing = report.get("missing_information", [])
    if missing:
        print(f"\n  {'─' * 70}")
        print(f"  MISSING INFORMATION ({len(missing)})")
        print(f"  {'─' * 70}")
        for mi in missing:
            print(f"    - {mi}")

    notes = report.get("informational_notes", [])
    if notes:
        print(f"\n  {'─' * 70}")
        print(f"  INFORMATIONAL NOTES ({len(notes)})")
        print(f"  {'─' * 70}")
        for n in notes:
            print(f"    - {n}")

    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_module2 = 'final_verdict.json'
    sample_company = 'kheria company.json'

    if True:
        result = verify_criterion(
            module2_json_path=str(sample_module2),
            company_json_path=str(sample_company),
            company_name="Sample Firm",
            output_path=str("sample_output_report_v4.json"),
        )
    else:
        print("Sample files not found. Use the verify_criterion() function directly.")
        print('Example:')
        print('  verify_criterion("module2.json", "company.json", "Company Name")')