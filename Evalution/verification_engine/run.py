#!/usr/bin/env python3
"""
run.py - CLI entry point for the Verification Engine (legacy, v2.0 with LLM).

Usage:
    python run.py -m sample_input_module2.json -c sample_input_company.json --company-name "Acme Corp"
    python run.py -m sample_input_module2.json -c sample_input_company.json --no-llm
    python run.py -m sample_input_module2.json -c sample_input_company.json -o report.json --print
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Verification Engine v4.0 - Python deterministic facts + GPT-4o-mini reasoning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--module2", required=True, help="Path to Module 2 JSON.")
    parser.add_argument("-c", "--company", default=None, help="Path to Company JSON.")
    parser.add_argument("-n", "--company-name", default="", help="Company name.")
    parser.add_argument("-o", "--output", default=None, help="Output path for report JSON.")
    parser.add_argument("-p", "--print", action="store_true", dest="print_report", help="Print report.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM evaluation.")
    parser.add_argument("--model", default=None, help="Model name (defaults to OPENROUTER_MODEL from .env).")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output.")

    args = parser.parse_args()

    module2_path = Path(args.module2)
    if not module2_path.exists():
        print(f"ERROR: Module 2 file not found: {module2_path}", file=sys.stderr)
        sys.exit(1)

    company_path = None
    if args.company:
        company_path = Path(args.company)
        if not company_path.exists():
            print(f"ERROR: Company JSON file not found: {company_path}", file=sys.stderr)
            sys.exit(1)

    # Import
    _pkg_dir = Path(__file__).resolve().parent
    _parent_dir = _pkg_dir.parent
    for d in (_pkg_dir, _parent_dir):
        ds = str(d)
        if ds not in sys.path:
            sys.path.insert(0, ds)

    from dotenv import load_dotenv
    load_dotenv(_parent_dir / ".env")

    from verification_engine import run_verification

    report = run_verification(
        module2_input=str(module2_path),
        company_json_input=str(company_path) if company_path else None,
        company_name=args.company_name,
        output_path=args.output,
        model=args.model,
        use_llm=not args.no_llm,
    )

    if args.print_report:
        result = report.to_dict()
        verdict = result.get("verdict", "REVIEW")
        colors = {"PASS": "\033[92m", "FAIL": "\033[91m", "REVIEW": "\033[93m"}
        reset = "\033[0m"
        color = "" if args.no_color else colors.get(verdict, "")
        py_verdict = result.get("python_verdict", "N/A")
        py_color = "" if args.no_color else colors.get(py_verdict, "")

        print(f"\n{'=' * 60}")
        print(f"  FINAL VERDICT (LLM): {color}{verdict}{reset}")
        print(f"  PYTHON VERDICT:       {py_color}{py_verdict}{reset}")
        llm = result.get("llm_evaluation", {})
        if llm:
            print(f"  MODEL:                {llm.get('model_used', 'N/A')}")
            print(f"  CONFIDENCE:           {llm.get('confidence', 0):.0%}")
            print(f"  AGREEMENT:            {llm.get('python_verdict_agreement', 'N/A')}")
        print(f"{'=' * 60}")

        if llm and llm.get("reasoning"):
            print(f"\n  LLM REASONING:\n    {llm['reasoning']}")

        if llm and llm.get("risks"):
            print(f"\n  RISKS:")
            for r in llm["risks"]:
                print(f"    - {r}")

    if args.output:
        print(f"\nReport saved to: {args.output}")

    sys.exit(0 if verdict != "FAIL" else 1)


if __name__ == "__main__":
    main()