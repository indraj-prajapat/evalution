"""
CLI entry point for tender_extractor.

Usage:
    python -m tender_extractor company.json tender_plan.json
    python -m tender_extractor company.json tender_plan.json --output result.json
    python -m tender_extractor company.json tender_plan.json --no-llm  # regex-only
"""

from __future__ import annotations

import argparse
import json
import sys

from .api import extract_from_bid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tender Information Extraction Agent (GPT-4o-mini powered)",
    )
    parser.add_argument(
        "company_json",
        help="Path to the company/bid documents JSON",
    )
    parser.add_argument(
        "tender_plan",
        help="Path to the tender information plan JSON",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)",
        default=None,
    )
    parser.add_argument(
        "--bidder-name", "-b",
        help="Bidder company name (default: auto-detect)",
        default="",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM, use regex-only extraction",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--skip-company-match",
        action="store_true",
        help="Skip company ownership verification",
    )

    args = parser.parse_args(argv)

    # Load files
    with open(args.company_json, "r", encoding="utf-8") as f:
        company_data = json.load(f)

    with open(args.tender_plan, "r", encoding="utf-8") as f:
        tender_plan_data = json.load(f)

    # Run extraction
    result = extract_from_bid(
        company_json=company_data,
        tender_plan=tender_plan_data,
        bidder_name=args.bidder_name,
        min_confidence=args.min_confidence,
        debug=args.debug,
        use_llm=not args.no_llm,
        skip_company_match=args.skip_company_match,
    )

    # Output
    output_str = json.dumps(result, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output_str)


if __name__ == "__main__":
    main()