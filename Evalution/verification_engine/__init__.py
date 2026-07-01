"""
Verification Engine v5.0 — Production Grade, Fully Generic.

A tender criterion verification engine that:
  - Uses LLM to separate related from unrelated evidence
  - Uses Python for ALL numeric operations
  - Verifies evidence against actual Company JSON page text
  - Checks company/entity name consistency
  - Works for ANY tender criterion (no hardcoded domains)
"""

from .engine import VerificationEngine, run_verification
from .models import (
    VerificationReport,
    Verdict,
    VerifiedFact,
    Evidence,
    LLMEvaluation,
    parse_module2_output,
    parse_company_json,
)

__version__ = "5.0.0"
__all__ = [
    "VerificationEngine",
    "run_verification",
    "VerificationReport",
    "Verdict",
    "VerifiedFact",
    "Evidence",
    "LLMEvaluation",
    "parse_module2_output",
    "parse_company_json",
]