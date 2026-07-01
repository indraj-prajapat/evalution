"""
Verification Engine v5.1 — Production Grade with Data Point Validation.

A tender criterion verification engine that:
  - Uses LLM to separate related from unrelated evidence
  - Uses LLM to validate data points (field matching, period matching, document type)
  - Uses Python for ALL numeric operations on validated data
  - Verifies evidence against actual Company JSON page text
  - Checks company/entity name consistency
  - Works for ANY tender criterion (no hardcoded domains)
  
v5.1 Changes:
  - Integrated DataPointValidator between evidence filter and numeric verifier
  - Filters out wrong fields, wrong periods, wrong document types, and duplicates
  - Prevents LLM transcription errors from corrupting numeric calculations
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
from .data_point_validator import DataPointValidator

__version__ = "5.1.0"
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
    "DataPointValidator",
]