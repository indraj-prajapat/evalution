"""
Data Point Validator - LLM-based validation of extracted data points.

This module sits between the evidence filter and the numeric verifier.
Its job is to ensure ONLY correct, relevant data points are passed to
Python for numeric computation.

Problem it solves:
- Evidence filter extracts many data points, many of which are wrong
- For a single requirement (e.g., turnover for FY2022-23), we may get 
  multiple data points from different documents, but only ONE is correct
- If wrong data points reach Python, all calculations become wrong
- This validator uses LLM to select the RIGHT input(s) from candidates

Example:
  Criterion: "Average annual turnover of at least Rs. 5 Crores in the last 3 financial years"
  
  Extracted data points (from evidence filter):
  - {"field": "annual_turnover", "value": "25 Lakhs", "period": "FY2022-23", "source": "ITR_Doc1.pdf"}
  - {"field": "annual_turnover", "value": "30 Lakhs", "period": "FY2022-23", "source": "BalanceSheet.pdf"}
  - {"field": "annual_turnover", "value": "5.2 Crores", "period": "FY2022-23", "source": "CA_Certificate.pdf"}
  - {"field": "net_profit", "value": "2 Crores", "period": "FY2022-23", "source": "ITR_Doc1.pdf"}  # Wrong field!
  - {"field": "turnover", "value": "4.8 Crores", "period": "FY2021-22", "source": "CA_Certificate.pdf"}  # Wrong year!
  
  After LLM validation:
  - Only: {"field": "annual_turnover", "value": "5.2 Crores", "period": "FY2022-23", "source": "CA_Certificate.pdf"}
  - Reason: CA Certificate is authoritative for turnover, other values are from less reliable sources
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from .models import Module2Output, CompanyJSON

load_dotenv()


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ValidatedDataPoint:
    """A data point with LLM validation assessment."""
    field: str
    value: str
    is_numeric: bool
    period: str = ""
    source_doc: str = ""
    document_name: str = ""
    snippet: str = ""
    page: int = 0
    
    # Validation results
    is_valid: bool = False
    validation_reason: str = ""
    confidence_score: float = 0.0
    alternative_rejected: bool = False  # True if this was one of multiple candidates but not selected
    rejection_reason: str = ""


@dataclass
class DataPointValidationResult:
    """Result of data point validation for a requirement."""
    requirement_description: str = ""
    target_field: str = ""
    expected_period: str = ""
    expected_document_type: str = ""
    
    total_candidates: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    
    validated_points: list[ValidatedDataPoint] = field(default_factory=list)
    rejected_points: list[ValidatedDataPoint] = field(default_factory=list)
    
    # LLM's explanation of selection logic
    selection_logic: str = ""
    validation_method: str = "llm"


# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

_VALIDATION_SYSTEM_PROMPT = """You are an expert tender compliance analyst specializing in data extraction validation.

Your task is to validate extracted data points and select ONLY the CORRECT ones for numeric verification.

## CRITICAL RULES:

1. FIELD MATCHING: The data point's field must match what the requirement asks for.
   - If requirement asks for "annual turnover", reject data points with field "net profit", "revenue from operations", etc.
   - Exception: Accept synonymous field names (e.g., "turnover" = "annual turnover" = "gross revenue")

2. PERIOD/YEAR MATCHING: The data point's period must match the requirement's specified period.
   - If requirement asks for "FY2022-23", reject data points with period "FY2021-22" or "FY2023-24"
   - If no specific period is required, accept any valid period

3. DOCUMENT TYPE MATCHING: If the requirement specifies a document type, ONLY accept data points from that document type.
   - Example: "as per audited financial statements" → only accept from CA_Certificate, Balance_Sheet, Audit_Report
   - Example: "as per ITR" → only accept from ITR documents
   - If no document type is specified, use document reliability hierarchy (see below)

4. VALUE CONSISTENCY: When multiple data points claim the same field+period:
   - Select the ONE most reliable source (see document hierarchy below)
   - Reject all others as "alternative_rejected"
   - NEVER select multiple conflicting values for the same field+period

5. DOCUMENT RELIABILITY HIERARCHY (most to least reliable):
   Tier 1 (Authoritative): CA_Certificate, Audited_Financial_Statement, Audit_Report
   Tier 2 (Official): ITR (Income Tax Return), Balance_Sheet, Profit_Loss_Statement
   Tier 3 (Supporting): Bank_Statement, Financial_Summary, Turnover_Certificate
   Tier 4 (Unreliable): Self_Declaration, Affidavit, Email, Letter

6. NUMERIC VALIDITY: Reject data points where:
   - Value is clearly not a number (e.g., "N/A", "Not Provided", "See attached")
   - Value contains obvious errors (e.g., negative turnover, impossibly large numbers)
   - Value is ambiguous or incomplete (e.g., "5" without units when context suggests "5 Crores")

7. CONTEXT AWARENESS: Use the snippet and document context to verify the data point makes sense.
   - Example: If snippet shows "Total Assets: 5 Crores" but field is "annual_turnover", reject it

## YOUR OUTPUT MUST:
- Be extremely strict: better to reject a questionable point than accept a wrong one
- Provide clear reasoning for each acceptance/rejection
- When in doubt, reject the data point (Python can work with fewer points, but wrong points cause wrong verdicts)
"""

_VALIDATION_USER_PROMPT = """## REQUIREMENT TO VERIFY
{requirement}

## EXPECTED PARAMETERS
- Target Field: {target_field}
- Expected Period: {expected_period}
- Expected Document Type: {expected_document_type}

## CANDIDATE DATA POINTS
{data_points_section}

## YOUR TASK
For EACH candidate data point:
1. Verify if the field matches the target field (allow synonyms)
2. Verify if the period matches the expected period (if specified)
3. Verify if the document type is appropriate (if specified, otherwise use reliability hierarchy)
4. Check if the value is a valid number
5. If multiple points have same field+period, select ONLY the most reliable source
6. Provide a confidence score (0.0 to 1.0) for each accepted point

Respond in JSON format:
{{
  "selection_logic": "Brief explanation of how you selected the valid points",
  "validated_points": [
    {{
      "field": "field name",
      "value": "extracted value",
      "is_numeric": true/false,
      "period": "period or empty string",
      "source_doc": "document name",
      "confidence_score": 0.95,
      "validation_reason": "Why this point is valid and selected"
    }}
  ],
  "rejected_points": [
    {{
      "field": "field name",
      "value": "extracted value",
      "period": "period or empty string",
      "source_doc": "document name",
      "rejection_reason": "Specific reason for rejection (wrong field, wrong period, wrong document type, duplicate with lower reliability, invalid value, etc.)"
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Data Point Validator
# ---------------------------------------------------------------------------

class DataPointValidator:
    """
    Validates extracted data points using LLM before they reach the numeric verifier.
    
    This ensures:
    - Only field-matching data points are used
    - Only period-matching data points are used  
    - Only document-type-appropriate data points are used
    - Conflicting values are resolved by selecting the most reliable source
    - Invalid/malformed values are filtered out
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                return None
            from Evalution.client import get_openrouter_client
            self._client = get_openrouter_client(api_key=self.api_key, model=self.model)
        return self._client
    
    def _build_data_points_section(self, data_points: list[dict]) -> str:
        """Build the data points section for the LLM prompt."""
        lines = []
        for i, dp in enumerate(data_points):
            lines.append(
                f"[{i}] Field: {dp.get('field', 'unknown')}\n"
                f"    Value: {dp.get('value', 'N/A')}\n"
                f"    Is Numeric: {dp.get('is_numeric', False)}\n"
                f"    Period: {dp.get('period', 'not specified')}\n"
                f"    Source Document: {dp.get('source_doc', dp.get('document_name', 'unknown'))}\n"
                f"    Snippet: {dp.get('snippet', '')[:200]}{'...' if len(dp.get('snippet', '')) > 200 else ''}"
            )
        return "\n\n".join(lines)
    
    def _parse_requirement_context(self, requirement) -> tuple[str, str, str]:
        """Extract target field, expected period, and expected document type from requirement."""
        from .criterion_parser import RequirementCheck
        
        target_field = requirement.target if hasattr(requirement, 'target') else ""
        expected_period = ""
        expected_document_type = ""
        
        # Extract period from requirement description
        desc = str(getattr(requirement, 'description', ''))
        
        # Look for financial year patterns
        fy_patterns = [
            r'FY\s*(\d{4}-\d{2,4})',
            r'financial\s+year\s+(\d{4}-\d{2,4})',
            r'(\d{4}-\d{2,4})',
            r'last\s+(\d+)\s+financial\s+years?',
            r'past\s+(\d+)\s+financial\s+years?',
        ]
        for pattern in fy_patterns:
            match = re.search(pattern, desc, re.IGNORECASE)
            if match:
                expected_period = match.group(0)
                break
        
        # Look for document type requirements
        doc_patterns = [
            (r'(?:as\s+per\s+|according\s+to\s+|from\s+|in\s+)(audited\s+financial\s+statements?|CA\s+certificate|ITR|income\s+tax\s+return|balance\s+sheet|audit\s+report)', 'authoritative'),
            (r'(audited\s+financial\s+statements?|CA\s+certificate)', 'CA_Certificate'),
            (r'(ITR|income\s+tax\s+return)', 'ITR'),
            (r'(balance\s+sheet|profit\s+and\s+loss)', 'Balance_Sheet'),
        ]
        for pattern, doc_type in doc_patterns:
            match = re.search(pattern, desc, re.IGNORECASE)
            if match:
                expected_document_type = doc_type
                break
        
        return target_field, expected_period, expected_document_type
    
    def validate_data_points(
        self,
        requirement,
        data_points: list[dict],
    ) -> DataPointValidationResult:
        """
        Validate data points for a specific requirement.
        
        Args:
            requirement: The RequirementCheck object describing what to verify
            data_points: List of candidate data points from evidence filter
            
        Returns:
            DataPointValidationResult with validated and rejected points
        """
        result = DataPointValidationResult(
            requirement_description=str(getattr(requirement, 'description', '')),
            validation_method="llm",
        )
        
        if not data_points:
            result.selection_logic = "No data points provided for validation"
            return result
        
        # Extract requirement context
        target_field, expected_period, expected_document_type = self._parse_requirement_context(requirement)
        result.target_field = target_field
        result.expected_period = expected_period
        result.expected_document_type = expected_document_type
        
        # Build prompt
        data_points_section = self._build_data_points_section(data_points)
        user_prompt = _VALIDATION_USER_PROMPT.format(
            requirement=result.requirement_description,
            target_field=target_field or "not specified",
            expected_period=expected_period or "not specified",
            expected_document_type=expected_document_type or "not specified (use reliability hierarchy)",
            data_points_section=data_points_section,
        )
        
        try:
            client = self._get_client()
            if not client:
                # Fallback: return all points as-is with low confidence
                return self._fallback_validation(result, data_points)
            
            response = client.create_chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": _VALIDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            
            raw = response.choices[0].message.content or "{}"
            return self._parse_llm_response(raw, result, data_points)
            
        except Exception as e:
            # Fallback to simple rule-based validation
            return self._fallback_validation(result, data_points, str(e))
    
    def _parse_llm_response(
        self,
        raw: str,
        result: DataPointValidationResult,
        original_data_points: list[dict],
    ) -> DataPointValidationResult:
        """Parse the LLM's JSON response."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self._fallback_validation(result, original_data_points, "Invalid LLM JSON")
            else:
                return self._fallback_validation(result, original_data_points, "No JSON in LLM response")
        
        result.selection_logic = data.get("selection_logic", "LLM did not provide selection logic")
        
        # Process validated points
        for vp in data.get("validated_points", []):
            validated_point = ValidatedDataPoint(
                field=vp.get("field", ""),
                value=vp.get("value", ""),
                is_numeric=vp.get("is_numeric", False),
                period=vp.get("period", ""),
                source_doc=vp.get("source_doc", ""),
                document_name=vp.get("source_doc", ""),
                is_valid=True,
                validation_reason=vp.get("validation_reason", ""),
                confidence_score=float(vp.get("confidence_score", 0.5)),
            )
            result.validated_points.append(validated_point)
        
        # Process rejected points
        for rp in data.get("rejected_points", []):
            rejected_point = ValidatedDataPoint(
                field=rp.get("field", ""),
                value=rp.get("value", ""),
                is_numeric=rp.get("is_numeric", False),
                period=rp.get("period", ""),
                source_doc=rp.get("source_doc", ""),
                document_name=rp.get("source_doc", ""),
                is_valid=False,
                rejection_reason=rp.get("rejection_reason", "Rejected by LLM"),
                alternative_rejected=True,
            )
            result.rejected_points.append(rejected_point)
        
        result.total_candidates = len(original_data_points)
        result.invalid_count = len(result.rejected_points)
        
        return result
    
    def _fallback_validation(
        self,
        result: DataPointValidationResult,
        data_points: list[dict],
        error_message: str = "",
    ) -> DataPointValidationResult:
        """
        Fallback rule-based validation when LLM is unavailable.
        
        This is less accurate but ensures the pipeline continues.
        """
        result.validation_method = f"rule_based_fallback ({error_message[:50]})" if error_message else "rule_based_fallback"
        result.selection_logic = "Used rule-based fallback due to LLM unavailability"
        
        # Simple rules: accept all numeric-looking points
        for dp in data_points:
            value = str(dp.get("value", ""))
            is_numeric_looking = bool(
                re.search(r'\d', value) and 
                value.lower() not in ('na', 'n/a', 'nil', 'none', 'not applicable', 'see attached')
            )
            
            if is_numeric_looking:
                validated_point = ValidatedDataPoint(
                    field=dp.get("field", ""),
                    value=value,
                    is_numeric=dp.get("is_numeric", False),
                    period=dp.get("period", ""),
                    source_doc=dp.get("source_doc", dp.get("document_name", "")),
                    document_name=dp.get("source_doc", dp.get("document_name", "")),
                    snippet=dp.get("snippet", ""),
                    page=dp.get("page", 0),
                    is_valid=True,
                    validation_reason="Accepted by rule-based fallback (contains numeric value)",
                    confidence_score=0.5,  # Lower confidence for fallback
                )
                result.validated_points.append(validated_point)
            else:
                rejected_point = ValidatedDataPoint(
                    field=dp.get("field", ""),
                    value=value,
                    is_numeric=dp.get("is_numeric", False),
                    period=dp.get("period", ""),
                    source_doc=dp.get("source_doc", dp.get("document_name", "")),
                    document_name=dp.get("source_doc", dp.get("document_name", "")),
                    is_valid=False,
                    rejection_reason="Rejected by rule-based fallback (non-numeric value)",
                    alternative_rejected=True,
                )
                result.rejected_points.append(rejected_point)
        
        result.total_candidates = len(data_points)
        result.invalid_count = len(result.rejected_points)
        
        return result
    
    def validate_for_multiple_requirements(
        self,
        requirements: list,
        all_data_points: list[dict],
    ) -> dict:
        """
        Validate data points for multiple requirements.
        
        Args:
            requirements: List of RequirementCheck objects
            all_data_points: All candidate data points from evidence filter
            
        Returns:
            Dictionary mapping requirement index to validation result
        """
        results = {}
        
        for i, req in enumerate(requirements):
            # Filter data points that might be relevant to this requirement
            # (The evidence filter already did broad filtering, this is fine-tuning)
            req_result = self.validate_data_points(req, all_data_points)
            results[i] = req_result
        
        return results


# ---------------------------------------------------------------------------
# Integration helper for FactGenerator
# ---------------------------------------------------------------------------

def integrate_with_fact_generator():
    """
    Helper to show how to integrate DataPointValidator into FactGenerator.
    
    Usage in fact_generator.py:
    
    1. Import:
       from .data_point_validator import DataPointValidator
    
    2. In __init__:
       self._data_point_validator = DataPointValidator(api_key=api_key, model=model)
    
    3. In generate_all_facts(), Step 8c (before calling numeric_verifier):
       
       # Validate data points before sending to Python
       validated_result = self._data_point_validator.validate_data_points(req, req_data_points)
       
       # Convert validated points back to the format numeric_verifier expects
       validated_data_points = [
           {
               "field": vp.field,
               "value": vp.value,
               "is_numeric": vp.is_numeric,
               "period": vp.period,
               "source_doc": vp.source_doc,
               "document_name": vp.document_name,
               "snippet": vp.snippet,
               "page": vp.page,
           }
           for vp in validated_result.validated_points
       ]
       
       # Now pass validated points to numeric verifier
       num_result = self._numeric_verifier.verify_requirement(req, validated_data_points)
    """
    pass
