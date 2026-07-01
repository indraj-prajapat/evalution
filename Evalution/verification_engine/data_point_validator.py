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

## CRITICAL RULES - READ CAREFULLY:

1. FIELD MATCHING (MOST IMPORTANT):
   - The data point's field MUST match what the requirement asks for
   - If requirement asks for "annual turnover", REJECT data points with field "net profit", "revenue from operations", "total assets", "liabilities", "EMD amount", "performance security", "bid value", "project cost", "staff count", "experience years", etc.
   - ONLY accept synonymous field names (e.g., "turnover" = "annual turnover" = "gross revenue" = "total revenue")
   - USE THE SNIPPET: If snippet shows "Total Assets" but field says "turnover", REJECT it
   - USE DOCUMENT NAME: If document is "EMD Certificate" or "Bank Guarantee", it likely doesn't contain turnover

2. PERIOD/YEAR MATCHING (CRITICAL FOR FINANCIAL DATA):
   - If requirement asks for specific FY (e.g., "FY2022-23"), the data point MUST be from that year
   - DON'T just trust the "period" field - CHECK THE DOCUMENT NAME AND SNIPPET:
     * Document name "Balance Sheet FY 2021-22" → period is FY2021-22 (even if period field is empty)
     * Snippet "as at 31st March 2023" → period is FY2022-23
     * Snippet "for the year ended 31 March 2022" → period is FY2021-22
   - If you see year info in document_name or snippet, USE IT to validate/correct the period
   - If requirement needs 3 years and you have data points from different years, keep them separate - don't mix years

3. DOCUMENT TYPE MATCHING:
   - If requirement specifies document type, ONLY accept from that type
   - Example: "as per audited financial statements" → only CA_Certificate, Balance_Sheet, Audit_Report
   - Example: "as per ITR" → only ITR documents
   - If no document type specified, use reliability hierarchy BUT STILL verify field and period match

4. VALUE CONSISTENCY (AVOID DOUBLE COUNTING):
   - When multiple data points claim SAME field + SAME period with DIFFERENT values:
     * Select ONLY ONE most reliable source (see hierarchy below)
     * Reject all others as "alternative_rejected" with reason "Duplicate value for {{field}} in {{period}}, selected more reliable source: {{selected_source}}"
   - NEVER select multiple conflicting values for same field+period

5. DOCUMENT RELIABILITY HIERARCHY (most to least reliable):
   Tier 1 (Authoritative): CA_Certificate, Audited_Financial_Statement, Audit_Report, Form_VI_Audited_Financials
   Tier 2 (Official): ITR, Balance_Sheet, Profit_Loss_Statement, Annual_Report
   Tier 3 (Supporting): Bank_Statement, Financial_Summary, Turnover_Certificate, Computation_of_Income
   Tier 4 (Unreliable): Self_Declaration, Affidavit, Email, Letter, Excel_Sheet, Working_Notes

6. NUMERIC VALIDITY:
   - Reject if value is not a number: "N/A", "Not Provided", "See attached", "Refer notes", "Nil", "None"
   - Reject if value has obvious errors: negative turnover, impossibly large numbers (e.g., 999999 Crores)
   - Reject if value is ambiguous: "5" without units when context suggests "5 Crores" vs "5 Lakhs"

7. CONTEXT AWARENESS (USE ALL AVAILABLE INFO):
   - Check document_name: "EMD_Bank_Guarantee.pdf" won't have turnover; "Experience_Certificate.pdf" won't have financial data
   - Check snippet: If snippet mentions "EMD", "Bid Security", "Performance Bank Guarantee", "staff strength", "years of experience" - it's NOT turnover/profit data
   - Check page number: Financial statements are usually in specific pages; if page 34 shows "Branch details" and "Borrowers", it's NOT company turnover
   - Cross-reference: If document_name says "FY 2021-22" but period field says "FY2022-23", trust document_name

8. COMMON MISTAKES TO AVOID:
   - Don't confuse "turnover" with "total assets", "liabilities", "net worth", "paid-up capital"
   - Don't confuse company turnover with partner's personal income
   - Don't confuse project value/bid amount with company turnover
   - Don't confuse EMD amount/performance security with turnover
   - Don't confuse staff count/experience years with financial figures
   - Don't mix up financial years - FY2021-22 ≠ FY2022-23 ≠ FY2023-24

## YOUR OUTPUT MUST:
- Be EXTREMELY STRICT: better to reject 10 good points than accept 1 wrong point
- WRONG POINTS CAUSE WRONG VERDICTS: If Python gets wrong input, calculation is wrong, verdict is wrong
- Provide CLEAR reasoning: "Rejected: wrong field - snippet shows 'Total Assets' not turnover" 
- When in doubt, REJECT the data point (Python can work with fewer points, but wrong points break everything)
- If a data point has NO period info anywhere (field, document_name, snippet), mark confidence as 0.3 and note "Period could not be verified"
"""

_VALIDATION_USER_PROMPT = """## REQUIREMENT TO VERIFY
{requirement}

## EXPECTED PARAMETERS
- Target Field: {target_field}
- Expected Period: {expected_period}
- Expected Document Type: {expected_document_type}

## CANDIDATE DATA POINTS
{data_points_section}

## YOUR TASK - STEP BY STEP VALIDATION

For EACH candidate data point, follow these steps IN ORDER:

STEP 1 - CHECK DOCUMENT NAME:
  - Look at document_name: Does it suggest this document would contain the required data?
  - Examples:
    * Requirement: turnover → Document: "EMD_Bank_Guarantee.pdf" → REJECT (wrong document type)
    * Requirement: turnover → Document: "Balance Sheet FY 2021-22" → ACCEPT (correct document type for that year)
    * Requirement: experience years → Document: "Financial_Report.pdf" → REJECT (financial docs don't have experience info)

STEP 2 - CHECK SNIPPET CONTENT:
  - Read the snippet carefully: What does it actually show?
  - Examples:
    * Snippet mentions "EMD", "Bid Security", "Bank Guarantee" → NOT financial performance data
    * Snippet mentions "Branch details", "Borrowers", "Account No" → NOT company turnover
    * Snippet shows "Total Assets" but field says "turnover" → REJECT (wrong field)
    * Snippet shows "Staff strength: 50 employees" → NOT a financial figure

STEP 3 - EXTRACT/VERIFY PERIOD FROM DOCUMENT NAME AND SNIPPET:
  - Don't trust the "period" field alone - cross-check with:
    * Document name: "Balance Sheet FY 2021-22" → period is FY2021-22
    * Snippet: "as at 31st March 2023" → period is FY2022-23
    * Snippet: "for the year ended 31 March 2022" → period is FY2021-22
  - If document_name/snippet contradicts the period field, TRUST document_name/snippet
  - If no period info anywhere, note "Period could not be verified" and set confidence to 0.3

STEP 4 - FIELD MATCHING:
  - Does the field (after checking snippet context) match what requirement asks?
  - Accept only exact matches or clear synonyms (turnover = annual turnover = gross revenue)
  - Reject if field is different concept (assets ≠ turnover, liabilities ≠ turnover, EMD ≠ turnover)

STEP 5 - VALUE VALIDITY:
  - Is the value a valid number (possibly with units like Lakhs/Crores)?
  - Reject: "N/A", "Not Provided", "See attached", "Nil", "None", negative numbers, impossibly large numbers

STEP 6 - DUPLICATE CHECK:
  - If multiple points have SAME field + SAME period (after your verification in Step 3) but DIFFERENT values:
    * Select ONLY the most reliable source (use document hierarchy)
    * Reject others as duplicates

STEP 7 - ASSIGN CONFIDENCE:
  - 0.9-1.0: All checks passed, period verified from document_name/snippet, authoritative source
  - 0.7-0.9: All checks passed, period inferred from context, reliable source
  - 0.5-0.7: All checks passed, but period unclear or source less reliable
  - 0.3-0.5: Major uncertainty (e.g., no period info anywhere, ambiguous snippet)
  - Below 0.3: Reject instead

STEP 8 - PROVIDE CLEAR REASONING:
  - For accepted: "Field matches '{{target_field}}', period verified as {{period}} from {{source}}, value is valid number from {{doc_type}}"
  - For rejected: Be specific - "Rejected: snippet shows 'Total Assets' not turnover" or "Rejected: document is EMD certificate, doesn't contain turnover data" or "Rejected: period is FY2021-22 (from document name) but requirement needs FY2022-23"

Respond in JSON format:
{{
  "selection_logic": "Brief explanation of your validation approach and key findings",
  "validated_points": [
    {{
      "field": "field name",
      "value": "extracted value",
      "is_numeric": true/false,
      "period": "period or empty string",
      "source_doc": "document name",
      "confidence_score": 0.95,
      "validation_reason": "Detailed reason why this point is valid: field match, period verification, document type appropriateness, value validity"
    }}
  ],
  "rejected_points": [
    {{
      "field": "field name",
      "value": "extracted value",
      "period": "period or empty string (your verified period, not just from input)",
      "source_doc": "document name",
      "rejection_reason": "SPECIFIC reason: e.g., 'Wrong field: snippet shows Total Assets not turnover', or 'Wrong period: document name says FY2021-22 but requirement needs FY2022-23', or 'Wrong document type: EMD certificate doesn't contain financial data', or 'Duplicate: lower reliability source for same field+period'"
    }}
  ]
}}

REMEMBER: 
- Your job is to FILTER OUT wrong data points so Python receives ONLY correct inputs
- If wrong data reaches Python, calculations will be wrong and verdict will be wrong
- It's better to reject 10 good points than accept 1 wrong point
- USE ALL AVAILABLE CONTEXT: document_name, snippet, page number - not just field/value pairs
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
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from Evalution.client import get_llm_model
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.model = model or get_llm_model()
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
