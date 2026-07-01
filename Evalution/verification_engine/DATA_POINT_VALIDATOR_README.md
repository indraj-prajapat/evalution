# Data Point Validator - LLM Layer for Input Validation

## Problem Solved

The evidence filter extracts many data points from documents, but many are wrong:
- Wrong field type (e.g., net profit instead of turnover)
- Wrong financial year/period
- Wrong document type (when specific document is required)
- Multiple conflicting values for the same field+period from different documents
- Invalid/malformed numeric values

**If these wrong data points reach Python's numeric verifier, all calculations become wrong and verdicts are incorrect.**

## Solution

A new **LLM validation layer** between the evidence filter and numeric verifier that:

1. **Validates field matching**: Ensures data point field matches what requirement asks for
2. **Validates period/year matching**: Ensures data point period matches requirement's specified period
3. **Validates document type**: If requirement specifies a document type, only accepts from that type
4. **Resolves conflicts**: When multiple values exist for same field+period, selects ONLY the most reliable source
5. **Filters invalid values**: Rejects non-numeric, ambiguous, or erroneous values

## Architecture

```
Evidence Filter (extracts all candidate data points)
        ↓
Data Point Validator ← NEW LLM LAYER
        ↓
Numeric Verifier (only receives validated, correct data points)
```

## Usage Example

### Before (Wrong data points reach Python):

```python
# Criterion: "Average annual turnover of at least Rs. 5 Crores in last 3 financial years"

# Extracted data points (many wrong):
data_points = [
    {"field": "annual_turnover", "value": "25 Lakhs", "period": "FY2022-23", "source": "ITR.pdf"},
    {"field": "annual_turnover", "value": "30 Lakhs", "period": "FY2022-23", "source": "BalanceSheet.pdf"},
    {"field": "annual_turnover", "value": "5.2 Crores", "period": "FY2022-23", "source": "CA_Certificate.pdf"},  # Correct!
    {"field": "net_profit", "value": "2 Crores", "period": "FY2022-23", "source": "ITR.pdf"},  # Wrong field!
    {"field": "turnover", "value": "4.8 Crores", "period": "FY2021-22", "source": "CA_Certificate.pdf"},  # Wrong year!
]

# Python calculates average: (0.25 + 0.30 + 5.2 + 2.0 + 4.8) / 5 = 2.51 Crores → WRONG FAIL!
```

### After (Only correct data points reach Python):

```python
# After LLM validation:
validated_points = [
    {"field": "annual_turnover", "value": "5.2 Crores", "period": "FY2022-23", "source": "CA_Certificate.pdf"},
]
# Rejected:
# - 25 Lakhs (lower reliability source, CA Certificate is authoritative)
# - 30 Lakhs (lower reliability source)
# - net_profit (wrong field - not turnover)
# - FY2021-22 value (wrong year)

# Python calculates average: 5.2 Crores → CORRECT PASS!
```

## Document Reliability Hierarchy

The LLM uses this hierarchy to select the most authoritative source:

**Tier 1 (Authoritative):**
- CA_Certificate
- Audited_Financial_Statement
- Audit_Report

**Tier 2 (Official):**
- ITR (Income Tax Return)
- Balance_Sheet
- Profit_Loss_Statement

**Tier 3 (Supporting):**
- Bank_Statement
- Financial_Summary
- Turnover_Certificate

**Tier 4 (Unreliable):**
- Self_Declaration
- Affidavit
- Email
- Letter

## Integration

Already integrated into `FactGenerator`:

1. **Import added:**
   ```python
   from .data_point_validator import DataPointValidator
   ```

2. **Initializer updated:**
   ```python
   self._data_point_validator = DataPointValidator(api_key=api_key, model=model)
   ```

3. **Validation step added before numeric verification:**
   ```python
   # Validate data points with LLM
   validated_result = self._data_point_validator.validate_data_points(req, req_data_points)
   
   # Convert to format numeric_verifier expects
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
   
   # Pass only validated points to Python
   num_result = self._numeric_verifier.verify_requirement(req, validated_data_points)
   ```

## Output Format

The validator produces detailed audit trail facts:

```python
{
    "validation_method": "llm",
    "accepted_count": 3,
    "rejected_count": 5,
    "selection_logic": "Selected CA Certificate as authoritative source for turnover figures. Rejected ITR and Balance Sheet values as they are less reliable. Rejected net_profit as wrong field type.",
    "rejected_reasons": [
        "Wrong field: 'net_profit' does not match required 'annual_turnover'",
        "Wrong period: 'FY2021-22' does not match required 'FY2022-23'",
        "Lower reliability source: ITR is Tier 2, CA Certificate is Tier 1",
        "Duplicate value with lower reliability: Balance Sheet vs CA Certificate",
        "Invalid value: 'N/A' is not a numeric value"
    ]
}
```

## Fallback Behavior

If LLM is unavailable, falls back to simple rule-based validation:
- Accepts any value containing numeric digits
- Rejects obvious non-values: "N/A", "nil", "none", "not applicable", "see attached"
- Lower confidence score (0.5) to indicate reduced reliability

## Testing

To test the validator:

```python
from Evalution.verification_engine.data_point_validator import DataPointValidator

validator = DataPointValidator(api_key="your-api-key", model="gpt-4o-mini")

# Mock requirement
class MockRequirement:
    description = "Average annual turnover of at least Rs. 5 Crores in the last 3 financial years (FY2021-22, FY2022-23, FY2023-24)"
    target = "annual_turnover"
    check_type = "threshold"
    threshold_value = 50000000
    operator = ">="

# Test data points
test_points = [
    {"field": "annual_turnover", "value": "25 Lakhs", "period": "FY2022-23", "source_doc": "ITR.pdf", "snippet": "...", "page": 1},
    {"field": "annual_turnover", "value": "5.2 Crores", "period": "FY2022-23", "source_doc": "CA_Certificate.pdf", "snippet": "...", "page": 1},
    {"field": "net_profit", "value": "2 Crores", "period": "FY2022-23", "source_doc": "ITR.pdf", "snippet": "...", "page": 1},
]

result = validator.validate_data_points(MockRequirement(), test_points)

print(f"Accepted: {len(result.validated_points)}")
print(f"Rejected: {len(result.rejected_points)}")
print(f"Logic: {result.selection_logic}")
```

## Benefits

1. **Accuracy**: Only correct data points reach Python calculations
2. **Audit Trail**: Every acceptance/rejection is logged with reasoning
3. **Conflict Resolution**: Automatically selects most authoritative source when duplicates exist
4. **Field Validation**: Prevents wrong field types from corrupting calculations
5. **Period Validation**: Ensures correct financial years are used
6. **Document Type Enforcement**: Respects requirement-specific document type constraints
7. **Fallback Safety**: Continues to work (with reduced accuracy) if LLM is unavailable
