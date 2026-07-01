# Verification Engine v5.0.0

**Production-grade, fully generic tender criterion verification engine.**

3-stage pipeline:
  1. **LLM separates evidence** — related vs unrelated to the criterion
  2. **Python handles ALL numeric operations** — no LLM for numbers
  3. **Criteria evaluation** — PASS/FAIL/REVIEW with user-friendly reasons

Works for **ANY** tender criterion. Zero hardcoded domains.

---

## Quick Start

```bash
# Install dependencies (openai, python-dotenv):
pip install openai python-dotenv

# Set API key:
export OPENAI_API_KEY="sk-..."

# Run with both Module 2 output and Company JSON:
python run.py -m sample_input_module2.json -c sample_input_company.json --company-name "Acme Corp"

# Run without LLM (Python-only, no OpenAI):
python run.py -m sample_input_module2.json -c sample_input_company.json --no-llm

# Save report to file:
python run.py -m sample_input_module2.json -c sample_input_company.json -o report.json --print
```

---

## Pipeline (v5.0)

```
Module 2 Output + Company JSON
         |
    [1] Parse inputs
         |
    [2] Parse criterion into checkable requirements (generic regex)
         |
    [3] Ground truth search in Company JSON
         |
    [4] LLM filters evidence: related vs unrelated
         |
    [5] Verify related evidence against Company JSON page-level text
         |
    [6] Check company/entity name consistency
         |
    [7] Python handles ALL numeric checks (threshold, comparison, count)
         |
    [8] Generate verified facts
         |
    [9] Python preliminary verdict (PASS/FAIL/REVIEW)
         |
   [10] LLM final evaluation (user-friendly summary)
         |
   Output: VerificationReport JSON
```

---

## Input Format

### Input 1: Module 2 Extraction Output (Required)

```json
{
    "documents": [
        {
            "requirement_document": "PAN Card",
            "mode": "EXPLICIT",
            "entity_type": "Certificate",
            "matched_documents": [
                {
                    "document_id": "doc_41",
                    "document_name": "PAN Card – Acme Corp",
                    "status": "FOUND",
                    "records": [
                        {
                            "entity_id": "doc_41",
                            "group": null,
                            "group_value": null,
                            "fields": [
                                {
                                    "name": "pan_number",
                                    "value": "ABCDE1234F",
                                    "datatype": "string",
                                    "page": 41,
                                    "snippet": "PAN: ABCDE1234F",
                                    "confidence": 0.95,
                                    "raw_value": "ABCDE1234F"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ],
    "criterion_id": "CRIT001",
    "criterion": "Copy of PAN card and GST registration certificate"
}
```

### Input 2: Company JSON (Optional, for Ground Truth)

```json
{
    "Acme_Corp_merged.pdf": {
        "file_name": "Acme_Corp_merged.pdf",
        "summary": { "total_pages": 100, "documents_detected": 100 },
        "documents": {
            "total_pages": 100,
            "documents": [
                {
                    "doc_id": "doc_41",
                    "doc_type": "certificate",
                    "pages": [41],
                    "entities": { "company_name": "Acme Corp" },
                    "document_name": "PAN Card – Acme Corp",
                    "summary": "PAN card showing PAN: ABCDE1234F"
                }
            ]
        },
        "page_41": "Full raw text of page 41 here...",
        "page_42": "Full raw text of page 42 here..."
    }
}
```

---

## Output Format

```json
{
    "criterion_id": "CRIT001",
    "criterion": "Copy of PAN card and GST registration certificate",
    "company_name": "Acme Corp",
    "verdict": "PASS",
    "python_verdict": "PASS",
    "summary": "Acme Corp meets the requirement...",
    "reason": [],
    "verified_facts": [...],
    "evidence": [...],
    "verdict_evidence": [...],
    "missing_information": [],
    "criterion_requirements": [...],
    "informational_notes": ["50 evidence items were unrelated..."],
    "ground_truth_verification": {...},
    "llm_evaluation": {...}
}
```

### Verdict Policy

| Verdict | When |
|---------|------|
| **PASS** | 100% of criterion requirements satisfied with verified evidence. No entity name issues. No missing info. |
| **FAIL** | Something DIRECTLY and CONCLUSIVELY proves the requirement is not met (e.g., threshold provably below). |
| **REVIEW** | Everything else: missing info, partial data, entity name mismatch, uncertainty. |

**Key principle:** Unrelated PARTIAL/NOT_FOUND documents do NOT affect the verdict. Only criterion-relevant evidence matters.

---

## Architecture

```
verification_engine/
  __init__.py              # Package entry point, exports
  models.py                # All data models, parsers, numeric utilities
  criterion_parser.py      # Generic criterion → checkable requirements (regex)
  evidence_filter.py       # LLM-based evidence relevance filter
  numeric_verifier.py      # Pure Python number operations
  entity_checker.py        # Company/entity name verification
  ground_truth.py          # Company JSON search engine
  fact_generator.py        # Orchestrates the full pipeline
  engine.py                # Main engine + verdict determination
  llm_evaluator.py         # LLM final evaluation layer
  run.py                   # CLI entry point
  run_verification.py      # Simple function-call runner
  verifiers/
    __init__.py            # Generic verifier exports
    base.py                # Abstract base verifier
    generic.py             # Count, threshold, date, duplicate, existence
  sample_input_module2.json
  sample_input_company.json
```

---

## Programmatic Usage

```python
import json
from verification_engine import run_verification

# Load inputs
with open("module2.json") as f:
    module2_data = json.load(f)

with open("company.json") as f:
    company_data = json.load(f)

# Run verification
report = run_verification(
    module2_input=module2_data,
    company_json_input=company_data,
    company_name="Acme Corp",
    output_path="report.json",
)

print(report.verdict)       # "PASS" | "FAIL" | "REVIEW"
print(report.summary)
```

Or use the simple runner:

```python
from run_verification import verify_criterion

result = verify_criterion(
    module2_json_path="module2.json",
    company_json_path="company.json",
    company_name="Acme Corp",
)
```

---

## What Changed in v5.0

- **No hardcoded criteria**: Works for ANY tender criterion, not just PAN/GST/CISA
- **LLM evidence filtering**: Separates related from unrelated evidence before evaluation
- **Strict criterion-only verdict**: If criterion is satisfied by relevant evidence, verdict is PASS — regardless of unrelated document status
- **Page-level evidence verification**: Evidence verified against actual Company JSON page text
- **Company name verification**: Cross-checks entity names across documents
- **Python-only numeric operations**: All thresholds, comparisons, counts done by Python
- **3-stage pipeline**: LLM filter → Python compute → Criteria evaluate