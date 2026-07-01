# Evaluation “main2” Pipeline — End-to-End README

This document explains the complete end-to-end evaluation pipeline implemented in `main2.py`.  
If you understand this README, you should be able to follow the data flow, know what happens at each stage, and understand the produced outputs (including intermediate caches) without reading the code.

---

## 1) What this pipeline does (high level)

Given:

- a **tender output JSON** that contains a list of **key points / criteria** (the things to evaluate), and
- a **company JSON** that contains parsed **bid documents** (the evidence to search/verify against),

the pipeline:

1. **Plans** what evidence is required to evaluate each key point (LLM-based planning + validation/repair).
2. **Extracts** evidence for each planned requirement (RAG + LLM scoring + extraction).
3. **Verifies** extracted evidence against the company’s page text and computes a verdict:
   - `PASS`, `FAIL`, or `REVIEW`
4. **Writes a final tender-shaped report** by appending verdict/summary/matched documents into the original `tender_output.json` structure.

Outputs are written to `evalution.json` by default, plus optional intermediate cache JSON files.

---

## 2) Entry point and default file paths

The pipeline entry point is the `run_pipeline(...)` function in:

- `main2.py`

Default hardcoded configuration inside `main2.py`:

- **Input 1**: `tender_output.json`
- **Input 2**: `kheria company.json`
- **Output**: `evalution.json`

Intermediate cache files saved by the pipeline:

- `tender_plan_results.json`
- `final_verdict.json`
- `evaluation_results.json`

Parallelism:

- `EVALUATION_MAX_WORKERS` env var controls thread count (defaults to `4`).

---

## 3) Stage-by-stage pipeline (full view)

### Stage 0 — Load inputs & initialize logging

**Where:** `run_pipeline(...)` in `main2.py`

1. Load `tender_output_json`
2. Read `kheria company.json` into `company_data`
3. Determine worker count:
   - `worker_count = min(configured_max_workers, number_of_key_points)` (at least `1`)
4. Configure planner config (LLM settings + retry + validation retries)
5. Build a “planner tender” object:
   - `tender_id` chosen from tender metadata
   - title
   - list of key points mapped into `criteria[]` with:
     - `criterion_id`
     - `text` (from `key_point["point"]` or `key_point["text"]`)

---

### Stage 1 — Planning (LLM) per key point

**Where:** `_plan_key_points(...)` and `_plan_one_key_point(...)` in `main2.py`  
**Core component:** `Evalution/planner/planner.py` → `TenderPlanner`

For each key point:

1. Create a `criterion_id` and criterion text.
2. Call `TenderPlanner.plan_single(...)` which internally:
   - Builds a **prompt** using `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE`
   - Calls an LLM via `Evalution/planner/client.py`
   - Validates the returned plan with `PlanValidator`
   - If validation fails, performs a **validation-repair loop** using `REPAIR_PROMPT_TEMPLATE`
   - On success, returns a structured plan (`CriterionPlan`) containing:
     - `required_documents` (the evidence types/documents to look for)
     - other structured plan fields produced by the schema
3. Planning results are assembled in the same order as input key points.

Parallelism:
- Planning uses `ThreadPoolExecutor` when `max_workers > 1`.
- Each key point planning runs in a separate task.

**Cache written:**
- `tender_plan_results.json`  
  (stored after converting planning result objects into plain JSON using recursive conversion)

---

### Stage 2 — Evaluate planned key points (Extraction + Verification)

**Where:** `_evaluate_planned_key_points(...)` in `main2.py`  
**Parallelism:** Threaded evaluation across only the successfully planned key points.

For each planning result that succeeded (`result.success == True` and `result.plan != None`):

#### 2A) Evidence extraction (RAG + LLM scoring + extraction)

**Where:** `_evaluate_one_key_point(...)` in `main2.py`  
**Core component:** `Evalution/tender_extractor/api.py` → `extract_from_bid(...)`

Inputs:

- `company_json` (company_data)
- `tender_plan` (the planned evidence requirements extracted from `planning_result.plan`)

Extraction function behavior (as implemented in `extract_from_bid`):

1. Load/adapt inputs into internal models:
   - `adapt_company_json(company_json)`
   - `adapt_tender_plan(tender_plan)`
2. Auto-detect bidder name (unless provided)
3. Run `TenderExtractor.run()` with an `ExtractionConfig`:
   - `min_confidence` threshold
   - scoring includes:
     - RAG retrieval and scoring
     - LLM document scoring
   - runs GPT-based extraction per relevant document/page
4. Returns an output structure including:
   - extracted evidence mapped to criterion requirements
   - evidence documents with matched/excluded docs and reasons

**This extraction output is stored in the final per-key-point record as `extraction`.**

#### 2B) Evidence verification and verdict computation

**Where:** `_evaluate_one_key_point(...)` in `main2.py`  
**Core component:** `Evalution/verification_engine/engine.py` → `run_verification(...)`

Inputs:

- `module2_input`: the extraction output from Stage 2A
- `company_json_input`: the raw company JSON
- `company_name`: human readable company name

Verification engine steps (summary from `VerificationEngine.run()`):

1. Parse module2/extraction output into internal structures
2. Parse criterion into generic checkable requirements
3. Ground truth search (generic regex-based keyword selection)
4. LLM filters evidence into related vs unrelated
5. Verify related evidence against the **company’s page-level text**
6. Check entity/company name consistency
7. Use Python for authoritative numeric operations
8. Generate verified facts and missing information
9. Determine `python_verdict` using a generic policy:
   - numeric authoritative failures → `FAIL`
   - everything else uncertain → `REVIEW`
   - no authoritative failures/missing items → `PASS`
10. Optionally call LLM evaluator to produce a final user-friendly verdict/summary

**Final verification output is saved as `verification`** (via `verification_report.to_dict()`).

Error handling:
- If extraction or verification throws an exception, the pipeline stores:
  - `error: str(exc)`
  - with `extraction=None`, `verification=None` for that key point.

Parallelism:
- `_evaluate_planned_key_points` uses `ThreadPoolExecutor` when `max_workers > 1`.

---

### Stage 3 — Cache subsets of results

After all evaluations:

- `final_verdict.json` stores:
  - `extraction` list for key points where `extraction != None`
- `evaluation_results.json` stores:
  - `verification` list for key points where `verification != None`

These are “cache / debugging” artifacts used for later inspection.

---

### Stage 4 — Assemble final output and write report

**Where:** `_append_evaluations_to_key_points(...)` in `main2.py`  
**Output file:** `evalution.json` (default)

The pipeline deep-copies the original `tender_output` and appends into each key point:

- If `evaluation_result.verification` exists:
  - `key_point["verdict"] = verification.get("verdict", "REVIEW")`
  - `key_point["summary"] = verification.get("summary", "")`
  - `key_point["matched_docuements"]` is created from extraction results

- Otherwise (planning failed, missing plan, extraction/verification exception, etc.):
  - `key_point["verdict"] = "REVIEW"`
  - `key_point["summary"] = <failure summary>`
  - `key_point["matched_docuements"] = []`

Important: the code intentionally uses the misspelled key name:
- `matched_docuements` (not `matched_documents`)

---

## 4) Verdict policy (what PASS/FAIL/REVIEW means)

Final per-key-point `verdict` comes from `verification_engine`.

From `Evalution/verification_engine/engine.py`:

- **PASS**
  - No authoritative numeric failures
  - No unverified authoritative numeric facts
  - No entity mismatch conflicts
  - No missing required information

- **FAIL**
  - Only when a numeric threshold/comparison is provably false
  - And nothing else is uncertain/missing

- **REVIEW**
  - Everything else:
    - missing information
    - numeric unverified/conflicting
    - entity mismatch
    - or insufficient verified evidence to conclusively decide

Additionally:
- The verification engine treats numeric comparisons as authoritative (Python-driven).
- Document presence/keyword existence/counts are treated as informational and typically do not directly block PASS/FAIL (they instead influence missing/uncertainty handled via the LLM and evidence verification).

---

## 5) Parallelism model

Both planning and evaluation can run concurrently:

- Planning parallelism:
  - in `_plan_key_points` via `ThreadPoolExecutor`
- Evaluation parallelism:
  - in `_evaluate_planned_key_points` via `ThreadPoolExecutor`

Worker count:
- Controlled by:
  - `EVALUATION_MAX_WORKERS` env var or default `4`

---

## 6) Intermediate artifacts: what they contain

1. **`tender_plan_results.json`**
   - Planning results per criterion (criterion_id, plan or errors)

2. **`final_verdict.json`**
   - A list of `extraction` objects for successfully evaluated key points

3. **`evaluation_results.json`**
   - A list of `verification` objects for successfully evaluated key points

4. **`evalution.json`** (final)
   - The original `tender_output.json` with appended:
     - `verdict`
     - `summary`
     - `matched_docuements`

---

## 7) Expected input JSON shapes (minimum needed)

### 7.1 `tender_output.json`
At minimum, the pipeline expects:

- `key_points`: list of objects
  - each key point should have:
    - `key_id` OR `criterion_id` OR `id` (any one of them)
    - `point` OR `text` (the criterion wording)

Optional but used:
- `metadata` including:
  - `gem_bid_id`
  - `tender_reference_number`
  - `tender_title`

### 7.2 Company JSON (`kheria company.json`)
The extraction and verification engines assume the company JSON includes parsed documents/pages with text and metadata.

At minimum:
- `Evalution/tender_extractor` must be able to parse documents from the JSON via `adapt_company_json`.
- `Evalution/verification_engine` must be able to parse and verify against page-level text via `parse_company_json`.

(Exact schema is implemented in `Evalution/tender_extractor` and `Evalution/verification_engine` adapters/parsers.)

---

## 8) Common failure points

- **Planning failure**
  - if LLM plan validation repeatedly fails
  - key point will get `REVIEW` with a failure summary

- **Extraction/verification exception**
  - stored in per-key-point `error` during evaluation
  - key point will become `REVIEW`

- **Company JSON not parseable**
  - extraction may return empty results or errors
  - verification may skip ground truth/verification and lead to `REVIEW`

---

## 9) TL;DR: “Full view” of the pipeline

1. Read `tender_output.json` and `company_json`
2. For each key point:
   - **Plan** required evidence documents (LLM + validate/repair)
   - **Extract** evidence from bid docs (RAG + scoring + extraction)
   - **Verify** evidence against page text + run numeric logic + entity checks (Python + optional LLM)
   - Produce `verdict` + `summary` and `matched_docuements`
3. Write:
   - `tender_plan_results.json`
   - `final_verdict.json`
   - `evaluation_results.json`
   - `evalution.json` (final)
