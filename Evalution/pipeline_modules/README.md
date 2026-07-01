# Modular Evaluation Pipeline

This is the refactored evaluation pipeline with a modular structure, small modules, and comprehensive debug support.

## Features

- **Modular Structure**: Code is split into small, focused modules for better maintainability
- **Debug Support**: Comprehensive logging of ALL inputs, outputs, and intermediate steps when `debug=True`
- **Environment Configuration**: All API keys and model names read from `.env` file only - NO hardcoded values
- **4 Required Inputs**: Same interface as original - `tender_output_json`, `company_json_path`, `company_name`, `output_path`
- **Exact Functionality**: 100% feature parity with original pipeline

## Module Structure

```
Evalution/
├── pipeline.py                    # Main refactored pipeline (modular structure)
├── pipeline_modules/              # Supporting module components
│   ├── __init__.py                # Package exports
│   ├── input_loader.py            # Input loading and validation
│   ├── config_builder.py          # Configuration building
│   ├── planner_module.py          # Key point planning
│   ├── evaluator_module.py        # Key point evaluation
│   ├── output_builder.py          # Final output assembly
│   └── utils.py                   # Utility functions
└── debug_utils/                   # Debug utilities
    ├── __init__.py                # Package exports
    └── logger.py                  # Debug logger implementation
```

## Usage

### Basic Usage (4 Required Inputs)

```python
from Evalution.pipeline import run_pipeline

result = run_pipeline(
    tender_output_json="tender_output.json",  # or dict or JSON string
    company_json_path="company.json",
    company_name="My Company Name",
    output_path="evaluation_result.json",
)
```

### With Debug Enabled (Recommended for Troubleshooting)

```python
result = run_pipeline(
    tender_output_json="tender_output.json",
    company_json_path="company.json",
    company_name="My Company Name",
    output_path="evaluation_result.json",
    debug=True,              # Enable debug logging - saves EVERYTHING
    debug_save_dir="./logs", # Optional: custom debug save directory
)
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tender_output_json` | dict/str/Path | ✅ Yes | - | Tender output (dict, JSON string, or file path) |
| `company_json_path` | str | ✅ Yes | - | Path to company JSON file |
| `company_name` | str | ✅ Yes | - | Human-readable company name |
| `output_path` | str | ✅ Yes | - | Output file path for results |
| `max_workers` | int | ❌ No | None | Parallel workers (uses env var or 4) |
| `debug` | bool | ❌ No | False | Enable comprehensive debug logging |
| `debug_save_dir` | str | ❌ No | None | Custom debug directory (default: ./debug_logs) |

## Environment Variables (.env file)

Create a `.env` file in your workspace root:

```env
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EVALUATION_MAX_WORKERS=4
```

**IMPORTANT**: The pipeline reads ALL configuration from environment variables. There are NO hardcoded API keys or model names anywhere in the code.

## Debug Output

When `debug=True`, the pipeline saves detailed logs showing EVERY step's input/output:

```
debug_logs/
├── debug_summary.json           # Complete summary of all steps
└── steps/                       # Individual step logs with timestamps
    ├── load_tender_output_20250701_120000_123456.json
    ├── validate_key_points_20250701_120001_234567.json
    ├── build_planner_config_20250701_120002_345678.json
    ├── build_planner_tender_20250701_120003_345679.json
    ├── plan_key_points_20250701_120004_456789.json
    ├── evaluate_key_points_20250701_120005_567890.json
    └── build_final_output_20250701_120006_678901.json
```

Each step log contains:
- `timestamp`: Exact execution time
- `step_name`: Which pipeline step executed
- `input`: Full input data to that step
- `output`: Full output data from that step
- `metadata`: Additional context
- `error`: Error message if step failed

This allows you to trace exactly where things go wrong!

## Pipeline Steps (What Gets Logged)

1. **load_tender_output**: Load tender output JSON
2. **validate_key_points**: Validate key point structure
3. **load_company_data**: Load company JSON
4. **build_planner_config**: Build LLM configuration from .env
5. **build_planner_tender**: Format tender for planner
6. **plan_key_points**: Plan each criterion with LLM
7. **evaluate_key_points**: Extract → Verify → Evaluate each criterion
8. **build_final_output**: Assemble final results with verdicts

## How Debug Helps You Find Issues

With `debug=True`, you can check:

1. **Input Validation**: See exactly what was loaded and if any key points were malformed
2. **Configuration**: Verify API keys and models are correctly loaded from .env
3. **Planning**: Check what criteria were sent to the LLM and what plans were returned
4. **Evaluation**: Trace extraction → verification → evaluation for each criterion
5. **Errors**: Get full error context including inputs that caused failures

Example debug workflow:
```python
result = run_pipeline(..., debug=True)

# If something fails, check:
# 1. ./debug_logs/debug_summary.json - overview of all steps
# 2. ./debug_logs/steps/*.json - detailed step-by-step logs
# 3. Look for "error" fields in any step log
```

## Running Directly

```bash
# Run with debug enabled by default
python -m Evalution.pipeline
```

This executes the pipeline with `debug=True` automatically.

## Testing

```bash
# Test imports work
python -c "from Evalution.pipeline import run_pipeline; print('✓ Imports OK')"

# Check function signature
python -c "from Evalution.pipeline import run_pipeline; import inspect; print(inspect.signature(run_pipeline))"

# Run with debug
python -m Evalution.pipeline
```

## What Changed from Original

The original monolithic `pipeline.py` has been refactored into:

1. **Main pipeline function** (`run_pipeline`) - Now with modular structure and debug support
2. **Helper modules** - Each responsibility separated into small files
3. **Debug utilities** - Comprehensive logging system

**Functionality remains EXACTLY the same** - only the code structure changed to be more maintainable and debuggable.
