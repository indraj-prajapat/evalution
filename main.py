from Evalution.pipeline import run_pipeline

TENDER_OUTPUT_JSON_PATH: str = "tender_output.json"
COMPANY_JSON_PATH: str = "kheria company.json"
COMPANY_NAME: str = "M S Kheria & Company"
OUTPUT_PATH: str = "evalution.json"


if __name__ == "__main__":
    run_pipeline(
        tender_output_json=TENDER_OUTPUT_JSON_PATH,
        company_json_path=COMPANY_JSON_PATH,
        company_name=COMPANY_NAME,
        output_path=OUTPUT_PATH,
    )