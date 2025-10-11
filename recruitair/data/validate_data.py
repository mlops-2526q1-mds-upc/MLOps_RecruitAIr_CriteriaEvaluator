"""
Data validation script using Great Expectations for the preprocessed CV data.
"""

import sys
from typing import Any

import pandas as pd
import great_expectations as gx
from great_expectations.expectations.metadata_types import FailureSeverity

from recruitair.config.data_preprocess_config import (
    INTERIM_DATA_DIR,
    MAX_RESUME_LENGTH,
)

DATA_PATH = INTERIM_DATA_DIR / "preprocessed_cvs.jsonl"
MIN_SCORE = 0
MAX_SCORE = 10


if __name__ == "__main__":
    print(f"Validating data at: {DATA_PATH}")

    if not DATA_PATH.exists():
        print(f"❌ ERROR: Data file not found at '{DATA_PATH}'.")
        print("Please run the preprocessing pipeline first.")
        sys.exit(1)

    context = gx.get_context()

    # Create a GX Datasource to connect to our pandas DataFrame
    df = pd.read_json(DATA_PATH, lines=True)
    data_source = context.data_sources.add_pandas(name="resumes_jsonl")
    data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

    # Build a batch request to tell GX what data to validate
    batch_parameters: dict[str, Any] = {"dataframe": df}
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name="batch definition")
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    suite = context.suites.add(
        gx.core.expectation_suite.ExpectationSuite(name="processed_cv_suite")
    )

    # Expectation 1: Resume must exist
    print(f"  - Expecting all jsons to have a resume.")
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(
            meta={"name": "Resume must exist"},
            column="resume",
            severity="critical",
        )
    )
    # Expectation 2: Resume length should not exceed MAX_RESUME_LENGTH
    print(f"  - Expecting 'resume' length between 1 and {MAX_RESUME_LENGTH} characters.")
    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToBeBetween(
            meta={"name": "Max length of resumes"},
            column="resume",
            min_value=1,
            max_value=MAX_RESUME_LENGTH,
            severity="warning",
        )
    )
    # Expectation 3: Score must be a numeric type
    print("  - Expecting 'score' column to contain numbers (integer or float).")
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInTypeList(
            meta={"name": "Score types"},
            column="score",
            type_list=["int", "float", "int64", "float64"],
        )
    )
    # Expectation 4: Score must be within the defined range
    print(f"  - Expecting 'score' values to be between {MIN_SCORE} and {MAX_SCORE}.")
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            meta={"name": "Score between range"},
            column="score",
            min_value=MIN_SCORE,
            max_value=MAX_SCORE,
        )
    )
    # Expectation 5: Objects must have a criteria
    print(f"  - Expecting objects to have an extracted criteria.")
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(
            meta={"name": "Criteria must exist"},
            column="criteria",
        )
    )
    # Expectation 5: Criteria length must be between 1 and 100.
    print(f"  - Expecting 'criteria' length between 1 and 100 characters.")
    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToBeBetween(
            meta={"name": "Max length of criteria"},
            column="criteria",
            min_value=1,
            max_value=100,
            severity="warning",
        )
    )

    print("\nRunning validation...")

    # Create Validation Definition.
    validation_definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name="validation definition",
            data=batch_definition,
            suite=suite,
        )
    )

    # Create Checkpoint, run Checkpoint, and capture result.
    checkpoint = context.checkpoints.add(
        gx.checkpoint.checkpoint.Checkpoint(
            name="checkpoint", validation_definitions=[validation_definition]
        )
    )

    checkpoint_result = checkpoint.run(batch_parameters=batch_parameters)

    print("")
    success = True
    for _, result_obj in checkpoint_result.run_results.items():
        for res in result_obj.results:
            if res.success:
                print(f"-✅ {res.expectation_config.meta['name']}")
                continue
            print(f"-❌ {res.expectation_config.meta['name']}")
            print(f"Analyzed rows: {res.result['element_count']}")
            print(
                f"Percentage of non-compliant data: {round(res.result['unexpected_percent'], 2)}%"
            )
            print(f"Percentage of empty data: {round(res.result['missing_percent'], 2)}%")
            # print(f"Sample of failing data: {res.result['partial_unexpected_list']}")
            print("")
            if res.expectation_config.severity == FailureSeverity.CRITICAL:
                success = False
    print("")

    # Check the results
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
