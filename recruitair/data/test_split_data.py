import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from recruitair.data.split_data import split_data


@pytest.fixture
def preprocessed_data_file(tmp_path: Path) -> Path:
    """
    Creates a dummy preprocessed JSONL file for testing.
    Returns the path to the directory containing the file.
    """
    interim_dir = tmp_path / "data" / "interim"
    interim_dir.mkdir(parents=True)
    jsonl_path = interim_dir / "preprocessed_cvs.jsonl"

    # Create a dummy dataset with 100 rows
    dummy_data = [{"id": i, "text": f"This is item {i}"} for i in range(100)]
    with jsonl_path.open("w") as f:
        for item in dummy_data:
            f.write(json.dumps(item) + "\n")

    return interim_dir


def test_split_data(preprocessed_data_file: Path, tmp_path: Path):
    """
    Test the split_data function to ensure it correctly splits data and saves the files.
    """
    processed_dir = tmp_path / "data" / "processed"

    total_rows = 100
    train_split_ratio = 0.7
    validation_split_ratio = 0.15

    expected_train_rows = int(total_rows * train_split_ratio)  # 70

    remaining_rows = total_rows - expected_train_rows  # 30
    validation_on_remaining_ratio = validation_split_ratio / (1 - train_split_ratio)  # 0.15 / 0.3 = 0.5
    expected_validation_rows = int(remaining_rows * validation_on_remaining_ratio)  # 15

    expected_test_rows = remaining_rows - expected_validation_rows  # 15

    with patch("recruitair.data.split_data.INTERIM_DATA_DIR", preprocessed_data_file), \
            patch("recruitair.data.split_data.PROCESSED_DATA_DIR", processed_dir), \
            patch("recruitair.data.split_data.TRAIN_SPLIT", train_split_ratio), \
            patch("recruitair.data.split_data.VALIDATION_SPLIT", validation_split_ratio), \
            patch("recruitair.data.split_data.SEED", 42):
        split_data()

    assert processed_dir.exists()
    train_file = processed_dir / "train.jsonl"
    validation_file = processed_dir / "validation.jsonl"
    test_file = processed_dir / "test.jsonl"

    assert train_file.exists()
    assert validation_file.exists()
    assert test_file.exists()

    train_df = pd.read_json(train_file, lines=True)
    validation_df = pd.read_json(validation_file, lines=True)
    test_df = pd.read_json(test_file, lines=True)

    assert len(train_df) == expected_train_rows
    assert len(validation_df) == expected_validation_rows
    assert len(test_df) == expected_test_rows
    assert len(train_df) + len(validation_df) + len(test_df) == total_rows

    original_ids = set(range(total_rows))
    train_ids = set(train_df["id"])
    validation_ids = set(validation_df["id"])
    test_ids = set(test_df["id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == original_ids
