"""Functions to split pre-processed data."""

import pandas as pd
from sklearn.model_selection import train_test_split

from recruitair.config.data_split_config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


def split_data() -> None:
    """
    Split the cleaned data into train, validation, and test sets
    and save them to the 'data/processed' directory.

    Input: INTERIM_DATA_DIR / "preprocessed_cvs.jsonl"
    Output: JSONL files in PROCESSED_DATA_DIR:
        - train.jsonl
        - validation.jsonl
        - test.jsonl
    """
    # Leer el JSON (si es JSONL usar lines=True, si es JSON normal quitar lines=True)
    df = pd.read_json(INTERIM_DATA_DIR / "preprocessed_cvs.jsonl", lines=True)

    # First split into train and remaining
    train_df, remaining_df = train_test_split(df, train_size=TRAIN_SPLIT, random_state=SEED, shuffle=True)

    # Compute validation proportion relative to remaining
    validation_ratio = VALIDATION_SPLIT / (1 - TRAIN_SPLIT)
    validation_df, test_df = train_test_split(
        remaining_df, train_size=validation_ratio, random_state=SEED, shuffle=True
    )

    # Create parent folder if not exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save to JSONL
    train_df.to_json(PROCESSED_DATA_DIR / "train.jsonl", orient="records", lines=True, force_ascii=False)
    validation_df.to_json(PROCESSED_DATA_DIR / "validation.jsonl", orient="records", lines=True, force_ascii=False)
    test_df.to_json(PROCESSED_DATA_DIR / "test.jsonl", orient="records", lines=True, force_ascii=False)

    print("✅ Split completed:")
    print(f" - Train: {len(train_df)} rows -> {PROCESSED_DATA_DIR / 'train.jsonl'}")
    print(f" - Validation: {len(validation_df)} rows -> {PROCESSED_DATA_DIR / 'validation.jsonl'}")
    print(f" - Test: {len(test_df)} rows -> {PROCESSED_DATA_DIR / 'test.jsonl'}")


if __name__ == "__main__":
    split_data()
