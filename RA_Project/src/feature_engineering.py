"""
Feature Engineering and Quality Validation Module for Rheumatoid Arthritis Classification.

Defines reproducible transformations:
- Validation of raw schema and row count (12,085 records, 15 columns)
- Safe whitespace and categorical standardization while strictly preserving genuine missing values
- Creation of binary target `Target_RA` (1 = RA [2,848], 0 = Non-RA [9,237])
- Preservation of stable row key `patient_row_id` and clinical label `Disease` for traceability
- Definition of the standardized 14 candidate input features
"""

from pathlib import Path
from typing import Tuple, List, Union
import numpy as np
import pandas as pd

RAW_EXPECTED_COLUMNS: List[str] = [
    "Age", "Gender", "ESR", "CRP", "RF", "Anti-CCP",
    "HLA-B27", "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm",
    "C3", "C4", "Disease"
]

FEATURE_COLUMNS: List[str] = [
    "Age", "Gender", "ESR", "CRP", "RF", "Anti-CCP",
    "HLA-B27", "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm",
    "C3", "C4"
]

NUMERIC_FEATURES: List[str] = [
    "Age", "ESR", "CRP", "RF", "Anti-CCP", "C3", "C4"
]

CATEGORICAL_FEATURES: List[str] = [
    "Gender", "HLA-B27", "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm"
]

TARGET_COLUMN: str = "Target_RA"
ID_COLUMN: str = "patient_row_id"
TRACE_COLUMN: str = "Disease"

EXPECTED_TOTAL_ROWS: int = 12085
EXPECTED_RA_COUNT: int = 2848
EXPECTED_NON_RA_COUNT: int = 9237


def validate_raw_data(df: pd.DataFrame) -> bool:
    """
    Validate that raw data matches the expected schema and properties.
    Raises ValueError / AssertionError if any check fails.
    """
    if len(df) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_ROWS} rows, but found {len(df)} rows."
        )

    missing_cols = [c for c in RAW_EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw dataset: {missing_cols}")

    if "Disease" not in df.columns:
        raise ValueError("Target column 'Disease' not found in raw dataset.")

    disease_counts = df["Disease"].value_counts()
    if "Rheumatoid Arthritis" not in disease_counts:
        raise ValueError("'Rheumatoid Arthritis' class missing from 'Disease' column.")

    actual_ra = int(disease_counts["Rheumatoid Arthritis"])
    if actual_ra != EXPECTED_RA_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RA_COUNT} RA records, but found {actual_ra}."
        )

    return True


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely standardize categorical values and whitespace.

    Strict clinical rules applied:
    - Never impute missing values at this stage.
    - Never interpret missing laboratory tests as negative or zero.
    - Never drop rows with missing laboratory measurements.
    - Standardize whitespace on string columns without modifying missing (NaN) values.
    """
    clean_df = df.copy()

    for col in clean_df.select_dtypes(include=["object"]).columns:
        # Strip whitespace for non-null string entries
        clean_df[col] = clean_df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    return clean_df


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create standardized features and binary classification target.

    Output DataFrame includes:
    - `patient_row_id`: Stable integer row identifier (0 to N-1) for audit and split tracking
    - 14 Candidate Input Features (2 demographic, 6 numeric labs, 6 categorical labs)
    - `Disease`: Original multi-class clinical label (retained for traceability)
    - `Target_RA`: Binary classification target (1 = RA, 0 = Non-RA)
    """
    validate_raw_data(df)
    clean_df = standardize_data(df)

    # 1. Stable patient row identifier
    if ID_COLUMN not in clean_df.columns:
        clean_df[ID_COLUMN] = np.arange(len(clean_df), dtype=int)

    # 2. Binary target: 1 = Rheumatoid Arthritis, 0 = Non-RA
    clean_df[TARGET_COLUMN] = (
        clean_df[TRACE_COLUMN] == "Rheumatoid Arthritis"
    ).astype(int)

    # Verification of class balance
    ra_count = int((clean_df[TARGET_COLUMN] == 1).sum())
    non_ra_count = int((clean_df[TARGET_COLUMN] == 0).sum())

    assert ra_count == EXPECTED_RA_COUNT, (
        f"Target_RA=1 count mismatch: got {ra_count}, expected {EXPECTED_RA_COUNT}"
    )
    assert non_ra_count == EXPECTED_NON_RA_COUNT, (
        f"Target_RA=0 count mismatch: got {non_ra_count}, expected {EXPECTED_NON_RA_COUNT}"
    )
    assert clean_df[TARGET_COLUMN].isnull().sum() == 0, "Target_RA contains null values!"

    # 3. Order columns systematically
    ordered_cols = [ID_COLUMN] + FEATURE_COLUMNS + [TRACE_COLUMN, TARGET_COLUMN]
    return clean_df[ordered_cols]


def save_engineered_dataset(df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """Save the engineered dataset to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
