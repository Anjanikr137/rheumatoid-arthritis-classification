"""
Shared Data Loader Module for Rheumatoid Arthritis (RA) Classification.

This module provides standardized, reproducible interfaces for loading:
- Engineered dataset (full cohort)
- Stratified training set (X_train, y_train)
- Stratified test set (X_test, y_test)
- Feature metadata and column definitions

Key Guarantees:
- X feature matrices strictly contain only the 14 candidate clinical features in fixed order.
- The target vector y strictly contains `Target_RA` (1 = RA, 0 = Non-RA).
- The original diagnosis `Disease` and tracking ID `patient_row_id` are strictly excluded
  from X to prevent target leakage.
- No transformations, scalers, or encoders are pre-fitted here; model owners fit their
  own preprocessing pipelines strictly on training data.
- Paths are resolved dynamically relative to the project root.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import json
import pandas as pd

# Standardized Feature Definitions
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


def get_project_root() -> Path:
    """
    Dynamically locate the RA_Project root directory.
    Searches upward from the current file, or from current working directory.
    """
    # 1. Start from this file's parent directory
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
        if (parent / "RA_Project" / "data").exists():
            return parent / "RA_Project"

    # 2. Check current working directory
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
        if (parent / "RA_Project" / "data").exists():
            return parent / "RA_Project"

    # Fallback to local relative RA_Project
    if Path("RA_Project").exists():
        return Path("RA_Project").resolve()
    return Path(".").resolve()


def get_feature_names() -> List[str]:
    """Return the exact list of 14 model input feature names in standardized order."""
    return list(FEATURE_COLUMNS)


def get_feature_metadata() -> Dict[str, Any]:
    """
    Return comprehensive clinical and technical metadata for all 14 features,
    including clinical grouping, data type, expected range/values, and description.
    """
    return {
        "features": {
            "Age": {
                "group": "Demographic",
                "type": "numeric",
                "unit": "Years",
                "expected_range": [20, 80],
                "description": "Patient age at clinical evaluation."
            },
            "Gender": {
                "group": "Demographic",
                "type": "categorical",
                "categories": ["Male", "Female"],
                "description": "Patient biological sex."
            },
            "ESR": {
                "group": "Numeric Laboratory",
                "type": "numeric",
                "unit": "mm/hr",
                "expected_range": [0, 49],
                "description": "Erythrocyte Sedimentation Rate, non-specific systemic inflammation marker."
            },
            "CRP": {
                "group": "Numeric Laboratory",
                "type": "numeric",
                "unit": "mg/L",
                "expected_range": [0.1, 30.0],
                "description": "C-Reactive Protein, acute-phase systemic inflammatory marker."
            },
            "RF": {
                "group": "Numeric Laboratory",
                "type": "numeric",
                "unit": "IU/mL",
                "expected_range": [0, 40],
                "description": "Rheumatoid Factor autoantibody titer."
            },
            "Anti-CCP": {
                "group": "Numeric Laboratory",
                "type": "numeric",
                "unit": "U/mL",
                "expected_range": [0, 40],
                "description": "Anti-Cyclic Citrullinated Peptide antibody titer, highly specific for RA."
            },
            "HLA-B27": {
                "group": "Categorical Laboratory",
                "type": "categorical",
                "categories": ["Positive", "Negative"],
                "description": "Human Leukocyte Antigen B27 serotype, primarily associated with spondyloarthropathies."
            },
            "ANA": {
                "group": "Categorical Laboratory",
                "type": "categorical",
                "categories": ["Positive", "Negative"],
                "description": "Antinuclear Antibodies screen, prevalent in systemic autoimmune conditions."
            },
            "Anti-Ro": {
                "group": "Categorical Laboratory",
                "type": "categorical",
                "categories": ["Positive", "Negative"],
                "description": "Anti-Ro/SSA autoantibodies, associated with Sjögren's Syndrome and SLE."
            },
            "Anti-La": {
                "group": "Categorical Laboratory",
                "type": "categorical",
                "categories": ["Positive", "Negative"],
                "description": "Anti-La/SSB autoantibodies, associated with Sjögren's Syndrome."
            },
            "Anti-dsDNA": {
                "group": "Categorical Laboratory",
                "type": "categorical",
                "categories": ["Positive", "Negative"],
                "description": "Anti-double stranded DNA autoantibodies, highly specific for Systemic Lupus Erythematosus."
            },
            "Anti-Sm": {
                "group": "Categorical Laboratory",
                "type": "categorical",
                "categories": ["Positive", "Negative"],
                "description": "Anti-Smith autoantibodies, highly specific for Systemic Lupus Erythematosus."
            },
            "C3": {
                "group": "Numeric Laboratory",
                "type": "numeric",
                "unit": "mg/dL",
                "expected_range": [50, 205],
                "description": "Complement Component 3, consumed in active immune-complex-mediated autoimmune flares."
            },
            "C4": {
                "group": "Numeric Laboratory",
                "type": "numeric",
                "unit": "mg/dL",
                "expected_range": [5, 74],
                "description": "Complement Component 4, consumed in active complement pathway activation."
            },
        },
        "target": {
            "name": TARGET_COLUMN,
            "type": "binary",
            "values": {1: "Rheumatoid Arthritis", 0: "Non-RA (Other 6 cohorts)"},
            "description": "Retrospective binary classification target."
        },
        "trace_columns": {
            ID_COLUMN: "Original record row index (0-indexed) for audit and sample tracking.",
            TRACE_COLUMN: "Original multi-class diagnostic category (excluded from model input to prevent leakage)."
        }
    }


def load_engineered_data(
    return_X_y: bool = False,
    data_path: Optional[Union[str, Path]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load the full engineered dataset (12,085 records).

    Parameters:
    -----------
    return_X_y : bool, default=False
        If True, returns tuple `(X, y)` where X contains only the 14 features
        and y contains the binary Target_RA.
        If False, returns the full DataFrame including `patient_row_id`,
        `Disease`, `Target_RA`, and all features.
    data_path : str or Path, optional
        Custom path to `ra_engineered_dataset.csv`. If None, uses default project path.

    Returns:
    --------
    pd.DataFrame or Tuple[pd.DataFrame, pd.Series]
    """
    if data_path is None:
        root = get_project_root()
        data_path = root / "data" / "processed" / "ra_engineered_dataset.csv"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at: {data_path}. "
            "Please run notebook '03_feature_engineering.ipynb' first."
        )

    df = pd.read_csv(data_path)

    if return_X_y:
        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMN].copy()
        return X, y

    return df


def load_train_data(
    return_X_y: bool = True,
    data_path: Optional[Union[str, Path]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load the reproducible stratified training split (80%, 9,668 records).

    Parameters:
    -----------
    return_X_y : bool, default=True
        If True, returns tuple `(X_train, y_train)` containing only the 14 features
        in standard order and Target_RA.
        If False, returns the full train DataFrame (including `patient_row_id` and `Disease`).
    data_path : str or Path, optional
        Custom path to `train.csv`. If None, uses default project path.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series] or pd.DataFrame
    """
    if data_path is None:
        root = get_project_root()
        data_path = root / "data" / "splits" / "train.csv"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training split not found at: {data_path}. "
            "Please run notebook '04_final_validation.ipynb' first."
        )

    df = pd.read_csv(data_path)

    if return_X_y:
        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMN].copy()
        return X, y

    return df


def load_test_data(
    return_X_y: bool = True,
    data_path: Optional[Union[str, Path]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load the reproducible stratified test split (20%, 2,417 records).

    Parameters:
    -----------
    return_X_y : bool, default=True
        If True, returns tuple `(X_test, y_test)` containing only the 14 features
        in standard order and Target_RA.
        If False, returns the full test DataFrame (including `patient_row_id` and `Disease`).
    data_path : str or Path, optional
        Custom path to `test.csv`. If None, uses default project path.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series] or pd.DataFrame
    """
    if data_path is None:
        root = get_project_root()
        data_path = root / "data" / "splits" / "test.csv"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Test split not found at: {data_path}. "
            "Please run notebook '04_final_validation.ipynb' first."
        )

    df = pd.read_csv(data_path)

    if return_X_y:
        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMN].copy()
        return X, y

    return df


def load_split_metadata(
    metadata_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Load metadata describing the stratified train/test split parameters and distributions."""
    if metadata_path is None:
        root = get_project_root()
        metadata_path = root / "data" / "splits" / "split_metadata.json"
    else:
        metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Split metadata not found at: {metadata_path}. "
            "Please run notebook '04_final_validation.ipynb' first."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)
