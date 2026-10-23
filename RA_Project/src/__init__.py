"""
Rheumatoid Arthritis (RA) Classification System - Core Source Package.

This package provides reproducible modules for dataset auditing, data quality validation,
feature engineering, and standardized data loading across all machine learning models.
"""

from .data_loader import (
    load_engineered_data,
    load_train_data,
    load_test_data,
    load_split_metadata,
    get_feature_names,
    get_feature_metadata,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    ID_COLUMN,
    TRACE_COLUMN,
)
from .feature_engineering import (
    validate_raw_data,
    standardize_data,
    create_engineered_features,
    save_engineered_dataset,
)
from .data_audit import (
    audit_dataset,
    generate_audit_report,
)

__version__ = "1.0.0"
__all__ = [
    "load_engineered_data",
    "load_train_data",
    "load_test_data",
    "load_split_metadata",
    "get_feature_names",
    "get_feature_metadata",
    "FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "TARGET_COLUMN",
    "ID_COLUMN",
    "TRACE_COLUMN",
    "validate_raw_data",
    "standardize_data",
    "create_engineered_features",
    "save_engineered_dataset",
    "audit_dataset",
    "generate_audit_report",
]
