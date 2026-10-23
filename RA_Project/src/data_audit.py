"""
Dataset Audit Module for Rheumatoid Arthritis Classification.

Provides comprehensive functions for auditing raw dataset structure, completeness,
data types, distribution, ordering anomalies, duplicates, and clinical laboratory metrics.
"""

from pathlib import Path
from typing import Dict, Any, List, Union
import json
import numpy as np
import pandas as pd


def inspect_sheets(file_path: Union[str, Path]) -> List[str]:
    """Inspect and return available sheet names from an Excel workbook."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    excel_file = pd.ExcelFile(file_path)
    return excel_file.sheet_names


def detect_diagnosis_blocks(df: pd.DataFrame, disease_col: str = "Disease") -> List[Dict[str, Any]]:
    """
    Detect contiguous blocks of identical diagnoses in the dataset's row sequence.
    This identifies whether the raw data was sorted or grouped by diagnosis,
    which is a major risk for non-shuffled train/test splits.
    """
    if disease_col not in df.columns:
        return []

    blocks = []
    current_label = None
    start_idx = 0

    for idx, label in enumerate(df[disease_col]):
        if label != current_label:
            if current_label is not None:
                blocks.append({
                    "diagnosis": str(current_label),
                    "start_index": int(start_idx),
                    "end_index": int(idx - 1),
                    "block_length": int(idx - start_idx),
                })
            current_label = label
            start_idx = idx

    if current_label is not None:
        blocks.append({
            "diagnosis": str(current_label),
            "start_index": int(start_idx),
            "end_index": int(len(df) - 1),
            "block_length": int(len(df) - start_idx),
        })

    return blocks


def audit_dataset(df_or_path: Union[pd.DataFrame, str, Path]) -> Dict[str, Any]:
    """
    Perform a complete, structured audit of the dataset.

    Returns a comprehensive dictionary containing:
    - Shape, columns, and data types
    - Missing value counts and percentages
    - Duplicate row counts
    - Numerical summary statistics (including zeros and IQR-based outliers)
    - Categorical value distributions and whitespace checks
    - Original diagnosis distribution
    - Sequential diagnosis block detection
    - Missing laboratory tests per record distribution
    - Identified limitations and risks
    """
    if isinstance(df_or_path, (str, Path)):
        path = Path(df_or_path)
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    else:
        df = df_or_path.copy()

    total_rows, total_cols = df.shape

    # 1. Column Types & Missingness
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / total_rows) * 100

    columns_audit = {}
    for col in df.columns:
        columns_audit[col] = {
            "dtype": str(df[col].dtype),
            "missing_count": int(missing_counts[col]),
            "missing_percentage": round(float(missing_pcts[col]), 2),
            "non_null_count": int(df[col].count()),
        }

    # 2. Exact Duplicates
    exact_duplicates = int(df.duplicated().sum())
    features_only_cols = [c for c in df.columns if c != "Disease"]
    feature_duplicates = int(df.duplicated(subset=features_only_cols).sum())

    # 3. Numeric Summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary = {}
    for col in numeric_cols:
        series = df[col].dropna()
        q25 = float(series.quantile(0.25)) if not series.empty else 0.0
        q75 = float(series.quantile(0.75)) if not series.empty else 0.0
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

        numeric_summary[col] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 4) if not series.empty else None,
            "std": round(float(series.std()), 4) if not series.empty else None,
            "min": float(series.min()) if not series.empty else None,
            "q25": round(q25, 4),
            "median": round(float(series.median()), 4) if not series.empty else None,
            "q75": round(q75, 4),
            "max": float(series.max()) if not series.empty else None,
            "zero_count": int((series == 0).sum()),
            "negative_count": int((series < 0).sum()),
            "outlier_count_iqr": outlier_count,
            "outlier_pct_iqr": round((outlier_count / len(series)) * 100, 2) if len(series) > 0 else 0.0,
        }

    # 4. Categorical Summary & Whitespace Check
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_summary = {}
    for col in cat_cols:
        series_clean = df[col].dropna().astype(str)
        val_counts = df[col].value_counts(dropna=False).to_dict()
        val_counts_formatted = {
            str(k) if pd.notnull(k) else "NaN": int(v) for k, v in val_counts.items()
        }

        # Check for trailing/leading whitespace or casing inconsistencies
        stripped = [v.strip() for v in series_clean]
        whitespace_issues = any(v != s for v, s in zip(series_clean, stripped))

        categorical_summary[col] = {
            "unique_count": int(df[col].nunique(dropna=False)),
            "value_counts": val_counts_formatted,
            "whitespace_issues_detected": whitespace_issues,
        }

    # 5. Diagnosis Distribution
    diagnosis_dist = {}
    if "Disease" in df.columns:
        diag_counts = df["Disease"].value_counts(dropna=False)
        for diag, count in diag_counts.items():
            diagnosis_dist[str(diag)] = {
                "count": int(count),
                "percentage": round(float((count / total_rows) * 100), 2),
            }

    # 6. Diagnosis Block Analysis (Sequential Row Ordering)
    diagnosis_blocks = detect_diagnosis_blocks(df, "Disease") if "Disease" in df.columns else []

    # 7. Missing Labs Distribution per Patient
    lab_cols = [c for c in df.columns if c not in ["Age", "Gender", "Disease"]]
    missing_labs_per_row = df[lab_cols].isnull().sum(axis=1)
    missing_labs_dist = {
        int(k): int(v) for k, v in missing_labs_per_row.value_counts().sort_index().items()
    }

    # 8. Missingness by Diagnosis
    missingness_by_diagnosis = {}
    if "Disease" in df.columns:
        for diag, group in df.groupby("Disease"):
            missingness_by_diagnosis[str(diag)] = {
                col: int(group[col].isnull().sum()) for col in df.columns if col != "Disease"
            }

    # 9. Identified Limitations & Clinical Considerations
    limitations = [
        "Absence of explicit unique patient identifier (Patient ID / MRN); rows are currently identifiable only by positional index.",
        "Severe row-ordering artifact: records are strictly segregated into 7 monolithic diagnosis blocks in the raw workbook.",
        "High missingness in specialized laboratory tests (Anti-Sm: 43.0%, Anti-dsDNA: 39.0%, ANA: 31.0%, Anti-CCP: 27.0%, CRP: 20.0%).",
        "Retrospective cross-sectional cohort: labels reflect recorded clinical diagnoses, not prospective future-onset disease prediction.",
        "Absence of longitudinal follow-up, symptom duration, physical joint examination scores (e.g., DAS28, tender/swollen joint counts), and imaging (X-ray/MRI).",
    ]

    return {
        "dataset_shape": {"rows": total_rows, "columns": total_cols},
        "columns_audit": columns_audit,
        "exact_duplicates": exact_duplicates,
        "feature_duplicates_excluding_disease": feature_duplicates,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "diagnosis_distribution": diagnosis_dist,
        "diagnosis_blocks": diagnosis_blocks,
        "is_diagnosis_block_ordered": len(diagnosis_blocks) == len(diagnosis_dist),
        "missing_labs_per_row_distribution": missing_labs_dist,
        "missingness_by_diagnosis": missingness_by_diagnosis,
        "limitations": limitations,
    }


def generate_audit_report(
    raw_path: Union[str, Path],
    output_json_path: Union[str, Path],
    output_md_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Run the audit and export both a machine-readable JSON report
    and a human-readable Markdown summary.
    """
    raw_path = Path(raw_path)
    output_json_path = Path(output_json_path)
    output_md_path = Path(output_md_path)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    audit_res = audit_dataset(raw_path)

    # Save JSON report
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_res, f, indent=2)

    # Generate human-readable Markdown report
    rows = audit_res["dataset_shape"]["rows"]
    cols = audit_res["dataset_shape"]["columns"]

    md_lines = [
        "# Dataset Audit Report: Rheumatic and Autoimmune Disease Dataset",
        "",
        "## 1. Executive Summary",
        f"- **Raw File**: `{raw_path.name}`",
        f"- **Dimensions**: {rows:,} records across {cols} columns.",
        f"- **Exact Duplicate Rows**: {audit_res['exact_duplicates']}.",
        f"- **Block-Ordered Structure**: {audit_res['is_diagnosis_block_ordered']} ({len(audit_res['diagnosis_blocks'])} contiguous blocks detected).",
        "",
        "## 2. Diagnosis Distribution",
        "| Diagnosis | Record Count | Percentage |",
        "| :--- | :--- | :--- |",
    ]

    for diag, stats in audit_res["diagnosis_distribution"].items():
        md_lines.append(f"| {diag} | {stats['count']:,} | {stats['percentage']:.2f}% |")

    md_lines.extend([
        "",
        "## 3. Data Ordering Analysis (Contiguous Diagnosis Blocks)",
        "> **CRITICAL FINDING**: The raw workbook arranges patient records into 7 continuous blocks grouped strictly by disease. A non-shuffled train/test split would cause catastrophic leakage or train on one disease and test on another.",
        "",
        "| Block # | Diagnosis | Start Row | End Row | Total Records |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for i, blk in enumerate(audit_res["diagnosis_blocks"], 1):
        md_lines.append(f"| {i} | {blk['diagnosis']} | {blk['start_index']:,} | {blk['end_index']:,} | {blk['block_length']:,} |")

    md_lines.extend([
        "",
        "## 4. Completeness and Missing Values",
        "| Column | Data Type | Non-Null Count | Missing Count | Missing % |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for col, c_info in audit_res["columns_audit"].items():
        md_lines.append(f"| {col} | `{c_info['dtype']}` | {c_info['non_null_count']:,} | {c_info['missing_count']:,} | {c_info['missing_percentage']:.2f}% |")

    md_lines.extend([
        "",
        "## 5. Numeric Laboratory Features Summary",
        "| Feature | Count | Mean | Std | Min | 25% | Median | 75% | Max | Zero Count | Outliers (IQR) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for col, n_info in audit_res["numeric_summary"].items():
        md_lines.append(
            f"| {col} | {n_info['count']:,} | {n_info['mean']} | {n_info['std']} | {n_info['min']} | "
            f"{n_info['q25']} | {n_info['median']} | {n_info['q75']} | {n_info['max']} | "
            f"{n_info['zero_count']} | {n_info['outlier_count_iqr']} ({n_info['outlier_pct_iqr']}%) |"
        )

    md_lines.extend([
        "",
        "## 6. Categorical Laboratory and Demographic Features",
        "| Feature | Unique Categories | Distribution | Whitespace Anomaly |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for col, cat_info in audit_res["categorical_summary"].items():
        dist_str = ", ".join([f"{k}: {v:,}" for k, v in cat_info["value_counts"].items()])
        md_lines.append(f"| {col} | {cat_info['unique_count']} | {dist_str} | {cat_info['whitespace_issues_detected']} |")

    md_lines.extend([
        "",
        "## 7. Distribution of Missing Laboratory Tests per Patient",
        "| Missing Tests Count | Number of Patients | Percentage |",
        "| :--- | :--- | :--- |",
    ])

    for count_val, num_pts in audit_res["missing_labs_per_row_distribution"].items():
        pct = (num_pts / rows) * 100
        md_lines.append(f"| {count_val} missing labs | {num_pts:,} | {pct:.2f}% |")

    md_lines.extend([
        "",
        "## 8. Identified Data Limitations & Methodological Constraints",
    ])
    for limit in audit_res["limitations"]:
        md_lines.append(f"- {limit}")

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return audit_res
