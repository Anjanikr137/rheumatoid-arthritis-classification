# Rheumatoid Arthritis (RA) Clinical Classification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Phase 1 Validated](https://img.shields.io/badge/status-phase%201%20completed-success.svg)]()
[![Reproducible Splits](https://img.shields.io/badge/splits-stratified%2080%3A20-orange.svg)]()

A clean, modular, and reproducible machine learning pipeline for classifying clinical patient records as **Rheumatoid Arthritis (RA)** or **Non-RA** using standardized demographic and clinical laboratory features.

---

## 1. Project Overview & Clinical Context

Rheumatoid Arthritis (RA) is a systemic chronic inflammatory autoimmune disorder characterized by persistent synovitis, joint destruction, and extra-articular manifestations. Early and precise diagnostic discrimination is crucial to initiate disease-modifying anti-rheumatic drugs (DMARDs) and prevent irreversible joint damage.

This project investigates retrospective classification of recorded patient laboratory encounters from an autoimmune cohort (12,085 records) across seven differential diagnostic cohorts:
1. **Rheumatoid Arthritis (RA)** — Positive Class (`Target_RA = 1`, 2,848 records)
2. **Ankylosing Spondylitis** — Negative Class (`Target_RA = 0`, 2,127 records)
3. **Sjögren's Syndrome** — Negative Class (`Target_RA = 0`, 1,852 records)
4. **Psoriatic Arthritis** — Negative Class (`Target_RA = 0`, 1,783 records)
5. **Normal (Control)** — Negative Class (`Target_RA = 0`, 1,604 records)
6. **Systemic Lupus Erythematosus (SLE)** — Negative Class (`Target_RA = 0`, 1,355 records)
7. **Reactive Arthritis** — Negative Class (`Target_RA = 0`, 516 records)

> **Important Clinical Boundary**: This project performs **retrospective classification of recorded diagnostic labels**. It does not claim or model prospective future-onset disease prediction or pre-clinical disease emergence.

---

## 2. Team Ownership & Responsibilities

Four separate machine learning models will be developed in dedicated notebooks by assigned owners, ingesting the **exact same pre-split features and test cohort**:

| Notebook | Model Architecture | Assigned Owner | Target Artifact Path | Target Format |
| :--- | :--- | :--- | :--- | :--- |
| [`05_logistic_regression.ipynb`](file:///d:/PRJ%201%20Sem%205/RA_Project/notebooks/05_logistic_regression.ipynb) | Regularized Logistic Regression (L1/L2/ElasticNet) | **Teammate 1 (Ali Zamaan)** | `models/logistic_regression/` | `.joblib` |
| [`06_random_forest.ipynb`](file:///d:/PRJ%201%20Sem%205/RA_Project/notebooks/06_random_forest.ipynb) | Random Forest Ensemble | **Teammate 2 (Nakshtr Gupta)** | `models/random_forest/` | `.joblib` |
| [`07_xgboost.ipynb`](file:///d:/PRJ%201%20Sem%205/RA_Project/notebooks/07_xgboost.ipynb) | Extreme Gradient Boosting (XGBoost) | **Anjani Kumar ** | `models/xgboost/` | `.joblib` / JSON |
| [`08_ann.ipynb`](file:///d:/PRJ%201%20Sem%205/RA_Project/notebooks/08_ann.ipynb) | Artificial Neural Network (MLP / Deep Feedforward) | **Anjani Kumar ** 
** | `models/ann/` | `.keras` + `.joblib` |
| [`evaluation/model_comparison.ipynb`](file:///d:/PRJ%201%20Sem%205/RA_Project/evaluation/model_comparison.ipynb) | Multi-Model Benchmark & Evaluation | **All Team Members** | `evaluation/results/` | `.csv` / plots |

---

## 3. Repository Directory Structure

```text
RA_Project/
│
├── data/
│   ├── raw/
│   │   └── Rheumatic and Autoimmune Disease Dataset.xlsx    # Immutable original workbook (12,085 rows)
│   ├── processed/
│   │   └── ra_engineered_dataset.csv                        # Processed cohort with Target_RA (12,085 rows)
│   ├── splits/
│   │   ├── train.csv                                        # Stratified 80% train split (9,668 rows)
│   │   ├── test.csv                                         # Stratified 20% test split (2,417 rows)
│   │   └── split_metadata.json                              # Split parameters, distributions, zero-overlap audit
│   └── reports/
│       ├── dataset_audit_report.json                        # Machine-readable audit metrics
│       ├── dataset_audit_summary.md                         # Human-readable audit findings
│       └── data_quality_report.json                         # Clinical validity rules and decisions
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb                               # Comprehensive workbook audit & discovery
│   ├── 02_data_quality.ipynb                                # Clinical bounds & missingness policy
│   ├── 03_feature_engineering.ipynb                         # Target_RA construction & candidate features
│   ├── 04_final_validation.ipynb                            # Stratified partitioning & contract verification
│   ├── 05_logistic_regression.ipynb                         # Model notebook (Teammate 1 - Ali Zamaan)
│   ├── 06_random_forest.ipynb                               # Model notebook (Teammate 2 - Nakshtr Gupta)
│   ├── 07_xgboost.ipynb                                     # Model notebook (Anjani Kumar - Me)
│   └── 08_ann.ipynb                                         # Model notebook (Anjani Kumar - Me)
│
├── src/
│   ├── __init__.py                                          # Package exports
│   ├── data_audit.py                                        # Auditing, block detection, and report utilities
│   ├── feature_engineering.py                               # Validation, target mapping, and engineering
│   └── data_loader.py                                       # Shared data loading interface for all models
│
├── models/
│   ├── logistic_regression/                                 # Serialized pipeline placeholder
│   ├── random_forest/                                       # Serialized pipeline placeholder
│   ├── xgboost/                                             # Serialized pipeline placeholder
│   └── ann/                                                 # Serialized Keras/pipeline placeholder
│
├── evaluation/
│   ├── results/                                             # Comparative metrics, ROC/PR curves
│   └── model_comparison.ipynb                               # Comparative benchmark notebook
│
├── docs/
│   ├── data_dictionary.md                                   # Clinical definitions, units, ranges
│   ├── feature_engineering_report.md                        # Architecture, leakage mitigation, rationale
│   └── model_handoff.md                                     # Teammate instructions and coding standards
│
├── requirements.txt                                         # Python package dependencies
└── README.md                                                # Project documentation
```

---

## 4. Key Forensic Audit Discoveries

During Phase 1 auditing (`01_dataset_audit.ipynb`), two fundamental findings were established:

### 1. Monolithic Diagnosis Block Ordering
The original Excel workbook rows are **not randomly distributed**. Patient records are grouped into **7 contiguous blocks** by diagnosis:
- Rows 0 – 2,847: **Rheumatoid Arthritis** (100% positive)
- Rows 2,848 – 3,363: **Reactive Arthritis**
- Rows 3,364 – 5,490: **Ankylosing Spondylitis**
- Rows 5,491 – 7,342: **Sjögren's Syndrome**
- Rows 7,343 – 8,697: **Systemic Lupus Erythematosus**
- Rows 8,698 – 10,301: **Normal**
- Rows 10,302 – 12,084: **Psoriatic Arthritis**

*Mitigation*: A non-shuffled split would result in 100% target leakage or a test set containing 0 positive cases. We enforce **stratified random splitting with shuffling under a fixed random seed (`seed=42`)**, guaranteeing representative class ratios in both splits.

### 2. High Laboratory Missingness & Prohibition of `dropna()`
Only **452 out of 12,085 patients (3.74%)** have complete laboratory records across all tests. A standard `dropna()` policy would discard **96.26% of the cohort**. Furthermore, missing laboratory tests reflect selective clinical ordering, not negative results.
*Policy*: Genuine `NaN` values are preserved. Model developers must handle missing values within training pipelines (e.g., median/mode imputation or iterative imputation fitted strictly on `X_train`).

---

## 5. Standardized Developer Contract (How to Access Shared Features)

Each model developer loads their training and testing sets using the shared functions in `src.data_loader`:

```python
import sys
from pathlib import Path

# Add project root to path
root = Path("..").resolve() if Path("..").resolve().name == "RA_Project" else Path(".").resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import (
    load_train_data,
    load_test_data,
    get_feature_names,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

# 1. Load standardized features and targets
X_train, y_train = load_train_data(return_X_y=True)
X_test, y_test = load_test_data(return_X_y=True)

print(f"X_train: {X_train.shape}")  # (9668, 14)
print(f"y_train: {y_train.shape}")  # (9668,) -> 2,278 positive (23.56%)
print(f"X_test:  {X_test.shape}")   # (2417, 14)
print(f"y_test:  {y_test.shape}")   # (2417,) -> 570 positive (23.58%)
```

### The 14 Standardized Input Features:
- **Demographics (2)**: `Age`, `Gender`
- **Numeric Labs (6)**: `ESR`, `CRP`, `RF`, `Anti-CCP`, `C3`, `C4`
- **Categorical Labs (6)**: `HLA-B27`, `ANA`, `Anti-Ro`, `Anti-La`, `Anti-dsDNA`, `Anti-Sm`

> **Leakage Prevention**: `Disease` and `Target_RA` (and surrogate key `patient_row_id`) are strictly excluded from `X_train` and `X_test`.

---

## 6. Model Training & Export Standards

1. **Strict Train-Only Fitting**: Fit all imputers, scalers, and encoders **strictly on `X_train`**. Never call `.fit()` on `X_test`.
2. **Cross-Validation**: Tune hyperparameters using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `X_train`.
3. **Unbiased Testing**: Run predictions on `X_test` only at final evaluation.
4. **Dashboard-Ready Export**: Save complete end-to-end pipelines (preprocessing + classifier) into `models/<model_name>/` using `joblib` or native formats so a future dashboard can predict without retraining.

---

## 7. Reproduction Instructions

To reproduce Phase 1 data preparation from scratch:

```bash
# 1. Clone repository & install dependencies
pip install -r requirements.txt

# 2. Execute preparation notebooks sequentially
jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_audit.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_quality.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_final_validation.ipynb
```

