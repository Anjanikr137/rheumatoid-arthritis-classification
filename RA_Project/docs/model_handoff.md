# Model Handoff Guide: Shared Feature & Model Development Contract

## 1. Team Ownership Matrix

This document defines the interface and protocols for the model development phase. Four machine learning models will be built independently by assigned owners using the **exact same pre-split features and test cohort**.

| Notebook | Model Architecture | Assigned Owner | Model Directory | Target Serialization Format |
| :--- | :--- | :--- | :--- | :--- |
| `05_logistic_regression.ipynb` | Logistic Regression (L1/L2 ElasticNet) | **Teammate 1 (Ali Zamaan)** | `models/logistic_regression/` | `logistic_regression_pipeline.joblib` |
| `06_random_forest.ipynb` | Random Forest Classifier | **Teammate 2 (Nakshtr Gupta)** | `models/random_forest/` | `random_forest_pipeline.joblib` |
| `07_xgboost.ipynb` | Gradient Boosted Decision Trees (XGBoost) | **Anjani Kumar (Me)** *(with team input)* | `models/xgboost/` | `xgboost_pipeline.joblib` |
| `08_ann.ipynb` | Artificial Neural Network (MLP / Deep Feedforward) | **Anjani Kumar (Me)** | `models/ann/` | `ann_model.keras` + `preprocessor.joblib` |
| `evaluation/model_comparison.ipynb`| Cross-Model Benchmark & Evaluation | **All Team Members** | `evaluation/results/` | `model_comparison_metrics.csv` |

---

## 2. Standardized Data-Loading Interface

Each teammate must load their training and testing partitions using the shared functions in `src.data_loader`.

### Standard Python Loading Snippet:
```python
import sys
from pathlib import Path

# Automatically ensure project root is in Python sys.path
root = Path("..").resolve() if Path("..").resolve().name == "RA_Project" else Path(".").resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import (
    load_train_data,
    load_test_data,
    get_feature_names,
    get_feature_metadata,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

# 1. Load feature matrices and target vectors
X_train, y_train = load_train_data(return_X_y=True)
X_test, y_test = load_test_data(return_X_y=True)

# 2. Inspect standardized properties
feature_names = get_feature_names()
print(f"X_train shape: {X_train.shape}")  # (9668, 14)
print(f"y_train shape: {y_train.shape}")  # (9668,)
print(f"X_test shape:  {X_test.shape}")   # (2417, 14)
print(f"y_test shape:  {y_test.shape}")   # (2417,)
print(f"Candidate Features ({len(feature_names)}): {feature_names}")
```

---

## 3. Guarantees & Contract Specifications

### 3.1. What `X_train` and `X_test` Contain
- Exactly **14 clinical features** in identical column order across all models:
  1. `Age` (numeric)
  2. `Gender` (categorical: 'Male', 'Female')
  3. `ESR` (numeric)
  4. `CRP` (numeric)
  5. `RF` (numeric)
  6. `Anti-CCP` (numeric)
  7. `HLA-B27` (categorical: 'Positive', 'Negative', NaN)
  8. `ANA` (categorical: 'Positive', 'Negative', NaN)
  9. `Anti-Ro` (categorical: 'Positive', 'Negative', NaN)
  10. `Anti-La` (categorical: 'Positive', 'Negative', NaN)
  11. `Anti-dsDNA` (categorical: 'Positive', 'Negative', NaN)
  12. `Anti-Sm` (categorical: 'Positive', 'Negative', NaN)
  13. `C3` (numeric)
  14. `C4` (numeric)

### 3.2. What Is Strictly Excluded from `X`
- `Disease`: Excluded to prevent 100% target leakage.
- `Target_RA`: Returned separately as `y`.
- `patient_row_id`: Positional tracking key; excluded from features.

### 3.3. Target Vector `y`
- Binary Series: `1` for Rheumatoid Arthritis, `0` for Non-RA.
- Class distribution:
  - Training: 2,278 positive (23.56%), 7,390 negative (76.44%).
  - Testing: 570 positive (23.58%), 1,847 negative (76.42%).

---

## 4. Golden Rules for Model Developers

1. **Never Modify Split Files**:
   Do not edit or overwrite `data/splits/train.csv` or `data/splits/test.csv`.

2. **Strict Preprocessing Isolation (Zero Leakage)**:
   - **Never** fit an imputer, scaler, or encoder on the entire dataset or on `X_test`.
   - Wrap your preprocessing inside a `sklearn.pipeline.Pipeline` or `sklearn.compose.ColumnTransformer`.
   - Call `.fit()` or `.fit_transform()` **only** on `X_train` (or inside inner folds during CV).
   - Call `.transform()` **only** on `X_test`.

3. **Hyperparameter Tuning via Cross-Validation**:
   - Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `X_train`.
   - Optimize for clinically relevant metrics: **ROC-AUC**, **PR-AUC (Average Precision)**, **F1-score**, and **Sensitivity (Recall at high specificity)**.

4. **Reserve `X_test` for Final Evaluation**:
   - Do not use `X_test` to tune hyperparameters or guide architecture selection.
   - Run predictions on `X_test` exactly once at final validation to report unbiased test metrics.

---

## 5. Model Serialization Standards for Future Dashboard

Future dashboard integration requires that each model is saved with its complete preprocessing pipeline so that it can ingest raw 14-feature inputs directly.

### Recommended Pattern (Scikit-Learn / XGBoost):
```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define numeric and categorical transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, NUMERIC_FEATURES),
    ('cat', categorical_transformer, CATEGORICAL_FEATURES)
])

# Full end-to-end pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', your_model)
])

# Fit on training data ONLY
full_pipeline.fit(X_train, y_train)

# Save pipeline artifact
joblib.dump(full_pipeline, "models/your_model/model_pipeline.joblib")
```

### Dashboard Loading Verification:
```python
# The dashboard can simply do:
pipeline = joblib.load("models/your_model/model_pipeline.joblib")
prediction = pipeline.predict(new_patient_df[feature_names])
probability = pipeline.predict_proba(new_patient_df[feature_names])[:, 1]
```
