# Feature Engineering & Data Preparation Report

## 1. Executive Summary

This report documents the architectural, clinical, and data engineering decisions implemented in transforming the raw `Rheumatic and Autoimmune Disease Dataset.xlsx` workbook into a standardized, leak-free, reproducible machine learning dataset (`data/processed/ra_engineered_dataset.csv`) and shared training/testing partitions (`data/splits/`).

The primary task is **binary retrospective classification of Rheumatoid Arthritis (RA)** against six clinically relevant comparison cohorts (Normal controls, Ankylosing Spondylitis, Sjögren's Syndrome, Psoriatic Arthritis, Systemic Lupus Erythematosus, and Reactive Arthritis).

---

## 2. Target Engineering

### 2.1. Formulation & Clinical Context
The raw dataset contains a multi-class `Disease` column with 7 clinical diagnostic categories. For the RA Classification System, the problem is formulated as a binary classification task:

$$\text{Target\_RA} = \begin{cases} 1 & \text{if Patient Diagnosis} = \text{'Rheumatoid Arthritis'} \\ 0 & \text{if Patient Diagnosis} \in \{\text{All 6 other categories}\} \end{cases}$$

### 2.2. Quantitative Target Distribution
Verification across the 12,085 cohort records confirms the exact distribution:

- **Positive Class (`Target_RA = 1`)**: **2,848 records (23.57%)**
- **Negative Class (`Target_RA = 0`)**: **9,237 records (76.43%)**
  - Ankylosing Spondylitis: 2,127 records (17.60%)
  - Sjögren's Syndrome: 1,852 records (15.32%)
  - Psoriatic Arthritis: 1,783 records (14.75%)
  - Normal (Non-autoimmune): 1,604 records (13.27%)
  - Systemic Lupus Erythematosus (SLE): 1,355 records (11.21%)
  - Reactive Arthritis: 516 records (4.27%)
- **Total Records**: **12,085 records (100.00%)**

---

## 3. Candidate Input Feature Selection

Fourteen (14) clinical candidate features were curated to capture demographic characteristics, acute-phase systemic inflammation, autoantibody serology, and complement pathway activation:

```
[Candidate Feature Set: 14 Features]
├── Demographic Features (2)
│   ├── Age (Numeric, years: 20 - 80)
│   └── Gender (Categorical: Male, Female)
│
├── Numeric Laboratory Biomarkers (6)
│   ├── ESR (Erythrocyte Sedimentation Rate, mm/hr)
│   ├── CRP (C-Reactive Protein, mg/L)
│   ├── RF (Rheumatoid Factor, IU/mL)
│   ├── Anti-CCP (Anti-Cyclic Citrullinated Peptide, U/mL)
│   ├── C3 (Complement Component 3, mg/dL)
│   └── C4 (Complement Component 4, mg/dL)
│
└── Categorical Laboratory Serology (6)
    ├── HLA-B27 (Categorical: Positive, Negative, NaN)
    ├── ANA (Categorical: Positive, Negative, NaN)
    ├── Anti-Ro (Categorical: Positive, Negative, NaN)
    ├── Anti-La (Categorical: Positive, Negative, NaN)
    ├── Anti-dsDNA (Categorical: Positive, Negative, NaN)
    └── Anti-Sm (Categorical: Positive, Negative, NaN)
```

---

## 4. Leakage Prevention and Traceability Architecture

### 4.1. Prevention of Target Leakage
- **The Issue**: If the original `Disease` column were present in the model training matrix, any algorithm would achieve 100% artificial accuracy by reading `Disease == 'Rheumatoid Arthritis'`.
- **The Solution**: In `data/processed/ra_engineered_dataset.csv`, both `Disease` and `patient_row_id` are maintained for clinical traceability, cohort error analysis, and auditing. However, the shared data loader (`src/data_loader.py`) programmatically filters the feature matrix `X` to include **strictly the 14 candidate input features**, returning `Target_RA` as a decoupled target vector `y`.

### 4.2. Sample Traceability via `patient_row_id`
Because the raw workbook lacks a primary hospital medical record number (MRN), a deterministic surrogate key `patient_row_id` ($0, 1, 2, \dots, 12084$) was assigned. This enables:
- Verification of zero train/test overlap.
- Exact mapping of misclassified records back to the original raw workbook during model error analysis.
- Consistent tracking across multiple iterations and notebooks.

---

## 5. Critical Audit Discovery: Contiguous Block Row Ordering

During dataset auditing (`01_dataset_audit.ipynb`), an essential structural property was uncovered:

> **Audit Finding**: The raw Excel workbook is **not randomized**. Patient rows are arranged in **7 monolithic contiguous blocks grouped strictly by disease**:
> - Rows 0 to 2,847: All **Rheumatoid Arthritis** (100% positive)
> - Rows 2,848 to 3,363: All **Reactive Arthritis**
> - Rows 3,364 to 5,490: All **Ankylosing Spondylitis**
> - Rows 5,491 to 7,342: All **Sjögren's Syndrome**
> - Rows 7,343 to 8,697: All **Systemic Lupus Erythematosus**
> - Rows 8,698 to 10,301: All **Normal**
> - Rows 10,302 to 12,084: All **Psoriatic Arthritis**

### Methodological Consequence:
A standard sequential slice (e.g., `train = df[:9668]`, `test = df[9668:]`) would result in a test set containing **zero positive RA cases** and only Normal and Psoriatic Arthritis cases.
**Remediation**: The data partitioning pipeline enforces **stratified random sampling with shuffling** under a fixed random seed (`random_state=42`), ensuring proportional representation in both splits while completely breaking the sequential block bias.

---

## 6. Treatment of Missing Values: Clinical Rationale

### 6.1. Why Missing Labs Must NOT Be Imputed Globally
- **Preservation of Genuine Missingness**: In clinical practice, laboratory tests are ordered selectively based on differential diagnosis (e.g., Anti-Sm and Anti-dsDNA are typically ordered when lupus is suspected, not during routine joint pain screening).
- **Fallacy of "Missing = Negative"**: A missing test is an unperformed test, **not** a confirmed negative serology. Imputing `NaN` as `'Negative'` or `0.0` introduces severe diagnostic bias.
- **Fallacy of Listwise Deletion (`dropna`)**: Only **452 out of 12,085 patients (3.74%)** have complete laboratory measurements across all tests. Running `dropna()` would discard **11,633 records (96.26% of the cohort)**, decimating the dataset and inducing catastrophic selection bias.
- **Pipeline Segregation**: Imputation (e.g., MedianImputer for numeric labs, SimpleImputer with `'Missing'` or frequent category for categorical serology, or KNN/IterativeImputer) must be fitted **exclusively on the training fold (`X_train`)** by model owners to prevent data leakage.

---

## 7. Artifact Storage Locations

| Artifact | File Path | Format | Record Count / Columns |
| :--- | :--- | :--- | :--- |
| **Engineered Cohort** | `data/processed/ra_engineered_dataset.csv` | CSV | 12,085 rows, 17 columns |
| **Training Split (80%)** | `data/splits/train.csv` | CSV | 9,668 rows, 17 columns |
| **Test Split (20%)** | `data/splits/test.csv` | CSV | 2,417 rows, 17 columns |
| **Split Metadata** | `data/splits/split_metadata.json` | JSON | Parameter & distribution audit |
| **Audit Report (JSON)** | `data/reports/dataset_audit_report.json` | JSON | Machine-readable metrics |
| **Audit Summary (MD)** | `data/reports/dataset_audit_summary.md` | Markdown | Human-readable audit findings |
| **Quality Report (JSON)**| `data/reports/data_quality_report.json` | JSON | Data quality validation rules |
