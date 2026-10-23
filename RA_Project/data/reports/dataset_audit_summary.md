# Dataset Audit Report: Rheumatic and Autoimmune Disease Dataset

## 1. Executive Summary
- **Raw File**: `Rheumatic and Autoimmune Disease Dataset.xlsx`
- **Dimensions**: 12,085 records across 15 columns.
- **Exact Duplicate Rows**: 0.
- **Block-Ordered Structure**: True (7 contiguous blocks detected).

## 2. Diagnosis Distribution
| Diagnosis | Record Count | Percentage |
| :--- | :--- | :--- |
| Rheumatoid Arthritis | 2,848 | 23.57% |
| Ankylosing Spondylitis | 2,127 | 17.60% |
| Sjögren's Syndrome | 1,852 | 15.32% |
| Psoriatic Arthritis | 1,783 | 14.75% |
| Normal | 1,604 | 13.27% |
| Systemic Lupus Erythematosus | 1,355 | 11.21% |
| Reactive Arthritis | 516 | 4.27% |

## 3. Data Ordering Analysis (Contiguous Diagnosis Blocks)
> **CRITICAL FINDING**: The raw workbook arranges patient records into 7 continuous blocks grouped strictly by disease. A non-shuffled train/test split would cause catastrophic leakage or train on one disease and test on another.

| Block # | Diagnosis | Start Row | End Row | Total Records |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Rheumatoid Arthritis | 0 | 2,847 | 2,848 |
| 2 | Reactive Arthritis | 2,848 | 3,363 | 516 |
| 3 | Ankylosing Spondylitis | 3,364 | 5,490 | 2,127 |
| 4 | Sjögren's Syndrome | 5,491 | 7,342 | 1,852 |
| 5 | Systemic Lupus Erythematosus | 7,343 | 8,697 | 1,355 |
| 6 | Normal | 8,698 | 10,301 | 1,604 |
| 7 | Psoriatic Arthritis | 10,302 | 12,084 | 1,783 |

## 4. Completeness and Missing Values
| Column | Data Type | Non-Null Count | Missing Count | Missing % |
| :--- | :--- | :--- | :--- | :--- |
| Age | `int64` | 12,085 | 0 | 0.00% |
| Gender | `object` | 12,085 | 0 | 0.00% |
| ESR | `float64` | 10,997 | 1,088 | 9.00% |
| CRP | `float64` | 9,668 | 2,417 | 20.00% |
| RF | `float64` | 10,756 | 1,329 | 11.00% |
| Anti-CCP | `float64` | 8,822 | 3,263 | 27.00% |
| HLA-B27 | `object` | 10,151 | 1,934 | 16.00% |
| ANA | `object` | 8,339 | 3,746 | 31.00% |
| Anti-Ro | `object` | 9,185 | 2,900 | 24.00% |
| Anti-La | `object` | 9,064 | 3,021 | 25.00% |
| Anti-dsDNA | `object` | 7,372 | 4,713 | 39.00% |
| Anti-Sm | `object` | 6,888 | 5,197 | 43.00% |
| C3 | `float64` | 10,393 | 1,692 | 14.00% |
| C4 | `float64` | 10,031 | 2,054 | 17.00% |
| Disease | `object` | 12,085 | 0 | 0.00% |

## 5. Numeric Laboratory Features Summary
| Feature | Count | Mean | Std | Min | 25% | Median | 75% | Max | Zero Count | Outliers (IQR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Age | 12,085 | 49.9058 | 17.65 | 20.0 | 35.0 | 50.0 | 65.0 | 80.0 | 0 | 0 (0.0%) |
| ESR | 10,997 | 24.2143 | 14.3683 | 0.0 | 10.0 | 28.0 | 36.0 | 49.0 | 260 | 0 (0.0%) |
| CRP | 9,668 | 13.325 | 10.393 | 0.1 | 1.9 | 15.6 | 22.6 | 30.0 | 0 | 0 (0.0%) |
| RF | 10,756 | 19.6852 | 11.5176 | 0.0 | 9.7 | 19.2 | 29.7 | 40.0 | 11 | 0 (0.0%) |
| Anti-CCP | 8,822 | 19.5792 | 11.6017 | 0.0 | 9.5 | 19.0 | 29.6 | 40.0 | 12 | 0 (0.0%) |
| C3 | 10,393 | 131.9103 | 36.4259 | 50.0 | 105.0 | 133.0 | 161.0 | 205.0 | 0 | 0 (0.0%) |
| C4 | 10,031 | 38.1785 | 20.0502 | 5.0 | 21.0 | 38.0 | 55.0 | 74.0 | 0 | 0 (0.0%) |

## 6. Categorical Laboratory and Demographic Features
| Feature | Unique Categories | Distribution | Whitespace Anomaly |
| :--- | :--- | :--- | :--- |
| Gender | 2 | Male: 6,151, Female: 5,934 | False |
| HLA-B27 | 3 | Positive: 6,231, Negative: 3,920, NaN: 1,934 | False |
| ANA | 3 | Positive: 5,248, NaN: 3,746, Negative: 3,091 | False |
| Anti-Ro | 3 | Positive: 5,292, Negative: 3,893, NaN: 2,900 | False |
| Anti-La | 3 | Positive: 5,305, Negative: 3,759, NaN: 3,021 | False |
| Anti-dsDNA | 3 | NaN: 4,713, Positive: 4,077, Negative: 3,295 | False |
| Anti-Sm | 3 | NaN: 5,197, Positive: 3,759, Negative: 3,129 | False |
| Disease | 7 | Rheumatoid Arthritis: 2,848, Ankylosing Spondylitis: 2,127, Sjögren's Syndrome: 1,852, Psoriatic Arthritis: 1,783, Normal: 1,604, Systemic Lupus Erythematosus: 1,355, Reactive Arthritis: 516 | False |

## 7. Distribution of Missing Laboratory Tests per Patient
| Missing Tests Count | Number of Patients | Percentage |
| :--- | :--- | :--- |
| 0 missing labs | 452 | 3.74% |
| 1 missing labs | 1,796 | 14.86% |
| 2 missing labs | 3,204 | 26.51% |
| 3 missing labs | 3,191 | 26.40% |
| 4 missing labs | 2,127 | 17.60% |
| 5 missing labs | 919 | 7.60% |
| 6 missing labs | 313 | 2.59% |
| 7 missing labs | 69 | 0.57% |
| 8 missing labs | 13 | 0.11% |
| 9 missing labs | 1 | 0.01% |

## 8. Identified Data Limitations & Methodological Constraints
- Absence of explicit unique patient identifier (Patient ID / MRN); rows are currently identifiable only by positional index.
- Severe row-ordering artifact: records are strictly segregated into 7 monolithic diagnosis blocks in the raw workbook.
- High missingness in specialized laboratory tests (Anti-Sm: 43.0%, Anti-dsDNA: 39.0%, ANA: 31.0%, Anti-CCP: 27.0%, CRP: 20.0%).
- Retrospective cross-sectional cohort: labels reflect recorded clinical diagnoses, not prospective future-onset disease prediction.
- Absence of longitudinal follow-up, symptom duration, physical joint examination scores (e.g., DAS28, tender/swollen joint counts), and imaging (X-ray/MRI).