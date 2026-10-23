# Data Dictionary: Rheumatoid Arthritis Classification Dataset

## 1. Overview & Dataset Provenance

- **Dataset File**: `Rheumatic and Autoimmune Disease Dataset.xlsx`
- **Total Records**: 12,085 patient encounters
- **Total Original Columns**: 15 (1 demographic identifier, 1 sex indicator, 12 clinical laboratory measurements, 1 diagnostic label)
- **Engineered Records**: 12,085 records across 17 columns (including `patient_row_id` and binary `Target_RA`)
- **Clinical Scope**: Retrospective classification of clinical laboratory records from individuals evaluated for rheumatic and systemic autoimmune disorders.
- **Important Clinical Boundary**: This dataset represents retrospective classification of recorded diagnosis labels. It does **not** represent prospective prediction of future-onset disease or asymptomatic pre-clinical Rheumatoid Arthritis.

---

## 2. Original Variables Specification

| Column Name | Category | Data Type | Units / Values | Missing Count (%) | Reference / Normal Range | Clinical Significance & Biological Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Age** | Demographic | `int64` | Years [20 – 80] | 0 (0.00%) | N/A | Patient chronological age at assessment. RA prevalence typically peaks between 40–60 years, with significant geriatric incidence. |
| **Gender** | Demographic | `object` | `Male`, `Female` | 0 (0.00%) | N/A | Patient biological sex. RA exhibits a known female-to-male predilection (~3:1 ratio in general clinical populations). |
| **ESR** | Numeric Lab | `float64` | mm/hr [0.0 – 49.0] | 1,088 (9.00%) | Men: <15–20 mm/hr<br>Women: <20–25 mm/hr | Erythrocyte Sedimentation Rate. Non-specific acute-phase marker measuring red blood cell settling velocity; elevated in active systemic inflammation. |
| **CRP** | Numeric Lab | `float64` | mg/L [0.1 – 30.0] | 2,417 (20.00%) | <3.0 – 5.0 mg/L | C-Reactive Protein. Acute-phase reactant synthesized by hepatocytes in response to IL-6; responds rapidly to inflammatory flares. |
| **RF** | Numeric Lab | `float64` | IU/mL [0.0 – 40.0] | 1,329 (11.00%) | <15 – 20 IU/mL | Rheumatoid Factor. Autoantibody directed against the Fc portion of IgG; present in ~70–80% of established RA (seropositive RA). |
| **Anti-CCP** | Numeric Lab | `float64` | U/mL [0.0 – 40.0] | 3,263 (27.00%) | <20.0 U/mL | Anti-Cyclic Citrullinated Peptide antibody. Highly specific hallmark autoantibody for Rheumatoid Arthritis (>95% specificity). |
| **HLA-B27** | Categorical Lab | `object` | `Positive`, `Negative` | 1,934 (16.00%) | Negative | Major Histocompatibility Complex class I surface antigen. Strongly associated with Ankylosing Spondylitis and seronegative spondyloarthropathies; typically negative in primary RA. |
| **ANA** | Categorical Lab | `object` | `Positive`, `Negative` | 3,746 (31.00%) | Negative / Low Titer | Antinuclear Antibodies. Screening serology for systemic autoimmune disease; characteristically high in Systemic Lupus Erythematosus (SLE) and Systemic Sclerosis. |
| **Anti-Ro** | Categorical Lab | `object` | `Positive`, `Negative` | 2,900 (24.00%) | Negative | Anti-Ro (SSA) autoantibody. Hall-mark serology in Sjögren's Syndrome and subacute cutaneous lupus; occasionally co-occurs in secondary Sjögren's with RA. |
| **Anti-La** | Categorical Lab | `object` | `Positive`, `Negative` | 3,021 (25.00%) | Negative | Anti-La (SSB) autoantibody. Closely linked with Anti-Ro; high diagnostic specificity for primary Sjögren's Syndrome. |
| **Anti-dsDNA**| Categorical Lab | `object` | `Positive`, `Negative` | 4,713 (39.00%) | Negative | Antibodies to double-stranded DNA. Pathognomonic serological hallmark for Systemic Lupus Erythematosus (SLE) and lupus nephritis. |
| **Anti-Sm** | Categorical Lab | `object` | `Positive`, `Negative` | 5,197 (43.00%) | Negative | Anti-Smith autoantibodies directed against core snRNP proteins. Highly specific diagnostic criterion for SLE. |
| **C3** | Numeric Lab | `float64` | mg/dL [50.0 – 205.0] | 1,692 (14.00%) | 90 – 180 mg/dL | Complement Component 3. Depleted (hypocomplementemia) during immune-complex deposition and complement activation in active SLE. |
| **C4** | Numeric Lab | `float64` | mg/dL [5.0 – 74.0] | 2,054 (17.00%) | 10 – 40 mg/dL | Complement Component 4. Sensitive indicator of classical complement pathway consumption, particularly in lupus flares. |
| **Disease** | Clinical Diagnosis| `object` | 7 Diagnostic Labels | 0 (0.00%) | N/A | Original recorded diagnosis label: Rheumatoid Arthritis, Ankylosing Spondylitis, Sjögren's Syndrome, Psoriatic Arthritis, Normal, Systemic Lupus Erythematosus, Reactive Arthritis. |

---

## 3. Original Diagnostic Label Breakdown

| Diagnosis Category | Raw Count | Proportion | Classification Assignment |
| :--- | :--- | :--- | :--- |
| **Rheumatoid Arthritis** | 2,848 | 23.57% | **Positive Class (`Target_RA = 1`)** |
| **Ankylosing Spondylitis** | 2,127 | 17.60% | Negative Class (`Target_RA = 0`) |
| **Sjögren's Syndrome** | 1,852 | 15.32% | Negative Class (`Target_RA = 0`) |
| **Psoriatic Arthritis** | 1,783 | 14.75% | Negative Class (`Target_RA = 0`) |
| **Normal** | 1,604 | 13.27% | Negative Class (`Target_RA = 0`) |
| **Systemic Lupus Erythematosus** | 1,355 | 11.21% | Negative Class (`Target_RA = 0`) |
| **Reactive Arthritis** | 516 | 4.27% | Negative Class (`Target_RA = 0`) |
| **Total Cohort** | **12,085** | **100.00%** | **Target_RA: 1=2,848 (23.57%), 0=9,237 (76.43%)** |

---

## 4. Engineered Variables

### 4.1. `patient_row_id`
- **Data Type**: `int64` (values: 0 through 12,084)
- **Purpose**: Stable, unique surrogate key preserving the original positional row index from the raw workbook.
- **Usage**: Used for exact row tracing, split overlap verification, and cohort indexing across train and test partitions.
- **Model Rule**: Excluded from model input matrices (`X`) to avoid spurious index learning.

### 4.2. `Target_RA`
- **Data Type**: `int64` (values: `0` or `1`)
- **Mapping Logic**:
  $$\text{Target\_RA} = \begin{cases} 1 & \text{if } \text{Disease} = \text{'Rheumatoid Arthritis'} \\ 0 & \text{otherwise} \end{cases}$$
- **Class Balance**: 2,848 positive (23.57%), 9,237 negative (76.43%).
- **Model Rule**: Served as the ground-truth target vector `y` for all four classifiers.

---

## 5. Candidate Model Input Feature Set (14 Features)

All downstream machine learning models must ingest strictly these 14 candidate features in the following fixed sequence:

1. `Age` (Numeric)
2. `Gender` (Categorical)
3. `ESR` (Numeric)
4. `CRP` (Numeric)
5. `RF` (Numeric)
6. `Anti-CCP` (Numeric)
7. `HLA-B27` (Categorical)
8. `ANA` (Categorical)
9. `Anti-Ro` (Categorical)
10. `Anti-La` (Categorical)
11. `Anti-dsDNA` (Categorical)
12. `Anti-Sm` (Categorical)
13. `C3` (Numeric)
14. `C4` (Numeric)

> **Leakage Prevention Notice**: `Disease` and `Target_RA` must NEVER be included in the feature matrix `X`. The shared data loader (`src/data_loader.py`) enforces this separation programmatically.
