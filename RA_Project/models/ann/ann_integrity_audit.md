# Artificial Neural Network (ANN) Implementation, Data Leakage, and Result Integrity Audit

**Audit Timestamp**: 2026-10-27  
**Audited Notebook**: `RA_Project/notebooks/08_ann.ipynb`  
**Audited Artifacts Directory**: `RA_Project/models/ann/`  
**Auditor**: Independent Pipeline & Result Integrity Audit  

---

## 1. Overall Verdict

### **Verdict: PASS WITH WARNINGS**

The implementation of the Artificial Neural Network (ANN) workflow in `RA_Project/notebooks/08_ann.ipynb` and its associated artifacts in `RA_Project/models/ann/` adheres to rigorous machine learning standards with **zero data leakage detected**. The training, validation, and testing partitions are strictly disjoint; preprocessing transformations were fitted exclusively on sub-training data; architecture comparison was governed strictly by validation PR-AUC; and final held-out test evaluation was performed exactly once on frozen artifacts.

The single advisory warning pertains to the code implementation of the decision threshold selection: while threshold `0.39` genuinely maximizes validation $F_1$ (0.8443) and satisfies the $\ge 90\%$ sensitivity constraint (achieving 92.76% recall), the notebook code applied `df_thresholds["F1"].idxmax()` without a hard programmatic conditional filter (`df[df['Recall'] >= 0.90]['F1'].idxmax()`), relying on the mathematical coincidence that the global $F_1$ maximum on the validation set naturally satisfied the clinical sensitivity target.

---

## 2. Audit Summary Matrix

| Audit Area | Scope / Item | Status | Primary Code / Artifact Evidence | Finding Summary |
| :--- | :--- | :---: | :--- | :--- |
| **1. Partitioning** | 80:20 Train/Test Split | **PASS** | Cell 04 (`src.data_loader`), Cell 06 assertions | Reused standardized shared splits (`train.csv`, `test.csv`). No resplitting. |
| **1. Partitioning** | Internal Sub-Train/Val Split | **PASS** | Cell 08 (`train_test_split(..., test_size=0.20, stratify=y_train, random_state=42)`) | Stratified partition strictly from `X_train` (7,734 sub-train, 1,934 validation). |
| **1. Partitioning** | Partition Disjointness | **PASS** | Cell 06, independent `patient_row_id` set intersection | 0 overlap across all 3 subsets ($Train_{sub} \cap Val = \emptyset$, $Train \cap Test = \emptyset$). |
| **1. Partitioning** | Test Isolation in Callbacks | **PASS** | Cell 14 (`validation_data=(X_val_proc, y_val)`) | `X_test` was never passed to Keras callbacks or early stopping. |
| **2. Preprocessing** | Numeric Imputation & Scaling | **PASS** | Cell 10 (`preprocessor_tuning.fit_transform(X_train_sub)`) | Median imputers & standard scalers fitted strictly on `X_train_sub`. |
| **2. Preprocessing** | Categorical Encoding | **PASS** | Cell 10 (`OneHotEncoder(handle_unknown='ignore')`) | One-hot encodings fitted strictly on `X_train_sub`. |
| **2. Preprocessing** | Target & ID Exclusion | **PASS** | Cell 04, Cell 06 assertions | `Disease`, `Target_RA`, and `patient_row_id` strictly excluded from input matrices. |
| **2. Preprocessing** | Zero Leakage into Val/Test | **PASS** | Cell 10 (`.transform(X_val)`), Cell 22 (`.transform(X_test)`) | Validation and test data transformed using pre-fitted transformers only. |
| **3. Architecture** | Candidate Comparison | **PASS** | Cell 14, Cell 16 (`df_experiments.sort_values('Val_PR_AUC')`) | All 4 candidates evaluated on identical `(X_val_proc, y_val)` split. |
| **3. Architecture** | Selection Metric Integrity | **PASS** | Cell 16 (`winner_row = df_experiments.iloc[0]`) | Candidate A selected purely by highest validation PR-AUC (0.9269). |
| **3. Architecture** | Weight Restoration | **PASS** | Cell 14 (`EarlyStopping(restore_best_weights=True)`) | Weights restored to best epoch (Epoch 14, val_loss = 0.1548). |
| **4. Threshold** | Validation-Only Tuning | **PASS** | Cell 20 (`final_model.predict(X_val_proc)`) | Decision threshold evaluated exclusively on validation predictions. |
| **4. Threshold** | Decision Rule Implementation | **WARNING** | Cell 20 (`best_f1_idx = df_thresholds["F1"].idxmax()`) | Programmatic filter for `Recall >= 0.90` was not enforced in code, though 0.39 satisfies it. |
| **5. Test Eval** | Model Freezing | **PASS** | Cell 18, Cell 22 | Model and preprocessor were fully frozen prior to test evaluation. |
| **5. Test Eval** | Single Unbiased Run | **PASS** | Cell 22 (`evaluate_at_threshold`) | Evaluated once on held-out test data; no post-test retraining or threshold revision. |
| **5. Test Eval** | Probability Metrics (PR/ROC) | **PASS** | Cell 22 (`roc_auc_score(y_test, test_probs)`) | PR-AUC and ROC-AUC calculated on continuous sigmoid probabilities, not discrete classes. |
| **6. Artifacts** | Artifact Presence & Format | **PASS** | `models/ann/` file inspection | All 11 required files present, non-empty, and loadable without errors. |
| **6. Artifacts** | Preprocessor-Model Alignment | **PASS** | Model input shape `(None, 27)`, Preprocessor output `27` | Perfect dimensionality match between transformer output and ANN input layer. |
| **7. Math** | Result Consistency | **PASS** | Verification of `ann_test_predictions.csv` vs `ann_test_metrics.json` | Exact agreement across TP, FP, TN, FN, Recall, Specificity, Precision, F1, and AUC. |

---

## 3. Detailed Audit Findings

### 3.1. Train / Validation / Test Split Integrity (Audit Section 1)
- **Shared Split Loading (Cell 04)**: `X_train, y_train = load_train_data(return_X_y=True)` and `X_test, y_test = load_test_data(return_X_y=True)`. The notebook strictly ingests the reproducible splits from `data/splits/` without re-partitioning or duplicating features.
- **Internal Validation Partitioning (Cell 08)**:
  ```python
  X_train_sub, X_val, y_train_sub, y_val = train_test_split(
      X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_SEED
  )
  ```
  - Sub-train partition: 7,734 rows $\times$ 14 features (Positive RA: 1,822, 23.56%).
  - Validation partition: 1,934 rows $\times$ 14 features (Positive RA: 456, 23.58%).
  - Both partitions maintain exact empirical target balance.
- **Disjoint Partition Audit**: Independent verification of `patient_row_id` sets confirms:
  $$\text{Sub-train} \cap \text{Validation} = \emptyset \quad (0 \text{ overlapping records})$$
  $$\text{Sub-train} \cap \text{Test} = \emptyset \quad (0 \text{ overlapping records})$$
  $$\text{Validation} \cap \text{Test} = \emptyset \quad (0 \text{ overlapping records})$$
- **Callback Isolation**: In Cell 14, `train_args` passes `validation_data=(X_val_proc, y_val)` to `model.fit()`. The held-out test set `X_test` does not appear in the training loop, early stopping callback, or learning rate scheduler.

### 3.2. Preprocessing Pipeline & Leakage Audit (Audit Section 2)
- **Pipeline Architecture (Cell 10)**:
  - `numeric_transformer`: `SimpleImputer(strategy="median")` followed by `StandardScaler()`.
  - `categorical_transformer`: `SimpleImputer(strategy="constant", fill_value="Missing")` followed by `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- **Fitting Isolation**:
  ```python
  preprocessor_tuning = create_preprocessing_pipeline()
  X_tr_proc = preprocessor_tuning.fit_transform(X_train_sub)
  X_val_proc = preprocessor_tuning.transform(X_val)
  ```
  - The preprocessor was `.fit_transform()`-ed strictly on `X_train_sub` (7,734 records).
  - It was **not** fitted on the full 9,668-record partition before creating `X_val`, eliminating validation-to-train distribution leakage.
  - Test features were transformed in Cell 22 via `final_preprocessor.transform(X_test)` with zero fitting operations.
- **Feature Schema & Exclusion**:
  - Raw inputs strictly comprise the 14 candidate clinical features in standardized order.
  - `Disease`, `Target_RA`, and `patient_row_id` are absent from `X_train`, `X_train_sub`, `X_val`, and `X_test`.
  - Transformed output consists of 27 columns (7 numeric, 2 gender one-hot, 18 autoimmune assay one-hot). All values are finite real numbers with zero NaNs or infinities.

### 3.3. Controlled Architecture Selection (Audit Section 3)
- **Experimental Control (Cell 14)**:
  - 4 candidate configurations were evaluated: Compact (Candidate A), Medium Baseline (Candidate B), Wider (Candidate C), and Medium with balanced class weights.
  - All trials initialized with `set_seed(42)` and executed on the identical `(X_tr_proc, y_train_sub)` and `(X_val_proc, y_val)` matrices.
  - `EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)` and `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)` monitored validation loss exclusively.
- **Selection Decision (Cell 16)**:
  - Table sorted by `Val_PR_AUC` descending:
    1. Candidate A: Val PR-AUC = **0.9269**, Val Loss = **0.1548**, Val ROC-AUC = **0.9770** (Best Epoch 14).
    2. Candidate B: Val PR-AUC = 0.9199, Val Loss = 0.1599, Val ROC-AUC = 0.9745 (Best Epoch 16).
    3. Candidate B + Class Weights: Val PR-AUC = 0.9175, Val Loss = 0.1892, Val ROC-AUC = 0.9735 (Best Epoch 30).
    4. Candidate C: Val PR-AUC = 0.9130, Val Loss = 0.1712, Val ROC-AUC = 0.9722 (Best Epoch 17).
  - Candidate A won strictly on the predefined primary metric (`Val_PR_AUC`). Test performance was never computed or referenced during model selection.

### 3.4. Threshold Selection & Decision Rule Audit (Audit Section 4)
- **Validation Threshold Sweep (Cell 20)**:
  - Computed validation probabilities: `val_probs = final_model.predict(X_val_proc, verbose=0).ravel()`.
  - Grid search evaluated 81 discrete thresholds across $[0.10, 0.90]$ with step $0.01$.
- **Selection Rule Code Pattern**:
  ```python
  best_f1_idx = df_thresholds["F1"].idxmax()
  selected_threshold_row = df_thresholds.iloc[best_f1_idx]
  SELECTED_THRESHOLD = float(selected_threshold_row["Threshold"])
  ```
- **Audit Assessment of Warning**:
  - The documentation and narrative state that the threshold was selected by maximizing validation $F_1$ while requiring sensitivity $\ge 90\%$.
  - In the code, `df_thresholds["F1"].idxmax()` was executed directly without a hard programmatic condition such as `df_thresholds[df_thresholds["Recall"] >= 0.90]["F1"].idxmax()`.
  - Independent mathematical verification reveals:
    - Global maximum of $F_1$ across all 81 thresholds occurs at $\mathbf{\tau = 0.39}$ ($F_1 = 0.844311$).
    - At $\tau = 0.39$, validation Recall is **92.76%** (423/456 detected), which naturally satisfies $\ge 90\%$.
    - Restricting the search space to thresholds where $\text{Recall} \ge 0.90$ yields the exact same threshold $\tau = 0.39$.
  - Therefore, the selected threshold of $0.39$ is mathematically valid and satisfies both criteria, but the programmatic constraint was not hardcoded in the query.

### 3.5. Held-Out Test Evaluation & Metric Verification (Audit Section 5)
- **Test Set Isolation**: The test evaluation in Cell 22 occurs after model selection (Cell 16), model confirmation (Cell 18), and threshold selection (Cell 20).
- **Metric Formulation**:
  - `test_probs = final_model.predict(X_test_proc, verbose=0).ravel()`.
  - Continuous probabilities $p \in [0, 1]$ were passed to `roc_auc_score(y_test, test_probs)` and `average_precision_score(y_test, test_probs)`. Neither metric was computed from binarized class labels.
  - Discrete metrics (Accuracy, Precision, Recall, Specificity, $F_1$) were computed using the confusion matrix derived from $(p \ge \tau)$.
- **Why Similar Validation and Test Scores Do Not Prove Absence of Leakage**:
  - Test PR-AUC (0.9252) and ROC-AUC (0.9768) closely match validation PR-AUC (0.9269) and ROC-AUC (0.9770).
  - While consistent scores demonstrate absence of severe overfitting, score similarity alone is insufficient to rule out leakage. For instance, if global feature scaling or duplicate records existed across train and test partitions, both partitions would exhibit artificially elevated, closely matching metrics.
  - The true guarantee of zero leakage rests on structural isolation: zero record overlap, `.fit()` restricted to sub-train, and frozen one-pass test inference.

### 3.6. Verification of Saved Artifacts (Audit Section 6)
All 11 artifacts in `RA_Project/models/ann/` were verified on disk:

1. `ann_model.keras` (45.69 KB): Loads cleanly via `tf.keras.models.load_model`. Name: `Candidate_A_Compact`, input shape `(None, 27)`, output shape `(None, 1)`, total parameters: 1,441.
2. `preprocessor.joblib` (5.14 KB): Loads cleanly via `joblib.load`. Contains `ColumnTransformer` with 7 median-imputed scaled numeric features and 7 constant-imputed one-hot encoded categorical features. Output dimension is exactly 27.
3. `ann_metadata.json` (4.07 KB): Fully populated with architecture layers, hyperparameter configuration, package versions, and complete validation and test metrics.
4. `ann_training_history.csv` (5.27 KB): Contains 29 training epochs. Minimum validation loss is recorded at epoch 14 (0.154834), matching the restored model checkpoint.
5. `ann_validation_results.csv` (0.53 KB): Contains 4 rows detailing validation metrics across all candidates.
6. `ann_test_metrics.json` (0.90 KB): Contains exact test metrics at both $\tau = 0.50$ and $\tau = 0.39$.
7. `ann_test_predictions.csv` (45.35 KB): Exactly 2,417 prediction rows plus 1 header row. Contains `patient_row_id`, `true_label`, `predicted_probability`, `predicted_class_default`, and `predicted_class_threshold_0.39`.
8. Visual artifacts (`ann_training_curves.png`, `ann_confusion_matrix.png`, `ann_precision_recall_curve.png`, `ann_roc_curve.png`): High-resolution figures correctly depicting training dynamics, confusion matrices, and ROC/PR curves.

---

## 4. Numerical Consistency Audit (Audit Section 7)

An independent recalculation was conducted directly on the 2,417 rows of `ann_test_predictions.csv` and compared against `ann_test_metrics.json`:

### At Selected Threshold ($\tau = 0.39$)
- **Sample Support**: Total = 2,417 | True Positive RA = 570 | True Negative Non-RA = 1,847
- **Raw Confusion Matrix**:
  - $\text{True Positives (TP)} = 521$
  - $\text{False Negatives (FN)} = 49$
  - $\text{True Negatives (TN)} = 1,694$
  - $\text{False Positives (FP)} = 153$
- **Mathematical Metrics**:
  - $\text{Accuracy} = \frac{521 + 1694}{2417} = \frac{2215}{2417} = \mathbf{0.91642532}$ (Reported: 91.64%)
  - $\text{Precision} = \frac{521}{521 + 153} = \frac{521}{674} = \mathbf{0.77299703}$ (Reported: 77.30%)
  - $\text{Recall (Sensitivity)} = \frac{521}{521 + 49} = \frac{521}{570} = \mathbf{0.91403509}$ (Reported: 91.40%)
  - $\text{Specificity} = \frac{1694}{1694 + 153} = \frac{1694}{1847} = \mathbf{0.91716297}$ (Reported: 91.72%)
  - $\text{F1-Score} = \frac{2 \times 0.77299703 \times 0.91403509}{0.77299703 + 0.91403509} = \mathbf{0.83762058}$ (Reported: 0.8376)
  - $\text{ROC-AUC} = \mathbf{0.97677030}$ (Reported: 0.9768)
  - $\text{PR-AUC} = \mathbf{0.92520373}$ (Reported: 0.9252)

### At Default Threshold ($\tau = 0.50$)
- $\text{TP} = 497, \text{FN} = 73, \text{TN} = 1727, \text{FP} = 120$
- $\text{Accuracy} = 0.92014894$ (92.01%)
- $\text{Precision} = 0.80551053$ (80.55%)
- $\text{Recall} = 0.87192982$ (87.19%)
- $\text{Specificity} = 0.93502978$ (93.50%)
- $\text{F1-Score} = 0.83740522$ (0.8374)

**Audit Finding**: Zero discrepancies detected. Every reported figure matches the raw prediction records to the 8th decimal place.

---

## 5. Specific Recommended Corrections (Non-Destructive)

The following advisory enhancements are recommended for future pipeline iterations without altering existing saved artifacts:

1. **Explicit Programmatic Constraint in Threshold Grid Search**:
   In Cell 20, make the sensitivity requirement explicit in code:
   ```python
   # Recommended explicit formulation:
   df_eligible = df_thresholds[df_thresholds["Recall"] >= 0.90]
   if not df_eligible.empty:
       selected_threshold_row = df_eligible.loc[df_eligible["F1"].idxmax()]
   else:
       selected_threshold_row = df_thresholds.loc[df_thresholds["F1"].idxmax()]
   ```
2. **Explicit Random Seed in Keras Layers**:
   While global seeds were set in Cell 02 via `set_seed(42)`, explicitly passing `kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)` into `Dense` layers provides an additional safeguard against cross-platform multithreading nondeterminism.
3. **Clinical Threshold Documentation in Metadata**:
   In `ann_metadata.json`, note that the threshold $\tau = 0.39$ is an empirical cohort-specific operating point designed for decision support and requires local calibration prior to prospective clinical utilization.
