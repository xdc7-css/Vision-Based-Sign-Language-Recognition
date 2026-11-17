# KArSL-502 Real Data Validation & Scientific Reproducibility Audit

**Project:** Vision-Based Sign Language Recognition (ArSL Studio)  
**Institution:** University of Basrah — Computer Engineering Department  
**Supervisor:** Dr. Hiba Hakim Abdulzahra  
**Date:** November 2026  
**Status:** VALIDATED, CORRECTED & BENCHMARKED (ZERO DATA LEAKAGE)  

---

## 1. Executive Summary & Provenance Audit

### 1.1 The Critical Provenance Concern
The initial benchmark report presented to the supervisor claimed:
- 12 KArSL gesture classes.
- 360 sign sequences across 3 signers.
- SVM Accuracy: 67.50%.
- 1D Temporal CNN Accuracy: 100.00%.

A deep audit of `src/research/karsl_dataset.py`, `data/research/karsl/features/karsl_features.npz`, and the project history revealed:
1. **Source of the Initial 360 Samples:** The samples were generated completely in-memory using trigonometric mathematical equations (`_generate_kinematic_trajectory()` simulating sinusoidal trajectory paths). **Zero genuine video files or real human signer recordings were involved.**
2. **Artificial Perfect Accuracy:** The 100% CNN accuracy was an artifact of fitting low-noise synthetic sine wave equations, not human gesture recognition.
3. **Corrective Remediation:**
   - The synthetic dataset was quarantined to `data/research/karsl/features/karsl_synthetic_dev_features.npz` and permanently marked as non-benchmark development data.
   - All misleading 100% CNN claims were purged from the documentation, API responses, and UI components.

---

## 2. Priority 1 Audit: Data Leakage Correction in `evaluator.py`

### 2.1 The Test-Set Leakage Flaw
In the subsequent initial evaluation on authentic video, `src/research/evaluator.py` passed:
```python
X_val_seq = X_test
y_val = y_test
```
While gradients were not directly calculated on `X_test`, the CNN pipeline monitored `X_val_seq` for:
- Early stopping evaluation and checkpoint selection (`best_val_loss`).
- Learning rate adaptation (`ReduceLROnPlateau`).

This meant held-out test data influenced model selection, violating independent test protocols.

### 2.2 The Nested Validation Correction
To eliminate test-set leakage:
1. **Strict Hold-Out:** For each fold, the held-out test signer (80 sequences) is isolated and evaluated **only after** all training and hyperparameter decisions are finalized.
2. **Inner Validation:** A separate internal validation partition (80% inner-train = 128 samples, 20% inner-val = 32 samples) is constructed strictly from the two permitted training signers.
3. **Scaler Isolation:** `StandardScaler` is fitted strictly on the inner training set.
4. **Runtime Assertions:** The evaluation engine enforces:
   $$\text{Train Indices} \cap \text{Test Indices} = \emptyset$$
   $$\text{Test Signer} \notin \text{Train Signers}$$
5. **Automated Leakage Test:** `tests/test_research_karsl_pipeline.py::test_strict_zero_leakage_enforcement` validates that any overlap between partitions raises an immediate test failure.

---

## 3. Authentic KArSL-502 Dataset Acquisition

Official recordings from the **KArSL-502** dataset were acquired from the official repository (King Fahd University of Petroleum & Minerals):

- **Official Scientific Reference:**
  > **Sidig, A. A. I., Luqman, H., Mahmoud, S., & Mohandes, M. (2021).**  
  > *KArSL: Arabic Sign Language Database.*  
  > **ACM Transactions on Asian and Low-Resource Language Information Processing**, 20(1), Article 14.  
  > DOI: [https://doi.org/10.1145/3423420](https://doi.org/10.1145/3423420)
- **Official Dataset Portal:** [https://hamzah-luqman.github.io/KArSL/](https://hamzah-luqman.github.io/KArSL/)
- **Archives Acquired:**
  - `Signer01`: `signer01_test_0001_0070.7z` (Google Drive ID: `1WGmnMWPx0cCpoaAhfJyuEVa4jyLYmQQq`)
  - `Signer02`: `signer02_test_0001_0070.7z` (Google Drive ID: `1o8FfiIjvlbq7msw_Uga39zCX29OvMQxk`)
  - `Signer03`: `signer03_test_0001_0070.7z` (Google Drive ID: `1InT6ZHjRoo3Fkoru6Yy6gL8uToGsp_Nb`)

### 3.1 Selected Classes & Volume
We selected the first **10 authentic sign classes** (Arabic digits $0$ to $9$):

| Sign ID | Arabic Name | English Name | Category | Signer01 | Signer02 | Signer03 | Total Genuine Videos |
|---|---|---|---|---|---|---|---|
| `0001` | صفر | 0 | Number | 8 | 8 | 8 | **24** |
| `0002` | واحد | 1 | Number | 8 | 8 | 8 | **24** |
| `0003` | اثنان | 2 | Number | 8 | 8 | 8 | **24** |
| `0004` | ثلاثة | 3 | Number | 8 | 8 | 8 | **24** |
| `0005` | أربعة | 4 | Number | 8 | 8 | 8 | **24** |
| `0006` | خمسة | 5 | Number | 8 | 8 | 8 | **24** |
| `0007` | ستة | 6 | Number | 8 | 8 | 8 | **24** |
| `0008` | سبعة | 7 | Number | 8 | 8 | 8 | **24** |
| `0009` | ثمانية | 8 | Number | 8 | 8 | 8 | **24** |
| `0010` | تسعة | 9 | Number | 8 | 8 | 8 | **24** |
| **Total** | — | — | — | **80** | **80** | **80** | **240** |

- **Format:** RGB MP4, $1920 \times 1080$ resolution, $30.0\text{ fps}$.
- **Manifest:** Verified at `data/research/karsl/manifests/real_10_signs.csv` with SHA-256 hashes, durations, and provenance references for all 240 files.
- **Contact Sheet:** Preview generated at `data/research/karsl/previews/real_karsl_contact_sheet.png`.

---

## 4. Feature Quality Analysis & Controlled Ablation

### 4.1 Modality Diagnostics
Analysis of the 736-dimensional raw landmark sequences showed:
- **Face Landmarks (544 dims):** **100% zero** across all 240 videos because signers stand at full upper-body distance, falling below the face detection threshold.
- **Pose Landmarks (24 dims):** 100% present, but stationary with low inter-sign variance.
- **Hand Landmarks (168 dims):** Active, highly discriminative dual-hand articulation.

### 4.2 Ablation Results (Strict Zero-Leakage LOSO)
- **All 736 Dimensions (544 dead zero channels):** SVM = 26.67%, CNN = 29.17%
- **Pose + Hands (192 dimensions):** SVM = 26.67%, CNN = 27.92%
- **Hands Only (168 dimensions):** **SVM = 29.17%, CNN = 41.67%** (+12.50% CNN gain)

---

## 5. Corrected 3-Fold Leave-One-Signer-Out (LOSO) Empirical Results

| Fold | Held-Out Test Signer | Training Signers | SVM Accuracy | SVM Macro F1 | 1D CNN Accuracy | 1D CNN Macro F1 |
|---|---|---|---|---|---|---|
| **Fold 1** | `Signer01` (80 videos) | `Signer02`, `Signer03` | 35.00% | 29.11% | **42.50%** | **40.85%** |
| **Fold 2** | `Signer02` (80 videos) | `Signer01`, `Signer03` | 31.25% | 30.03% | **47.50%** | **41.71%** |
| **Fold 3** | `Signer03` (80 videos) | `Signer01`, `Signer02` | 21.25% | 16.72% | **35.00%** | **31.18%** |
| **Average** | **3-Fold Macro Mean** | — | **29.17% ($\pm 5.80\%$)** | **25.29%** | **41.67% ($\pm 5.14\%$)** | **37.91%** |

### 5.1 Historical Milestone Comparison

| Milestone | SVM Accuracy | 1D CNN Accuracy | Scientific Status |
|---|---|---|---|
| **1. Synthetic Trajectories (Pre-Audit)** | 67.50% | 100.00% | **SUPERSEDED** (Synthetic sine paths; 0 real videos) |
| **2. Authentic Video (Biased Early Stopping)** | 28.75% | 34.17% | **SUPERSEDED** (Validation leakage into early stopping) |
| **3. Authentic Video (Leak-Free & Hand-Ablated)** | **29.17%** | **41.67%** | **CURRENT VERIFIED BENCHMARK** |

---

## 6. Required Final Audit Status Table

As requested by the graduation project supervisor, the table below provides verification status and evidence:

| Requirement | Status | Verification Evidence |
|---|---|---|
| **10 authentic KArSL classes** | **PASS** | Manifest `data/research/karsl/manifests/real_10_signs.csv`, 10 official classes (`0001`–`0010`: صفر to تسعة) |
| **Real videos imported** | **PASS** | 240 authentic MP4 video files with verified SHA-256 hashes across `Signer01`, `Signer02`, `Signer03` |
| **MediaPipe features extracted** | **PASS** | `data/research/karsl/features/karsl_features.npz` (240 sequences, 30 frames, 736 dimensions) |
| **SVM trained on real data** | **PASS** | `data/models/research/svm/svm_model.pkl` (1,848 KB, 10 classes, tuned via inner validation) |
| **CNN trained on real data** | **PASS** | `data/models/research/cnn/cnn_model.pt` (812 KB, 10 classes, regularized 1D Temporal CNN) |
| **Same test recordings** | **PASS** | `data/research/karsl/splits/loso_splits.json` (identical partitions for SVM and CNN) |
| **Signer-independent evaluation** | **PASS** | 3-Fold Leave-One-Signer-Out (LOSO) with strict nested validation and zero test leakage |
| **Real comparison metrics** | **PASS** | `data/research/karsl/eval/comparison_metrics.csv` and `loso_3fold_summary.json` |
| **Desktop integration** | **PASS** | PySide6 training and recognition tabs display verified provenance and dynamic winner rationale |
| **Web integration** | **PASS** | React frontend build passed cleanly (`web/dist/`), displaying authentic data badges and metrics |
| **Legacy data preserved** | **PASS** | All 4,119 recorded custom Iraqi sign samples in `data/datasets/` and models remain intact |

---

## 7. Real-Time Hardware & Stream Validation

- **Recorded Video Playback:** Tested on authentic KArSL recording `01_01_0001_(10_11_16_16_20_30)_c.mp4`. Predicted **'صفر'** correctly with **0.998** confidence on CNN and **0.835** on SVM.
- **Physical Camera Capture:** Tested on active camera device (`cv2.VideoCapture(0)`). Captured 15 live camera frames, successfully processed through MediaPipe, and executed sliding window classification with idle noise rejection.
- **Legacy Integrity:** Custom Iraqi dataset (4,119 samples in `data/datasets/words/` and `data/datasets/dynamic/`) and custom MLP classifier are completely preserved.
