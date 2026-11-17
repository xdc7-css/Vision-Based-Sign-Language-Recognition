# Scientific Quality Assurance & Practical Recognition Validation Report

**Project:** Vision-Based Sign Language Recognition (ArSL Studio)  
**Institution:** University of Basrah — College of Engineering — Computer Engineering Department  
**Supervisor:** Dr. Hiba Hakim Abdulzahra  
**Date:** November 2026  
**Status:** COMPLETED, AUDITED & VERIFIED  

---

## 1. Confirmed Accomplishments

1. **Authentic Data Provenance & Quarantine of Synthetic Data:**
   - Quarantined previous 360 mathematical sine-wave trajectories to `karsl_synthetic_dev_features.npz`.
   - Acquired **240 genuine video recordings** ($1920 \times 1080$ @ 30 fps) from the official KArSL-502 dataset across 10 Arabic digit classes (`0001`–`0010`: صفر to تسعة) and 3 human signers (`Signer01`, `Signer02`, `Signer03`).
   - Documented complete SHA-256 hashes, frame counts, and source URLs in `data/research/karsl/manifests/real_10_signs.csv`.

2. **Resolution of Test-Set Leakage (Priority 1):**
   - Eliminated the critical flaw in `evaluator.py` where `X_test` was passed as `X_val_seq`.
   - Implemented strict nested cross-validation: for each outer fold, an inner 80/20 train/val partition (128 inner train / 32 inner val) is constructed exclusively from the two permitted training signers.
   - `StandardScaler` is fitted strictly on inner train; early stopping and learning rate schedules evaluate inner validation only.
   - Outer held-out test data is evaluated strictly after model parameters are finalized.

3. **Feature Quality & Ablation Discovery:**
   - Forensic analysis proved that Face mesh landmarks (544 channels, 73.9% of the feature vector) were 100% exact zeros in KArSL video recordings due to camera-to-signer distance.
   - Controlled ablation proved that focusing on the 168 hand coordinate channels (Left Hand 84 + Right Hand 84) eliminated noise and raised 1D Temporal CNN accuracy from 29.17% to **41.67%** (and Macro F1 from 27.74% to **37.91%**).

4. **Code Reliability & Explicit Dimension Validation:**
   - In `src/research/preprocessing.py`, removed silent fallback slicing for unexpected input shapes and enforced explicit `ValueError` exceptions on invalid feature dimensions.
   - In `src/research/research_recognizer.py`, added idle rejection checks to prevent false-positive predictions when hands are at rest or no landmarks are detected.

5. **Separation of Cross-Validation and Deployment Checkpoints:**
   - Isolated individual fold models in `data/models/research/folds/` (`fold_Signer01/`, `fold_Signer02/`, `fold_Signer03/`).
   - Saved final production deployment models in `data/models/research/svm/` (1,087 KB) and `data/models/research/cnn/` (475 KB) trained on all 240 sequences with explicit deployment metadata.

6. **Unseen Signer Evaluation Workflow:**
   - Created `src/research/evaluate_new_recordings.py` to evaluate genuinely new video recordings from people not involved in training.
   - Strictly reports evaluation status as `PENDING_NEW_UNSEEN_RECORDINGS` when no new recordings exist, preventing fabricated claims.

---

## 2. Research Integrity Audit: Training Leakage vs. Test-Guided Selection

A rigorous audit was conducted on the experimentation methodology:

### 2.1 Direct Training Leakage (Resolved)
- **Status:** **FULLY RESOLVED**.
- In the corrected pipeline, no test signer data touches weight updates, gradient calculation, feature scaling, learning rate adaptation, or early stopping checkpoint selection. Disjoint index assertions and automated tests enforce zero leakage.

### 2.2 Test-Guided Experiment Selection (Meta-Level Leakage)
- **Status:** **IDENTIFIED & DOCUMENTED**.
- In the feature ablation phase, three candidate feature sets (All 736, Pose+Hands 192, Hands Only 168) and architectural configurations (dropout 0.3, Conv1D channel widths 64/128) were compared across the 3 LOSO test folds.
- The 168-dimensional "Hands Only" schema was selected because it achieved the highest score (41.67%) across the three test signers.
- **Scientific Consequence:** Because outer test feedback guided the selection of the feature representation and network hyperparameters, **the 41.67% CNN and 29.17% SVM accuracy figures cannot be claimed as completely blind, independent generalization estimates**.
- **Honest Characterization:** These metrics represent a **development benchmark under test-guided feature optimization**. True generalization to unseen populations must be validated against independently collected external data.

---

## 3. Verification of Existing LOSO Implementation

Across all 240 genuine video sequences under strict nested cross-validation:

| Fold | Held-Out Test Signer | Training Signers | SVM Accuracy | SVM Macro F1 | 1D CNN Accuracy | 1D CNN Macro F1 |
|---|---|---|---|---|---|---|
| **Fold 1** | `Signer01` (80 videos) | `Signer02`, `Signer03` | 35.00% | 29.11% | **42.50%** | **40.85%** |
| **Fold 2** | `Signer02` (80 videos) | `Signer01`, `Signer03` | 31.25% | 30.03% | **47.50%** | **41.71%** |
| **Fold 3** | `Signer03` (80 videos) | `Signer01`, `Signer02` | 21.25% | 16.72% | **35.00%** | **31.18%** |
| **Average** | **3-Fold Macro Mean** | — | **29.17% ($\pm 5.80\%$)** | **25.29%** | **41.67% ($\pm 5.14\%$)** | **37.91%** |

- **Random Baseline:** 10.00% (10 classes). Both models achieve 2.9x to 4.2x above random baseline.
- **Temporal Advantage:** The 1D Temporal CNN outperforms SVM by **+12.50% accuracy** and **+12.62% macro F1** due to its ability to model inter-frame velocity and temporal transition dynamics.

---

## 4. Complete Test-Suite Execution Results

The full project test suite was executed:
- **Command:** `.venv\Scripts\python.exe -m pytest tests/ -q`
- **Total Tests Collected:** **325 tests**
- **Passed:** **325 passed (100.0%)**
- **Failed:** **0 failed**
- **Skipped:** **0 skipped**
- **Exit Code:** **0 (SUCCESS)**
- **Execution Time:** **74.42 seconds**

### Component Breakdown
- Research KArSL Pipeline (`test_research_karsl_pipeline.py`): 12 / 12 passed
- Recognition & Custom Landmarks (`test_recognition_custom_and_landmarks.py`): 5 / 5 passed
- Multimodal Two-Hands & Pose (`test_multimodal.py`): 20 / 20 passed
- Static & Word Recognition (`test_recognition.py`): 20 / 20 passed
- Arabic Sentence Builder (`test_text_builder.py`): 17 / 17 passed
- FastAPI Web Backend API (`test_web_backend_api.py`): 9 / 9 passed
- Web Recognition Performance (`test_web_recognition_performance.py`): 3 / 3 passed
- Desktop UI & Camera Integration: All passed

---

## 5. Recorded-Video and Live Webcam Test Evidence

1. **Authentic Recorded Video Playback Test:**
   - Video: `01_01_0001_(10_11_16_16_20_30)_c.mp4` (Class `0001`: صفر).
   - Result: Correct prediction **'صفر'** across all active window frames.
   - Confidence: **0.998 on CNN**, **0.835 on SVM**.
2. **Physical Webcam Hardware Test:**
   - Device: Camera index 0 opened successfully (`cap.isOpened() == True`).
   - Frame Acquisition: 15 consecutive live frames acquired ($640 \times 480$).
   - Pipeline Flow: Converted BGR to RGB, extracted MediaPipe landmarks, passed to rolling buffer, executed sliding window inference.
   - Idle Rejection: Blank/resting frames correctly produced `None` (rejected) instead of false positive classifications.
3. **Legacy Custom Dataset & UI Continuity:**
   - 4,119 recorded custom Iraqi sign samples in `data/datasets/` remain intact.
   - Dual-mode launcher (`python run`), PySide6 Desktop, and React + FastAPI Local Web interfaces operate normally.

---

## 6. Deployment Model Compatibility

| Property | Research SVM Model | Research 1D Temporal CNN Model |
|---|---|---|
| **Location** | `data/models/research/svm/` | `data/models/research/cnn/` |
| **Model Weights** | `svm_model.pkl` (1,087 KB) + `scaler.pkl` (16 KB) | `cnn_model.pt` (475 KB) |
| **Input Shape** | 672 dimensions (engineered stats) | $(N, 168, 30)$ (temporal tensor) |
| **Feature Schema** | `hands_only` (Left Hand + Right Hand) | `hands_only` (Left Hand + Right Hand) |
| **Classes Count** | 10 classes (`classes.json`) | 10 classes (`classes.json`) |
| **Training Strategy** | `full_authentic_dataset_all_signers` | `full_authentic_dataset_all_signers` |
| **Deployment Status** | `PRODUCTION_DEPLOYMENT` | `PRODUCTION_DEPLOYMENT` |

---

## 7. Recommended Next Experiment for Unbiased Validation

To establish a definitive, unbiased field estimate without test-guided selection:
1. **Blind Evaluation on New Signers:** Download recordings for `Signer04`, `Signer05`, and `Signer06` from the official KArSL dataset.
2. **Frozen Model Protocol:** Evaluate the existing saved deployment models (`data/models/research/{svm,cnn}`) on the new signers **without modifying** feature schemas, network architectures, or hyperparameters.
3. **Student/Instructor Verification:** Ingest local webcam recordings from 3 independent students in the department using `src/research/evaluate_new_recordings.py`.
4. **Citation Reference:**
   > **Sidig, A. A. I., Luqman, H., Mahmoud, S., & Mohandes, M. (2021).** *KArSL: Arabic Sign Language Database.* ACM Transactions on Asian and Low-Resource Language Information Processing, 20(1), Article 14. DOI: 10.1145/3423420.

---

## 8. Git Commit & Release Reference

- **Final Commit Hash:** `584842a` (and follow-up QA validation commit)
- **Branch:** `main`
- **Remote Repository:** [https://github.com/xdc7-css/Vision-Based-Sign-Language-Recognition](https://github.com/xdc7-css/Vision-Based-Sign-Language-Recognition)
