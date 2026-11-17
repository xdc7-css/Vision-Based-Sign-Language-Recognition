# KArSL-502 Real Data Validation & Scientific Reproducibility Audit

**Project:** Vision-Based Sign Language Recognition (ArSL Studio)  
**Institution:** University of Basrah — Computer Engineering Department  
**Supervisor:** Dr. Hiba Hakim Abdulzahra  
**Date:** November 2026  
**Status:** VALIDATED & BENCHMARKED ON AUTHENTIC DATA  

---

## 1. Executive Summary & Provenance Audit

### 1.1 The Critical Provenance Concern
The previous evaluation report claimed:
- 12 KArSL gesture classes.
- 360 sign sequences across 3 signers.
- SVM Accuracy: 67.50%.
- 1D Temporal CNN Accuracy: 100.00%.

A deep audit of `src/research/karsl_dataset.py`, `data/research/karsl/features/karsl_features.npz`, and the filesystem was conducted.

### 1.2 Audit Findings
1. **Source of the Previous 360 Samples:** The samples were generated completely in-memory using synthetic trigonometric functions (`_generate_kinematic_trajectory()` simulating sinusoidal trajectory paths). **Zero genuine video files or real signer recordings were involved.**
2. **Artificial Perfect Accuracy:** The 100% CNN accuracy was an artifact of fitting low-noise synthetic sine wave equations, not human gesture recognition.
3. **Audit Counts:**
   - Real KArSL recordings in old dataset: **0 (0.0%)**
   - Synthetically generated trajectories: **360 (100.0%)**
   - Verified authentic signer videos: **0**
4. **Corrective Remediation:**
   - The synthetic dataset was quarantined and preserved solely for non-benchmark synthetic development testing: `data/research/karsl/features/karsl_synthetic_dev_features.npz`.
   - The misleading 100% CNN claims were purged from all documentation, UI components, and evaluation reports.
   - All models, feature arrays, and UI displays were re-engineered to require authentic video data with provenance tracking.

---

## 2. Authentic KArSL-502 Dataset Acquisition

To fulfill the supervisor's requirements for genuine scientific reproducibility, official recordings from the **KArSL-502** dataset were downloaded directly from the official archives (Hamzah Luqman et al., King Fahd University of Petroleum & Minerals):

- **Citation:** Sidig, A. A. I., Luqman, H., & Mahmoud, S. (2021). *KArSL: Arabic Sign Language Database*. ACM Transactions on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT).
- **Video Archives Acquired:**
  - `Signer01`: Archive `signer01_test_0001_0070.7z` (Google Drive ID: `1WGmnMWPx0cCpoaAhfJyuEVa4jyLYmQQq`)
  - `Signer02`: Archive `signer02_test_0001_0070.7z` (Google Drive ID: `1o8FfiIjvlbq7msw_Uga39zCX29OvMQxk`)
  - `Signer03`: Archive `signer03_test_0001_0070.7z` (Google Drive ID: `1InT6ZHjRoo3Fkoru6Yy6gL8uToGsp_Nb`)

### 2.1 Selected Classes & Volume
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

- **Format:** RGB MP4, $1920 \times 1080$ resolution, $30.0$ frames per second.
- **Manifest:** Saved at `data/research/karsl/manifests/real_10_signs.csv` with SHA-256 hashes, frame counts, video dimensions, durations, and official provenance references for all 240 files.
- **Contact Sheet:** Generated visual preview at `data/research/karsl/previews/real_karsl_contact_sheet.png` confirming real human signers performing signs under studio conditions.
- **Git Policy:** In accordance with copyright and file-size constraints, raw video files are excluded via `.gitignore` (`data/research/karsl/raw/`), while the manifest, features, models, and evaluation outputs are fully version-controlled.

---

## 3. MediaPipe Multimodal Feature Extraction

Using the project's standard computer vision extraction pipeline (`src/research/extract_real_karsl_features.py`):
1. **Frame Decoding:** OpenCV decoded 5,420 total frames across the 240 genuine video files.
2. **Landmark Extraction:** MediaPipe extracted 3D coordinates:
   - Left Hand: 21 landmarks $(x, y, z)$
   - Right Hand: 21 landmarks $(x, y, z)$
   - Selected Facial Mesh Landmarks: Key expressive reference points
   - Feature Dimension: $736$ channels per frame, including explicit presence/missing-data masks.
3. **Temporal Normalization:** Each sequence was resampled via linear interpolation to a uniform length of $T = 30$ frames (1 second at 30 fps).
4. **Extracted Feature Artifact:**
   - Saved to `data/research/karsl/features/karsl_features.npz` (also mirrored at `karsl_real_features.npz`).
   - Tensor shape: `(240, 30, 736)`.
   - Verified data authenticity: `is_authentic = True`, `data_source = "karsl_502_authentic_video"`.

---

## 4. Scientific Empirical Evaluation: Leave-One-Signer-Out (LOSO)

To ensure zero data leakage and strict scientific validity, the evaluation used Leave-One-Signer-Out (LOSO) cross-validation across the three human signers.

### 4.1 Benchmark Protocol
In each fold:
- Training: 160 sequences from 2 distinct signers (80 videos $\times$ 2).
- Testing: 80 sequences from the remaining completely unseen held-out signer.
- Identical data partitions, feature matrices, and evaluation criteria were provided to both Model A (SVM) and Model B (1D Temporal CNN).

### 4.2 LOSO 3-Fold Empirical Results

| Fold | Held-Out Test Signer | Train Signers | Model A: SVM Accuracy | Model A: SVM Macro F1 | Model B: 1D CNN Accuracy | Model B: 1D CNN Macro F1 |
|---|---|---|---|---|---|---|
| **Fold 1** | `Signer01` | `Signer02`, `Signer03` | 32.50% | 26.02% | **41.25%** | **28.11%** |
| **Fold 2** | `Signer02` | `Signer01`, `Signer03` | 26.25% | 25.66% | **36.25%** | **35.70%** |
| **Fold 3** | `Signer03` | `Signer01`, `Signer02` | **27.50%** | **17.64%** | 25.00% | 17.45% |
| **Mean** | **3-Fold Average** | — | **28.75%** | **23.11%** | **34.17%** | **27.09%** |

### 4.3 Computational Efficiency & Deployment Comparison

| Metric | Model A: SVM (RBF Kernel) | Model B: 1D Temporal CNN | Delta / Advantage |
|---|---|---|---|
| **Model Architecture** | Scikit-Learn SVC ($C=10.0, \gamma=\text{'scale'}$) | PyTorch Conv1D + BatchNorm + Dropout + Linear | — |
| **Feature Input** | 2,208-dim statistical/temporal features | 30 frames $\times$ 736 channels temporal tensor | — |
| **Mean Accuracy (LOSO)** | 28.75% | **34.17%** | **+5.42% CNN advantage** |
| **Mean Macro F1 (LOSO)** | 23.11% | **27.09%** | **+3.98% CNN advantage** |
| **Training Time (sec)** | **0.20 sec** | 3.75 – 5.73 sec | **SVM is 27× faster** |
| **Inference Latency (ms)** | **0.73 – 0.85 ms** | 1.14 – 1.54 ms | **SVM is 1.6× faster** |
| **Checkpoint Size (KB)** | **2,775.6 KB** | 3,147.6 KB | **SVM is 12% smaller** |

### 4.4 Scientific Discussion & Observations
1. **Genuine Inter-Signer Generalization:** Unlike synthetic benchmarks that artificially achieve 100%, authentic sign language video exhibits natural anatomical variations, execution speeds, and slight viewpoint shifts. On a 10-class problem, random chance is $10.00\%$. Both SVM ($28.75\%$) and CNN ($34.17\%$) achieve nearly $3\times$ to $3.5\times$ random chance under strict zero-shot signer generalization.
2. **Why 1D CNN Outperforms SVM:** The 1D Temporal CNN leverages convolutional kernels across the 30 time steps, capturing dynamic inter-frame transitions and trajectory changes between digits (e.g., distinguishing digit $1$ vs digit $2$).
3. **Why SVM Remains Valuable:** SVM trains in a fraction of a second ($0.2$s) and runs at sub-millisecond inference speeds ($0.73$ ms), making it viable for low-power edge deployment or rapid user calibration.
4. **Kinect V2 vs RGB Domain Shift:** KArSL videos were recorded with high-angle Microsoft Kinect V2 cameras. Applying MediaPipe RGB landmark estimation to high-angle Kinect streams induces landmark jitter, which explains why cross-signer transfer is challenging without fine-tuning.

---

## 5. Required Final Audit Status Table

As requested by the graduation project supervisor, the table below provides direct verification status and evidence:

| Requirement | Status | Evidence |
|---|---|---|
| **10 authentic KArSL classes** | **PASS** | Manifest `data/research/karsl/manifests/real_10_signs.csv`, 10 official classes (0001–0010: صفر to تسعة) |
| **Real videos imported** | **PASS** | 240 authentic MP4 video files with verified SHA-256 hashes across `Signer01`, `Signer02`, `Signer03` |
| **MediaPipe features extracted** | **PASS** | `data/research/karsl/features/karsl_features.npz` (240 sequences, 30 frames, 736 dimensions) |
| **SVM trained on real data** | **PASS** | `data/models/research/svm/svm_model.pkl` (2,775.6 KB, 10 classes, verified fit on authentic data) |
| **CNN trained on real data** | **PASS** | `data/models/research/cnn/cnn_model.pt` (3,147.6 KB, 10 classes, PyTorch TemporalSignCNN checkpoint) |
| **Same test recordings** | **PASS** | `data/research/karsl/splits/loso_splits.json` (exact identical splits used for both SVM and CNN) |
| **Signer-independent evaluation** | **PASS** | 3-Fold Leave-One-Signer-Out (LOSO) cross-validation with zero train/test leakage |
| **Real comparison metrics** | **PASS** | `data/research/karsl/eval/comparison_metrics.csv`, `comparison_metrics.json`, and `confusion_matrices.json` |
| **Desktop integration** | **PASS** | PySide6 desktop application reflects genuine KArSL metadata, dynamic winner rationale, and provenance |
| **Web integration** | **PASS** | React frontend build passed cleanly (`web/dist/`), Train Model & Recognize pages show authentic badge |
| **Legacy data preserved** | **PASS** | All 4,119 recorded custom Iraqi sign samples in `data/datasets/` and trained models remain intact |

---

## 6. System & Data Integrity Verification

- **Custom Dataset Preservation:**
  - `data/datasets/words/`: All recorded static training samples preserved.
  - `data/datasets/dynamic/`: All recorded dynamic gesture sequences preserved.
  - Custom Iraqi Classifier (`data/models/custom/`): Verified functional and independent of research models.
- **Automated Test Coverage:**
  - `tests/test_research_karsl_pipeline.py`: All 9 research pipeline unit and integration tests passing.
  - `tests/test_recognition_custom_and_landmarks.py`: All 5 live recognition tests passing.
  - Frontend TypeScript/React compilation: Clean build with zero TypeScript or packaging errors.

