# Technical Specification

> **System Designation:** Real-Time Arabic Sign Language Translation Overlay System  
> **Document Version:** 2.0 (Updated Post-KArSL-502 Integration & PySide6 Migration)  
> **Target Platform:** Windows 10 / 11 (64-bit)  
> **Implementation Scope:** Computer Vision Ingestion, MediaPipe Tasks, ONNX ST-Transformer, Scikit-Learn Custom Models, PySide6 GUI, and Win32 Subtitle Overlay.

---

## 1. Technology Stack

### 1.1 Core Production Stack

| Component | Verified Technology | Actual Version | Architectural Rationale & Implementation Notes |
| :--- | :--- | :--- | :--- |
| **Language Runtime** | Python | `3.12.4 (64-bit)` | Compatible with `>=3.10`. High-performance C-extensions for CV and ML. |
| **Desktop GUI** | `PySide6` (Qt 6) | `>=6.5.0` (`6.7+`) | Official Qt for Python bindings; hardware-accelerated viewports and thread isolation via `QThread`. *(Note: Superseded initial PyQt6 draft)*. |
| **Computer Vision** | `opencv-python` | `>=4.8.0` (`4.10.0`) | Camera frame acquisition via Windows DirectShow (`cv2.CAP_DSHOW`), color space conversion (BGR $\to$ RGB). |
| **Keypoint Tracking**| `mediapipe` | `>=0.10.0` (`0.10.14`) | Google MediaPipe Tasks Vision API (`hand_landmarker.task`, `pose_landmarker.task`, `face_landmarker.task`). |
| **Foundation Inference**| `onnxruntime` | `>=1.15.0` (`1.18.0`) | Cross-platform, hardware-accelerated ONNX inference for the 502-class ST-Transformer model. |
| **Sequence Modeling** | `torch`, `torchvision` | `>=2.0.0` (`2.4.0`) | PyTorch sequence architectures (`SignTransformer`, `LSTMClassifier`, `GRUClassifier`) in `train_dynamic.py`. |
| **Custom Classifiers**| `scikit-learn` | `>=1.3.0` (`1.5.0`) | Local dialect static classifiers (`MLPClassifier`, `SVC`, `RandomForestClassifier`). |
| **Model Serialization**| `joblib` | `>=1.3.0` (`1.4.2`) | Checkpoint persistence for Scikit-Learn models and feature scalers. |
| **Screen Ingestion** | `mss` | `>=9.0.0` (`9.0.1`) | High-speed multi-monitor frame grabbing for desktop translation. |
| **Window Ingestion** | `pywin32` | `>=306` | Native Windows GDI capture (`win32gui`, `win32ui`) and layered window click-through styles. |
| **Arabic Glyph Shaping**| `arabic-reshaper` | `>=2.1.0` (`3.0.0`) | Reconstructs context-sensitive Arabic cursive connections. |
| **Bidirectional Layout**| `python-bidi` | `>=0.4.0` (`0.6.6`) | Formats text according to the Unicode Bidirectional Algorithm (UBA) for Right-to-Left (RTL) display. |
| **Text-to-Speech** | `pyttsx3` | `>=2.90` (`2.98`) | Offline speech synthesis with automated Windows SAPI5 Arabic voice discovery. |
| **Telemetry & Metrics**| `psutil` | `>=5.9.0` (`5.9.8`) | CPU load and RAM memory monitoring for the Dashboard tab. |
| **Configuration** | `PyYAML` | `>=6.0` (`6.0.2`) | Human-readable hierarchical configuration (`default.yaml`, `user.yaml`, `logging.yaml`). |

---

## 2. Model & Recognition Specifications

The system implements a **Dual-Engine Architecture**:
1. **Foundation Model (KArSL-502):** High-capacity multimodal Spatio-Temporal Transformer.
2. **Custom Dialect Models:** Lightweight landmark-based classifiers for localized signs.

```
+-----------------------------------------------------------------------------------+
|                            RECOGNITION MODEL SPECIFICATIONS                       |
+----------------------+-----------------------------+------------------------------+
| Attribute            | KArSL-502 Foundation Model  | Custom Dialect Static Model  |
+----------------------+-----------------------------+------------------------------+
| Model File           | data/models/karsl/          | data/models/iraqi/           |
|                      | karsl_502.onnx              | static_model.pkl             |
| File Size            | 7,330,073 bytes (7.33 MB)   | 209,742 bytes (209 KB)       |
| Architecture         | ST-Transformer (30 epochs)  | Multi-Layer Perceptron (MLP) |
| Fallback Arch        | —                           | SVM (RBF kernel) / Rand Forest|
| Input Tensor Shape   | (1, 50, 736) float32        | (1, 63) float32              |
| Feature Composition  | 184 keypoints x 4 (x,y,z,v) | 21 hand landmarks x 3 (x,y,z)|
|                      | • Pose: 6 keypoints (24 dim)| • 21 Hand Landmarks          |
|                      | • Face: 136 keypoints(544)  | • Wrist-centered origin      |
|                      | • Right Hand: 21 keypoints  | • Bounding-radius normalized |
|                      | • Left Hand: 21 keypoints   |                              |
| Temporal Sampling    | TSNSampler (15-50 -> 50)    | Instantaneous spatial frame  |
| Output Classes       | 502 discrete sign classes   | 3 verified classes (Iraqi)   |
| Inference Framework  | ONNX Runtime 1.18.0         | Scikit-Learn 1.5.0           |
| Mean Latency (CPU)   | 45 - 65 ms                  | < 2 ms                       |
+----------------------+-----------------------------+------------------------------+
```

### 2.1 KArSL-502 ST-Transformer Specification
- **Input Dimension:** $\mathbf{X} \in \mathbb{R}^{1 \times 50 \times 736}$
  - Temporal dimension: Exactly 50 frames, resampled via `TSNSampler`.
  - Spatial dimension: 736 features derived from 184 normalized landmarks:
    - **Body Pose (6 landmarks = 24 floats):** Left/Right Shoulders, Elbows, Wrists. Centered on the Nose; normalized by shoulder width.
    - **Face Mesh (136 landmarks = 544 floats):** Face oval contours, lips, and eyes. Centered on the Nose tip; normalized by eye distance.
    - **Right Hand (21 landmarks = 84 floats):** Centered on Right Wrist; normalized by palm span.
    - **Left Hand (21 landmarks = 84 floats):** Centered on Left Wrist; normalized by palm span.
- **Output Dimension:** Logit vector $\mathbf{z} \in \mathbb{R}^{1 \times 502}$.
  - Softmax probabilities: $P_k = \frac{e^{z_k}}{\sum_{j=1}^{502} e^{z_j}}$
  - Index Ranges:
    - `0 – 30`: Arabic Numerals (0 to 30)
    - `31 – 69`: Arabic Alphabet Letters
    - `70 – 501`: Vocabulary Words & Expressions
- **Execution Strategy:** Dedicated background worker (`KArSLWorker`) isolated from the UI thread, utilizing bounded queue-dropping to maintain camera fluidity.

### 2.2 Custom Dialect Models (Iraqi Dialect)
- **Feature Vector:** 21 hand landmarks $\times 3 \text{ spatial coordinates } (x, y, z) = 63 \text{ features}$.
- **Normalization (`LandmarkProcessor`):**
  $$\vec{p}'_i = \vec{p}_i - \vec{p}_{\text{wrist}}, \quad \vec{p}''_i = \frac{\vec{p}'_i}{\max_j \|\vec{p}'_j\| + \epsilon}$$
- **Current Active Vocabulary:** `["السلام_عليكم", "لا", "نعم"]`.
- **Training Pipeline (`src/models/train_static.py`):**
  - Automates training of Random Forest (`n_estimators=100`), SVM (`C=1.0`, RBF kernel, Platt probability scaling), and MLP (`hidden_layer_sizes=(128, 64)`, ReLU, Adam optimizer).
  - Data Augmentation: Landmark Gaussian jitter ($\sigma = 0.01$) with default 3x sample multiplier.
- **Evaluation Status:** Holdout test accuracy of 1.0 (100%) recorded in `data/models/iraqi/evaluation.json` on the 177 captured samples. *This reflects consistent local conditions, not multi-user open-world generalization.*

### 2.3 Custom Dynamic Sequence Models (`src/models/train_dynamic.py`)
- **Input:** Sequences of landmark vectors: $\mathbf{X}_{\text{seq}} \in \mathbb{R}^{B \times T \times 63}$, where $T = 30$ (default sequence length).
- **Supported PyTorch Architectures:**
  1. `SignTransformer`: Multi-head self-attention encoder with positional encodings.
  2. `LSTMClassifier`: Bidirectional 2-layer LSTM with hidden size 128.
  3. `GRUClassifier`: 2-layer Gated Recurrent Unit with hidden size 128.
- **Current Status:** Training architecture verified; awaiting expansion of dynamic training samples beyond the single recorded sample in `data/datasets/dynamic/السلام_عليكم/`.

### 2.4 Legacy / Superseded Design Notice
> [!NOTE]
> **Superseded Architectural Designs:**
> - **MobileNetV3 RGB Image Classifier:** An early design proposal outlined direct 224x224 RGB image classification. This approach was superseded by coordinate landmark-based pipelines (63-dim hand vectors and 736-dim holistic vectors), which offer superior lighting invariance, reduced CPU footprint, and zero training data requirements for the 502 pretrained signs.
> - **PyQt6 Framework Reference:** All presentation modules have been standardized on **PySide6** (Qt 6.7+).

---

## 3. Computer Vision & Ingestion Specifications

### 3.1 MediaPipe Tasks Vision Configuration
- **Model Storage:** `data/models/karsl/`
- **Execution Mode:** `LIVE_STREAM` with asynchronous listener callbacks.
- **Monotonic Timestamp Invariant:**
  Under Windows DirectShow capture, frame arrival times can have microsecond jitter. The compatibility wrapper ensures strictly monotonic timestamps:
  $$t_{\text{dispatched}} = \max\left(t_{\text{now\_micros}},\, t_{\text{previous}} + 1\,\mu\text{s}\right)$$
- **Tracking Confidence Thresholds:**
  - `min_hand_detection_confidence`: $0.5$ (configurable in `config/default.yaml`)
  - `min_hand_presence_confidence`: $0.5$
  - `min_tracking_confidence`: $0.5$

### 3.2 Video Capture Adapters (`src/infrastructure/video_source_engine.py`)

| Adapter | Technical Implementation | Target Performance | Use Case |
| :--- | :--- | :--- | :--- |
| **`WebcamAdapter`** | `cv2.VideoCapture(index, cv2.CAP_DSHOW)` | 640x480 @ 30 FPS | Primary real-time webcam stream. |
| **`FileAdapter`** | `cv2.VideoCapture(file_path)` | Variable (Native FPS) | Recorded benchmark video evaluation. |
| **`ScreenAdapter`** | `mss.mss().grab(monitor_rect)` | 15–30 FPS | Full-screen or desktop region translation. |
| **`WindowAdapter`** | `win32gui`, `win32ui`, `win32con` GDI bitblt | 15–30 FPS | Targeting Zoom, Teams, or browser windows. |

---

## 4. NLP & Text Formatting Specifications

### 4.1 Temporal Stabilization & Sentence Building
Operates via a state machine in `src/infrastructure/text_builder.py`:
1. **Stabilization Threshold ($N = 3$):** A candidate sign must be predicted identically across 3 consecutive evaluation cycles before being accepted.
2. **Cooldown Interlock ($\Delta t = 1.5\text{ s}$):** Suppresses immediate re-emission of the same word while the user holds their hand in position.
3. **Punctuation & Sentence Assembly:** Words are accumulated into a sentence buffer. A configurable pause ($> 2.5\text{ s}$) appends sentence-ending punctuation.

### 4.2 Arabic BiDi Layout & Glyph Shaping
- **Shaping Step:** `arabic_reshaper.reshape(text)` transforms isolated characters into contextually joined cursive glyphs (Initial, Medial, Final, Isolated).
- **Layout Step:** `bidi.algorithm.get_display(reshaped_text)` reverses visual string ordering to ensure correct Right-to-Left (RTL) flow across Qt labels and Windows GDI overlays.

---

## 5. Overlay Subsystem Specifications

### 5.1 Windows Native Layered Subtitle Overlay
- **Target OS:** Windows 10 / 11.
- **Window Extended Styles:**
  ```c
  SetWindowLong(hwnd, GWL_EXSTYLE, 
      GetWindowLong(hwnd, GWL_EXSTYLE) | 
      WS_EX_LAYERED | 
      WS_EX_TRANSPARENT | 
      WS_EX_TOPMOST
  );
  ```
- **Transparency & Click-Through Behavior:**
  - `WS_EX_LAYERED`: Enables per-pixel alpha transparency ($0.0 \to 1.0$).
  - `WS_EX_TRANSPARENT`: Directs the OS Desktop Window Manager (DWM) to ignore all mouse events on the overlay window, passing clicks directly through to underlying meeting software.
  - `WS_EX_TOPMOST`: Ensures subtitles remain visible above full-screen presentations.

### 5.2 Cross-Platform Qt Frameless Overlay
- **Fallback OS:** macOS / Linux.
- **Window Flags:** `Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow`
- **Window Attributes:** `Qt.WA_TranslucentBackground`, `Qt.WA_ShowWithoutActivating`

---

## 6. Performance & Latency Benchmarks

| Milestone / Metric | Target Threshold | Actual Verified Performance | Measurement Methodology |
| :--- | :--- | :--- | :--- |
| **Webcam Stream Ingestion** | $\ge 25\text{ FPS}$ | **30 FPS (~33.3 ms/frame)** | `VideoSourceEngine` DirectShow backend |
| **MediaPipe Multimodal Extraction** | $< 25\text{ ms}$ | **12 – 18 ms** | Multi-threaded CPU keypoint inference |
| **KArSL ST-Transformer Inference** | $< 80\text{ ms}$ | **45 – 65 ms** | ONNX Runtime (CPU execution provider) |
| **Custom Static MLP Inference** | $< 5\text{ ms}$ | **< 2 ms** | Scikit-Learn `predict_proba` on 63 features |
| **Text Stabilization & BiDi Shaping** | $< 5\text{ ms}$ | **< 1 ms** | Native string algorithm execution |
| **Win32 Overlay Redraw** | $< 10\text{ ms}$ | **2 – 4 ms** | Direct GDI layered window update |
| **End-to-End Latency (Capture $\to$ Caption)**| $< 150\text{ ms}$ | **66 – 101 ms** | Total pipeline latency |
| **Cold Startup Time** | $< 6.0\text{ s}$ | **~3.8 s** | Process launch to UI readiness |
| **Host Memory Footprint (RAM)** | $< 1.5\text{ GB}$ | **~650 – 850 MB** | Full session with KArSL model in memory |

---

## 7. Threading & Concurrency Architecture

```
+-----------------------------------------------------------------------------------+
| APPLICATION PROCESS                                                               |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Main GUI Thread (PySide6 Event Loop)                                        |  |
|  | - Viewport rendering (OpenGL / QPainter)                                    |  |
|  | - Tab interactions (Dataset, Training, Settings, Analytics)                 |  |
|  | - Overlay window position and alpha updates                                 |  |
|  +-----------------------------------------------------------------------------+  |
|                                          ▲                                        |
|                          Qt Signals      │  predictionSignal(id, conf)            |
|                                          │                                        |
|  +-----------------------------------------------------------------------------+  |
|  | KArSLWorker Thread (QThread)                                                |  |
|  | - Dispatched RGB frames via bounded queue (size = 2)                        |  |
|  | - Frame-dropping logic (drops oldest frame when busy)                       |  |
|  | - MediaPipe Holistic Landmark Tracking                                      |  |
|  | - TSNSampler Temporal Window Interpolation                                  |  |
|  | - ONNX ST-Transformer Inference Session                                     |  |
|  +-----------------------------------------------------------------------------+  |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Asynchronous Audio Thread (pyttsx3 Daemon)                                  |  |
|  | - Non-blocking Arabic spoken word synthesis                                 |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 8. Test Suite Verification

The project enforces automated regression and contract verification via `pytest`:
- **Total Test Count:** **208 unit and integration tests**
- **Pass Rate:** **100% (208 passed, 0 failures, 2 warnings in 81.20s)**
- **Coverage Domains:**
  - `tests/test_karsl.py`: ONNX tensor dimensions, TSNSampler interpolation, multimodal feature packing.
  - `tests/test_hand_detection.py`: LandmarkProcessor normalization, monotonic timestamps, MediaPipe adapter.
  - `tests/test_recognition.py`: Classifier contracts, confidence thresholds, fallback behavior.
  - `tests/test_text_builder.py`: Arabic character reshaping, BiDi reordering, stabilization queues.
  - `tests/test_overlay.py`: Overlay window creation, opacity calculations, positioning.
  - `tests/integration/`: End-to-end pipeline integration and latency benchmarks.

---

*Document updated and verified against repository state: November 2026.*
