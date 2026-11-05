# UML-Style Component Design

> Detailed component diagrams and interface contracts for the Real-Time ArSL Translation Overlay System

---

## 1. Component Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION                                     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │                    PIPELINE ORCHESTRATOR                       │      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐ │      │
│  │  │ Capture  │  │Preprocess│  │Recogition│  │  Translation   │ │      │
│  │  │ Stage    │─▶│ Stage    │─▶│ Stage    │─▶│  Stage         │─┐      │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────────────┘ │      │
│  │  ┌────────────────────────────────────────────────────────────┐│      │
│  │  │  Display Stage                                            ││      │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               ││      │
│  │  │  │ Overlay  │  │ Caption  │  │ Control  │               ││      │
│  │  │  │ Window   │  │ Renderer │  │ Panel    │               ││      │
│  │  │  └──────────┘  └──────────┘  └──────────┘               ││      │
│  │  └────────────────────────────────────────────────────────────┘      │
│  └────────────────────────────────────────────────────────────────┘      │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │  Event Bus   │  │   Config     │  │   Model      │                   │
│  │  (Observer)  │  │   Manager    │  │   Manager    │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Package Diagram

```
┌──────────────────────┐     ┌──────────────────────┐
│       capture        │     │    preprocessing      │
│ ┌──────────────────┐ │     │ ┌──────────────────┐ │
│ │ VideoSource (ABC)│ │     │ │ PreprocessStages │ │
│ │ WebcamSource     │ │     │ │ HandDetector     │ │
│ │ FileSource       │ │     │ │ LandmarkExtract  │ │
│ │ ScreenCaptureSrc │ │     │ │ Normalizer       │ │
│ │ BrowserCaptureSrc│ │     │ │ ROIExtractor     │ │
│ └──────────────────┘ │     │ └──────────────────┘ │
└──────────┬───────────┘     └──────────┬───────────┘
           │                            │
           │      ┌─────────────────────┘
           │      ▼
┌──────────┴───────────┐     ┌──────────────────────┐
│        core          │     │     recognition       │
│ ┌──────────────────┐ │     │ ┌──────────────────┐ │
│ │ PipelineOrchestr │◀┼─────┼▶│ RecognitionEngine│ │
│ │ PipelineStage    │ │     │ │ GestureSegmenter│ │
│ │ EventBus         │ │     │ │ Classifiers (3) │ │
│ │ errors           │ │     │ │ DecisionFusion   │ │
│ └──────────────────┘ │     │ └──────────────────┘ │
└──────────────────────┘     └──────────┬───────────┘
                                        │
                                        ▼
┌──────────────────────┐     ┌──────────────────────┐
│     translation      │     │     model_manager     │
│ ┌──────────────────┐ │     │ ┌──────────────────┐ │
│ │ TranslationMapper│ │     │ │ ModelManager     │ │
│ │ GestureVocab     │ │     │ │ ModelPack        │ │
│ │ ContextManager   │ │     │ │ ModelRegistry    │ │
│ │ CaptionBuffer    │ │     │ └──────────────────┘ │
│ │ DialectPack      │ │     └──────────────────────┘
│ └──────────────────┘ │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     ┌──────────────────────┐
│       overlay        │     │         ui           │
│ ┌──────────────────┐ │     │ ┌──────────────────┐ │
│ │ DisplayManager   │ │     │ │ ControlPanel     │ │
│ │ OverlayWindow    │ │     │ │ SystemTray      │ │
│ │ CaptionRenderer  │ │     │ │ SettingsDialog  │ │
│ │ OverlayRenderer  │ │     │ │ Widgets         │ │
│ └──────────────────┘ │     │ └──────────────────┘ │
└──────────────────────┘     └──────────────────────┘
```

---

## 3. Interface Contracts

### 3.1 VideoSource (ABC) — `src/capture/video_source.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SourceProperties:
    width: int
    height: int
    fps: float
    backend: str
    is_color: bool = True

@dataclass
class Frame:
    id: int
    timestamp: float
    source_name: str
    data: 'np.ndarray'  # RGB, H×W×3

class VideoSource(ABC):
    """Abstract base for all video capture sources."""
    
    @abstractmethod
    def open(self, uri: str) -> bool:
        """Open video source. Returns True on success."""
        ...
    
    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Read next frame. Returns None on EOF or error."""
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...
    
    @abstractmethod
    def get_properties(self) -> SourceProperties:
        """Return current source properties."""
        ...
    
    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """True if source is connected and producing frames."""
        ...
```

### 3.2 GestureClassifier (ABC) — `src/recognition/classifiers/base_classifier.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class ClassificationResult:
    gesture_id: str
    confidence: float
    classifier_type: str  # 'static', 'dynamic', 'alphabet'
    processing_time_ms: float

class GestureClassifier(ABC):
    """Abstract base for all gesture classifiers."""
    
    @abstractmethod
    def classify(self, input_data: 'PreprocessedFrame | LandmarkSequence') -> Optional[ClassificationResult]:
        """Classify a single input. Returns None if below threshold."""
        ...
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load model weights from path. Returns True on success."""
        ...
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload model and free resources."""
        ...
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True if model is loaded and ready."""
        ...
```

### 3.3 GestureSegmenter (ABC) — `src/recognition/segmenter/gesture_segmenter.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class GestureSegment:
    start_frame_id: int
    end_frame_id: int
    start_time: float
    end_time: float
    landmark_sequence: 'list'  # List of landmark arrays

class GestureSegmenter(ABC):
    """Analyzes streaming landmarks to detect gesture boundaries."""
    
    @abstractmethod
    def feed(self, frame: 'PreprocessedFrame') -> Optional[GestureSegment]:
        """Feed next frame. Returns completed GestureSegment when boundary detected."""
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Clear internal state."""
        ...
```

### 3.4 DialectPack (ABC) — `src/translation/dialect_pack.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class DialectInfo:
    dialect_id: str
    name: str
    name_ar: str
    version: str
    has_static: bool
    has_dynamic: bool
    has_alphabet: bool
    vocabulary_size: int

class DialectPack(ABC):
    """Encapsulates all models and vocabulary for one dialect."""
    
    @property
    @abstractmethod
    def dialect_id(self) -> str:
        ...
    
    @abstractmethod
    def load_models(self, model_manager: 'ModelManager') -> bool:
        """Load all required models for this dialect."""
        ...
    
    @abstractmethod
    def load_vocabulary(self) -> 'GestureVocab':
        """Load gesture→text mapping."""
        ...
    
    @abstractmethod
    def unload(self) -> None:
        """Unload all models and resources."""
        ...
```

### 3.5 ConfigProvider (ABC) — `src/config/providers/`

```python
from abc import ABC, abstractmethod

class ConfigProvider(ABC):
    """Abstract source of configuration values."""
    
    @abstractmethod
    def load(self) -> dict:
        """Load configuration as flat or nested dictionary."""
        ...
    
    @abstractmethod
    def get_source_priority(self) -> int:
        """Priority order (higher = overrides lower)."""
        ...
```

### 3.6 OverlayRenderer (ABC) — `src/overlay/renderers/abstract_renderer.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class OverlayConfig:
    opacity: float
    position: str
    font_family: str
    font_size: int
    font_color: str
    background_color: str
    max_lines: int
    x: int = 0
    y: int = 0

class OverlayRenderer(ABC):
    """Platform-specific transparent overlay implementation."""
    
    @abstractmethod
    def create_window(self, config: OverlayConfig) -> bool:
        """Create overlay window. Returns True on success."""
        ...
    
    @abstractmethod
    def update_caption(self, text: str) -> None:
        """Update displayed caption text."""
        ...
    
    @abstractmethod
    def set_visible(self, visible: bool) -> None:
        """Show or hide overlay window."""
        ...
    
    @abstractmethod
    def destroy(self) -> None:
        """Destroy overlay window and release resources."""
        ...
```

---

## 4. State Machine Diagrams

### 4.1 Pipeline Orchestrator

```
         ┌──────────────────────────────────────────────┐
         │                                              │
         ▼                                              │
    ┌─────────┐    init()    ┌──────────────┐           │
    │  IDLE   │─────────────▶│ INITIALIZING  │           │
    └─────────┘              └──────┬───────┘           │
         ▲                          │                   │
         │              ┌───────────┼───────────┐       │
         │              │ fail      │ success   │       │
         │              ▼           ▼           │       │
         │         ┌────────┐ ┌──────────┐      │       │
         │         │ ERROR  │ │  READY   │      │       │
         │         └───┬────┘ └─────┬────┘      │       │
         │             │            │ start()   │       │
         │             │            ▼           │       │
         │             │       ┌──────────┐     │       │
         │             │       │ RUNNING  │     │       │
         │             │       └────┬─────┘     │       │
         │             │            │           │       │
         │             │    ┌───────┼───────┐   │       │
         │             │    │ pause │ error │   │       │
         │             │    ▼       ▼       │   │       │
         │             │ ┌──────┐ ┌────────┐│   │       │
         │             │ │PAUSED│ │ERROR   ││   │       │
         │             │ └──┬───┘ └────────┘│   │       │
         │             │    │ resume        │   │       │
         │             └────┼───────────────┘   │       │
         │                  ▼                   │       │
         │             ┌──────────┐             │       │
         │             │ STOPPING │◄────────────┘       │
         │             └────┬─────┘                     │
         │                  │ done                      │
         └──────────────────┘───────────────────────────┘
```

### 4.2 Gesture Segmenter State Machine

```
         ┌────────────────────────────────────┐
         │   Landmark stream continues...      │
         │                                    │
         ▼                                    │
    ┌─────────┐   motion > threshold    ┌──────────────┐
    │  IDLE   │────────────────────────▶│ IN_PROGRESS  │
    │         │◀────────────────────────│              │
    └─────────┘   motion < threshold    └──────┬───────┘
         ▲        for min_duration             │
         │                            ┌────────┴────────┐
         │                            │                 │
         │              segment       │  motion <       │
         │              complete      │  threshold      │
         │                            │  for hold_time  │
         │                            ▼                 │
         │                     ┌──────────────┐        │
         │                     │  COMPLETED   │────────┘
         │                     │ (emit event) │
         │                     └──────────────┘
         │
         └─ reset() → IDLE
```

---

## 5. Event Catalog

| Event | Payload | Publisher | Subscribers |
|-------|---------|-----------|-------------|
| `PipelineStateChanged` | `{old_state, new_state}` | PipelineOrchestrator | ControlPanel, DisplayManager |
| `SourceConnected` | `{source_id, properties}` | CaptureManager | ControlPanel |
| `SourceDisconnected` | `{source_id, reason}` | CaptureManager | PipelineOrchestrator, ControlPanel |
| `GestureDetected` | `RecognitionResult` | RecognitionEngine | TranslationMapper |
| `GestureSegmentReady` | `GestureSegment` | GestureSegmenter | Classifiers |
| `CaptionUpdated` | `CaptionEvent` | TranslationMapper | DisplayManager |
| `ModelLoaded` | `{model_id, dialect, type}` | ModelManager | RecognitionEngine, ControlPanel |
| `ModelError` | `{model_id, error}` | ModelManager | PipelineOrchestrator, ControlPanel |
| `ConfigChanged` | `{key, old_value, new_value}` | ConfigManager | All subscribed modules |
| `PipelineError` | `{stage, error, recoverable}` | Any stage | PipelineOrchestrator, ControlPanel |
| `FpsUpdated` | `{fps, stage}` | Any stage | ControlPanel (debug) |

---

## 6. Data Flow Matrix

```
Stage          Input                       Output                      Queue
──────────────────────────────────────────────────────────────────────────────
Capture        Video source (hw/sw)        Frame                       FrameQueue
Preprocess     Frame                       PreprocessedFrame           FeatureQueue
Segmenter      PreprocessedFrame           GestureSegment | None       (internal)
Static Cls     PreprocessedFrame           ClassificationResult        (internal)
Dynamic Cls    GestureSegment              ClassificationResult        (internal)
Alphabet Dec   ClassificationResult[]      ClassificationResult        (internal)
DecisionFusion ClassificationResult[]      RecognitionResult           ResultQueue
Translate      RecognitionResult           CaptionEvent                CaptionQueue
Display        CaptionEvent                (overlay update)            (direct)
```

---

## 7. Dependency Injection Topology

```
Application
├── PipelineOrchestrator
│   ├── CaptureManager
│   │   └── VideoSource[] (plugins)
│   ├── PreprocessPipeline
│   │   └── HandDetector
│   ├── RecognitionEngine
│   │   ├── GestureSegmenter
│   │   ├── GestureClassifier[] (static, dynamic, alphabet)
│   │   └── DecisionFusion
│   ├── TranslationMapper
│   │   ├── GestureVocab
│   │   ├── ContextManager
│   │   └── CaptionBuffer
│   └── DisplayManager
│       └── OverlayRenderer
├── EventBus
├── ConfigManager
│   └── ConfigProvider[] (yaml, env, cli)
└── ModelManager
    └── ONNX Runtime Sessions
```

All dependencies are injected at construction time. No global state. Each module receives its dependencies explicitly via `__init__` parameters, making components testable in isolation.
