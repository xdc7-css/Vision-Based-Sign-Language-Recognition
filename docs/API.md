# API Reference — ArSL Translation Overlay

## Application Layer

### `AppState` (`src.application.state`)

Central reactive state. All properties emit Qt signals on change.

```python
state = AppState()

# Signals
state.pipeline_status_changed.connect(handler)  # str: idle|streaming|error
state.fps_updated.connect(handler)              # float
state.caption_updated.connect(handler)          # str
state.gesture_detected.connect(handler)         # str, float
state.overlay_visibility_changed.connect(handler)  # bool
state.error_occurred.connect(handler)           # str
```

## Domain Layer

### `RecognitionResult` (`src.domain.recognition`)

```python
@dataclass
class RecognitionResult:
    gesture_id: str
    arabic_text: str
    category: GestureCategory    # STATIC_WORD, DYNAMIC_GESTURE, ALPHABET
    confidence: float
    is_rejected: bool
    rejection_reason: str
    processing_time_ms: float

    @property
    def is_valid(self) -> bool: ...
```

### `Hand` (`src.domain.hand`)

```python
@dataclass
class Hand:
    landmarks: dict[HandLandmarkType, HandLandmark]
    handedness: Handedness
    confidence: float = 1.0

    def __getitem__(self, key: HandLandmarkType) -> HandLandmark: ...
```

### `TextCommand` (`src.domain.text`)

```python
@dataclass
class TextCommand:
    input_type: GestureInputType  # LETTER, WORD, SPACE, DELETE, CLEAR
    value: str
    confidence: float
    gesture_id: str

    @staticmethod
    def letter(text: str, confidence: float = 1.0) -> TextCommand: ...
    @staticmethod
    def word(text: str, confidence: float = 1.0) -> TextCommand: ...
    @staticmethod
    def space() -> TextCommand: ...
    @staticmethod
    def delete() -> TextCommand: ...
    @staticmethod
    def clear() -> TextCommand: ...
```

## Infrastructure Layer

### `HybridRecognitionPipeline` (`src.infrastructure.recognition.hybrid_pipeline`)

```python
pipeline = HybridRecognitionPipeline(
    word_recognizer=WordRecognizer(),
    static_recognizer=StaticSignRecognizer(),
    dynamic_recognizer=DynamicSignRecognizer(),
    config=HybridConfig(...)
)

result: HybridResult = pipeline.process(hand=hand, track=track)
result, command = pipeline.process_to_command(hand=hand, track=track)
```

### `ArabicTextBuilder` (`src.infrastructure.text_builder`)

```python
builder = ArabicTextBuilder()
text = builder.process(TextCommand.letter("ا"))
sentence = builder.process_batch([TextCommand.word("السلام"), TextCommand.space()])
```

### `DialectManager` (`src.infrastructure.dialect_manager`)

```python
dm = DialectManager(pipeline=pipeline, base_path=".")
dm.switch_dialect(DialectId.SAUDI)
result = dm.process(hand=hand, track=track)
```

### `AnalyticsEngine` (`src.infrastructure.analytics`)

```python
engine = AnalyticsEngine(window_size=60)
engine.record_frame()
engine.record_prediction(correct=True, delay_ms=50)
snap: AnalyticsSnapshot = engine.snapshot()
```

### `OverlayManager` (`src.infrastructure.overlay_system`)

```python
overlay = OverlayManager(config=OverlayConfig(...))
overlay.initialize()
overlay.update_text("مرحبا")
overlay.show()
overlay.set_position(200, 300)
overlay.close()
```

### `PerformanceOptimizer` (`src.infrastructure.optimization`)

```python
opt = PerformanceOptimizer()
opt.set_profile(OptimizationLevel.MAX_PERFORMANCE)
if opt.should_process_frame():
    frame, orig = opt.resize_frame(frame)
```
