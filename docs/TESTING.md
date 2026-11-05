# Testing Guide — ArSL Translation Overlay

## Test Suite

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test modules
python -m pytest tests/test_hand_detection.py -v
python -m pytest tests/test_hybrid_pipeline.py -v
python -m pytest tests/test_text_builder.py -v
python -m pytest tests/test_dialect_manager.py -v
python -m pytest tests/test_analytics.py -v

# Run with performance profiling
python -m pytest tests/ --durations=10
```

## Test Structure

| Test File | Module Under Test | Tests |
|-----------|------------------|-------|
| `test_hand_detection.py` | Hand models, renderer, LandmarkProcessor | 36 |
| `test_motion_tracking.py` | Position2D, Velocity2D, MotionTracker | 14 |
| `test_dataset.py` | DatasetCollector, Organizer, Validator | 20 |
| `test_recognition.py` | Static, Dynamic, Word recognizers | 15 |
| `test_hybrid_pipeline.py` | HybridRecognitionPipeline | 12 |
| `test_text_builder.py` | ArabicTextBuilder | 17 |
| `test_dialect_manager.py` | DialectManager, DialectRegistry | 15 |
| `test_analytics.py` | AnalyticsEngine | 13 |
| `test_overlay.py` | OverlayManager, QtOverlayWindow | 8 |

**Total: 150+ tests**

## Writing Tests

```python
# Example: test for ArabicTextBuilder
def test_build_word_from_letters(self):
    builder = ArabicTextBuilder()
    for letter in ["م", "ر", "ح", "ب", "ا"]:
        builder.process(TextCommand.letter(letter))
    assert builder.current_word == "مرحبا"
```

## Coverage Targets

| Module | Target Coverage |
|--------|----------------|
| Domain entities | 100% |
| Infrastructure services | 90%+ |
| Presentation | 60%+ |
| Overall | 80%+ |
