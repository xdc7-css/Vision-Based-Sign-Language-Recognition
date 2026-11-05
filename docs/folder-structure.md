# Project Folder Structure

> Auto-generated reference for the Real-Time ArSL Translation Overlay System

```
real-time-arsl-overlay/
│
├── ARCHITECTURE.md                  # System architecture document
├── README.md                        # Project overview, setup, usage
├── LICENSE
├── CONTRIBUTING.md
│
├── config/                          # Configuration files
│   ├── default.yaml                 # System defaults
│   ├── schema.yaml                  # Config schema for validation
│   ├── profiles/                    # Runtime profiles
│   │   ├── fast.yaml                # Lower accuracy, higher FPS
│   │   └── accurate.yaml            # Higher accuracy, lower FPS
│   └── logging.yaml                 # Logging configuration
│
├── data/                            # Runtime data
│   ├── models/                      # Trained model files (gitignored)
│   │   ├── <dialect>/               # e.g., iraqi, saudi, egyptian
│   │   │   ├── static/model.onnx
│   │   │   ├── dynamic/model.onnx
│   │   │   ├── alphabet/model.onnx
│   │   │   └── gestures.json
│   │   └── manifest.json            # All available models index
│   ├── cache/                       # Runtime cache
│   └── samples/                     # Sample video files for testing
│
├── docs/                            # Documentation
│   ├── technical-specification.md
│   ├── component-design.md
│   └── api-reference.md
│
├── models/                          # Model training code
│   ├── static_gesture/
│   ├── dynamic_gesture/
│   ├── alphabet/
│   └── utils/
│
├── notebooks/                       # Research notebooks
│
├── scripts/                         # Utility scripts
│
├── src/                             # Application source code
│   ├── main.py                      # Entry point
│   ├── app.py                       # Application class (DI container)
│   │
│   ├── core/                        # Core framework
│   │   ├── __init__.py
│   │   ├── pipeline_orchestrator.py # Pipeline state machine & lifecycle
│   │   ├── pipeline_stage.py        # Abstract base for pipeline stages
│   │   ├── event_bus.py             # In-process pub-sub event system
│   │   └── errors.py                # Domain exception hierarchy
│   │
│   ├── capture/                     # Video source capture
│   │   ├── __init__.py
│   │   ├── capture_manager.py       # Source lifecycle
│   │   ├── video_source.py          # Abstract base
│   │   ├── frame.py                 # Frame data class
│   │   └── sources/                 # Concrete implementations
│   │       ├── webcam_source.py
│   │       ├── file_source.py
│   │       ├── screen_capture_source.py
│   │       └── browser_capture_source.py
│   │
│   ├── preprocessing/               # Frame preprocessing
│   │   ├── __init__.py
│   │   ├── preprocess_pipeline.py
│   │   ├── preprocessed_frame.py
│   │   └── stages/
│   │       ├── normalizer.py
│   │       ├── hand_detector.py
│   │       ├── landmark_extractor.py
│   │       └── roi_extractor.py
│   │
│   ├── recognition/                 # Core recognition engine
│   │   ├── __init__.py
│   │   ├── recognition_engine.py
│   │   ├── recognition_result.py
│   │   ├── segmenter/
│   │   │   ├── gesture_segmenter.py
│   │   │   ├── motion_analyzer.py
│   │   │   └── boundary_detector.py
│   │   ├── classifiers/
│   │   │   ├── base_classifier.py
│   │   │   ├── static_classifier.py
│   │   │   ├── dynamic_classifier.py
│   │   │   └── alphabet_decoder.py
│   │   └── fusion/
│   │       ├── decision_fusion.py
│   │       └── confidence_scorer.py
│   │
│   ├── translation/                 # Gesture-to-text mapping
│   │   ├── __init__.py
│   │   ├── translation_mapper.py
│   │   ├── gesture_vocabulary.py
│   │   ├── context_manager.py
│   │   ├── caption_buffer.py
│   │   └── dialect_pack.py
│   │
│   ├── overlay/                     # Display & overlay
│   │   ├── __init__.py
│   │   ├── display_manager.py
│   │   ├── overlay_window.py
│   │   ├── caption_renderer.py
│   │   ├── renderers/
│   │   │   ├── windows_renderer.py
│   │   │   └── abstract_renderer.py
│   │   └── themes/default.yaml
│   │
│   ├── ui/                          # Control panel
│   │   ├── __init__.py
│   │   ├── control_panel.py
│   │   ├── system_tray.py
│   │   ├── settings_dialog.py
│   │   └── widgets/
│   │       ├── source_selector.py
│   │       ├── dialect_selector.py
│   │       └── status_indicator.py
│   │
│   ├── model_manager/               # Model lifecycle
│   │   ├── __init__.py
│   │   ├── model_manager.py
│   │   ├── model_pack.py
│   │   └── model_registry.py
│   │
│   ├── config/                      # Configuration subsystem
│   │   ├── __init__.py
│   │   ├── config_manager.py
│   │   ├── schema_validator.py
│   │   └── providers/
│   │       ├── yaml_provider.py
│   │       ├── env_provider.py
│   │       └── cli_provider.py
│   │
│   └── common/                      # Shared utilities
│       ├── __init__.py
│       ├── types.py
│       ├── constants.py
│       ├── logging_config.py
│       ├── timing.py
│       └── serialization.py
│
├── tests/                           # Test suite
│   ├── unit/
│   ├── integration/
│   │   └── test_pipeline.py
│   ├── e2e/
│   │   └── test_full_pipeline.py
│   └── fixtures/
│
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── training.txt
│
├── setup.py
├── setup.cfg
├── pyproject.toml
│
└── .github/workflows/
    ├── ci.yml
    ├── test.yml
    └── build.yml
```
