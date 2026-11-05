"""DialectManager — runtime dialect switching with separate datasets, models, and vocabulary.

Architecture:
    DialectManager
    ├── DialectRegistry (domain model)
    ├── ModelRouter (loads/unloads models per dialect)
    ├── DatasetRouter (resolves dataset paths)
    └── VocabularyRouter (resolves vocabulary paths)

Dialects share the recognition pipeline (HybridRecognitionPipeline)
but load dialect-specific models and vocabulary.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from src.domain.dialect import DialectConfig, DialectId, DialectRegistry
from src.domain.hand import Hand
from src.domain.motion import HandMotionTrack
from src.infrastructure.recognition.hybrid_pipeline import (
    HybridConfig,
    HybridRecognitionPipeline,
    HybridResult,
)
from src.infrastructure.recognition.static_recognizer import (
    StaticModelConfig,
    StaticSignRecognizer,
)
from src.infrastructure.recognition.dynamic_recognizer import (
    DynamicModelConfig,
    DynamicSignRecognizer,
)
from src.infrastructure.recognition.word_recognizer import WordRecognizer


DEFAULT_DIALECTS = {
    DialectId.IRAQI: DialectConfig(
        dialect_id=DialectId.IRAQI,
        dataset_path="data/datasets/iraqi",
        model_path="data/models/iraqi",
        vocabulary_path="data/models/iraqi/vocabulary.json",
    ),
    DialectId.SAUDI: DialectConfig(
        dialect_id=DialectId.SAUDI,
        dataset_path="data/datasets/saudi",
        model_path="data/models/saudi",
        vocabulary_path="data/models/saudi/vocabulary.json",
    ),
    DialectId.EGYPTIAN: DialectConfig(
        dialect_id=DialectId.EGYPTIAN,
        dataset_path="data/datasets/egyptian",
        model_path="data/models/egyptian",
        vocabulary_path="data/models/egyptian/vocabulary.json",
    ),
}


class DialectManager:
    """Manages dialect switching at runtime. All dialects share the same
    recognition pipeline but load separate models and vocabulary.

    Future scalability:
    - Add new dialect: register a DialectConfig + create data/models/{dialect}/
    - Dialect-specific preprocessing via preprocess_hook callable
    - Auto-discovery of installed dialects from data/models/ directory
    """

    def __init__(
        self,
        pipeline: Optional[HybridRecognitionPipeline] = None,
        hybrid_config: Optional[HybridConfig] = None,
        static_config: Optional[StaticModelConfig] = None,
        dynamic_config: Optional[DynamicModelConfig] = None,
        base_path: str = ".",
    ) -> None:
        self._registry = DialectRegistry()
        self._base_path = Path(base_path).resolve()
        self._pipeline: Optional[HybridRecognitionPipeline] = pipeline
        self._hybrid_config = hybrid_config or HybridConfig()
        self._static_config = static_config or StaticModelConfig()
        self._dynamic_config = dynamic_config or DynamicModelConfig()
        self._logger = logging.getLogger(__name__)

        # Per-dialect recognizers (lazy-loaded per active dialect)
        self._static_recognizers: dict[DialectId, Optional[StaticSignRecognizer]] = {}
        self._dynamic_recognizers: dict[DialectId, Optional[DynamicSignRecognizer]] = {}
        self._word_recognizers: dict[DialectId, Optional[WordRecognizer]] = {}

        # Pre/post processing hooks per dialect (for future use)
        self._preprocess_hooks: dict[DialectId, callable] = {}
        self._postprocess_hooks: dict[DialectId, callable] = {}

        self._register_defaults()

    # ── Public API ──────────────────────────────────────────────────

    def process(self, hand: Optional[Hand] = None, track: Optional[HandMotionTrack] = None) -> HybridResult:
        """Run recognition with the currently active dialect's models."""
        if self._pipeline is None:
            self._logger.warning("No recognition pipeline configured")
            return HybridResult(
                final_result=None,
                decision_path="error",
            )

        # Preprocess hook
        if self._registry.active_id in self._preprocess_hooks:
            hand = self._preprocess_hooks[self._registry.active_id](hand)

        result = self._pipeline.process(hand, track)

        # Postprocess hook
        if self._registry.active_id in self._postprocess_hooks:
            result = self._postprocess_hooks[self._registry.active_id](result)

        return result

    def reload_dialect(self, dialect_id: Optional[DialectId] = None) -> bool:
        """Reload models and vocabulary for a dialect (defaults to active dialect)."""
        target = dialect_id or self._registry.active_id
        success = self._load_dialect_models(target)
        if success:
            self._logger.info(f"Reloaded models and vocabulary for dialect '{target.value}'")
        return success

    def switch_dialect(self, dialect_id: DialectId) -> bool:
        """Switch active dialect at runtime. Returns True if successful."""
        old_id = self._registry.active_id
        if not self._registry.activate(dialect_id):
            self._logger.warning(f"Failed to activate dialect '{dialect_id.value}'")
            return False

        if not self._load_dialect_models(dialect_id):
            self._logger.error(f"Failed to load models for dialect '{dialect_id.value}', reverting")
            self._registry.activate(old_id)
            return False

        self._logger.info(f"Switched dialect: {old_id.value} -> {dialect_id.value}")
        return True

    @property
    def active_dialect(self) -> DialectId:
        return self._registry.active_id

    @property
    def active_dialect_config(self) -> Optional[DialectConfig]:
        return self._registry.active

    @property
    def registry(self) -> DialectRegistry:
        return self._registry

    @property
    def available_dialects(self) -> list[DialectConfig]:
        return self._registry.list_enabled()

    @property
    def pipeline(self) -> Optional[HybridRecognitionPipeline]:
        return self._pipeline

    @pipeline.setter
    def pipeline(self, p: HybridRecognitionPipeline) -> None:
        self._pipeline = p

    def register_hook(self, dialect: DialectId, stage: str, hook: callable) -> None:
        if stage == "preprocess":
            self._preprocess_hooks[dialect] = hook
        elif stage == "postprocess":
            self._postprocess_hooks[dialect] = hook

    def discover_installed_dialects(self) -> list[DialectId]:
        """Auto-discover dialects from data/models/ directory."""
        discovered: list[DialectId] = []
        models_dir = self._base_path / "data" / "models"
        if models_dir.exists():
            for child in models_dir.iterdir():
                if child.is_dir():
                    try:
                        did = DialectId(child.name)
                        discovered.append(did)
                    except ValueError:
                        pass
        return discovered

    # ── Private ─────────────────────────────────────────────────────

    def _register_defaults(self) -> None:
        for dialect_id, config in DEFAULT_DIALECTS.items():
            config.dataset_path = str(self._base_path / config.dataset_path)
            config.model_path = str(self._base_path / config.model_path)
            config.vocabulary_path = str(self._base_path / config.vocabulary_path)
            self._registry.register(config)

        self._registry.activate(DialectId.IRAQI)

    def _load_dialect_models(self, dialect_id: DialectId) -> bool:
        """Load (or reload) models for a specific dialect."""
        config = self._registry.dialects.get(dialect_id)
        if config is None:
            return False

        try:
            self._pipeline = self._pipeline or HybridRecognitionPipeline(config=self._hybrid_config)

            # Static model
            static_path = Path(config.model_path) / "static_model.pkl"
            if not static_path.exists():
                candidates = list(Path(config.model_path).glob("static_*.pkl"))
                if candidates:
                    static_path = candidates[0]

            sr: Optional[StaticSignRecognizer] = None
            if static_path.exists():
                sr = StaticSignRecognizer(config=self._static_config)
                sr.load(str(static_path))
                self._static_recognizers[dialect_id] = sr
                if self._pipeline is not None:
                    self._pipeline._static_recognizer = sr
            else:
                self._logger.info(f"No static model for '{dialect_id.value}' at {static_path}")

            # Dynamic model
            dynamic_path = Path(config.model_path) / "dynamic_model.pt"
            if dynamic_path.exists():
                dr = DynamicSignRecognizer(config=self._dynamic_config)
                dr.load(str(dynamic_path))
                self._dynamic_recognizers[dialect_id] = dr
                if self._pipeline is not None:
                    self._pipeline._dynamic_recognizer = dr
            else:
                self._logger.info(f"No dynamic model for '{dialect_id.value}' at {dynamic_path}")

            # Vocabulary
            vocab_path = Path(config.vocabulary_path)
            wr = WordRecognizer(static_recognizer=sr)
            if vocab_path.exists():
                count = wr.load_vocabulary_from_json(str(vocab_path))
                self._logger.info(f"Loaded {count} vocabulary items for '{dialect_id.value}' from {vocab_path}")
            else:
                self._logger.info(f"No vocabulary for '{dialect_id.value}' at {vocab_path}")

            if sr is not None:
                wr._static_recognizer = sr

            self._word_recognizers[dialect_id] = wr
            if self._pipeline is not None:
                self._pipeline._word_recognizer = wr

            return True

        except Exception as e:
            self._logger.error(f"Failed to load models for '{dialect_id.value}': {e}")
            return False
