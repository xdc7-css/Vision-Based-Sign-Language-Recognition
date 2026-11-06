from __future__ import annotations

from typing import Optional

from src.domain.entities import CameraInfo, DialectInfo, ModelInfo, VideoSourceType
from src.domain.interfaces import CameraService, DialectRepository, VideoSource
from src.application.state import AppState


class CameraUseCases:
    """Orchestrates camera discovery and selection."""

    def __init__(self, camera_service: CameraService, state: AppState) -> None:
        self._camera_service = camera_service
        self._state = state

    def refresh_cameras(self) -> list[CameraInfo]:
        cameras = self._camera_service.list_cameras()
        self._state.cameras = cameras
        return cameras

    def get_camera_by_index(self, index: int) -> Optional[CameraInfo]:
        return self._camera_service.get_camera(index)


class DialectUseCases:
    """Orchestrates dialect pack discovery and selection."""

    def __init__(self, dialect_repository: DialectRepository, state: AppState) -> None:
        self._dialect_repository = dialect_repository
        self._state = state

    def refresh_dialects(self) -> list[DialectInfo]:
        dialects = self._dialect_repository.list_dialects()
        self._state.dialects = dialects
        return dialects

    def select_dialect(self, dialect_id: str) -> Optional[DialectInfo]:
        dialect = self._dialect_repository.get_dialect(dialect_id)
        if dialect is not None:
            self._state.active_dialect = dialect
            self._state.log("info", f"Selected dialect: {dialect.display_name}")
        return dialect


class ModelUseCases:
    """Orchestrates model discovery and selection."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    def discover_models(self, dialect_id: str) -> list[ModelInfo]:
        from pathlib import Path

        models_dir = Path("data/models") / dialect_id
        discovered: list[ModelInfo] = []

        if not models_dir.exists():
            return discovered

        for model_type in ["static", "dynamic", "alphabet"]:
            model_path = models_dir / model_type / "model.onnx"
            metadata_path = models_dir / model_type / "metadata.json"
            if model_path.exists():
                accuracy = 0.0
                if metadata_path.exists():
                    import json
                    try:
                        with open(metadata_path) as f:
                            meta = json.load(f)
                        accuracy = meta.get("accuracy", {}).get("top1", 0.0)
                    except (json.JSONDecodeError, OSError):
                        pass

                discovered.append(ModelInfo(
                    dialect_id=dialect_id,
                    model_type=model_type,
                    version="1.0.0",
                    file_path=str(model_path),
                    is_loaded=False,
                    accuracy=accuracy,
                ))

        self._state.models = discovered
        return discovered

    def select_model(self, model_type: str) -> Optional[ModelInfo]:
        for model in self._state.models:
            if model.model_type == model_type:
                self._state.active_model = model
                return model
        return None


class VideoSourceUseCases:
    """Orchestrates video source control."""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._source: Optional[VideoSource] = None
        self._frame_count: int = 0

    @property
    def active_source(self) -> Optional[VideoSource]:
        return self._source

    def start_source(self, source_type: VideoSourceType, uri: str) -> bool:
        from src.infrastructure.video_source_engine import VideoSourceFactory

        self.stop_source()

        try:
            self._source = VideoSourceFactory.create(source_type, uri)
            if not self._source.open():
                self._state.error(f"Failed to open video source: {uri}")
                self._source = None
                return False

            self._state.active_source = self._source.properties
            self._state.pipeline_status = "streaming"
            self._state.log("info", f"Streaming from {source_type.display_name}: {uri}")
            return True

        except Exception as e:
            self._state.error(f"Error starting video source: {e}")
            self._source = None
            return False

    def stop_source(self) -> None:
        if self._source is not None:
            try:
                self._source.close()
            except Exception as e:
                self._state.log("warning", f"Error closing source: {e}")
            self._source = None
            self._state.active_source = None
            self._state.pipeline_status = "idle"
            self._state.log("info", "Video source stopped")

    def set_resolution(self, width: int, height: int) -> bool:
        if self._source is not None and hasattr(self._source, "set_resolution"):
            return self._source.set_resolution(width, height)
        return False

    def read_frame(self):
        if self._source is None or not self._source.is_open:
            return None

        try:
            frame = self._source.read()
            if frame is not None:
                self._frame_count += 1
                self._state.current_fps = self._source.current_fps
                return frame
            return None
        except Exception as e:
            self._state.error(f"Frame read error: {e}")
            return None
