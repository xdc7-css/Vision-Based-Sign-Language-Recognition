"""DatasetCollector — professional dataset capture platform for sign language gestures."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.domain.hand import HandDetectionResult
from src.domain.recognition import GestureCategory


@dataclass
class SampleMetadata:
    sample_id: str
    gesture_type: str
    gesture_label: str
    dialect: str
    timestamp: float
    num_frames: int
    num_hands: int
    handedness: str
    duration_seconds: float
    fps: float
    has_motion: bool
    total_distance: float
    motion_energy: float
    sample_path: str = ""


@dataclass
class CaptureSession:
    """A single recording session accumulating frames for one gesture sample."""
    gesture_type: GestureCategory
    gesture_label: str
    dialect: str = "iraqi"
    frames: list[dict[str, Any]] = field(default_factory=list)
    landmarks: list[np.ndarray] = field(default_factory=list)
    motion_vectors: list[dict[str, float]] = field(default_factory=list)
    trajectories: list[list[tuple[float, float]]] = field(default_factory=list)
    start_time: float = 0.0
    is_recording: bool = False

    def start(self) -> None:
        self.frames.clear()
        self.landmarks.clear()
        self.motion_vectors.clear()
        self.trajectories.clear()
        self.start_time = time.time()
        self.is_recording = True

    def add_frame(self, result: HandDetectionResult, palm_center: tuple[float, float]) -> None:
        if not self.is_recording:
            return

        frame_data: dict[str, Any] = {
            "timestamp": time.time() - self.start_time,
            "num_hands": result.num_hands,
            "width": result.image_width,
            "height": result.image_height,
        }

        for hand in result.hands:
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmarks], dtype=np.float32)
            self.landmarks.append(lm_array)

            frame_data[f"landmarks_{hand.handedness.value}"] = lm_array.tolist()
            frame_data["handedness"] = hand.handedness.value
            frame_data["confidence"] = hand.confidence

        self.frames.append(frame_data)

    def add_motion(self, vx: float, vy: float, speed: float, direction: float) -> None:
        self.motion_vectors.append({
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "direction": direction,
        })

    def add_trajectory_point(self, x: float, y: float) -> None:
        if not self.trajectories:
            self.trajectories.append([])
        self.trajectories[-1].append((x, y))

    def stop(self) -> CaptureSession:
        self.is_recording = False
        return self


class DatasetCollector:
    """Professional dataset capture platform.

    Organizes samples into:
        data/datasets/{gesture_type}/{label}/samples/{sample_id}/
            ├── landmarks.npy
            ├── metadata.json
            └── motion_data.npy
    """

    BASE_DIR = Path("data/datasets")

    TYPE_DIR_MAP = {
        GestureCategory.ALPHABET: "alphabet",
        GestureCategory.STATIC_WORD: "words",
        GestureCategory.DYNAMIC_GESTURE: "dynamic",
    }

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base_dir = base_dir or self.BASE_DIR
        self._session: Optional[CaptureSession] = None
        self._samples: list[SampleMetadata] = []

    # ── Session management ──────────────────────────────────────────

    def start_session(
        self,
        gesture_type: GestureCategory,
        gesture_label: str,
        dialect: str = "iraqi",
    ) -> CaptureSession:
        """Start a new capture session for one gesture sample."""
        self._session = CaptureSession(
            gesture_type=gesture_type,
            gesture_label=gesture_label,
            dialect=dialect,
        )
        self._session.start()
        return self._session

    @property
    def current_session(self) -> Optional[CaptureSession]:
        return self._session

    @property
    def is_recording(self) -> bool:
        return self._session is not None and self._session.is_recording

    def stop_session(self) -> Optional[SampleMetadata]:
        """Stop recording and return metadata for the captured sample."""
        if self._session is None:
            return None

        self._session.stop()
        meta = self._build_metadata(self._session)
        self._samples.append(meta)
        self._last_session = self._session
        self._session = None
        return meta

    # ── Save / Delete ───────────────────────────────────────────────

    def save_sample(self, session: Optional[CaptureSession] = None) -> Optional[Path]:
        """Save the current (or specified) session to disk."""
        session = session or self._session or getattr(self, '_last_session', None)
        if session is None or not session.landmarks:
            return None

        sample_id = str(uuid.uuid4())[:8]
        label = self._sanitize_label(session.gesture_label)
        type_dir = self.TYPE_DIR_MAP.get(session.gesture_type, "other")

        sample_dir = self._base_dir / type_dir / label / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save landmarks (3D array: frames × 21 × 3)
        if session.landmarks:
            landmarks_array = np.stack(session.landmarks, axis=0)
            np.save(sample_dir / "landmarks.npy", landmarks_array)

        # Save motion data
        if session.motion_vectors:
            motion_array = np.array([
                [m["vx"], m["vy"], m["speed"], m["direction"]]
                for m in session.motion_vectors
            ], dtype=np.float32)
            np.save(sample_dir / "motion_data.npy", motion_array)

        # Save trajectory
        if session.trajectories and session.trajectories[0]:
            traj_array = np.array(session.trajectories[0], dtype=np.float32)
            np.save(sample_dir / "trajectory.npy", traj_array)

        # Save metadata
        meta = self._build_metadata(session)
        meta.sample_path = str(sample_dir)
        meta_json = {
            "sample_id": meta.sample_id,
            "gesture_type": meta.gesture_type,
            "gesture_label": meta.gesture_label,
            "dialect": meta.dialect,
            "timestamp": meta.timestamp,
            "num_frames": meta.num_frames,
            "num_hands": meta.num_hands,
            "handedness": meta.handedness,
            "duration_seconds": meta.duration_seconds,
            "fps": meta.fps,
            "has_motion": meta.has_motion,
            "total_distance": meta.total_distance,
            "motion_energy": meta.motion_energy,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_json, f, indent=2, ensure_ascii=False)

        return sample_dir

    def delete_sample(self, sample_dir: Path) -> bool:
        """Delete a sample directory and all its contents."""
        if sample_dir.exists() and sample_dir.is_dir():
            import shutil
            shutil.rmtree(sample_dir)
            return True
        return False

    def list_samples(self, gesture_type: Optional[GestureCategory] = None) -> list[Path]:
        """List all sample directories, optionally filtered by type."""
        if gesture_type:
            type_dir = self.TYPE_DIR_MAP.get(gesture_type)
            search_path = self._base_dir / type_dir if type_dir else self._base_dir
        else:
            search_path = self._base_dir

        samples: list[Path] = []
        if not search_path.exists():
            return samples

        for label_dir in search_path.iterdir():
            if label_dir.is_dir():
                samples_dir = label_dir / "samples"
                if samples_dir.exists():
                    for sample_dir in samples_dir.iterdir():
                        if sample_dir.is_dir():
                            samples.append(sample_dir)

        return sorted(samples)

    # ── Private ─────────────────────────────────────────────────────

    def _build_metadata(self, session: CaptureSession) -> SampleMetadata:
        num_hands = 0
        handedness = "unknown"
        for f in session.frames:
            num_hands = max(num_hands, f.get("num_hands", 0))
            if "handedness" in f:
                handedness = f["handedness"]

        total_dist = 0.0
        motion_energy = 0.0
        for m in session.motion_vectors:
            total_dist += m["speed"]
            motion_energy += m["speed"] ** 2

        duration = 0.0
        if session.frames:
            duration = session.frames[-1]["timestamp"] - session.frames[0]["timestamp"]

        fps = len(session.frames) / max(duration, 0.001)

        return SampleMetadata(
            sample_id=str(uuid.uuid4())[:8],
            gesture_type=session.gesture_type.name.lower(),
            gesture_label=session.gesture_label,
            dialect=session.dialect,
            timestamp=session.start_time,
            num_frames=len(session.frames),
            num_hands=num_hands,
            handedness=handedness,
            duration_seconds=duration,
            fps=fps,
            has_motion=total_dist > 0.01,
            total_distance=total_dist,
            motion_energy=motion_energy,
        )

    @staticmethod
    def _sanitize_label(label: str) -> str:
        return label.strip().replace(" ", "_").replace("/", "_")
