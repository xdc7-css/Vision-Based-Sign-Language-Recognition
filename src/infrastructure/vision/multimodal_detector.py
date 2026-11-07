"""MultimodalDetector — real-time dual hand, face, and upper-body pose tracker using MediaPipe."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
import mediapipe as mp

from src.domain.hand import (
    Hand,
    Handedness,
    HandLandmark,
)
from src.domain.multimodal import (
    FACE_EXPRESSIVE_INDICES,
    FaceLandmark,
    ModalityConfig,
    MultimodalFrame,
    POSE_UPPER_BODY_INDICES,
    PoseLandmark,
)


class MultimodalDetector:
    """Tracks up to two hands, face landmarks, and optional upper-body pose.

    Ensures:
    - Stable handedness tracking across frames (prevents hand swapping).
    - Correct physical handedness under mirrored webcam previews.
    - Zero duplication of missing hands.
    - Compact, expressive facial and pose feature representations.
    """

    def __init__(
        self,
        config: Optional[ModalityConfig] = None,
        models_dir: str | Path = "data/models/karsl",
    ) -> None:
        self._config = config or ModalityConfig()
        self._models_dir = Path(models_dir)
        self._logger = logging.getLogger(__name__)

        self._hands_model: Optional[vision.HandLandmarker] = None
        self._face_model: Optional[vision.FaceLandmarker] = None
        self._pose_model: Optional[vision.PoseLandmarker] = None

        # Temporal tracking state for hand continuity
        self._prev_left_wrist: Optional[tuple[float, float, float]] = None
        self._prev_right_wrist: Optional[tuple[float, float, float]] = None
        self._last_detect_time: float = 0.0

        self._init_models()

    def _init_models(self) -> None:
        """Initialize MediaPipe task models on CPU."""
        running_mode = vision.RunningMode.IMAGE
        delegate = BaseOptions.Delegate.CPU

        # 1. Hands Model
        hand_task = self._models_dir / "hand_landmarker.task"
        if not hand_task.exists():
            hand_task = Path("data/models/general/hand_landmarker.task")

        if hand_task.exists():
            try:
                opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(hand_task), delegate=delegate),
                    running_mode=running_mode,
                    num_hands=2,
                    min_hand_detection_confidence=self._config.min_detection_confidence,
                    min_hand_presence_confidence=self._config.min_tracking_confidence,
                    min_tracking_confidence=self._config.min_tracking_confidence,
                )
                self._hands_model = vision.HandLandmarker.create_from_options(opts)
            except Exception as e:
                self._logger.error(f"Failed to load HandLandmarker: {e}")

        # 2. Face Model
        face_task = self._models_dir / "face_landmarker.task"
        if face_task.exists():
            try:
                opts = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(face_task), delegate=delegate),
                    running_mode=running_mode,
                    num_faces=1,
                    min_face_detection_confidence=self._config.min_detection_confidence,
                    min_face_presence_confidence=self._config.min_tracking_confidence,
                    min_tracking_confidence=self._config.min_tracking_confidence,
                )
                self._face_model = vision.FaceLandmarker.create_from_options(opts)
            except Exception as e:
                self._logger.error(f"Failed to load FaceLandmarker: {e}")

        # 3. Pose Model (lazy/optional)
        if self._config.enable_pose:
            self._ensure_pose_model()

    def _ensure_pose_model(self) -> None:
        if self._pose_model is not None:
            return
        pose_task = self._models_dir / "pose_landmarker.task"
        if pose_task.exists():
            try:
                opts = vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(pose_task), delegate=BaseOptions.Delegate.CPU),
                    running_mode=vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=self._config.min_detection_confidence,
                    min_pose_presence_confidence=self._config.min_tracking_confidence,
                    min_tracking_confidence=self._config.min_tracking_confidence,
                )
                self._pose_model = vision.PoseLandmarker.create_from_options(opts)
            except Exception as e:
                self._logger.error(f"Failed to load PoseLandmarker: {e}")

    @property
    def config(self) -> ModalityConfig:
        return self._config

    def set_pose_enabled(self, enabled: bool) -> None:
        self._config.enable_pose = enabled
        if enabled:
            self._ensure_pose_model()

    def set_mirror_camera(self, mirror: bool) -> None:
        self._config.mirror_camera = mirror

    def detect(self, image: np.ndarray) -> MultimodalFrame:
        """Run multimodal detection on a frame (BGR or RGB).

        Args:
            image: Frame as numpy array (H, W, 3).

        Returns:
            MultimodalFrame with synchronized hands, face, pose, and metrics.
        """
        start_time = time.perf_counter()

        if image is None or image.size == 0:
            return MultimodalFrame(timestamp=time.time())

        h, w = image.shape[:2]

        # Convert to RGB if BGR
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        rgb = np.ascontiguousarray(rgb)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── 1. Hands Detection ───────────────────────────────────────
        detected_hands: list[Hand] = []
        if self._hands_model is not None:
            try:
                res_hands = self._hands_model.detect(mp_img)
                detected_hands = self._extract_hands(res_hands)
            except Exception as e:
                self._logger.debug(f"Hands detect error: {e}")

        # Separate left and right hands with spatial continuity
        left_hand, right_hand = self._assign_and_stabilize_hands(detected_hands)

        # ── 2. Face Detection ────────────────────────────────────────
        face_lms: list[FaceLandmark] = []
        has_face = False
        raw_face_pts: list[tuple[int, int]] = []
        raw_face_3d: list[Any] = []

        if self._config.enable_face and self._face_model is not None:
            try:
                res_face = self._face_model.detect(mp_img)
                if res_face.face_landmarks and len(res_face.face_landmarks) > 0:
                    lms = res_face.face_landmarks[0]
                    has_face = True
                    raw_face_3d = lms
                    raw_face_pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

                    # Extract selected expressive subset
                    face_lms = [
                        FaceLandmark(
                            x=float(lms[idx].x),
                            y=float(lms[idx].y),
                            z=float(lms[idx].z),
                        )
                        for idx in FACE_EXPRESSIVE_INDICES
                        if idx < len(lms)
                    ]
            except Exception as e:
                self._logger.debug(f"Face detect error: {e}")

        # ── 3. Optional Pose Detection ───────────────────────────────
        pose_lms: list[PoseLandmark] = []
        has_pose = False
        raw_pose_pts: list[tuple[int, int]] = []

        if self._config.enable_pose:
            self._ensure_pose_model()
            if self._pose_model is not None:
                try:
                    res_pose = self._pose_model.detect(mp_img)
                    if res_pose.pose_landmarks and len(res_pose.pose_landmarks) > 0:
                        p_lms = res_pose.pose_landmarks[0]
                        has_pose = True
                        raw_pose_pts = [(int(lm.x * w), int(lm.y * h)) for lm in p_lms]

                        # Extract selected upper-body subset
                        pose_lms = [
                            PoseLandmark(
                                x=float(p_lms[idx].x),
                                y=float(p_lms[idx].y),
                                z=float(p_lms[idx].z),
                                visibility=float(getattr(p_lms[idx], 'visibility', 1.0)),
                            )
                            for idx in POSE_UPPER_BODY_INDICES
                            if idx < len(p_lms)
                        ]
                except Exception as e:
                    self._logger.debug(f"Pose detect error: {e}")

        # ── 4. Compute Geometric Indicators & Relations ──────────────
        metrics = self._compute_geometric_metrics(raw_face_3d, left_hand, right_hand)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        frame_result = MultimodalFrame(
            left_hand=left_hand,
            right_hand=right_hand,
            face_landmarks=face_lms,
            pose_landmarks=pose_lms,
            has_left_hand=left_hand is not None,
            has_right_hand=right_hand is not None,
            has_face=has_face,
            has_pose=has_pose,
            mouth_openness=metrics["mouth_openness"],
            eyebrow_elevation=metrics["eyebrow_elevation"],
            eye_openness_left=metrics["eye_openness_left"],
            eye_openness_right=metrics["eye_openness_right"],
            left_hand_to_face_dist=metrics["left_hand_to_face_dist"],
            right_hand_to_face_dist=metrics["right_hand_to_face_dist"],
            timestamp=time.time(),
            processing_time_ms=elapsed_ms,
        )

        # Attach raw pixel points for visual rendering cache
        setattr(frame_result, "_raw_face_pts", raw_face_pts)
        setattr(frame_result, "_raw_pose_pts", raw_pose_pts)

        return frame_result

    def _extract_hands(self, res_hands) -> list[Hand]:
        """Convert raw MediaPipe HandLandmarker result to Hand objects with correct handedness."""
        hands: list[Hand] = []
        if not res_hands or not res_hands.hand_landmarks:
            return hands

        for handedness_list, hand_lms in zip(res_hands.handedness, res_hands.hand_landmarks):
            landmarks = [
                HandLandmark(
                    max(0.0, min(1.0, float(lm.x))),
                    max(0.0, min(1.0, float(lm.y))),
                    float(lm.z)
                )
                for lm in hand_lms
            ]

            category_name = handedness_list[0].category_name if handedness_list else "Unknown"
            score = float(handedness_list[0].score) if handedness_list else 1.0

            handedness_enum = self.calibrate_handedness(category_name, self._config.mirror_camera)

            hands.append(Hand(
                landmarks=landmarks,
                handedness=handedness_enum,
                confidence=score,
            ))

        return hands

    @staticmethod
    def calibrate_handedness(category: str | Handedness, mirror: bool = True) -> Handedness:
        """Calibrate handedness enum taking into account camera mirroring."""
        cat_str = category.value if isinstance(category, Handedness) else str(category)
        if mirror:
            if cat_str in ("Left", "left"):
                return Handedness.RIGHT
            elif cat_str in ("Right", "right"):
                return Handedness.LEFT
            return Handedness.UNKNOWN
        else:
            if cat_str in ("Left", "left"):
                return Handedness.LEFT
            elif cat_str in ("Right", "right"):
                return Handedness.RIGHT
            return Handedness.UNKNOWN

    def assign_and_stabilize_hands(
        self, hands: list[Hand]
    ) -> tuple[Optional[Hand], Optional[Hand]]:
        return self._assign_and_stabilize_hands(hands)

    def _assign_and_stabilize_hands(
        self, hands: list[Hand]
    ) -> tuple[Optional[Hand], Optional[Hand]]:
        """Assign detected hands to Left and Right slots, preventing hand swapping."""
        if not hands:
            # Clear temporal tracking if hands absent for > 0.5s
            if time.time() - self._last_detect_time > 0.5:
                self._prev_left_wrist = None
                self._prev_right_wrist = None
            return None, None

        self._last_detect_time = time.time()

        left_candidates = [h for h in hands if h.handedness == Handedness.LEFT]
        right_candidates = [h for h in hands if h.handedness == Handedness.RIGHT]

        # Case 1: Exactly one Left and one Right detected cleanly
        if len(left_candidates) == 1 and len(right_candidates) == 1:
            lh = left_candidates[0]
            rh = right_candidates[0]
            self._update_wrist_trackers(lh, rh)
            return lh, rh

        # Case 2: One hand detected
        if len(hands) == 1:
            h = hands[0]
            if h.handedness == Handedness.LEFT:
                self._update_wrist_trackers(lh=h, rh=None)
                return h, None
            elif h.handedness == Handedness.RIGHT:
                self._update_wrist_trackers(lh=None, rh=h)
                return None, h
            else:
                # Ambiguous handedness: use spatial continuity with previous wrist
                wrist = (h.landmarks[0].x, h.landmarks[0].y, h.landmarks[0].z)
                dist_l = self._wrist_dist(wrist, self._prev_left_wrist)
                dist_r = self._wrist_dist(wrist, self._prev_right_wrist)
                if dist_l < dist_r and dist_l < 0.25:
                    h.handedness = Handedness.LEFT
                    self._update_wrist_trackers(lh=h, rh=None)
                    return h, None
                else:
                    h.handedness = Handedness.RIGHT
                    self._update_wrist_trackers(lh=None, rh=h)
                    return None, h

        # Case 3: Two hands detected, but handedness classification is ambiguous (e.g. two Left or two Right)
        if len(hands) >= 2:
            h1, h2 = hands[0], hands[1]
            w1 = (h1.landmarks[0].x, h1.landmarks[0].y, h1.landmarks[0].z)
            w2 = (h2.landmarks[0].x, h2.landmarks[0].y, h2.landmarks[0].z)

            # In image space, user's left hand is typically to the left of user's right hand
            # In mirrored view: user's physical left hand is on screen left (smaller x)
            # and physical right hand is on screen right (larger x).
            if w1[0] < w2[0]:
                h1.handedness = Handedness.LEFT
                h2.handedness = Handedness.RIGHT
                self._update_wrist_trackers(h1, h2)
                return h1, h2
            else:
                h2.handedness = Handedness.LEFT
                h1.handedness = Handedness.RIGHT
                self._update_wrist_trackers(h2, h1)
                return h2, h1

        return None, None

    def _wrist_dist(self, p1: tuple[float, ...], p2: Optional[tuple[float, ...]]) -> float:
        if p2 is None:
            return 999.0
        z1 = p1[2] if len(p1) > 2 else 0.0
        z2 = p2[2] if len(p2) > 2 else 0.0
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (z1 - z2) ** 2))

    def _update_wrist_trackers(self, lh: Optional[Hand], rh: Optional[Hand]) -> None:
        if lh is not None:
            self._prev_left_wrist = (lh.landmarks[0].x, lh.landmarks[0].y, lh.landmarks[0].z)
        if rh is not None:
            self._prev_right_wrist = (rh.landmarks[0].x, rh.landmarks[0].y, rh.landmarks[0].z)

    def _compute_geometric_metrics(
        self,
        raw_face: list[Any],
        left_hand: Optional[Hand],
        right_hand: Optional[Hand],
    ) -> dict[str, float]:
        """Compute compact geometric indicators (mouth openness, eyebrow elevation, hand-to-face distances)."""
        metrics = {
            "mouth_openness": 0.0,
            "eyebrow_elevation": 0.0,
            "eye_openness_left": 0.0,
            "eye_openness_right": 0.0,
            "left_hand_to_face_dist": 0.0,
            "right_hand_to_face_dist": 0.0,
        }

        if raw_face and len(raw_face) > 400:
            def pt_dist(idx1: int, idx2: int) -> float:
                p1 = raw_face[idx1]
                p2 = raw_face[idx2]
                return float(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2))

            # Mouth aspect ratio: vertical opening (13, 14) / mouth width (61, 291)
            mouth_w = max(pt_dist(61, 291), 1e-5)
            mouth_h = pt_dist(13, 14)
            metrics["mouth_openness"] = float(np.clip(mouth_h / mouth_w, 0.0, 2.0))

            # Eye openness (EAR proxy): vertical height / horizontal width
            eye_l_w = max(pt_dist(33, 133), 1e-5)
            eye_l_h = pt_dist(159, 145)
            metrics["eye_openness_left"] = float(np.clip(eye_l_h / eye_l_w, 0.0, 1.5))

            eye_r_w = max(pt_dist(263, 362), 1e-5)
            eye_r_h = pt_dist(386, 374)
            metrics["eye_openness_right"] = float(np.clip(eye_r_h / eye_r_w, 0.0, 1.5))

            # Eyebrow elevation relative to eye
            eb_l = pt_dist(63, 159)
            eb_r = pt_dist(293, 386)
            avg_eb = (eb_l + eb_r) / 2.0
            metrics["eyebrow_elevation"] = float(np.clip(avg_eb / eye_l_w, 0.0, 2.0))

            # Distance from wrists to nose tip (index 1)
            nose = raw_face[1]
            if left_hand is not None:
                lw = left_hand.landmarks[0]
                metrics["left_hand_to_face_dist"] = float(np.sqrt((lw.x - nose.x) ** 2 + (lw.y - nose.y) ** 2))
            if right_hand is not None:
                rw = right_hand.landmarks[0]
                metrics["right_hand_to_face_dist"] = float(np.sqrt((rw.x - nose.x) ** 2 + (rw.y - nose.y) ** 2))

        return metrics

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands_model:
            self._hands_model.close()
        if self._face_model:
            self._face_model.close()
        if self._pose_model:
            self._pose_model.close()

    def __enter__(self) -> MultimodalDetector:
        return self

    def __exit__(self, *args) -> None:
        self.close()
