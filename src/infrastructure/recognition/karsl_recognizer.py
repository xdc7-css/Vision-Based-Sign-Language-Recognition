"""KArSL-502 Pretrained Spatial-Temporal Transformer Recognition Engine.

Integrates the pretrained ST-Transformer model for 502-class Arabic Sign Language
recognition based on multimodal facial, body (pose), and dual-hand keypoints.

Architecture:
    Frame Input (RGB)
        │
        ├── Motion Detection (adaptive background subtraction)
        │
        ├── MediaPipe Multimodal Landmark Extraction (CPU):
        │   ├── Pose (6 keypoints: shoulders, elbows, wrists) — centered on nose, scaled by shoulder distance
        │   ├── Face (136 keypoints: lips, eyes, eyebrows, oval) — centered on nose, scaled by eye distance
        │   ├── Right Hand (21 keypoints) — centered on wrist, scaled by palm width
        │   └── Left Hand (21 keypoints) — centered on wrist, scaled by palm width
        │   Total: 184 keypoints × 4 features (x, y, z, visibility) = 736 dimensions per frame
        │
        ├── Temporal Buffer & TSN Sampling (15 to 50 frames → exact 50 frames)
        │
        ├── ONNX Runtime Inference [batch=1, 50, 736] → [1, 502] logits
        │
        ├── Softmax & Temporal Consensus (history voting + cooldown debounce)
        │
        └── Output: KArSL-502 Arabic / English Sign Word Prediction
"""

from __future__ import annotations

import json
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime as ort
from mediapipe.tasks.python import BaseOptions, vision
import mediapipe as mp


# ─── Geometric Landmark Indices & Connections ───────────────────────────

POSE_KPS_MP_IDX = (11, 12, 13, 14, 15, 16)  # L/R shoulders, elbows, wrists
MP_POSE_NOSE_IDX = 0
MP_POSE_SHOULDERS_IDX = (11, 12)

# Face: 136 contour points (lips, eyes, eyebrows, face oval, irises)
FACEMESH_LIPS = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
]
FACEMESH_LEFT_EYE = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362),
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362),
]
FACEMESH_LEFT_EYEBROW = [
    (276, 283), (283, 282), (282, 295), (295, 285),
    (276, 300), (300, 293), (293, 334), (334, 296), (296, 336), (336, 285),
]
FACEMESH_RIGHT_EYE = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133),
]
FACEMESH_RIGHT_EYEBROW = [
    (46, 53), (53, 52), (52, 65), (65, 55), (55, 107),
    (46, 70), (70, 63), (63, 105), (105, 66), (66, 107),
]
FACEMESH_FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323),
    (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93),
    (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]
FACEMESH_IRISES = [
    (468, 469), (469, 470), (470, 471), (471, 468),
    (473, 474), (474, 475), (475, 476), (476, 473),
]

ALL_FACE_EDGES = [
    *FACEMESH_LIPS, *FACEMESH_LEFT_EYE, *FACEMESH_LEFT_EYEBROW,
    *FACEMESH_RIGHT_EYE, *FACEMESH_RIGHT_EYEBROW, *FACEMESH_FACE_OVAL,
    *FACEMESH_IRISES,
]
FACE_KPS_MP_IDX = tuple(sorted(set(p for edge in ALL_FACE_EDGES for p in edge)))  # exactly 136 indices
MP_FACE_NOSE_IDX = 1
MP_FACE_EYES_IDX = (468, 473)

# Hands: 21 keypoints
HAND_KPS_MP_IDX = tuple(range(21))
MP_HAND_WRIST_IDX = 0
MP_HANDS_PALM_IDX = (2, 17)  # Thumb MCP and Pinky MCP

POSE_NUM = len(POSE_KPS_MP_IDX)      # 6
FACE_NUM = len(FACE_KPS_MP_IDX)      # 136
HAND_NUM = len(HAND_KPS_MP_IDX)      # 21
FEAT_NUM = POSE_NUM + FACE_NUM + 2 * HAND_NUM  # 184
FEAT_DIM = 4                         # (x, y, z, visibility)
TOTAL_INPUT_FEATURES = FEAT_NUM * FEAT_DIM    # 736
SEQ_LEN = 50

KP2SLICE = {
    "pose": slice(0, POSE_NUM),
    "face": slice(POSE_NUM, POSE_NUM + FACE_NUM),
    "rh": slice(POSE_NUM + FACE_NUM, POSE_NUM + FACE_NUM + HAND_NUM),
    "lh": slice(POSE_NUM + FACE_NUM + HAND_NUM, POSE_NUM + FACE_NUM + HAND_NUM * 2),
}


@dataclass
class KArSLConfig:
    """Configuration for KArSL-502 Pretrained Engine."""
    model_path: str = "data/models/karsl/karsl_502.onnx"
    labels_path: str = "data/models/karsl/labels.json"
    landmarkers_dir: str = "data/models/karsl"
    confidence_threshold: float = 0.60
    min_frames: int = 15
    seq_len: int = 50
    history_threshold: int = 2
    history_len: int = 5
    cooldown_frames: int = 25


@dataclass
class KArSLResult:
    """Result of KArSL-502 sign recognition."""
    pred_idx: int
    arabic_text: str
    english_text: str
    confidence: float
    hand_detected: bool
    face_detected: bool
    pose_detected: bool
    buffer_fill: int
    is_new: bool
    decision_path: str = "karsl_502"
    all_top_candidates: list[tuple[str, str, float]] = field(default_factory=list)


class MotionDetector:
    """Adaptive frame motion detector to filter idle periods."""

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        sigma_factor: float = 0.005,
        thresh_val: int = 20,
        motion_thresh: float = 0.001,
    ) -> None:
        self.width = width
        self.height = height
        self.thresh_val = thresh_val
        self.motion_thresh = motion_thresh

        sigmay = sigma_factor * height
        ksize = max(3, int(6 * sigmay) + int(sigmay % 2))
        self.gaussian_kernel = cv2.getGaussianKernel(ksize, sigmay)

    def convert_small_gray(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) > 2 and frame.shape[2] > 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = frame.shape[0]
        if abs(h - self.height) > 1:
            scale = self.height / h
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        return frame

    def detect(self, prev_frame: Optional[np.ndarray], frame: np.ndarray) -> tuple[bool, np.ndarray]:
        if prev_frame is None or prev_frame.size == 0 or frame is None or frame.size == 0:
            small = self.convert_small_gray(frame) if frame is not None and frame.size > 0 else np.array([])
            return True, small

        small_curr = self.convert_small_gray(frame)
        small_prev = self.convert_small_gray(prev_frame)

        diff = cv2.absdiff(small_prev, small_curr)
        blur = cv2.sepFilter2D(diff, -1, self.gaussian_kernel, self.gaussian_kernel)
        _, thresh = cv2.threshold(blur, self.thresh_val, 255, cv2.THRESH_BINARY)

        nz_curr = cv2.countNonZero(small_curr)
        if nz_curr == 0:
            return True, small_curr

        motion_ratio = cv2.countNonZero(thresh) / nz_curr
        return motion_ratio > self.motion_thresh, small_curr


class TSNSampler:
    """Temporal Segment Network Sampler — uniformly samples any sequence to target_len."""

    def __init__(self, target_len: int = SEQ_LEN) -> None:
        self.target_len = target_len

    def __call__(self, kps: np.ndarray) -> np.ndarray:
        seq_len = kps.shape[0]
        if seq_len == self.target_len:
            return kps

        ticks = np.linspace(0, seq_len, self.target_len + 1)
        start_ticks, end_ticks = ticks[:-1], ticks[1:]
        bin_width = end_ticks - start_ticks
        indices = start_ticks + bin_width / 2.0
        indices = np.clip(indices, 0, seq_len - 1 - 1e-6)

        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.ceil(indices).astype(int)
        alpha = (indices - idx_floor)[:, np.newaxis, np.newaxis]

        return (1 - alpha) * kps[idx_floor] + alpha * kps[idx_ceil]


class LandmarkerProcessor:
    """Multimodal keypoint extractor using CPU MediaPipe Landmarkers."""

    def __init__(self, landmarkers_dir: str = "data/models/karsl") -> None:
        self._dir = Path(landmarkers_dir)
        self.pose_model: Optional[vision.PoseLandmarker] = None
        self.face_model: Optional[vision.FaceLandmarker] = None
        self.hands_model: Optional[vision.HandLandmarker] = None
        self._logger = logging.getLogger(__name__)
        self._init_models()

    def _init_models(self) -> None:
        running_mode = vision.RunningMode.IMAGE
        delegate = BaseOptions.Delegate.CPU

        # Pose Landmarker
        pose_path = self._dir / "pose_landmarker.task"
        if pose_path.exists():
            opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_path), delegate=delegate),
                running_mode=running_mode,
            )
            self.pose_model = vision.PoseLandmarker.create_from_options(opts)

        # Face Landmarker
        face_path = self._dir / "face_landmarker.task"
        if face_path.exists():
            opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(face_path), delegate=delegate),
                running_mode=running_mode,
                num_faces=1,
            )
            self.face_model = vision.FaceLandmarker.create_from_options(opts)

        # Hand Landmarker
        hand_path = self._dir / "hand_landmarker.task"
        if hand_path.exists():
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(hand_path), delegate=delegate),
                running_mode=running_mode,
                num_hands=2,
            )
            self.hands_model = vision.HandLandmarker.create_from_options(opts)

        self._logger.info("KArSL MediaPipe landmarkers initialized on CPU successfully.")

    def close(self) -> None:
        if self.pose_model:
            self.pose_model.close()
        if self.face_model:
            self.face_model.close()
        if self.hands_model:
            self.hands_model.close()

    def extract_frame_keypoints(
        self, frame_rgb: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Extract and normalize all 184 keypoints (x, y, z, v).

        Returns:
            all_kps: (184, 4) float32 normalized array
            vis_info: dict with raw unnormalized pixel landmark coordinates for UI drawing
        """
        all_kps = np.zeros((FEAT_NUM, FEAT_DIM), dtype=np.float32)
        pose_kps = all_kps[KP2SLICE["pose"]]
        face_kps = all_kps[KP2SLICE["face"]]
        rh_kps = all_kps[KP2SLICE["rh"]]
        lh_kps = all_kps[KP2SLICE["lh"]]

        vis_info: dict[str, Any] = {
            "pose": [],
            "face": [],
            "rh": [],
            "lh": [],
            "has_hand": False,
            "has_face": False,
            "has_pose": False,
        }

        h, w = frame_rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        def lm_xyzv(lm, v: float = 1.0) -> tuple[float, float, float, float]:
            return (float(lm.x), float(lm.y), float(lm.z), float(v))

        def distance(p1, p2) -> float:
            return float(max(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2), 1e-6))

        # 1. Pose
        if self.pose_model:
            res_pose = self.pose_model.detect(mp_img)
            if res_pose.pose_landmarks and len(res_pose.pose_landmarks) > 0:
                lms = res_pose.pose_landmarks[0]
                vis_info["has_pose"] = True
                vis_info["pose"] = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

                pose_kps[:] = np.array([lm_xyzv(lms[idx], getattr(lms[idx], 'visibility', 1.0)) for idx in POSE_KPS_MP_IDX], dtype=np.float32)
                # Centered on nose (0), scaled by shoulder distance (11, 12)
                nose = lms[MP_POSE_NOSE_IDX]
                pose_kps[:, :3] -= np.array([nose.x, nose.y, nose.z], dtype=np.float32)
                shoulder_dist = distance(lms[11], lms[12])
                pose_kps[:, :3] /= shoulder_dist

        # 2. Face
        if self.face_model:
            res_face = self.face_model.detect(mp_img)
            if res_face.face_landmarks and len(res_face.face_landmarks) > 0:
                lms = res_face.face_landmarks[0]
                vis_info["has_face"] = True
                vis_info["face"] = [(int(lms[idx].x * w), int(lms[idx].y * h)) for idx in FACE_KPS_MP_IDX[::4]]

                face_kps[:] = np.array([lm_xyzv(lms[idx]) for idx in FACE_KPS_MP_IDX], dtype=np.float32)
                # Centered on nose tip/bridge (1), scaled by eye distance (468, 473)
                nose = lms[MP_FACE_NOSE_IDX]
                face_kps[:, :3] -= np.array([nose.x, nose.y, nose.z], dtype=np.float32)
                eye_dist = distance(lms[MP_FACE_EYES_IDX[0]], lms[MP_FACE_EYES_IDX[1]])
                face_kps[:, :3] /= eye_dist

        # 3. Hands
        if self.hands_model:
            res_hands = self.hands_model.detect(mp_img)
            if res_hands.hand_landmarks and len(res_hands.hand_landmarks) > 0:
                vis_info["has_hand"] = True
                for handedness, hand_lms in zip(res_hands.handedness, res_hands.hand_landmarks):
                    is_left = handedness[0].category_name == "Left"
                    target = lh_kps if is_left else rh_kps
                    vis_key = "lh" if is_left else "rh"
                    vis_info[vis_key] = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

                    target[:] = np.array([lm_xyzv(hand_lms[idx]) for idx in HAND_KPS_MP_IDX], dtype=np.float32)
                    # Centered on wrist (0), scaled by palm width (2, 17)
                    wrist = hand_lms[MP_HAND_WRIST_IDX]
                    target[:, :3] -= np.array([wrist.x, wrist.y, wrist.z], dtype=np.float32)
                    palm_dist = distance(hand_lms[MP_HANDS_PALM_IDX[0]], hand_lms[MP_HANDS_PALM_IDX[1]])
                    target[:, :3] /= palm_dist

        return all_kps, vis_info


class KArSLRecognizer:
    """High-level recognition engine managing KArSL-502 ST-Transformer model."""

    def __init__(self, config: Optional[KArSLConfig] = None) -> None:
        self.config = config or KArSLConfig()
        self._logger = logging.getLogger(__name__)

        self._session: Optional[ort.InferenceSession] = None
        self._ar_words: list[str] = []
        self._en_words: list[str] = []

        self._extractor: Optional[LandmarkerProcessor] = None
        self._motion_detector: Optional[MotionDetector] = None
        self._tsn_sampler = TSNSampler(target_len=self.config.seq_len)

        self._frame_buffer: list[np.ndarray] = []
        self._history_queue: deque[int] = deque(maxlen=self.config.history_len)
        self._prev_small_gray: Optional[np.ndarray] = None
        self._last_sent_sign: Optional[int] = None
        self._cooldown_counter: int = 0
        self._idle_counter: int = 0

        self._is_loaded = False
        self.load()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> bool:
        """Load ONNX model session, class vocabulary labels, and landmarkers."""
        try:
            model_p = Path(self.config.model_path)
            labels_p = Path(self.config.labels_path)
            landmarkers_d = Path(self.config.landmarkers_dir)

            if not model_p.exists():
                self._logger.error(f"KArSL ONNX model not found: {model_p}")
                return False

            if not labels_p.exists():
                self._logger.error(f"KArSL labels not found: {labels_p}")
                return False

            # Load ONNX session on CPU
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(
                str(model_p),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            # Load 502 class labels
            with open(labels_p, "r", encoding="utf-8") as f:
                labels_dict = json.load(f)
            self._ar_words = [str(w) for w in labels_dict["AR_WORDS"]]
            self._en_words = [str(w) for w in labels_dict["EN_WORDS"]]

            # Initialize feature extractor & motion detector
            self._extractor = LandmarkerProcessor(str(landmarkers_d))
            self._motion_detector = MotionDetector()

            self._is_loaded = True
            self._logger.info(
                f"KArSL-502 engine loaded successfully with {len(self._ar_words)} classes."
            )
            return True

        except Exception as e:
            self._logger.error(f"Failed to load KArSL-502 engine: {e}", exc_info=True)
            self._is_loaded = False
            return False

    def reset(self) -> None:
        """Clear temporal buffer and recognition state."""
        self._frame_buffer.clear()
        self._history_queue.clear()
        self._prev_small_gray = None
        self._last_sent_sign = None
        self._cooldown_counter = 0
        self._idle_counter = 0

    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[Optional[KArSLResult], np.ndarray, dict[str, Any]]:
        """Process a single video frame.

        Args:
            frame: BGR numpy image array.

        Returns:
            result: Optional[KArSLResult] if a sign is recognized or evaluated.
            annotated_frame: Frame with visual skeletal overlay.
            vis_info: Keypoint detection metadata.
        """
        if not self._is_loaded or self._extractor is None:
            return None, frame, {}

        # Cooldown management
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        # Check motion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        has_motion, small_gray = self._motion_detector.detect(self._prev_small_gray, frame)
        self._prev_small_gray = small_gray

        if not has_motion:
            self._idle_counter += 1
            if self._idle_counter > 40:
                self.reset()
        else:
            self._idle_counter = 0

        # Extract multimodal keypoints
        all_kps, vis_info = self._extractor.extract_frame_keypoints(frame_rgb)

        # Draw visual overlay on frame
        annotated_frame = self._render_landmarks(frame.copy(), vis_info)

        # Add to buffer if valid motion or hand is visible
        if vis_info["has_hand"] or vis_info["has_pose"]:
            self._frame_buffer.append(all_kps)
        elif len(self._frame_buffer) > 0 and not has_motion:
            # Drop frames slowly if no hand and no motion
            self._frame_buffer.pop(0)

        # Cap buffer size
        if len(self._frame_buffer) > self.config.seq_len:
            self._frame_buffer = self._frame_buffer[-self.config.seq_len:]

        buffer_len = len(self._frame_buffer)

        # Evaluate if sufficient frames accumulated
        if buffer_len < self.config.min_frames:
            return (
                KArSLResult(
                    pred_idx=-1,
                    arabic_text="",
                    english_text="",
                    confidence=0.0,
                    hand_detected=vis_info["has_hand"],
                    face_detected=vis_info["has_face"],
                    pose_detected=vis_info["has_pose"],
                    buffer_fill=buffer_len,
                    is_new=False,
                ),
                annotated_frame,
                vis_info,
            )

        # TSN Temporal Sampling to exactly 50 frames
        kps_arr = np.array(self._frame_buffer, dtype=np.float32)
        sampled_50 = self._tsn_sampler(kps_arr)  # (50, 184, 4)
        input_tensor = sampled_50.reshape(1, self.config.seq_len, TOTAL_INPUT_FEATURES)

        # ONNX Inference
        raw_outputs = self._session.run(["output"], {"input": input_tensor})[0].flatten()

        # Softmax probabilities
        exp_vals = np.exp(raw_outputs - np.max(raw_outputs))
        probs = exp_vals / np.sum(exp_vals)

        top_indices = np.argsort(probs)[::-1][:3]
        top_idx = int(top_indices[0])
        top_conf = float(probs[top_idx])

        top_candidates = [
            (self._ar_words[i], self._en_words[i], float(probs[i])) for i in top_indices
        ]

        is_new_sign = False
        ar_text = self._ar_words[top_idx]
        en_text = self._en_words[top_idx]

        if top_conf >= self.config.confidence_threshold:
            self._history_queue.append(top_idx)
            most_common_idx, count = Counter(self._history_queue).most_common(1)[0]

            if (
                count >= self.config.history_threshold
                and most_common_idx != self._last_sent_sign
                and self._cooldown_counter == 0
            ):
                self._last_sent_sign = most_common_idx
                self._cooldown_counter = self.config.cooldown_frames
                is_new_sign = True
                ar_text = self._ar_words[most_common_idx]
                en_text = self._en_words[most_common_idx]

        result = KArSLResult(
            pred_idx=top_idx,
            arabic_text=ar_text,
            english_text=en_text,
            confidence=top_conf,
            hand_detected=vis_info["has_hand"],
            face_detected=vis_info["has_face"],
            pose_detected=vis_info["has_pose"],
            buffer_fill=buffer_len,
            is_new=is_new_sign,
            all_top_candidates=top_candidates,
        )

        return result, annotated_frame, vis_info

    def get_vocabulary(self) -> list[dict[str, Any]]:
        """Return full 502-sign dictionary with metadata."""
        vocab: list[dict[str, Any]] = []
        for i in range(len(self._ar_words)):
            ar = self._ar_words[i]
            en = self._en_words[i]
            if i <= 30:
                cat = "أرقام (Numbers)"
            elif i <= 69:
                cat = "حروف أبجدية (Letters)"
            else:
                cat = "كلمات (Words)"
            vocab.append({"index": i, "arabic": ar, "english": en, "category": cat})
        return vocab

    def _render_landmarks(self, frame: np.ndarray, vis: dict[str, Any]) -> np.ndarray:
        """Render multimodal skeleton overlay on camera frame."""
        # Pose (Cyan)
        pose_pts = vis.get("pose", [])
        if len(pose_pts) >= 17:
            # Connect shoulders, elbows, wrists
            conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
            for u, v in conns:
                cv2.line(frame, pose_pts[u], pose_pts[v], (255, 255, 0), 2)
            for pt in [pose_pts[11], pose_pts[12], pose_pts[13], pose_pts[14], pose_pts[15], pose_pts[16]]:
                cv2.circle(frame, pt, 4, (0, 255, 255), -1)

        # Face points (Subtle purple dots)
        face_pts = vis.get("face", [])
        for pt in face_pts:
            cv2.circle(frame, pt, 1, (200, 100, 255), -1)

        # Hands (Green / Red)
        for hand_key, col in [("rh", (0, 255, 128)), ("lh", (255, 128, 0))]:
            h_pts = vis.get(hand_key, [])
            if len(h_pts) == 21:
                # Connections
                hand_edges = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
                ]
                for u, v in hand_edges:
                    cv2.line(frame, h_pts[u], h_pts[v], col, 2)
                for pt in h_pts:
                    cv2.circle(frame, pt, 3, (255, 255, 255), -1)

        return frame

