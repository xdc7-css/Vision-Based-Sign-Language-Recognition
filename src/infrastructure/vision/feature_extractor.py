"""MultimodalFeatureExtractor — unified, versioned landmark normalization & vectorization."""

from __future__ import annotations

import numpy as np

from src.domain.multimodal import (
    DUAL_HAND_DIM,
    FACE_DIM,
    FEATURE_DIM_FULL,
    FEATURE_DIM_HANDS,
    FEATURE_DIM_HANDS_FACE,
    HAND_DIM,
    HAND_LANDMARKS_COUNT,
    MULTIMODAL_SCHEMA_VERSION,
    ModalityMode,
    MultimodalFeatureSchema,
    MultimodalFrame,
    POSE_DIM,
    QUALITY_METRICS_DIM,
)


class MultimodalFeatureExtractor:
    """Extracts consistent, normalized feature tensors across Dataset Studio, Training, and Inference.

    Preserves relative spatial relationships between hands and face while ensuring
    invariance to subject scale and translation.
    """

    def __init__(self, default_mode: ModalityMode = ModalityMode.TWO_HANDS_FACE) -> None:
        self._default_mode = default_mode

    @staticmethod
    def get_feature_dim(mode: ModalityMode) -> int:
        if mode == ModalityMode.TWO_HANDS_FACE_POSE:
            return FEATURE_DIM_FULL  # 230
        elif mode == ModalityMode.TWO_HANDS_FACE:
            return FEATURE_DIM_HANDS_FACE  # 206
        elif mode == ModalityMode.TWO_HANDS:
            return FEATURE_DIM_HANDS  # 134
        elif mode == ModalityMode.ONE_HAND:
            return 63
        return FEATURE_DIM_HANDS_FACE

    def extract(
        self,
        frame: MultimodalFrame,
        mode: ModalityMode | None = None,
    ) -> np.ndarray:
        """Extract a 1D float32 normalized feature vector from a MultimodalFrame.

        Args:
            frame: Synchronized MultimodalFrame.
            mode: Desired modality mode. Defaults to self._default_mode.

        Returns:
            1D float32 numpy array with fixed shape matching the mode.
        """
        target_mode = mode or self._default_mode

        # Legacy single-hand mode: extracts exactly (63,)
        if target_mode == ModalityMode.ONE_HAND:
            return self._extract_single_hand(frame)

        # Multimodal feature vector
        target_dim = self.get_feature_dim(target_mode)
        feature_vec = np.zeros(target_dim, dtype=np.float32)

        # ── 1. Determine Global Spatial Reference Anchor & Scale ────────
        anchor_x, anchor_y, anchor_z = 0.0, 0.0, 0.0
        spatial_scale = 1.0

        if frame.has_face and len(frame.face_landmarks) >= 20:
            # Face anchor: nose tip (landmark index 18 in 24-point subset = index 1 in full face)
            # In FACE_EXPRESSIVE_INDICES, index 1 is at index position 20
            nose_idx = 20 if len(frame.face_landmarks) > 20 else 0
            nose = frame.face_landmarks[nose_idx]
            anchor_x, anchor_y, anchor_z = nose.x, nose.y, nose.z

            # Face scale: distance between outer eye corners (index 6: 33, and index 10: 263)
            p_le = frame.face_landmarks[6]
            p_re = frame.face_landmarks[10]
            eye_dist = float(np.sqrt((p_le.x - p_re.x) ** 2 + (p_le.y - p_re.y) ** 2))
            spatial_scale = max(eye_dist * 2.0, 0.05)  # approximate head width

        elif frame.has_pose and len(frame.pose_landmarks) >= 3:
            # Pose anchor: midpoint between shoulders (left: index 1, right: index 2)
            ls = frame.pose_landmarks[1]
            rs = frame.pose_landmarks[2]
            anchor_x = (ls.x + rs.x) / 2.0
            anchor_y = (ls.y + rs.y) / 2.0
            anchor_z = (ls.z + rs.z) / 2.0
            shoulder_dist = float(np.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2))
            spatial_scale = max(shoulder_dist, 0.1)

        # ── 2. Vectorize Left Hand (63 dims) ─────────────────────────
        lh_slice = slice(0, HAND_DIM)
        if frame.has_left_hand and frame.left_hand is not None:
            lh_arr = np.zeros((HAND_LANDMARKS_COUNT, 3), dtype=np.float32)
            for i, lm in enumerate(frame.left_hand.landmarks):
                lh_arr[i] = [lm.x, lm.y, lm.z]

            if frame.has_face or frame.has_pose:
                # Spatial normalization relative to face/body anchor
                lh_arr[:, 0] = (lh_arr[:, 0] - anchor_x) / spatial_scale
                lh_arr[:, 1] = (lh_arr[:, 1] - anchor_y) / spatial_scale
                lh_arr[:, 2] = (lh_arr[:, 2] - anchor_z) / spatial_scale
            else:
                # Local wrist normalization if no face/body visible
                wrist = lh_arr[0].copy()
                lh_arr -= wrist
                palm_dist = max(float(np.linalg.norm(lh_arr[5] - lh_arr[17])), 1e-4)
                lh_arr /= palm_dist

            feature_vec[lh_slice] = lh_arr.flatten()

        # ── 3. Vectorize Right Hand (63 dims) ────────────────────────
        rh_slice = slice(HAND_DIM, DUAL_HAND_DIM)
        if frame.has_right_hand and frame.right_hand is not None:
            rh_arr = np.zeros((HAND_LANDMARKS_COUNT, 3), dtype=np.float32)
            for i, lm in enumerate(frame.right_hand.landmarks):
                rh_arr[i] = [lm.x, lm.y, lm.z]

            if frame.has_face or frame.has_pose:
                # Spatial normalization relative to face/body anchor
                rh_arr[:, 0] = (rh_arr[:, 0] - anchor_x) / spatial_scale
                rh_arr[:, 1] = (rh_arr[:, 1] - anchor_y) / spatial_scale
                rh_arr[:, 2] = (rh_arr[:, 2] - anchor_z) / spatial_scale
            else:
                # Local wrist normalization if no face/body visible
                wrist = rh_arr[0].copy()
                rh_arr -= wrist
                palm_dist = max(float(np.linalg.norm(rh_arr[5] - rh_arr[17])), 1e-4)
                rh_arr /= palm_dist

            feature_vec[rh_slice] = rh_arr.flatten()

        # ── 4. Vectorize Face (72 dims) ──────────────────────────────
        current_offset = DUAL_HAND_DIM
        if target_mode in (ModalityMode.TWO_HANDS_FACE, ModalityMode.TWO_HANDS_FACE_POSE):
            face_slice = slice(current_offset, current_offset + FACE_DIM)
            if frame.has_face and frame.face_landmarks:
                face_arr = np.zeros((len(frame.face_landmarks), 3), dtype=np.float32)
                for i, lm in enumerate(frame.face_landmarks):
                    face_arr[i] = [
                        (lm.x - anchor_x) / spatial_scale,
                        (lm.y - anchor_y) / spatial_scale,
                        (lm.z - anchor_z) / spatial_scale,
                    ]
                feature_vec[face_slice] = face_arr.flatten()
            current_offset += FACE_DIM

        # ── 5. Vectorize Pose (24 dims) ──────────────────────────────
        if target_mode == ModalityMode.TWO_HANDS_FACE_POSE:
            pose_slice = slice(current_offset, current_offset + POSE_DIM)
            if frame.has_pose and frame.pose_landmarks:
                pose_arr = np.zeros((len(frame.pose_landmarks), 3), dtype=np.float32)
                for i, lm in enumerate(frame.pose_landmarks):
                    pose_arr[i] = [
                        (lm.x - anchor_x) / spatial_scale,
                        (lm.y - anchor_y) / spatial_scale,
                        (lm.z - anchor_z) / spatial_scale,
                    ]
                feature_vec[pose_slice] = pose_arr.flatten()
            current_offset += POSE_DIM

        # ── 6. Vectorize Quality, Presence & Indicators (8 dims) ─────
        quality_slice = slice(current_offset, current_offset + QUALITY_METRICS_DIM)
        feature_vec[quality_slice] = np.array([
            1.0 if frame.has_left_hand else 0.0,
            1.0 if frame.has_right_hand else 0.0,
            1.0 if frame.has_face else 0.0,
            1.0 if frame.has_pose else 0.0,
            frame.mouth_openness,
            frame.eyebrow_elevation,
            frame.left_hand_to_face_dist,
            frame.right_hand_to_face_dist,
        ], dtype=np.float32)

        return feature_vec

    def _extract_single_hand(self, frame: MultimodalFrame) -> np.ndarray:
        """Extract legacy single-hand 63-dim feature vector."""
        hand = frame.right_hand or frame.left_hand
        if hand is None or len(hand.landmarks) != 21:
            return np.zeros(63, dtype=np.float32)

        raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmarks], dtype=np.float32)
        # Wrist-centered and scale normalized
        wrist = raw[0].copy()
        raw -= wrist
        scale = max(float(np.linalg.norm(raw[5] - raw[17])), 1e-4)
        raw /= scale
        return raw.flatten()

    def get_schema(self, mode: ModalityMode) -> MultimodalFeatureSchema:
        """Return the schema descriptor for the given modality mode."""
        dim = self.get_feature_dim(mode)
        has_face = mode in (ModalityMode.TWO_HANDS_FACE, ModalityMode.TWO_HANDS_FACE_POSE)
        has_pose = mode == ModalityMode.TWO_HANDS_FACE_POSE

        curr = DUAL_HAND_DIM
        face_slice = (curr, curr + FACE_DIM) if has_face else None
        if has_face:
            curr += FACE_DIM

        pose_slice = (curr, curr + POSE_DIM) if has_pose else None
        if has_pose:
            curr += POSE_DIM

        quality_slice = (curr, curr + QUALITY_METRICS_DIM)

        return MultimodalFeatureSchema(
            version=MULTIMODAL_SCHEMA_VERSION,
            mode=mode.value,
            feature_dim=dim,
            has_face=has_face,
            has_pose=has_pose,
            left_hand_slice=(0, HAND_DIM),
            right_hand_slice=(HAND_DIM, DUAL_HAND_DIM),
            face_slice=face_slice or (0, 0),
            pose_slice=pose_slice,
            quality_slice=quality_slice,
        )
