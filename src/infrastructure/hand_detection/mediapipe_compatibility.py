from __future__ import annotations
import sys
import urllib.request

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class MockLandmark:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class MockLandmarkList:
    def __init__(self, landmark_list):
        self.landmark = landmark_list

class MockClassification:
    def __init__(self, label, score):
        self.label = label
        self.score = float(score)

class MockHandedness:
    def __init__(self, classification_list):
        self.classification = classification_list

class MockResults:
    def __init__(self, multi_hand_landmarks, multi_handedness):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness = multi_handedness

class HandsCompatibilityWrapper:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        model_complexity: int = 0,
    ) -> None:
        self.model_path = self._resolve_model_path()

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _resolve_model_path() -> Path:
        candidates = [
            Path(__file__).resolve().parents[3] / "data" / "models" / "general" / "hand_landmarker.task",
            Path("data/models/general/hand_landmarker.task").resolve(),
        ]
        for p in candidates:
            if p.exists():
                return p

        # Default fallback destination
        target = candidates[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading hand_landmarker.task model for MediaPipe Tasks...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, str(target))
        logger.info("Download completed.")
        return target

    def __enter__(self) -> HandsCompatibilityWrapper:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, 'detector') and self.detector:
            try:
                self.detector.close()
            except Exception:
                pass

    def process(self, rgb_image) -> MockResults:
        import mediapipe as mp

        if not isinstance(rgb_image, np.ndarray):
            rgb_image = np.asarray(rgb_image)
        if not rgb_image.flags.c_contiguous:
            rgb_image = np.ascontiguousarray(rgb_image)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        try:
            result = self.detector.detect(mp_image)
        except Exception as e:
            logger.error("MediaPipe HandLandmarker inference error: %s", e)
            return MockResults([], [])

        multi_hand_landmarks = []
        multi_handedness = []

        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                landmark_list = [MockLandmark(lm.x, lm.y, lm.z) for lm in landmarks]
                multi_hand_landmarks.append(MockLandmarkList(landmark_list))

            if result.handedness:
                for handedness in result.handedness:
                    class_list = [
                        MockClassification(
                            getattr(c, "display_name", None) or getattr(c, "category_name", None) or ("Left" if getattr(c, "index", 1) == 0 else "Right"),
                            getattr(c, "score", 0.0)
                        )
                        for c in handedness
                    ]
                    multi_handedness.append(MockHandedness(class_list))

        return MockResults(multi_hand_landmarks, multi_handedness)


def patch_mediapipe() -> None:
    """Monkey-patch MediaPipe solutions/hands module if missing in the environment."""
    import mediapipe as mp
    try:
        # Check if legacy solutions are missing
        _ = mp.solutions.hands
    except AttributeError:
        # Monkey-patch solutions
        class MockSolutions:
            class MockHands:
                Hands = HandsCompatibilityWrapper
            hands = MockHands()

        mp.solutions = MockSolutions()
        sys.modules['mediapipe.solutions'] = MockSolutions()
        sys.modules['mediapipe.solutions.hands'] = MockSolutions.MockHands()
        print("MediaPipe solutions.hands monkey-patched successfully.")
