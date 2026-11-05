"""Hand detection module — MediaPipe-based real-time hand tracking.

Services:
    HandDetector    — Wraps MediaPipe Hands, produces HandDetectionResult.
    HandRenderer    — Draws landmarks, skeleton, labels on the image.

Utilities:
    LandmarkProcessor — Normalization, bounding box, finger states, geometry.
"""
