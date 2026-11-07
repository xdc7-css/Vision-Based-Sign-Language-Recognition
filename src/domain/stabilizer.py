"""Prediction stabilization state machine for real-time sign language recognition.

Prevents rapid prediction jitter, eliminates unwanted word duplication,
requires temporal confirmation before emission, and detects neutral/release states.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StabilizerState(str, Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    EMITTED = "emitted"
    WAIT_FOR_RELEASE = "wait_for_release"
    READY = "ready"


@dataclass
class StabilizationResult:
    """Result of updating the prediction stabilizer for a single frame."""
    state: StabilizerState
    candidate_sign: str
    confidence: float
    progress: float  # 0.0 to 1.0 (progress towards confirmation)
    emitted_sign: Optional[str] = None  # Non-None only on the exact frame the sign is confirmed
    is_holding: bool = False
    is_neutral: bool = True
    status_label: str = "Waiting for sign..."


class PredictionStabilizer:
    """State machine governing live candidate display vs sentence word emission.

    Lifecycle:
        IDLE
        → (confidence >= threshold) → CANDIDATE (accumulates evidence)
        → (stable for N frames with >= 80% agreement) → CONFIRMED
        → EMITTED (emits word to sentence builder exactly ONCE)
        → WAIT_FOR_RELEASE (holds sign without re-emitting duplicates)
        → (hands lowered / neutral for M frames) → IDLE / READY
    """

    def __init__(
        self,
        min_confidence: float = 0.70,
        min_stable_frames: int = 6,
        agreement_threshold: float = 0.80,
        release_frames: int = 4,
        neutral_y_threshold: float = 0.84,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_stable_frames = max(2, min_stable_frames)
        self.agreement_threshold = agreement_threshold
        self.release_frames = max(1, release_frames)
        self.neutral_y_threshold = neutral_y_threshold

        self._state = StabilizerState.IDLE
        self._history: list[tuple[str, float]] = []
        self._current_candidate: str = ""
        self._current_confidence: float = 0.0
        self._emitted_sign: str = ""
        self._release_count: int = 0
        self._last_emitted_time: float = 0.0

    @property
    def state(self) -> StabilizerState:
        return self._state

    def reset(self) -> None:
        """Reset stabilizer to clean initial state."""
        self._state = StabilizerState.IDLE
        self._history.clear()
        self._current_candidate = ""
        self._current_confidence = 0.0
        self._emitted_sign = ""
        self._release_count = 0

    def update(
        self,
        raw_sign: str,
        confidence: float,
        has_hands: bool,
        hand_y: Optional[float] = None,
    ) -> StabilizationResult:
        """Process incoming frame prediction and update state machine.

        Args:
            raw_sign: Raw predicted Arabic sign text from classifier.
            confidence: Raw prediction probability (0.0 to 1.0).
            has_hands: Whether at least one hand is detected in frame.
            hand_y: Normalized vertical position of hand center or wrist (0.0=top, 1.0=bottom).

        Returns:
            StabilizationResult detailing current state, candidate, progress, and any emitted word.
        """
        # Determine if hand is resting at the bottom of the frame
        is_resting = hand_y is not None and hand_y > self.neutral_y_threshold
        is_neutral = (not has_hands) or is_resting or (confidence < 0.40 and not raw_sign)

        # ── 1. Neutral / Release Handling ────────────────────────────────────
        if is_neutral:
            self._release_count += 1
            if self._release_count >= self.release_frames:
                # Fully returned to neutral: reset candidate and release lock
                self._state = StabilizerState.IDLE
                self._history.clear()
                self._current_candidate = ""
                self._current_confidence = 0.0
                self._emitted_sign = ""

            return StabilizationResult(
                state=self._state,
                candidate_sign="" if self._state == StabilizerState.IDLE else self._current_candidate,
                confidence=0.0 if self._state == StabilizerState.IDLE else self._current_confidence,
                progress=0.0,
                emitted_sign=None,
                is_holding=self._state == StabilizerState.WAIT_FOR_RELEASE,
                is_neutral=True,
                status_label="Neutral / Resting" if is_resting else "Waiting for sign...",
            )

        # Hands are actively in signing position
        self._release_count = 0

        # ── 2. Holding Lock (Duplicate Prevention) ───────────────────────────
        if self._state == StabilizerState.WAIT_FOR_RELEASE:
            # User is holding the sign they already emitted
            if raw_sign == self._emitted_sign and confidence >= self.min_confidence:
                self._current_confidence = 0.8 * self._current_confidence + 0.2 * confidence
                return StabilizationResult(
                    state=StabilizerState.WAIT_FOR_RELEASE,
                    candidate_sign=self._emitted_sign,
                    confidence=self._current_confidence,
                    progress=1.0,
                    emitted_sign=None,  # Do NOT emit duplicates!
                    is_holding=True,
                    is_neutral=False,
                    status_label=f"Holding: «{self._emitted_sign}»",
                )
            else:
                # Hand pose changed while in hold: allow transition to new candidate if stable
                self._history.append((raw_sign, confidence))
                if len(self._history) > self.min_stable_frames:
                    self._history.pop(0)

                # Check if a new sign has taken over history
                signs = [s for s, _ in self._history if s and s != self._emitted_sign]
                if len(signs) >= self.min_stable_frames:
                    # New distinct gesture detected
                    self._state = StabilizerState.CANDIDATE
                    self._emitted_sign = ""
                else:
                    return StabilizationResult(
                        state=StabilizerState.WAIT_FOR_RELEASE,
                        candidate_sign=self._emitted_sign,
                        confidence=self._current_confidence,
                        progress=1.0,
                        emitted_sign=None,
                        is_holding=True,
                        is_neutral=False,
                        status_label=f"Holding: «{self._emitted_sign}»",
                    )

        # ── 3. Confidence Threshold Check ────────────────────────────────────
        if confidence < self.min_confidence or not raw_sign:
            if self._history:
                self._history.pop(0)
            if not self._history:
                self._state = StabilizerState.IDLE
                self._current_candidate = ""
                self._current_confidence = 0.0

            return StabilizationResult(
                state=self._state,
                candidate_sign=self._current_candidate,
                confidence=self._current_confidence,
                progress=len(self._history) / float(self.min_stable_frames) if self.min_stable_frames > 0 else 0.0,
                emitted_sign=None,
                is_holding=False,
                is_neutral=False,
                status_label="Low confidence" if raw_sign else "Detecting sign...",
            )

        # ── 4. Candidate Accumulation ────────────────────────────────────────
        self._history.append((raw_sign, confidence))
        if len(self._history) > self.min_stable_frames:
            self._history.pop(0)

        # Analyze sign agreement in history window
        signs = [s for s, _ in self._history]
        counts = Counter(signs)
        mode_sign, mode_count = counts.most_common(1)[0]
        agreement = mode_count / float(len(self._history))

        # Average confidence for the mode sign
        mode_confs = [c for s, c in self._history if s == mode_sign]
        avg_conf = sum(mode_confs) / float(len(mode_confs))

        self._current_candidate = mode_sign
        self._current_confidence = avg_conf

        progress = min(1.0, len(self._history) / float(self.min_stable_frames))

        # Check if confirmation conditions are satisfied
        if len(self._history) >= self.min_stable_frames and agreement >= self.agreement_threshold:
            # Sign is stably held!
            self._state = StabilizerState.EMITTED
            self._emitted_sign = mode_sign
            emitted = mode_sign

            # Transition immediately to WAIT_FOR_RELEASE to prevent any duplicate emits
            self._state = StabilizerState.WAIT_FOR_RELEASE
            self._history.clear()

            return StabilizationResult(
                state=StabilizerState.EMITTED,
                candidate_sign=mode_sign,
                confidence=avg_conf,
                progress=1.0,
                emitted_sign=emitted,
                is_holding=True,
                is_neutral=False,
                status_label=f"Confirmed: «{mode_sign}»",
            )

        self._state = StabilizerState.CANDIDATE
        return StabilizationResult(
            state=StabilizerState.CANDIDATE,
            candidate_sign=mode_sign,
            confidence=avg_conf,
            progress=progress,
            emitted_sign=None,
            is_holding=False,
            is_neutral=False,
            status_label=f"Confirming ({int(progress * 100)}%): «{mode_sign}»",
        )

