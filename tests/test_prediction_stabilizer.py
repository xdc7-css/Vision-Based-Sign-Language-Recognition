"""Unit tests for PredictionStabilizer state machine and duplicate prevention."""

from __future__ import annotations

import pytest

from src.domain.stabilizer import PredictionStabilizer, StabilizerState


def test_initial_state():
    stabilizer = PredictionStabilizer(min_confidence=0.7, min_stable_frames=5)
    assert stabilizer.state == StabilizerState.IDLE


def test_single_prediction_does_not_emit():
    stabilizer = PredictionStabilizer(min_confidence=0.7, min_stable_frames=5)
    res = stabilizer.update(raw_sign="مرحبا", confidence=0.85, has_hands=True, hand_y=0.5)

    assert res.emitted_sign is None
    assert res.candidate_sign == "مرحبا"
    assert res.state == StabilizerState.CANDIDATE
    assert res.progress < 1.0


def test_stable_sign_emits_exactly_once():
    stabilizer = PredictionStabilizer(min_confidence=0.7, min_stable_frames=5, release_frames=3)

    emissions = []
    for i in range(10):
        res = stabilizer.update(raw_sign="مرحبا", confidence=0.9, has_hands=True, hand_y=0.5)
        if res.emitted_sign:
            emissions.append((i, res.emitted_sign))

    # Exactly ONE emission occurred
    assert len(emissions) == 1
    assert emissions[0][1] == "مرحبا"

    # State should now be WAIT_FOR_RELEASE
    assert stabilizer.state == StabilizerState.WAIT_FOR_RELEASE


def test_duplicate_prevention_under_continuous_holding():
    stabilizer = PredictionStabilizer(min_confidence=0.7, min_stable_frames=4, release_frames=3)

    emissions = []
    # Hold for 50 consecutive frames
    for _ in range(50):
        res = stabilizer.update(raw_sign="شكرا", confidence=0.88, has_hands=True, hand_y=0.4)
        if res.emitted_sign:
            emissions.append(res.emitted_sign)

    # Must be emitted exactly once, NO infinite copies
    assert len(emissions) == 1
    assert emissions == ["شكرا"]
    assert res.is_holding is True


def test_neutral_and_release_detection():
    stabilizer = PredictionStabilizer(min_confidence=0.7, min_stable_frames=4, release_frames=3)

    # 1. Emit first word
    for _ in range(4):
        stabilizer.update(raw_sign="نعم", confidence=0.92, has_hands=True, hand_y=0.4)

    assert stabilizer.state == StabilizerState.WAIT_FOR_RELEASE

    # 2. Drop hands for 3 frames (release)
    for _ in range(3):
        res = stabilizer.update(raw_sign="", confidence=0.0, has_hands=False)
        assert res.emitted_sign is None

    assert stabilizer.state == StabilizerState.IDLE

    # 3. Raise hands and perform the same sign again
    second_emissions = []
    for _ in range(4):
        res = stabilizer.update(raw_sign="نعم", confidence=0.92, has_hands=True, hand_y=0.4)
        if res.emitted_sign:
            second_emissions.append(res.emitted_sign)

    # Now it can legitimately emit again because user clearly released
    assert len(second_emissions) == 1
    assert second_emissions == ["نعم"]


def test_resting_hand_rejected_as_neutral():
    stabilizer = PredictionStabilizer(min_confidence=0.7, min_stable_frames=4, neutral_y_threshold=0.80)

    # Hand resting at y=0.88 (near bottom of screen)
    for _ in range(10):
        res = stabilizer.update(raw_sign="آسف", confidence=0.95, has_hands=True, hand_y=0.88)
        assert res.emitted_sign is None
        assert res.is_neutral is True


def test_flickering_predictions_do_not_emit():
    stabilizer = PredictionStabilizer(min_confidence=0.7, min_stable_frames=5, agreement_threshold=0.8)

    # Rapidly alternating signs
    flicker = ["أكل", "نوم", "تعال", "شكرا", "أكل", "نوم"]
    emissions = []
    for sign in flicker:
        res = stabilizer.update(raw_sign=sign, confidence=0.85, has_hands=True, hand_y=0.5)
        if res.emitted_sign:
            emissions.append(res.emitted_sign)

    assert len(emissions) == 0

