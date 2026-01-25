"""
Tests for recording processing pipeline.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.process_recordings import (
    filter_low_confidence,
    filter_outliers,
    detect_static_frames,
    filter_frames,
    average_landmarks_mean,
    average_landmarks_median,
    average_landmarks_trimmed,
    average_landmarks_weighted,
    calculate_position_tolerances,
    calculate_angle_tolerances,
    calculate_tolerances,
    process_recording,
    process_multiple_recordings,
    ProcessingConfig,
)
from src.types import (
    Recording,
    RecordingFrame,
    DetectedHand,
    HandLandmarks,
    HandLandmark,
    Point3D,
)
from src.landmarks import LANDMARK_NAMES


# =============================================================================
# Fixtures
# =============================================================================


def create_landmarks(offset: float = 0.0, noise: float = 0.0) -> np.ndarray:
    """Create test landmarks with optional offset and noise."""
    # Basic hand shape
    landmarks = np.zeros((21, 3))

    # Wrist at base
    landmarks[0] = [0.5 + offset, 0.7, 0.0]

    # Thumb
    landmarks[1] = [0.55 + offset, 0.68, 0.01]
    landmarks[2] = [0.58 + offset, 0.66, 0.02]
    landmarks[3] = [0.60 + offset, 0.64, 0.02]
    landmarks[4] = [0.62 + offset, 0.62, 0.02]

    # Index
    landmarks[5] = [0.54 + offset, 0.60, 0.0]
    landmarks[6] = [0.54 + offset, 0.56, 0.0]
    landmarks[7] = [0.54 + offset, 0.53, 0.0]
    landmarks[8] = [0.54 + offset, 0.50, 0.0]

    # Middle
    landmarks[9] = [0.50 + offset, 0.59, 0.0]
    landmarks[10] = [0.50 + offset, 0.54, 0.0]
    landmarks[11] = [0.50 + offset, 0.51, 0.0]
    landmarks[12] = [0.50 + offset, 0.48, 0.0]

    # Ring
    landmarks[13] = [0.46 + offset, 0.60, 0.0]
    landmarks[14] = [0.46 + offset, 0.56, 0.0]
    landmarks[15] = [0.46 + offset, 0.53, 0.0]
    landmarks[16] = [0.46 + offset, 0.50, 0.0]

    # Pinky
    landmarks[17] = [0.42 + offset, 0.62, 0.0]
    landmarks[18] = [0.42 + offset, 0.59, 0.0]
    landmarks[19] = [0.42 + offset, 0.57, 0.0]
    landmarks[20] = [0.42 + offset, 0.55, 0.0]

    if noise > 0:
        landmarks += np.random.randn(*landmarks.shape) * noise
        landmarks = np.clip(landmarks, 0, 1)

    return landmarks


def create_frame_sequence(
    num_frames: int = 10,
    base_confidence: float = 0.9,
    motion: bool = True
) -> list[tuple[int, np.ndarray, float]]:
    """Create a sequence of frames for testing."""
    frames = []
    for i in range(num_frames):
        offset = i * 0.01 if motion else 0.0
        landmarks = create_landmarks(offset=offset, noise=0.001)
        confidence = base_confidence + np.random.uniform(-0.05, 0.05)
        confidence = max(0.0, min(1.0, confidence))
        frames.append((i * 33, landmarks, confidence))
    return frames


def create_recording(num_frames: int = 10) -> Recording:
    """Create a test Recording object."""
    frames = []

    for i in range(num_frames):
        landmarks = create_landmarks(offset=i * 0.01, noise=0.001)

        hand_landmarks = []
        for j in range(21):
            hand_landmarks.append(HandLandmark(
                point=Point3D(
                    x=float(landmarks[j, 0]),
                    y=float(landmarks[j, 1]),
                    z=float(landmarks[j, 2])
                ),
                name=LANDMARK_NAMES[j],
                index=j
            ))

        detected_hand = DetectedHand(
            landmarks=HandLandmarks(landmarks=hand_landmarks),
            handedness="Right",
            confidence=0.9
        )

        frames.append(RecordingFrame(
            timestamp_ms=i * 33,
            right_hand=detected_hand
        ))

    return Recording(
        frames=frames,
        sign_id="test_sign",
        start_time=datetime.now(),
        end_time=datetime.now(),
        metadata={}
    )


# =============================================================================
# Frame Filtering Tests
# =============================================================================


class TestFilterLowConfidence:
    """Tests for confidence filtering."""

    def test_keeps_high_confidence(self):
        frames = [(0, np.zeros((21, 3)), 0.9)]
        result = filter_low_confidence(frames, threshold=0.7)
        assert len(result) == 1

    def test_removes_low_confidence(self):
        frames = [(0, np.zeros((21, 3)), 0.5)]
        result = filter_low_confidence(frames, threshold=0.7)
        assert len(result) == 0

    def test_threshold_boundary(self):
        frames = [(0, np.zeros((21, 3)), 0.7)]
        result = filter_low_confidence(frames, threshold=0.7)
        assert len(result) == 1

    def test_mixed_confidence(self):
        frames = [
            (0, np.zeros((21, 3)), 0.9),
            (33, np.zeros((21, 3)), 0.5),
            (66, np.zeros((21, 3)), 0.8),
        ]
        result = filter_low_confidence(frames, threshold=0.7)
        assert len(result) == 2


class TestFilterOutliers:
    """Tests for outlier filtering."""

    def test_keeps_normal_frames(self):
        frames = create_frame_sequence(10)
        result = filter_outliers(frames, method="zscore", threshold=2.0)
        # Most frames should be kept
        assert len(result) >= 8

    def test_removes_outlier(self):
        frames = create_frame_sequence(10)
        # Add an outlier
        outlier = (500, create_landmarks(offset=0.5), 0.9)  # Far from others
        frames.append(outlier)

        result = filter_outliers(frames, method="zscore", threshold=2.0)
        assert len(result) < len(frames)

    def test_iqr_method(self):
        frames = create_frame_sequence(10)
        result = filter_outliers(frames, method="iqr", threshold=1.5)
        # Should keep at least half the frames
        assert len(result) >= 4

    def test_few_frames_unchanged(self):
        frames = create_frame_sequence(3)
        result = filter_outliers(frames, method="zscore")
        assert len(result) == len(frames)


class TestDetectStaticFrames:
    """Tests for static frame detection."""

    def test_removes_static(self):
        # Create truly static frames (identical landmarks)
        base_landmarks = create_landmarks()
        frames = [(i * 33, base_landmarks.copy(), 0.9) for i in range(20)]
        result = detect_static_frames(frames, motion_threshold=0.005)
        # Most should be removed as static (motion = 0)
        assert len(result) <= len(frames)

    def test_keeps_moving(self):
        frames = create_frame_sequence(20, motion=True)
        result = detect_static_frames(frames, motion_threshold=0.001)
        # Most should be kept
        assert len(result) >= 10

    def test_few_frames_unchanged(self):
        frames = create_frame_sequence(4, motion=False)
        result = detect_static_frames(frames)
        assert len(result) == len(frames)


class TestFilterFrames:
    """Tests for combined filtering."""

    def test_applies_all_filters(self):
        frames = create_frame_sequence(20)
        result = filter_frames(
            frames,
            confidence_threshold=0.7,
            outlier_method="zscore",
            remove_static=True
        )
        assert len(result) > 0
        assert len(result) <= len(frames)

    def test_no_outlier_filter(self):
        frames = create_frame_sequence(20)
        result = filter_frames(
            frames,
            outlier_method="none"
        )
        assert len(result) > 0


# =============================================================================
# Averaging Tests
# =============================================================================


class TestAverageLandmarks:
    """Tests for landmark averaging functions."""

    def test_mean_average(self):
        samples = [create_landmarks(offset=i*0.01) for i in range(5)]
        result = average_landmarks_mean(samples)

        assert result.shape == (21, 3)
        # Should be roughly in the middle
        assert 0.4 < result[0, 0] < 0.6

    def test_median_average(self):
        samples = [create_landmarks(offset=i*0.01) for i in range(5)]
        result = average_landmarks_median(samples)

        assert result.shape == (21, 3)

    def test_trimmed_average(self):
        samples = [create_landmarks(offset=i*0.01) for i in range(10)]
        result = average_landmarks_trimmed(samples, trim=0.1)

        assert result.shape == (21, 3)

    def test_weighted_average(self):
        samples = [create_landmarks(offset=0), create_landmarks(offset=0.2)]
        weights = [0.9, 0.1]
        result = average_landmarks_weighted(samples, weights)

        # Should be closer to first sample (0.5) than second (0.7)
        # Weighted avg: 0.5*0.9 + 0.7*0.1 = 0.52
        assert result[0, 0] < 0.55

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            average_landmarks_mean([])

    def test_single_sample(self):
        sample = create_landmarks()
        result = average_landmarks_mean([sample])

        np.testing.assert_array_almost_equal(result, sample)


# =============================================================================
# Tolerance Calculation Tests
# =============================================================================


class TestToleranceCalculation:
    """Tests for tolerance calculation."""

    def test_position_tolerances(self):
        canonical = create_landmarks()
        samples = [create_landmarks(noise=0.01) for _ in range(10)]

        tolerances = calculate_position_tolerances(samples, canonical)

        assert len(tolerances) == 21
        assert all(name in tolerances for name in LANDMARK_NAMES)
        assert all(t >= 0 for t in tolerances.values())

    def test_angle_tolerances(self):
        canonical = create_landmarks()
        samples = [create_landmarks(noise=0.01) for _ in range(10)]

        tolerances = calculate_angle_tolerances(samples, canonical)

        assert len(tolerances) == 15  # 15 joints
        assert all(t >= 0 for t in tolerances.values())

    def test_combined_tolerances(self):
        canonical = create_landmarks()
        samples = [create_landmarks(noise=0.01) for _ in range(10)]

        result = calculate_tolerances(samples, canonical)

        assert result.overall_position_tolerance >= 0
        assert result.overall_angle_tolerance >= 0
        assert len(result.position_tolerances) == 21
        assert len(result.angle_tolerances) == 15

    def test_zero_tolerance_identical_samples(self):
        canonical = create_landmarks()
        samples = [canonical.copy() for _ in range(5)]

        result = calculate_tolerances(samples, canonical)

        assert result.overall_position_tolerance < 1e-10


# =============================================================================
# Recording Processing Tests
# =============================================================================


class TestProcessRecording:
    """Tests for recording processing."""

    def test_process_valid_recording(self):
        recording = create_recording(20)
        result = process_recording(recording)

        assert result is not None
        assert result.sample_count > 0
        assert 0 <= result.quality_score <= 1
        assert result.canonical_pose is not None

    def test_process_with_config(self):
        recording = create_recording(20)
        config = ProcessingConfig(
            confidence_threshold=0.5,
            averaging_method="median"
        )
        result = process_recording(recording, config)

        assert result is not None

    def test_canonical_pose_has_landmarks(self):
        recording = create_recording(20)
        result = process_recording(recording)

        pose = result.canonical_pose
        # Recording has right hand, so should have right landmarks
        assert pose.right_hand_landmarks is not None
        assert len(pose.right_hand_landmarks) == 21

    def test_tolerances_in_pose(self):
        recording = create_recording(20)
        result = process_recording(recording)

        pose = result.canonical_pose
        assert len(pose.tolerances) > 0


class TestProcessMultipleRecordings:
    """Tests for processing multiple recordings."""

    def test_process_multiple(self):
        recordings = [create_recording(15) for _ in range(3)]
        result = process_multiple_recordings(recordings)

        assert result is not None
        assert result.sample_count > 0

    def test_combined_sample_count(self):
        recordings = [create_recording(10) for _ in range(3)]
        result = process_multiple_recordings(recordings)

        # Should have samples from multiple recordings
        assert result.sample_count >= 10

    def test_empty_recordings_returns_none(self):
        result = process_multiple_recordings([])
        assert result is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_few_frames(self):
        recording = create_recording(3)
        config = ProcessingConfig(min_frames=2)
        result = process_recording(recording, config)

        assert result is not None

    def test_all_low_confidence(self):
        recording = create_recording(10)
        # Manually set low confidence
        for frame in recording.frames:
            if frame.right_hand:
                frame.right_hand = DetectedHand(
                    landmarks=frame.right_hand.landmarks,
                    handedness="Right",
                    confidence=0.3
                )

        config = ProcessingConfig(confidence_threshold=0.5)
        result = process_recording(recording, config)

        # All frames filtered out, so should return None
        assert result is None

    def test_low_confidence_below_threshold_still_works(self):
        recording = create_recording(10)
        # Set confidence just below default threshold
        for frame in recording.frames:
            if frame.right_hand:
                frame.right_hand = DetectedHand(
                    landmarks=frame.right_hand.landmarks,
                    handedness="Right",
                    confidence=0.6
                )

        # With lower threshold, should still work
        config = ProcessingConfig(confidence_threshold=0.5)
        result = process_recording(recording, config)

        assert result is not None


class TestProcessingConfig:
    """Tests for ProcessingConfig."""

    def test_default_config(self):
        config = ProcessingConfig()

        assert config.confidence_threshold == 0.7
        assert config.averaging_method == "trimmed"
        assert config.min_frames == 5

    def test_custom_config(self):
        config = ProcessingConfig(
            confidence_threshold=0.5,
            averaging_method="mean",
            min_frames=3
        )

        assert config.confidence_threshold == 0.5
        assert config.averaging_method == "mean"
        assert config.min_frames == 3
