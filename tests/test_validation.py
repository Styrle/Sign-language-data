"""Tests for the validation module."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.landmarks import LANDMARK_NAMES, NUM_LANDMARKS
from src.types import (
    DetectedHand,
    HandLandmark,
    HandLandmarks,
    Point3D,
    Recording,
    RecordingFrame,
    SignDefinition,
    SignDictionaryEntry,
    SignPose,
)
from src.validation import (
    ANATOMICAL_LIMITS,
    calculate_landmark_quality,
    calculate_recording_quality,
    calculate_sign_quality,
    check_finger_lengths,
    check_joint_angles,
    generate_validation_report,
    summarize_validation_results,
    validate_frame,
    validate_landmarks,
    validate_point,
    validate_recording,
    validate_sign_entry,
    validate_sign_pose,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_point() -> Point3D:
    """Create a valid Point3D."""
    return Point3D(x=0.5, y=0.5, z=0.0)


@pytest.fixture
def valid_landmarks() -> HandLandmarks:
    """Create valid hand landmarks with realistic positions.

    This creates a roughly anatomically correct hand shape with
    proper finger proportions.
    """
    # Create landmarks that form a realistic hand shape
    # Wrist at bottom, fingers extending upward
    base_positions = [
        (0.5, 0.9, 0.0),    # WRIST
        # Thumb (offset to side)
        (0.35, 0.85, 0.0),  # THUMB_CMC
        (0.28, 0.78, 0.0),  # THUMB_MCP
        (0.23, 0.70, 0.0),  # THUMB_IP
        (0.20, 0.62, 0.0),  # THUMB_TIP
        # Index finger
        (0.38, 0.70, 0.0),  # INDEX_MCP
        (0.37, 0.55, 0.0),  # INDEX_PIP
        (0.36, 0.43, 0.0),  # INDEX_DIP
        (0.35, 0.32, 0.0),  # INDEX_TIP
        # Middle finger (longest)
        (0.48, 0.68, 0.0),  # MIDDLE_MCP
        (0.48, 0.50, 0.0),  # MIDDLE_PIP
        (0.48, 0.36, 0.0),  # MIDDLE_DIP
        (0.48, 0.24, 0.0),  # MIDDLE_TIP
        # Ring finger
        (0.58, 0.70, 0.0),  # RING_MCP
        (0.59, 0.53, 0.0),  # RING_PIP
        (0.60, 0.40, 0.0),  # RING_DIP
        (0.61, 0.30, 0.0),  # RING_TIP
        # Pinky (shortest)
        (0.68, 0.73, 0.0),  # PINKY_MCP
        (0.70, 0.60, 0.0),  # PINKY_PIP
        (0.72, 0.50, 0.0),  # PINKY_DIP
        (0.73, 0.42, 0.0),  # PINKY_TIP
    ]

    landmarks = []
    for i, (x, y, z) in enumerate(base_positions):
        landmarks.append(
            HandLandmark(
                point=Point3D(x=x, y=y, z=z),
                name=LANDMARK_NAMES[i],
                index=i,
            )
        )

    return HandLandmarks(landmarks=landmarks)


@pytest.fixture
def valid_frame(valid_landmarks: HandLandmarks) -> RecordingFrame:
    """Create a valid recording frame."""
    return RecordingFrame(
        timestamp_ms=0,
        right_hand=DetectedHand(
            landmarks=valid_landmarks,
            handedness="Right",
            confidence=0.95,
        ),
        left_hand=None,
        frame_confidence=0.95,
    )


@pytest.fixture
def valid_recording(valid_frame: RecordingFrame) -> Recording:
    """Create a valid recording."""
    now = datetime.now(timezone.utc)
    return Recording(
        frames=[valid_frame],
        sign_id="bsl_alphabet_a",
        start_time=now,
        end_time=now,
        metadata={"source": "test"},
    )


@pytest.fixture
def valid_sign_pose() -> SignPose:
    """Create a valid sign pose."""
    # Create 21 points with reasonable spread
    points = [Point3D(x=0.3 + (i % 5) * 0.1, y=0.2 + (i // 5) * 0.15, z=0.0) for i in range(NUM_LANDMARKS)]
    return SignPose(
        right_hand_landmarks=points,
        left_hand_landmarks=None,
        tolerances={"default": 0.1},
    )


@pytest.fixture
def valid_sign_entry(valid_sign_pose: SignPose) -> SignDictionaryEntry:
    """Create a valid sign dictionary entry."""
    now = datetime.now(timezone.utc)
    return SignDictionaryEntry(
        definition=SignDefinition(
            id="bsl_alphabet_a",
            name="A",
            category="alphabet",
            difficulty=1,
            two_handed=True,
        ),
        poses=[valid_sign_pose],
        sample_count=100,
        quality_score=0.9,
        created_at=now,
        updated_at=now,
        source="kaggle",  # Must be one of: kaggle, recorded, merged
    )


# =============================================================================
# Point Validation Tests
# =============================================================================


class TestValidatePoint:
    """Tests for validate_point function."""

    def test_valid_point(self, valid_point: Point3D):
        """Test validation of a valid point."""
        result = validate_point(valid_point)
        assert result.is_valid
        assert result.score == 1.0
        assert len(result.errors) == 0

    def test_point_at_origin(self):
        """Test point at origin (0, 0, 0)."""
        point = Point3D(x=0.0, y=0.0, z=0.0)
        result = validate_point(point)
        assert result.is_valid

    def test_point_at_max(self):
        """Test point at maximum values (1, 1)."""
        point = Point3D(x=1.0, y=1.0, z=0.0)
        result = validate_point(point)
        assert result.is_valid

    def test_point_out_of_range_rejected_by_pydantic(self):
        """Test that Pydantic rejects out-of-range coordinates."""
        with pytest.raises(ValidationError):
            Point3D(x=1.5, y=0.5, z=0.0)

    def test_point_negative_rejected_by_pydantic(self):
        """Test that Pydantic rejects negative coordinates."""
        with pytest.raises(ValidationError):
            Point3D(x=-0.5, y=0.5, z=0.0)

    def test_point_z_can_be_negative(self):
        """Test that z coordinate can be negative (depth)."""
        point = Point3D(x=0.5, y=0.5, z=-0.5)
        result = validate_point(point)
        assert result.is_valid  # z is not bounded


# =============================================================================
# Landmark Validation Tests
# =============================================================================


class TestValidateLandmarks:
    """Tests for validate_landmarks function."""

    def test_valid_landmarks(self, valid_landmarks: HandLandmarks):
        """Test validation of valid landmarks."""
        result = validate_landmarks(valid_landmarks)
        assert result.is_valid
        assert result.score > 0.5

    def test_collapsed_landmarks(self):
        """Test landmarks all at same position."""
        landmarks = []
        for i in range(NUM_LANDMARKS):
            landmarks.append(
                HandLandmark(
                    point=Point3D(x=0.5, y=0.5, z=0.0),
                    name=LANDMARK_NAMES[i],
                    index=i,
                )
            )
        hand = HandLandmarks(landmarks=landmarks)
        result = validate_landmarks(hand)
        # Should have error about collapsed landmarks
        assert not result.is_valid or result.score < 0.5


class TestCheckFingerLengths:
    """Tests for check_finger_lengths function."""

    def test_valid_finger_lengths(self, valid_landmarks: HandLandmarks):
        """Test with reasonable finger proportions."""
        result = check_finger_lengths(valid_landmarks)
        assert result is True

    def test_collapsed_hand(self):
        """Test hand with all landmarks at same point."""
        landmarks = []
        for i in range(NUM_LANDMARKS):
            landmarks.append(
                HandLandmark(
                    point=Point3D(x=0.5, y=0.5, z=0.0),
                    name=LANDMARK_NAMES[i],
                    index=i,
                )
            )
        hand = HandLandmarks(landmarks=landmarks)
        result = check_finger_lengths(hand)
        assert result is False


class TestCheckJointAngles:
    """Tests for check_joint_angles function."""

    def test_valid_joint_angles(self, valid_landmarks: HandLandmarks):
        """Test with reasonable joint angles."""
        result = check_joint_angles(valid_landmarks)
        assert result is True


# =============================================================================
# Frame Validation Tests
# =============================================================================


class TestValidateFrame:
    """Tests for validate_frame function."""

    def test_valid_frame(self, valid_frame: RecordingFrame):
        """Test validation of a valid frame."""
        result = validate_frame(valid_frame)
        assert result.is_valid
        assert result.score > 0.5

    def test_frame_no_hands(self):
        """Test frame with no hands detected."""
        frame = RecordingFrame(
            timestamp_ms=0,
            left_hand=None,
            right_hand=None,
            frame_confidence=0.5,
        )
        result = validate_frame(frame)
        assert result.is_valid  # No hands is warning, not error
        assert len(result.warnings) > 0

    def test_frame_negative_timestamp_rejected(self, valid_landmarks: HandLandmarks):
        """Test that Pydantic rejects negative timestamps."""
        with pytest.raises(ValidationError):
            RecordingFrame(
                timestamp_ms=-100,
                right_hand=DetectedHand(
                    landmarks=valid_landmarks,
                    handedness="Right",
                    confidence=0.95,
                ),
                frame_confidence=0.95,
            )

    def test_frame_low_confidence(self, valid_landmarks: HandLandmarks):
        """Test frame with low confidence."""
        frame = RecordingFrame(
            timestamp_ms=0,
            right_hand=DetectedHand(
                landmarks=valid_landmarks,
                handedness="Right",
                confidence=0.3,
            ),
            frame_confidence=0.3,
        )
        result = validate_frame(frame)
        assert result.is_valid  # Low confidence is warning
        assert len(result.warnings) > 0


# =============================================================================
# Recording Validation Tests
# =============================================================================


class TestValidateRecording:
    """Tests for validate_recording function."""

    def test_valid_recording(self, valid_recording: Recording):
        """Test validation of a valid recording."""
        result = validate_recording(valid_recording)
        assert result.is_valid
        assert result.score > 0.5

    def test_recording_no_sign_id(self, valid_frame: RecordingFrame):
        """Test recording with no sign_id."""
        now = datetime.now(timezone.utc)
        recording = Recording(
            frames=[valid_frame],
            sign_id="",
            start_time=now,
            end_time=now,
        )
        result = validate_recording(recording)
        assert not result.is_valid
        assert any("sign_id" in e.lower() for e in result.errors)

    def test_recording_unordered_timestamps(self, valid_landmarks: HandLandmarks):
        """Test recording with out-of-order timestamps."""
        now = datetime.now(timezone.utc)
        frames = [
            RecordingFrame(
                timestamp_ms=100,
                right_hand=DetectedHand(
                    landmarks=valid_landmarks,
                    handedness="Right",
                    confidence=0.95,
                ),
            ),
            RecordingFrame(
                timestamp_ms=50,  # Out of order
                right_hand=DetectedHand(
                    landmarks=valid_landmarks,
                    handedness="Right",
                    confidence=0.95,
                ),
            ),
        ]
        recording = Recording(
            frames=frames,
            sign_id="test",
            start_time=now,
            end_time=now,
        )
        result = validate_recording(recording)
        assert not result.is_valid


# =============================================================================
# Sign Validation Tests
# =============================================================================


class TestValidateSignPose:
    """Tests for validate_sign_pose function."""

    def test_valid_pose(self, valid_sign_pose: SignPose):
        """Test validation of a valid pose."""
        result = validate_sign_pose(valid_sign_pose)
        assert result.is_valid
        assert result.score > 0.5

    def test_pose_no_landmarks(self):
        """Test pose with no landmark data."""
        pose = SignPose(
            right_hand_landmarks=None,
            left_hand_landmarks=None,
        )
        result = validate_sign_pose(pose)
        assert not result.is_valid

    def test_pose_wrong_landmark_count_rejected(self):
        """Test that Pydantic rejects wrong landmark count."""
        points = [Point3D(x=0.5, y=0.5, z=0.0) for _ in range(10)]  # Only 10
        with pytest.raises(ValidationError):
            SignPose(right_hand_landmarks=points)


class TestValidateSignEntry:
    """Tests for validate_sign_entry function."""

    def test_valid_entry(self, valid_sign_entry: SignDictionaryEntry):
        """Test validation of a valid entry."""
        result = validate_sign_entry(valid_sign_entry)
        assert result.is_valid
        assert result.score > 0.5

    def test_entry_no_poses_rejected(self):
        """Test that Pydantic rejects entry with no poses."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError):
            SignDictionaryEntry(
                definition=SignDefinition(
                    id="test",
                    name="Test",
                    category="alphabet",
                ),
                poses=[],  # Empty poses should fail
                sample_count=1,
                quality_score=0.5,
                created_at=now,
                updated_at=now,
                source="kaggle",
            )

    def test_entry_invalid_source_rejected(self, valid_sign_pose: SignPose):
        """Test that Pydantic rejects invalid source values."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError):
            SignDictionaryEntry(
                definition=SignDefinition(
                    id="test",
                    name="Test",
                    category="alphabet",
                ),
                poses=[valid_sign_pose],
                sample_count=1,
                quality_score=0.5,
                created_at=now,
                updated_at=now,
                source="invalid_source",  # Not one of: kaggle, recorded, merged
            )


# =============================================================================
# Quality Scoring Tests
# =============================================================================


class TestCalculateLandmarkQuality:
    """Tests for calculate_landmark_quality function."""

    def test_quality_valid_landmarks(self, valid_landmarks: HandLandmarks):
        """Test quality score for valid landmarks."""
        score = calculate_landmark_quality(valid_landmarks)
        assert 0 <= score <= 100
        assert score > 50  # Should be reasonably high

    def test_quality_collapsed_landmarks(self):
        """Test quality score for collapsed landmarks."""
        landmarks = []
        for i in range(NUM_LANDMARKS):
            landmarks.append(
                HandLandmark(
                    point=Point3D(x=0.5, y=0.5, z=0.0),
                    name=LANDMARK_NAMES[i],
                    index=i,
                )
            )
        hand = HandLandmarks(landmarks=landmarks)
        score = calculate_landmark_quality(hand)
        assert score < 80  # Should be penalized


class TestCalculateRecordingQuality:
    """Tests for calculate_recording_quality function."""

    def test_quality_valid_recording(self, valid_recording: Recording):
        """Test quality score for valid recording."""
        score = calculate_recording_quality(valid_recording)
        assert 0 <= score <= 100
        assert score > 50


class TestCalculateSignQuality:
    """Tests for calculate_sign_quality function."""

    def test_quality_valid_entry(self, valid_sign_entry: SignDictionaryEntry):
        """Test quality score for valid entry."""
        score = calculate_sign_quality(valid_sign_entry)
        assert 0 <= score <= 100
        assert score > 50


# =============================================================================
# Batch Validation Tests
# =============================================================================


class TestValidationReport:
    """Tests for report generation."""

    def test_generate_report(self, valid_landmarks: HandLandmarks):
        """Test report generation."""
        results = {
            "test_1": validate_landmarks(valid_landmarks),
            "test_2": validate_landmarks(valid_landmarks),
        }
        report = generate_validation_report(results)
        assert "# Validation Report" in report
        assert "Summary" in report
        assert "Total Items" in report

    def test_summarize_results(self, valid_landmarks: HandLandmarks):
        """Test result summarization."""
        results = {
            "test_1": validate_landmarks(valid_landmarks),
            "test_2": validate_landmarks(valid_landmarks),
        }
        summary = summarize_validation_results(results)
        assert summary.total_items == 2
        assert summary.valid_items >= 0
        assert 0 <= summary.avg_score <= 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_point_boundary_values(self):
        """Test points at exact boundaries."""
        # Exactly at 0
        result = validate_point(Point3D(x=0.0, y=0.0, z=0.0))
        assert result.is_valid

        # Exactly at 1
        result = validate_point(Point3D(x=1.0, y=1.0, z=0.0))
        assert result.is_valid

    def test_empty_recording_rejected(self):
        """Test that Pydantic rejects empty recording."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError):
            Recording(
                frames=[],
                sign_id="test",
                start_time=now,
                end_time=now,
            )

    def test_anatomical_limits_customization(self, valid_landmarks: HandLandmarks):
        """Test with custom anatomical limits."""
        from src.validation import AnatomicalLimits

        strict_limits = AnatomicalLimits(
            coord_min=0.1,
            coord_max=0.9,
        )
        result = validate_landmarks(valid_landmarks, limits=strict_limits)
        # May have more warnings with stricter limits
        assert isinstance(result.is_valid, bool)

    def test_multiple_frames_recording(self, valid_landmarks: HandLandmarks):
        """Test recording with multiple frames."""
        now = datetime.now(timezone.utc)
        frames = [
            RecordingFrame(
                timestamp_ms=i * 100,
                right_hand=DetectedHand(
                    landmarks=valid_landmarks,
                    handedness="Right",
                    confidence=0.95,
                ),
            )
            for i in range(5)
        ]
        recording = Recording(
            frames=frames,
            sign_id="test",
            start_time=now,
            end_time=now,
        )
        result = validate_recording(recording)
        assert result.is_valid
        assert result.score > 0.5

    def test_both_hands_detected(self, valid_landmarks: HandLandmarks):
        """Test frame with both hands detected."""
        frame = RecordingFrame(
            timestamp_ms=0,
            right_hand=DetectedHand(
                landmarks=valid_landmarks,
                handedness="Right",
                confidence=0.95,
            ),
            left_hand=DetectedHand(
                landmarks=valid_landmarks,
                handedness="Left",
                confidence=0.90,
            ),
            frame_confidence=0.92,
        )
        result = validate_frame(frame)
        assert result.is_valid
