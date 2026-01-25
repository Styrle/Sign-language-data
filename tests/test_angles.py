"""
Tests for joint angle calculation module.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.angles import (
    degrees_to_radians,
    radians_to_degrees,
    normalize_angle,
    calculate_angle_3points,
    calculate_finger_angles,
    calculate_wrist_rotation,
    calculate_all_joint_angles,
    validate_angles,
    clamp_angles,
    angles_to_threejs_rotations,
    threejs_rotations_to_json,
    interpolate_angle,
    interpolate_angles,
    angles_to_dict,
    dict_to_angles,
    ANGLE_LIMITS,
    THREEJS_BONE_MAPPING,
)
from src.types import JointAngles, FingerAngles
from src.landmarks import (
    WRIST, INDEX_FINGER_MCP, PINKY_MCP, MIDDLE_FINGER_MCP,
    THUMB_CMC, THUMB_TIP,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def flat_hand_landmarks() -> np.ndarray:
    """
    Create landmarks for a flat, open hand.
    All fingers extended, palm facing forward.
    """
    landmarks = np.zeros((21, 3))

    # Wrist at origin
    landmarks[0] = [0.0, 0.0, 0.0]

    # Thumb (angled to side)
    landmarks[1] = [0.05, 0.02, 0.01]   # CMC
    landmarks[2] = [0.08, 0.05, 0.02]   # MCP
    landmarks[3] = [0.10, 0.08, 0.02]   # IP
    landmarks[4] = [0.12, 0.11, 0.02]   # TIP

    # Index finger (straight up)
    landmarks[5] = [0.04, 0.10, 0.0]    # MCP
    landmarks[6] = [0.04, 0.15, 0.0]    # PIP
    landmarks[7] = [0.04, 0.19, 0.0]    # DIP
    landmarks[8] = [0.04, 0.23, 0.0]    # TIP

    # Middle finger (straight up)
    landmarks[9] = [0.0, 0.11, 0.0]     # MCP
    landmarks[10] = [0.0, 0.17, 0.0]    # PIP
    landmarks[11] = [0.0, 0.21, 0.0]    # DIP
    landmarks[12] = [0.0, 0.25, 0.0]    # TIP

    # Ring finger (straight up)
    landmarks[13] = [-0.03, 0.10, 0.0]  # MCP
    landmarks[14] = [-0.03, 0.15, 0.0]  # PIP
    landmarks[15] = [-0.03, 0.19, 0.0]  # DIP
    landmarks[16] = [-0.03, 0.23, 0.0]  # TIP

    # Pinky (straight up)
    landmarks[17] = [-0.06, 0.08, 0.0]  # MCP
    landmarks[18] = [-0.06, 0.12, 0.0]  # PIP
    landmarks[19] = [-0.06, 0.15, 0.0]  # DIP
    landmarks[20] = [-0.06, 0.18, 0.0]  # TIP

    return landmarks


@pytest.fixture
def fist_landmarks() -> np.ndarray:
    """
    Create landmarks for a closed fist.
    All fingers curled.
    """
    landmarks = np.zeros((21, 3))

    # Wrist
    landmarks[0] = [0.0, 0.0, 0.0]

    # Thumb (curled across palm)
    landmarks[1] = [0.04, 0.02, 0.01]
    landmarks[2] = [0.06, 0.04, 0.02]
    landmarks[3] = [0.05, 0.06, 0.03]
    landmarks[4] = [0.03, 0.07, 0.04]

    # Index (curled down)
    landmarks[5] = [0.04, 0.10, 0.0]
    landmarks[6] = [0.04, 0.11, 0.03]
    landmarks[7] = [0.04, 0.09, 0.05]
    landmarks[8] = [0.04, 0.06, 0.04]

    # Middle (curled down)
    landmarks[9] = [0.0, 0.11, 0.0]
    landmarks[10] = [0.0, 0.12, 0.04]
    landmarks[11] = [0.0, 0.10, 0.06]
    landmarks[12] = [0.0, 0.07, 0.05]

    # Ring (curled down)
    landmarks[13] = [-0.03, 0.10, 0.0]
    landmarks[14] = [-0.03, 0.11, 0.03]
    landmarks[15] = [-0.03, 0.09, 0.05]
    landmarks[16] = [-0.03, 0.06, 0.04]

    # Pinky (curled down)
    landmarks[17] = [-0.06, 0.08, 0.0]
    landmarks[18] = [-0.06, 0.09, 0.03]
    landmarks[19] = [-0.06, 0.07, 0.04]
    landmarks[20] = [-0.06, 0.05, 0.03]

    return landmarks


@pytest.fixture
def sample_angles() -> JointAngles:
    """Create sample JointAngles for testing."""
    return JointAngles(
        thumb=FingerAngles(mcp=30, pip=20, dip=10),
        index=FingerAngles(mcp=45, pip=60, dip=30),
        middle=FingerAngles(mcp=50, pip=65, dip=35),
        ring=FingerAngles(mcp=45, pip=55, dip=25),
        pinky=FingerAngles(mcp=40, pip=50, dip=20),
        wrist_rotation=15,
        wrist_flexion=-10,
    )


# =============================================================================
# Conversion Tests
# =============================================================================


class TestConversions:
    """Tests for degree/radian conversions."""

    def test_degrees_to_radians(self):
        assert_allclose(degrees_to_radians(0), 0)
        assert_allclose(degrees_to_radians(90), math.pi / 2)
        assert_allclose(degrees_to_radians(180), math.pi)
        assert_allclose(degrees_to_radians(360), 2 * math.pi)

    def test_radians_to_degrees(self):
        assert_allclose(radians_to_degrees(0), 0)
        assert_allclose(radians_to_degrees(math.pi / 2), 90)
        assert_allclose(radians_to_degrees(math.pi), 180)
        assert_allclose(radians_to_degrees(2 * math.pi), 360)

    def test_roundtrip(self):
        for deg in [-180, -90, 0, 45, 90, 180]:
            result = radians_to_degrees(degrees_to_radians(deg))
            assert_allclose(result, deg)


class TestNormalizeAngle:
    """Tests for angle normalization."""

    def test_already_normal(self):
        assert normalize_angle(0) == 0
        assert normalize_angle(90) == 90
        assert normalize_angle(-90) == -90
        assert normalize_angle(180) == 180
        assert normalize_angle(-180) == -180

    def test_positive_wrap(self):
        assert normalize_angle(270) == -90
        assert normalize_angle(360) == 0
        assert normalize_angle(450) == 90

    def test_negative_wrap(self):
        assert normalize_angle(-270) == 90
        assert normalize_angle(-360) == 0
        assert normalize_angle(-450) == -90


# =============================================================================
# Angle Calculation Tests
# =============================================================================


class TestCalculateAngle3Points:
    """Tests for 3-point angle calculation."""

    def test_straight_line(self):
        # Points in a straight line = 180 degrees
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        angle = calculate_angle_3points(p1, p2, p3)
        assert_allclose(angle, 180.0, atol=0.01)

    def test_right_angle(self):
        # 90 degree angle
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        angle = calculate_angle_3points(p1, p2, p3)
        assert_allclose(angle, 90.0, atol=0.01)

    def test_acute_angle(self):
        # 60 degree angle (equilateral triangle)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([0.5, math.sqrt(3)/2, 0.0])
        angle = calculate_angle_3points(p1, p2, p3)
        assert_allclose(angle, 60.0, atol=0.01)

    def test_obtuse_angle(self):
        # 120 degree angle
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.5, -math.sqrt(3)/2, 0.0])
        angle = calculate_angle_3points(p1, p2, p3)
        assert_allclose(angle, 120.0, atol=0.01)

    def test_3d_angle(self):
        # Right angle in 3D
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 1.0])
        angle = calculate_angle_3points(p1, p2, p3)
        assert_allclose(angle, 90.0, atol=0.01)


class TestCalculateFingerAngles:
    """Tests for finger angle calculation."""

    def test_flat_hand_low_angles(self, flat_hand_landmarks):
        # Flat hand should have low flexion angles
        angles = calculate_finger_angles(flat_hand_landmarks, "index")

        # All angles should be relatively small for extended fingers
        assert angles["mcp"] < 30
        assert angles["pip"] < 30
        assert angles["dip"] < 30

    def test_fist_high_angles(self, fist_landmarks):
        # Fist should have high flexion angles
        angles = calculate_finger_angles(fist_landmarks, "index")

        # Angles should be significant for curled fingers
        assert angles["pip"] > 30  # PIP should be bent

    def test_all_fingers(self, flat_hand_landmarks):
        # Test all fingers
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            angles = calculate_finger_angles(flat_hand_landmarks, finger)

            if finger == "thumb":
                assert "cmc" in angles
                assert "mcp" in angles
                assert "ip" in angles
            else:
                assert "mcp" in angles
                assert "pip" in angles
                assert "dip" in angles


class TestCalculateWristRotation:
    """Tests for wrist rotation calculation."""

    def test_flat_hand_neutral(self, flat_hand_landmarks):
        rotation = calculate_wrist_rotation(flat_hand_landmarks)

        assert "x" in rotation
        assert "y" in rotation
        assert "z" in rotation

        # All values should be angles (degrees)
        assert -180 <= rotation["x"] <= 180
        assert -180 <= rotation["y"] <= 180
        assert -180 <= rotation["z"] <= 180


class TestCalculateAllJointAngles:
    """Tests for complete joint angle calculation."""

    def test_returns_joint_angles(self, flat_hand_landmarks):
        angles = calculate_all_joint_angles(flat_hand_landmarks)

        assert isinstance(angles, JointAngles)
        assert hasattr(angles, "thumb")
        assert hasattr(angles, "index")
        assert hasattr(angles, "middle")
        assert hasattr(angles, "ring")
        assert hasattr(angles, "pinky")
        assert hasattr(angles, "wrist_rotation")
        assert hasattr(angles, "wrist_flexion")

    def test_all_angles_present(self, flat_hand_landmarks):
        angles = calculate_all_joint_angles(flat_hand_landmarks)

        # Check finger angles
        for finger in [angles.thumb, angles.index, angles.middle, angles.ring, angles.pinky]:
            assert isinstance(finger.mcp, float)
            assert isinstance(finger.pip, float)
            assert isinstance(finger.dip, float)

    def test_flat_vs_fist(self, flat_hand_landmarks, fist_landmarks):
        flat_angles = calculate_all_joint_angles(flat_hand_landmarks)
        fist_angles = calculate_all_joint_angles(fist_landmarks)

        # Fist should have higher finger flexion
        assert fist_angles.index.pip > flat_angles.index.pip


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateAngles:
    """Tests for angle validation."""

    def test_valid_angles(self, sample_angles):
        result = validate_angles(sample_angles)
        assert result.is_valid
        assert result.score > 0.8

    def test_invalid_angles(self):
        # Create angles outside limits
        invalid = JointAngles(
            thumb=FingerAngles(mcp=120, pip=100, dip=100),  # Way too high
            index=FingerAngles(mcp=150, pip=150, dip=100),
            middle=FingerAngles(mcp=50, pip=60, dip=30),
            ring=FingerAngles(mcp=45, pip=55, dip=25),
            pinky=FingerAngles(mcp=40, pip=50, dip=20),
            wrist_rotation=0,
            wrist_flexion=0,
        )

        result = validate_angles(invalid)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_warning_near_limits(self):
        # Create angles just past limits (should get warnings)
        near_limit = JointAngles(
            thumb=FingerAngles(mcp=92, pip=82, dip=92),  # Slightly over
            index=FingerAngles(mcp=45, pip=60, dip=30),
            middle=FingerAngles(mcp=50, pip=60, dip=30),
            ring=FingerAngles(mcp=45, pip=55, dip=25),
            pinky=FingerAngles(mcp=40, pip=50, dip=20),
            wrist_rotation=0,
            wrist_flexion=0,
        )

        result = validate_angles(near_limit)
        assert len(result.warnings) > 0 or len(result.errors) > 0


class TestClampAngles:
    """Tests for angle clamping."""

    def test_already_valid(self, sample_angles):
        clamped = clamp_angles(sample_angles)

        # Should be unchanged
        assert clamped.index.mcp == sample_angles.index.mcp
        assert clamped.index.pip == sample_angles.index.pip

    def test_clamps_high_values(self):
        high = JointAngles(
            thumb=FingerAngles(mcp=150, pip=150, dip=150),
            index=FingerAngles(mcp=150, pip=150, dip=150),
            middle=FingerAngles(mcp=150, pip=150, dip=150),
            ring=FingerAngles(mcp=150, pip=150, dip=150),
            pinky=FingerAngles(mcp=150, pip=150, dip=150),
            wrist_rotation=180,
            wrist_flexion=180,
        )

        clamped = clamp_angles(high)

        # Should be at max limits
        assert clamped.index.mcp <= ANGLE_LIMITS["index_mcp"][1]
        assert clamped.index.pip <= ANGLE_LIMITS["index_pip"][1]
        assert clamped.wrist_rotation <= ANGLE_LIMITS["wrist_rotation"][1]

    def test_clamps_low_values(self):
        low = JointAngles(
            thumb=FingerAngles(mcp=-50, pip=-50, dip=-50),
            index=FingerAngles(mcp=-50, pip=-50, dip=-50),
            middle=FingerAngles(mcp=-50, pip=-50, dip=-50),
            ring=FingerAngles(mcp=-50, pip=-50, dip=-50),
            pinky=FingerAngles(mcp=-50, pip=-50, dip=-50),
            wrist_rotation=-180,
            wrist_flexion=-180,
        )

        clamped = clamp_angles(low)

        # Should be at min limits
        assert clamped.index.mcp >= ANGLE_LIMITS["index_mcp"][0]
        assert clamped.wrist_flexion >= ANGLE_LIMITS["wrist_flexion"][0]


# =============================================================================
# Three.js Conversion Tests
# =============================================================================


class TestThreeJSConversion:
    """Tests for Three.js bone rotation conversion."""

    def test_bone_mapping_exists(self):
        # Verify all expected bones are mapped
        assert "thumb_cmc" in THREEJS_BONE_MAPPING
        assert "index_mcp" in THREEJS_BONE_MAPPING
        assert "pinky_dip" in THREEJS_BONE_MAPPING

    def test_conversion_returns_dict(self, sample_angles):
        rotations = angles_to_threejs_rotations(sample_angles)

        assert isinstance(rotations, dict)
        assert len(rotations) > 0

    def test_conversion_has_all_bones(self, sample_angles):
        rotations = angles_to_threejs_rotations(sample_angles, hand="Right")

        # Should have wrist + 5 fingers * 3 joints = 16 bones
        expected_bones = [
            "RightHand",
            "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
            "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
            "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
            "RightHandRing1", "RightHandRing2", "RightHandRing3",
            "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
        ]

        for bone in expected_bones:
            assert bone in rotations

    def test_left_hand_prefix(self, sample_angles):
        rotations = angles_to_threejs_rotations(sample_angles, hand="Left")

        assert "LeftHand" in rotations
        assert "LeftHandIndex1" in rotations

    def test_rotation_values_in_radians(self, sample_angles):
        rotations = angles_to_threejs_rotations(sample_angles)

        for bone_name, rot in rotations.items():
            # Radians should be roughly in range [-pi, pi] for reasonable poses
            assert -math.pi * 2 <= rot.x <= math.pi * 2
            assert -math.pi * 2 <= rot.y <= math.pi * 2
            assert -math.pi * 2 <= rot.z <= math.pi * 2

    def test_to_json(self, sample_angles):
        rotations = angles_to_threejs_rotations(sample_angles)
        json_data = threejs_rotations_to_json(rotations)

        assert isinstance(json_data, dict)

        for bone_name, rot_data in json_data.items():
            assert "x" in rot_data
            assert "y" in rot_data
            assert "z" in rot_data
            assert "order" in rot_data


# =============================================================================
# Interpolation Tests
# =============================================================================


class TestInterpolation:
    """Tests for angle interpolation."""

    def test_interpolate_angle_start(self):
        result = interpolate_angle(0, 90, 0)
        assert_allclose(result, 0)

    def test_interpolate_angle_end(self):
        result = interpolate_angle(0, 90, 1)
        assert_allclose(result, 90)

    def test_interpolate_angle_middle(self):
        result = interpolate_angle(0, 90, 0.5)
        assert_allclose(result, 45)

    def test_interpolate_angle_wraparound(self):
        # Should take short path from 170 to -170 (across 180)
        result = interpolate_angle(170, -170, 0.5)
        assert_allclose(abs(result), 180)

    def test_interpolate_angles_structure(self, sample_angles):
        end_angles = JointAngles(
            thumb=FingerAngles(mcp=60, pip=40, dip=20),
            index=FingerAngles(mcp=90, pip=100, dip=60),
            middle=FingerAngles(mcp=100, pip=110, dip=70),
            ring=FingerAngles(mcp=90, pip=100, dip=50),
            pinky=FingerAngles(mcp=80, pip=90, dip=40),
            wrist_rotation=30,
            wrist_flexion=-20,
        )

        result = interpolate_angles(sample_angles, end_angles, 0.5)

        assert isinstance(result, JointAngles)

        # Check middle values
        assert_allclose(result.index.mcp, (45 + 90) / 2)
        assert_allclose(result.wrist_rotation, (15 + 30) / 2)


# =============================================================================
# Dict Conversion Tests
# =============================================================================


class TestDictConversion:
    """Tests for dictionary conversion."""

    def test_to_dict(self, sample_angles):
        data = angles_to_dict(sample_angles)

        assert isinstance(data, dict)
        assert "index_mcp" in data
        assert "thumb_cmc" in data
        assert "wrist_rotation" in data

        assert data["index_mcp"] == sample_angles.index.mcp

    def test_from_dict(self):
        data = {
            "thumb_cmc": 30,
            "thumb_mcp": 20,
            "thumb_ip": 10,
            "index_mcp": 45,
            "index_pip": 60,
            "index_dip": 30,
            "middle_mcp": 50,
            "middle_pip": 65,
            "middle_dip": 35,
            "ring_mcp": 45,
            "ring_pip": 55,
            "ring_dip": 25,
            "pinky_mcp": 40,
            "pinky_pip": 50,
            "pinky_dip": 20,
            "wrist_rotation": 15,
            "wrist_flexion": -10,
        }

        angles = dict_to_angles(data)

        assert angles.index.mcp == 45
        assert angles.wrist_rotation == 15

    def test_roundtrip(self, sample_angles):
        data = angles_to_dict(sample_angles)
        recovered = dict_to_angles(data)

        assert recovered.index.mcp == sample_angles.index.mcp
        assert recovered.thumb.mcp == sample_angles.thumb.mcp
        assert recovered.wrist_rotation == sample_angles.wrist_rotation


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_angles(self):
        zero = JointAngles(
            thumb=FingerAngles(mcp=0, pip=0, dip=0),
            index=FingerAngles(mcp=0, pip=0, dip=0),
            middle=FingerAngles(mcp=0, pip=0, dip=0),
            ring=FingerAngles(mcp=0, pip=0, dip=0),
            pinky=FingerAngles(mcp=0, pip=0, dip=0),
            wrist_rotation=0,
            wrist_flexion=0,
        )

        result = validate_angles(zero)
        assert result.is_valid

        clamped = clamp_angles(zero)
        assert clamped.index.mcp == 0

        rotations = angles_to_threejs_rotations(zero)
        assert len(rotations) > 0

    def test_coincident_points(self):
        # All points at same location
        p1 = np.array([1.0, 1.0, 1.0])
        p2 = np.array([1.0, 1.0, 1.0])
        p3 = np.array([1.0, 1.0, 1.0])

        # Should not crash
        angle = calculate_angle_3points(p1, p2, p3)
        assert not np.isnan(angle)

    def test_very_small_landmarks(self):
        landmarks = np.random.randn(21, 3) * 1e-6

        # Should not crash
        angles = calculate_all_joint_angles(landmarks)
        assert isinstance(angles, JointAngles)
