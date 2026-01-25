"""
Tests for landmark normalization module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from src.normalize import (
    calculate_bounding_box,
    calculate_bounding_box_size,
    calculate_centroid,
    calculate_palm_width,
    calculate_palm_center,
    calculate_palm_normal,
    calculate_finger_direction,
    normalize_translation,
    normalize_scale_bbox,
    normalize_scale_palm,
    normalize_rotation,
    normalize_all,
    denormalize,
    normalize_sequence,
    NormalizationResult,
)
from src.math_utils import (
    normalize_vector,
    magnitude,
    dot,
    cross,
    angle_between_vectors,
    rotation_matrix_from_vectors,
    rotation_matrix_from_axis_angle,
    apply_rotation,
    invert_transformation,
    orthonormalize,
)
from src.landmarks import WRIST, INDEX_FINGER_MCP, PINKY_MCP, MIDDLE_FINGER_MCP


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_landmarks() -> np.ndarray:
    """Create realistic hand landmarks for testing."""
    # Create a right hand in a neutral pose
    # Wrist at origin, fingers pointing up (+Y), palm facing camera (+Z)
    landmarks = np.zeros((21, 3))

    # Wrist
    landmarks[0] = [0.0, 0.0, 0.0]

    # Thumb (angled to the side)
    landmarks[1] = [0.05, 0.02, 0.01]   # CMC
    landmarks[2] = [0.08, 0.04, 0.02]   # MCP
    landmarks[3] = [0.10, 0.06, 0.02]   # IP
    landmarks[4] = [0.12, 0.08, 0.02]   # TIP

    # Index finger
    landmarks[5] = [0.04, 0.10, 0.0]    # MCP
    landmarks[6] = [0.04, 0.14, 0.0]    # PIP
    landmarks[7] = [0.04, 0.17, 0.0]    # DIP
    landmarks[8] = [0.04, 0.20, 0.0]    # TIP

    # Middle finger
    landmarks[9] = [0.0, 0.11, 0.0]     # MCP
    landmarks[10] = [0.0, 0.16, 0.0]    # PIP
    landmarks[11] = [0.0, 0.19, 0.0]    # DIP
    landmarks[12] = [0.0, 0.22, 0.0]    # TIP

    # Ring finger
    landmarks[13] = [-0.03, 0.10, 0.0]  # MCP
    landmarks[14] = [-0.03, 0.14, 0.0]  # PIP
    landmarks[15] = [-0.03, 0.17, 0.0]  # DIP
    landmarks[16] = [-0.03, 0.20, 0.0]  # TIP

    # Pinky
    landmarks[17] = [-0.06, 0.08, 0.0]  # MCP
    landmarks[18] = [-0.06, 0.11, 0.0]  # PIP
    landmarks[19] = [-0.06, 0.13, 0.0]  # DIP
    landmarks[20] = [-0.06, 0.15, 0.0]  # TIP

    return landmarks


@pytest.fixture
def translated_landmarks(sample_landmarks) -> np.ndarray:
    """Landmarks translated away from origin."""
    return sample_landmarks + np.array([10.0, 20.0, 5.0])


@pytest.fixture
def scaled_landmarks(sample_landmarks) -> np.ndarray:
    """Landmarks scaled by factor of 3."""
    return sample_landmarks * 3.0


@pytest.fixture
def rotated_landmarks(sample_landmarks) -> np.ndarray:
    """Landmarks rotated 45 degrees around Z axis."""
    angle = np.pi / 4
    R = rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), angle)
    return apply_rotation(sample_landmarks, R)


# =============================================================================
# Math Utils Tests
# =============================================================================


class TestVectorOperations:
    """Tests for basic vector operations."""

    def test_normalize_vector(self):
        v = np.array([3.0, 4.0, 0.0])
        result = normalize_vector(v)
        assert_allclose(magnitude(result), 1.0)
        assert_allclose(result, [0.6, 0.8, 0.0])

    def test_normalize_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        result = normalize_vector(v)
        assert_allclose(result, [0.0, 0.0, 0.0])

    def test_magnitude(self):
        v = np.array([3.0, 4.0, 0.0])
        assert magnitude(v) == 5.0

    def test_dot_product(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        assert dot(v1, v2) == 32.0

    def test_cross_product(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        result = cross(v1, v2)
        assert_allclose(result, [0.0, 0.0, 1.0])

    def test_angle_between_parallel(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])
        assert_allclose(angle_between_vectors(v1, v2), 0.0)

    def test_angle_between_perpendicular(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert_allclose(angle_between_vectors(v1, v2), np.pi / 2)

    def test_angle_between_opposite(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert_allclose(angle_between_vectors(v1, v2), np.pi)


class TestRotationMatrices:
    """Tests for rotation matrix creation and application."""

    def test_rotation_from_vectors_identity(self):
        v = np.array([1.0, 0.0, 0.0])
        R = rotation_matrix_from_vectors(v, v)
        assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_from_vectors_90_degrees(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        R = rotation_matrix_from_vectors(v1, v2)

        # Apply rotation to v1 should give v2
        result = R @ v1
        assert_allclose(result, v2, atol=1e-9)

    def test_rotation_from_vectors_opposite(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        R = rotation_matrix_from_vectors(v1, v2)

        result = R @ v1
        assert_allclose(result, v2, atol=1e-10)

    def test_rotation_from_axis_angle(self):
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 2  # 90 degrees

        R = rotation_matrix_from_axis_angle(axis, angle)

        # Rotate X axis should give Y axis
        v = np.array([1.0, 0.0, 0.0])
        result = R @ v
        assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_rotation_matrix_is_orthogonal(self):
        axis = np.array([1.0, 1.0, 1.0])
        angle = 0.7
        R = rotation_matrix_from_axis_angle(axis, angle)

        # R^T @ R should be identity
        assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        # Determinant should be 1
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_apply_rotation(self):
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        R = rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)

        result = apply_rotation(points, R)
        expected = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        assert_allclose(result, expected, atol=1e-10)


class TestOrthonormalize:
    """Tests for Gram-Schmidt orthonormalization."""

    def test_orthonormalize_perpendicular_input(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        x, y, z = orthonormalize(v1, v2)

        assert_allclose(x, [1.0, 0.0, 0.0])
        assert_allclose(y, [0.0, 1.0, 0.0])
        assert_allclose(z, [0.0, 0.0, 1.0])

    def test_orthonormalize_non_perpendicular(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 0.0])

        x, y, z = orthonormalize(v1, v2)

        # All should be unit vectors
        assert_allclose(magnitude(x), 1.0)
        assert_allclose(magnitude(y), 1.0)
        assert_allclose(magnitude(z), 1.0)

        # All should be perpendicular
        assert_allclose(dot(x, y), 0.0, atol=1e-10)
        assert_allclose(dot(y, z), 0.0, atol=1e-10)
        assert_allclose(dot(x, z), 0.0, atol=1e-10)


# =============================================================================
# Geometry Calculation Tests
# =============================================================================


class TestBoundingBox:
    """Tests for bounding box calculations."""

    def test_calculate_bounding_box(self, sample_landmarks):
        min_c, max_c, dims = calculate_bounding_box(sample_landmarks)

        assert min_c[0] < max_c[0]
        assert min_c[1] < max_c[1]
        assert_allclose(dims, max_c - min_c)

    def test_bounding_box_size(self, sample_landmarks):
        size = calculate_bounding_box_size(sample_landmarks)
        assert size > 0

        _, _, dims = calculate_bounding_box(sample_landmarks)
        expected = magnitude(dims)
        assert_allclose(size, expected)


class TestPalmGeometry:
    """Tests for palm geometry calculations."""

    def test_calculate_centroid(self, sample_landmarks):
        centroid = calculate_centroid(sample_landmarks)
        assert centroid.shape == (3,)

        # Centroid should be within bounding box
        min_c, max_c, _ = calculate_bounding_box(sample_landmarks)
        assert np.all(centroid >= min_c)
        assert np.all(centroid <= max_c)

    def test_calculate_palm_width(self, sample_landmarks):
        width = calculate_palm_width(sample_landmarks)
        assert width > 0

        # Check against expected value from fixture
        expected = magnitude(
            sample_landmarks[INDEX_FINGER_MCP] - sample_landmarks[PINKY_MCP]
        )
        assert_allclose(width, expected)

    def test_calculate_palm_center(self, sample_landmarks):
        center = calculate_palm_center(sample_landmarks)
        assert center.shape == (3,)

    def test_calculate_palm_normal(self, sample_landmarks):
        normal = calculate_palm_normal(sample_landmarks)

        # Should be unit vector
        assert_allclose(magnitude(normal), 1.0)

        # For our test fixture, normal should point in +Z direction
        assert normal[2] > 0.9

    def test_calculate_finger_direction(self, sample_landmarks):
        direction = calculate_finger_direction(sample_landmarks)

        # Should be unit vector
        assert_allclose(magnitude(direction), 1.0)

        # For our test fixture, fingers point in +Y direction
        assert direction[1] > 0.9


# =============================================================================
# Translation Normalization Tests
# =============================================================================


class TestTranslationNormalization:
    """Tests for translation normalization."""

    def test_normalize_translation_wrist(self, sample_landmarks):
        result, anchor = normalize_translation(sample_landmarks, "wrist")

        # Wrist should be at origin
        assert_allclose(result[WRIST], [0.0, 0.0, 0.0])

    def test_normalize_translation_centroid(self, sample_landmarks):
        result, anchor = normalize_translation(sample_landmarks, "centroid")

        # Centroid should be at origin
        centroid = calculate_centroid(result)
        assert_allclose(centroid, [0.0, 0.0, 0.0], atol=1e-10)

    def test_normalize_translation_palm_center(self, sample_landmarks):
        result, anchor = normalize_translation(sample_landmarks, "palm_center")

        # Palm center should be at origin
        palm_center = calculate_palm_center(result)
        assert_allclose(palm_center, [0.0, 0.0, 0.0], atol=1e-10)

    def test_translation_preserves_relative_positions(self, sample_landmarks):
        result, _ = normalize_translation(sample_landmarks, "wrist")

        # Relative distances should be preserved
        orig_dist = magnitude(sample_landmarks[8] - sample_landmarks[5])
        new_dist = magnitude(result[8] - result[5])
        assert_allclose(orig_dist, new_dist)

    def test_translation_returns_original_anchor(self, translated_landmarks):
        _, anchor = normalize_translation(translated_landmarks, "wrist")
        assert_allclose(anchor, translated_landmarks[WRIST])


# =============================================================================
# Scale Normalization Tests
# =============================================================================


class TestScaleNormalization:
    """Tests for scale normalization."""

    def test_normalize_scale_bbox(self, sample_landmarks):
        result, original_size = normalize_scale_bbox(sample_landmarks, target_size=1.0)

        new_size = calculate_bounding_box_size(result)
        assert_allclose(new_size, 1.0, atol=1e-10)

    def test_normalize_scale_bbox_custom_target(self, sample_landmarks):
        result, _ = normalize_scale_bbox(sample_landmarks, target_size=2.0)

        new_size = calculate_bounding_box_size(result)
        assert_allclose(new_size, 2.0, atol=1e-10)

    def test_normalize_scale_palm(self, sample_landmarks):
        result, original_width = normalize_scale_palm(sample_landmarks, target_width=1.0)

        new_width = calculate_palm_width(result)
        assert_allclose(new_width, 1.0, atol=1e-10)

    def test_scale_preserves_shape(self, sample_landmarks):
        result, original_size = normalize_scale_bbox(sample_landmarks, target_size=1.0)

        # Angles between landmarks should be preserved
        orig_dir1 = normalize_vector(sample_landmarks[8] - sample_landmarks[5])
        orig_dir2 = normalize_vector(sample_landmarks[12] - sample_landmarks[9])
        orig_angle = angle_between_vectors(orig_dir1, orig_dir2)

        new_dir1 = normalize_vector(result[8] - result[5])
        new_dir2 = normalize_vector(result[12] - result[9])
        new_angle = angle_between_vectors(new_dir1, new_dir2)

        assert_allclose(orig_angle, new_angle, atol=1e-10)


# =============================================================================
# Rotation Normalization Tests
# =============================================================================


class TestRotationNormalization:
    """Tests for rotation normalization."""

    def test_normalize_rotation_aligns_palm(self, sample_landmarks):
        result, R = normalize_rotation(sample_landmarks, align_palm=True, align_fingers=False)

        # Palm normal should point in +Z direction
        normal = calculate_palm_normal(result)
        assert normal[2] > 0.9

    def test_normalize_rotation_aligns_fingers(self, sample_landmarks):
        result, R = normalize_rotation(sample_landmarks, align_palm=True, align_fingers=True)

        # Finger direction projected to XY should align with +Y
        finger_dir = calculate_finger_direction(result)
        finger_xy = normalize_vector(np.array([finger_dir[0], finger_dir[1], 0.0]))

        # Y component should be dominant
        assert finger_xy[1] > 0.9

    def test_rotation_returns_valid_matrix(self, sample_landmarks):
        _, R = normalize_rotation(sample_landmarks)

        # Should be orthogonal
        assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

        # Determinant should be 1 (proper rotation)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rotation_preserves_distances(self, sample_landmarks):
        result, _ = normalize_rotation(sample_landmarks)

        # All pairwise distances should be preserved
        for i in range(21):
            for j in range(i + 1, 21):
                orig_dist = magnitude(sample_landmarks[i] - sample_landmarks[j])
                new_dist = magnitude(result[i] - result[j])
                assert_allclose(orig_dist, new_dist, atol=1e-10)


# =============================================================================
# Full Normalization Pipeline Tests
# =============================================================================


class TestNormalizeAll:
    """Tests for combined normalization."""

    def test_normalize_all_default(self, sample_landmarks):
        result = normalize_all(sample_landmarks)

        assert isinstance(result, NormalizationResult)
        assert result.landmarks.shape == (21, 3)
        assert result.rotation_matrix.shape == (3, 3)

    def test_normalize_all_wrist_at_origin(self, translated_landmarks):
        result = normalize_all(translated_landmarks, translation_anchor="wrist")

        # Wrist should be at origin
        assert_allclose(result.landmarks[WRIST], [0.0, 0.0, 0.0], atol=1e-10)

    def test_normalize_all_palm_width_normalized(self, scaled_landmarks):
        result = normalize_all(scaled_landmarks, scale_method="palm", target_scale=1.0)

        new_width = calculate_palm_width(result.landmarks)
        assert_allclose(new_width, 1.0, atol=1e-10)

    def test_normalize_all_bbox_normalized(self, scaled_landmarks):
        result = normalize_all(scaled_landmarks, scale_method="bbox", target_scale=1.0)

        new_size = calculate_bounding_box_size(result.landmarks)
        assert_allclose(new_size, 1.0, atol=1e-10)

    def test_normalize_all_method_string(self, sample_landmarks):
        result = normalize_all(
            sample_landmarks,
            translation_anchor="centroid",
            scale_method="bbox",
            align_rotation=False
        )

        assert "centroid" in result.method
        assert "bbox" in result.method
        assert "False" in result.method


# =============================================================================
# Denormalization Tests
# =============================================================================


class TestDenormalize:
    """Tests for inverse normalization."""

    def test_denormalize_recovers_original(self, sample_landmarks):
        result = normalize_all(sample_landmarks)
        recovered = denormalize(result.landmarks, result)

        assert_allclose(recovered, sample_landmarks, atol=1e-10)

    def test_denormalize_translated(self, translated_landmarks):
        result = normalize_all(translated_landmarks)
        recovered = denormalize(result.landmarks, result)

        assert_allclose(recovered, translated_landmarks, atol=1e-10)

    def test_denormalize_scaled(self, scaled_landmarks):
        result = normalize_all(scaled_landmarks)
        recovered = denormalize(result.landmarks, result)

        assert_allclose(recovered, scaled_landmarks, atol=1e-10)

    def test_denormalize_rotated(self, rotated_landmarks):
        result = normalize_all(rotated_landmarks)
        recovered = denormalize(result.landmarks, result)

        assert_allclose(recovered, rotated_landmarks, atol=1e-9)

    def test_denormalize_complex_transform(self):
        """Test denormalization with translation, scale, and rotation combined."""
        base = np.random.randn(21, 3)

        # Apply complex transform
        transformed = base * 2.5
        R = rotation_matrix_from_axis_angle(np.array([1.0, 1.0, 0.0]), 0.8)
        transformed = apply_rotation(transformed, R)
        transformed = transformed + np.array([100, -50, 25])

        result = normalize_all(transformed)
        recovered = denormalize(result.landmarks, result)

        assert_allclose(recovered, transformed, atol=1e-8)


# =============================================================================
# Sequence Normalization Tests
# =============================================================================


class TestSequenceNormalization:
    """Tests for batch/sequence normalization."""

    def test_normalize_sequence_single_frame(self, sample_landmarks):
        frames = [sample_landmarks]
        results = normalize_sequence(frames)

        assert len(results) == 1
        assert results[0].landmarks.shape == (21, 3)

    def test_normalize_sequence_consistent_params(self, sample_landmarks):
        """When using first frame params, all frames should use same transform."""
        # Create sequence with slight variations
        frames = [
            sample_landmarks,
            sample_landmarks + np.array([0.01, 0.02, 0.0]),
            sample_landmarks + np.array([0.02, 0.01, 0.0]),
        ]

        results = normalize_sequence(frames, use_first_frame_params=True)

        # All should have same scale factor
        assert_allclose(results[0].scale_factor, results[1].scale_factor)
        assert_allclose(results[0].scale_factor, results[2].scale_factor)

        # All should have same rotation
        assert_allclose(results[0].rotation_matrix, results[1].rotation_matrix)

    def test_normalize_sequence_independent(self, sample_landmarks):
        """When normalizing independently, each frame has own params."""
        frames = [
            sample_landmarks,
            sample_landmarks * 2.0,  # Different scale
        ]

        results = normalize_sequence(frames, use_first_frame_params=False)

        # Different original scales
        assert not np.allclose(results[0].scale_factor, results[1].scale_factor)

    def test_normalize_sequence_preserves_motion(self, sample_landmarks):
        """Relative motion should be preserved when using first frame params."""
        motion = np.array([0.1, 0.0, 0.0])
        frames = [
            sample_landmarks,
            sample_landmarks + motion,
        ]

        results = normalize_sequence(frames, use_first_frame_params=True)

        # Calculate motion in normalized space
        normalized_motion = results[1].landmarks[WRIST] - results[0].landmarks[WRIST]

        # Motion direction should be preserved (though magnitude may change due to scale)
        orig_dir = normalize_vector(motion)
        norm_dir = normalize_vector(normalized_motion)

        # Account for rotation by applying same rotation to original motion
        rotated_orig_dir = apply_rotation(orig_dir.reshape(1, -1), results[0].rotation_matrix)[0]

        assert_allclose(norm_dir, rotated_orig_dir, atol=1e-10)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_landmarks(self):
        """Handle all-zero landmarks gracefully."""
        zeros = np.zeros((21, 3))

        result = normalize_all(zeros)
        # Should not crash, landmarks should still be zeros
        assert_allclose(result.landmarks, zeros)

    def test_flat_landmarks(self):
        """Handle 2D landmarks (Z=0 for all)."""
        flat = np.random.randn(21, 3)
        flat[:, 2] = 0.0

        result = normalize_all(flat)
        assert result.landmarks.shape == (21, 3)

    def test_collinear_landmarks(self):
        """Handle landmarks all on a line."""
        line = np.zeros((21, 3))
        for i in range(21):
            line[i] = [i * 0.01, 0.0, 0.0]

        # Should not crash
        result = normalize_all(line)
        assert result.landmarks.shape == (21, 3)

    def test_single_point(self):
        """Handle all landmarks at same position."""
        single = np.ones((21, 3)) * 5.0

        result = normalize_all(single)
        # All points should be at origin after translation
        assert_allclose(result.landmarks, np.zeros((21, 3)), atol=1e-10)

    def test_very_small_landmarks(self):
        """Handle very small coordinate values."""
        tiny = np.random.randn(21, 3) * 1e-8

        result = normalize_all(tiny)
        assert not np.any(np.isnan(result.landmarks))
        assert not np.any(np.isinf(result.landmarks))

    def test_very_large_landmarks(self):
        """Handle very large coordinate values."""
        huge = np.random.randn(21, 3) * 1e8

        result = normalize_all(huge)
        assert not np.any(np.isnan(result.landmarks))
        assert not np.any(np.isinf(result.landmarks))


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestNormalizationProperties:
    """Property-based tests for normalization invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_roundtrip_random(self, seed):
        """Normalization followed by denormalization recovers original."""
        np.random.seed(seed)
        landmarks = np.random.randn(21, 3)

        result = normalize_all(landmarks)
        recovered = denormalize(result.landmarks, result)

        assert_allclose(recovered, landmarks, atol=1e-8)

    @pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
    def test_scale_invariance(self, sample_landmarks, scale):
        """Different input scales should produce similar normalized results."""
        scaled = sample_landmarks * scale

        result1 = normalize_all(sample_landmarks)
        result2 = normalize_all(scaled)

        # Normalized landmarks should be similar
        assert_allclose(result1.landmarks, result2.landmarks, atol=1e-6)

    @pytest.mark.parametrize("offset", [[0, 0, 0], [10, 0, 0], [0, -5, 3]])
    def test_translation_invariance(self, sample_landmarks, offset):
        """Different input positions should produce similar normalized results."""
        translated = sample_landmarks + np.array(offset)

        result1 = normalize_all(sample_landmarks)
        result2 = normalize_all(translated)

        # Normalized landmarks should be identical
        assert_allclose(result1.landmarks, result2.landmarks, atol=1e-10)

    def test_rotation_determinism(self, sample_landmarks):
        """Same input should always produce same output."""
        result1 = normalize_all(sample_landmarks)
        result2 = normalize_all(sample_landmarks)

        assert_allclose(result1.landmarks, result2.landmarks)
        assert_allclose(result1.rotation_matrix, result2.rotation_matrix)
