"""
Landmark normalization module.

Provides functions to normalize hand landmarks for consistent comparison
across different hand sizes, positions, and orientations.

Normalization pipeline:
1. Translation: Center landmarks (wrist at origin or centroid at origin)
2. Scale: Normalize to consistent size (palm width = 1.0 or bounding box = 1.0)
3. Rotation: Align palm plane and finger direction to canonical orientation
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .landmarks import WRIST, INDEX_FINGER_MCP, PINKY_MCP, MIDDLE_FINGER_MCP
from .math_utils import (
    normalize_vector,
    magnitude,
    cross,
    rotation_matrix_from_vectors,
    apply_rotation,
    orthonormalize,
    fit_plane_to_points,
)


@dataclass
class NormalizationResult:
    """Result of landmark normalization with inverse transform parameters."""

    landmarks: NDArray[np.floating]
    """Normalized landmarks (21, 3)"""

    original_centroid: NDArray[np.floating]
    """Original centroid/anchor point for translation"""

    scale_factor: float
    """Scale factor applied (original_size / normalized_size)"""

    rotation_matrix: NDArray[np.floating]
    """3x3 rotation matrix applied"""

    method: str
    """Description of normalization method used"""


# =============================================================================
# Bounding Box Utilities
# =============================================================================


def calculate_bounding_box(
    landmarks: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate axis-aligned bounding box of landmarks.

    Args:
        landmarks: Array of shape (21, 3) or (N, 3)

    Returns:
        Tuple of (min_corner, max_corner, dimensions)
    """
    min_corner = np.min(landmarks, axis=0)
    max_corner = np.max(landmarks, axis=0)
    dimensions = max_corner - min_corner

    return min_corner, max_corner, dimensions


def calculate_bounding_box_size(landmarks: NDArray[np.floating]) -> float:
    """
    Calculate the diagonal size of the bounding box.

    Args:
        landmarks: Array of shape (21, 3)

    Returns:
        Length of bounding box diagonal
    """
    _, _, dimensions = calculate_bounding_box(landmarks)
    return float(magnitude(dimensions))


# =============================================================================
# Palm Geometry Utilities
# =============================================================================


def calculate_centroid(landmarks: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculate centroid (mean position) of all landmarks.

    Args:
        landmarks: Array of shape (21, 3)

    Returns:
        3D centroid point
    """
    return np.mean(landmarks, axis=0)


def calculate_palm_width(landmarks: NDArray[np.floating]) -> float:
    """
    Calculate palm width as distance from index MCP to pinky MCP.

    Args:
        landmarks: Array of shape (21, 3)

    Returns:
        Palm width (distance between knuckles)
    """
    index_mcp = landmarks[INDEX_FINGER_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    return float(magnitude(index_mcp - pinky_mcp))


def calculate_palm_center(landmarks: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculate palm center as mean of MCP joints.

    Args:
        landmarks: Array of shape (21, 3)

    Returns:
        3D palm center point
    """
    # Palm is defined by wrist and MCP joints
    palm_indices = [WRIST, INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, PINKY_MCP]
    palm_points = landmarks[palm_indices]
    return np.mean(palm_points, axis=0)


def calculate_palm_normal(landmarks: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculate palm normal vector pointing out from palm.

    Uses cross product of palm vectors to find perpendicular direction.

    Args:
        landmarks: Array of shape (21, 3)

    Returns:
        Unit normal vector
    """
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_FINGER_MCP]
    pinky_mcp = landmarks[PINKY_MCP]

    # Vector from wrist to index MCP (up the hand)
    v1 = index_mcp - wrist
    # Vector from wrist to pinky MCP (across the hand)
    v2 = pinky_mcp - wrist

    # Cross product gives normal
    normal = cross(v1, v2)
    return normalize_vector(normal)


def calculate_palm_plane(
    landmarks: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Fit a plane to the palm landmarks.

    Args:
        landmarks: Array of shape (21, 3)

    Returns:
        Tuple of (palm_center, palm_normal)
    """
    # Use wrist and all MCP joints for plane fitting
    palm_indices = [WRIST, 5, 9, 13, 17]  # Wrist + 4 MCP joints
    palm_points = landmarks[palm_indices]

    return fit_plane_to_points(palm_points)


def calculate_finger_direction(landmarks: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculate average finger pointing direction.

    Uses middle finger as the reference.

    Args:
        landmarks: Array of shape (21, 3)

    Returns:
        Unit vector in finger direction
    """
    wrist = landmarks[WRIST]
    middle_tip = landmarks[12]  # MIDDLE_FINGER_TIP

    direction = middle_tip - wrist
    return normalize_vector(direction)


# =============================================================================
# Translation Normalization
# =============================================================================


def normalize_translation(
    landmarks: NDArray[np.floating],
    anchor: Literal["wrist", "centroid", "palm_center"] = "wrist"
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Translate landmarks so anchor point is at origin.

    Args:
        landmarks: Array of shape (21, 3)
        anchor: Which point to place at origin:
            - "wrist": Wrist landmark at origin
            - "centroid": Mean of all landmarks at origin
            - "palm_center": Palm center at origin

    Returns:
        Tuple of (translated_landmarks, original_anchor_position)
    """
    if anchor == "wrist":
        anchor_point = landmarks[WRIST].copy()
    elif anchor == "centroid":
        anchor_point = calculate_centroid(landmarks)
    elif anchor == "palm_center":
        anchor_point = calculate_palm_center(landmarks)
    else:
        raise ValueError(f"Unknown anchor: {anchor}")

    translated = landmarks - anchor_point
    return translated, anchor_point


# =============================================================================
# Scale Normalization
# =============================================================================


def normalize_scale_bbox(
    landmarks: NDArray[np.floating],
    target_size: float = 1.0
) -> tuple[NDArray[np.floating], float]:
    """
    Scale landmarks so bounding box diagonal equals target size.

    Args:
        landmarks: Array of shape (21, 3)
        target_size: Desired bounding box diagonal length

    Returns:
        Tuple of (scaled_landmarks, original_size)
    """
    current_size = calculate_bounding_box_size(landmarks)

    if current_size < 1e-10:
        return landmarks.copy(), 0.0

    scale = target_size / current_size
    scaled = landmarks * scale

    return scaled, current_size


def normalize_scale_palm(
    landmarks: NDArray[np.floating],
    target_width: float = 1.0
) -> tuple[NDArray[np.floating], float]:
    """
    Scale landmarks so palm width equals target width.

    Palm width is defined as distance from index MCP to pinky MCP.

    Args:
        landmarks: Array of shape (21, 3)
        target_width: Desired palm width

    Returns:
        Tuple of (scaled_landmarks, original_palm_width)
    """
    current_width = calculate_palm_width(landmarks)

    if current_width < 1e-10:
        return landmarks.copy(), 0.0

    scale = target_width / current_width
    scaled = landmarks * scale

    return scaled, current_width


# =============================================================================
# Rotation Normalization
# =============================================================================


def normalize_rotation(
    landmarks: NDArray[np.floating],
    align_palm: bool = True,
    align_fingers: bool = True
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Rotate landmarks to canonical orientation.

    Canonical orientation:
    - Palm normal points along positive Z axis (palm facing camera)
    - Fingers point along positive Y axis (up)
    - Thumb direction along positive X axis (right for right hand)

    Args:
        landmarks: Array of shape (21, 3)
        align_palm: If True, align palm to face Z+ direction
        align_fingers: If True, align fingers to point Y+ direction

    Returns:
        Tuple of (rotated_landmarks, rotation_matrix)
    """
    result = landmarks.copy()
    total_rotation = np.eye(3)

    if align_palm:
        # Step 1: Rotate palm normal to align with Z+
        palm_normal = calculate_palm_normal(result)
        target_normal = np.array([0.0, 0.0, 1.0])

        R1 = rotation_matrix_from_vectors(palm_normal, target_normal)
        result = apply_rotation(result, R1)
        total_rotation = R1 @ total_rotation

    if align_fingers:
        # Step 2: Rotate around Z to align fingers with Y+
        finger_dir = calculate_finger_direction(result)
        # Project finger direction onto XY plane
        finger_xy = np.array([finger_dir[0], finger_dir[1], 0.0])
        finger_xy = normalize_vector(finger_xy)

        if magnitude(finger_xy) > 1e-6:
            target_up = np.array([0.0, 1.0, 0.0])
            R2 = rotation_matrix_from_vectors(finger_xy, target_up)
            result = apply_rotation(result, R2)
            total_rotation = R2 @ total_rotation

    return result, total_rotation


def create_canonical_basis(
    landmarks: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Create rotation matrix that transforms landmarks to canonical orientation.

    The canonical basis is:
    - X axis: From pinky MCP to index MCP (across palm)
    - Y axis: From wrist toward middle finger (up the hand)
    - Z axis: Palm normal (cross product of X and Y)

    Args:
        landmarks: Array of shape (21, 3)

    Returns:
        3x3 rotation matrix (columns are the canonical basis vectors)
    """
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_FINGER_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    middle_mcp = landmarks[MIDDLE_FINGER_MCP]

    # X axis: across the palm
    x_axis = index_mcp - pinky_mcp

    # Y axis: up the hand (from wrist toward fingers)
    y_axis = middle_mcp - wrist

    # Orthonormalize
    x_norm, y_norm, z_norm = orthonormalize(x_axis, y_axis)

    # Stack as columns to form rotation matrix
    basis = np.column_stack([x_norm, y_norm, z_norm])
    return basis


# =============================================================================
# Combined Normalization
# =============================================================================


def normalize_all(
    landmarks: NDArray[np.floating],
    translation_anchor: Literal["wrist", "centroid", "palm_center"] = "wrist",
    scale_method: Literal["bbox", "palm"] = "palm",
    target_scale: float = 1.0,
    align_rotation: bool = True
) -> NormalizationResult:
    """
    Apply full normalization pipeline to landmarks.

    Pipeline order: translate -> scale -> rotate

    Args:
        landmarks: Array of shape (21, 3)
        translation_anchor: Point to place at origin
        scale_method: Method for scale normalization ("bbox" or "palm")
        target_scale: Target size/width after scaling
        align_rotation: Whether to apply rotation normalization

    Returns:
        NormalizationResult with normalized landmarks and inverse parameters
    """
    result = landmarks.copy()

    # Step 1: Translation
    result, original_anchor = normalize_translation(result, translation_anchor)

    # Step 2: Scale
    if scale_method == "bbox":
        result, original_scale = normalize_scale_bbox(result, target_scale)
    else:
        result, original_scale = normalize_scale_palm(result, target_scale)

    # Step 3: Rotation
    if align_rotation:
        result, rotation_matrix = normalize_rotation(result)
    else:
        rotation_matrix = np.eye(3)

    method = f"translate={translation_anchor}, scale={scale_method}, rotate={align_rotation}"

    return NormalizationResult(
        landmarks=result,
        original_centroid=original_anchor,
        scale_factor=original_scale,
        rotation_matrix=rotation_matrix,
        method=method
    )


# =============================================================================
# Denormalization (Inverse Transform)
# =============================================================================


def denormalize(
    landmarks: NDArray[np.floating],
    result: NormalizationResult
) -> NDArray[np.floating]:
    """
    Reverse normalization to recover original landmark positions.

    Applies inverse transforms in reverse order: unrotate -> unscale -> untranslate

    Args:
        landmarks: Normalized landmarks (21, 3)
        result: NormalizationResult from original normalization

    Returns:
        Denormalized landmarks in original coordinate space
    """
    denorm = landmarks.copy()

    # Inverse rotation (transpose of rotation matrix)
    inv_rotation = result.rotation_matrix.T
    denorm = apply_rotation(denorm, inv_rotation)

    # Inverse scale
    if result.scale_factor > 1e-10:
        denorm = denorm * result.scale_factor

    # Inverse translation
    denorm = denorm + result.original_centroid

    return denorm


# =============================================================================
# Batch Processing
# =============================================================================


def normalize_sequence(
    frames: list[NDArray[np.floating]],
    use_first_frame_params: bool = True,
    **normalize_kwargs
) -> list[NormalizationResult]:
    """
    Normalize a sequence of landmark frames.

    Args:
        frames: List of landmark arrays, each (21, 3)
        use_first_frame_params: If True, use first frame's normalization
            parameters for all frames (maintains relative motion)
        **normalize_kwargs: Arguments passed to normalize_all()

    Returns:
        List of NormalizationResult for each frame
    """
    if not frames:
        return []

    results = []

    if use_first_frame_params:
        # Normalize first frame to get reference parameters
        first_result = normalize_all(frames[0], **normalize_kwargs)
        results.append(first_result)

        # Apply same transform to remaining frames
        for frame in frames[1:]:
            # Apply same translation
            translated = frame - first_result.original_centroid

            # Apply same scale
            if first_result.scale_factor > 1e-10:
                scaled = translated / first_result.scale_factor
            else:
                scaled = translated

            # Apply same rotation
            rotated = apply_rotation(scaled, first_result.rotation_matrix)

            results.append(NormalizationResult(
                landmarks=rotated,
                original_centroid=first_result.original_centroid,
                scale_factor=first_result.scale_factor,
                rotation_matrix=first_result.rotation_matrix,
                method=first_result.method + " (sequence-aligned)"
            ))
    else:
        # Normalize each frame independently
        for frame in frames:
            results.append(normalize_all(frame, **normalize_kwargs))

    return results


def normalize_recording_landmarks(
    landmarks_sequence: NDArray[np.floating],
    **normalize_kwargs
) -> tuple[NDArray[np.floating], NormalizationResult]:
    """
    Normalize all frames in a recording using first frame as reference.

    Args:
        landmarks_sequence: Array of shape (num_frames, 21, 3)
        **normalize_kwargs: Arguments passed to normalize_all()

    Returns:
        Tuple of (normalized_sequence, first_frame_result)
    """
    frames = [landmarks_sequence[i] for i in range(len(landmarks_sequence))]
    results = normalize_sequence(frames, use_first_frame_params=True, **normalize_kwargs)

    normalized = np.array([r.landmarks for r in results])
    return normalized, results[0]


def normalize_two_hands(
    right_landmarks: NDArray[np.floating],
    left_landmarks: NDArray[np.floating],
    translation_anchor: Literal["wrist", "centroid", "palm_center"] = "wrist",
    scale_method: Literal["bbox", "palm"] = "palm",
    target_scale: float = 1.0,
    align_rotation: bool = True
) -> tuple[NormalizationResult, NDArray[np.floating]]:
    """
    Normalize both hands relative to the right hand's reference frame.

    Both hands are transformed using the right hand's wrist position,
    palm width, and palm orientation. This preserves the spatial
    relationship between the two hands, which is important for BSL
    where the dominant hand touches/points at the non-dominant.

    Args:
        right_landmarks: Right hand landmarks (21, 3)
        left_landmarks: Left hand landmarks (21, 3)
        translation_anchor: Point to place at origin (from right hand)
        scale_method: Method for scale normalization
        target_scale: Target size after scaling
        align_rotation: Whether to apply rotation normalization

    Returns:
        Tuple of (right_hand_NormalizationResult, left_hand_normalized_landmarks)
    """
    # Get right hand's transform parameters
    right_result = normalize_all(
        right_landmarks,
        translation_anchor=translation_anchor,
        scale_method=scale_method,
        target_scale=target_scale,
        align_rotation=align_rotation,
    )

    # Apply the SAME transform to left hand
    # Step 1: Same translation (right hand's anchor)
    left_translated = left_landmarks - right_result.original_centroid

    # Step 2: Same scale
    if right_result.scale_factor > 1e-10:
        left_scaled = left_translated / right_result.scale_factor
    else:
        left_scaled = left_translated

    # Step 3: Same rotation
    if align_rotation:
        from .math_utils import apply_rotation
        left_normalized = apply_rotation(left_scaled, right_result.rotation_matrix)
    else:
        left_normalized = left_scaled

    return right_result, left_normalized
