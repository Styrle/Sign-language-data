"""
Math utilities for landmark processing.

Vector operations, rotation matrices, and geometric calculations
used by the normalization module.
"""

import numpy as np
from numpy.typing import NDArray


def normalize_vector(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector of any dimension

    Returns:
        Unit vector in same direction, or zero vector if input is zero
    """
    mag = np.linalg.norm(v)
    if mag < 1e-10:
        return np.zeros_like(v)
    return v / mag


def magnitude(v: NDArray[np.floating]) -> float:
    """
    Calculate the magnitude (length) of a vector.

    Args:
        v: Input vector

    Returns:
        Euclidean norm of the vector
    """
    return float(np.linalg.norm(v))


def dot(v1: NDArray[np.floating], v2: NDArray[np.floating]) -> float:
    """
    Calculate dot product of two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Scalar dot product
    """
    return float(np.dot(v1, v2))


def cross(v1: NDArray[np.floating], v2: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculate cross product of two 3D vectors.

    Args:
        v1: First 3D vector
        v2: Second 3D vector

    Returns:
        Cross product vector perpendicular to both inputs
    """
    return np.cross(v1, v2)


def angle_between_vectors(v1: NDArray[np.floating], v2: NDArray[np.floating]) -> float:
    """
    Calculate angle between two vectors in radians.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Angle in radians [0, pi]
    """
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    # Clamp to avoid numerical issues with arccos
    cos_angle = np.clip(dot(v1_norm, v2_norm), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def rotation_matrix_from_vectors(
    vec_from: NDArray[np.floating],
    vec_to: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Calculate rotation matrix that rotates vec_from to align with vec_to.

    Uses Rodrigues' rotation formula.

    Args:
        vec_from: Source direction vector
        vec_to: Target direction vector

    Returns:
        3x3 rotation matrix
    """
    a = normalize_vector(vec_from)
    b = normalize_vector(vec_to)

    # Check if vectors are already aligned
    cos_angle = dot(a, b)
    if cos_angle > 1.0 - 1e-10:
        return np.eye(3)

    # Check if vectors are opposite
    if cos_angle < -1.0 + 1e-10:
        # Find perpendicular vector
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        perp = normalize_vector(cross(a, perp))
        # 180 degree rotation around perpendicular axis
        return rotation_matrix_from_axis_angle(perp, np.pi)

    # Rodrigues' formula
    v = cross(a, b)
    s = magnitude(v)  # sin(angle)
    c = cos_angle     # cos(angle)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-10))
    return R


def rotation_matrix_from_axis_angle(
    axis: NDArray[np.floating],
    angle: float
) -> NDArray[np.floating]:
    """
    Create rotation matrix from axis-angle representation.

    Args:
        axis: Unit vector defining rotation axis
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    axis = normalize_vector(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    x, y, z = axis

    return np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])


def rotation_matrix_from_euler(
    roll: float,
    pitch: float,
    yaw: float
) -> NDArray[np.floating]:
    """
    Create rotation matrix from Euler angles (XYZ order).

    Args:
        roll: Rotation around X axis in radians
        pitch: Rotation around Y axis in radians
        yaw: Rotation around Z axis in radians

    Returns:
        3x3 rotation matrix
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def apply_rotation(
    points: NDArray[np.floating],
    rotation_matrix: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Apply rotation matrix to a set of 3D points.

    Args:
        points: Array of shape (N, 3) containing N 3D points
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Rotated points with same shape as input
    """
    return points @ rotation_matrix.T


def apply_transformation(
    points: NDArray[np.floating],
    rotation: NDArray[np.floating],
    translation: NDArray[np.floating],
    scale: float = 1.0
) -> NDArray[np.floating]:
    """
    Apply full transformation (scale, rotate, translate) to points.

    Transformation order: scale -> rotate -> translate

    Args:
        points: Array of shape (N, 3) containing N 3D points
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
        scale: Uniform scale factor

    Returns:
        Transformed points
    """
    return apply_rotation(points * scale, rotation) + translation


def invert_transformation(
    rotation: NDArray[np.floating],
    translation: NDArray[np.floating],
    scale: float = 1.0
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """
    Compute inverse of a transformation.

    Args:
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
        scale: Uniform scale factor

    Returns:
        Tuple of (inverse_rotation, inverse_translation, inverse_scale)
    """
    inv_rotation = rotation.T
    inv_scale = 1.0 / scale if scale != 0 else 1.0
    inv_translation = -inv_rotation @ translation * inv_scale

    return inv_rotation, inv_translation, inv_scale


def project_onto_plane(
    point: NDArray[np.floating],
    plane_point: NDArray[np.floating],
    plane_normal: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Project a point onto a plane.

    Args:
        point: 3D point to project
        plane_point: Any point on the plane
        plane_normal: Normal vector of the plane

    Returns:
        Projected point on the plane
    """
    normal = normalize_vector(plane_normal)
    v = point - plane_point
    dist = dot(v, normal)
    return point - dist * normal


def distance_point_to_plane(
    point: NDArray[np.floating],
    plane_point: NDArray[np.floating],
    plane_normal: NDArray[np.floating]
) -> float:
    """
    Calculate signed distance from point to plane.

    Args:
        point: 3D point
        plane_point: Any point on the plane
        plane_normal: Normal vector of the plane

    Returns:
        Signed distance (positive if on normal side)
    """
    normal = normalize_vector(plane_normal)
    v = point - plane_point
    return float(dot(v, normal))


def fit_plane_to_points(
    points: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Fit a plane to a set of 3D points using SVD.

    Args:
        points: Array of shape (N, 3) containing N 3D points

    Returns:
        Tuple of (centroid, normal_vector)
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD - the normal is the last right singular vector
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]

    return centroid, normal


def orthonormalize(v1: NDArray[np.floating], v2: NDArray[np.floating]) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating]
]:
    """
    Create orthonormal basis from two vectors using Gram-Schmidt.

    Args:
        v1: First vector (becomes X axis after normalization)
        v2: Second vector (used to determine Y axis)

    Returns:
        Tuple of (x_axis, y_axis, z_axis) orthonormal vectors
    """
    x = normalize_vector(v1)

    # Remove component of v2 parallel to x
    v2_orth = v2 - dot(v2, x) * x
    y = normalize_vector(v2_orth)

    z = cross(x, y)

    return x, y, z
