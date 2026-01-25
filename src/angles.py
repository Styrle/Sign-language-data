"""
Joint angle calculation module.

Calculates joint angles from hand landmarks and provides conversion
to Three.js bone rotation format for 3D visualization.

Three.js conventions:
- Euler angles in radians with order (typically 'XYZ')
- Right-handed coordinate system
- Bone rotations are relative to parent bone
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .landmarks import (
    WRIST,
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP,
    MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP,
    RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP,
)
from .types import JointAngles, FingerAngles, ValidationResult
from .math_utils import normalize_vector, magnitude, cross, dot


# =============================================================================
# Constants
# =============================================================================


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to range [-180, 180] degrees.

    Args:
        angle: Angle in degrees

    Returns:
        Normalized angle in degrees
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


# =============================================================================
# Anatomical Limits
# =============================================================================


# Anatomical limits for joint angles (in degrees)
# Based on typical human hand range of motion
ANGLE_LIMITS: dict[str, tuple[float, float]] = {
    # Thumb joints (CMC has more complex motion)
    "thumb_cmc": (0, 90),      # Thumb base flexion
    "thumb_mcp": (0, 80),      # Thumb knuckle
    "thumb_ip": (0, 90),       # Thumb tip joint

    # Finger MCP joints (knuckles)
    "index_mcp": (0, 100),
    "middle_mcp": (0, 100),
    "ring_mcp": (0, 100),
    "pinky_mcp": (0, 100),

    # Finger PIP joints (middle joints)
    "index_pip": (0, 110),
    "middle_pip": (0, 110),
    "ring_pip": (0, 110),
    "pinky_pip": (0, 110),

    # Finger DIP joints (tip joints)
    "index_dip": (0, 80),
    "middle_dip": (0, 80),
    "ring_dip": (0, 80),
    "pinky_dip": (0, 80),

    # Wrist
    "wrist_flexion": (-80, 80),      # Forward/backward bend
    "wrist_deviation": (-30, 45),     # Side to side (radial/ulnar)
    "wrist_rotation": (-90, 90),      # Pronation/supination
}


# Finger joint indices for angle calculation
# Each entry is (proximal_idx, middle_idx, distal_idx)
FINGER_JOINT_INDICES: dict[str, dict[str, tuple[int, int, int]]] = {
    "thumb": {
        "cmc": (WRIST, THUMB_CMC, THUMB_MCP),
        "mcp": (THUMB_CMC, THUMB_MCP, THUMB_IP),
        "ip": (THUMB_MCP, THUMB_IP, THUMB_TIP),
    },
    "index": {
        "mcp": (WRIST, INDEX_FINGER_MCP, INDEX_FINGER_PIP),
        "pip": (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP),
        "dip": (INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP),
    },
    "middle": {
        "mcp": (WRIST, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP),
        "pip": (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP),
        "dip": (MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP),
    },
    "ring": {
        "mcp": (WRIST, RING_FINGER_MCP, RING_FINGER_PIP),
        "pip": (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP),
        "dip": (RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP),
    },
    "pinky": {
        "mcp": (WRIST, PINKY_MCP, PINKY_PIP),
        "pip": (PINKY_MCP, PINKY_PIP, PINKY_DIP),
        "dip": (PINKY_PIP, PINKY_DIP, PINKY_TIP),
    },
}


# =============================================================================
# Three.js Bone Mapping
# =============================================================================


# Mapping from our joint names to Three.js/Mixamo bone names
THREEJS_BONE_MAPPING: dict[str, str] = {
    # Wrist
    "wrist": "Hand",

    # Thumb
    "thumb_cmc": "HandThumb1",
    "thumb_mcp": "HandThumb2",
    "thumb_ip": "HandThumb3",

    # Index
    "index_mcp": "HandIndex1",
    "index_pip": "HandIndex2",
    "index_dip": "HandIndex3",

    # Middle
    "middle_mcp": "HandMiddle1",
    "middle_pip": "HandMiddle2",
    "middle_dip": "HandMiddle3",

    # Ring
    "ring_mcp": "HandRing1",
    "ring_pip": "HandRing2",
    "ring_dip": "HandRing3",

    # Pinky
    "pinky_mcp": "HandPinky1",
    "pinky_pip": "HandPinky2",
    "pinky_dip": "HandPinky3",
}


# Three.js rotation order (Euler angle order)
THREEJS_ROTATION_ORDER = "XYZ"


# =============================================================================
# Angle Calculation
# =============================================================================


def calculate_angle_3points(
    p1: NDArray[np.floating],
    p2: NDArray[np.floating],
    p3: NDArray[np.floating]
) -> float:
    """
    Calculate the angle at p2 formed by vectors p2->p1 and p2->p3.

    Args:
        p1: First point (3D)
        p2: Middle point where angle is measured (3D)
        p3: Third point (3D)

    Returns:
        Angle in degrees (0 to 180)
    """
    v1 = p1 - p2
    v2 = p3 - p2

    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    # Clamp to handle numerical issues
    cos_angle = np.clip(dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    return float(radians_to_degrees(angle_rad))


def calculate_finger_angles(
    landmarks: NDArray[np.floating],
    finger: Literal["thumb", "index", "middle", "ring", "pinky"]
) -> dict[str, float]:
    """
    Calculate all joint angles for a finger.

    Args:
        landmarks: (21, 3) landmark array
        finger: Which finger to calculate

    Returns:
        Dictionary with joint angles in degrees:
        - For thumb: {cmc, mcp, ip}
        - For other fingers: {mcp, pip, dip}
    """
    joint_indices = FINGER_JOINT_INDICES[finger]
    angles = {}

    for joint_name, (i1, i2, i3) in joint_indices.items():
        p1 = landmarks[i1]
        p2 = landmarks[i2]
        p3 = landmarks[i3]

        # Calculate bend angle (180 - angle gives flexion)
        raw_angle = calculate_angle_3points(p1, p2, p3)
        # Flexion is how much the finger bends from straight
        flexion = 180.0 - raw_angle
        angles[joint_name] = max(0.0, flexion)  # Clamp negative

    return angles


def calculate_wrist_rotation(
    landmarks: NDArray[np.floating]
) -> dict[str, float]:
    """
    Calculate wrist orientation as Euler angles.

    Computes the rotation of the hand relative to a canonical orientation
    where the palm faces forward and fingers point up.

    Args:
        landmarks: (21, 3) landmark array

    Returns:
        Dictionary with {x, y, z} Euler angles in degrees
    """
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_FINGER_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    middle_mcp = landmarks[MIDDLE_FINGER_MCP]

    # Palm vectors
    palm_width_vec = index_mcp - pinky_mcp  # Across palm (X-like)
    finger_vec = middle_mcp - wrist         # Up the hand (Y-like)

    # Normalize
    palm_width_vec = normalize_vector(palm_width_vec)
    finger_vec = normalize_vector(finger_vec)

    # Palm normal (Z-like, pointing out of palm)
    palm_normal = normalize_vector(cross(palm_width_vec, finger_vec))

    # Re-orthogonalize
    finger_vec = normalize_vector(cross(palm_normal, palm_width_vec))

    # Build rotation matrix (columns are local axes)
    # In canonical pose: X = right, Y = up, Z = forward (out of palm)
    rotation_matrix = np.column_stack([palm_width_vec, finger_vec, palm_normal])

    # Extract Euler angles (XYZ order)
    # For a rotation matrix R:
    # R = Rz * Ry * Rx (for XYZ order applied in that sequence)

    # Check for gimbal lock
    if abs(rotation_matrix[0, 2]) < 0.99999:
        y = np.arcsin(-rotation_matrix[0, 2])
        x = np.arctan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
        z = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
    else:
        # Gimbal lock case
        z = 0
        if rotation_matrix[0, 2] < 0:
            y = np.pi / 2
            x = np.arctan2(rotation_matrix[1, 0], rotation_matrix[1, 1])
        else:
            y = -np.pi / 2
            x = np.arctan2(-rotation_matrix[1, 0], rotation_matrix[1, 1])

    return {
        "x": radians_to_degrees(float(x)),  # Flexion/extension
        "y": radians_to_degrees(float(y)),  # Radial/ulnar deviation
        "z": radians_to_degrees(float(z)),  # Rotation
    }


def calculate_all_joint_angles(
    landmarks: NDArray[np.floating]
) -> JointAngles:
    """
    Calculate all joint angles for a hand.

    Args:
        landmarks: (21, 3) landmark array

    Returns:
        JointAngles structure with all finger and wrist angles
    """
    # Calculate finger angles
    thumb_angles = calculate_finger_angles(landmarks, "thumb")
    index_angles = calculate_finger_angles(landmarks, "index")
    middle_angles = calculate_finger_angles(landmarks, "middle")
    ring_angles = calculate_finger_angles(landmarks, "ring")
    pinky_angles = calculate_finger_angles(landmarks, "pinky")

    # Calculate wrist rotation
    wrist_rot = calculate_wrist_rotation(landmarks)

    return JointAngles(
        thumb=FingerAngles(
            mcp=thumb_angles["cmc"],  # Map CMC to MCP field
            pip=thumb_angles["mcp"],  # Map MCP to PIP field
            dip=thumb_angles["ip"],   # Map IP to DIP field
        ),
        index=FingerAngles(
            mcp=index_angles["mcp"],
            pip=index_angles["pip"],
            dip=index_angles["dip"],
        ),
        middle=FingerAngles(
            mcp=middle_angles["mcp"],
            pip=middle_angles["pip"],
            dip=middle_angles["dip"],
        ),
        ring=FingerAngles(
            mcp=ring_angles["mcp"],
            pip=ring_angles["pip"],
            dip=ring_angles["dip"],
        ),
        pinky=FingerAngles(
            mcp=pinky_angles["mcp"],
            pip=pinky_angles["pip"],
            dip=pinky_angles["dip"],
        ),
        wrist_rotation=wrist_rot["z"],
        wrist_flexion=wrist_rot["x"],
    )


# =============================================================================
# Validation
# =============================================================================


def validate_angles(angles: JointAngles) -> ValidationResult:
    """
    Validate joint angles against anatomical limits.

    Args:
        angles: JointAngles to validate

    Returns:
        ValidationResult with any errors/warnings
    """
    errors = []
    warnings = []

    def check_limit(name: str, value: float, limits: tuple[float, float]):
        min_val, max_val = limits
        if value < min_val - 10:  # Allow small tolerance
            errors.append(f"{name} angle {value:.1f}° below minimum {min_val}°")
        elif value < min_val:
            warnings.append(f"{name} angle {value:.1f}° slightly below minimum {min_val}°")
        elif value > max_val + 10:
            errors.append(f"{name} angle {value:.1f}° above maximum {max_val}°")
        elif value > max_val:
            warnings.append(f"{name} angle {value:.1f}° slightly above maximum {max_val}°")

    # Check thumb (using CMC/MCP/IP mapping)
    check_limit("thumb_cmc", angles.thumb.mcp, ANGLE_LIMITS["thumb_cmc"])
    check_limit("thumb_mcp", angles.thumb.pip, ANGLE_LIMITS["thumb_mcp"])
    check_limit("thumb_ip", angles.thumb.dip, ANGLE_LIMITS["thumb_ip"])

    # Check other fingers
    for finger_name in ["index", "middle", "ring", "pinky"]:
        finger = getattr(angles, finger_name)
        check_limit(f"{finger_name}_mcp", finger.mcp, ANGLE_LIMITS[f"{finger_name}_mcp"])
        check_limit(f"{finger_name}_pip", finger.pip, ANGLE_LIMITS[f"{finger_name}_pip"])
        check_limit(f"{finger_name}_dip", finger.dip, ANGLE_LIMITS[f"{finger_name}_dip"])

    # Check wrist
    check_limit("wrist_flexion", angles.wrist_flexion, ANGLE_LIMITS["wrist_flexion"])
    check_limit("wrist_rotation", angles.wrist_rotation, ANGLE_LIMITS["wrist_rotation"])

    is_valid = len(errors) == 0
    score = 1.0 - (len(errors) * 0.1 + len(warnings) * 0.02)
    score = max(0.0, min(1.0, score))

    return ValidationResult(
        is_valid=is_valid,
        score=score,
        errors=errors,
        warnings=warnings,
    )


def clamp_angles(angles: JointAngles) -> JointAngles:
    """
    Clamp joint angles to anatomical limits.

    Args:
        angles: JointAngles to clamp

    Returns:
        New JointAngles with values clamped to valid ranges
    """
    def clamp(value: float, limits: tuple[float, float]) -> float:
        return max(limits[0], min(limits[1], value))

    return JointAngles(
        thumb=FingerAngles(
            mcp=clamp(angles.thumb.mcp, ANGLE_LIMITS["thumb_cmc"]),
            pip=clamp(angles.thumb.pip, ANGLE_LIMITS["thumb_mcp"]),
            dip=clamp(angles.thumb.dip, ANGLE_LIMITS["thumb_ip"]),
        ),
        index=FingerAngles(
            mcp=clamp(angles.index.mcp, ANGLE_LIMITS["index_mcp"]),
            pip=clamp(angles.index.pip, ANGLE_LIMITS["index_pip"]),
            dip=clamp(angles.index.dip, ANGLE_LIMITS["index_dip"]),
        ),
        middle=FingerAngles(
            mcp=clamp(angles.middle.mcp, ANGLE_LIMITS["middle_mcp"]),
            pip=clamp(angles.middle.pip, ANGLE_LIMITS["middle_pip"]),
            dip=clamp(angles.middle.dip, ANGLE_LIMITS["middle_dip"]),
        ),
        ring=FingerAngles(
            mcp=clamp(angles.ring.mcp, ANGLE_LIMITS["ring_mcp"]),
            pip=clamp(angles.ring.pip, ANGLE_LIMITS["ring_pip"]),
            dip=clamp(angles.ring.dip, ANGLE_LIMITS["ring_dip"]),
        ),
        pinky=FingerAngles(
            mcp=clamp(angles.pinky.mcp, ANGLE_LIMITS["pinky_mcp"]),
            pip=clamp(angles.pinky.pip, ANGLE_LIMITS["pinky_pip"]),
            dip=clamp(angles.pinky.dip, ANGLE_LIMITS["pinky_dip"]),
        ),
        wrist_rotation=clamp(angles.wrist_rotation, ANGLE_LIMITS["wrist_rotation"]),
        wrist_flexion=clamp(angles.wrist_flexion, ANGLE_LIMITS["wrist_flexion"]),
    )


# =============================================================================
# Three.js Conversion
# =============================================================================


@dataclass
class ThreeJSBoneRotation:
    """Rotation data for a Three.js bone."""

    bone_name: str
    """Three.js bone name (e.g., 'LeftHandIndex1')"""

    x: float
    """X rotation in radians"""

    y: float
    """Y rotation in radians"""

    z: float
    """Z rotation in radians"""

    order: str = "XYZ"
    """Euler rotation order"""


def angles_to_threejs_rotations(
    angles: JointAngles,
    hand: Literal["Left", "Right"] = "Right"
) -> dict[str, ThreeJSBoneRotation]:
    """
    Convert joint angles to Three.js bone rotation format.

    Three.js uses:
    - Right-handed coordinate system
    - Euler angles in radians
    - Default rotation order 'XYZ'

    For hand bones:
    - X rotation: Flexion/extension (curl)
    - Y rotation: Abduction/adduction (spread)
    - Z rotation: Twist

    Args:
        angles: JointAngles to convert
        hand: Which hand ("Left" or "Right")

    Returns:
        Dictionary mapping bone names to ThreeJSBoneRotation
    """
    result = {}

    # Hand prefix for bone names
    prefix = f"{hand}Hand"

    # Mirror factor for left hand (flip certain axes)
    mirror = -1.0 if hand == "Left" else 1.0

    # Wrist rotation
    result[f"{prefix}"] = ThreeJSBoneRotation(
        bone_name=f"{prefix}",
        x=degrees_to_radians(angles.wrist_flexion),
        y=degrees_to_radians(angles.wrist_rotation) * mirror,
        z=0.0,
        order=THREEJS_ROTATION_ORDER,
    )

    # Helper to create finger bone rotations
    def add_finger_bones(
        finger_name: str,
        finger_angles: FingerAngles,
        is_thumb: bool = False
    ):
        bone_base = f"{prefix}{finger_name.capitalize()}"

        # Joint 1 (MCP for fingers, CMC for thumb)
        # Primary rotation is X (flexion)
        result[f"{bone_base}1"] = ThreeJSBoneRotation(
            bone_name=f"{bone_base}1",
            x=degrees_to_radians(finger_angles.mcp),
            y=0.0,
            z=0.0,
            order=THREEJS_ROTATION_ORDER,
        )

        # Joint 2 (PIP for fingers, MCP for thumb)
        result[f"{bone_base}2"] = ThreeJSBoneRotation(
            bone_name=f"{bone_base}2",
            x=degrees_to_radians(finger_angles.pip),
            y=0.0,
            z=0.0,
            order=THREEJS_ROTATION_ORDER,
        )

        # Joint 3 (DIP for fingers, IP for thumb)
        result[f"{bone_base}3"] = ThreeJSBoneRotation(
            bone_name=f"{bone_base}3",
            x=degrees_to_radians(finger_angles.dip),
            y=0.0,
            z=0.0,
            order=THREEJS_ROTATION_ORDER,
        )

    # Add all finger bones
    add_finger_bones("Thumb", angles.thumb, is_thumb=True)
    add_finger_bones("Index", angles.index)
    add_finger_bones("Middle", angles.middle)
    add_finger_bones("Ring", angles.ring)
    add_finger_bones("Pinky", angles.pinky)

    return result


def threejs_rotations_to_json(
    rotations: dict[str, ThreeJSBoneRotation]
) -> dict[str, dict]:
    """
    Convert Three.js rotations to JSON-serializable format.

    Args:
        rotations: Dictionary of ThreeJSBoneRotation

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        bone_name: {
            "x": rot.x,
            "y": rot.y,
            "z": rot.z,
            "order": rot.order,
        }
        for bone_name, rot in rotations.items()
    }


# =============================================================================
# Interpolation
# =============================================================================


def interpolate_angle(a1: float, a2: float, t: float) -> float:
    """
    Linearly interpolate between two angles.

    Handles wraparound for angles near ±180°.

    Args:
        a1: Start angle in degrees
        a2: End angle in degrees
        t: Interpolation factor (0 = a1, 1 = a2)

    Returns:
        Interpolated angle in degrees
    """
    # Normalize both angles
    a1 = normalize_angle(a1)
    a2 = normalize_angle(a2)

    # Find shortest path
    diff = a2 - a1
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360

    result = a1 + diff * t
    return normalize_angle(result)


def interpolate_finger_angles(
    f1: FingerAngles,
    f2: FingerAngles,
    t: float
) -> FingerAngles:
    """
    Interpolate between two FingerAngles.

    Args:
        f1: Start finger angles
        f2: End finger angles
        t: Interpolation factor (0 = f1, 1 = f2)

    Returns:
        Interpolated FingerAngles
    """
    return FingerAngles(
        mcp=interpolate_angle(f1.mcp, f2.mcp, t),
        pip=interpolate_angle(f1.pip, f2.pip, t),
        dip=interpolate_angle(f1.dip, f2.dip, t),
    )


def interpolate_angles(
    a1: JointAngles,
    a2: JointAngles,
    t: float
) -> JointAngles:
    """
    Interpolate between two JointAngles.

    Args:
        a1: Start joint angles
        a2: End joint angles
        t: Interpolation factor (0 = a1, 1 = a2)

    Returns:
        Interpolated JointAngles
    """
    return JointAngles(
        thumb=interpolate_finger_angles(a1.thumb, a2.thumb, t),
        index=interpolate_finger_angles(a1.index, a2.index, t),
        middle=interpolate_finger_angles(a1.middle, a2.middle, t),
        ring=interpolate_finger_angles(a1.ring, a2.ring, t),
        pinky=interpolate_finger_angles(a1.pinky, a2.pinky, t),
        wrist_rotation=interpolate_angle(a1.wrist_rotation, a2.wrist_rotation, t),
        wrist_flexion=interpolate_angle(a1.wrist_flexion, a2.wrist_flexion, t),
    )


# =============================================================================
# Utilities
# =============================================================================


def angles_to_dict(angles: JointAngles) -> dict:
    """
    Convert JointAngles to a flat dictionary.

    Args:
        angles: JointAngles to convert

    Returns:
        Dictionary with all angle values
    """
    return {
        "thumb_cmc": angles.thumb.mcp,
        "thumb_mcp": angles.thumb.pip,
        "thumb_ip": angles.thumb.dip,
        "index_mcp": angles.index.mcp,
        "index_pip": angles.index.pip,
        "index_dip": angles.index.dip,
        "middle_mcp": angles.middle.mcp,
        "middle_pip": angles.middle.pip,
        "middle_dip": angles.middle.dip,
        "ring_mcp": angles.ring.mcp,
        "ring_pip": angles.ring.pip,
        "ring_dip": angles.ring.dip,
        "pinky_mcp": angles.pinky.mcp,
        "pinky_pip": angles.pinky.pip,
        "pinky_dip": angles.pinky.dip,
        "wrist_rotation": angles.wrist_rotation,
        "wrist_flexion": angles.wrist_flexion,
    }


def dict_to_angles(data: dict) -> JointAngles:
    """
    Convert a flat dictionary to JointAngles.

    Args:
        data: Dictionary with angle values

    Returns:
        JointAngles structure
    """
    return JointAngles(
        thumb=FingerAngles(
            mcp=data.get("thumb_cmc", 0),
            pip=data.get("thumb_mcp", 0),
            dip=data.get("thumb_ip", 0),
        ),
        index=FingerAngles(
            mcp=data.get("index_mcp", 0),
            pip=data.get("index_pip", 0),
            dip=data.get("index_dip", 0),
        ),
        middle=FingerAngles(
            mcp=data.get("middle_mcp", 0),
            pip=data.get("middle_pip", 0),
            dip=data.get("middle_dip", 0),
        ),
        ring=FingerAngles(
            mcp=data.get("ring_mcp", 0),
            pip=data.get("ring_pip", 0),
            dip=data.get("ring_dip", 0),
        ),
        pinky=FingerAngles(
            mcp=data.get("pinky_mcp", 0),
            pip=data.get("pinky_pip", 0),
            dip=data.get("pinky_dip", 0),
        ),
        wrist_rotation=data.get("wrist_rotation", 0),
        wrist_flexion=data.get("wrist_flexion", 0),
    )
