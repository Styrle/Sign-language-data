#!/usr/bin/env python3
"""
Generate BSL Alphabet Dominant Hand Poses

Generates pose data for BSL fingerspelling alphabet letters A-Z.
BSL fingerspelling is two-handed: dominant (right) hand forms shapes,
non-dominant (left) hand is a flat open palm.

Skips letter C which already has Kaggle-recorded data.

Angles use BETWEEN-VECTORS convention: 0° = straight, higher = more bent.
"""

import json
import math
from datetime import datetime
from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

DICTIONARY_PATH = Path("data/output/bsl-dictionary.json")
DICTIONARY_MIN_PATH = Path("data/output/bsl-dictionary.min.json")

LANDMARK_NAMES = (
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
)

# Bone lengths for forward kinematics (from learning app)
BONE_LENGTHS = {
    "thumb": [0.10, 0.07, 0.05, 0.03],   # CMC, MCP, IP, TIP
    "index": [0.14, 0.09, 0.07, 0.03],    # MCP, PIP, DIP, TIP
    "middle": [0.16, 0.10, 0.07, 0.03],   # MCP, PIP, DIP, TIP
    "ring": [0.14, 0.09, 0.07, 0.03],     # MCP, PIP, DIP, TIP
    "pinky": [0.11, 0.07, 0.06, 0.03],    # MCP, PIP, DIP, TIP
}

# Base directions from wrist to each finger MCP (spread across palm)
FINGER_BASE_DIRECTIONS = {
    "thumb": (0.4, -0.8, 0.1),
    "index": (0.15, -1.0, 0.0),
    "middle": (0.0, -1.0, 0.0),
    "ring": (-0.15, -1.0, 0.0),
    "pinky": (-0.3, -0.95, 0.0),
}

# =============================================================================
# Right hand angle definitions per letter
# =============================================================================

RIGHT_HAND_ANGLES = {
    "A": {
        "thumb": {"mcp": 30, "pip": 20, "dip": 15},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 80, "pip": 140, "dip": 50},
        "ring": {"mcp": 80, "pip": 140, "dip": 50},
        "pinky": {"mcp": 80, "pip": 130, "dip": 45},
    },
    "B": {
        "thumb": {"mcp": 30, "pip": 20, "dip": 15},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 75, "pip": 135, "dip": 48},
        "ring": {"mcp": 75, "pip": 135, "dip": 48},
        "pinky": {"mcp": 75, "pip": 128, "dip": 42},
    },
    # C is skipped — keep existing Kaggle data
    "D": {
        "thumb": {"mcp": 40, "pip": 30, "dip": 20},
        "index": {"mcp": 12, "pip": 60, "dip": 40},
        "middle": {"mcp": 80, "pip": 140, "dip": 50},
        "ring": {"mcp": 80, "pip": 140, "dip": 50},
        "pinky": {"mcp": 80, "pip": 130, "dip": 45},
    },
    "E": {
        "thumb": {"mcp": 30, "pip": 20, "dip": 15},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 78, "pip": 138, "dip": 48},
        "ring": {"mcp": 78, "pip": 138, "dip": 48},
        "pinky": {"mcp": 78, "pip": 130, "dip": 42},
    },
    "F": {
        "thumb": {"mcp": 35, "pip": 25, "dip": 15},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 8, "pip": 5, "dip": 3},
        "ring": {"mcp": 80, "pip": 140, "dip": 50},
        "pinky": {"mcp": 80, "pip": 130, "dip": 45},
    },
    "G": {
        "thumb": {"mcp": 40, "pip": 30, "dip": 20},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 90, "pip": 145, "dip": 55},
        "ring": {"mcp": 90, "pip": 145, "dip": 55},
        "pinky": {"mcp": 90, "pip": 140, "dip": 50},
    },
    "H": {
        "thumb": {"mcp": 35, "pip": 25, "dip": 15},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 8, "pip": 5, "dip": 3},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "I": {
        "thumb": {"mcp": 40, "pip": 30, "dip": 20},
        "index": {"mcp": 85, "pip": 140, "dip": 50},
        "middle": {"mcp": 85, "pip": 140, "dip": 50},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 8, "pip": 5, "dip": 3},
    },
    "K": {
        "thumb": {"mcp": 35, "pip": 25, "dip": 15},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 18, "pip": 5, "dip": 3},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "L": {
        "thumb": {"mcp": 5, "pip": 3, "dip": 3},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 85, "pip": 140, "dip": 50},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "M": {
        "thumb": {"mcp": 40, "pip": 30, "dip": 20},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 8, "pip": 5, "dip": 3},
        "ring": {"mcp": 8, "pip": 5, "dip": 3},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "N": {
        "thumb": {"mcp": 40, "pip": 30, "dip": 20},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 8, "pip": 5, "dip": 3},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "O": {
        "thumb": {"mcp": 25, "pip": 35, "dip": 30},
        "index": {"mcp": 30, "pip": 45, "dip": 35},
        "middle": {"mcp": 82, "pip": 138, "dip": 48},
        "ring": {"mcp": 82, "pip": 138, "dip": 48},
        "pinky": {"mcp": 80, "pip": 132, "dip": 44},
    },
    "P": {
        "thumb": {"mcp": 35, "pip": 25, "dip": 15},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 8, "pip": 5, "dip": 3},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "Q": {
        "thumb": {"mcp": 35, "pip": 25, "dip": 15},
        "index": {"mcp": 20, "pip": 80, "dip": 60},
        "middle": {"mcp": 85, "pip": 140, "dip": 50},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "R": {
        "thumb": {"mcp": 40, "pip": 30, "dip": 20},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 10, "pip": 5, "dip": 3},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "S": {
        "thumb": {"mcp": 22, "pip": 15, "dip": 10},
        "index": {"mcp": 88, "pip": 142, "dip": 52},
        "middle": {"mcp": 88, "pip": 142, "dip": 52},
        "ring": {"mcp": 88, "pip": 142, "dip": 52},
        "pinky": {"mcp": 85, "pip": 138, "dip": 48},
    },
    "T": {
        "thumb": {"mcp": 20, "pip": 25, "dip": 20},
        "index": {"mcp": 45, "pip": 90, "dip": 60},
        "middle": {"mcp": 85, "pip": 140, "dip": 50},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "U": {
        "thumb": {"mcp": 40, "pip": 30, "dip": 20},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 8, "pip": 5, "dip": 3},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "V": {
        "thumb": {"mcp": 40, "pip": 30, "dip": 20},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 20, "pip": 5, "dip": 3},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "W": {
        "thumb": {"mcp": 40, "pip": 30, "dip": 20},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 14, "pip": 5, "dip": 3},
        "ring": {"mcp": 20, "pip": 5, "dip": 3},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "X": {
        "thumb": {"mcp": 40, "pip": 30, "dip": 20},
        "index": {"mcp": 12, "pip": 70, "dip": 55},
        "middle": {"mcp": 85, "pip": 140, "dip": 50},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 85, "pip": 135, "dip": 48},
    },
    "Y": {
        "thumb": {"mcp": 5, "pip": 3, "dip": 3},
        "index": {"mcp": 85, "pip": 140, "dip": 50},
        "middle": {"mcp": 85, "pip": 140, "dip": 50},
        "ring": {"mcp": 85, "pip": 140, "dip": 50},
        "pinky": {"mcp": 8, "pip": 5, "dip": 3},
    },
    "Z": {
        "thumb": {"mcp": 30, "pip": 20, "dip": 15},
        "index": {"mcp": 8, "pip": 5, "dip": 3},
        "middle": {"mcp": 80, "pip": 140, "dip": 50},
        "ring": {"mcp": 80, "pip": 140, "dip": 50},
        "pinky": {"mcp": 80, "pip": 130, "dip": 45},
    },
}

# J uses same hand shape as I
RIGHT_HAND_ANGLES["J"] = RIGHT_HAND_ANGLES["I"].copy()

# Wrist overrides (default is pitch=0, yaw=0, roll=0)
WRIST_OVERRIDES = {
    "P": {"pitch": -55, "yaw": 0, "roll": 0},
}

# Left hand (non-dominant) — flat open palm for all letters
LEFT_HAND_ANGLES = {
    "thumb": {"mcp": 15, "pip": 10, "dip": 8},
    "index": {"mcp": 5, "pip": 3, "dip": 3},
    "middle": {"mcp": 5, "pip": 3, "dip": 3},
    "ring": {"mcp": 5, "pip": 3, "dip": 3},
    "pinky": {"mcp": 5, "pip": 3, "dip": 3},
}
LEFT_HAND_WRIST = {"pitch": 0, "yaw": 90, "roll": 0}

# Finger pattern descriptions for summary
FINGER_PATTERNS = {
    "A": "Index up, rest curled",
    "B": "Index up, rest curled",
    "C": "C-curve shape",
    "D": "Index curved, rest curled",
    "E": "Index up, rest curled",
    "F": "Index+middle up, rest curled",
    "G": "Index up, rest tight fist",
    "H": "Index+middle extended",
    "I": "Pinky extended, rest curled",
    "J": "Pinky extended (dynamic J)",
    "K": "Index+middle V-shape",
    "L": "Thumb+index L-shape",
    "M": "Three fingers extended",
    "N": "Two fingers extended",
    "O": "Thumb+index O-circle",
    "P": "Index+middle down (wrist bent)",
    "Q": "Hooked index",
    "R": "Index+middle crossed",
    "S": "Fist, thumb across",
    "T": "Index tucked under thumb",
    "U": "Index+middle together",
    "V": "Index+middle spread V",
    "W": "Three fingers spread",
    "X": "Hooked index",
    "Y": "Thumb+pinky extended",
    "Z": "Index pointing (dynamic Z)",
}


# =============================================================================
# Forward Kinematics
# =============================================================================

def normalize_vec(v):
    """Normalize a 3D vector."""
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length < 1e-10:
        return (0, 0, 0)
    return (v[0]/length, v[1]/length, v[2]/length)


def rotate_vec_around_axis(v, axis, angle_deg):
    """Rotate vector v around axis by angle_deg degrees (Rodrigues' formula)."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    ax = normalize_vec(axis)

    # v_rot = v*cos(a) + (axis x v)*sin(a) + axis*(axis . v)*(1-cos(a))
    dot = v[0]*ax[0] + v[1]*ax[1] + v[2]*ax[2]
    cross = (
        ax[1]*v[2] - ax[2]*v[1],
        ax[2]*v[0] - ax[0]*v[2],
        ax[0]*v[1] - ax[1]*v[0],
    )
    return (
        v[0]*cos_a + cross[0]*sin_a + ax[0]*dot*(1-cos_a),
        v[1]*cos_a + cross[1]*sin_a + ax[1]*dot*(1-cos_a),
        v[2]*cos_a + cross[2]*sin_a + ax[2]*dot*(1-cos_a),
    )


def generate_finger_landmarks(base_pos, base_dir, bone_lengths, angles, finger_name):
    """
    Generate landmark positions for a finger using forward kinematics.

    Args:
        base_pos: (x, y, z) starting position
        base_dir: (dx, dy, dz) initial direction (normalized)
        bone_lengths: list of 4 bone lengths
        angles: dict with mcp, pip, dip angles in degrees
        finger_name: name of the finger

    Returns:
        list of 4 (x, y, z) positions (MCP/CMC, PIP/MCP, DIP/IP, TIP)
    """
    positions = []
    current_pos = base_pos
    current_dir = normalize_vec(base_dir)

    # Rotation axis perpendicular to finger direction (bend in the z-plane)
    rot_axis = (0, 0, 1)  # Bend around Z axis for simplicity

    joint_angles = [angles["mcp"], angles["pip"], angles["dip"]]

    for i, bone_len in enumerate(bone_lengths):
        # Apply joint angle rotation at each joint
        if i > 0 and i <= len(joint_angles):
            current_dir = rotate_vec_around_axis(current_dir, rot_axis, joint_angles[i-1])
            current_dir = normalize_vec(current_dir)

        # Extend along direction
        new_pos = (
            current_pos[0] + current_dir[0] * bone_len,
            current_pos[1] + current_dir[1] * bone_len,
            current_pos[2] + current_dir[2] * bone_len,
        )
        positions.append(new_pos)
        current_pos = new_pos

    return positions


def generate_hand_landmarks(finger_angles, wrist_angles=None):
    """
    Generate all 21 hand landmarks from finger angles using forward kinematics.

    Returns list of 21 {x, y, z} dicts.
    """
    wrist = (0.0, 0.0, 0.0)
    landmarks = [{"x": 0.0, "y": 0.0, "z": 0.0}]  # Wrist

    finger_order = ["thumb", "index", "middle", "ring", "pinky"]

    for finger in finger_order:
        base_dir = FINGER_BASE_DIRECTIONS[finger]
        bone_lens = BONE_LENGTHS[finger]
        angles = finger_angles[finger]

        positions = generate_finger_landmarks(
            wrist, base_dir, bone_lens, angles, finger
        )

        for pos in positions:
            landmarks.append({
                "x": round(pos[0], 6),
                "y": round(pos[1], 6),
                "z": round(pos[2], 6),
            })

    return landmarks


def generate_default_tolerances():
    """Generate default tolerances (20° for angles, 0.25 for positions)."""
    angle_tolerances = {
        "thumb": {"mcp": 20.0, "pip": 20.0, "dip": 20.0},
        "index": {"mcp": 20.0, "pip": 20.0, "dip": 20.0},
        "middle": {"mcp": 20.0, "pip": 20.0, "dip": 20.0},
        "ring": {"mcp": 20.0, "pip": 20.0, "dip": 20.0},
        "pinky": {"mcp": 20.0, "pip": 20.0, "dip": 20.0},
    }

    position_tolerances = {name: 0.25 for name in LANDMARK_NAMES}

    return angle_tolerances, position_tolerances


def build_pose(letter):
    """Build a complete pose dict for a letter."""
    right_angles = RIGHT_HAND_ANGLES[letter]
    wrist = WRIST_OVERRIDES.get(letter, {"pitch": 0, "yaw": 0, "roll": 0})

    # Build right hand angles dict with wrist
    right_angles_dict = {}
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        right_angles_dict[finger] = {
            "mcp": float(right_angles[finger]["mcp"]),
            "pip": float(right_angles[finger]["pip"]),
            "dip": float(right_angles[finger]["dip"]),
        }
    right_angles_dict["wrist"] = {
        "pitch": float(wrist["pitch"]),
        "yaw": float(wrist["yaw"]),
        "roll": float(wrist["roll"]),
    }

    # Build left hand angles dict with wrist
    left_angles_dict = {}
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        left_angles_dict[finger] = {
            "mcp": float(LEFT_HAND_ANGLES[finger]["mcp"]),
            "pip": float(LEFT_HAND_ANGLES[finger]["pip"]),
            "dip": float(LEFT_HAND_ANGLES[finger]["dip"]),
        }
    left_angles_dict["wrist"] = {
        "pitch": float(LEFT_HAND_WRIST["pitch"]),
        "yaw": float(LEFT_HAND_WRIST["yaw"]),
        "roll": float(LEFT_HAND_WRIST["roll"]),
    }

    # Generate landmarks via forward kinematics
    right_landmarks = generate_hand_landmarks(right_angles)
    left_landmarks = generate_hand_landmarks(LEFT_HAND_ANGLES)

    # Generate tolerances
    right_angle_tol, right_pos_tol = generate_default_tolerances()
    left_angle_tol, left_pos_tol = generate_default_tolerances()

    pose = {
        "right_hand_landmarks": right_landmarks,
        "right_hand_position_tolerances": right_pos_tol,
        "right_hand_angle_tolerances": right_angle_tol,
        "right_hand_angles": right_angles_dict,
        "left_hand_landmarks": left_landmarks,
        "left_hand_position_tolerances": left_pos_tol,
        "left_hand_angle_tolerances": left_angle_tol,
        "left_hand_angles": left_angles_dict,
    }

    return pose


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Generate BSL Alphabet Dominant Hand Poses")
    print("=" * 60)

    # Load dictionary
    print(f"\nLoading dictionary from {DICTIONARY_PATH}...")
    with open(DICTIONARY_PATH) as f:
        dictionary = json.load(f)

    entries = dictionary["entries"]
    now = datetime.now().isoformat()
    updated_count = 0
    skipped_c = False

    # Process each letter A-Z
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        sign_id = f"bsl_alphabet_{letter.lower()}"

        if sign_id not in entries:
            print(f"  WARNING: {sign_id} not found in dictionary, skipping")
            continue

        # Skip C — keep existing Kaggle data
        if letter == "C":
            skipped_c = True
            continue

        entry = entries[sign_id]
        pose = build_pose(letter)

        entry["poses"] = [pose]
        entry["sample_count"] = 1
        entry["quality_score"] = 0.7
        entry["source"] = "manual"
        entry["updated_at"] = now

        updated_count += 1

    print(f"\nUpdated {updated_count} letters (skipped C with Kaggle data)")

    # Update metadata
    total_signs = len(entries)
    signs_with_data = sum(
        1 for e in entries.values()
        if e.get("poses") and len(e["poses"]) > 0
    )
    signs_without_data = total_signs - signs_with_data
    coverage_percent = round(signs_with_data / total_signs * 100, 2)

    dictionary["metadata"] = {
        "total_signs": total_signs,
        "signs_with_data": signs_with_data,
        "signs_without_data": signs_without_data,
        "coverage_percent": coverage_percent,
    }
    dictionary["updated_at"] = now

    print(f"\nMetadata:")
    print(f"  Total signs: {total_signs}")
    print(f"  Signs with data: {signs_with_data}")
    print(f"  Signs without data: {signs_without_data}")
    print(f"  Coverage: {coverage_percent}%")

    # Write dictionary
    print(f"\nWriting {DICTIONARY_PATH}...")
    with open(DICTIONARY_PATH, "w") as f:
        json.dump(dictionary, f, indent=2)

    print(f"Writing {DICTIONARY_MIN_PATH}...")
    with open(DICTIONARY_MIN_PATH, "w") as f:
        json.dump(dictionary, f, separators=(",", ":"))

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Letter':<8} {'Finger pattern':<35} {'Source'}")
    print("-" * 60)
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        sign_id = f"bsl_alphabet_{letter.lower()}"
        entry = entries.get(sign_id, {})
        source = entry.get("source", "unknown")
        pattern = FINGER_PATTERNS.get(letter, "")

        if letter == "C":
            source_display = f"kaggle (unchanged, n={entry.get('sample_count', 0)})"
        else:
            source_display = source

        print(f"{letter:<8} {pattern:<35} {source_display}")

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    c_entry = entries.get("bsl_alphabet_c", {})
    c_sample = c_entry.get("sample_count", 0)
    c_source = c_entry.get("source", "")

    all_have_poses = all(
        len(entries.get(f"bsl_alphabet_{chr(c)}", {}).get("poses", [])) > 0
        for c in range(ord('a'), ord('z') + 1)
    )

    print(f"  All 26 letters have poses: {'YES' if all_have_poses else 'NO'}")
    print(f"  C retains Kaggle data: source={c_source}, sample_count={c_sample}")
    print(f"  Signs with data: {signs_with_data}")
    print(f"  Expected signs_with_data: 36")
    print(f"  Match: {'YES' if signs_with_data == 36 else 'NO'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
