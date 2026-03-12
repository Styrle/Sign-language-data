#!/usr/bin/env python3
"""
Update Dictionary with Joint Angles and Tolerances

Loads Kaggle data, processes landmarks, calculates joint angles,
calculates tolerances from variance, and updates the BSL dictionary.
"""

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.landmarks import LANDMARK_NAMES, NUM_LANDMARKS
from src.angles import calculate_all_joint_angles
from src.types import FingerAngles, JointAngles


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ToleranceConfig:
    """Configuration for tolerance calculation."""
    # Multiplier for position tolerance (position_tolerance = std_dev * multiplier)
    position_multiplier: float = 1.5

    # Default position tolerance when not enough samples
    default_position_tolerance: float = 0.08

    # Default angle tolerance in degrees
    default_angle_tolerance: float = 20.0

    # Minimum samples needed for variance calculation
    min_samples_for_variance: int = 5

    # Clamp tolerances to reasonable range
    min_position_tolerance: float = 0.02
    max_position_tolerance: float = 0.25


def angles_to_nested_dict(angles: JointAngles) -> dict:
    """Convert JointAngles to nested dictionary matching TypeScript types."""
    result = {
        "thumb": {
            "mcp": angles.thumb.mcp,
            "pip": angles.thumb.pip,
            "dip": angles.thumb.dip,
        },
        "index": {
            "mcp": angles.index.mcp,
            "pip": angles.index.pip,
            "dip": angles.index.dip,
        },
        "middle": {
            "mcp": angles.middle.mcp,
            "pip": angles.middle.pip,
            "dip": angles.middle.dip,
        },
        "ring": {
            "mcp": angles.ring.mcp,
            "pip": angles.ring.pip,
            "dip": angles.ring.dip,
        },
        "pinky": {
            "mcp": angles.pinky.mcp,
            "pip": angles.pinky.pip,
            "dip": angles.pinky.dip,
        },
    }

    # Add wrist angles if available
    if angles.wrist_rotation is not None or angles.wrist_flexion is not None:
        result["wrist"] = {
            "pitch": angles.wrist_flexion or 0.0,
            "yaw": angles.wrist_rotation or 0.0,
            "roll": 0.0,
        }

    return result


def load_kaggle_csv(csv_path: Path) -> list[dict]:
    """Load Kaggle CSV and return list of records with landmarks and sign labels."""
    records = []

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 64:
                continue

            # Extract landmarks (21 points × 3 coordinates = 63 values)
            landmarks = []
            for i in range(NUM_LANDMARKS):
                idx = i * 3
                try:
                    x = float(row[idx])
                    y = float(row[idx + 1])
                    z = float(row[idx + 2])
                    landmarks.append({"x": x, "y": y, "z": z})
                except (ValueError, IndexError):
                    break

            if len(landmarks) != NUM_LANDMARKS:
                continue

            # Last column is the sign label
            sign_label = row[63].strip() if len(row) > 63 else ""

            records.append({
                "landmarks": landmarks,
                "sign_label": sign_label,
            })

    return records


def normalize_sign_id(label: str) -> str:
    """Convert Kaggle sign label to our sign ID format."""
    # Examples: "C - c" -> "bsl_alphabet_c", "Nine - 9" -> "bsl_numbers_9"
    label = label.strip()

    # Map word numbers to digits
    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }

    # Handle format like "C - c" or "Nine - 9"
    if " - " in label:
        parts = label.split(" - ")
        first_part = parts[0].strip().lower()
        second_part = parts[1].strip() if len(parts) > 1 else ""

        # Check if it's a number (second part is digit)
        if second_part.isdigit():
            return f"bsl_numbers_{second_part}"

        # Check if first part is a word number
        if first_part in word_to_num:
            return f"bsl_numbers_{word_to_num[first_part]}"

        # Otherwise it's a letter
        return f"bsl_alphabet_{first_part}"

    # Handle single letters
    label_lower = label.lower()
    if len(label_lower) == 1 and label_lower.isalpha():
        return f"bsl_alphabet_{label_lower}"

    # Handle numeric digits
    if label.isdigit():
        return f"bsl_numbers_{label}"

    # Handle word numbers (nine, zero, etc.)
    if label_lower in word_to_num:
        return f"bsl_numbers_{word_to_num[label_lower]}"

    # Handle word labels
    label_clean = label_lower.replace(" ", "_").replace("-", "_")
    return f"bsl_{label_clean}"


def landmarks_to_numpy(landmarks: list[dict]) -> np.ndarray:
    """Convert landmark list to numpy array."""
    return np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize landmarks to canonical space."""
    # Translate to wrist origin
    wrist = landmarks[0].copy()
    translated = landmarks - wrist

    # Scale by palm width (index MCP to pinky MCP)
    index_mcp = translated[5]
    pinky_mcp = translated[17]
    palm_width = np.linalg.norm(index_mcp - pinky_mcp)

    if palm_width > 1e-6:
        normalized = translated / palm_width
    else:
        normalized = translated

    return normalized


def calculate_angles_from_landmarks(landmarks: list[dict]) -> Optional[dict]:
    """Calculate joint angles from landmark list."""
    try:
        arr = landmarks_to_numpy(landmarks)
        normalized = normalize_landmarks(arr)

        # Calculate angles
        angles = calculate_all_joint_angles(normalized)

        # Convert to dict
        return angles_to_dict(angles)
    except Exception as e:
        print(f"  Warning: Failed to calculate angles: {e}")
        return None


def calculate_position_tolerances(
    all_landmarks: np.ndarray,
    config: ToleranceConfig,
) -> dict[str, float]:
    """
    Calculate position tolerances from landmark variance.

    Args:
        all_landmarks: Array of shape (num_samples, 21, 3)
        config: Tolerance configuration

    Returns:
        Dictionary mapping landmark names to tolerance values
    """
    num_samples = len(all_landmarks)
    tolerances = {}

    if num_samples >= config.min_samples_for_variance:
        # Calculate standard deviation across samples for each landmark
        std = np.std(all_landmarks, axis=0)  # Shape: (21, 3)

        for i, name in enumerate(LANDMARK_NAMES):
            # Use the mean of x, y, z std dev for this landmark
            landmark_std = float(np.mean(std[i]))

            # Apply multiplier
            tol = landmark_std * config.position_multiplier

            # Clamp to reasonable range
            tol = max(config.min_position_tolerance, min(config.max_position_tolerance, tol))
            tolerances[name] = tol
    else:
        # Not enough samples - use default
        for name in LANDMARK_NAMES:
            tolerances[name] = config.default_position_tolerance

    return tolerances


def calculate_angle_tolerances(
    all_landmarks: np.ndarray,
    config: ToleranceConfig,
) -> dict[str, dict[str, float]]:
    """
    Calculate angle tolerances.

    For now, uses default values. Can be extended to calculate from variance.

    Args:
        all_landmarks: Array of shape (num_samples, 21, 3)
        config: Tolerance configuration

    Returns:
        Dictionary with finger -> joint -> tolerance mapping
    """
    # Use default angle tolerance for all joints
    default_tol = config.default_angle_tolerance

    return {
        "thumb": {"mcp": default_tol, "pip": default_tol, "dip": default_tol},
        "index": {"mcp": default_tol, "pip": default_tol, "dip": default_tol},
        "middle": {"mcp": default_tol, "pip": default_tol, "dip": default_tol},
        "ring": {"mcp": default_tol, "pip": default_tol, "dip": default_tol},
        "pinky": {"mcp": default_tol, "pip": default_tol, "dip": default_tol},
    }


def process_kaggle_data(
    kaggle_dir: Path,
    min_samples: int = 3,
    tolerance_config: Optional[ToleranceConfig] = None,
) -> dict[str, dict]:
    """Process all Kaggle data and return sign data with angles and tolerances."""
    if tolerance_config is None:
        tolerance_config = ToleranceConfig()

    sign_data: dict[str, list[dict]] = {}

    # Process all CSV files
    for csv_file in kaggle_dir.glob("*.csv"):
        print(f"Loading {csv_file.name}...")
        records = load_kaggle_csv(csv_file)
        print(f"  Loaded {len(records)} records")

        for record in records:
            sign_id = normalize_sign_id(record["sign_label"])
            if sign_id not in sign_data:
                sign_data[sign_id] = []
            sign_data[sign_id].append(record["landmarks"])

    # Process each sign
    processed: dict[str, dict] = {}

    for sign_id, samples in sign_data.items():
        if len(samples) < min_samples:
            continue

        print(f"Processing {sign_id} ({len(samples)} samples)...")

        # Convert all samples to numpy and normalize
        all_landmarks = []
        for sample in samples:
            arr = landmarks_to_numpy(sample)
            normalized = normalize_landmarks(arr)
            all_landmarks.append(normalized)

        all_landmarks = np.array(all_landmarks)

        # Average the landmarks to get canonical pose
        canonical = np.mean(all_landmarks, axis=0)

        # Calculate position tolerances from variance
        position_tolerances = calculate_position_tolerances(all_landmarks, tolerance_config)

        # Calculate angle tolerances
        angle_tolerances = calculate_angle_tolerances(all_landmarks, tolerance_config)

        # Log tolerance stats
        tol_values = list(position_tolerances.values())
        print(f"  Position tolerances: min={min(tol_values):.4f}, max={max(tol_values):.4f}, "
              f"mean={np.mean(tol_values):.4f}")

        # Calculate angles from canonical pose
        try:
            angles = calculate_all_joint_angles(canonical)
            angles_dict = angles_to_nested_dict(angles)
        except Exception as e:
            print(f"  Warning: Angle calculation failed: {e}")
            angles_dict = None

        # Convert canonical landmarks to list format
        canonical_list = [
            {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])}
            for pt in canonical
        ]

        processed[sign_id] = {
            "landmarks": canonical_list,
            "position_tolerances": position_tolerances,
            "angle_tolerances": angle_tolerances,
            "angles": angles_dict,
            "sample_count": len(samples),
        }

    return processed


def update_dictionary(
    dictionary_path: Path,
    processed_data: dict[str, dict],
    output_path: Optional[Path] = None,
) -> dict:
    """Update dictionary with processed landmark and angle data."""
    # Load existing dictionary
    with open(dictionary_path) as f:
        dictionary = json.load(f)

    updated_count = 0
    angles_count = 0
    tolerances_count = 0

    for sign_id, data in processed_data.items():
        if sign_id in dictionary["entries"]:
            entry = dictionary["entries"][sign_id]

            # Create pose with landmarks, tolerances, and angles
            pose = {
                "right_hand_landmarks": data["landmarks"],
                "right_hand_position_tolerances": data["position_tolerances"],
                "right_hand_angle_tolerances": data["angle_tolerances"],
            }

            if data["angles"]:
                pose["right_hand_angles"] = data["angles"]
                angles_count += 1

            tolerances_count += 1

            entry["poses"] = [pose]
            entry["sample_count"] = data["sample_count"]
            entry["quality_score"] = min(1.0, data["sample_count"] / 10.0)
            entry["source"] = "kaggle"
            entry["updated_at"] = datetime.now().isoformat()

            updated_count += 1

    # Update dictionary metadata
    dictionary["updated_at"] = datetime.now().isoformat()

    # Save updated dictionary
    out_path = output_path or dictionary_path
    with open(out_path, "w") as f:
        json.dump(dictionary, f, indent=2)

    print(f"\nUpdated {updated_count} signs:")
    print(f"  - {angles_count} with angles")
    print(f"  - {tolerances_count} with tolerances")
    print(f"Saved to {out_path}")

    return dictionary


def generate_typescript_types(dictionary: dict, output_path: Path) -> None:
    """Generate TypeScript type definitions."""
    types_content = '''// Auto-generated BSL Dictionary Types
// Generated: {timestamp}

export interface Point3D {{
  x: number;
  y: number;
  z: number;
}}

export interface FingerAngles {{
  mcp: number;
  pip: number;
  dip: number;
}}

export interface JointAngles {{
  thumb: FingerAngles;
  index: FingerAngles;
  middle: FingerAngles;
  ring: FingerAngles;
  pinky: FingerAngles;
  wrist?: {{
    pitch: number;
    yaw: number;
    roll: number;
  }};
}}

export interface FingerAngleTolerances {{
  mcp: number;
  pip: number;
  dip: number;
}}

export interface AngleTolerances {{
  thumb: FingerAngleTolerances;
  index: FingerAngleTolerances;
  middle: FingerAngleTolerances;
  ring: FingerAngleTolerances;
  pinky: FingerAngleTolerances;
}}

export interface SignPose {{
  // Landmarks (21 points per hand)
  left_hand_landmarks?: Point3D[];
  right_hand_landmarks?: Point3D[];

  // Joint angles
  left_hand_angles?: JointAngles;
  right_hand_angles?: JointAngles;

  // Position tolerances (per landmark, for matching algorithm)
  left_hand_position_tolerances?: Record<string, number>;
  right_hand_position_tolerances?: Record<string, number>;

  // Angle tolerances (per joint, in degrees)
  left_hand_angle_tolerances?: AngleTolerances;
  right_hand_angle_tolerances?: AngleTolerances;
}}

export type SignCategory =
  | "alphabet"
  | "numbers"
  | "greetings"
  | "questions"
  | "common_phrases"
  | "actions"
  | "objects"
  | "people"
  | "places"
  | "time"
  | "colors"
  | "emotions"
  | "food_drink"
  | "animals"
  | "weather"
  | "other";

export interface SignDefinition {{
  id: string;
  name: string;
  category: SignCategory;
  difficulty: 1 | 2 | 3 | 4 | 5;
  description: string;
  two_handed: boolean;
}}

export interface SignDictionaryEntry {{
  definition: SignDefinition;
  poses: SignPose[];
  sample_count: number;
  quality_score: number;
  source: "kaggle" | "recorded" | "processed" | "definition";
  created_at: string;
  updated_at: string;
}}

export interface DictionaryMetadata {{
  total_signs: number;
  signs_with_data: number;
  signs_without_data: number;
  coverage_percent: number;
}}

export interface SignDictionary {{
  version: string;
  created_at: string;
  updated_at: string;
  entries: Record<string, SignDictionaryEntry>;
  metadata?: DictionaryMetadata;
}}

// Sign IDs
export type SignId = {sign_ids};

// Dictionary version
export const DICTIONARY_VERSION = "{version}";
'''.format(
        timestamp=datetime.now().isoformat(),
        version=dictionary.get("version", "1.0.0"),
        sign_ids=" | ".join(f'"{s}"' for s in sorted(dictionary.get("entries", {}).keys())),
    )

    with open(output_path, "w") as f:
        f.write(types_content)

    print(f"Generated TypeScript types: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Update dictionary with joint angles and tolerances from Kaggle data"
    )
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=Path("data/kaggle-bsl"),
        help="Kaggle data directory"
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=Path("data/output/bsl-dictionary.json"),
        help="Dictionary file to update"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum samples per sign"
    )
    parser.add_argument(
        "--position-multiplier",
        type=float,
        default=1.5,
        help="Multiplier for position tolerance (std_dev * multiplier)"
    )
    parser.add_argument(
        "--default-position-tolerance",
        type=float,
        default=0.08,
        help="Default position tolerance when not enough samples"
    )
    parser.add_argument(
        "--default-angle-tolerance",
        type=float,
        default=20.0,
        help="Default angle tolerance in degrees"
    )

    args = parser.parse_args()

    # Create tolerance config
    tolerance_config = ToleranceConfig(
        position_multiplier=args.position_multiplier,
        default_position_tolerance=args.default_position_tolerance,
        default_angle_tolerance=args.default_angle_tolerance,
    )

    print("=" * 50)
    print("Update Dictionary with Joint Angles and Tolerances")
    print("=" * 50)

    print(f"\nTolerance Configuration:")
    print(f"  Position multiplier: {tolerance_config.position_multiplier}")
    print(f"  Default position tolerance: {tolerance_config.default_position_tolerance}")
    print(f"  Default angle tolerance: {tolerance_config.default_angle_tolerance} deg")

    # Process Kaggle data
    print("\n1. Processing Kaggle data...")
    processed = process_kaggle_data(args.kaggle_dir, args.min_samples, tolerance_config)
    print(f"   Processed {len(processed)} signs")

    # Update dictionary
    print("\n2. Updating dictionary...")
    dictionary = update_dictionary(args.dictionary, processed)

    # Generate minified version
    print("\n3. Generating minified dictionary...")
    min_path = args.dictionary.parent / "bsl-dictionary.min.json"
    with open(min_path, "w") as f:
        json.dump(dictionary, f, separators=(",", ":"))
    print(f"   Saved to {min_path}")

    # Generate TypeScript types
    print("\n4. Generating TypeScript types...")
    ts_path = args.dictionary.parent / "bsl-dictionary.d.ts"
    generate_typescript_types(dictionary, ts_path)

    # Verify results
    print("\n" + "=" * 50)
    print("Verification")
    print("=" * 50)

    entries = dictionary.get("entries", {})
    with_landmarks = 0
    with_angles = 0
    with_pos_tolerances = 0
    with_angle_tolerances = 0

    all_pos_tolerances = []

    for sign_id, entry in entries.items():
        poses = entry.get("poses", [])
        if poses:
            pose = poses[0]
            if pose.get("right_hand_landmarks"):
                with_landmarks += 1
            if pose.get("right_hand_angles"):
                with_angles += 1
            if pose.get("right_hand_position_tolerances"):
                with_pos_tolerances += 1
                all_pos_tolerances.extend(pose["right_hand_position_tolerances"].values())
            if pose.get("right_hand_angle_tolerances"):
                with_angle_tolerances += 1

    print(f"Total entries: {len(entries)}")
    print(f"Entries with landmarks: {with_landmarks}")
    print(f"Entries with joint angles: {with_angles}")
    print(f"Entries with position tolerances: {with_pos_tolerances}")
    print(f"Entries with angle tolerances: {with_angle_tolerances}")

    if all_pos_tolerances:
        print(f"\nPosition Tolerance Statistics:")
        print(f"  Min: {min(all_pos_tolerances):.4f}")
        print(f"  Max: {max(all_pos_tolerances):.4f}")
        print(f"  Mean: {np.mean(all_pos_tolerances):.4f}")
        print(f"  Std: {np.std(all_pos_tolerances):.4f}")

    # Show sample entry
    for sign_id, entry in entries.items():
        poses = entry.get("poses", [])
        if poses and poses[0].get("right_hand_angles"):
            print(f"\nSample entry: {sign_id}")

            pose = poses[0]

            # Show angles
            angles = pose["right_hand_angles"]
            print(f"  Joint Angles:")
            print(f"    Thumb MCP: {float(angles.get('thumb', {}).get('mcp', 0)):.1f} deg")
            print(f"    Index MCP: {float(angles.get('index', {}).get('mcp', 0)):.1f} deg")

            # Show position tolerances
            pos_tols = pose.get("right_hand_position_tolerances", {})
            if pos_tols:
                print(f"  Position Tolerances (sample):")
                for name in ["WRIST", "INDEX_FINGER_TIP", "THUMB_TIP"]:
                    if name in pos_tols:
                        print(f"    {name}: {pos_tols[name]:.4f}")

            # Show angle tolerances
            angle_tols = pose.get("right_hand_angle_tolerances", {})
            if angle_tols:
                print(f"  Angle Tolerances:")
                print(f"    Index MCP: {angle_tols.get('index', {}).get('mcp', 'N/A')} deg")

            break

    print("\nDone!")


if __name__ == "__main__":
    main()
