#!/usr/bin/env python3
"""
Update Adaptive Angle Tolerances in BSL Dictionary

Replaces flat 20° tolerances with per-joint adaptive tolerances based on
minimum inter-sign distance. This improves discrimination between similar
signs (e.g., 4 vs 5, 1 vs 7).

Algorithm:
  1. For each sign+joint, find minimum distance to any other sign's same joint
  2. base_tolerance = max(8.0, min_distance / 2.5)
  3. Scale by detection reliability (thumb harder, MCP most reliable, DIP less)
  4. Clamp to [8.0, 30.0] degrees

Usage:
    python scripts/update_tolerances.py
    python scripts/update_tolerances.py --dictionary data/output/bsl-dictionary.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Fingers and joints
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
JOINTS = ["mcp", "pip", "dip"]

# Detection reliability scaling factors
RELIABILITY_SCALE = {
    # Thumb joints are hardest to detect accurately
    ("thumb", "mcp"): 1.5,
    ("thumb", "pip"): 1.5,
    ("thumb", "dip"): 1.5,
    # Non-thumb MCP: most reliable
    ("index", "mcp"): 0.85,
    ("middle", "mcp"): 0.85,
    ("ring", "mcp"): 0.85,
    ("pinky", "mcp"): 0.85,
    # Non-thumb PIP: baseline
    ("index", "pip"): 1.0,
    ("middle", "pip"): 1.0,
    ("ring", "pip"): 1.0,
    ("pinky", "pip"): 1.0,
    # Non-thumb DIP: fingertips less reliable
    ("index", "dip"): 1.25,
    ("middle", "dip"): 1.25,
    ("ring", "dip"): 1.25,
    ("pinky", "dip"): 1.25,
}

# Tolerance bounds
FLOOR = 8.0
CEILING = 30.0
DISTANCE_DIVISOR = 2.5


def extract_signs_with_angles(dictionary: dict) -> dict[str, dict]:
    """Extract sign_id -> {finger -> {joint -> angle}} for signs with pose data."""
    signs = {}
    for sign_id, entry in dictionary["entries"].items():
        poses = entry.get("poses", [])
        if not poses:
            continue
        angles = poses[0].get("right_hand_angles")
        if not angles:
            continue
        signs[sign_id] = angles
    return signs


def compute_adaptive_tolerances(
    signs: dict[str, dict],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-sign, per-joint adaptive tolerances."""
    sign_ids = sorted(signs.keys())
    tolerances = {}

    for sign_id in sign_ids:
        tols = {}
        for finger in FINGERS:
            tols[finger] = {}
            for joint in JOINTS:
                this_val = signs[sign_id][finger][joint]

                # Find minimum distance to any other sign for this joint
                min_dist = float("inf")
                for other_id in sign_ids:
                    if other_id == sign_id:
                        continue
                    other_val = signs[other_id][finger][joint]
                    dist = abs(this_val - other_val)
                    if dist < min_dist:
                        min_dist = dist

                # Base tolerance: half the gap (divided by 2.5) with floor
                base_tol = max(FLOOR, min_dist / DISTANCE_DIVISOR)

                # Apply detection reliability scaling
                scale = RELIABILITY_SCALE[(finger, joint)]
                scaled_tol = base_tol * scale

                # Clamp
                clamped_tol = max(FLOOR, min(CEILING, scaled_tol))

                tols[finger][joint] = round(clamped_tol, 2)

        tolerances[sign_id] = tols

    return tolerances


def check_discrimination(
    signs: dict[str, dict],
    tolerances: dict[str, dict[str, dict[str, float]]],
) -> list[dict]:
    """Check whether each pair of signs can be distinguished.

    A pair is distinguishable if at least 3 joints have tolerance < difference.
    """
    sign_ids = sorted(signs.keys())
    results = []

    for i, id_a in enumerate(sign_ids):
        for id_b in sign_ids[i + 1 :]:
            distinguishing_joints = 0
            total_diff = 0.0
            joint_details = []

            for finger in FINGERS:
                for joint in JOINTS:
                    val_a = signs[id_a][finger][joint]
                    val_b = signs[id_b][finger][joint]
                    diff = abs(val_a - val_b)
                    tol_a = tolerances[id_a][finger][joint]
                    tol_b = tolerances[id_b][finger][joint]
                    max_tol = max(tol_a, tol_b)
                    total_diff += diff

                    if diff > max_tol:
                        distinguishing_joints += 1
                        joint_details.append(
                            f"    {finger}.{joint}: diff={diff:.1f}° > tol={max_tol:.1f}°"
                        )

            results.append({
                "pair": f"{id_a} vs {id_b}",
                "distinguishing_joints": distinguishing_joints,
                "total_diff": total_diff,
                "distinguishable": distinguishing_joints >= 3,
                "details": joint_details,
            })

    return results


def print_summary(
    signs: dict[str, dict],
    tolerances: dict[str, dict[str, dict[str, float]]],
    disc_results: list[dict],
) -> None:
    """Print a summary of the tolerance changes and discrimination analysis."""
    print("\n" + "=" * 70)
    print("ADAPTIVE TOLERANCE SUMMARY")
    print("=" * 70)

    # Per-sign summary
    all_tols = []
    for sign_id in sorted(tolerances.keys()):
        tols = tolerances[sign_id]
        sign_tol_values = []
        for finger in FINGERS:
            for joint in JOINTS:
                sign_tol_values.append(tols[finger][joint])
                all_tols.append(tols[finger][joint])

        print(f"\n{sign_id}:")
        print(f"  Old: flat 20.0° for all 15 joints")
        print(f"  New: min={min(sign_tol_values):.1f}°  max={max(sign_tol_values):.1f}°  "
              f"avg={sum(sign_tol_values)/len(sign_tol_values):.1f}°")
        for finger in FINGERS:
            vals = [f"{tols[finger][j]:.1f}" for j in JOINTS]
            print(f"    {finger:7s}: mcp={vals[0]:>5s}  pip={vals[1]:>5s}  dip={vals[2]:>5s}")

    # Global stats
    print(f"\n{'=' * 70}")
    print(f"GLOBAL STATISTICS")
    print(f"{'=' * 70}")
    print(f"  Total joints: {len(all_tols)}")
    print(f"  Min tolerance: {min(all_tols):.1f}°")
    print(f"  Max tolerance: {max(all_tols):.1f}°")
    print(f"  Average tolerance: {sum(all_tols)/len(all_tols):.1f}°")

    # Discrimination results
    print(f"\n{'=' * 70}")
    print(f"DISCRIMINATION CHECK")
    print(f"{'=' * 70}")

    problem_pairs = []
    for r in disc_results:
        status = "OK" if r["distinguishable"] else "WARN"
        marker = "  " if r["distinguishable"] else ">>"
        print(f"{marker} [{status}] {r['pair']}: "
              f"{r['distinguishing_joints']} distinguishing joints, "
              f"total diff={r['total_diff']:.1f}°")
        if not r["distinguishable"]:
            problem_pairs.append(r)

    if problem_pairs:
        print(f"\n  WARNING: {len(problem_pairs)} pair(s) may be hard to distinguish:")
        for r in problem_pairs:
            print(f"    {r['pair']} ({r['distinguishing_joints']} joints)")
            for detail in r["details"][:5]:
                print(detail)
    else:
        print(f"\n  All {len(disc_results)} pairs are distinguishable (>= 3 joints).")


def update_dictionary(
    dictionary_path: Path,
    tolerances: dict[str, dict[str, dict[str, float]]],
) -> dict:
    """Write computed tolerances back into the dictionary."""
    with open(dictionary_path) as f:
        dictionary = json.load(f)

    updated = 0
    for sign_id, tols in tolerances.items():
        entry = dictionary["entries"].get(sign_id)
        if not entry or not entry.get("poses"):
            continue
        entry["poses"][0]["right_hand_angle_tolerances"] = tols
        entry["updated_at"] = datetime.now().isoformat()
        updated += 1

    dictionary["updated_at"] = datetime.now().isoformat()

    # Write full dictionary
    with open(dictionary_path, "w") as f:
        json.dump(dictionary, f, indent=2)
    print(f"\nUpdated {updated} signs in {dictionary_path}")

    # Write minified version
    min_path = dictionary_path.parent / "bsl-dictionary.min.json"
    with open(min_path, "w") as f:
        json.dump(dictionary, f, separators=(",", ":"))
    print(f"Regenerated minified: {min_path}")

    return dictionary


def main(dictionary_path: Optional[str] = None) -> None:
    dict_path = Path(dictionary_path or "data/output/bsl-dictionary.json")

    if not dict_path.exists():
        print(f"Error: Dictionary not found at {dict_path}")
        sys.exit(1)

    print(f"Loading dictionary: {dict_path}")
    with open(dict_path) as f:
        dictionary = json.load(f)

    # Extract signs with angle data
    signs = extract_signs_with_angles(dictionary)
    print(f"Found {len(signs)} signs with angle data: {', '.join(sorted(signs.keys()))}")

    if len(signs) < 2:
        print("Error: Need at least 2 signs with angle data to compute adaptive tolerances.")
        sys.exit(1)

    # Compute adaptive tolerances
    tolerances = compute_adaptive_tolerances(signs)

    # Check discrimination
    disc_results = check_discrimination(signs, tolerances)

    # Print summary
    print_summary(signs, tolerances, disc_results)

    # Update dictionary
    update_dictionary(dict_path, tolerances)

    print("\nDone! Remember to copy to learning app:")
    print("  cp data/output/bsl-dictionary.json ../bsl-learning-app/src/data/bsl-dictionary.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update BSL dictionary with adaptive per-joint angle tolerances"
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="data/output/bsl-dictionary.json",
        help="Path to bsl-dictionary.json",
    )
    args = parser.parse_args()
    main(args.dictionary)
