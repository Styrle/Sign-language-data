#!/usr/bin/env python3
"""
Spot-check the BSL dictionary for data quality issues.

Checks:
- Summary table of all signs
- Left hand palm verification for two-handed signs
- Angle plausibility per letter
- Duplicate/similar pose detection
"""

import json
from pathlib import Path

DICTIONARY_PATH = Path("data/output/bsl-dictionary.json")

# Expected finger states per letter (index.pip threshold)
# "extended" = pip < 20, "curled" = pip > 100
LETTER_CHECKS = {
    "A": {"index_extended": True, "middle_curled": True},
    "B": {"index_extended": True, "middle_curled": True},
    "D": {"index_bent": True, "middle_curled": True},
    "F": {"index_extended": True, "middle_extended": True, "ring_curled": True},
    "G": {"index_extended": True, "middle_curled": True},
    "H": {"index_extended": True, "middle_extended": True},
    "I": {"pinky_extended": True, "index_curled": True},
    "J": {"pinky_extended": True, "index_curled": True},
    "K": {"index_extended": True, "middle_extended": True},
    "L": {"thumb_extended": True, "index_extended": True, "middle_curled": True},
    "M": {"index_extended": True, "middle_extended": True, "ring_extended": True},
    "N": {"index_extended": True, "middle_extended": True, "ring_curled": True},
    "S": {"index_curled": True, "middle_curled": True},
    "U": {"index_extended": True, "middle_extended": True},
    "V": {"index_extended": True, "middle_extended": True},
    "W": {"index_extended": True, "middle_extended": True, "ring_extended": True},
    "Y": {"thumb_extended": True, "pinky_extended": True, "index_curled": True},
}


def get_angles(pose, hand="right"):
    """Extract angles dict from a pose."""
    key = f"{hand}_hand_angles"
    return pose.get(key, {})


def is_extended(pip_angle):
    """Check if a finger is extended (pip < 20)."""
    return pip_angle < 20


def is_curled(pip_angle):
    """Check if a finger is curled (pip > 100)."""
    return pip_angle > 100


def is_bent(pip_angle):
    """Check if a finger is bent (40 < pip < 100)."""
    return 40 < pip_angle < 100


def check_finger_state(angles, finger, state_type):
    """Check if a finger matches expected state."""
    finger_angles = angles.get(finger, {})
    pip = finger_angles.get("pip", 0)

    if state_type == "extended":
        return is_extended(pip)
    elif state_type == "curled":
        return is_curled(pip)
    elif state_type == "bent":
        return is_bent(pip)
    return True


def total_angle_diff(angles1, angles2):
    """Calculate total angle difference between two poses."""
    total = 0
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        for joint in ["mcp", "pip", "dip"]:
            a1 = angles1.get(finger, {}).get(joint, 0)
            a2 = angles2.get(finger, {}).get(joint, 0)
            total += abs(a1 - a2)
    return total


def main():
    print("=" * 80)
    print("BSL Dictionary Spot Check")
    print("=" * 80)

    with open(DICTIONARY_PATH) as f:
        dictionary = json.load(f)

    entries = dictionary["entries"]

    # Summary table
    print(f"\n{'Sign':<12} {'Category':<12} {'Source':<10} {'Hands':<6} {'R-Angles':<10} {'L-Angles':<10} {'Samples':<8}")
    print("-" * 80)

    issues = []
    all_right_angles = {}

    for sign_id, entry in sorted(entries.items()):
        defn = entry["definition"]
        name = defn["name"]
        category = defn["category"]
        source = entry.get("source", "?")
        poses = entry.get("poses", [])
        sample_count = entry.get("sample_count", 0)

        has_right_angles = False
        has_left_angles = False
        num_hands = 0

        if poses:
            pose = poses[0]
            has_right_angles = bool(pose.get("right_hand_angles"))
            has_left_angles = bool(pose.get("left_hand_angles"))
            if pose.get("right_hand_landmarks"):
                num_hands += 1
            if pose.get("left_hand_landmarks"):
                num_hands += 1

            if has_right_angles:
                all_right_angles[name] = pose["right_hand_angles"]

        r_mark = "Y" if has_right_angles else "-"
        l_mark = "Y" if has_left_angles else "-"

        print(f"{name:<12} {category:<12} {source:<10} {num_hands:<6} {r_mark:<10} {l_mark:<10} {sample_count:<8}")

    # Check left hand is open palm for two-handed signs
    print(f"\n{'='*80}")
    print("Left Hand Palm Check (two-handed signs)")
    print("-" * 80)

    palm_issues = 0
    for sign_id, entry in sorted(entries.items()):
        defn = entry["definition"]
        if not defn.get("two_handed"):
            continue

        poses = entry.get("poses", [])
        if not poses:
            continue

        pose = poses[0]
        left_angles = pose.get("left_hand_angles", {})
        if not left_angles:
            continue

        for finger in ["index", "middle", "ring", "pinky"]:
            pip = left_angles.get(finger, {}).get("pip", 0)
            if pip > 40:
                print(f"  WARN: {defn['name']} — left {finger}.pip = {pip:.0f} (expected < 20 for flat palm)")
                palm_issues += 1

    if palm_issues == 0:
        print("  PASS: All left hands look like open palms")

    # Check finger plausibility per letter
    print(f"\n{'='*80}")
    print("Finger State Plausibility (alphabet)")
    print("-" * 80)

    plausibility_fails = 0
    for letter, checks in sorted(LETTER_CHECKS.items()):
        angles = all_right_angles.get(letter, {})
        if not angles:
            continue

        for check_key, expected in checks.items():
            parts = check_key.split("_")
            finger = parts[0]
            state_type = parts[1]

            result = check_finger_state(angles, finger, state_type)
            if not result:
                pip = angles.get(finger, {}).get("pip", 0)
                print(f"  FAIL: {letter} — {finger} should be {state_type} but pip={pip:.0f}")
                plausibility_fails += 1

    if plausibility_fails == 0:
        print("  PASS: All letter finger states are plausible")

    # Check for duplicate/similar poses
    print(f"\n{'='*80}")
    print("Similar Pose Detection (total angle diff < 50)")
    print("-" * 80)

    similar_count = 0
    letters = sorted(all_right_angles.keys())
    for i, l1 in enumerate(letters):
        for l2 in letters[i+1:]:
            diff = total_angle_diff(all_right_angles[l1], all_right_angles[l2])
            if diff < 50:
                print(f"  WARN: {l1} and {l2} are very similar (total diff = {diff:.0f})")
                similar_count += 1

    if similar_count == 0:
        print("  PASS: No suspiciously similar poses found")

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print("-" * 80)
    total_issues = palm_issues + plausibility_fails
    print(f"  Palm issues: {palm_issues}")
    print(f"  Plausibility fails: {plausibility_fails}")
    print(f"  Similar poses: {similar_count}")
    print(f"  Critical issues: {total_issues}")
    print(f"  Result: {'PASS' if total_issues == 0 else 'NEEDS REVIEW'}")


if __name__ == "__main__":
    main()
