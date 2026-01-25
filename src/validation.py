"""BSL Data Extraction - Validation module.

This module provides comprehensive validation for landmark data, recordings,
and sign dictionary entries. It includes anatomical checks, quality scoring,
and batch validation with reporting.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from src.landmarks import (
    FINGER_INDICES,
    FINGER_JOINTS,
    LANDMARK_CONNECTIONS,
    NUM_LANDMARKS,
    THUMB_JOINTS,
    WRIST_INDEX,
)
from src.types import (
    DetectedHand,
    HandLandmarks,
    Point3D,
    Recording,
    RecordingFrame,
    SignDictionary,
    SignDictionaryEntry,
    SignPose,
    ValidationResult,
)


# =============================================================================
# Constants
# =============================================================================


@dataclass
class AnatomicalLimits:
    """Anatomical limits for hand validation."""

    # Coordinate bounds
    coord_min: float = 0.0
    coord_max: float = 1.0

    # Finger length ratios relative to hand size (palm width)
    # Note: These are lenient to accommodate various hand poses and camera angles
    min_finger_length_ratio: float = 0.2
    max_finger_length_ratio: float = 2.5

    # Joint angle limits (degrees)
    min_joint_angle: float = 0.0  # Fully extended
    max_joint_angle: float = 180.0  # Maximum bend
    hyperextension_limit: float = -15.0  # Allow slight hyperextension

    # Finger segment ratios (each segment relative to previous)
    min_segment_ratio: float = 0.5
    max_segment_ratio: float = 1.2

    # Maximum inter-frame movement (normalized units per frame)
    max_frame_movement: float = 0.15

    # Minimum detection confidence
    min_confidence: float = 0.5

    # Recording requirements
    min_frames: int = 1
    max_jitter_threshold: float = 0.05  # Max std dev for static poses


ANATOMICAL_LIMITS = AnatomicalLimits()


# Quality score weights
QUALITY_WEIGHTS = {
    "coordinate_validity": 0.25,
    "completeness": 0.20,
    "anatomical_plausibility": 0.25,
    "confidence": 0.15,
    "consistency": 0.15,
}


# =============================================================================
# Helper Functions
# =============================================================================


def _point_to_array(point: Point3D) -> np.ndarray:
    """Convert Point3D to numpy array."""
    return np.array([point.x, point.y, point.z])


def _landmarks_to_array(landmarks: HandLandmarks) -> np.ndarray:
    """Convert HandLandmarks to numpy array of shape (21, 3)."""
    return np.array([[lm.point.x, lm.point.y, lm.point.z] for lm in landmarks.landmarks])


def _calculate_distance(p1: Point3D, p2: Point3D) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle at p2 formed by p1-p2-p3 in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2

    # Handle zero-length vectors
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 < 1e-10 or len2 < 1e-10:
        return 180.0  # Assume straight if points overlap

    cos_angle = np.dot(v1, v2) / (len1 * len2)
    # Clamp to valid range for arccos
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


# =============================================================================
# Point Validation
# =============================================================================


def validate_point(point: Point3D, limits: AnatomicalLimits = ANATOMICAL_LIMITS) -> ValidationResult:
    """Validate a single 3D point.

    Checks:
    - x, y coordinates within valid range [0, 1]
    - No NaN or infinite values
    - z coordinate is finite (depth can exceed [0,1])

    Args:
        point: The Point3D to validate.
        limits: Anatomical limits to use.

    Returns:
        ValidationResult with validation status and score.
    """
    errors: list[str] = []
    warnings: list[str] = []
    score = 1.0

    # Check for NaN/Inf
    if math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z):
        errors.append("Point contains NaN values")
        score = 0.0
    elif math.isinf(point.x) or math.isinf(point.y) or math.isinf(point.z):
        errors.append("Point contains infinite values")
        score = 0.0
    else:
        # Check x, y bounds
        if point.x < limits.coord_min or point.x > limits.coord_max:
            if point.x < limits.coord_min - 0.1 or point.x > limits.coord_max + 0.1:
                errors.append(f"x coordinate {point.x:.4f} significantly out of range")
                score -= 0.3
            else:
                warnings.append(f"x coordinate {point.x:.4f} slightly out of range")
                score -= 0.1

        if point.y < limits.coord_min or point.y > limits.coord_max:
            if point.y < limits.coord_min - 0.1 or point.y > limits.coord_max + 0.1:
                errors.append(f"y coordinate {point.y:.4f} significantly out of range")
                score -= 0.3
            else:
                warnings.append(f"y coordinate {point.y:.4f} slightly out of range")
                score -= 0.1

    return ValidationResult(
        is_valid=len(errors) == 0,
        score=max(0.0, min(1.0, score)),
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# Landmark Validation
# =============================================================================


def validate_landmarks(
    landmarks: HandLandmarks,
    limits: AnatomicalLimits = ANATOMICAL_LIMITS,
) -> ValidationResult:
    """Validate a complete set of hand landmarks.

    Checks:
    - All 21 landmarks present
    - All coordinates in valid range
    - No NaN values
    - Anatomically plausible (finger lengths, joint angles)

    Args:
        landmarks: The HandLandmarks to validate.
        limits: Anatomical limits to use.

    Returns:
        ValidationResult with detailed validation status.
    """
    errors: list[str] = []
    warnings: list[str] = []
    score = 1.0

    # Check landmark count
    if len(landmarks.landmarks) != NUM_LANDMARKS:
        errors.append(f"Expected {NUM_LANDMARKS} landmarks, got {len(landmarks.landmarks)}")
        return ValidationResult(is_valid=False, score=0.0, errors=errors, warnings=warnings)

    # Validate each point
    point_scores: list[float] = []
    for lm in landmarks.landmarks:
        result = validate_point(lm.point, limits)
        point_scores.append(result.score)
        if not result.is_valid:
            errors.extend([f"Landmark {lm.index} ({lm.name}): {e}" for e in result.errors])
        warnings.extend([f"Landmark {lm.index} ({lm.name}): {w}" for w in result.warnings])

    avg_point_score = sum(point_scores) / len(point_scores) if point_scores else 0.0

    # Anatomical checks
    if not check_finger_lengths(landmarks, limits):
        warnings.append("Finger length proportions appear unusual")
        score -= 0.1

    if not check_joint_angles(landmarks, limits):
        warnings.append("Some joint angles appear anatomically implausible")
        score -= 0.1

    # Check for overlapping landmarks (all at same position)
    coords = _landmarks_to_array(landmarks)
    if np.std(coords[:, :2]) < 0.01:  # Very low variance in x,y
        errors.append("Landmarks appear collapsed to single point")
        score -= 0.5

    # Combine scores
    final_score = (score + avg_point_score) / 2

    return ValidationResult(
        is_valid=len(errors) == 0,
        score=max(0.0, min(1.0, final_score)),
        errors=errors,
        warnings=warnings,
    )


def check_finger_lengths(
    landmarks: HandLandmarks,
    limits: AnatomicalLimits = ANATOMICAL_LIMITS,
) -> bool:
    """Check if finger lengths are anatomically reasonable.

    Verifies that finger lengths are proportional to palm size.

    Args:
        landmarks: The HandLandmarks to check.
        limits: Anatomical limits to use.

    Returns:
        True if finger lengths are plausible.
    """
    try:
        lm_array = _landmarks_to_array(landmarks)

        # Calculate palm width (wrist to middle finger MCP)
        wrist = lm_array[WRIST_INDEX]
        middle_mcp = lm_array[9]  # Middle finger MCP
        palm_size = np.linalg.norm(middle_mcp - wrist)

        if palm_size < 0.01:  # Too small to measure
            return False

        # Check each finger length
        for finger_name, indices in FINGER_INDICES.items():
            if len(indices) < 2:
                continue

            # Total finger length (base to tip)
            base = lm_array[indices[0]]
            tip = lm_array[indices[-1]]
            finger_length = np.linalg.norm(tip - base)

            ratio = finger_length / palm_size
            if ratio < limits.min_finger_length_ratio or ratio > limits.max_finger_length_ratio:
                return False

        return True

    except (IndexError, ValueError):
        return False


def check_joint_angles(
    landmarks: HandLandmarks,
    limits: AnatomicalLimits = ANATOMICAL_LIMITS,
) -> bool:
    """Check if joint angles are anatomically possible.

    Verifies that no joints are bent beyond physical limits.

    Args:
        landmarks: The HandLandmarks to check.
        limits: Anatomical limits to use.

    Returns:
        True if all joint angles are plausible.
    """
    try:
        lm_array = _landmarks_to_array(landmarks)

        # Check thumb joints
        for joint_name, (i1, i2, i3) in THUMB_JOINTS.items():
            angle = _calculate_angle(lm_array[i1], lm_array[i2], lm_array[i3])
            if angle < limits.hyperextension_limit or angle > limits.max_joint_angle:
                return False

        # Check finger joints
        for finger_name, joints in FINGER_JOINTS.items():
            for joint_name, (i1, i2, i3) in joints.items():
                angle = _calculate_angle(lm_array[i1], lm_array[i2], lm_array[i3])
                if angle < limits.hyperextension_limit or angle > limits.max_joint_angle:
                    return False

        return True

    except (IndexError, ValueError):
        return False


# =============================================================================
# Recording Validation
# =============================================================================


def validate_frame(
    frame: RecordingFrame,
    limits: AnatomicalLimits = ANATOMICAL_LIMITS,
) -> ValidationResult:
    """Validate a single recording frame.

    Args:
        frame: The RecordingFrame to validate.
        limits: Anatomical limits to use.

    Returns:
        ValidationResult with frame validation status.
    """
    errors: list[str] = []
    warnings: list[str] = []
    score = 1.0

    # Check timestamp
    if frame.timestamp_ms < 0:
        errors.append(f"Invalid timestamp: {frame.timestamp_ms}")
        score -= 0.2

    # Check frame confidence
    if frame.frame_confidence < limits.min_confidence:
        warnings.append(f"Low frame confidence: {frame.frame_confidence:.2f}")
        score -= 0.1

    # Validate detected hands
    hands_validated = 0

    if frame.left_hand is not None:
        hand_result = _validate_detected_hand(frame.left_hand, "left", limits)
        if not hand_result.is_valid:
            errors.extend(hand_result.errors)
        warnings.extend(hand_result.warnings)
        score = (score + hand_result.score) / 2
        hands_validated += 1

    if frame.right_hand is not None:
        hand_result = _validate_detected_hand(frame.right_hand, "right", limits)
        if not hand_result.is_valid:
            errors.extend(hand_result.errors)
        warnings.extend(hand_result.warnings)
        score = (score + hand_result.score) / 2
        hands_validated += 1

    if hands_validated == 0:
        warnings.append("No hands detected in frame")
        score -= 0.2

    return ValidationResult(
        is_valid=len(errors) == 0,
        score=max(0.0, min(1.0, score)),
        errors=errors,
        warnings=warnings,
    )


def _validate_detected_hand(
    hand: DetectedHand,
    side: str,
    limits: AnatomicalLimits,
) -> ValidationResult:
    """Validate a detected hand."""
    errors: list[str] = []
    warnings: list[str] = []

    # Check confidence
    if hand.confidence < limits.min_confidence:
        warnings.append(f"{side} hand low confidence: {hand.confidence:.2f}")

    # Validate landmarks
    lm_result = validate_landmarks(hand.landmarks, limits)
    errors.extend([f"{side} hand: {e}" for e in lm_result.errors])
    warnings.extend([f"{side} hand: {w}" for w in lm_result.warnings])

    return ValidationResult(
        is_valid=len(errors) == 0,
        score=lm_result.score * hand.confidence,
        errors=errors,
        warnings=warnings,
    )


def validate_recording(
    recording: Recording,
    limits: AnatomicalLimits = ANATOMICAL_LIMITS,
) -> ValidationResult:
    """Validate a complete recording.

    Checks:
    - Minimum number of frames
    - Consistent hand detection across frames
    - No large jumps (jitter) between frames
    - Valid timestamps

    Args:
        recording: The Recording to validate.
        limits: Anatomical limits to use.

    Returns:
        ValidationResult with recording validation status.
    """
    errors: list[str] = []
    warnings: list[str] = []
    score = 1.0

    # Check minimum frames
    if len(recording.frames) < limits.min_frames:
        errors.append(f"Recording has {len(recording.frames)} frames, minimum is {limits.min_frames}")
        return ValidationResult(is_valid=False, score=0.0, errors=errors, warnings=warnings)

    # Validate each frame
    frame_scores: list[float] = []
    for i, frame in enumerate(recording.frames):
        result = validate_frame(frame, limits)
        frame_scores.append(result.score)
        if not result.is_valid:
            errors.extend([f"Frame {i}: {e}" for e in result.errors])
        # Limit warnings to first few frames
        if i < 3:
            warnings.extend([f"Frame {i}: {w}" for w in result.warnings])

    avg_frame_score = sum(frame_scores) / len(frame_scores) if frame_scores else 0.0

    # Check for consistent detection
    left_detection = sum(1 for f in recording.frames if f.left_hand is not None)
    right_detection = sum(1 for f in recording.frames if f.right_hand is not None)
    total_frames = len(recording.frames)

    if left_detection > 0 and left_detection < total_frames * 0.8:
        warnings.append(f"Left hand only detected in {left_detection}/{total_frames} frames")
        score -= 0.1

    if right_detection > 0 and right_detection < total_frames * 0.8:
        warnings.append(f"Right hand only detected in {right_detection}/{total_frames} frames")
        score -= 0.1

    # Check for jitter (large inter-frame movements)
    if len(recording.frames) > 1:
        jitter_result = _check_recording_jitter(recording, limits)
        if not jitter_result.is_valid:
            warnings.extend(jitter_result.warnings)
            score -= 0.1

    # Check timestamps are ordered
    timestamps = [f.timestamp_ms for f in recording.frames]
    if timestamps != sorted(timestamps):
        errors.append("Frame timestamps are not in order")
        score -= 0.2

    # Check sign_id is set
    if not recording.sign_id:
        errors.append("Recording has no sign_id")
        score -= 0.2

    final_score = (score + avg_frame_score) / 2

    return ValidationResult(
        is_valid=len(errors) == 0,
        score=max(0.0, min(1.0, final_score)),
        errors=errors,
        warnings=warnings,
    )


def _check_recording_jitter(
    recording: Recording,
    limits: AnatomicalLimits,
) -> ValidationResult:
    """Check for excessive jitter between frames."""
    warnings: list[str] = []
    max_movement = 0.0

    for i in range(1, len(recording.frames)):
        prev_frame = recording.frames[i - 1]
        curr_frame = recording.frames[i]

        # Check right hand movement
        if prev_frame.right_hand and curr_frame.right_hand:
            movement = _calculate_hand_movement(
                prev_frame.right_hand.landmarks,
                curr_frame.right_hand.landmarks,
            )
            max_movement = max(max_movement, movement)

        # Check left hand movement
        if prev_frame.left_hand and curr_frame.left_hand:
            movement = _calculate_hand_movement(
                prev_frame.left_hand.landmarks,
                curr_frame.left_hand.landmarks,
            )
            max_movement = max(max_movement, movement)

    if max_movement > limits.max_frame_movement:
        warnings.append(f"Large inter-frame movement detected: {max_movement:.3f}")

    return ValidationResult(
        is_valid=True,  # Jitter is a warning, not an error
        score=max(0.5, 1.0 - max_movement),
        errors=[],
        warnings=warnings,
    )


def _calculate_hand_movement(lm1: HandLandmarks, lm2: HandLandmarks) -> float:
    """Calculate average landmark movement between two hand states."""
    arr1 = _landmarks_to_array(lm1)
    arr2 = _landmarks_to_array(lm2)
    distances = np.linalg.norm(arr2 - arr1, axis=1)
    return float(np.mean(distances))


# =============================================================================
# Sign Validation
# =============================================================================


def validate_sign_pose(
    pose: SignPose,
    limits: AnatomicalLimits = ANATOMICAL_LIMITS,
) -> ValidationResult:
    """Validate a sign pose.

    Args:
        pose: The SignPose to validate.
        limits: Anatomical limits to use.

    Returns:
        ValidationResult with pose validation status.
    """
    errors: list[str] = []
    warnings: list[str] = []
    score = 1.0

    # Check that at least one hand has landmarks
    has_left = pose.left_hand_landmarks is not None and len(pose.left_hand_landmarks) > 0
    has_right = pose.right_hand_landmarks is not None and len(pose.right_hand_landmarks) > 0

    if not has_left and not has_right:
        errors.append("Sign pose has no landmark data for either hand")
        return ValidationResult(is_valid=False, score=0.0, errors=errors, warnings=warnings)

    # Validate left hand landmarks
    if has_left:
        if len(pose.left_hand_landmarks) != NUM_LANDMARKS:
            errors.append(f"Left hand has {len(pose.left_hand_landmarks)} landmarks, expected {NUM_LANDMARKS}")
            score -= 0.3
        else:
            for i, point in enumerate(pose.left_hand_landmarks):
                result = validate_point(point, limits)
                if not result.is_valid:
                    errors.extend([f"Left hand point {i}: {e}" for e in result.errors])

    # Validate right hand landmarks
    if has_right:
        if len(pose.right_hand_landmarks) != NUM_LANDMARKS:
            errors.append(f"Right hand has {len(pose.right_hand_landmarks)} landmarks, expected {NUM_LANDMARKS}")
            score -= 0.3
        else:
            for i, point in enumerate(pose.right_hand_landmarks):
                result = validate_point(point, limits)
                if not result.is_valid:
                    errors.extend([f"Right hand point {i}: {e}" for e in result.errors])

    # Check tolerances if present
    if pose.tolerances:
        for key, value in pose.tolerances.items():
            if value < 0:
                warnings.append(f"Negative tolerance for {key}: {value}")
            elif value > 1:
                warnings.append(f"Very large tolerance for {key}: {value}")

    return ValidationResult(
        is_valid=len(errors) == 0,
        score=max(0.0, min(1.0, score)),
        errors=errors,
        warnings=warnings,
    )


def validate_sign_entry(
    entry: SignDictionaryEntry,
    limits: AnatomicalLimits = ANATOMICAL_LIMITS,
) -> ValidationResult:
    """Validate a sign dictionary entry.

    Args:
        entry: The SignDictionaryEntry to validate.
        limits: Anatomical limits to use.

    Returns:
        ValidationResult with entry validation status.
    """
    errors: list[str] = []
    warnings: list[str] = []
    score = 1.0

    # Validate definition
    if not entry.definition.id:
        errors.append("Sign definition has no ID")
    if not entry.definition.name:
        errors.append("Sign definition has no name")

    # Validate poses
    if not entry.poses:
        errors.append("Sign entry has no poses")
        return ValidationResult(is_valid=False, score=0.0, errors=errors, warnings=warnings)

    pose_scores: list[float] = []
    for i, pose in enumerate(entry.poses):
        result = validate_sign_pose(pose, limits)
        pose_scores.append(result.score)
        if not result.is_valid:
            errors.extend([f"Pose {i}: {e}" for e in result.errors])
        warnings.extend([f"Pose {i}: {w}" for w in result.warnings])

    avg_pose_score = sum(pose_scores) / len(pose_scores) if pose_scores else 0.0

    # Check sample count
    if entry.sample_count < 1:
        errors.append(f"Invalid sample count: {entry.sample_count}")
    elif entry.sample_count < 10:
        warnings.append(f"Low sample count: {entry.sample_count}")

    # Check quality score
    if entry.quality_score < 0.5:
        warnings.append(f"Low quality score: {entry.quality_score:.2f}")

    # Check timestamps
    if entry.updated_at < entry.created_at:
        warnings.append("updated_at is before created_at")

    final_score = (score + avg_pose_score + entry.quality_score) / 3

    return ValidationResult(
        is_valid=len(errors) == 0,
        score=max(0.0, min(1.0, final_score)),
        errors=errors,
        warnings=warnings,
    )


def validate_dictionary(
    dictionary: SignDictionary,
    limits: AnatomicalLimits = ANATOMICAL_LIMITS,
) -> ValidationResult:
    """Validate a complete sign dictionary.

    Args:
        dictionary: The SignDictionary to validate.
        limits: Anatomical limits to use.

    Returns:
        ValidationResult with dictionary validation status.
    """
    errors: list[str] = []
    warnings: list[str] = []
    score = 1.0

    # Check version
    if not dictionary.version:
        errors.append("Dictionary has no version")

    # Check for entries
    if not dictionary.entries:
        errors.append("Dictionary has no entries")
        return ValidationResult(is_valid=False, score=0.0, errors=errors, warnings=warnings)

    # Validate each entry
    entry_scores: list[float] = []
    invalid_entries: list[str] = []

    for sign_id, entry in dictionary.entries.items():
        # Check ID consistency
        if entry.definition.id != sign_id:
            warnings.append(f"Entry key '{sign_id}' doesn't match definition ID '{entry.definition.id}'")

        result = validate_sign_entry(entry, limits)
        entry_scores.append(result.score)
        if not result.is_valid:
            invalid_entries.append(sign_id)
            # Only include first error per entry to avoid flooding
            if result.errors:
                errors.append(f"Entry '{sign_id}': {result.errors[0]}")

    if invalid_entries:
        warnings.append(f"{len(invalid_entries)} entries have validation errors")

    avg_entry_score = sum(entry_scores) / len(entry_scores) if entry_scores else 0.0
    final_score = (score + avg_entry_score) / 2

    return ValidationResult(
        is_valid=len(errors) == 0,
        score=max(0.0, min(1.0, final_score)),
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# Quality Scoring
# =============================================================================


def calculate_landmark_quality(landmarks: HandLandmarks) -> float:
    """Calculate quality score for hand landmarks (0-100).

    Args:
        landmarks: The HandLandmarks to score.

    Returns:
        Quality score from 0 to 100.
    """
    score = 100.0

    # Coordinate validity (25%)
    coords = _landmarks_to_array(landmarks)
    out_of_range = np.sum((coords[:, :2] < 0) | (coords[:, :2] > 1))
    coord_score = max(0, 100 - out_of_range * 5)
    score = score * 0.75 + coord_score * 0.25

    # Completeness (20%) - all landmarks present and non-zero
    zero_landmarks = np.sum(np.all(coords == 0, axis=1))
    completeness_score = max(0, 100 - zero_landmarks * 10)
    score = score * 0.80 + completeness_score * 0.20

    # Anatomical plausibility (25%)
    anatomical_score = 100.0
    if not check_finger_lengths(landmarks):
        anatomical_score -= 30
    if not check_joint_angles(landmarks):
        anatomical_score -= 30
    score = score * 0.75 + anatomical_score * 0.25

    # Variance check (15%) - landmarks should have reasonable spread
    variance = np.var(coords[:, :2])
    if variance < 0.001:
        score -= 15  # Too clustered
    elif variance > 0.1:
        score -= 10  # Too spread out

    # Connectivity check (15%) - connected landmarks should be close
    for i1, i2 in LANDMARK_CONNECTIONS:
        dist = np.linalg.norm(coords[i1] - coords[i2])
        if dist > 0.3:  # Too far apart
            score -= 2

    return max(0.0, min(100.0, score))


def calculate_recording_quality(recording: Recording) -> float:
    """Calculate quality score for a recording (0-100).

    Args:
        recording: The Recording to score.

    Returns:
        Quality score from 0 to 100.
    """
    if not recording.frames:
        return 0.0

    score = 100.0

    # Frame quality average
    frame_qualities: list[float] = []
    for frame in recording.frames:
        frame_score = 0.0
        count = 0
        if frame.right_hand:
            frame_score += calculate_landmark_quality(frame.right_hand.landmarks)
            frame_score += frame.right_hand.confidence * 20
            count += 1
        if frame.left_hand:
            frame_score += calculate_landmark_quality(frame.left_hand.landmarks)
            frame_score += frame.left_hand.confidence * 20
            count += 1
        if count > 0:
            frame_qualities.append(frame_score / count)

    if frame_qualities:
        avg_frame_quality = sum(frame_qualities) / len(frame_qualities)
        score = avg_frame_quality * 0.7  # 70% weight

    # Detection consistency (15%)
    frames_with_hands = sum(1 for f in recording.frames if f.left_hand or f.right_hand)
    consistency = frames_with_hands / len(recording.frames) if recording.frames else 0
    score += consistency * 15

    # Temporal smoothness (15%)
    if len(recording.frames) > 1:
        movements: list[float] = []
        for i in range(1, len(recording.frames)):
            prev, curr = recording.frames[i - 1], recording.frames[i]
            if prev.right_hand and curr.right_hand:
                movements.append(_calculate_hand_movement(
                    prev.right_hand.landmarks, curr.right_hand.landmarks
                ))
        if movements:
            avg_movement = sum(movements) / len(movements)
            # Lower movement = smoother = higher score
            smoothness = max(0, 15 - avg_movement * 100)
            score += smoothness
        else:
            score += 10  # Partial credit if can't measure
    else:
        score += 15  # Single frame, no jitter possible

    return max(0.0, min(100.0, score))


def calculate_sign_quality(entry: SignDictionaryEntry) -> float:
    """Calculate quality score for a sign dictionary entry (0-100).

    Args:
        entry: The SignDictionaryEntry to score.

    Returns:
        Quality score from 0 to 100.
    """
    score = 100.0

    # Sample count (30%) - more samples = more reliable
    sample_score = min(100, entry.sample_count * 2)  # 50+ samples = 100%
    score = score * 0.70 + sample_score * 0.30

    # Existing quality score (40%)
    score = score * 0.60 + entry.quality_score * 100 * 0.40

    # Pose validation (30%)
    pose_scores: list[float] = []
    for pose in entry.poses:
        result = validate_sign_pose(pose)
        pose_scores.append(result.score * 100)
    if pose_scores:
        avg_pose_score = sum(pose_scores) / len(pose_scores)
        score = score * 0.70 + avg_pose_score * 0.30

    return max(0.0, min(100.0, score))


# =============================================================================
# Batch Validation
# =============================================================================


@dataclass
class BatchValidationSummary:
    """Summary of batch validation results."""

    total_items: int = 0
    valid_items: int = 0
    invalid_items: int = 0
    avg_score: float = 0.0
    error_counts: dict[str, int] = field(default_factory=dict)
    warning_counts: dict[str, int] = field(default_factory=dict)
    results: dict[str, ValidationResult] = field(default_factory=dict)


def validate_all_recordings(
    recordings: dict[str, list[Recording]],
    limits: AnatomicalLimits = ANATOMICAL_LIMITS,
) -> dict[str, ValidationResult]:
    """Validate all recordings in a dictionary.

    Args:
        recordings: Dictionary mapping sign_id to list of recordings.
        limits: Anatomical limits to use.

    Returns:
        Dictionary mapping recording identifiers to validation results.
    """
    results: dict[str, ValidationResult] = {}

    for sign_id, recording_list in recordings.items():
        for i, recording in enumerate(recording_list):
            key = f"{sign_id}_{i}"
            results[key] = validate_recording(recording, limits)

    return results


def summarize_validation_results(
    results: dict[str, ValidationResult],
) -> BatchValidationSummary:
    """Summarize batch validation results.

    Args:
        results: Dictionary of validation results.

    Returns:
        BatchValidationSummary with aggregated statistics.
    """
    summary = BatchValidationSummary()
    summary.total_items = len(results)
    summary.results = results

    scores: list[float] = []
    for key, result in results.items():
        scores.append(result.score)
        if result.is_valid:
            summary.valid_items += 1
        else:
            summary.invalid_items += 1

        # Count errors
        for error in result.errors:
            # Extract error type (first few words)
            error_type = " ".join(error.split()[:3])
            summary.error_counts[error_type] = summary.error_counts.get(error_type, 0) + 1

        # Count warnings
        for warning in result.warnings:
            warning_type = " ".join(warning.split()[:3])
            summary.warning_counts[warning_type] = summary.warning_counts.get(warning_type, 0) + 1

    summary.avg_score = sum(scores) / len(scores) if scores else 0.0

    return summary


def generate_validation_report(
    results: dict[str, ValidationResult],
    title: str = "Validation Report",
) -> str:
    """Generate a markdown-formatted validation report.

    Args:
        results: Dictionary of validation results.
        title: Report title.

    Returns:
        Markdown-formatted report string.
    """
    summary = summarize_validation_results(results)

    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Items | {summary.total_items} |",
        f"| Valid Items | {summary.valid_items} ({summary.valid_items/summary.total_items*100:.1f}%) |" if summary.total_items > 0 else "| Valid Items | 0 |",
        f"| Invalid Items | {summary.invalid_items} |",
        f"| Average Score | {summary.avg_score:.2f} |",
        "",
    ]

    # Error breakdown
    if summary.error_counts:
        lines.extend([
            "## Errors",
            "",
            "| Error Type | Count |",
            "|------------|-------|",
        ])
        for error_type, count in sorted(summary.error_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {error_type} | {count} |")
        lines.append("")

    # Warning breakdown
    if summary.warning_counts:
        lines.extend([
            "## Warnings",
            "",
            "| Warning Type | Count |",
            "|--------------|-------|",
        ])
        for warning_type, count in sorted(summary.warning_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"| {warning_type} | {count} |")
        lines.append("")

    # Score distribution
    lines.extend([
        "## Score Distribution",
        "",
    ])

    score_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in score_ranges:
        count = sum(1 for r in results.values() if low <= r.score < high)
        bar = "█" * (count * 20 // max(len(results), 1))
        lines.append(f"| {low:.1f}-{high:.1f} | {bar} {count} |")

    lines.append("")

    # Sample of invalid items
    invalid_items = [(k, v) for k, v in results.items() if not v.is_valid]
    if invalid_items:
        lines.extend([
            "## Sample Invalid Items",
            "",
        ])
        for key, result in invalid_items[:5]:
            lines.append(f"### {key}")
            lines.append(f"- Score: {result.score:.2f}")
            lines.append(f"- Errors: {', '.join(result.errors[:3])}")
            lines.append("")

    return "\n".join(lines)
