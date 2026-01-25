#!/usr/bin/env python3
"""
Recording Processing Pipeline

Converts raw sign recordings to canonical poses for the BSL dictionary.
Includes frame filtering, normalization, averaging, and tolerance calculation.

Usage:
    python scripts/process_recordings.py single /data/raw-recordings/bsl_alphabet_a_*.json
    python scripts/process_recordings.py batch /data/raw-recordings/ --output /data/processed/
    python scripts/process_recordings.py report /data/processed/
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import click
import numpy as np
from numpy.typing import NDArray
from scipy import stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.landmarks import NUM_LANDMARKS, LANDMARK_NAMES
from src.normalize import normalize_all, NormalizationResult
from src.types import (
    Point3D,
    Recording,
    RecordingFrame,
    DetectedHand,
    SignPose,
    ProcessingResult,
    NormalizationParams,
)
from src.validation import calculate_recording_quality
from src.math_utils import magnitude, angle_between_vectors


# =============================================================================
# Data Extraction Utilities
# =============================================================================


def frame_to_landmarks(
    frame: RecordingFrame,
    hand: Literal["left", "right", "any"] = "any"
) -> Optional[NDArray[np.floating]]:
    """
    Extract landmarks from a frame as numpy array.

    Args:
        frame: Recording frame to extract from
        hand: Which hand to extract ("left", "right", or "any" for first available)

    Returns:
        (21, 3) array of landmarks or None if hand not present
    """
    detected_hand: Optional[DetectedHand] = None

    if hand == "left":
        detected_hand = frame.left_hand
    elif hand == "right":
        detected_hand = frame.right_hand
    else:  # any
        detected_hand = frame.right_hand or frame.left_hand

    if detected_hand is None:
        return None

    landmarks = np.zeros((NUM_LANDMARKS, 3))
    for lm in detected_hand.landmarks.landmarks:
        landmarks[lm.index] = [lm.point.x, lm.point.y, lm.point.z]

    return landmarks


def recording_to_landmark_sequence(
    recording: Recording,
    hand: Literal["left", "right", "any"] = "any"
) -> list[tuple[int, NDArray[np.floating], float]]:
    """
    Extract all landmarks from a recording as a sequence.

    Args:
        recording: Recording to extract from
        hand: Which hand to extract

    Returns:
        List of (timestamp_ms, landmarks, confidence) tuples
    """
    sequence = []

    for frame in recording.frames:
        landmarks = frame_to_landmarks(frame, hand)
        if landmarks is not None:
            if hand == "left" and frame.left_hand:
                conf = frame.left_hand.confidence
            elif hand == "right" and frame.right_hand:
                conf = frame.right_hand.confidence
            else:
                conf = max(
                    frame.left_hand.confidence if frame.left_hand else 0,
                    frame.right_hand.confidence if frame.right_hand else 0
                )
            sequence.append((frame.timestamp_ms, landmarks, conf))

    return sequence


# =============================================================================
# Frame Filtering
# =============================================================================


def filter_low_confidence(
    frames: list[tuple[int, NDArray[np.floating], float]],
    threshold: float = 0.7
) -> list[tuple[int, NDArray[np.floating], float]]:
    """
    Filter out frames with low detection confidence.

    Args:
        frames: List of (timestamp, landmarks, confidence) tuples
        threshold: Minimum confidence to keep (default 0.7)

    Returns:
        Filtered list of frames
    """
    return [(ts, lm, conf) for ts, lm, conf in frames if conf >= threshold]


def filter_outliers(
    frames: list[tuple[int, NDArray[np.floating], float]],
    method: Literal["zscore", "iqr"] = "zscore",
    threshold: float = 2.0
) -> list[tuple[int, NDArray[np.floating], float]]:
    """
    Filter out frames with outlier landmark positions.

    Uses wrist position and palm width as metrics for outlier detection.

    Args:
        frames: List of (timestamp, landmarks, confidence) tuples
        method: Outlier detection method ("zscore" or "iqr")
        threshold: Threshold for outlier detection (z-score or IQR multiplier)

    Returns:
        Filtered list of frames with outliers removed
    """
    if len(frames) < 4:
        return frames

    # Calculate metrics for each frame
    wrist_positions = np.array([lm[0] for _, lm, _ in frames])  # Wrist landmark
    palm_widths = np.array([
        magnitude(lm[5] - lm[17])  # Index MCP to Pinky MCP
        for _, lm, _ in frames
    ])

    # Combine metrics
    metrics = np.column_stack([
        wrist_positions,
        palm_widths.reshape(-1, 1)
    ])

    if method == "zscore":
        # Z-score based filtering
        z_scores = np.abs(stats.zscore(metrics, axis=0))
        max_z = np.max(z_scores, axis=1)
        mask = max_z < threshold
    else:  # iqr
        # IQR based filtering
        q1 = np.percentile(metrics, 25, axis=0)
        q3 = np.percentile(metrics, 75, axis=0)
        iqr = q3 - q1

        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr

        mask = np.all((metrics >= lower) & (metrics <= upper), axis=1)

    return [f for f, keep in zip(frames, mask) if keep]


def detect_static_frames(
    frames: list[tuple[int, NDArray[np.floating], float]],
    motion_threshold: float = 0.005,
    window_size: int = 5
) -> list[tuple[int, NDArray[np.floating], float]]:
    """
    Filter out frames where the hand is barely moving (static frames).

    Useful for removing frames at the start/end of recordings where
    the signer hasn't begun/finished the sign.

    Args:
        frames: List of (timestamp, landmarks, confidence) tuples
        motion_threshold: Minimum average motion to be considered moving
        window_size: Number of frames to average motion over

    Returns:
        Filtered list of frames with significant motion
    """
    if len(frames) < window_size + 1:
        return frames

    # Calculate motion (displacement) between consecutive frames
    landmarks_array = np.array([lm for _, lm, _ in frames])

    # Frame-to-frame displacement
    displacements = np.diff(landmarks_array, axis=0)
    motion_magnitude = np.linalg.norm(displacements, axis=(1, 2))

    # Sliding window average
    kernel = np.ones(window_size) / window_size
    smoothed_motion = np.convolve(motion_magnitude, kernel, mode='valid')

    # Pad to match frame count (first frames assumed moving if later ones are)
    pad_size = len(frames) - len(smoothed_motion)
    smoothed_motion = np.concatenate([
        np.full(pad_size, smoothed_motion[0] if len(smoothed_motion) > 0 else 0),
        smoothed_motion
    ])

    # Keep frames with motion above threshold
    mask = smoothed_motion >= motion_threshold
    return [f for f, keep in zip(frames, mask) if keep]


def filter_frames(
    frames: list[tuple[int, NDArray[np.floating], float]],
    confidence_threshold: float = 0.7,
    outlier_method: Literal["zscore", "iqr", "none"] = "zscore",
    outlier_threshold: float = 2.0,
    remove_static: bool = True,
    motion_threshold: float = 0.005
) -> list[tuple[int, NDArray[np.floating], float]]:
    """
    Apply all frame filtering steps.

    Args:
        frames: List of (timestamp, landmarks, confidence) tuples
        confidence_threshold: Minimum detection confidence
        outlier_method: Method for outlier detection ("zscore", "iqr", "none")
        outlier_threshold: Threshold for outlier detection
        remove_static: Whether to remove static frames
        motion_threshold: Threshold for static frame detection

    Returns:
        Filtered list of frames
    """
    result = frames

    # Step 1: Low confidence filter
    result = filter_low_confidence(result, confidence_threshold)

    # Step 2: Outlier filter
    if outlier_method != "none" and len(result) >= 4:
        result = filter_outliers(result, outlier_method, outlier_threshold)

    # Step 3: Static frame filter
    if remove_static and len(result) >= 6:
        result = detect_static_frames(result, motion_threshold)

    return result


# =============================================================================
# Landmark Averaging
# =============================================================================


def average_landmarks_mean(
    samples: list[NDArray[np.floating]]
) -> NDArray[np.floating]:
    """
    Calculate mean average of landmark samples.

    Args:
        samples: List of (21, 3) landmark arrays

    Returns:
        (21, 3) averaged landmarks
    """
    if not samples:
        raise ValueError("Cannot average empty sample list")

    stacked = np.stack(samples, axis=0)
    return np.mean(stacked, axis=0)


def average_landmarks_median(
    samples: list[NDArray[np.floating]]
) -> NDArray[np.floating]:
    """
    Calculate median average of landmark samples.

    More robust to outliers than mean.

    Args:
        samples: List of (21, 3) landmark arrays

    Returns:
        (21, 3) averaged landmarks
    """
    if not samples:
        raise ValueError("Cannot average empty sample list")

    stacked = np.stack(samples, axis=0)
    return np.median(stacked, axis=0)


def average_landmarks_trimmed(
    samples: list[NDArray[np.floating]],
    trim: float = 0.1
) -> NDArray[np.floating]:
    """
    Calculate trimmed mean average of landmark samples.

    Removes extreme values before averaging.

    Args:
        samples: List of (21, 3) landmark arrays
        trim: Proportion to trim from each end (default 0.1 = 10%)

    Returns:
        (21, 3) averaged landmarks
    """
    if not samples:
        raise ValueError("Cannot average empty sample list")

    if len(samples) < 5:
        # Not enough samples for meaningful trimming
        return average_landmarks_mean(samples)

    stacked = np.stack(samples, axis=0)

    # Apply trimmed mean along sample axis
    result = np.zeros((NUM_LANDMARKS, 3))
    for i in range(NUM_LANDMARKS):
        for j in range(3):
            result[i, j] = stats.trim_mean(stacked[:, i, j], trim)

    return result


def average_landmarks_weighted(
    samples: list[NDArray[np.floating]],
    weights: list[float]
) -> NDArray[np.floating]:
    """
    Calculate weighted average of landmark samples.

    Args:
        samples: List of (21, 3) landmark arrays
        weights: Weights for each sample (will be normalized)

    Returns:
        (21, 3) averaged landmarks
    """
    if not samples:
        raise ValueError("Cannot average empty sample list")

    if len(samples) != len(weights):
        raise ValueError("Number of samples must match number of weights")

    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    stacked = np.stack(samples, axis=0)
    return np.average(stacked, axis=0, weights=weights)


# =============================================================================
# Tolerance Calculation
# =============================================================================


@dataclass
class ToleranceResult:
    """Result of tolerance calculation."""

    position_tolerances: dict[str, float]
    """Per-landmark position tolerance (standard deviation)"""

    angle_tolerances: dict[str, float]
    """Per-joint angle tolerance in degrees"""

    overall_position_tolerance: float
    """Average position tolerance across all landmarks"""

    overall_angle_tolerance: float
    """Average angle tolerance across all joints"""


def calculate_position_tolerances(
    samples: list[NDArray[np.floating]],
    canonical: NDArray[np.floating]
) -> dict[str, float]:
    """
    Calculate per-landmark position tolerances.

    Tolerance is the standard deviation of distance from canonical position.

    Args:
        samples: List of (21, 3) landmark arrays
        canonical: (21, 3) canonical landmark positions

    Returns:
        Dictionary mapping landmark names to tolerance values
    """
    if not samples:
        return {name: 0.0 for name in LANDMARK_NAMES}

    # Calculate distances from canonical for each sample
    stacked = np.stack(samples, axis=0)  # (N, 21, 3)
    distances = np.linalg.norm(stacked - canonical, axis=2)  # (N, 21)

    # Standard deviation per landmark
    std_devs = np.std(distances, axis=0)  # (21,)

    return {name: float(std_devs[i]) for i, name in enumerate(LANDMARK_NAMES)}


def calculate_joint_angle(
    landmarks: NDArray[np.floating],
    joint_indices: tuple[int, int, int]
) -> float:
    """
    Calculate angle at a joint in degrees.

    Args:
        landmarks: (21, 3) landmark array
        joint_indices: (proximal, middle, distal) landmark indices

    Returns:
        Angle in degrees
    """
    p1 = landmarks[joint_indices[0]]
    p2 = landmarks[joint_indices[1]]
    p3 = landmarks[joint_indices[2]]

    v1 = p1 - p2
    v2 = p3 - p2

    angle_rad = angle_between_vectors(v1, v2)
    return np.degrees(angle_rad)


# Joint definitions for angle calculation
JOINT_DEFINITIONS = {
    "thumb_cmc": (0, 1, 2),
    "thumb_mcp": (1, 2, 3),
    "thumb_ip": (2, 3, 4),
    "index_mcp": (0, 5, 6),
    "index_pip": (5, 6, 7),
    "index_dip": (6, 7, 8),
    "middle_mcp": (0, 9, 10),
    "middle_pip": (9, 10, 11),
    "middle_dip": (10, 11, 12),
    "ring_mcp": (0, 13, 14),
    "ring_pip": (13, 14, 15),
    "ring_dip": (14, 15, 16),
    "pinky_mcp": (0, 17, 18),
    "pinky_pip": (17, 18, 19),
    "pinky_dip": (18, 19, 20),
}


def calculate_angle_tolerances(
    samples: list[NDArray[np.floating]],
    canonical: NDArray[np.floating]
) -> dict[str, float]:
    """
    Calculate per-joint angle tolerances.

    Tolerance is the standard deviation of angle deviation from canonical.

    Args:
        samples: List of (21, 3) landmark arrays
        canonical: (21, 3) canonical landmark positions

    Returns:
        Dictionary mapping joint names to angle tolerance in degrees
    """
    if not samples:
        return {name: 0.0 for name in JOINT_DEFINITIONS}

    # Calculate canonical angles
    canonical_angles = {
        name: calculate_joint_angle(canonical, indices)
        for name, indices in JOINT_DEFINITIONS.items()
    }

    # Calculate angle deviations for each sample
    deviations = {name: [] for name in JOINT_DEFINITIONS}

    for sample in samples:
        for name, indices in JOINT_DEFINITIONS.items():
            angle = calculate_joint_angle(sample, indices)
            deviation = abs(angle - canonical_angles[name])
            deviations[name].append(deviation)

    # Standard deviation of deviations
    return {
        name: float(np.std(devs)) if devs else 0.0
        for name, devs in deviations.items()
    }


def calculate_tolerances(
    samples: list[NDArray[np.floating]],
    canonical: NDArray[np.floating]
) -> ToleranceResult:
    """
    Calculate all tolerances for a set of samples.

    Args:
        samples: List of (21, 3) normalized landmark arrays
        canonical: (21, 3) canonical normalized landmarks

    Returns:
        ToleranceResult with position and angle tolerances
    """
    position_tols = calculate_position_tolerances(samples, canonical)
    angle_tols = calculate_angle_tolerances(samples, canonical)

    return ToleranceResult(
        position_tolerances=position_tols,
        angle_tolerances=angle_tols,
        overall_position_tolerance=float(np.mean(list(position_tols.values()))),
        overall_angle_tolerance=float(np.mean(list(angle_tols.values())))
    )


# =============================================================================
# Main Processing Functions
# =============================================================================


@dataclass
class ProcessingConfig:
    """Configuration for recording processing."""

    confidence_threshold: float = 0.7
    outlier_method: Literal["zscore", "iqr", "none"] = "zscore"
    outlier_threshold: float = 2.0
    remove_static: bool = True
    motion_threshold: float = 0.005
    averaging_method: Literal["mean", "median", "trimmed"] = "trimmed"
    trim_ratio: float = 0.1
    min_frames: int = 5
    target_hand: Literal["left", "right", "any"] = "any"


def process_recording(
    recording: Recording,
    config: Optional[ProcessingConfig] = None
) -> Optional[ProcessingResult]:
    """
    Process a single recording to extract canonical pose.

    Args:
        recording: Recording to process
        config: Processing configuration

    Returns:
        ProcessingResult or None if processing failed
    """
    if config is None:
        config = ProcessingConfig()

    warnings = []

    # Extract landmark sequence
    sequence = recording_to_landmark_sequence(recording, config.target_hand)

    if not sequence:
        return None

    # Filter frames
    filtered = filter_frames(
        sequence,
        confidence_threshold=config.confidence_threshold,
        outlier_method=config.outlier_method,
        outlier_threshold=config.outlier_threshold,
        remove_static=config.remove_static,
        motion_threshold=config.motion_threshold
    )

    if len(filtered) < config.min_frames:
        warnings.append(f"Too few frames after filtering: {len(filtered)} < {config.min_frames}")
        if not filtered:
            return None
        # Use unfiltered if we have too few
        if len(filtered) < 3:
            filtered = sequence[:config.min_frames]
            warnings.append("Using unfiltered frames due to insufficient filtered frames")

    # Normalize all frames
    normalized_samples = []
    norm_result = None

    for _, landmarks, _ in filtered:
        result = normalize_all(landmarks)
        normalized_samples.append(result.landmarks)
        if norm_result is None:
            norm_result = result

    # Average to get canonical pose
    if config.averaging_method == "mean":
        canonical = average_landmarks_mean(normalized_samples)
    elif config.averaging_method == "median":
        canonical = average_landmarks_median(normalized_samples)
    else:  # trimmed
        canonical = average_landmarks_trimmed(normalized_samples, config.trim_ratio)

    # Calculate tolerances
    tol_result = calculate_tolerances(normalized_samples, canonical)

    # Convert to SignPose
    landmarks_list = [
        Point3D(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        for p in canonical
    ]

    # Determine which hand to use
    has_left = any(f.left_hand for f in recording.frames)
    has_right = any(f.right_hand for f in recording.frames)

    pose = SignPose(
        left_hand_landmarks=landmarks_list if has_left and config.target_hand != "right" else None,
        right_hand_landmarks=landmarks_list if has_right and config.target_hand != "left" else None,
        tolerances={
            **{f"pos_{k}": v for k, v in tol_result.position_tolerances.items()},
            **{f"angle_{k}": v for k, v in tol_result.angle_tolerances.items()},
        }
    )

    # Calculate quality score (returns 0-100, normalize to 0-1)
    quality_score = calculate_recording_quality(recording) / 100.0

    # Adjust quality based on processing
    frame_retention = len(filtered) / len(sequence) if sequence else 0
    quality_score = quality_score * (0.5 + 0.5 * frame_retention)

    # Build normalization params if available
    norm_params = None
    if norm_result is not None:
        norm_params = NormalizationParams(
            scale_factor=float(norm_result.scale_factor) if norm_result.scale_factor > 0 else 1.0,
            rotation_matrix=norm_result.rotation_matrix.tolist(),
            translation_vector=norm_result.original_centroid.tolist()
        )

    return ProcessingResult(
        canonical_pose=pose,
        quality_score=min(1.0, max(0.0, quality_score)),
        sample_count=len(filtered),
        warnings=warnings,
        normalization_params=norm_params
    )


def process_multiple_recordings(
    recordings: list[Recording],
    config: Optional[ProcessingConfig] = None
) -> Optional[ProcessingResult]:
    """
    Process multiple recordings to create a combined canonical pose.

    Weights recordings by quality and combines their normalized samples.

    Args:
        recordings: List of recordings for the same sign
        config: Processing configuration

    Returns:
        ProcessingResult combining all recordings, or None if processing failed
    """
    if config is None:
        config = ProcessingConfig()

    all_samples = []
    all_weights = []
    total_frame_count = 0
    all_warnings = []

    for recording in recordings:
        # Extract and filter frames
        sequence = recording_to_landmark_sequence(recording, config.target_hand)
        if not sequence:
            continue

        filtered = filter_frames(
            sequence,
            confidence_threshold=config.confidence_threshold,
            outlier_method=config.outlier_method,
            outlier_threshold=config.outlier_threshold,
            remove_static=config.remove_static,
            motion_threshold=config.motion_threshold
        )

        if len(filtered) < config.min_frames:
            continue

        # Get quality weight for this recording (0-100 normalized to 0-1)
        weight = calculate_recording_quality(recording) / 100.0

        # Normalize and collect samples
        for _, landmarks, conf in filtered:
            result = normalize_all(landmarks)
            all_samples.append(result.landmarks)
            all_weights.append(weight * conf)
            total_frame_count += 1

    if not all_samples:
        return None

    # Weighted average for canonical pose
    canonical = average_landmarks_weighted(all_samples, all_weights)

    # Calculate tolerances
    tol_result = calculate_tolerances(all_samples, canonical)

    # Convert to SignPose
    landmarks_list = [
        Point3D(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        for p in canonical
    ]

    # Determine handedness from first recording
    first_rec = recordings[0]
    has_left = any(f.left_hand for f in first_rec.frames)
    has_right = any(f.right_hand for f in first_rec.frames)

    pose = SignPose(
        left_hand_landmarks=landmarks_list if has_left else None,
        right_hand_landmarks=landmarks_list if has_right else None,
        tolerances={
            **{f"pos_{k}": v for k, v in tol_result.position_tolerances.items()},
            **{f"angle_{k}": v for k, v in tol_result.angle_tolerances.items()},
        }
    )

    # Overall quality is weighted average
    total_weight = sum(all_weights)
    overall_quality = total_weight / len(all_weights) if all_weights else 0.5

    return ProcessingResult(
        canonical_pose=pose,
        quality_score=min(1.0, max(0.0, overall_quality)),
        sample_count=total_frame_count,
        warnings=all_warnings,
        normalization_params=None  # Multiple recordings don't have single params
    )


def process_sign(
    sign_id: str,
    recordings: list[Recording],
    config: Optional[ProcessingConfig] = None
) -> Optional[SignPose]:
    """
    Full pipeline to process all recordings for one sign.

    Args:
        sign_id: ID of the sign being processed
        recordings: All recordings for this sign
        config: Processing configuration

    Returns:
        SignPose for the sign, or None if processing failed
    """
    # Filter recordings to only those matching sign_id
    matching = [r for r in recordings if r.sign_id == sign_id]

    if not matching:
        return None

    if len(matching) == 1:
        result = process_recording(matching[0], config)
    else:
        result = process_multiple_recordings(matching, config)

    return result.canonical_pose if result else None


# =============================================================================
# Batch Processing
# =============================================================================


@dataclass
class BatchResult:
    """Result of batch processing."""

    processed: dict[str, ProcessingResult] = field(default_factory=dict)
    """Successfully processed signs"""

    failed: dict[str, str] = field(default_factory=dict)
    """Failed signs with error messages"""

    skipped: list[str] = field(default_factory=list)
    """Skipped files"""


def load_recording_from_json(path: Path) -> Optional[Recording]:
    """
    Load a recording from a JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Recording object or None if loading failed
    """
    try:
        with open(path) as f:
            data = json.load(f)

        # Handle both raw export format and Recording format
        if "frames" in data and isinstance(data["frames"], list):
            # Convert from export format
            frames = []
            for frame_data in data["frames"]:
                # Parse the frame
                frame = RecordingFrame(
                    timestamp_ms=frame_data.get("timestamp_ms", 0),
                    left_hand=None,
                    right_hand=None
                )

                # Parse hands if present
                if "hands" in frame_data:
                    for hand_data in frame_data["hands"]:
                        from src.types import HandLandmarks, HandLandmark

                        landmarks_list = []
                        raw_landmarks = hand_data.get("landmarks", [])

                        for i, lm in enumerate(raw_landmarks):
                            if isinstance(lm, list) and len(lm) >= 3:
                                landmarks_list.append(HandLandmark(
                                    point=Point3D(
                                        x=max(0, min(1, lm[0])),
                                        y=max(0, min(1, lm[1])),
                                        z=lm[2]
                                    ),
                                    name=LANDMARK_NAMES[i],
                                    index=i
                                ))

                        if len(landmarks_list) == 21:
                            detected = DetectedHand(
                                landmarks=HandLandmarks(landmarks=landmarks_list),
                                handedness=hand_data.get("handedness", "Right"),
                                confidence=hand_data.get("confidence", 0.9)
                            )

                            if hand_data.get("handedness") == "Left":
                                frame.left_hand = detected
                            else:
                                frame.right_hand = detected

                frames.append(frame)

            if frames:
                return Recording(
                    frames=frames,
                    sign_id=data.get("sign_id", path.stem.split("_")[0]),
                    start_time=datetime.fromisoformat(
                        data.get("recorded_at", datetime.now().isoformat()).replace("Z", "+00:00")
                    ),
                    end_time=datetime.now(),
                    metadata={"source_file": str(path)}
                )

        return None

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def process_all_signs(
    recordings_dir: Path,
    output_dir: Path,
    config: Optional[ProcessingConfig] = None
) -> BatchResult:
    """
    Process all recordings in a directory.

    Args:
        recordings_dir: Directory containing recording JSON files
        output_dir: Directory to write processed results
        config: Processing configuration

    Returns:
        BatchResult with processing summary
    """
    if config is None:
        config = ProcessingConfig()

    result = BatchResult()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group recordings by sign_id
    recordings_by_sign: dict[str, list[Recording]] = defaultdict(list)

    # Find all JSON files
    json_files = list(recordings_dir.glob("*.json"))
    print(f"Found {len(json_files)} recording files")

    for json_path in json_files:
        recording = load_recording_from_json(json_path)
        if recording:
            recordings_by_sign[recording.sign_id].append(recording)
        else:
            result.skipped.append(str(json_path))

    print(f"Loaded recordings for {len(recordings_by_sign)} signs")

    # Process each sign
    for sign_id, recordings in recordings_by_sign.items():
        try:
            if len(recordings) == 1:
                proc_result = process_recording(recordings[0], config)
            else:
                proc_result = process_multiple_recordings(recordings, config)

            if proc_result:
                result.processed[sign_id] = proc_result

                # Save to output
                output_path = output_dir / f"{sign_id}.json"
                with open(output_path, "w") as f:
                    json.dump(proc_result.model_dump(), f, indent=2, default=str)

                print(f"  Processed {sign_id}: {proc_result.sample_count} frames, "
                      f"quality={proc_result.quality_score:.2f}")
            else:
                result.failed[sign_id] = "Processing returned no result"

        except Exception as e:
            result.failed[sign_id] = str(e)
            print(f"  Failed {sign_id}: {e}")

    return result


def generate_report(processed_dir: Path) -> dict:
    """
    Generate a report from processed results.

    Args:
        processed_dir: Directory containing processed JSON files

    Returns:
        Report dictionary
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_signs": 0,
        "average_quality": 0.0,
        "total_samples": 0,
        "signs_by_quality": {"high": [], "medium": [], "low": []},
        "warnings_summary": defaultdict(int),
        "details": {}
    }

    quality_scores = []

    for json_path in processed_dir.glob("*.json"):
        try:
            with open(json_path) as f:
                data = json.load(f)

            sign_id = json_path.stem
            report["total_signs"] += 1
            report["total_samples"] += data.get("sample_count", 0)

            quality = data.get("quality_score", 0)
            quality_scores.append(quality)

            if quality >= 0.8:
                report["signs_by_quality"]["high"].append(sign_id)
            elif quality >= 0.5:
                report["signs_by_quality"]["medium"].append(sign_id)
            else:
                report["signs_by_quality"]["low"].append(sign_id)

            for warning in data.get("warnings", []):
                report["warnings_summary"][warning] += 1

            report["details"][sign_id] = {
                "quality_score": quality,
                "sample_count": data.get("sample_count", 0),
                "warnings": data.get("warnings", [])
            }

        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    if quality_scores:
        report["average_quality"] = float(np.mean(quality_scores))

    report["warnings_summary"] = dict(report["warnings_summary"])

    return report


# =============================================================================
# CLI
# =============================================================================


@click.group()
def cli():
    """Recording processing pipeline for BSL signs."""
    pass


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--confidence", "-c", default=0.7, help="Confidence threshold")
@click.option("--averaging", "-a", type=click.Choice(["mean", "median", "trimmed"]),
              default="trimmed", help="Averaging method")
def single(files, output, confidence, averaging):
    """Process single or multiple recording files for one sign."""
    if not files:
        click.echo("No files specified")
        return

    config = ProcessingConfig(
        confidence_threshold=confidence,
        averaging_method=averaging
    )

    recordings = []
    for file_path in files:
        recording = load_recording_from_json(Path(file_path))
        if recording:
            recordings.append(recording)
            click.echo(f"Loaded: {file_path}")
        else:
            click.echo(f"Failed to load: {file_path}")

    if not recordings:
        click.echo("No valid recordings loaded")
        return

    # Process
    if len(recordings) == 1:
        result = process_recording(recordings[0], config)
    else:
        result = process_multiple_recordings(recordings, config)

    if result:
        click.echo(f"\nProcessing complete:")
        click.echo(f"  Samples: {result.sample_count}")
        click.echo(f"  Quality: {result.quality_score:.2f}")

        if result.warnings:
            click.echo(f"  Warnings: {len(result.warnings)}")
            for w in result.warnings:
                click.echo(f"    - {w}")

        # Output
        if output:
            output_path = Path(output)
            with open(output_path, "w") as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
            click.echo(f"\nSaved to: {output_path}")
        else:
            click.echo("\nResult (use --output to save):")
            click.echo(json.dumps(result.model_dump(), indent=2, default=str)[:1000] + "...")
    else:
        click.echo("Processing failed")


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--confidence", "-c", default=0.7, help="Confidence threshold")
@click.option("--averaging", "-a", type=click.Choice(["mean", "median", "trimmed"]),
              default="trimmed", help="Averaging method")
def batch(input_dir, output, confidence, averaging):
    """Process all recordings in a directory."""
    config = ProcessingConfig(
        confidence_threshold=confidence,
        averaging_method=averaging
    )

    input_path = Path(input_dir)
    output_path = Path(output)

    click.echo(f"Processing recordings from: {input_path}")
    click.echo(f"Output directory: {output_path}")

    result = process_all_signs(input_path, output_path, config)

    click.echo(f"\n{'='*50}")
    click.echo(f"Batch processing complete:")
    click.echo(f"  Processed: {len(result.processed)}")
    click.echo(f"  Failed: {len(result.failed)}")
    click.echo(f"  Skipped: {len(result.skipped)}")

    if result.failed:
        click.echo("\nFailed signs:")
        for sign_id, error in result.failed.items():
            click.echo(f"  {sign_id}: {error}")

    # Generate and save report
    report = generate_report(output_path)
    report_path = output_path / "processing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    click.echo(f"\nReport saved to: {report_path}")


@cli.command()
@click.argument("processed_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", type=click.Path(), help="Output report file")
def report(processed_dir, output):
    """Generate report from processed results."""
    report_data = generate_report(Path(processed_dir))

    click.echo(f"\n{'='*50}")
    click.echo("Processing Report")
    click.echo(f"{'='*50}")
    click.echo(f"Total signs: {report_data['total_signs']}")
    click.echo(f"Total samples: {report_data['total_samples']}")
    click.echo(f"Average quality: {report_data['average_quality']:.2f}")

    click.echo(f"\nQuality distribution:")
    click.echo(f"  High (≥0.8): {len(report_data['signs_by_quality']['high'])}")
    click.echo(f"  Medium (≥0.5): {len(report_data['signs_by_quality']['medium'])}")
    click.echo(f"  Low (<0.5): {len(report_data['signs_by_quality']['low'])}")

    if report_data["warnings_summary"]:
        click.echo(f"\nWarnings summary:")
        for warning, count in report_data["warnings_summary"].items():
            click.echo(f"  ({count}x) {warning}")

    if output:
        with open(output, "w") as f:
            json.dump(report_data, f, indent=2)
        click.echo(f"\nFull report saved to: {output}")


if __name__ == "__main__":
    cli()
