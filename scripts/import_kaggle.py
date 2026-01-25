"""Import Kaggle BSL Numbers/Alphabet dataset.

This script imports the Kaggle BSL dataset containing hand landmark data
for alphabet letters and numbers. The dataset has two files:
- one_hand_dataset.csv: Numbers 0-9 and letter C (64 columns: 21×3 coords + label)
- two_hand_dataset.csv: Alphabet A-Z (except C,H,J,Y) + number 10 (127 columns: 42×3 coords + label)

Usage:
    python -m scripts.import_kaggle analyze /data/kaggle-bsl/one_hand_dataset.csv
    python -m scripts.import_kaggle import /data/kaggle-bsl/ --output /data/processed/
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import click
import pandas as pd
from pydantic import ValidationError

from src.landmarks import LANDMARK_NAMES, NUM_LANDMARKS
from src.types import (
    DetectedHand,
    HandLandmark,
    HandLandmarks,
    Point3D,
    Recording,
    RecordingFrame,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Expected column counts
ONE_HAND_COLS = 64  # 21 landmarks × 3 coords + 1 label
TWO_HAND_COLS = 127  # 42 landmarks × 3 coords + 1 label

# Label parsing regex: "A - a" or "Eight - 8"
LABEL_PATTERN = re.compile(r"^([A-Za-z]+)\s*-\s*(.+)$")

# Mapping from Kaggle labels to our sign IDs
LABEL_TO_SIGN_ID: dict[str, str] = {
    # Alphabet (from both files)
    "a": "bsl_alphabet_a",
    "b": "bsl_alphabet_b",
    "c": "bsl_alphabet_c",
    "d": "bsl_alphabet_d",
    "e": "bsl_alphabet_e",
    "f": "bsl_alphabet_f",
    "g": "bsl_alphabet_g",
    "h": "bsl_alphabet_h",
    "i": "bsl_alphabet_i",
    "j": "bsl_alphabet_j",
    "k": "bsl_alphabet_k",
    "l": "bsl_alphabet_l",
    "m": "bsl_alphabet_m",
    "n": "bsl_alphabet_n",
    "o": "bsl_alphabet_o",
    "p": "bsl_alphabet_p",
    "q": "bsl_alphabet_q",
    "r": "bsl_alphabet_r",
    "s": "bsl_alphabet_s",
    "t": "bsl_alphabet_t",
    "u": "bsl_alphabet_u",
    "v": "bsl_alphabet_v",
    "w": "bsl_alphabet_w",
    "x": "bsl_alphabet_x",
    "y": "bsl_alphabet_y",
    "z": "bsl_alphabet_z",
    # Numbers
    "0": "bsl_numbers_0",
    "1": "bsl_numbers_1",
    "2": "bsl_numbers_2",
    "3": "bsl_numbers_3",
    "4": "bsl_numbers_4",
    "5": "bsl_numbers_5",
    "6": "bsl_numbers_6",
    "7": "bsl_numbers_7",
    "8": "bsl_numbers_8",
    "9": "bsl_numbers_9",
    "10": "bsl_numbers_10",
}

DatasetType = Literal["one_hand", "two_hand", "unknown"]


# =============================================================================
# Analysis Functions
# =============================================================================


def detect_dataset_type(filepath: Path) -> DatasetType:
    """Detect whether file is one-hand or two-hand dataset based on column count."""
    # Read just the first row to count columns
    df = pd.read_csv(filepath, header=None, nrows=1)
    num_cols = len(df.columns)

    if num_cols == ONE_HAND_COLS:
        return "one_hand"
    elif num_cols == TWO_HAND_COLS:
        return "two_hand"
    else:
        return "unknown"


def analyze_kaggle_file(filepath: Path) -> dict[str, Any]:
    """Analyze a Kaggle CSV file and return structure summary.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Dictionary with file analysis including:
        - file_name, file_size, row_count, column_count
        - dataset_type (one_hand/two_hand)
        - labels found and their counts
        - coordinate statistics
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Analyzing: {filepath.name}")

    # Basic file info
    file_size = filepath.stat().st_size
    dataset_type = detect_dataset_type(filepath)

    # Read the full file
    df = pd.read_csv(filepath, header=None)
    row_count = len(df)
    col_count = len(df.columns)

    # Parse labels (last column)
    label_col = df.iloc[:, -1]
    labels_raw = label_col.unique().tolist()

    # Parse and normalize labels
    label_counts: dict[str, int] = {}
    unparseable_labels: list[str] = []

    for label in labels_raw:
        label_str = str(label).strip()
        match = LABEL_PATTERN.match(label_str)
        if match:
            # Use the second group (lowercase letter or number)
            normalized = match.group(2).strip().lower()
            count = int((label_col == label).sum())
            if normalized in label_counts:
                label_counts[normalized] += count
            else:
                label_counts[normalized] = count
        else:
            unparseable_labels.append(label_str)

    # Coordinate statistics (all columns except last)
    coord_cols = df.iloc[:, :-1]
    coord_stats = {
        "min": float(coord_cols.min().min()),
        "max": float(coord_cols.max().max()),
        "mean": float(coord_cols.mean().mean()),
        "has_negatives": bool((coord_cols < 0).any().any()),
        "has_out_of_range": bool(((coord_cols < 0) | (coord_cols > 1)).any().any()),
    }

    # Check for missing values
    missing_count = int(df.isna().sum().sum())

    # Map labels to our sign IDs
    mapped_labels: dict[str, str] = {}
    unmapped_labels: list[str] = []
    for label in label_counts:
        if label in LABEL_TO_SIGN_ID:
            mapped_labels[label] = LABEL_TO_SIGN_ID[label]
        else:
            unmapped_labels.append(label)

    return {
        "file_name": filepath.name,
        "file_path": str(filepath),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "row_count": row_count,
        "column_count": col_count,
        "dataset_type": dataset_type,
        "expected_landmarks": NUM_LANDMARKS if dataset_type == "one_hand" else NUM_LANDMARKS * 2,
        "labels": {
            "unique_count": len(label_counts),
            "counts": label_counts,
            "unparseable": unparseable_labels,
        },
        "label_mapping": {
            "mapped": mapped_labels,
            "unmapped": unmapped_labels,
        },
        "coordinates": coord_stats,
        "missing_values": missing_count,
    }


# =============================================================================
# Import Functions
# =============================================================================


def parse_label(raw_label: str) -> str | None:
    """Parse a Kaggle label string and return normalized form.

    Args:
        raw_label: Label like "A - a" or "Eight - 8"

    Returns:
        Normalized label (lowercase) or None if unparseable.
    """
    label_str = str(raw_label).strip()
    match = LABEL_PATTERN.match(label_str)
    if match:
        return match.group(2).strip().lower()
    return None


def parse_landmarks_one_hand(
    row: pd.Series,
) -> HandLandmarks | None:
    """Parse a single row of one-hand data into HandLandmarks.

    Args:
        row: Pandas Series with 63 coordinate values (21 landmarks × 3).

    Returns:
        HandLandmarks object or None if validation fails.
    """
    try:
        landmarks = []
        for i in range(NUM_LANDMARKS):
            x = float(row.iloc[i * 3])
            y = float(row.iloc[i * 3 + 1])
            z = float(row.iloc[i * 3 + 2])

            # Clamp x,y to valid range (some data may be slightly out)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))

            point = Point3D(x=x, y=y, z=z)
            landmark = HandLandmark(
                point=point,
                name=LANDMARK_NAMES[i],
                index=i,
            )
            landmarks.append(landmark)

        return HandLandmarks(landmarks=landmarks)

    except (ValueError, ValidationError) as e:
        logger.debug(f"Failed to parse landmarks: {e}")
        return None


def parse_landmarks_two_hand(
    row: pd.Series,
) -> tuple[HandLandmarks | None, HandLandmarks | None]:
    """Parse a single row of two-hand data into two HandLandmarks.

    Args:
        row: Pandas Series with 126 coordinate values (42 landmarks × 3).

    Returns:
        Tuple of (left_hand, right_hand) HandLandmarks, either may be None.
    """
    # First 21 landmarks are one hand, next 21 are the other
    # Note: Kaggle data doesn't specify which is left/right, we assume first is dominant (right)
    try:
        right_landmarks = []
        left_landmarks = []

        # First hand (assume right/dominant)
        for i in range(NUM_LANDMARKS):
            x = float(row.iloc[i * 3])
            y = float(row.iloc[i * 3 + 1])
            z = float(row.iloc[i * 3 + 2])

            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))

            point = Point3D(x=x, y=y, z=z)
            landmark = HandLandmark(point=point, name=LANDMARK_NAMES[i], index=i)
            right_landmarks.append(landmark)

        # Second hand (assume left/non-dominant)
        offset = NUM_LANDMARKS * 3
        for i in range(NUM_LANDMARKS):
            x = float(row.iloc[offset + i * 3])
            y = float(row.iloc[offset + i * 3 + 1])
            z = float(row.iloc[offset + i * 3 + 2])

            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))

            point = Point3D(x=x, y=y, z=z)
            landmark = HandLandmark(point=point, name=LANDMARK_NAMES[i], index=i)
            left_landmarks.append(landmark)

        right_hand = HandLandmarks(landmarks=right_landmarks)
        left_hand = HandLandmarks(landmarks=left_landmarks)

        return left_hand, right_hand

    except (ValueError, ValidationError) as e:
        logger.debug(f"Failed to parse two-hand landmarks: {e}")
        return None, None


def import_kaggle_csv(
    filepath: Path,
) -> list[tuple[str, HandLandmarks | tuple[HandLandmarks, HandLandmarks]]]:
    """Import a Kaggle CSV file into (label, landmarks) pairs.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of (normalized_label, landmarks) tuples.
        For one-hand: landmarks is HandLandmarks
        For two-hand: landmarks is tuple of (left, right) HandLandmarks
    """
    filepath = Path(filepath)
    dataset_type = detect_dataset_type(filepath)

    if dataset_type == "unknown":
        raise ValueError(f"Unknown dataset type for {filepath}")

    logger.info(f"Importing {dataset_type} dataset: {filepath.name}")

    df = pd.read_csv(filepath, header=None)
    total_rows = len(df)

    samples: list[tuple[str, Any]] = []
    skipped_rows = 0
    invalid_landmarks = 0
    unmapped_labels = 0

    for idx, row in df.iterrows():
        # Parse label
        raw_label = row.iloc[-1]
        label = parse_label(raw_label)

        if label is None:
            skipped_rows += 1
            logger.debug(f"Row {idx}: Unparseable label '{raw_label}'")
            continue

        if label not in LABEL_TO_SIGN_ID:
            unmapped_labels += 1
            logger.debug(f"Row {idx}: Unmapped label '{label}'")
            continue

        # Parse landmarks
        if dataset_type == "one_hand":
            landmarks = parse_landmarks_one_hand(row.iloc[:-1])
            if landmarks is None:
                invalid_landmarks += 1
                continue
            samples.append((label, landmarks))
        else:
            left, right = parse_landmarks_two_hand(row.iloc[:-1])
            if left is None or right is None:
                invalid_landmarks += 1
                continue
            samples.append((label, (left, right)))

        # Progress logging
        if (idx + 1) % 5000 == 0:
            logger.info(f"  Processed {idx + 1}/{total_rows} rows...")

    logger.info(
        f"Import complete: {len(samples)} valid samples from {total_rows} rows"
    )
    if skipped_rows > 0:
        logger.warning(f"  Skipped {skipped_rows} rows (unparseable labels)")
    if invalid_landmarks > 0:
        logger.warning(f"  Skipped {invalid_landmarks} rows (invalid landmarks)")
    if unmapped_labels > 0:
        logger.warning(f"  Skipped {unmapped_labels} rows (unmapped labels)")

    return samples


# =============================================================================
# Conversion Functions
# =============================================================================


def convert_to_recordings(
    samples: list[tuple[str, Any]],
    dataset_type: DatasetType,
    source_file: str,
) -> dict[str, list[Recording]]:
    """Convert imported samples to Recording objects grouped by sign ID.

    Args:
        samples: List of (label, landmarks) tuples from import.
        dataset_type: Whether this is one_hand or two_hand data.
        source_file: Name of the source file for metadata.

    Returns:
        Dictionary mapping sign_id to list of Recording objects.
    """
    recordings_by_sign: dict[str, list[Recording]] = {}
    now = datetime.now(timezone.utc)

    for label, landmarks in samples:
        sign_id = LABEL_TO_SIGN_ID.get(label)
        if sign_id is None:
            continue

        # Create detected hands based on dataset type
        if dataset_type == "one_hand":
            # Single hand - assume right hand for signing
            right_hand = DetectedHand(
                landmarks=landmarks,
                handedness="Right",
                confidence=1.0,
            )
            frame = RecordingFrame(
                timestamp_ms=0,
                left_hand=None,
                right_hand=right_hand,
                frame_confidence=1.0,
            )
        else:
            # Two hands
            left_landmarks, right_landmarks = landmarks
            left_hand = DetectedHand(
                landmarks=left_landmarks,
                handedness="Left",
                confidence=1.0,
            )
            right_hand = DetectedHand(
                landmarks=right_landmarks,
                handedness="Right",
                confidence=1.0,
            )
            frame = RecordingFrame(
                timestamp_ms=0,
                left_hand=left_hand,
                right_hand=right_hand,
                frame_confidence=1.0,
            )

        # Create single-frame recording
        recording = Recording(
            frames=[frame],
            sign_id=sign_id,
            start_time=now,
            end_time=now,
            metadata={
                "source": "kaggle",
                "source_file": source_file,
                "original_label": label,
                "dataset_type": dataset_type,
            },
        )

        if sign_id not in recordings_by_sign:
            recordings_by_sign[sign_id] = []
        recordings_by_sign[sign_id].append(recording)

    return recordings_by_sign


def save_recordings(
    recordings_by_sign: dict[str, list[Recording]],
    output_dir: Path,
) -> dict[str, int]:
    """Save recordings to JSON files in output directory.

    Args:
        recordings_by_sign: Dictionary of sign_id -> recordings list.
        output_dir: Directory to save files to.

    Returns:
        Dictionary mapping sign_id to count of saved recordings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_counts: dict[str, int] = {}

    for sign_id, recordings in recordings_by_sign.items():
        # Convert to JSON-serializable format
        recordings_data = [r.model_dump(mode="json") for r in recordings]

        output_file = output_dir / f"{sign_id}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "sign_id": sign_id,
                    "count": len(recordings),
                    "recordings": recordings_data,
                },
                f,
                indent=2,
                default=str,  # Handle datetime serialization
            )

        saved_counts[sign_id] = len(recordings)
        logger.debug(f"Saved {len(recordings)} recordings to {output_file}")

    return saved_counts


# =============================================================================
# CLI Interface
# =============================================================================


@click.group()
def cli():
    """Kaggle BSL dataset importer."""
    pass


@cli.command()
@click.argument("filepath", type=click.Path(exists=True, path_type=Path))
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def analyze(filepath: Path, output_json: bool):
    """Analyze a Kaggle CSV file structure.

    FILEPATH: Path to the CSV file to analyze.
    """
    try:
        analysis = analyze_kaggle_file(filepath)

        if output_json:
            click.echo(json.dumps(analysis, indent=2))
        else:
            click.echo(f"\n{'='*60}")
            click.echo(f"File Analysis: {analysis['file_name']}")
            click.echo(f"{'='*60}")
            click.echo(f"Size: {analysis['file_size_mb']} MB")
            click.echo(f"Rows: {analysis['row_count']:,}")
            click.echo(f"Columns: {analysis['column_count']}")
            click.echo(f"Dataset Type: {analysis['dataset_type']}")
            click.echo(f"Expected Landmarks: {analysis['expected_landmarks']}")
            click.echo(f"Missing Values: {analysis['missing_values']}")

            click.echo(f"\nCoordinate Stats:")
            coords = analysis["coordinates"]
            click.echo(f"  Range: [{coords['min']:.4f}, {coords['max']:.4f}]")
            click.echo(f"  Mean: {coords['mean']:.4f}")
            click.echo(f"  Has negatives: {coords['has_negatives']}")
            click.echo(f"  Has out-of-range: {coords['has_out_of_range']}")

            click.echo(f"\nLabels ({analysis['labels']['unique_count']} unique):")
            for label, count in sorted(analysis["labels"]["counts"].items()):
                sign_id = LABEL_TO_SIGN_ID.get(label, "UNMAPPED")
                click.echo(f"  {label}: {count:,} samples -> {sign_id}")

            if analysis["labels"]["unparseable"]:
                click.echo(f"\nUnparseable labels: {analysis['labels']['unparseable']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command("import")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("data/processed/kaggle"),
    help="Output directory for processed files",
)
@click.option("--file", "-f", multiple=True, help="Specific file(s) to import")
def import_cmd(input_path: Path, output: Path, file: tuple[str, ...]):
    """Import Kaggle dataset from directory or file.

    INPUT_PATH: Path to CSV file or directory containing CSV files.
    """
    try:
        files_to_process: list[Path] = []

        if input_path.is_file():
            files_to_process.append(input_path)
        elif input_path.is_dir():
            if file:
                # Specific files requested
                for f in file:
                    fp = input_path / f
                    if fp.exists():
                        files_to_process.append(fp)
                    else:
                        logger.warning(f"File not found: {fp}")
            else:
                # Find all CSV files
                files_to_process = list(input_path.glob("*.csv"))

        if not files_to_process:
            click.echo("No CSV files found to process.", err=True)
            raise SystemExit(1)

        click.echo(f"\nProcessing {len(files_to_process)} file(s)...")

        all_recordings: dict[str, list[Recording]] = {}

        for filepath in files_to_process:
            click.echo(f"\n{'='*60}")
            click.echo(f"Importing: {filepath.name}")
            click.echo(f"{'='*60}")

            dataset_type = detect_dataset_type(filepath)
            samples = import_kaggle_csv(filepath)

            recordings = convert_to_recordings(
                samples, dataset_type, filepath.name
            )

            # Merge with existing recordings
            for sign_id, recs in recordings.items():
                if sign_id not in all_recordings:
                    all_recordings[sign_id] = []
                all_recordings[sign_id].extend(recs)

        # Save all recordings
        click.echo(f"\nSaving to: {output}")
        saved_counts = save_recordings(all_recordings, output)

        # Summary
        click.echo(f"\n{'='*60}")
        click.echo("Import Summary")
        click.echo(f"{'='*60}")
        total = sum(saved_counts.values())
        click.echo(f"Total recordings: {total:,}")
        click.echo(f"Unique signs: {len(saved_counts)}")
        click.echo(f"\nBy sign:")
        for sign_id in sorted(saved_counts.keys()):
            click.echo(f"  {sign_id}: {saved_counts[sign_id]:,}")

    except Exception as e:
        logger.exception("Import failed")
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
def list_mappings():
    """List all label-to-sign-ID mappings."""
    click.echo("\nLabel to Sign ID Mappings:")
    click.echo("-" * 40)

    # Group by category
    alphabet = {k: v for k, v in LABEL_TO_SIGN_ID.items() if v.startswith("bsl_alphabet")}
    numbers = {k: v for k, v in LABEL_TO_SIGN_ID.items() if v.startswith("bsl_numbers")}

    click.echo("\nAlphabet:")
    for label in sorted(alphabet.keys()):
        click.echo(f"  {label} -> {alphabet[label]}")

    click.echo("\nNumbers:")
    for label in sorted(numbers.keys(), key=lambda x: int(x) if x.isdigit() else 99):
        click.echo(f"  {label} -> {numbers[label]}")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
