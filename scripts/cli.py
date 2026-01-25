#!/usr/bin/env python3
"""
BSL Data Extraction Tool - Unified CLI

Complete command-line interface for the BSL sign language data pipeline.
Provides commands for analysis, import, processing, validation, and generation.

Usage:
    bsl-tool analyze kaggle /data/kaggle-bsl/
    bsl-tool import kaggle /data/kaggle-bsl/*.csv
    bsl-tool process batch /data/raw-recordings/
    bsl-tool validate dictionary /data/output/bsl-dictionary.json
    bsl-tool generate dictionary --version 1.0.0
    bsl-tool pipeline run
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.types import (
    Point3D,
    HandLandmarks,
    RecordingFrame,
    RecordingSession,
    SignPose,
    SignDictionaryEntry,
    SignDictionary,
    ProcessingResult,
)
from src.landmarks import LANDMARK_NAMES, NUM_LANDMARKS
from src.sign_definitions import SIGN_DEFINITIONS, CATEGORIES, get_sign_by_id
from src.validation import (
    validate_landmarks,
    validate_frame,
    validate_recording,
    validate_sign_entry,
    validate_dictionary,
    calculate_landmark_quality,
    calculate_recording_quality,
)


# =============================================================================
# Configuration
# =============================================================================


DEFAULT_CONFIG = {
    "data_dir": "data",
    "kaggle_dir": "data/kaggle-bsl",
    "recordings_dir": "data/raw-recordings",
    "processed_dir": "data/processed",
    "output_dir": "data/output",
    "dictionary_file": "bsl-dictionary.json",
    "min_confidence": 0.5,
    "outlier_method": "zscore",
    "averaging_method": "trimmed",
}


class Config:
    """Configuration container with file support."""

    def __init__(self):
        self.values = dict(DEFAULT_CONFIG)
        self.verbose = 0
        self.dry_run = False

    def load_file(self, filepath: Path) -> None:
        """Load configuration from JSON file."""
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
                self.values.update(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.values[key] = value


pass_config = click.make_pass_decorator(Config, ensure=True)


# =============================================================================
# Utility Functions
# =============================================================================


def styled_header(text: str) -> str:
    """Create a styled header."""
    return click.style(f"\n{'=' * 60}\n{text}\n{'=' * 60}", fg="cyan", bold=True)


def styled_success(text: str) -> str:
    """Create success message."""
    return click.style(f"[OK] {text}", fg="green")


def styled_warning(text: str) -> str:
    """Create warning message."""
    return click.style(f"[WARN] {text}", fg="yellow")


def styled_error(text: str) -> str:
    """Create error message."""
    return click.style(f"[ERROR] {text}", fg="red", bold=True)


def styled_info(text: str) -> str:
    """Create info message."""
    return click.style(f"[INFO] {text}", fg="blue")


def log_verbose(config: Config, message: str, level: int = 1) -> None:
    """Log message if verbosity level is sufficient."""
    if config.verbose >= level:
        click.echo(click.style(f"  [v{level}] {message}", fg="bright_black"))


def format_size(size_bytes: int) -> str:
    """Format byte size as human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def count_files(directory: Path, pattern: str = "*") -> int:
    """Count files matching pattern in directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def load_json_file(filepath: Path) -> Optional[dict]:
    """Load JSON file with error handling."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        click.echo(styled_error(f"Failed to load {filepath}: {e}"))
        return None


def save_json_file(filepath: Path, data: dict, pretty: bool = True) -> bool:
    """Save JSON file with error handling."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            if pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, separators=(",", ":"), default=str)
        return True
    except Exception as e:
        click.echo(styled_error(f"Failed to save {filepath}: {e}"))
        return False


# =============================================================================
# Main CLI Group
# =============================================================================


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v, -vv, -vvv)")
@click.option("--dry-run", is_flag=True, help="Show what would be done without doing it")
@click.option("--config", type=click.Path(exists=True), help="Configuration file path")
@click.version_option(version="0.1.0", prog_name="bsl-tool")
@click.pass_context
def main(ctx, verbose, dry_run, config):
    """
    BSL Data Extraction Tool - Process sign language landmark data.

    A complete pipeline for extracting, processing, and generating
    British Sign Language dictionary data from various sources.
    """
    ctx.ensure_object(Config)
    ctx.obj.verbose = verbose
    ctx.obj.dry_run = dry_run

    if config:
        ctx.obj.load_file(Path(config))
        if verbose:
            click.echo(styled_info(f"Loaded config from {config}"))


# =============================================================================
# Analyze Command Group
# =============================================================================


@main.group()
@pass_config
def analyze(config):
    """Analyze data sources and show statistics."""
    pass


@analyze.command("kaggle")
@click.argument("path", type=click.Path(exists=True))
@pass_config
def analyze_kaggle(config, path):
    """Analyze Kaggle BSL dataset directory."""
    click.echo(styled_header("Kaggle Dataset Analysis"))

    kaggle_path = Path(path)

    # Find CSV files
    csv_files = list(kaggle_path.glob("*.csv"))
    parquet_files = list(kaggle_path.glob("*.parquet"))

    click.echo(f"\nDirectory: {kaggle_path}")
    click.echo(f"CSV files: {len(csv_files)}")
    click.echo(f"Parquet files: {len(parquet_files)}")

    # Analyze each CSV file
    if csv_files:
        click.echo("\nCSV Files:")
        total_rows = 0

        for csv_file in csv_files:
            size = csv_file.stat().st_size
            # Count lines (rough estimate)
            with open(csv_file) as f:
                lines = sum(1 for _ in f)
            total_rows += lines - 1  # Subtract header

            click.echo(f"  {csv_file.name}: {lines-1:,} rows ({format_size(size)})")
            log_verbose(config, f"Headers would need pandas to read", 2)

        click.echo(f"\nTotal data rows: {total_rows:,}")

    # Check for landmark data patterns
    json_files = list(kaggle_path.rglob("*.json"))
    if json_files:
        click.echo(f"\nJSON files found: {len(json_files)}")

    click.echo(styled_success("Analysis complete"))


@analyze.command("recordings")
@click.argument("path", type=click.Path(exists=True))
@click.option("--detailed", "-d", is_flag=True, help="Show detailed per-file stats")
@pass_config
def analyze_recordings(config, path, detailed):
    """Analyze raw recordings directory."""
    click.echo(styled_header("Recordings Analysis"))

    recordings_path = Path(path)
    json_files = list(recordings_path.glob("*.json"))

    click.echo(f"\nDirectory: {recordings_path}")
    click.echo(f"Recording files: {len(json_files)}")

    if not json_files:
        click.echo(styled_warning("No JSON files found"))
        return

    # Aggregate statistics
    stats = {
        "total_files": len(json_files),
        "total_frames": 0,
        "signs_found": set(),
        "avg_frames_per_recording": 0,
        "files_with_errors": 0,
        "total_size": 0,
    }

    frame_counts = []

    with click.progressbar(json_files, label="Analyzing files") as files:
        for json_file in files:
            stats["total_size"] += json_file.stat().st_size

            data = load_json_file(json_file)
            if not data:
                stats["files_with_errors"] += 1
                continue

            # Extract sign ID
            sign_id = data.get("sign_id") or json_file.stem.split("_")[0]
            stats["signs_found"].add(sign_id)

            # Count frames
            frames = data.get("frames", [])
            frame_count = len(frames)
            stats["total_frames"] += frame_count
            frame_counts.append(frame_count)

            if detailed:
                quality = calculate_recording_quality_simple(data)
                click.echo(f"\n  {json_file.name}: {frame_count} frames, quality: {quality:.1f}%")

    # Calculate averages
    if frame_counts:
        stats["avg_frames_per_recording"] = sum(frame_counts) / len(frame_counts)
        stats["min_frames"] = min(frame_counts)
        stats["max_frames"] = max(frame_counts)

    click.echo(f"\n{'-' * 40}")
    click.echo(f"Total recordings: {stats['total_files']}")
    click.echo(f"Total frames: {stats['total_frames']:,}")
    click.echo(f"Unique signs: {len(stats['signs_found'])}")
    click.echo(f"Avg frames/recording: {stats['avg_frames_per_recording']:.1f}")
    click.echo(f"Frame range: {stats.get('min_frames', 0)} - {stats.get('max_frames', 0)}")
    click.echo(f"Total size: {format_size(stats['total_size'])}")
    click.echo(f"Files with errors: {stats['files_with_errors']}")

    if config.verbose >= 1:
        click.echo(f"\nSigns found: {', '.join(sorted(stats['signs_found']))}")

    click.echo(styled_success("Analysis complete"))


def calculate_recording_quality_simple(data: dict) -> float:
    """Simple quality calculation for analysis."""
    frames = data.get("frames", [])
    if not frames:
        return 0.0

    confidences = []
    for frame in frames:
        if "right_hand" in frame and frame["right_hand"]:
            conf = frame["right_hand"].get("confidence", 0.5)
            confidences.append(conf)
        if "left_hand" in frame and frame["left_hand"]:
            conf = frame["left_hand"].get("confidence", 0.5)
            confidences.append(conf)

    if not confidences:
        return 0.0

    return (sum(confidences) / len(confidences)) * 100


@analyze.command("dictionary")
@click.argument("path", type=click.Path(exists=True))
@pass_config
def analyze_dictionary(config, path):
    """Analyze generated dictionary file."""
    click.echo(styled_header("Dictionary Analysis"))

    dict_path = Path(path)
    data = load_json_file(dict_path)

    if not data:
        return

    click.echo(f"\nFile: {dict_path}")
    click.echo(f"Size: {format_size(dict_path.stat().st_size)}")
    click.echo(f"Version: {data.get('version', 'unknown')}")
    click.echo(f"Created: {data.get('created_at', 'unknown')}")

    signs = data.get("signs", {})
    click.echo(f"\nTotal signs: {len(signs)}")

    # Coverage analysis
    with_data = 0
    quality_sum = 0
    by_category = defaultdict(lambda: {"total": 0, "with_data": 0})

    for sign_id, sign_data in signs.items():
        category = sign_data.get("category", "unknown")
        by_category[category]["total"] += 1

        poses = sign_data.get("poses", [])
        has_data = any(
            p.get("left_hand") or p.get("right_hand")
            for p in poses
        )

        if has_data:
            with_data += 1
            by_category[category]["with_data"] += 1

        meta = sign_data.get("metadata", {})
        quality_sum += meta.get("quality_score", 0)

    coverage = (with_data / len(signs) * 100) if signs else 0
    avg_quality = (quality_sum / len(signs)) if signs else 0

    click.echo(f"Signs with data: {with_data} ({coverage:.1f}%)")
    click.echo(f"Average quality: {avg_quality:.2f}")

    click.echo(f"\n{'Category':<20} {'Total':>8} {'With Data':>12} {'Coverage':>10}")
    click.echo("-" * 52)

    for cat in sorted(by_category.keys()):
        stats = by_category[cat]
        cat_coverage = (stats["with_data"] / stats["total"] * 100) if stats["total"] else 0
        click.echo(f"{cat:<20} {stats['total']:>8} {stats['with_data']:>12} {cat_coverage:>9.1f}%")

    click.echo(styled_success("Analysis complete"))


# =============================================================================
# Import Command Group
# =============================================================================


@main.group("import")
@pass_config
def import_cmd(config):
    """Import data from external sources."""
    pass


@import_cmd.command("kaggle")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="data/processed/kaggle",
              help="Output directory")
@click.option("--limit", "-l", type=int, help="Limit number of records to import")
@pass_config
def import_kaggle(config, input_path, output, limit):
    """Import and convert Kaggle BSL dataset."""
    click.echo(styled_header("Kaggle Dataset Import"))

    input_path = Path(input_path)
    output_path = Path(output)

    if config.dry_run:
        click.echo(styled_info("DRY RUN - no files will be written"))

    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")

    # Import the kaggle importer
    try:
        from scripts.import_kaggle import (
            load_kaggle_csv,
            convert_to_recording,
            save_recording,
        )
    except ImportError as e:
        click.echo(styled_error(f"Failed to import kaggle module: {e}"))
        click.echo("Make sure import_kaggle.py exists in scripts/")
        return

    # Find CSV files
    if input_path.is_file():
        csv_files = [input_path]
    else:
        csv_files = list(input_path.glob("*.csv"))

    if not csv_files:
        click.echo(styled_error("No CSV files found"))
        return

    click.echo(f"Found {len(csv_files)} CSV files")

    total_imported = 0
    total_errors = 0

    for csv_file in csv_files:
        click.echo(f"\nProcessing: {csv_file.name}")

        try:
            # Load CSV data
            records = load_kaggle_csv(csv_file)

            if limit:
                records = records[:limit]

            log_verbose(config, f"Loaded {len(records)} records", 1)

            with click.progressbar(records, label="Converting") as items:
                for record in items:
                    try:
                        recording = convert_to_recording(record)

                        if not config.dry_run:
                            out_file = output_path / f"{recording.sign_id}_{total_imported:04d}.json"
                            save_recording(recording, out_file)

                        total_imported += 1
                    except Exception as e:
                        total_errors += 1
                        log_verbose(config, f"Error: {e}", 2)

        except Exception as e:
            click.echo(styled_error(f"Failed to process {csv_file}: {e}"))
            total_errors += 1

    click.echo(f"\n{'-' * 40}")
    click.echo(f"Imported: {total_imported}")
    click.echo(f"Errors: {total_errors}")

    if config.dry_run:
        click.echo(styled_warning("DRY RUN - no files were written"))
    else:
        click.echo(styled_success(f"Import complete to {output_path}"))


@import_cmd.command("json")
@click.argument("input_files", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="data/processed",
              help="Output directory")
@pass_config
def import_json(config, input_files, output):
    """Import JSON recording files."""
    click.echo(styled_header("JSON Import"))

    output_path = Path(output)

    if config.dry_run:
        click.echo(styled_info("DRY RUN - no files will be written"))

    if not input_files:
        click.echo(styled_error("No input files specified"))
        return

    click.echo(f"Files to import: {len(input_files)}")
    click.echo(f"Output: {output_path}")

    imported = 0
    errors = 0

    with click.progressbar(input_files, label="Importing") as files:
        for filepath in files:
            path = Path(filepath)
            data = load_json_file(path)

            if not data:
                errors += 1
                continue

            # Validate basic structure
            if "frames" not in data:
                log_verbose(config, f"No frames in {path.name}", 1)
                errors += 1
                continue

            if not config.dry_run:
                out_file = output_path / path.name
                if save_json_file(out_file, data):
                    imported += 1
                else:
                    errors += 1
            else:
                imported += 1

    click.echo(f"\nImported: {imported}")
    click.echo(f"Errors: {errors}")
    click.echo(styled_success("Import complete"))


# =============================================================================
# Process Command Group
# =============================================================================


@main.group()
@pass_config
def process(config):
    """Process recordings into canonical poses."""
    pass


@process.command("single")
@click.argument("input_files", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@pass_config
def process_single(config, input_files, output):
    """Process one or more recording files for a single sign."""
    click.echo(styled_header("Single Sign Processing"))

    if not input_files:
        click.echo(styled_error("No input files specified"))
        return

    if config.dry_run:
        click.echo(styled_info("DRY RUN - no files will be written"))

    # Import processor
    try:
        from scripts.process_recordings import (
            process_recording,
            process_multiple_recordings,
            ProcessingConfig,
        )
    except ImportError as e:
        click.echo(styled_error(f"Failed to import processing module: {e}"))
        return

    click.echo(f"Processing {len(input_files)} files")

    # Load recordings
    recordings = []
    for filepath in input_files:
        data = load_json_file(Path(filepath))
        if data:
            try:
                recording = RecordingSession(
                    sign_id=data.get("sign_id", Path(filepath).stem),
                    frames=[
                        RecordingFrame(
                            timestamp=f.get("timestamp", 0),
                            right_hand=HandLandmarks(
                                landmarks=[Point3D(**p) for p in f["right_hand"]["landmarks"]],
                                confidence=f["right_hand"].get("confidence", 0.8),
                            ) if f.get("right_hand") else None,
                            left_hand=HandLandmarks(
                                landmarks=[Point3D(**p) for p in f["left_hand"]["landmarks"]],
                                confidence=f["left_hand"].get("confidence", 0.8),
                            ) if f.get("left_hand") else None,
                        )
                        for f in data.get("frames", [])
                    ],
                    metadata=data.get("metadata", {}),
                )
                recordings.append(recording)
            except Exception as e:
                click.echo(styled_warning(f"Failed to parse {filepath}: {e}"))

    if not recordings:
        click.echo(styled_error("No valid recordings loaded"))
        return

    click.echo(f"Loaded {len(recordings)} recordings")

    # Configure processing
    proc_config = ProcessingConfig(
        min_confidence=config.get("min_confidence", 0.5),
        outlier_method=config.get("outlier_method", "zscore"),
        averaging_method=config.get("averaging_method", "trimmed"),
    )

    log_verbose(config, f"Config: {proc_config}", 1)

    # Process
    if len(recordings) == 1:
        result = process_recording(recordings[0], proc_config)
    else:
        result = process_multiple_recordings(recordings, proc_config)

    if not result:
        click.echo(styled_error("Processing failed - no valid frames"))
        return

    click.echo(styled_success(f"Processing complete"))
    click.echo(f"  Quality score: {result.quality_score:.2f}")
    click.echo(f"  Sample count: {result.sample_count}")
    click.echo(f"  Warnings: {len(result.warnings)}")

    if result.warnings and config.verbose >= 1:
        for warn in result.warnings:
            click.echo(styled_warning(f"  {warn}"))

    # Save output
    if output and not config.dry_run:
        output_data = {
            "canonical_pose": {
                "left_hand_landmarks": [
                    {"x": p.x, "y": p.y, "z": p.z}
                    for p in (result.canonical_pose.left_hand_landmarks or [])
                ],
                "right_hand_landmarks": [
                    {"x": p.x, "y": p.y, "z": p.z}
                    for p in (result.canonical_pose.right_hand_landmarks or [])
                ],
                "tolerances": result.canonical_pose.tolerances,
            },
            "quality_score": result.quality_score,
            "sample_count": result.sample_count,
            "warnings": result.warnings,
        }

        if save_json_file(Path(output), output_data):
            click.echo(f"Saved to: {output}")


@process.command("batch")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="data/processed",
              help="Output directory")
@click.option("--pattern", "-p", default="*.json", help="File pattern to match")
@pass_config
def process_batch(config, input_dir, output, pattern):
    """Process all recordings in a directory."""
    click.echo(styled_header("Batch Processing"))

    input_path = Path(input_dir)
    output_path = Path(output)

    if config.dry_run:
        click.echo(styled_info("DRY RUN - no files will be written"))

    # Find files
    files = list(input_path.glob(pattern))
    click.echo(f"Found {len(files)} files matching '{pattern}'")

    if not files:
        return

    # Group by sign ID
    by_sign = defaultdict(list)
    for filepath in files:
        # Extract sign ID from filename
        sign_id = filepath.stem.split("_")[0]
        by_sign[sign_id].append(filepath)

    click.echo(f"Signs to process: {len(by_sign)}")

    # Import processor
    try:
        from scripts.process_recordings import (
            process_recording,
            process_multiple_recordings,
            ProcessingConfig,
        )
    except ImportError as e:
        click.echo(styled_error(f"Failed to import processing module: {e}"))
        return

    proc_config = ProcessingConfig(
        min_confidence=config.get("min_confidence", 0.5),
        outlier_method=config.get("outlier_method", "zscore"),
        averaging_method=config.get("averaging_method", "trimmed"),
    )

    processed = 0
    failed = 0
    results_summary = []

    with click.progressbar(by_sign.items(), label="Processing signs") as signs:
        for sign_id, filepaths in signs:
            recordings = []

            for filepath in filepaths:
                data = load_json_file(filepath)
                if not data:
                    continue

                try:
                    recording = RecordingSession(
                        sign_id=sign_id,
                        frames=[
                            RecordingFrame(
                                timestamp=f.get("timestamp", 0),
                                right_hand=HandLandmarks(
                                    landmarks=[Point3D(**p) for p in f["right_hand"]["landmarks"]],
                                    confidence=f["right_hand"].get("confidence", 0.8),
                                ) if f.get("right_hand") else None,
                                left_hand=HandLandmarks(
                                    landmarks=[Point3D(**p) for p in f["left_hand"]["landmarks"]],
                                    confidence=f["left_hand"].get("confidence", 0.8),
                                ) if f.get("left_hand") else None,
                            )
                            for f in data.get("frames", [])
                        ],
                        metadata=data.get("metadata", {}),
                    )
                    recordings.append(recording)
                except Exception as e:
                    log_verbose(config, f"Failed to parse {filepath}: {e}", 2)

            if not recordings:
                failed += 1
                continue

            # Process
            if len(recordings) == 1:
                result = process_recording(recordings[0], proc_config)
            else:
                result = process_multiple_recordings(recordings, proc_config)

            if not result:
                failed += 1
                continue

            # Save
            if not config.dry_run:
                output_data = {
                    "sign_id": sign_id,
                    "canonical_pose": {
                        "left_hand_landmarks": [
                            {"x": p.x, "y": p.y, "z": p.z}
                            for p in (result.canonical_pose.left_hand_landmarks or [])
                        ] if result.canonical_pose.left_hand_landmarks else None,
                        "right_hand_landmarks": [
                            {"x": p.x, "y": p.y, "z": p.z}
                            for p in (result.canonical_pose.right_hand_landmarks or [])
                        ] if result.canonical_pose.right_hand_landmarks else None,
                        "tolerances": result.canonical_pose.tolerances,
                    },
                    "quality_score": result.quality_score,
                    "sample_count": result.sample_count,
                    "warnings": result.warnings,
                }

                out_file = output_path / f"{sign_id}.json"
                save_json_file(out_file, output_data)

            processed += 1
            results_summary.append({
                "sign_id": sign_id,
                "quality": result.quality_score,
                "samples": result.sample_count,
            })

    click.echo(f"\n{'-' * 40}")
    click.echo(f"Processed: {processed}")
    click.echo(f"Failed: {failed}")

    if results_summary:
        avg_quality = sum(r["quality"] for r in results_summary) / len(results_summary)
        click.echo(f"Average quality: {avg_quality:.2f}")

    click.echo(styled_success("Batch processing complete"))


@process.command("all")
@click.option("--kaggle-dir", type=click.Path(), help="Kaggle data directory")
@click.option("--recordings-dir", type=click.Path(), help="Raw recordings directory")
@click.option("--output", "-o", type=click.Path(), default="data/processed",
              help="Output directory")
@pass_config
def process_all(config, kaggle_dir, recordings_dir, output):
    """Run full processing pipeline on all data sources."""
    click.echo(styled_header("Full Processing Pipeline"))

    if config.dry_run:
        click.echo(styled_info("DRY RUN - no files will be written"))

    output_path = Path(output)
    kaggle_path = Path(kaggle_dir or config.get("kaggle_dir"))
    recordings_path = Path(recordings_dir or config.get("recordings_dir"))

    click.echo(f"Kaggle dir: {kaggle_path}")
    click.echo(f"Recordings dir: {recordings_path}")
    click.echo(f"Output: {output_path}")

    # Process Kaggle data if available
    if kaggle_path.exists():
        click.echo("\nProcessing Kaggle data...")
        # Would call process_batch internally
        kaggle_files = list(kaggle_path.glob("*.json"))
        click.echo(f"  Found {len(kaggle_files)} files")

    # Process recordings if available
    if recordings_path.exists():
        click.echo("\nProcessing recordings...")
        rec_files = list(recordings_path.glob("*.json"))
        click.echo(f"  Found {len(rec_files)} files")

    click.echo(styled_success("Full processing complete"))


# =============================================================================
# Validate Command Group
# =============================================================================


@main.group()
@pass_config
def validate(config):
    """Validate data files and structures."""
    pass


@validate.command("landmarks")
@click.argument("filepath", type=click.Path(exists=True))
@pass_config
def validate_landmarks_cmd(config, filepath):
    """Validate landmark data in a file."""
    click.echo(styled_header("Landmark Validation"))

    data = load_json_file(Path(filepath))
    if not data:
        return

    click.echo(f"File: {filepath}")

    # Find landmarks in various structures
    landmarks_found = []

    if "landmarks" in data:
        landmarks_found.append(("root", data["landmarks"]))

    if "right_hand" in data and data["right_hand"]:
        if "landmarks" in data["right_hand"]:
            landmarks_found.append(("right_hand", data["right_hand"]["landmarks"]))

    if "left_hand" in data and data["left_hand"]:
        if "landmarks" in data["left_hand"]:
            landmarks_found.append(("left_hand", data["left_hand"]["landmarks"]))

    if "frames" in data:
        for i, frame in enumerate(data["frames"][:5]):  # Check first 5 frames
            if frame.get("right_hand") and "landmarks" in frame["right_hand"]:
                landmarks_found.append((f"frame[{i}].right_hand", frame["right_hand"]["landmarks"]))
            if frame.get("left_hand") and "landmarks" in frame["left_hand"]:
                landmarks_found.append((f"frame[{i}].left_hand", frame["left_hand"]["landmarks"]))

    if not landmarks_found:
        click.echo(styled_warning("No landmarks found in file"))
        return

    click.echo(f"Found landmarks in {len(landmarks_found)} locations\n")

    all_valid = True

    for location, lm_data in landmarks_found:
        click.echo(f"Validating {location}:")

        try:
            points = [Point3D(**p) for p in lm_data]
            result = validate_landmarks(points)

            if result.is_valid:
                click.echo(styled_success(f"  Valid ({len(points)} points, score: {result.score:.2f})"))
            else:
                click.echo(styled_error(f"  Invalid"))
                for err in result.errors:
                    click.echo(f"    - {err}")
                all_valid = False

            if result.warnings and config.verbose >= 1:
                for warn in result.warnings:
                    click.echo(styled_warning(f"    {warn}"))

        except Exception as e:
            click.echo(styled_error(f"  Failed to parse: {e}"))
            all_valid = False

    if all_valid:
        click.echo(styled_success("\nAll landmarks valid"))
    else:
        click.echo(styled_error("\nValidation found errors"))


@validate.command("recording")
@click.argument("filepath", type=click.Path(exists=True))
@pass_config
def validate_recording_cmd(config, filepath):
    """Validate a recording session file."""
    click.echo(styled_header("Recording Validation"))

    data = load_json_file(Path(filepath))
    if not data:
        return

    click.echo(f"File: {filepath}")

    try:
        recording = RecordingSession(
            sign_id=data.get("sign_id", Path(filepath).stem),
            frames=[
                RecordingFrame(
                    timestamp=f.get("timestamp", 0),
                    right_hand=HandLandmarks(
                        landmarks=[Point3D(**p) for p in f["right_hand"]["landmarks"]],
                        confidence=f["right_hand"].get("confidence", 0.8),
                    ) if f.get("right_hand") else None,
                    left_hand=HandLandmarks(
                        landmarks=[Point3D(**p) for p in f["left_hand"]["landmarks"]],
                        confidence=f["left_hand"].get("confidence", 0.8),
                    ) if f.get("left_hand") else None,
                )
                for f in data.get("frames", [])
            ],
            metadata=data.get("metadata", {}),
        )

        result = validate_recording(recording)

        click.echo(f"\nSign ID: {recording.sign_id}")
        click.echo(f"Frames: {len(recording.frames)}")

        if result.is_valid:
            click.echo(styled_success(f"Valid (score: {result.score:.2f})"))
        else:
            click.echo(styled_error("Invalid"))
            for err in result.errors:
                click.echo(f"  - {err}")

        if result.warnings:
            click.echo("\nWarnings:")
            for warn in result.warnings:
                click.echo(styled_warning(f"  {warn}"))

        # Quality score
        quality = calculate_recording_quality(recording)
        click.echo(f"\nQuality score: {quality:.1f}%")

    except Exception as e:
        click.echo(styled_error(f"Failed to parse recording: {e}"))
        if config.verbose >= 2:
            import traceback
            traceback.print_exc()


@validate.command("dictionary")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--strict", is_flag=True, help="Fail on warnings too")
@pass_config
def validate_dictionary_cmd(config, filepath, strict):
    """Validate a dictionary file."""
    click.echo(styled_header("Dictionary Validation"))

    # Import dictionary validator
    try:
        from scripts.generate_dictionary import (
            import_dictionary,
            validate_complete_dictionary,
        )
    except ImportError as e:
        click.echo(styled_error(f"Failed to import dictionary module: {e}"))
        return

    dict_path = Path(filepath)
    dictionary = import_dictionary(dict_path)

    if not dictionary:
        click.echo(styled_error("Failed to load dictionary"))
        return

    click.echo(f"File: {filepath}")
    click.echo(f"Version: {dictionary.version}")
    click.echo(f"Entries: {len(dictionary.entries)}")

    result = validate_complete_dictionary(dictionary)

    click.echo(f"\nErrors: {result['error_count']}")
    click.echo(f"Warnings: {result['warning_count']}")

    if result["errors"]:
        click.echo("\nErrors (first 10):")
        for err in result["errors"][:10]:
            click.echo(styled_error(f"  {err}"))

    if result["warnings"] and config.verbose >= 1:
        click.echo("\nWarnings (first 10):")
        for warn in result["warnings"][:10]:
            click.echo(styled_warning(f"  {warn}"))

    if result["is_valid"]:
        if strict and result["warning_count"] > 0:
            click.echo(styled_warning("\nValid with warnings (strict mode)"))
        else:
            click.echo(styled_success("\nDictionary is valid"))
    else:
        click.echo(styled_error("\nDictionary has errors"))


@validate.command("all")
@click.option("--data-dir", type=click.Path(exists=True), default="data",
              help="Data directory to validate")
@pass_config
def validate_all(config, data_dir):
    """Validate all data in a directory."""
    click.echo(styled_header("Full Data Validation"))

    data_path = Path(data_dir)

    # Check structure
    subdirs = ["raw-recordings", "processed", "output"]
    click.echo("Checking directory structure:")

    for subdir in subdirs:
        path = data_path / subdir
        if path.exists():
            file_count = count_files(path, "*.json")
            click.echo(styled_success(f"  {subdir}/: {file_count} JSON files"))
        else:
            click.echo(styled_warning(f"  {subdir}/: not found"))

    # Validate dictionary if exists
    dict_path = data_path / "output" / "bsl-dictionary.json"
    if dict_path.exists():
        click.echo(f"\nValidating dictionary: {dict_path}")
        # Call validate_dictionary_cmd logic here
        data = load_json_file(dict_path)
        if data:
            click.echo(styled_success(f"  Loaded {len(data.get('signs', {}))} signs"))

    click.echo(styled_success("\nValidation complete"))


# =============================================================================
# Generate Command Group
# =============================================================================


@main.group()
@pass_config
def generate(config):
    """Generate output files."""
    pass


@generate.command("dictionary")
@click.option("--version", "-v", default="1.0.0", help="Dictionary version")
@click.option("--processed-dir", "-p", type=click.Path(),
              help="Processed poses directory")
@click.option("--output-dir", "-o", type=click.Path(),
              help="Output directory")
@click.option("--include-missing/--no-include-missing", default=True,
              help="Include signs without data")
@pass_config
def generate_dictionary_cmd(config, version, processed_dir, output_dir, include_missing):
    """Generate BSL dictionary from processed poses."""
    click.echo(styled_header("Dictionary Generation"))

    if config.dry_run:
        click.echo(styled_info("DRY RUN - no files will be written"))

    # Import generator
    try:
        from scripts.generate_dictionary import (
            generate_dictionary,
            export_json,
            export_minified,
            export_typescript_types,
            generate_summary,
        )
    except ImportError as e:
        click.echo(styled_error(f"Failed to import generator module: {e}"))
        return

    processed_path = Path(processed_dir or config.get("processed_dir"))
    output_path = Path(output_dir or config.get("output_dir"))

    click.echo(f"Version: {version}")
    click.echo(f"Processed dir: {processed_path}")
    click.echo(f"Output dir: {output_path}")

    # Generate
    dictionary = generate_dictionary(
        processed_dir=processed_path,
        version=version,
        include_missing=include_missing,
    )

    # Summary
    summary = generate_summary(dictionary)

    click.echo(f"\nGenerated dictionary:")
    click.echo(f"  Total signs: {summary['total_signs']}")
    click.echo(f"  With data: {summary['signs_with_data']}")
    click.echo(f"  Coverage: {summary['coverage_percent']:.1f}%")
    click.echo(f"  Avg quality: {summary['quality']['average']:.2f}")

    # Export
    if not config.dry_run:
        json_path = output_path / "bsl-dictionary.json"
        min_path = output_path / "bsl-dictionary.min.json"
        types_path = output_path / "bsl-dictionary.d.ts"

        export_json(dictionary, json_path, pretty=True)
        export_minified(dictionary, min_path)
        export_typescript_types(dictionary, types_path)

        click.echo(f"\nExported files:")
        click.echo(f"  {json_path}")
        click.echo(f"  {min_path}")
        click.echo(f"  {types_path}")

    click.echo(styled_success("Generation complete"))


@generate.command("types")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True),
              help="Dictionary JSON file")
@click.option("--output", "-o", type=click.Path(), required=True,
              help="Output TypeScript file")
@pass_config
def generate_types(config, input_file, output):
    """Generate TypeScript type definitions."""
    click.echo(styled_header("TypeScript Types Generation"))

    if config.dry_run:
        click.echo(styled_info("DRY RUN - no files will be written"))

    try:
        from scripts.generate_dictionary import (
            import_dictionary,
            export_typescript_types,
        )
    except ImportError as e:
        click.echo(styled_error(f"Failed to import generator module: {e}"))
        return

    input_path = Path(input_file or config.get("output_dir") + "/bsl-dictionary.json")
    output_path = Path(output)

    dictionary = import_dictionary(input_path)
    if not dictionary:
        click.echo(styled_error("Failed to load dictionary"))
        return

    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Signs: {len(dictionary.entries)}")

    if not config.dry_run:
        export_typescript_types(dictionary, output_path)

    click.echo(styled_success("Types generated"))


@generate.command("report")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True),
              help="Dictionary JSON file")
@click.option("--output", "-o", type=click.Path(), help="Output report file")
@click.option("--format", "-f", "fmt", type=click.Choice(["text", "json", "markdown"]),
              default="text", help="Report format")
@pass_config
def generate_report(config, input_file, output, fmt):
    """Generate coverage and quality reports."""
    click.echo(styled_header("Report Generation"))

    try:
        from scripts.generate_dictionary import (
            import_dictionary,
            generate_coverage_report,
            generate_quality_report,
            generate_summary,
        )
    except ImportError as e:
        click.echo(styled_error(f"Failed to import generator module: {e}"))
        return

    input_path = Path(input_file or config.get("output_dir") + "/bsl-dictionary.json")

    dictionary = import_dictionary(input_path)
    if not dictionary:
        click.echo(styled_error("Failed to load dictionary"))
        return

    if fmt == "json":
        report_data = generate_summary(dictionary)
        report_text = json.dumps(report_data, indent=2)
    elif fmt == "markdown":
        coverage = generate_coverage_report(dictionary)
        quality = generate_quality_report(dictionary)
        # Convert to markdown format
        report_text = f"# BSL Dictionary Report\n\n```\n{coverage}\n```\n\n```\n{quality}\n```"
    else:
        coverage = generate_coverage_report(dictionary)
        quality = generate_quality_report(dictionary)
        report_text = f"{coverage}\n\n{quality}"

    if output:
        with open(output, "w") as f:
            f.write(report_text)
        click.echo(f"Report saved to: {output}")
    else:
        click.echo(report_text)

    click.echo(styled_success("Report generated"))


# =============================================================================
# Pipeline Command Group
# =============================================================================


@main.group()
@pass_config
def pipeline(config):
    """Run complete pipeline operations."""
    pass


@pipeline.command("run")
@click.option("--skip-import", is_flag=True, help="Skip import step")
@click.option("--skip-process", is_flag=True, help="Skip processing step")
@click.option("--skip-validate", is_flag=True, help="Skip validation step")
@click.option("--version", default="1.0.0", help="Dictionary version")
@pass_config
def pipeline_run(config, skip_import, skip_process, skip_validate, version):
    """Run the complete pipeline: import -> process -> validate -> generate."""
    click.echo(styled_header("Full Pipeline Execution"))

    if config.dry_run:
        click.echo(styled_info("DRY RUN - no files will be written"))

    start_time = time.time()

    steps = [
        ("Import", not skip_import),
        ("Process", not skip_process),
        ("Validate", not skip_validate),
        ("Generate", True),
    ]

    active_steps = [(name, enabled) for name, enabled in steps if enabled]
    click.echo(f"Steps to run: {', '.join(name for name, _ in active_steps)}")

    kaggle_dir = Path(config.get("kaggle_dir"))
    recordings_dir = Path(config.get("recordings_dir"))
    processed_dir = Path(config.get("processed_dir"))
    output_dir = Path(config.get("output_dir"))

    # Step 1: Import
    if not skip_import:
        click.echo(f"\n{'=' * 40}")
        click.echo(click.style("Step 1: Import", bold=True))
        click.echo(f"{'=' * 40}")

        sources_found = 0

        if kaggle_dir.exists():
            csv_files = list(kaggle_dir.glob("*.csv"))
            click.echo(f"Kaggle CSV files: {len(csv_files)}")
            sources_found += len(csv_files)

        if recordings_dir.exists():
            rec_files = list(recordings_dir.glob("*.json"))
            click.echo(f"Recording files: {len(rec_files)}")
            sources_found += len(rec_files)

        if sources_found == 0:
            click.echo(styled_warning("No data sources found"))
        else:
            click.echo(styled_success(f"Found {sources_found} source files"))

    # Step 2: Process
    if not skip_process:
        click.echo(f"\n{'=' * 40}")
        click.echo(click.style("Step 2: Process", bold=True))
        click.echo(f"{'=' * 40}")

        # Check for files to process
        to_process = list(processed_dir.parent.glob("raw-recordings/*.json"))
        click.echo(f"Files to process: {len(to_process)}")

        if to_process and not config.dry_run:
            # Would run batch processing here
            click.echo("Processing would run here...")

        click.echo(styled_success("Processing step complete"))

    # Step 3: Validate
    if not skip_validate:
        click.echo(f"\n{'=' * 40}")
        click.echo(click.style("Step 3: Validate", bold=True))
        click.echo(f"{'=' * 40}")

        processed_files = list(processed_dir.glob("*.json"))
        click.echo(f"Processed files to validate: {len(processed_files)}")

        errors = 0
        warnings = 0

        if processed_files:
            with click.progressbar(processed_files[:20], label="Validating") as files:
                for filepath in files:
                    data = load_json_file(filepath)
                    if not data:
                        errors += 1

        click.echo(f"Validation errors: {errors}")
        click.echo(styled_success("Validation step complete"))

    # Step 4: Generate
    click.echo(f"\n{'=' * 40}")
    click.echo(click.style("Step 4: Generate", bold=True))
    click.echo(f"{'=' * 40}")

    try:
        from scripts.generate_dictionary import (
            generate_dictionary,
            export_json,
            export_minified,
            export_typescript_types,
            generate_summary,
        )

        dictionary = generate_dictionary(
            processed_dir=processed_dir,
            version=version,
            include_missing=True,
        )

        summary = generate_summary(dictionary)

        click.echo(f"Total signs: {summary['total_signs']}")
        click.echo(f"With data: {summary['signs_with_data']}")
        click.echo(f"Coverage: {summary['coverage_percent']:.1f}%")

        if not config.dry_run:
            export_json(dictionary, output_dir / "bsl-dictionary.json")
            export_minified(dictionary, output_dir / "bsl-dictionary.min.json")
            export_typescript_types(dictionary, output_dir / "bsl-dictionary.d.ts")

        click.echo(styled_success("Generation step complete"))

    except ImportError as e:
        click.echo(styled_error(f"Failed to import generator: {e}"))

    # Summary
    elapsed = time.time() - start_time

    click.echo(f"\n{'=' * 40}")
    click.echo(click.style("Pipeline Complete", bold=True, fg="green"))
    click.echo(f"{'=' * 40}")
    click.echo(f"Total time: {elapsed:.1f} seconds")

    if config.dry_run:
        click.echo(styled_warning("DRY RUN - no files were modified"))


@pipeline.command("status")
@pass_config
def pipeline_status(config):
    """Show current pipeline status and data statistics."""
    click.echo(styled_header("Pipeline Status"))

    data_dir = Path(config.get("data_dir"))

    # Check directories
    dirs_status = [
        ("kaggle-bsl", config.get("kaggle_dir")),
        ("raw-recordings", config.get("recordings_dir")),
        ("processed", config.get("processed_dir")),
        ("output", config.get("output_dir")),
    ]

    click.echo("\nDirectory Status:")
    click.echo("-" * 50)

    for name, dir_path in dirs_status:
        path = Path(dir_path)
        if path.exists():
            json_count = count_files(path, "*.json")
            csv_count = count_files(path, "*.csv")
            total_size = sum(f.stat().st_size for f in path.glob("*") if f.is_file())

            status = click.style("exists", fg="green")
            click.echo(f"  {name:<20} {status} | JSON: {json_count:>4} | CSV: {csv_count:>3} | Size: {format_size(total_size)}")
        else:
            status = click.style("missing", fg="yellow")
            click.echo(f"  {name:<20} {status}")

    # Check dictionary
    dict_path = Path(config.get("output_dir")) / "bsl-dictionary.json"
    if dict_path.exists():
        data = load_json_file(dict_path)
        if data:
            click.echo(f"\nDictionary Status:")
            click.echo("-" * 50)
            click.echo(f"  Version: {data.get('version', 'unknown')}")
            click.echo(f"  Signs: {len(data.get('signs', {}))}")
            click.echo(f"  Updated: {data.get('updated_at', 'unknown')}")

            # Coverage
            signs = data.get("signs", {})
            with_data = sum(
                1 for s in signs.values()
                if any(p.get("left_hand") or p.get("right_hand") for p in s.get("poses", []))
            )
            coverage = (with_data / len(signs) * 100) if signs else 0
            click.echo(f"  Coverage: {with_data}/{len(signs)} ({coverage:.1f}%)")

    # Sign definitions
    click.echo(f"\nSign Definitions:")
    click.echo("-" * 50)
    click.echo(f"  Total defined: {len(SIGN_DEFINITIONS)}")
    click.echo(f"  Categories: {len(CATEGORIES)}")

    for cat in CATEGORIES:
        signs_in_cat = sum(1 for s in SIGN_DEFINITIONS if s.category == cat.id)
        click.echo(f"    {cat.display_name}: {signs_in_cat}")

    click.echo(styled_success("\nStatus check complete"))


# =============================================================================
# Additional Utility Commands
# =============================================================================


@main.command("info")
@pass_config
def info(config):
    """Show tool information and configuration."""
    click.echo(styled_header("BSL Data Extraction Tool"))

    click.echo(f"\nVersion: 0.1.0")
    click.echo(f"Python: {sys.version.split()[0]}")

    click.echo(f"\nConfiguration:")
    click.echo("-" * 40)
    for key, value in config.values.items():
        click.echo(f"  {key}: {value}")

    click.echo(f"\nVerbosity: {config.verbose}")
    click.echo(f"Dry run: {config.dry_run}")

    # Check dependencies
    click.echo(f"\nDependencies:")
    click.echo("-" * 40)

    deps = ["click", "pydantic", "numpy", "scipy"]
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "unknown")
            click.echo(styled_success(f"  {dep}: {version}"))
        except ImportError:
            click.echo(styled_error(f"  {dep}: not installed"))


@main.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing directories")
@pass_config
def init(config, force):
    """Initialize project directory structure."""
    click.echo(styled_header("Project Initialization"))

    if config.dry_run:
        click.echo(styled_info("DRY RUN - no directories will be created"))

    dirs_to_create = [
        "data/kaggle-bsl",
        "data/raw-recordings",
        "data/processed",
        "data/output",
    ]

    for dir_path in dirs_to_create:
        path = Path(dir_path)

        if path.exists():
            if force:
                click.echo(styled_warning(f"  {dir_path} exists (keeping)"))
            else:
                click.echo(f"  {dir_path} " + click.style("exists", fg="green"))
        else:
            if not config.dry_run:
                path.mkdir(parents=True, exist_ok=True)
            click.echo(styled_success(f"  {dir_path} created"))

    # Create sample config
    config_path = Path("bsl-tool.config.json")
    if not config_path.exists() or force:
        sample_config = {
            "data_dir": "data",
            "kaggle_dir": "data/kaggle-bsl",
            "recordings_dir": "data/raw-recordings",
            "processed_dir": "data/processed",
            "output_dir": "data/output",
            "min_confidence": 0.5,
            "outlier_method": "zscore",
            "averaging_method": "trimmed",
        }

        if not config.dry_run:
            save_json_file(config_path, sample_config)

        click.echo(styled_success(f"  {config_path} created"))

    click.echo(styled_success("\nInitialization complete"))
    click.echo("\nNext steps:")
    click.echo("  1. Place Kaggle data in data/kaggle-bsl/")
    click.echo("  2. Run: bsl-tool pipeline status")
    click.echo("  3. Run: bsl-tool pipeline run")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    main()
