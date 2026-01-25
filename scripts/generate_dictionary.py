#!/usr/bin/env python3
"""
BSL Dictionary Generator

Produces the final BSL sign dictionary JSON from processed recordings.
Combines sign definitions with canonical poses and generates various exports.

Usage:
    python scripts/generate_dictionary.py generate --version 1.0.0
    python scripts/generate_dictionary.py validate /data/output/bsl-dictionary.json
    python scripts/generate_dictionary.py report /data/output/bsl-dictionary.json
    python scripts/generate_dictionary.py export-types /data/output/bsl-dictionary.json
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sign_definitions import (
    SIGN_DEFINITIONS,
    CATEGORIES,
    get_sign_by_id,
    get_signs_by_category,
)
from src.types import (
    SignDefinition,
    SignDictionaryEntry,
    SignDictionary,
    SignPose,
    ProcessingResult,
    Point3D,
)
from src.validation import validate_sign_entry, validate_dictionary


# =============================================================================
# Configuration
# =============================================================================


DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DICTIONARY_FILENAME = "bsl-dictionary.json"
MINIFIED_FILENAME = "bsl-dictionary.min.json"
TYPES_FILENAME = "bsl-dictionary.d.ts"


# =============================================================================
# Entry Generation
# =============================================================================


def generate_sign_entry(
    sign_def: SignDefinition,
    pose: SignPose,
    quality_score: float = 0.8,
    sample_count: int = 1,
    source: str = "processed"
) -> SignDictionaryEntry:
    """
    Generate a dictionary entry combining definition and pose.

    Args:
        sign_def: Sign definition with metadata
        pose: Processed canonical pose
        quality_score: Quality score (0-1)
        sample_count: Number of samples used
        source: Data source ("kaggle", "recorded", "merged")

    Returns:
        Complete SignDictionaryEntry
    """
    now = datetime.now()

    # Validate source
    valid_sources = ("kaggle", "recorded", "merged")
    if source not in valid_sources:
        source = "merged"

    return SignDictionaryEntry(
        definition=sign_def,
        poses=[pose],
        sample_count=sample_count,
        quality_score=quality_score,
        created_at=now,
        updated_at=now,
        source=source,
    )


def load_processing_result(filepath: Path) -> Optional[ProcessingResult]:
    """
    Load a ProcessingResult from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        ProcessingResult or None if loading failed
    """
    try:
        with open(filepath) as f:
            data = json.load(f)

        # Reconstruct SignPose from JSON
        pose_data = data.get("canonical_pose", {})

        left_landmarks = None
        right_landmarks = None

        if pose_data.get("left_hand_landmarks"):
            left_landmarks = [
                Point3D(x=p["x"], y=p["y"], z=p["z"])
                for p in pose_data["left_hand_landmarks"]
            ]

        if pose_data.get("right_hand_landmarks"):
            right_landmarks = [
                Point3D(x=p["x"], y=p["y"], z=p["z"])
                for p in pose_data["right_hand_landmarks"]
            ]

        pose = SignPose(
            left_hand_landmarks=left_landmarks,
            right_hand_landmarks=right_landmarks,
            tolerances=pose_data.get("tolerances", {}),
        )

        return ProcessingResult(
            canonical_pose=pose,
            quality_score=data.get("quality_score", 0.5),
            sample_count=data.get("sample_count", 1),
            warnings=data.get("warnings", []),
        )

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


# =============================================================================
# Dictionary Generation
# =============================================================================


def generate_dictionary(
    processed_dir: Path,
    version: str = "1.0.0",
    include_missing: bool = True
) -> SignDictionary:
    """
    Generate complete BSL dictionary from processed poses.

    Args:
        processed_dir: Directory containing processed pose JSON files
        version: Dictionary version string
        include_missing: If True, include signs without processed data

    Returns:
        Complete SignDictionary
    """
    now = datetime.now()
    entries: dict[str, SignDictionaryEntry] = {}

    # Load all processed results
    processed_results: dict[str, ProcessingResult] = {}

    if processed_dir.exists():
        for json_file in processed_dir.glob("*.json"):
            if json_file.name == "processing_report.json":
                continue

            sign_id = json_file.stem
            result = load_processing_result(json_file)
            if result:
                processed_results[sign_id] = result

    print(f"Loaded {len(processed_results)} processed poses")

    # Generate entries for each sign definition
    for sign_def in SIGN_DEFINITIONS:
        sign_id = sign_def.id

        if sign_id in processed_results:
            result = processed_results[sign_id]
            entry = generate_sign_entry(
                sign_def=sign_def,
                pose=result.canonical_pose,
                quality_score=result.quality_score,
                sample_count=result.sample_count,
                source="processed",
            )
            entries[sign_id] = entry
        elif include_missing:
            # Create placeholder entry without pose data
            # sample_count must be >= 1 per type definition
            placeholder_pose = SignPose(
                left_hand_landmarks=None,
                right_hand_landmarks=None,
                tolerances={},
            )
            entry = generate_sign_entry(
                sign_def=sign_def,
                pose=placeholder_pose,
                quality_score=0.0,
                sample_count=1,  # Minimum required value
                source="merged",
            )
            entries[sign_id] = entry

    # Create dictionary
    dictionary = SignDictionary(
        version=version,
        created_at=now,
        updated_at=now,
        entries=entries,
        metadata={
            "language": "BSL",
            "description": "British Sign Language dictionary with landmark data",
            "generator": "generate_dictionary.py",
            "sign_count": len(entries),
            "processed_count": len(processed_results),
            "categories": [cat._asdict() for cat in CATEGORIES],
        },
    )

    return dictionary


# =============================================================================
# Export Functions
# =============================================================================


def dictionary_to_export_format(dictionary: SignDictionary) -> dict:
    """
    Convert SignDictionary to export JSON format.

    Args:
        dictionary: SignDictionary to convert

    Returns:
        Export-ready dictionary
    """
    # Build signs dict with flattened structure
    signs = {}
    for sign_id, entry in dictionary.entries.items():
        # Flatten the entry structure
        sign_data = {
            "id": entry.definition.id,
            "name": entry.definition.name,
            "category": entry.definition.category,
            "description": entry.definition.description,
            "two_handed": entry.definition.two_handed,
            "difficulty": entry.definition.difficulty,
            "poses": [],
            "metadata": {
                "quality_score": entry.quality_score,
                "sample_count": entry.sample_count,
                "source": entry.source,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
            },
        }

        # Add poses
        for pose in entry.poses:
            pose_data = {
                "tolerances": pose.tolerances,
            }

            if pose.left_hand_landmarks:
                pose_data["left_hand"] = [
                    {"x": p.x, "y": p.y, "z": p.z}
                    for p in pose.left_hand_landmarks
                ]

            if pose.right_hand_landmarks:
                pose_data["right_hand"] = [
                    {"x": p.x, "y": p.y, "z": p.z}
                    for p in pose.right_hand_landmarks
                ]

            if pose.left_hand_angles:
                pose_data["left_angles"] = pose.left_hand_angles.model_dump()

            if pose.right_hand_angles:
                pose_data["right_angles"] = pose.right_hand_angles.model_dump()

            sign_data["poses"].append(pose_data)

        signs[sign_id] = sign_data

    # Build export format
    export = {
        "version": dictionary.version,
        "created_at": dictionary.created_at.isoformat(),
        "updated_at": dictionary.updated_at.isoformat(),
        "language": dictionary.metadata.get("language", "BSL"),
        "sign_count": dictionary.sign_count,
        "categories": dictionary.metadata.get("categories", []),
        "signs": signs,
    }

    return export


def export_json(
    dictionary: SignDictionary,
    filepath: Path,
    pretty: bool = True
) -> None:
    """
    Export dictionary to JSON file.

    Args:
        dictionary: SignDictionary to export
        filepath: Output file path
        pretty: If True, format with indentation
    """
    export_data = dictionary_to_export_format(dictionary)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        if pretty:
            json.dump(export_data, f, indent=2, default=str)
        else:
            json.dump(export_data, f, separators=(",", ":"), default=str)

    print(f"Exported dictionary to: {filepath}")


def export_minified(dictionary: SignDictionary, filepath: Path) -> None:
    """
    Export minified dictionary (no whitespace).

    Args:
        dictionary: SignDictionary to export
        filepath: Output file path
    """
    export_json(dictionary, filepath, pretty=False)


def export_typescript_types(dictionary: SignDictionary, filepath: Path) -> None:
    """
    Generate TypeScript type definitions for the dictionary.

    Creates .d.ts file with:
    - SignId literal type
    - Category literal type
    - Interface definitions

    Args:
        dictionary: SignDictionary to generate types from
        filepath: Output .d.ts file path
    """
    sign_ids = sorted(dictionary.entries.keys())
    categories = sorted(set(
        entry.definition.category
        for entry in dictionary.entries.values()
    ))

    ts_content = f'''/**
 * BSL Dictionary TypeScript Type Definitions
 * Generated: {datetime.now().isoformat()}
 * Version: {dictionary.version}
 */

/**
 * All available sign IDs in the dictionary
 */
export type SignId =
{chr(10).join(f'  | "{sid}"' for sid in sign_ids)};

/**
 * Sign categories
 */
export type SignCategory =
{chr(10).join(f'  | "{cat}"' for cat in categories)};

/**
 * 3D point for hand landmarks
 */
export interface Point3D {{
  x: number;
  y: number;
  z: number;
}}

/**
 * Hand pose data
 */
export interface HandPose {{
  landmarks: Point3D[];
  tolerances?: Record<string, number>;
}}

/**
 * Complete sign pose
 */
export interface SignPose {{
  left_hand?: HandPose;
  right_hand?: HandPose;
  left_angles?: JointAngles;
  right_angles?: JointAngles;
  tolerances: Record<string, number>;
}}

/**
 * Joint angles for a finger
 */
export interface FingerAngles {{
  mcp: number;
  pip: number;
  dip: number;
}}

/**
 * All joint angles for a hand
 */
export interface JointAngles {{
  thumb: FingerAngles;
  index: FingerAngles;
  middle: FingerAngles;
  ring: FingerAngles;
  pinky: FingerAngles;
  wrist_rotation: number;
  wrist_flexion: number;
}}

/**
 * Sign entry metadata
 */
export interface SignMetadata {{
  quality_score: number;
  sample_count: number;
  source: "kaggle" | "recorded" | "merged";
  created_at: string;
  updated_at: string;
}}

/**
 * Complete sign entry
 */
export interface SignEntry {{
  id: SignId;
  name: string;
  category: SignCategory;
  description: string;
  two_handed: boolean;
  difficulty: 1 | 2 | 3 | 4 | 5;
  poses: SignPose[];
  metadata: SignMetadata;
}}

/**
 * Category definition
 */
export interface Category {{
  id: SignCategory;
  display_name: string;
  description: string;
  icon_emoji: string;
}}

/**
 * Complete BSL dictionary
 */
export interface BSLDictionary {{
  version: string;
  created_at: string;
  updated_at: string;
  language: "BSL";
  sign_count: number;
  categories: Category[];
  signs: Record<SignId, SignEntry>;
}}

/**
 * Get sign by ID (type-safe)
 */
export declare function getSign(dictionary: BSLDictionary, id: SignId): SignEntry;

/**
 * Get signs by category (type-safe)
 */
export declare function getSignsByCategory(
  dictionary: BSLDictionary,
  category: SignCategory
): SignEntry[];

/**
 * Total number of signs in the dictionary
 */
export const SIGN_COUNT: {len(sign_ids)};

/**
 * All sign IDs as array
 */
export const ALL_SIGN_IDS: SignId[];

/**
 * All categories as array
 */
export const ALL_CATEGORIES: SignCategory[];
'''

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        f.write(ts_content)

    print(f"Exported TypeScript types to: {filepath}")


# =============================================================================
# Import/Merge Functions
# =============================================================================


def import_dictionary(filepath: Path) -> Optional[SignDictionary]:
    """
    Import dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        SignDictionary or None if import failed
    """
    try:
        with open(filepath) as f:
            data = json.load(f)

        # Convert from export format back to SignDictionary
        entries = {}

        for sign_id, sign_data in data.get("signs", {}).items():
            # Get sign definition
            sign_def = get_sign_by_id(sign_id)
            if not sign_def:
                # Create definition from data
                sign_def = SignDefinition(
                    id=sign_data["id"],
                    name=sign_data["name"],
                    category=sign_data["category"],
                    description=sign_data.get("description", ""),
                    two_handed=sign_data.get("two_handed", False),
                    difficulty=sign_data.get("difficulty", 3),
                )

            # Parse poses
            poses = []
            for pose_data in sign_data.get("poses", []):
                left_landmarks = None
                right_landmarks = None

                if "left_hand" in pose_data:
                    left_landmarks = [
                        Point3D(x=p["x"], y=p["y"], z=p["z"])
                        for p in pose_data["left_hand"]
                    ]

                if "right_hand" in pose_data:
                    right_landmarks = [
                        Point3D(x=p["x"], y=p["y"], z=p["z"])
                        for p in pose_data["right_hand"]
                    ]

                pose = SignPose(
                    left_hand_landmarks=left_landmarks,
                    right_hand_landmarks=right_landmarks,
                    tolerances=pose_data.get("tolerances", {}),
                )
                poses.append(pose)

            # If no poses, create empty one
            if not poses:
                poses = [SignPose(tolerances={})]

            # Parse metadata
            meta = sign_data.get("metadata", {})

            entry = SignDictionaryEntry(
                definition=sign_def,
                poses=poses,
                sample_count=meta.get("sample_count", 0),
                quality_score=meta.get("quality_score", 0.0),
                created_at=datetime.fromisoformat(
                    meta.get("created_at", datetime.now().isoformat())
                ),
                updated_at=datetime.fromisoformat(
                    meta.get("updated_at", datetime.now().isoformat())
                ),
                source=meta.get("source", "merged"),
            )
            entries[sign_id] = entry

        return SignDictionary(
            version=data.get("version", "0.0.0"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            entries=entries,
            metadata={
                "language": data.get("language", "BSL"),
                "categories": data.get("categories", []),
            },
        )

    except Exception as e:
        print(f"Error importing {filepath}: {e}")
        return None


def merge_dictionaries(
    base: SignDictionary,
    additions: SignDictionary,
    prefer_higher_quality: bool = True
) -> SignDictionary:
    """
    Merge two dictionaries.

    Args:
        base: Base dictionary
        additions: Dictionary with additions/updates
        prefer_higher_quality: If True, keep higher quality entry on conflict

    Returns:
        Merged SignDictionary
    """
    merged_entries = dict(base.entries)

    for sign_id, new_entry in additions.entries.items():
        if sign_id in merged_entries:
            existing = merged_entries[sign_id]

            # Decide which to keep
            if prefer_higher_quality:
                if new_entry.quality_score > existing.quality_score:
                    merged_entries[sign_id] = new_entry
                # else keep existing
            else:
                # Prefer additions
                merged_entries[sign_id] = new_entry
        else:
            merged_entries[sign_id] = new_entry

    # Update version
    base_parts = base.version.split(".")
    new_minor = int(base_parts[1]) + 1 if len(base_parts) > 1 else 1
    new_version = f"{base_parts[0]}.{new_minor}.0"

    return SignDictionary(
        version=new_version,
        created_at=base.created_at,
        updated_at=datetime.now(),
        entries=merged_entries,
        metadata={
            **base.metadata,
            "merged_from": [base.version, additions.version],
        },
    )


def diff_dictionaries(
    old: SignDictionary,
    new: SignDictionary
) -> dict:
    """
    Calculate differences between two dictionaries.

    Args:
        old: Old/previous dictionary
        new: New/current dictionary

    Returns:
        Dictionary describing changes
    """
    old_ids = set(old.entries.keys())
    new_ids = set(new.entries.keys())

    added = new_ids - old_ids
    removed = old_ids - new_ids
    common = old_ids & new_ids

    modified = []
    quality_improved = []
    quality_degraded = []

    for sign_id in common:
        old_entry = old.entries[sign_id]
        new_entry = new.entries[sign_id]

        # Check for quality changes
        if new_entry.quality_score > old_entry.quality_score + 0.05:
            quality_improved.append(sign_id)
        elif new_entry.quality_score < old_entry.quality_score - 0.05:
            quality_degraded.append(sign_id)

        # Check for content changes
        if new_entry.sample_count != old_entry.sample_count:
            modified.append(sign_id)

    return {
        "old_version": old.version,
        "new_version": new.version,
        "added": sorted(added),
        "removed": sorted(removed),
        "modified": sorted(modified),
        "quality_improved": sorted(quality_improved),
        "quality_degraded": sorted(quality_degraded),
        "summary": {
            "total_added": len(added),
            "total_removed": len(removed),
            "total_modified": len(modified),
            "quality_improvements": len(quality_improved),
            "quality_degradations": len(quality_degraded),
        },
    }


# =============================================================================
# Reports
# =============================================================================


def generate_coverage_report(dictionary: SignDictionary) -> str:
    """
    Generate coverage report showing which signs have data.

    Args:
        dictionary: SignDictionary to report on

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BSL Dictionary Coverage Report")
    lines.append(f"Version: {dictionary.version}")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 60)

    # Overall stats
    total = len(dictionary.entries)
    with_poses = sum(
        1 for e in dictionary.entries.values()
        if any(p.left_hand_landmarks or p.right_hand_landmarks for p in e.poses)
    )
    coverage = (with_poses / total * 100) if total > 0 else 0

    lines.append(f"\nOverall Coverage: {with_poses}/{total} ({coverage:.1f}%)")
    lines.append("")

    # By category
    lines.append("Coverage by Category:")
    lines.append("-" * 40)

    category_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "with_data": 0})

    for entry in dictionary.entries.values():
        cat = entry.definition.category
        category_stats[cat]["total"] += 1
        if any(p.left_hand_landmarks or p.right_hand_landmarks for p in entry.poses):
            category_stats[cat]["with_data"] += 1

    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        cat_coverage = (stats["with_data"] / stats["total"] * 100) if stats["total"] > 0 else 0
        lines.append(f"  {cat:20s}: {stats['with_data']:3d}/{stats['total']:3d} ({cat_coverage:5.1f}%)")

    # Missing signs
    missing = [
        e.definition.name
        for e in dictionary.entries.values()
        if not any(p.left_hand_landmarks or p.right_hand_landmarks for p in e.poses)
    ]

    if missing:
        lines.append("")
        lines.append(f"Missing Data ({len(missing)} signs):")
        lines.append("-" * 40)
        for i in range(0, len(missing), 10):
            lines.append("  " + ", ".join(missing[i:i+10]))

    return "\n".join(lines)


def generate_quality_report(dictionary: SignDictionary) -> str:
    """
    Generate quality report for the dictionary.

    Args:
        dictionary: SignDictionary to report on

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BSL Dictionary Quality Report")
    lines.append(f"Version: {dictionary.version}")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 60)

    # Quality distribution
    scores = [e.quality_score for e in dictionary.entries.values()]
    avg_score = sum(scores) / len(scores) if scores else 0

    high_quality = [s for s in scores if s >= 0.8]
    medium_quality = [s for s in scores if 0.5 <= s < 0.8]
    low_quality = [s for s in scores if 0.0 < s < 0.5]
    no_data = [s for s in scores if s == 0.0]

    lines.append(f"\nAverage Quality Score: {avg_score:.2f}")
    lines.append("")
    lines.append("Quality Distribution:")
    lines.append("-" * 40)
    lines.append(f"  High (≥0.8):    {len(high_quality):3d} ({len(high_quality)/len(scores)*100:.1f}%)")
    lines.append(f"  Medium (0.5-0.8): {len(medium_quality):3d} ({len(medium_quality)/len(scores)*100:.1f}%)")
    lines.append(f"  Low (<0.5):      {len(low_quality):3d} ({len(low_quality)/len(scores)*100:.1f}%)")
    lines.append(f"  No Data (0):     {len(no_data):3d} ({len(no_data)/len(scores)*100:.1f}%)")

    # Sample count stats
    sample_counts = [e.sample_count for e in dictionary.entries.values()]
    total_samples = sum(sample_counts)
    avg_samples = total_samples / len(sample_counts) if sample_counts else 0

    lines.append("")
    lines.append("Sample Statistics:")
    lines.append("-" * 40)
    lines.append(f"  Total Samples: {total_samples}")
    lines.append(f"  Average per Sign: {avg_samples:.1f}")
    lines.append(f"  Max Samples: {max(sample_counts) if sample_counts else 0}")
    lines.append(f"  Min Samples: {min(sample_counts) if sample_counts else 0}")

    # Lowest quality signs
    sorted_entries = sorted(
        dictionary.entries.values(),
        key=lambda e: e.quality_score
    )

    low_entries = [e for e in sorted_entries if 0 < e.quality_score < 0.5][:10]

    if low_entries:
        lines.append("")
        lines.append("Lowest Quality Signs (needs improvement):")
        lines.append("-" * 40)
        for entry in low_entries:
            lines.append(f"  {entry.definition.name:20s}: {entry.quality_score:.2f} ({entry.sample_count} samples)")

    return "\n".join(lines)


def generate_summary(dictionary: SignDictionary) -> dict:
    """
    Generate summary statistics for the dictionary.

    Args:
        dictionary: SignDictionary to summarize

    Returns:
        Summary dictionary
    """
    entries = list(dictionary.entries.values())

    # Count by category
    by_category = defaultdict(int)
    for entry in entries:
        by_category[entry.definition.category] += 1

    # Quality stats
    scores = [e.quality_score for e in entries]
    sample_counts = [e.sample_count for e in entries]

    # Coverage
    with_poses = sum(
        1 for e in entries
        if any(p.left_hand_landmarks or p.right_hand_landmarks for p in e.poses)
    )

    # By source
    by_source = defaultdict(int)
    for entry in entries:
        by_source[entry.source] += 1

    return {
        "version": dictionary.version,
        "created_at": dictionary.created_at.isoformat(),
        "updated_at": dictionary.updated_at.isoformat(),
        "total_signs": len(entries),
        "signs_with_data": with_poses,
        "coverage_percent": (with_poses / len(entries) * 100) if entries else 0,
        "quality": {
            "average": sum(scores) / len(scores) if scores else 0,
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "high_quality_count": sum(1 for s in scores if s >= 0.8),
        },
        "samples": {
            "total": sum(sample_counts),
            "average": sum(sample_counts) / len(sample_counts) if sample_counts else 0,
        },
        "by_category": dict(by_category),
        "by_source": dict(by_source),
    }


# =============================================================================
# Validation
# =============================================================================


def validate_complete_dictionary(dictionary: SignDictionary) -> dict:
    """
    Validate the complete dictionary.

    Args:
        dictionary: SignDictionary to validate

    Returns:
        Validation result dictionary
    """
    errors = []
    warnings = []
    entry_results = {}

    for sign_id, entry in dictionary.entries.items():
        result = validate_sign_entry(entry)
        entry_results[sign_id] = {
            "is_valid": result.is_valid,
            "score": result.score,
            "errors": result.errors,
            "warnings": result.warnings,
        }

        if not result.is_valid:
            errors.append(f"Entry {sign_id}: {', '.join(result.errors)}")

        warnings.extend([f"Entry {sign_id}: {w}" for w in result.warnings])

    # Check for missing signs from definitions
    defined_ids = {s.id for s in SIGN_DEFINITIONS}
    dict_ids = set(dictionary.entries.keys())

    missing = defined_ids - dict_ids
    extra = dict_ids - defined_ids

    if missing:
        warnings.append(f"Missing signs: {', '.join(sorted(missing))}")
    if extra:
        warnings.append(f"Extra signs not in definitions: {', '.join(sorted(extra))}")

    return {
        "is_valid": len(errors) == 0,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings[:20],  # Limit warnings
        "entry_results": entry_results,
    }


# =============================================================================
# CLI
# =============================================================================


@click.group()
def cli():
    """BSL Dictionary Generator and Management Tool."""
    pass


@cli.command()
@click.option("--version", "-v", default="1.0.0", help="Dictionary version")
@click.option("--processed-dir", "-p", type=click.Path(exists=True),
              default=str(DEFAULT_PROCESSED_DIR), help="Processed poses directory")
@click.option("--output-dir", "-o", type=click.Path(),
              default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
@click.option("--include-missing/--no-include-missing", default=True,
              help="Include signs without processed data")
def generate(version, processed_dir, output_dir, include_missing):
    """Generate BSL dictionary from processed poses."""
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)

    click.echo(f"Generating dictionary v{version}...")
    click.echo(f"  Processed dir: {processed_path}")
    click.echo(f"  Output dir: {output_path}")

    dictionary = generate_dictionary(
        processed_dir=processed_path,
        version=version,
        include_missing=include_missing,
    )

    # Export all formats
    json_path = output_path / DICTIONARY_FILENAME
    min_path = output_path / MINIFIED_FILENAME
    types_path = output_path / TYPES_FILENAME

    export_json(dictionary, json_path, pretty=True)
    export_minified(dictionary, min_path)
    export_typescript_types(dictionary, types_path)

    # Print summary
    summary = generate_summary(dictionary)
    click.echo("")
    click.echo("Generation complete!")
    click.echo(f"  Total signs: {summary['total_signs']}")
    click.echo(f"  With data: {summary['signs_with_data']}")
    click.echo(f"  Coverage: {summary['coverage_percent']:.1f}%")
    click.echo(f"  Avg quality: {summary['quality']['average']:.2f}")


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
def validate(filepath):
    """Validate a dictionary file."""
    path = Path(filepath)

    click.echo(f"Validating: {path}")

    dictionary = import_dictionary(path)
    if not dictionary:
        click.echo("Failed to import dictionary")
        return

    result = validate_complete_dictionary(dictionary)

    if result["is_valid"]:
        click.echo(f"\n✓ Dictionary is valid")
    else:
        click.echo(f"\n✗ Dictionary has errors")

    click.echo(f"  Errors: {result['error_count']}")
    click.echo(f"  Warnings: {result['warning_count']}")

    if result["errors"]:
        click.echo("\nErrors:")
        for err in result["errors"][:10]:
            click.echo(f"  - {err}")

    if result["warnings"]:
        click.echo("\nWarnings (first 10):")
        for warn in result["warnings"][:10]:
            click.echo(f"  - {warn}")


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output report file")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text",
              help="Report format")
def report(filepath, output, format):
    """Generate reports for a dictionary."""
    path = Path(filepath)

    dictionary = import_dictionary(path)
    if not dictionary:
        click.echo("Failed to import dictionary")
        return

    if format == "text":
        coverage = generate_coverage_report(dictionary)
        quality = generate_quality_report(dictionary)
        report_text = f"{coverage}\n\n{quality}"

        if output:
            with open(output, "w") as f:
                f.write(report_text)
            click.echo(f"Report saved to: {output}")
        else:
            click.echo(report_text)
    else:
        summary = generate_summary(dictionary)

        if output:
            with open(output, "w") as f:
                json.dump(summary, f, indent=2)
            click.echo(f"Report saved to: {output}")
        else:
            click.echo(json.dumps(summary, indent=2))


@cli.command("export-types")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output .d.ts file")
def export_types(filepath, output):
    """Export TypeScript type definitions."""
    path = Path(filepath)

    dictionary = import_dictionary(path)
    if not dictionary:
        click.echo("Failed to import dictionary")
        return

    output_path = Path(output) if output else path.with_suffix(".d.ts")
    export_typescript_types(dictionary, output_path)


@cli.command()
@click.argument("base", type=click.Path(exists=True))
@click.argument("additions", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file")
def merge(base, additions, output):
    """Merge two dictionaries."""
    base_dict = import_dictionary(Path(base))
    add_dict = import_dictionary(Path(additions))

    if not base_dict or not add_dict:
        click.echo("Failed to import dictionaries")
        return

    merged = merge_dictionaries(base_dict, add_dict)
    export_json(merged, Path(output))

    click.echo(f"Merged dictionary saved to: {output}")
    click.echo(f"  New version: {merged.version}")
    click.echo(f"  Total signs: {merged.sign_count}")


@cli.command()
@click.argument("old", type=click.Path(exists=True))
@click.argument("new", type=click.Path(exists=True))
def diff(old, new):
    """Show differences between two dictionaries."""
    old_dict = import_dictionary(Path(old))
    new_dict = import_dictionary(Path(new))

    if not old_dict or not new_dict:
        click.echo("Failed to import dictionaries")
        return

    changes = diff_dictionaries(old_dict, new_dict)

    click.echo(f"Comparing {changes['old_version']} -> {changes['new_version']}")
    click.echo("")

    summary = changes["summary"]
    click.echo(f"Added: {summary['total_added']}")
    click.echo(f"Removed: {summary['total_removed']}")
    click.echo(f"Modified: {summary['total_modified']}")
    click.echo(f"Quality improved: {summary['quality_improvements']}")
    click.echo(f"Quality degraded: {summary['quality_degradations']}")

    if changes["added"]:
        click.echo(f"\nAdded signs: {', '.join(changes['added'][:10])}")
    if changes["removed"]:
        click.echo(f"\nRemoved signs: {', '.join(changes['removed'][:10])}")


if __name__ == "__main__":
    cli()
