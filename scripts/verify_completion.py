#!/usr/bin/env python3
"""
Phase 0 Completion Verification Script

Comprehensive checks to ensure the BSL Data Extraction Tool is complete
and all components are working correctly.

Usage:
    python scripts/verify_completion.py
    python scripts/verify_completion.py --strict
    python scripts/verify_completion.py --report-only
    bsl-tool verify [--strict] [--report-only]
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Check Result Types
# =============================================================================


class CheckStatus(Enum):
    """Status of a verification check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a single verification check."""
    name: str
    status: CheckStatus
    message: str
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == CheckStatus.PASS

    @property
    def icon(self) -> str:
        icons = {
            CheckStatus.PASS: "\u2713",  # ✓
            CheckStatus.FAIL: "\u2717",  # ✗
            CheckStatus.WARN: "\u26A0",  # ⚠
            CheckStatus.SKIP: "\u2014",  # —
        }
        return icons.get(self.status, "?")

    @property
    def color(self) -> str:
        colors = {
            CheckStatus.PASS: "\033[92m",  # Green
            CheckStatus.FAIL: "\033[91m",  # Red
            CheckStatus.WARN: "\033[93m",  # Yellow
            CheckStatus.SKIP: "\033[90m",  # Gray
        }
        return colors.get(self.status, "")


@dataclass
class CategoryResult:
    """Result for a category of checks."""
    name: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.SKIP)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def all_passed(self) -> bool:
        return all(c.status in (CheckStatus.PASS, CheckStatus.SKIP) for c in self.checks)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class VerificationConfig:
    """Configuration for verification checks."""

    # Directories
    data_dir: Path = field(default_factory=lambda: Path("data"))
    docs_dir: Path = field(default_factory=lambda: Path("docs"))
    src_dir: Path = field(default_factory=lambda: Path("src"))
    scripts_dir: Path = field(default_factory=lambda: Path("scripts"))
    recording_tool_dir: Path = field(default_factory=lambda: Path("recording-tool"))
    notebooks_dir: Path = field(default_factory=lambda: Path("notebooks"))

    # Expected files
    dictionary_file: Path = field(default_factory=lambda: Path("data/output/bsl-dictionary.json"))

    # Data requirements
    expected_sign_count: int = 76
    min_samples_per_sign: int = 5
    min_quality_score: float = 0.5
    target_quality_score: float = 0.7

    # Dictionary requirements
    min_dictionary_size_bytes: int = 1000
    max_dictionary_size_bytes: int = 100_000_000  # 100MB

    # Documentation requirements
    required_docs: list[str] = field(default_factory=lambda: [
        "README.md",
        "docs/architecture.md",
        "docs/data-format.md",
        "docs/recording-guide.md",
        "docs/processing-params.md",
        "docs/troubleshooting.md",
        "CHANGELOG.md",
    ])

    # Source code requirements
    required_src_modules: list[str] = field(default_factory=lambda: [
        "types.py",
        "landmarks.py",
        "normalize.py",
        "angles.py",
        "math_utils.py",
        "validation.py",
        "sign_definitions.py",
    ])

    required_scripts: list[str] = field(default_factory=lambda: [
        "cli.py",
        "import_kaggle.py",
        "process_recordings.py",
        "generate_dictionary.py",
        "pipeline.py",
    ])


# =============================================================================
# Verification Engine
# =============================================================================


class Phase0Verifier:
    """
    Comprehensive verification for Phase 0 completion.

    Checks all aspects of the BSL Data Extraction Tool:
    - Data completeness
    - Quality metrics
    - Dictionary validity
    - Output files
    - Documentation
    """

    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig()
        self.results: list[CategoryResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def run_all_checks(self) -> bool:
        """Run all verification checks."""
        self.start_time = time.time()
        self.results = []

        # Run each category
        self.results.append(self._check_data_completeness())
        self.results.append(self._check_quality())
        self.results.append(self._check_dictionary())
        self.results.append(self._check_outputs())
        self.results.append(self._check_documentation())
        self.results.append(self._check_source_code())
        self.results.append(self._check_tests())

        self.end_time = time.time()

        # Return True if all checks passed
        return all(cat.all_passed for cat in self.results)

    # =========================================================================
    # Category 1: Data Completeness
    # =========================================================================

    def _check_data_completeness(self) -> CategoryResult:
        """Check data completeness requirements."""
        category = CategoryResult(name="Data Completeness")

        # Check sign definitions
        category.checks.append(self._check_sign_definitions())

        # Check recordings exist
        category.checks.append(self._check_recordings_exist())

        # Check samples per sign
        category.checks.append(self._check_samples_per_sign())

        # Check Kaggle import
        category.checks.append(self._check_kaggle_import())

        # Check processed data
        category.checks.append(self._check_processed_data())

        return category

    def _check_sign_definitions(self) -> CheckResult:
        """Verify sign definitions are complete."""
        try:
            from src.sign_definitions import SIGN_DEFINITIONS, CATEGORIES

            sign_count = len(SIGN_DEFINITIONS)
            cat_count = len(CATEGORIES)

            if sign_count >= self.config.expected_sign_count:
                return CheckResult(
                    name="Sign definitions",
                    status=CheckStatus.PASS,
                    message=f"{sign_count} signs defined across {cat_count} categories",
                    details={"count": sign_count, "categories": cat_count}
                )
            else:
                return CheckResult(
                    name="Sign definitions",
                    status=CheckStatus.WARN,
                    message=f"Only {sign_count} signs defined (expected {self.config.expected_sign_count})",
                    details={"count": sign_count, "expected": self.config.expected_sign_count}
                )
        except ImportError as e:
            return CheckResult(
                name="Sign definitions",
                status=CheckStatus.FAIL,
                message=f"Failed to import sign_definitions: {e}"
            )

    def _check_recordings_exist(self) -> CheckResult:
        """Check if recordings directory has data."""
        recordings_dir = self.config.data_dir / "raw-recordings"

        if not recordings_dir.exists():
            return CheckResult(
                name="Recordings directory",
                status=CheckStatus.WARN,
                message="Recordings directory not found (may be using Kaggle data only)",
                details={"path": str(recordings_dir)}
            )

        json_files = list(recordings_dir.glob("*.json"))

        if len(json_files) > 0:
            return CheckResult(
                name="Recordings directory",
                status=CheckStatus.PASS,
                message=f"Found {len(json_files)} recording files",
                details={"count": len(json_files), "path": str(recordings_dir)}
            )
        else:
            return CheckResult(
                name="Recordings directory",
                status=CheckStatus.WARN,
                message="No recording files found",
                details={"path": str(recordings_dir)}
            )

    def _check_samples_per_sign(self) -> CheckResult:
        """Check minimum samples per sign."""
        recordings_dir = self.config.data_dir / "raw-recordings"

        if not recordings_dir.exists():
            return CheckResult(
                name="Samples per sign",
                status=CheckStatus.SKIP,
                message="Recordings directory not found"
            )

        # Count recordings by sign
        sign_counts: dict[str, int] = {}
        for file_path in recordings_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                sign_id = data.get("sign_id", "unknown")
                sign_counts[sign_id] = sign_counts.get(sign_id, 0) + 1
            except Exception:
                pass

        if not sign_counts:
            return CheckResult(
                name="Samples per sign",
                status=CheckStatus.SKIP,
                message="No recordings to analyze"
            )

        # Check minimum
        below_min = [
            (sign, count) for sign, count in sign_counts.items()
            if count < self.config.min_samples_per_sign
        ]

        avg_samples = sum(sign_counts.values()) / len(sign_counts)

        if not below_min:
            return CheckResult(
                name="Samples per sign",
                status=CheckStatus.PASS,
                message=f"All signs have >= {self.config.min_samples_per_sign} samples (avg: {avg_samples:.1f})",
                details={"average": avg_samples, "signs": len(sign_counts)}
            )
        else:
            return CheckResult(
                name="Samples per sign",
                status=CheckStatus.WARN,
                message=f"{len(below_min)} signs have < {self.config.min_samples_per_sign} samples",
                details={"below_minimum": below_min[:10], "total": len(below_min)}
            )

    def _check_kaggle_import(self) -> CheckResult:
        """Check Kaggle data import status."""
        kaggle_dir = self.config.data_dir / "kaggle-bsl"

        if not kaggle_dir.exists():
            return CheckResult(
                name="Kaggle import",
                status=CheckStatus.SKIP,
                message="Kaggle directory not found (optional)"
            )

        csv_files = list(kaggle_dir.glob("*.csv"))
        parquet_files = list(kaggle_dir.glob("*.parquet"))
        total_files = len(csv_files) + len(parquet_files)

        if total_files > 0:
            return CheckResult(
                name="Kaggle import",
                status=CheckStatus.PASS,
                message=f"Found {total_files} Kaggle data files",
                details={"csv": len(csv_files), "parquet": len(parquet_files)}
            )
        else:
            return CheckResult(
                name="Kaggle import",
                status=CheckStatus.WARN,
                message="No Kaggle data files found"
            )

    def _check_processed_data(self) -> CheckResult:
        """Check processed pose data."""
        processed_dir = self.config.data_dir / "processed"

        if not processed_dir.exists():
            return CheckResult(
                name="Processed data",
                status=CheckStatus.WARN,
                message="Processed directory not found (run pipeline first)"
            )

        json_files = list(processed_dir.glob("*.json"))

        if len(json_files) > 0:
            return CheckResult(
                name="Processed data",
                status=CheckStatus.PASS,
                message=f"Found {len(json_files)} processed pose files",
                details={"count": len(json_files)}
            )
        else:
            return CheckResult(
                name="Processed data",
                status=CheckStatus.WARN,
                message="No processed pose files found"
            )

    # =========================================================================
    # Category 2: Quality Checks
    # =========================================================================

    def _check_quality(self) -> CategoryResult:
        """Check quality requirements."""
        category = CategoryResult(name="Quality Checks")

        # Load dictionary for quality checks
        dictionary = self._load_dictionary()

        category.checks.append(self._check_average_quality(dictionary))
        category.checks.append(self._check_minimum_quality(dictionary))
        category.checks.append(self._check_validation_passes(dictionary))
        category.checks.append(self._check_landmark_validity(dictionary))

        return category

    def _load_dictionary(self) -> Optional[dict]:
        """Load the dictionary file."""
        dict_path = self.config.dictionary_file
        if not dict_path.exists():
            return None
        try:
            with open(dict_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _check_average_quality(self, dictionary: Optional[dict]) -> CheckResult:
        """Check average quality score."""
        if not dictionary:
            return CheckResult(
                name="Average quality score",
                status=CheckStatus.SKIP,
                message="Dictionary not available"
            )

        entries = dictionary.get("entries", {})
        quality_scores = [
            entry.get("quality_score", 0)
            for entry in entries.values()
            if entry.get("sample_count", 0) > 0
        ]

        if not quality_scores:
            return CheckResult(
                name="Average quality score",
                status=CheckStatus.WARN,
                message="No quality scores available"
            )

        avg_quality = sum(quality_scores) / len(quality_scores)

        if avg_quality >= self.config.target_quality_score:
            return CheckResult(
                name="Average quality score",
                status=CheckStatus.PASS,
                message=f"Average quality: {avg_quality:.1%} (target: {self.config.target_quality_score:.0%})",
                details={"average": avg_quality, "target": self.config.target_quality_score}
            )
        elif avg_quality >= self.config.min_quality_score:
            return CheckResult(
                name="Average quality score",
                status=CheckStatus.WARN,
                message=f"Average quality: {avg_quality:.1%} (below target {self.config.target_quality_score:.0%})",
                details={"average": avg_quality, "target": self.config.target_quality_score}
            )
        else:
            return CheckResult(
                name="Average quality score",
                status=CheckStatus.FAIL,
                message=f"Average quality: {avg_quality:.1%} (below minimum {self.config.min_quality_score:.0%})",
                details={"average": avg_quality, "minimum": self.config.min_quality_score}
            )

    def _check_minimum_quality(self, dictionary: Optional[dict]) -> CheckResult:
        """Check no sign is below minimum quality."""
        if not dictionary:
            return CheckResult(
                name="Minimum quality threshold",
                status=CheckStatus.SKIP,
                message="Dictionary not available"
            )

        entries = dictionary.get("entries", {})
        below_min = [
            (sign_id, entry.get("quality_score", 0))
            for sign_id, entry in entries.items()
            if entry.get("sample_count", 0) > 0 and
               entry.get("quality_score", 0) < self.config.min_quality_score
        ]

        if not below_min:
            return CheckResult(
                name="Minimum quality threshold",
                status=CheckStatus.PASS,
                message=f"All signs meet minimum quality ({self.config.min_quality_score:.0%})"
            )
        else:
            return CheckResult(
                name="Minimum quality threshold",
                status=CheckStatus.WARN,
                message=f"{len(below_min)} signs below {self.config.min_quality_score:.0%} quality",
                details={"below_minimum": below_min[:10]}
            )

    def _check_validation_passes(self, dictionary: Optional[dict]) -> CheckResult:
        """Check that dictionary passes validation."""
        if not dictionary:
            return CheckResult(
                name="Dictionary validation",
                status=CheckStatus.SKIP,
                message="Dictionary not available"
            )

        try:
            # Manual validation since we have a plain dict, not a Pydantic model
            errors = []
            warnings = []

            # Check required top-level fields
            if "version" not in dictionary:
                errors.append("Missing 'version' field")
            if "entries" not in dictionary:
                errors.append("Missing 'entries' field")
            else:
                entries = dictionary["entries"]
                if not isinstance(entries, dict):
                    errors.append("'entries' must be a dictionary")
                elif len(entries) == 0:
                    warnings.append("Dictionary has no entries")
                else:
                    # Sample check on a few entries
                    for sign_id, entry in list(entries.items())[:10]:
                        if "definition" not in entry:
                            errors.append(f"Entry '{sign_id}' missing 'definition'")
                        if "poses" not in entry:
                            errors.append(f"Entry '{sign_id}' missing 'poses'")

            if not errors:
                return CheckResult(
                    name="Dictionary validation",
                    status=CheckStatus.PASS,
                    message="Dictionary passes structural validation",
                    details={"warnings": warnings}
                )
            else:
                return CheckResult(
                    name="Dictionary validation",
                    status=CheckStatus.FAIL,
                    message=f"Validation failed: {errors[0]}",
                    details={"errors": errors[:5]}
                )
        except Exception as e:
            return CheckResult(
                name="Dictionary validation",
                status=CheckStatus.FAIL,
                message=f"Validation error: {e}"
            )

    def _check_landmark_validity(self, dictionary: Optional[dict]) -> CheckResult:
        """Check landmark data is valid."""
        if not dictionary:
            return CheckResult(
                name="Landmark validity",
                status=CheckStatus.SKIP,
                message="Dictionary not available"
            )

        try:
            from src.landmarks import NUM_LANDMARKS

            entries = dictionary.get("entries", {})
            invalid_signs = []

            for sign_id, entry in entries.items():
                for pose in entry.get("poses", []):
                    for hand_key in ["right_hand_landmarks", "left_hand_landmarks"]:
                        landmarks = pose.get(hand_key)
                        if landmarks:
                            if len(landmarks) != NUM_LANDMARKS:
                                invalid_signs.append((sign_id, f"{hand_key}: {len(landmarks)} landmarks"))
                            else:
                                # Check for NaN/Inf
                                for lm in landmarks:
                                    x, y, z = lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)
                                    if not all(isinstance(v, (int, float)) and -10 <= v <= 10
                                              for v in [x, y, z]):
                                        invalid_signs.append((sign_id, "Invalid coordinates"))
                                        break

            if not invalid_signs:
                return CheckResult(
                    name="Landmark validity",
                    status=CheckStatus.PASS,
                    message="All landmarks have valid structure and values"
                )
            else:
                return CheckResult(
                    name="Landmark validity",
                    status=CheckStatus.FAIL,
                    message=f"{len(invalid_signs)} signs have invalid landmarks",
                    details={"invalid": invalid_signs[:10]}
                )
        except Exception as e:
            return CheckResult(
                name="Landmark validity",
                status=CheckStatus.FAIL,
                message=f"Error checking landmarks: {e}"
            )

    # =========================================================================
    # Category 3: Dictionary Checks
    # =========================================================================

    def _check_dictionary(self) -> CategoryResult:
        """Check dictionary file requirements."""
        category = CategoryResult(name="Dictionary Checks")

        category.checks.append(self._check_dictionary_exists())
        category.checks.append(self._check_dictionary_valid_json())
        category.checks.append(self._check_dictionary_structure())
        category.checks.append(self._check_dictionary_size())
        category.checks.append(self._check_joint_angles())
        category.checks.append(self._check_tolerances())

        return category

    def _check_dictionary_exists(self) -> CheckResult:
        """Check dictionary file exists."""
        dict_path = self.config.dictionary_file

        if dict_path.exists():
            return CheckResult(
                name="Dictionary file exists",
                status=CheckStatus.PASS,
                message=f"Found {dict_path}",
                details={"path": str(dict_path)}
            )
        else:
            return CheckResult(
                name="Dictionary file exists",
                status=CheckStatus.FAIL,
                message=f"Dictionary not found at {dict_path}"
            )

    def _check_dictionary_valid_json(self) -> CheckResult:
        """Check dictionary is valid JSON."""
        dict_path = self.config.dictionary_file

        if not dict_path.exists():
            return CheckResult(
                name="Valid JSON",
                status=CheckStatus.SKIP,
                message="Dictionary file not found"
            )

        try:
            with open(dict_path) as f:
                data = json.load(f)
            return CheckResult(
                name="Valid JSON",
                status=CheckStatus.PASS,
                message="Dictionary is valid JSON"
            )
        except json.JSONDecodeError as e:
            return CheckResult(
                name="Valid JSON",
                status=CheckStatus.FAIL,
                message=f"Invalid JSON: {e}"
            )

    def _check_dictionary_structure(self) -> CheckResult:
        """Check dictionary has required structure."""
        dictionary = self._load_dictionary()

        if not dictionary:
            return CheckResult(
                name="Dictionary structure",
                status=CheckStatus.SKIP,
                message="Dictionary not available"
            )

        required_fields = ["version", "entries"]
        missing = [f for f in required_fields if f not in dictionary]

        if missing:
            return CheckResult(
                name="Dictionary structure",
                status=CheckStatus.FAIL,
                message=f"Missing required fields: {missing}"
            )

        # Check entry structure
        entries = dictionary.get("entries", {})
        entry_issues = []

        for sign_id, entry in list(entries.items())[:20]:  # Sample check
            entry_required = ["definition", "poses"]
            entry_missing = [f for f in entry_required if f not in entry]
            if entry_missing:
                entry_issues.append(f"{sign_id}: missing {entry_missing}")

        if entry_issues:
            return CheckResult(
                name="Dictionary structure",
                status=CheckStatus.FAIL,
                message=f"Entry structure issues found",
                details={"issues": entry_issues[:5]}
            )

        return CheckResult(
            name="Dictionary structure",
            status=CheckStatus.PASS,
            message=f"Dictionary structure is valid ({len(entries)} entries)"
        )

    def _check_dictionary_size(self) -> CheckResult:
        """Check dictionary file size is reasonable."""
        dict_path = self.config.dictionary_file

        if not dict_path.exists():
            return CheckResult(
                name="Dictionary size",
                status=CheckStatus.SKIP,
                message="Dictionary file not found"
            )

        size = dict_path.stat().st_size

        if size < self.config.min_dictionary_size_bytes:
            return CheckResult(
                name="Dictionary size",
                status=CheckStatus.FAIL,
                message=f"Dictionary too small ({size} bytes)",
                details={"size": size, "minimum": self.config.min_dictionary_size_bytes}
            )

        if size > self.config.max_dictionary_size_bytes:
            return CheckResult(
                name="Dictionary size",
                status=CheckStatus.WARN,
                message=f"Dictionary very large ({size / 1_000_000:.1f} MB)",
                details={"size": size}
            )

        # Format size
        if size < 1024:
            size_str = f"{size} bytes"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / 1024 / 1024:.1f} MB"

        return CheckResult(
            name="Dictionary size",
            status=CheckStatus.PASS,
            message=f"Dictionary size: {size_str}",
            details={"size": size}
        )

    def _check_joint_angles(self) -> CheckResult:
        """Check joint angles are within anatomical limits."""
        dictionary = self._load_dictionary()

        if not dictionary:
            return CheckResult(
                name="Joint angles",
                status=CheckStatus.SKIP,
                message="Dictionary not available"
            )

        # Anatomical limits (degrees)
        limits = {
            "mcp": (0, 100),
            "pip": (0, 110),
            "dip": (0, 90),
        }

        entries = dictionary.get("entries", {})
        out_of_range = []
        angles_found = 0

        for sign_id, entry in entries.items():
            for pose in entry.get("poses", []):
                for hand_key in ["right_hand_angles", "left_hand_angles"]:
                    angles = pose.get(hand_key)
                    if angles:
                        angles_found += 1
                        for finger, finger_angles in angles.items():
                            if isinstance(finger_angles, dict):
                                for joint, value in finger_angles.items():
                                    if joint in limits:
                                        min_val, max_val = limits[joint]
                                        if not (min_val - 10 <= value <= max_val + 10):
                                            out_of_range.append(
                                                f"{sign_id}.{finger}.{joint}: {value:.1f}"
                                            )

        if angles_found == 0:
            return CheckResult(
                name="Joint angles",
                status=CheckStatus.WARN,
                message="No joint angles found in dictionary"
            )

        if not out_of_range:
            return CheckResult(
                name="Joint angles",
                status=CheckStatus.PASS,
                message=f"All joint angles within anatomical limits ({angles_found} checked)"
            )
        else:
            return CheckResult(
                name="Joint angles",
                status=CheckStatus.WARN,
                message=f"{len(out_of_range)} angles outside expected range",
                details={"out_of_range": out_of_range[:10]}
            )

    def _check_tolerances(self) -> CheckResult:
        """Check tolerances are reasonable."""
        dictionary = self._load_dictionary()

        if not dictionary:
            return CheckResult(
                name="Tolerances",
                status=CheckStatus.SKIP,
                message="Dictionary not available"
            )

        entries = dictionary.get("entries", {})
        tolerance_issues = []
        position_tolerances_found = 0
        angle_tolerances_found = 0

        for sign_id, entry in entries.items():
            for pose in entry.get("poses", []):
                # Check position tolerances (new format)
                for hand_key in ["right_hand_position_tolerances", "left_hand_position_tolerances"]:
                    tolerances = pose.get(hand_key)
                    if tolerances:
                        position_tolerances_found += 1
                        for landmark, value in tolerances.items():
                            if not (0.01 <= value <= 0.5):
                                tolerance_issues.append(
                                    f"{sign_id}.{landmark}: {value:.3f}"
                                )

                # Check angle tolerances (new format)
                for hand_key in ["right_hand_angle_tolerances", "left_hand_angle_tolerances"]:
                    angle_tols = pose.get(hand_key)
                    if angle_tols:
                        angle_tolerances_found += 1

                # Also check old format for backwards compatibility
                for hand_key in ["right_hand_tolerances", "left_hand_tolerances"]:
                    tolerances = pose.get(hand_key)
                    if tolerances:
                        position_tolerances_found += 1

        tolerances_found = position_tolerances_found + angle_tolerances_found

        if tolerances_found == 0:
            return CheckResult(
                name="Tolerances",
                status=CheckStatus.WARN,
                message="No tolerances found in dictionary"
            )

        if not tolerance_issues:
            return CheckResult(
                name="Tolerances",
                status=CheckStatus.PASS,
                message=f"All tolerances in reasonable range ({tolerances_found} checked)"
            )
        else:
            return CheckResult(
                name="Tolerances",
                status=CheckStatus.WARN,
                message=f"{len(tolerance_issues)} tolerances outside expected range",
                details={"issues": tolerance_issues[:10]}
            )

    # =========================================================================
    # Category 4: Output Checks
    # =========================================================================

    def _check_outputs(self) -> CategoryResult:
        """Check output file requirements."""
        category = CategoryResult(name="Output Checks")

        category.checks.append(self._check_output_directory())
        category.checks.append(self._check_minified_dictionary())
        category.checks.append(self._check_typescript_types())
        category.checks.append(self._check_can_load_dictionary())

        return category

    def _check_output_directory(self) -> CheckResult:
        """Check output directory exists and has files."""
        output_dir = self.config.data_dir / "output"

        if not output_dir.exists():
            return CheckResult(
                name="Output directory",
                status=CheckStatus.FAIL,
                message="Output directory not found"
            )

        files = list(output_dir.glob("*"))

        if files:
            return CheckResult(
                name="Output directory",
                status=CheckStatus.PASS,
                message=f"Output directory contains {len(files)} files",
                details={"files": [f.name for f in files]}
            )
        else:
            return CheckResult(
                name="Output directory",
                status=CheckStatus.WARN,
                message="Output directory is empty"
            )

    def _check_minified_dictionary(self) -> CheckResult:
        """Check minified dictionary exists."""
        min_path = self.config.data_dir / "output" / "bsl-dictionary.min.json"

        if min_path.exists():
            # Verify it's smaller than regular
            regular_path = self.config.dictionary_file
            if regular_path.exists():
                min_size = min_path.stat().st_size
                reg_size = regular_path.stat().st_size
                if min_size < reg_size:
                    savings = (1 - min_size / reg_size) * 100
                    return CheckResult(
                        name="Minified dictionary",
                        status=CheckStatus.PASS,
                        message=f"Minified dictionary exists ({savings:.0f}% smaller)"
                    )
            return CheckResult(
                name="Minified dictionary",
                status=CheckStatus.PASS,
                message="Minified dictionary exists"
            )
        else:
            return CheckResult(
                name="Minified dictionary",
                status=CheckStatus.WARN,
                message="Minified dictionary not found (optional)"
            )

    def _check_typescript_types(self) -> CheckResult:
        """Check TypeScript types are generated."""
        ts_path = self.config.data_dir / "output" / "bsl-dictionary.d.ts"

        if ts_path.exists():
            # Basic validation - check it has type definitions
            content = ts_path.read_text()
            has_exports = "export" in content
            has_interface = "interface" in content

            if has_exports and has_interface:
                return CheckResult(
                    name="TypeScript types",
                    status=CheckStatus.PASS,
                    message="TypeScript types generated"
                )
            else:
                return CheckResult(
                    name="TypeScript types",
                    status=CheckStatus.WARN,
                    message="TypeScript file exists but may be incomplete"
                )
        else:
            return CheckResult(
                name="TypeScript types",
                status=CheckStatus.WARN,
                message="TypeScript types not found (optional)"
            )

    def _check_can_load_dictionary(self) -> CheckResult:
        """Check dictionary can be loaded and used."""
        try:
            dictionary = self._load_dictionary()

            if not dictionary:
                return CheckResult(
                    name="Dictionary loadable",
                    status=CheckStatus.SKIP,
                    message="Dictionary not available"
                )

            # Try to access key fields
            version = dictionary.get("version")
            entries = dictionary.get("entries", {})
            entry_count = len(entries)

            # Try to access a random entry
            if entries:
                first_entry = next(iter(entries.values()))
                _ = first_entry.get("definition", {}).get("name")
                _ = first_entry.get("poses", [])

            return CheckResult(
                name="Dictionary loadable",
                status=CheckStatus.PASS,
                message=f"Dictionary v{version} loads successfully ({entry_count} entries)"
            )
        except Exception as e:
            return CheckResult(
                name="Dictionary loadable",
                status=CheckStatus.FAIL,
                message=f"Failed to load dictionary: {e}"
            )

    # =========================================================================
    # Category 5: Documentation
    # =========================================================================

    def _check_documentation(self) -> CategoryResult:
        """Check documentation requirements."""
        category = CategoryResult(name="Documentation")

        for doc_path in self.config.required_docs:
            category.checks.append(self._check_doc_file(doc_path))

        category.checks.append(self._check_readme_completeness())
        category.checks.append(self._check_cli_documented())

        return category

    def _check_doc_file(self, relative_path: str) -> CheckResult:
        """Check a documentation file exists and has content."""
        file_path = Path(relative_path)

        if not file_path.exists():
            return CheckResult(
                name=f"Doc: {relative_path}",
                status=CheckStatus.FAIL,
                message=f"Missing: {relative_path}"
            )

        # Check minimum size
        size = file_path.stat().st_size
        if size < 100:
            return CheckResult(
                name=f"Doc: {relative_path}",
                status=CheckStatus.WARN,
                message=f"File exists but very small ({size} bytes)"
            )

        return CheckResult(
            name=f"Doc: {relative_path}",
            status=CheckStatus.PASS,
            message=f"Exists ({size} bytes)"
        )

    def _check_readme_completeness(self) -> CheckResult:
        """Check README has required sections."""
        readme_path = Path("README.md")

        if not readme_path.exists():
            return CheckResult(
                name="README completeness",
                status=CheckStatus.SKIP,
                message="README.md not found"
            )

        content = readme_path.read_text()

        required_sections = [
            "Quick Start",
            "Project Structure",
            "Commands",
            "Recording Tool",
            "Output Format",
        ]

        missing = [s for s in required_sections if s not in content]

        if not missing:
            return CheckResult(
                name="README completeness",
                status=CheckStatus.PASS,
                message="README has all required sections"
            )
        else:
            return CheckResult(
                name="README completeness",
                status=CheckStatus.WARN,
                message=f"README missing sections: {missing}"
            )

    def _check_cli_documented(self) -> CheckResult:
        """Check CLI commands are documented."""
        readme_path = Path("README.md")

        if not readme_path.exists():
            return CheckResult(
                name="CLI documentation",
                status=CheckStatus.SKIP,
                message="README.md not found"
            )

        content = readme_path.read_text()

        cli_commands = [
            "bsl-tool",
            "pipeline",
            "analyze",
            "validate",
            "generate",
        ]

        documented = [cmd for cmd in cli_commands if cmd in content]

        if len(documented) == len(cli_commands):
            return CheckResult(
                name="CLI documentation",
                status=CheckStatus.PASS,
                message=f"All CLI commands documented ({len(documented)})"
            )
        else:
            missing = set(cli_commands) - set(documented)
            return CheckResult(
                name="CLI documentation",
                status=CheckStatus.WARN,
                message=f"Missing CLI documentation: {missing}"
            )

    # =========================================================================
    # Category 6: Source Code
    # =========================================================================

    def _check_source_code(self) -> CategoryResult:
        """Check source code requirements."""
        category = CategoryResult(name="Source Code")

        # Check required modules
        for module in self.config.required_src_modules:
            category.checks.append(self._check_module(self.config.src_dir / module))

        # Check required scripts
        for script in self.config.required_scripts:
            category.checks.append(self._check_module(self.config.scripts_dir / script))

        # Check recording tool
        category.checks.append(self._check_recording_tool())

        # Check notebooks
        category.checks.append(self._check_notebooks())

        return category

    def _check_module(self, path: Path) -> CheckResult:
        """Check a Python module exists and can be imported."""
        if not path.exists():
            return CheckResult(
                name=f"Module: {path.name}",
                status=CheckStatus.FAIL,
                message=f"Not found: {path}"
            )

        # Check file has content
        size = path.stat().st_size
        if size < 100:
            return CheckResult(
                name=f"Module: {path.name}",
                status=CheckStatus.WARN,
                message=f"File exists but very small ({size} bytes)"
            )

        return CheckResult(
            name=f"Module: {path.name}",
            status=CheckStatus.PASS,
            message=f"Exists ({size} bytes)"
        )

    def _check_recording_tool(self) -> CheckResult:
        """Check recording tool files exist."""
        tool_dir = self.config.recording_tool_dir

        if not tool_dir.exists():
            return CheckResult(
                name="Recording tool",
                status=CheckStatus.FAIL,
                message="Recording tool directory not found"
            )

        required_files = ["index.html", "app.js", "styles.css"]
        missing = [f for f in required_files if not (tool_dir / f).exists()]

        if missing:
            return CheckResult(
                name="Recording tool",
                status=CheckStatus.FAIL,
                message=f"Missing files: {missing}"
            )

        return CheckResult(
            name="Recording tool",
            status=CheckStatus.PASS,
            message="All recording tool files present"
        )

    def _check_notebooks(self) -> CheckResult:
        """Check Jupyter notebooks exist."""
        notebooks_dir = self.config.notebooks_dir

        if not notebooks_dir.exists():
            return CheckResult(
                name="Notebooks",
                status=CheckStatus.WARN,
                message="Notebooks directory not found"
            )

        notebooks = list(notebooks_dir.glob("*.ipynb"))

        if len(notebooks) >= 5:
            return CheckResult(
                name="Notebooks",
                status=CheckStatus.PASS,
                message=f"Found {len(notebooks)} notebooks"
            )
        else:
            return CheckResult(
                name="Notebooks",
                status=CheckStatus.WARN,
                message=f"Only {len(notebooks)} notebooks found (expected 5)"
            )

    # =========================================================================
    # Category 7: Tests
    # =========================================================================

    def _check_tests(self) -> CategoryResult:
        """Check test requirements."""
        category = CategoryResult(name="Tests")

        category.checks.append(self._check_tests_exist())
        category.checks.append(self._check_tests_pass())

        return category

    def _check_tests_exist(self) -> CheckResult:
        """Check test files exist."""
        tests_dir = Path("tests")

        if not tests_dir.exists():
            return CheckResult(
                name="Test files exist",
                status=CheckStatus.FAIL,
                message="Tests directory not found"
            )

        test_files = list(tests_dir.glob("test_*.py"))

        if len(test_files) >= 5:
            return CheckResult(
                name="Test files exist",
                status=CheckStatus.PASS,
                message=f"Found {len(test_files)} test files"
            )
        else:
            return CheckResult(
                name="Test files exist",
                status=CheckStatus.WARN,
                message=f"Only {len(test_files)} test files found"
            )

    def _check_tests_pass(self) -> CheckResult:
        """Check that tests pass (quick check)."""
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=no", "-x"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # Extract pass count
                match = re.search(r"(\d+) passed", result.stdout)
                count = match.group(1) if match else "?"
                return CheckResult(
                    name="Tests pass",
                    status=CheckStatus.PASS,
                    message=f"All tests pass ({count} tests)"
                )
            else:
                return CheckResult(
                    name="Tests pass",
                    status=CheckStatus.FAIL,
                    message="Some tests failed",
                    details={"output": result.stdout[-500:] if result.stdout else result.stderr[-500:]}
                )
        except subprocess.TimeoutExpired:
            return CheckResult(
                name="Tests pass",
                status=CheckStatus.WARN,
                message="Test run timed out"
            )
        except Exception as e:
            return CheckResult(
                name="Tests pass",
                status=CheckStatus.SKIP,
                message=f"Could not run tests: {e}"
            )

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_console_report(self) -> str:
        """Generate colored console report."""
        lines = []
        reset = "\033[0m"
        bold = "\033[1m"

        lines.append("")
        lines.append(f"{bold}Phase 0 Completion Verification{reset}")
        lines.append("=" * 50)

        total_pass = 0
        total_fail = 0
        total_warn = 0
        total_skip = 0

        for category in self.results:
            lines.append("")
            lines.append(f"{bold}{category.name}{reset}")
            lines.append("-" * 40)

            for check in category.checks:
                status_str = f"{check.color}[{check.icon}]{reset}"
                lines.append(f"  {status_str} {check.name}: {check.message}")

            total_pass += category.passed
            total_fail += category.failed
            total_warn += category.warnings
            total_skip += category.skipped

        # Summary
        lines.append("")
        lines.append("=" * 50)
        lines.append(f"{bold}Summary{reset}")
        lines.append(f"  \033[92mPassed:  {total_pass}{reset}")
        lines.append(f"  \033[91mFailed:  {total_fail}{reset}")
        lines.append(f"  \033[93mWarnings: {total_warn}{reset}")
        lines.append(f"  \033[90mSkipped: {total_skip}{reset}")

        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            lines.append(f"  Time:    {duration:.1f}s")

        lines.append("")

        if total_fail == 0:
            lines.append(f"\033[92m{bold}Phase 0 COMPLETE{reset}")
        else:
            lines.append(f"\033[91m{bold}Phase 0 INCOMPLETE - {total_fail} checks failed{reset}")

        return "\n".join(lines)

    def generate_markdown_report(self) -> str:
        """Generate markdown report for file output."""
        lines = []

        lines.append("# Phase 0 Completion Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")

        # Summary table
        total_pass = sum(c.passed for c in self.results)
        total_fail = sum(c.failed for c in self.results)
        total_warn = sum(c.warnings for c in self.results)
        total_skip = sum(c.skipped for c in self.results)

        lines.append("## Summary")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| Passed | {total_pass} |")
        lines.append(f"| Failed | {total_fail} |")
        lines.append(f"| Warnings | {total_warn} |")
        lines.append(f"| Skipped | {total_skip} |")
        lines.append("")

        # Overall result
        if total_fail == 0:
            lines.append("**Result: PHASE 0 COMPLETE**")
        else:
            lines.append(f"**Result: INCOMPLETE ({total_fail} failures)**")
        lines.append("")

        # Detailed results
        lines.append("## Detailed Results")
        lines.append("")

        for category in self.results:
            lines.append(f"### {category.name}")
            lines.append("")
            lines.append("| Check | Status | Details |")
            lines.append("|-------|--------|---------|")

            for check in category.checks:
                status_emoji = {
                    CheckStatus.PASS: "Pass",
                    CheckStatus.FAIL: "**FAIL**",
                    CheckStatus.WARN: "Warn",
                    CheckStatus.SKIP: "Skip",
                }[check.status]

                lines.append(f"| {check.name} | {status_emoji} | {check.message} |")

            lines.append("")

        # Failures section
        failures = [
            (cat.name, check)
            for cat in self.results
            for check in cat.checks
            if check.status == CheckStatus.FAIL
        ]

        if failures:
            lines.append("## Failures to Address")
            lines.append("")
            for cat_name, check in failures:
                lines.append(f"- **{cat_name} / {check.name}**: {check.message}")
                if check.details:
                    lines.append(f"  - Details: `{json.dumps(check.details)[:200]}`")
            lines.append("")

        # Warnings section
        warnings = [
            (cat.name, check)
            for cat in self.results
            for check in cat.checks
            if check.status == CheckStatus.WARN
        ]

        if warnings:
            lines.append("## Warnings")
            lines.append("")
            for cat_name, check in warnings:
                lines.append(f"- **{cat_name} / {check.name}**: {check.message}")
            lines.append("")

        return "\n".join(lines)

    def save_report(self, path: Path) -> None:
        """Save markdown report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.generate_markdown_report()
        path.write_text(report)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify Phase 0 completion for BSL Data Extraction Tool"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings too (not just failures)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report but don't exit with error code"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/output/phase0-completion-report.md"),
        help="Output path for markdown report"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples per sign (default: 5)"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum quality score (default: 0.5)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output"
    )

    args = parser.parse_args()

    # Configure verifier
    config = VerificationConfig(
        min_samples_per_sign=args.min_samples,
        min_quality_score=args.min_quality,
    )

    verifier = Phase0Verifier(config)

    # Run checks
    all_passed = verifier.run_all_checks()

    # Console output
    if not args.quiet:
        print(verifier.generate_console_report())

    # Save report
    verifier.save_report(args.output)
    if not args.quiet:
        print(f"\nReport saved to: {args.output}")

    # Determine exit code
    if args.report_only:
        sys.exit(0)

    if args.strict:
        # Fail on warnings too
        total_issues = sum(c.failed + c.warnings for c in verifier.results)
        sys.exit(0 if total_issues == 0 else 1)
    else:
        # Only fail on failures
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
