#!/usr/bin/env python3
"""
BSL Data Extraction Pipeline

Complete pipeline for processing BSL sign data from raw sources to final dictionary.
Supports resumable execution, progress tracking, and detailed reporting.
"""

import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.types import (
    Point3D,
    HandLandmarks,
    HandLandmark,
    DetectedHand,
    RecordingFrame,
    Recording,
    RecordingSession,
    SignPose,
    SignDictionaryEntry,
    SignDictionary,
    ProcessingResult,
    NormalizationParams,
    ValidationResult,
)
from src.landmarks import LANDMARK_NAMES, NUM_LANDMARKS
from src.sign_definitions import SIGN_DEFINITIONS, get_sign_by_id
from src.normalize import normalize_all, NormalizationResult
from src.angles import calculate_all_joint_angles
from src.validation import (
    validate_landmarks,
    validate_recording,
    validate_dictionary,
)


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure pipeline logging."""
    logger = logging.getLogger("bsl-pipeline")
    logger.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Pipeline Enums and Types
# =============================================================================


class PipelineStep(Enum):
    """Pipeline execution steps."""
    IMPORT_KAGGLE = "import_kaggle"
    IMPORT_RECORDINGS = "import_recordings"
    PROCESS_ALL = "process_all"
    VALIDATE_ALL = "validate_all"
    GENERATE_DICTIONARY = "generate_dictionary"
    GENERATE_REPORTS = "generate_reports"
    COMPLETE = "complete"


@dataclass
class StepResult:
    """Result of a pipeline step execution."""
    step: PipelineStep
    success: bool
    duration_seconds: float
    items_processed: int = 0
    items_skipped: int = 0
    items_failed: int = 0
    message: str = ""
    details: dict = field(default_factory=dict)


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the BSL processing pipeline."""

    # Directory paths
    kaggle_dir: Path = field(default_factory=lambda: Path("data/kaggle-bsl"))
    recordings_dir: Path = field(default_factory=lambda: Path("data/raw-recordings"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    output_dir: Path = field(default_factory=lambda: Path("data/output"))
    state_dir: Path = field(default_factory=lambda: Path("data/.pipeline"))

    # Dictionary settings
    version: str = "1.0.0"
    dictionary_name: str = "bsl-dictionary"

    # Processing parameters
    min_samples_per_sign: int = 1
    quality_threshold: float = 0.6
    min_confidence: float = 0.5
    outlier_method: str = "zscore"
    averaging_method: str = "trimmed"
    tolerance_multiplier: float = 2.0

    # Normalization settings
    normalization_anchor: str = "wrist"
    normalization_scale: str = "palm_width"
    apply_rotation: bool = True

    # Pipeline behavior
    skip_existing: bool = True
    fail_fast: bool = False
    generate_minified: bool = True
    generate_typescript: bool = True

    # Logging
    log_file: Optional[Path] = None
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Convert string paths to Path objects
        path_fields = [
            "kaggle_dir", "recordings_dir", "processed_dir",
            "output_dir", "state_dir", "log_file"
        ]
        for field_name in path_fields:
            if field_name in data and data[field_name] is not None:
                data[field_name] = Path(data[field_name])

        return cls(**data)

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        path_fields = [
            "kaggle_dir", "recordings_dir", "processed_dir",
            "output_dir", "state_dir", "log_file"
        ]
        for field_name in path_fields:
            if field_name in data and data[field_name] is not None:
                data[field_name] = Path(data[field_name])

        return cls(**data)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []

        if self.min_samples_per_sign < 1:
            errors.append("min_samples_per_sign must be >= 1")

        if not 0 <= self.quality_threshold <= 1:
            errors.append("quality_threshold must be between 0 and 1")

        if not 0 <= self.min_confidence <= 1:
            errors.append("min_confidence must be between 0 and 1")

        if self.outlier_method not in ["zscore", "iqr", "none"]:
            errors.append(f"Invalid outlier_method: {self.outlier_method}")

        if self.averaging_method not in ["mean", "trimmed", "median"]:
            errors.append(f"Invalid averaging_method: {self.averaging_method}")

        return errors


# =============================================================================
# Pipeline State
# =============================================================================


@dataclass
class PipelineState:
    """Tracks pipeline execution state for resumability."""

    current_step: PipelineStep = PipelineStep.IMPORT_KAGGLE
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Per-step tracking
    completed_steps: list[str] = field(default_factory=list)
    step_results: dict[str, dict] = field(default_factory=dict)

    # Item tracking for incremental processing
    processed_kaggle_files: set[str] = field(default_factory=set)
    processed_recording_files: set[str] = field(default_factory=set)
    processed_signs: set[str] = field(default_factory=set)

    # Progress tracking
    progress: dict[str, Any] = field(default_factory=dict)

    # Issues
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Save state to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "current_step": self.current_step.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "completed_steps": self.completed_steps,
            "step_results": self.step_results,
            "processed_kaggle_files": list(self.processed_kaggle_files),
            "processed_recording_files": list(self.processed_recording_files),
            "processed_signs": list(self.processed_signs),
            "progress": self.progress,
            "errors": self.errors,
            "warnings": self.warnings,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        """Load state from file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        state = cls()
        state.current_step = PipelineStep(data.get("current_step", "import_kaggle"))
        state.started_at = (
            datetime.fromisoformat(data["started_at"])
            if data.get("started_at") else None
        )
        state.completed_at = (
            datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at") else None
        )
        state.completed_steps = data.get("completed_steps", [])
        state.step_results = data.get("step_results", {})
        state.processed_kaggle_files = set(data.get("processed_kaggle_files", []))
        state.processed_recording_files = set(data.get("processed_recording_files", []))
        state.processed_signs = set(data.get("processed_signs", []))
        state.progress = data.get("progress", {})
        state.errors = data.get("errors", [])
        state.warnings = data.get("warnings", [])

        return state

    def add_error(self, step: str, message: str, details: Optional[dict] = None) -> None:
        """Record an error."""
        self.errors.append({
            "step": step,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })

    def add_warning(self, step: str, message: str, details: Optional[dict] = None) -> None:
        """Record a warning."""
        self.warnings.append({
            "step": step,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })

    def mark_step_complete(self, step: PipelineStep, result: StepResult) -> None:
        """Mark a step as completed."""
        step_name = step.value
        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)

        self.step_results[step_name] = {
            "success": result.success,
            "duration_seconds": result.duration_seconds,
            "items_processed": result.items_processed,
            "items_skipped": result.items_skipped,
            "items_failed": result.items_failed,
            "message": result.message,
            "details": result.details,
            "completed_at": datetime.now().isoformat(),
        }

    def is_step_complete(self, step: PipelineStep) -> bool:
        """Check if a step has been completed."""
        return step.value in self.completed_steps

    def reset(self) -> None:
        """Reset state for a fresh run."""
        self.current_step = PipelineStep.IMPORT_KAGGLE
        self.started_at = None
        self.completed_at = None
        self.completed_steps = []
        self.step_results = {}
        self.processed_kaggle_files = set()
        self.processed_recording_files = set()
        self.processed_signs = set()
        self.progress = {}
        self.errors = []
        self.warnings = []


# =============================================================================
# BSL Pipeline
# =============================================================================


class BSLPipeline:
    """
    Complete BSL data processing pipeline.

    Handles the full workflow from raw data import to dictionary generation:
    1. Import Kaggle CSV data
    2. Import raw JSON recordings
    3. Process all imported data (normalize, average, calculate angles)
    4. Validate processed data
    5. Generate final dictionary
    6. Generate quality and coverage reports
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.state = PipelineState()
        self.logger = setup_logging(
            config.log_file,
            getattr(logging, config.log_level.upper(), logging.INFO)
        )

        # Validate configuration
        errors = config.validate()
        if errors:
            for error in errors:
                self.logger.error(f"Config error: {error}")
            raise ValueError(f"Invalid configuration: {errors}")

        # State file path
        self.state_file = config.state_dir / "pipeline_state.json"

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required directories."""
        for directory in [
            self.config.kaggle_dir,
            self.config.recordings_dir,
            self.config.processed_dir,
            self.config.output_dir,
            self.config.state_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Pipeline Execution
    # =========================================================================

    def run_full(self, resume: bool = True) -> dict:
        """
        Run the complete pipeline.

        Args:
            resume: If True, resume from last successful step. If False, start fresh.

        Returns:
            Summary dictionary with results from all steps.
        """
        self.logger.info("=" * 60)
        self.logger.info("BSL Data Extraction Pipeline")
        self.logger.info("=" * 60)

        # Load or reset state
        if resume and self.state_file.exists():
            self.state = PipelineState.load(self.state_file)
            self.logger.info(f"Resuming from step: {self.state.current_step.value}")
        else:
            self.state = PipelineState()
            self.logger.info("Starting fresh pipeline run")

        self.state.started_at = datetime.now()
        self._save_state()

        # Define step execution order
        steps = [
            (PipelineStep.IMPORT_KAGGLE, self.import_kaggle_data),
            (PipelineStep.IMPORT_RECORDINGS, self.import_recordings),
            (PipelineStep.PROCESS_ALL, self.process_all),
            (PipelineStep.VALIDATE_ALL, self.validate_all),
            (PipelineStep.GENERATE_DICTIONARY, self.generate_dictionary),
            (PipelineStep.GENERATE_REPORTS, self.generate_reports),
        ]

        # Find starting point for resume
        start_index = 0
        if resume:
            for i, (step, _) in enumerate(steps):
                if not self.state.is_step_complete(step):
                    start_index = i
                    break
            else:
                # All steps complete
                self.logger.info("All steps already complete")
                return self._generate_summary()

        # Execute steps
        for step, func in steps[start_index:]:
            self.state.current_step = step
            self._save_state()

            self.logger.info("")
            self.logger.info("-" * 40)
            self.logger.info(f"Step: {step.value}")
            self.logger.info("-" * 40)

            try:
                result = func()
                self.state.mark_step_complete(step, result)
                self._save_state()

                if result.success:
                    self.logger.info(f"[OK] {result.message}")
                else:
                    self.logger.warning(f"[WARN] {result.message}")
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to fail_fast setting")
                        break

            except Exception as e:
                self.logger.exception(f"Step failed with error: {e}")
                self.state.add_error(step.value, str(e))
                self._save_state()

                if self.config.fail_fast:
                    raise

        # Mark pipeline complete
        self.state.current_step = PipelineStep.COMPLETE
        self.state.completed_at = datetime.now()
        self._save_state()

        return self._generate_summary()

    def run_step(self, step: PipelineStep) -> StepResult:
        """Run a single pipeline step."""
        step_funcs = {
            PipelineStep.IMPORT_KAGGLE: self.import_kaggle_data,
            PipelineStep.IMPORT_RECORDINGS: self.import_recordings,
            PipelineStep.PROCESS_ALL: self.process_all,
            PipelineStep.VALIDATE_ALL: self.validate_all,
            PipelineStep.GENERATE_DICTIONARY: self.generate_dictionary,
            PipelineStep.GENERATE_REPORTS: self.generate_reports,
        }

        if step not in step_funcs:
            raise ValueError(f"Unknown step: {step}")

        return step_funcs[step]()

    def _save_state(self) -> None:
        """Save current state to file."""
        self.state.save(self.state_file)

    # =========================================================================
    # Step 1: Import Kaggle Data
    # =========================================================================

    def import_kaggle_data(self) -> StepResult:
        """Import and convert Kaggle BSL dataset."""
        start_time = time.time()
        processed = 0
        skipped = 0
        failed = 0

        # Find CSV/Parquet files
        csv_files = list(self.config.kaggle_dir.glob("*.csv"))
        parquet_files = list(self.config.kaggle_dir.glob("*.parquet"))
        all_files = csv_files + parquet_files

        if not all_files:
            return StepResult(
                step=PipelineStep.IMPORT_KAGGLE,
                success=True,
                duration_seconds=time.time() - start_time,
                message="No Kaggle files found (skipped)",
            )

        self.logger.info(f"Found {len(all_files)} Kaggle files")

        for file_path in all_files:
            file_name = file_path.name

            # Skip if already processed
            if self.config.skip_existing and file_name in self.state.processed_kaggle_files:
                self.logger.debug(f"Skipping already processed: {file_name}")
                skipped += 1
                continue

            try:
                # Import the file
                records = self._load_kaggle_file(file_path)
                self.logger.info(f"Loaded {len(records)} records from {file_name}")

                # Convert and save each record
                for i, record in enumerate(records):
                    try:
                        recording = self._convert_kaggle_record(record)
                        if recording:
                            out_file = self.config.recordings_dir / f"kaggle_{file_path.stem}_{i:06d}.json"
                            self._save_recording(recording, out_file)
                            processed += 1
                    except Exception as e:
                        self.logger.debug(f"Record {i} failed: {e}")
                        failed += 1

                self.state.processed_kaggle_files.add(file_name)

            except Exception as e:
                self.logger.error(f"Failed to process {file_name}: {e}")
                self.state.add_error("import_kaggle", f"File {file_name}: {e}")
                failed += 1

        duration = time.time() - start_time
        return StepResult(
            step=PipelineStep.IMPORT_KAGGLE,
            success=failed == 0 or processed > 0,
            duration_seconds=duration,
            items_processed=processed,
            items_skipped=skipped,
            items_failed=failed,
            message=f"Imported {processed} records, skipped {skipped}, failed {failed}",
        )

    def _load_kaggle_file(self, path: Path) -> list[dict]:
        """Load records from a Kaggle CSV or Parquet file."""
        records = []

        if path.suffix == ".csv":
            # Simple CSV parsing (basic implementation)
            import csv
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)

        elif path.suffix == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                records = df.to_dict("records")
            except ImportError:
                self.logger.warning("pandas not available, skipping parquet file")

        return records

    def _convert_kaggle_record(self, record: dict) -> Optional[dict]:
        """Convert a Kaggle record to our recording format."""
        # Extract sign ID
        sign_id = record.get("sign") or record.get("label") or record.get("sign_id")
        if not sign_id:
            return None

        # Extract landmarks
        landmarks = []
        for i in range(NUM_LANDMARKS):
            x_key = f"x_right_hand_{i}"
            y_key = f"y_right_hand_{i}"
            z_key = f"z_right_hand_{i}"

            if x_key in record:
                try:
                    landmarks.append({
                        "x": float(record[x_key]),
                        "y": float(record[y_key]),
                        "z": float(record.get(z_key, 0)),
                    })
                except (ValueError, TypeError):
                    pass

        if len(landmarks) != NUM_LANDMARKS:
            return None

        # Create recording structure
        return {
            "sign_id": str(sign_id).lower().replace(" ", "_"),
            "frames": [{
                "timestamp_ms": 0,
                "right_hand": {
                    "landmarks": landmarks,
                    "handedness": "Right",
                    "confidence": 0.9,
                },
            }],
            "metadata": {
                "source": "kaggle",
                "original_file": record.get("file", "unknown"),
            },
        }

    def _save_recording(self, recording: dict, path: Path) -> None:
        """Save a recording to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(recording, f, indent=2)

    # =========================================================================
    # Step 2: Import Recordings
    # =========================================================================

    def import_recordings(self) -> StepResult:
        """Import raw JSON recordings from the recording tool."""
        start_time = time.time()
        processed = 0
        skipped = 0
        failed = 0

        # Find JSON files (excluding kaggle imports)
        json_files = [
            f for f in self.config.recordings_dir.glob("*.json")
            if not f.name.startswith("kaggle_")
        ]

        if not json_files:
            return StepResult(
                step=PipelineStep.IMPORT_RECORDINGS,
                success=True,
                duration_seconds=time.time() - start_time,
                message="No recording files found (skipped)",
            )

        self.logger.info(f"Found {len(json_files)} recording files")

        for file_path in json_files:
            file_name = file_path.name

            if self.config.skip_existing and file_name in self.state.processed_recording_files:
                skipped += 1
                continue

            try:
                # Validate recording structure
                with open(file_path) as f:
                    data = json.load(f)

                if "frames" not in data or not data["frames"]:
                    self.state.add_warning("import_recordings", f"No frames in {file_name}")
                    failed += 1
                    continue

                # Recording is valid
                self.state.processed_recording_files.add(file_name)
                processed += 1

            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in {file_name}: {e}")
                failed += 1
            except Exception as e:
                self.logger.error(f"Failed to process {file_name}: {e}")
                failed += 1

        duration = time.time() - start_time
        return StepResult(
            step=PipelineStep.IMPORT_RECORDINGS,
            success=True,
            duration_seconds=duration,
            items_processed=processed,
            items_skipped=skipped,
            items_failed=failed,
            message=f"Validated {processed} recordings, skipped {skipped}, failed {failed}",
        )

    # =========================================================================
    # Step 3: Process All Data
    # =========================================================================

    def process_all(self) -> StepResult:
        """Process all imported data into canonical poses."""
        start_time = time.time()
        processed = 0
        skipped = 0
        failed = 0

        # Group recordings by sign ID
        recordings_by_sign: dict[str, list[Path]] = {}
        for file_path in self.config.recordings_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                sign_id = data.get("sign_id", file_path.stem.split("_")[0])
                recordings_by_sign.setdefault(sign_id, []).append(file_path)
            except Exception:
                pass

        self.logger.info(f"Found {len(recordings_by_sign)} unique signs")

        for sign_id, file_paths in recordings_by_sign.items():
            # Skip if already processed
            if self.config.skip_existing and sign_id in self.state.processed_signs:
                skipped += 1
                continue

            # Check minimum samples
            if len(file_paths) < self.config.min_samples_per_sign:
                self.state.add_warning(
                    "process_all",
                    f"Sign '{sign_id}' has only {len(file_paths)} samples "
                    f"(minimum: {self.config.min_samples_per_sign})"
                )

            try:
                result = self._process_sign(sign_id, file_paths)
                if result:
                    out_path = self.config.processed_dir / f"{sign_id}.json"
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)
                    self.state.processed_signs.add(sign_id)
                    processed += 1
                else:
                    failed += 1

            except Exception as e:
                self.logger.error(f"Failed to process sign '{sign_id}': {e}")
                self.state.add_error("process_all", f"Sign {sign_id}: {e}")
                failed += 1

        duration = time.time() - start_time
        return StepResult(
            step=PipelineStep.PROCESS_ALL,
            success=processed > 0 or (processed == 0 and failed == 0),
            duration_seconds=duration,
            items_processed=processed,
            items_skipped=skipped,
            items_failed=failed,
            message=f"Processed {processed} signs, skipped {skipped}, failed {failed}",
        )

    def _process_sign(self, sign_id: str, file_paths: list[Path]) -> Optional[dict]:
        """Process recordings for a single sign."""
        import numpy as np

        all_landmarks = []
        total_frames = 0

        # Collect all valid frames
        for file_path in file_paths:
            with open(file_path) as f:
                data = json.load(f)

            for frame in data.get("frames", []):
                for hand_key in ["right_hand", "left_hand"]:
                    hand = frame.get(hand_key)
                    if not hand:
                        continue

                    confidence = hand.get("confidence", 1.0)
                    if confidence < self.config.min_confidence:
                        continue

                    landmarks = hand.get("landmarks", [])
                    if len(landmarks) == NUM_LANDMARKS:
                        all_landmarks.append({
                            "hand": hand_key,
                            "landmarks": landmarks,
                        })
                        total_frames += 1

        if not all_landmarks:
            return None

        # Separate by hand
        right_samples = [s["landmarks"] for s in all_landmarks if s["hand"] == "right_hand"]
        left_samples = [s["landmarks"] for s in all_landmarks if s["hand"] == "left_hand"]

        result = {
            "sign_id": sign_id,
            "canonical_pose": {},
            "quality_score": 0.0,
            "sample_count": len(all_landmarks),
            "warnings": [],
        }

        # Process right hand
        if right_samples:
            canonical, tolerances = self._compute_canonical_pose(right_samples)
            if canonical:
                result["canonical_pose"]["right_hand_landmarks"] = canonical
                result["canonical_pose"]["right_hand_tolerances"] = tolerances

                # Calculate angles
                try:
                    angles = self._calculate_angles(canonical)
                    if angles:
                        result["canonical_pose"]["right_hand_angles"] = angles
                except Exception as e:
                    result["warnings"].append(f"Angle calculation failed: {e}")

        # Process left hand
        if left_samples:
            canonical, tolerances = self._compute_canonical_pose(left_samples)
            if canonical:
                result["canonical_pose"]["left_hand_landmarks"] = canonical
                result["canonical_pose"]["left_hand_tolerances"] = tolerances

                try:
                    angles = self._calculate_angles(canonical)
                    if angles:
                        result["canonical_pose"]["left_hand_angles"] = angles
                except Exception as e:
                    result["warnings"].append(f"Angle calculation failed: {e}")

        # Calculate quality score
        result["quality_score"] = self._calculate_quality_score(
            result, len(right_samples), len(left_samples)
        )

        return result

    def _compute_canonical_pose(
        self, samples: list[list[dict]]
    ) -> tuple[Optional[list[dict]], Optional[dict]]:
        """Compute canonical pose from samples using averaging."""
        import numpy as np

        if not samples:
            return None, None

        # Convert to numpy array
        try:
            arr = np.array([
                [[pt["x"], pt["y"], pt["z"]] for pt in sample]
                for sample in samples
            ])
        except (KeyError, TypeError):
            return None, None

        # Normalize each sample
        normalized_samples = []
        for sample in arr:
            try:
                # Simple normalization: translate to wrist origin, scale by palm width
                wrist = sample[0]
                translated = sample - wrist

                # Palm width: index MCP to pinky MCP
                palm_width = np.linalg.norm(translated[5] - translated[17])
                if palm_width > 1e-6:
                    scaled = translated / palm_width
                else:
                    scaled = translated

                normalized_samples.append(scaled)
            except Exception:
                continue

        if not normalized_samples:
            return None, None

        normalized_arr = np.array(normalized_samples)

        # Remove outliers (z-score method)
        if self.config.outlier_method == "zscore" and len(normalized_arr) > 3:
            mean = np.mean(normalized_arr, axis=0)
            std = np.std(normalized_arr, axis=0)
            std[std < 1e-6] = 1e-6  # Prevent division by zero

            z_scores = np.abs((normalized_arr - mean) / std)
            mask = np.all(z_scores < 2.5, axis=(1, 2))
            if np.sum(mask) > 0:
                normalized_arr = normalized_arr[mask]

        # Compute average
        if self.config.averaging_method == "median":
            canonical = np.median(normalized_arr, axis=0)
        elif self.config.averaging_method == "trimmed" and len(normalized_arr) > 4:
            # Trim 10% from each end
            trim = max(1, len(normalized_arr) // 10)
            sorted_arr = np.sort(normalized_arr, axis=0)
            canonical = np.mean(sorted_arr[trim:-trim], axis=0)
        else:
            canonical = np.mean(normalized_arr, axis=0)

        # Calculate tolerances
        std = np.std(normalized_arr, axis=0)
        tolerances = {}
        for i, name in enumerate(LANDMARK_NAMES):
            tol = float(np.mean(std[i]) * self.config.tolerance_multiplier)
            tolerances[name] = max(0.02, min(0.2, tol))  # Clamp to reasonable range

        # Convert back to list of dicts
        canonical_list = [
            {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])}
            for pt in canonical
        ]

        return canonical_list, tolerances

    def _calculate_angles(self, landmarks: list[dict]) -> Optional[dict]:
        """Calculate joint angles from landmarks."""
        import numpy as np

        # Convert to numpy array
        arr = np.array([[pt["x"], pt["y"], pt["z"]] for pt in landmarks])

        # Calculate angles for each finger
        angles = {}
        finger_joints = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20],
        }

        for finger, indices in finger_joints.items():
            finger_angles = {}
            for i, joint_name in enumerate(["mcp", "pip", "dip"]):
                if i + 2 < len(indices):
                    p1, p2, p3 = arr[indices[i]], arr[indices[i + 1]], arr[indices[i + 2]]
                    v1 = p1 - p2
                    v2 = p3 - p2
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    finger_angles[joint_name] = float(180 - angle)  # Flexion angle
            angles[finger] = finger_angles

        return angles

    def _calculate_quality_score(
        self, result: dict, right_count: int, left_count: int
    ) -> float:
        """Calculate overall quality score for a processed sign."""
        scores = []

        # Completeness: at least one hand has data
        has_data = bool(
            result.get("canonical_pose", {}).get("right_hand_landmarks") or
            result.get("canonical_pose", {}).get("left_hand_landmarks")
        )
        scores.append(1.0 if has_data else 0.0)

        # Sample count score
        total_samples = right_count + left_count
        sample_score = min(1.0, total_samples / 10.0)  # Max score at 10 samples
        scores.append(sample_score)

        # Tolerance score (lower is better)
        tolerances = result.get("canonical_pose", {}).get("right_hand_tolerances", {})
        if tolerances:
            avg_tol = sum(tolerances.values()) / len(tolerances)
            tol_score = max(0.0, 1.0 - avg_tol * 5)  # Lower tolerance = higher score
            scores.append(tol_score)

        # Angle completeness
        angles = result.get("canonical_pose", {}).get("right_hand_angles", {})
        if angles:
            scores.append(1.0)

        return sum(scores) / len(scores) if scores else 0.0

    # =========================================================================
    # Step 4: Validate All
    # =========================================================================

    def validate_all(self) -> StepResult:
        """Validate all processed data."""
        start_time = time.time()
        validated = 0
        failed = 0
        warnings_count = 0

        processed_files = list(self.config.processed_dir.glob("*.json"))

        if not processed_files:
            return StepResult(
                step=PipelineStep.VALIDATE_ALL,
                success=True,
                duration_seconds=time.time() - start_time,
                message="No processed files to validate",
            )

        self.logger.info(f"Validating {len(processed_files)} processed files")

        for file_path in processed_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)

                issues = self._validate_processed_data(data)

                if issues["errors"]:
                    for error in issues["errors"]:
                        self.state.add_error("validate_all", f"{file_path.name}: {error}")
                    failed += 1
                else:
                    validated += 1

                warnings_count += len(issues["warnings"])
                for warning in issues["warnings"]:
                    self.state.add_warning("validate_all", f"{file_path.name}: {warning}")

            except Exception as e:
                self.logger.error(f"Validation failed for {file_path.name}: {e}")
                failed += 1

        duration = time.time() - start_time
        return StepResult(
            step=PipelineStep.VALIDATE_ALL,
            success=failed == 0,
            duration_seconds=duration,
            items_processed=validated,
            items_failed=failed,
            message=f"Validated {validated} files, {failed} failed, {warnings_count} warnings",
            details={"warnings": warnings_count},
        )

    def _validate_processed_data(self, data: dict) -> dict:
        """Validate a processed pose file."""
        errors = []
        warnings = []

        # Check required fields
        if "sign_id" not in data:
            errors.append("Missing sign_id")

        if "canonical_pose" not in data:
            errors.append("Missing canonical_pose")
        else:
            pose = data["canonical_pose"]
            has_right = "right_hand_landmarks" in pose and pose["right_hand_landmarks"]
            has_left = "left_hand_landmarks" in pose and pose["left_hand_landmarks"]

            if not has_right and not has_left:
                errors.append("No hand landmarks in canonical_pose")

            # Validate landmark counts
            for hand in ["right_hand_landmarks", "left_hand_landmarks"]:
                if hand in pose and pose[hand]:
                    if len(pose[hand]) != NUM_LANDMARKS:
                        errors.append(f"{hand} has {len(pose[hand])} landmarks, expected {NUM_LANDMARKS}")

        # Check quality
        quality = data.get("quality_score", 0)
        if quality < self.config.quality_threshold:
            warnings.append(f"Quality score {quality:.2f} below threshold {self.config.quality_threshold}")

        return {"errors": errors, "warnings": warnings}

    # =========================================================================
    # Step 5: Generate Dictionary
    # =========================================================================

    def generate_dictionary(self) -> StepResult:
        """Generate the final BSL dictionary."""
        start_time = time.time()

        # Load all processed poses
        processed_files = list(self.config.processed_dir.glob("*.json"))
        entries = {}
        signs_with_data = 0
        signs_without_data = 0

        self.logger.info(f"Loading {len(processed_files)} processed poses")

        for file_path in processed_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)

                sign_id = data.get("sign_id", file_path.stem)

                # Get sign definition
                sign_def = get_sign_by_id(sign_id)
                if sign_def:
                    definition = {
                        "id": sign_def.id,
                        "name": sign_def.name,
                        "category": sign_def.category,
                        "difficulty": sign_def.difficulty,
                        "description": sign_def.description,
                        "two_handed": sign_def.two_handed,
                    }
                else:
                    definition = {
                        "id": sign_id,
                        "name": sign_id.replace("_", " ").title(),
                        "category": "other",
                        "difficulty": 3,
                        "description": "",
                        "two_handed": False,
                    }

                # Create entry
                entry = {
                    "definition": definition,
                    "poses": [data.get("canonical_pose", {})],
                    "sample_count": data.get("sample_count", 1),
                    "quality_score": data.get("quality_score", 0.5),
                    "source": "processed",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }

                entries[sign_id] = entry
                signs_with_data += 1

            except Exception as e:
                self.logger.warning(f"Failed to load {file_path.name}: {e}")

        # Add signs without data
        for sign_def in SIGN_DEFINITIONS:
            if sign_def.id not in entries:
                entries[sign_def.id] = {
                    "definition": {
                        "id": sign_def.id,
                        "name": sign_def.name,
                        "category": sign_def.category,
                        "difficulty": sign_def.difficulty,
                        "description": sign_def.description,
                        "two_handed": sign_def.two_handed,
                    },
                    "poses": [],
                    "sample_count": 0,
                    "quality_score": 0.0,
                    "source": "definition",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
                signs_without_data += 1

        # Create dictionary
        dictionary = {
            "version": self.config.version,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "entries": entries,
            "metadata": {
                "total_signs": len(entries),
                "signs_with_data": signs_with_data,
                "signs_without_data": signs_without_data,
                "coverage_percent": (signs_with_data / len(entries) * 100) if entries else 0,
            },
        }

        # Save dictionary
        dict_path = self.config.output_dir / f"{self.config.dictionary_name}.json"
        with open(dict_path, "w") as f:
            json.dump(dictionary, f, indent=2)
        self.logger.info(f"Saved dictionary to {dict_path}")

        # Save minified version
        if self.config.generate_minified:
            min_path = self.config.output_dir / f"{self.config.dictionary_name}.min.json"
            with open(min_path, "w") as f:
                json.dump(dictionary, f, separators=(",", ":"))
            self.logger.info(f"Saved minified dictionary to {min_path}")

        # Generate TypeScript types
        if self.config.generate_typescript:
            ts_path = self.config.output_dir / f"{self.config.dictionary_name}.d.ts"
            self._generate_typescript_types(dictionary, ts_path)
            self.logger.info(f"Saved TypeScript types to {ts_path}")

        duration = time.time() - start_time
        return StepResult(
            step=PipelineStep.GENERATE_DICTIONARY,
            success=True,
            duration_seconds=duration,
            items_processed=len(entries),
            message=f"Generated dictionary with {len(entries)} signs ({signs_with_data} with data)",
            details={
                "total_signs": len(entries),
                "signs_with_data": signs_with_data,
                "signs_without_data": signs_without_data,
            },
        )

    def _generate_typescript_types(self, dictionary: dict, path: Path) -> None:
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
}}

export interface SignPose {{
  left_hand_landmarks?: Point3D[];
  right_hand_landmarks?: Point3D[];
  left_hand_angles?: JointAngles;
  right_hand_angles?: JointAngles;
  left_hand_tolerances?: Record<string, number>;
  right_hand_tolerances?: Record<string, number>;
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
  metadata: DictionaryMetadata;
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

        with open(path, "w") as f:
            f.write(types_content)

    # =========================================================================
    # Step 6: Generate Reports
    # =========================================================================

    def generate_reports(self) -> StepResult:
        """Generate quality and coverage reports."""
        start_time = time.time()

        # Load dictionary
        dict_path = self.config.output_dir / f"{self.config.dictionary_name}.json"
        if not dict_path.exists():
            return StepResult(
                step=PipelineStep.GENERATE_REPORTS,
                success=False,
                duration_seconds=time.time() - start_time,
                message="Dictionary not found, cannot generate reports",
            )

        with open(dict_path) as f:
            dictionary = json.load(f)

        # Generate coverage report
        coverage_report = self._generate_coverage_report(dictionary)
        coverage_path = self.config.output_dir / "coverage-report.md"
        with open(coverage_path, "w") as f:
            f.write(coverage_report)

        # Generate quality report
        quality_report = self._generate_quality_report(dictionary)
        quality_path = self.config.output_dir / "quality-report.md"
        with open(quality_path, "w") as f:
            f.write(quality_report)

        # Generate pipeline summary
        summary = self._generate_summary()
        summary_path = self.config.output_dir / "pipeline-summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        duration = time.time() - start_time
        return StepResult(
            step=PipelineStep.GENERATE_REPORTS,
            success=True,
            duration_seconds=duration,
            items_processed=3,
            message=f"Generated 3 reports in {self.config.output_dir}",
        )

    def _generate_coverage_report(self, dictionary: dict) -> str:
        """Generate markdown coverage report."""
        entries = dictionary.get("entries", {})
        metadata = dictionary.get("metadata", {})

        # Group by category
        by_category: dict[str, dict] = {}
        for sign_id, entry in entries.items():
            cat = entry.get("definition", {}).get("category", "other")
            if cat not in by_category:
                by_category[cat] = {"total": 0, "with_data": 0}
            by_category[cat]["total"] += 1
            if entry.get("sample_count", 0) > 0:
                by_category[cat]["with_data"] += 1

        report = f"""# BSL Dictionary Coverage Report

Generated: {datetime.now().isoformat()}
Version: {dictionary.get("version", "unknown")}

## Summary

| Metric | Value |
|--------|-------|
| Total Signs | {metadata.get("total_signs", 0)} |
| Signs with Data | {metadata.get("signs_with_data", 0)} |
| Signs without Data | {metadata.get("signs_without_data", 0)} |
| Coverage | {metadata.get("coverage_percent", 0):.1f}% |

## Coverage by Category

| Category | Total | With Data | Coverage |
|----------|-------|-----------|----------|
"""

        for cat in sorted(by_category.keys()):
            stats = by_category[cat]
            coverage = (stats["with_data"] / stats["total"] * 100) if stats["total"] else 0
            report += f"| {cat} | {stats['total']} | {stats['with_data']} | {coverage:.1f}% |\n"

        # List signs without data
        missing = [
            sign_id for sign_id, entry in entries.items()
            if entry.get("sample_count", 0) == 0
        ]

        if missing:
            report += f"\n## Signs Without Data ({len(missing)})\n\n"
            for sign_id in sorted(missing)[:50]:
                report += f"- {sign_id}\n"
            if len(missing) > 50:
                report += f"\n... and {len(missing) - 50} more\n"

        return report

    def _generate_quality_report(self, dictionary: dict) -> str:
        """Generate markdown quality report."""
        entries = dictionary.get("entries", {})

        # Calculate quality statistics
        quality_scores = [
            entry.get("quality_score", 0)
            for entry in entries.values()
            if entry.get("sample_count", 0) > 0
        ]

        if not quality_scores:
            avg_quality = 0
            min_quality = 0
            max_quality = 0
        else:
            import statistics
            avg_quality = statistics.mean(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)

        # Quality distribution
        excellent = sum(1 for q in quality_scores if q >= 0.9)
        good = sum(1 for q in quality_scores if 0.75 <= q < 0.9)
        acceptable = sum(1 for q in quality_scores if 0.6 <= q < 0.75)
        poor = sum(1 for q in quality_scores if 0.4 <= q < 0.6)
        unusable = sum(1 for q in quality_scores if q < 0.4)

        report = f"""# BSL Dictionary Quality Report

Generated: {datetime.now().isoformat()}
Version: {dictionary.get("version", "unknown")}

## Quality Summary

| Metric | Value |
|--------|-------|
| Average Quality | {avg_quality:.2f} |
| Minimum Quality | {min_quality:.2f} |
| Maximum Quality | {max_quality:.2f} |
| Signs Analyzed | {len(quality_scores)} |

## Quality Distribution

| Level | Score Range | Count | Percentage |
|-------|-------------|-------|------------|
| Excellent | >= 0.90 | {excellent} | {excellent/len(quality_scores)*100:.1f}% |
| Good | 0.75 - 0.89 | {good} | {good/len(quality_scores)*100:.1f}% |
| Acceptable | 0.60 - 0.74 | {acceptable} | {acceptable/len(quality_scores)*100:.1f}% |
| Poor | 0.40 - 0.59 | {poor} | {poor/len(quality_scores)*100:.1f}% |
| Unusable | < 0.40 | {unusable} | {unusable/len(quality_scores)*100:.1f}% |
""" if quality_scores else "No quality scores available.\n"

        # Low quality signs
        low_quality = [
            (sign_id, entry.get("quality_score", 0))
            for sign_id, entry in entries.items()
            if 0 < entry.get("quality_score", 0) < self.config.quality_threshold
        ]

        if low_quality:
            report += f"\n## Signs Below Quality Threshold ({self.config.quality_threshold})\n\n"
            report += "| Sign ID | Quality Score |\n|---------|---------------|\n"
            for sign_id, score in sorted(low_quality, key=lambda x: x[1])[:20]:
                report += f"| {sign_id} | {score:.2f} |\n"
            if len(low_quality) > 20:
                report += f"\n... and {len(low_quality) - 20} more\n"

        return report

    def _generate_summary(self) -> dict:
        """Generate pipeline execution summary."""
        return {
            "pipeline": {
                "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
                "completed_at": self.state.completed_at.isoformat() if self.state.completed_at else None,
                "current_step": self.state.current_step.value,
                "completed_steps": self.state.completed_steps,
            },
            "config": self.config.to_dict(),
            "step_results": self.state.step_results,
            "statistics": {
                "kaggle_files_processed": len(self.state.processed_kaggle_files),
                "recording_files_processed": len(self.state.processed_recording_files),
                "signs_processed": len(self.state.processed_signs),
            },
            "issues": {
                "errors": len(self.state.errors),
                "warnings": len(self.state.warnings),
                "error_details": self.state.errors[:10],  # First 10 errors
                "warning_details": self.state.warnings[:10],  # First 10 warnings
            },
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def reset(self) -> None:
        """Reset pipeline state for a fresh run."""
        self.state.reset()
        self._save_state()
        self.logger.info("Pipeline state reset")

    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "current_step": self.state.current_step.value,
            "completed_steps": self.state.completed_steps,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "completed_at": self.state.completed_at.isoformat() if self.state.completed_at else None,
            "errors": len(self.state.errors),
            "warnings": len(self.state.warnings),
            "progress": {
                "kaggle_files": len(self.state.processed_kaggle_files),
                "recording_files": len(self.state.processed_recording_files),
                "signs": len(self.state.processed_signs),
            },
        }


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Command-line entry point for the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BSL Data Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming",
    )
    parser.add_argument(
        "--step",
        choices=[s.value for s in PipelineStep if s != PipelineStep.COMPLETE],
        help="Run only a specific step",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset pipeline state and exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        if args.config.suffix in [".yaml", ".yml"]:
            config = PipelineConfig.from_yaml(args.config)
        else:
            config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig()

    # Adjust log level
    if args.verbose >= 2:
        config.log_level = "DEBUG"
    elif args.verbose == 1:
        config.log_level = "INFO"

    # Create pipeline
    pipeline = BSLPipeline(config)

    # Handle commands
    if args.reset:
        pipeline.reset()
        print("Pipeline state reset.")
        return

    if args.status:
        status = pipeline.get_status()
        print(json.dumps(status, indent=2))
        return

    if args.step:
        step = PipelineStep(args.step)
        result = pipeline.run_step(step)
        print(f"\n{result.step.value}: {'OK' if result.success else 'FAILED'}")
        print(f"  {result.message}")
        return

    # Run full pipeline
    summary = pipeline.run_full(resume=not args.no_resume)

    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Steps completed: {len(summary['pipeline']['completed_steps'])}")
    print(f"Signs processed: {summary['statistics']['signs_processed']}")
    print(f"Errors: {summary['issues']['errors']}")
    print(f"Warnings: {summary['issues']['warnings']}")


if __name__ == "__main__":
    main()
