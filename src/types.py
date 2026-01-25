"""BSL Data Extraction - Pydantic type definitions.

This module defines all data types used throughout the BSL data extraction
pipeline, from raw landmark data to the final dictionary format.
"""

from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


# =============================================================================
# Landmark Types
# =============================================================================


class Point3D(BaseModel):
    """A 3D point with coordinates.

    For raw landmark data, coordinates are typically normalized to [0, 1]
    relative to image dimensions. For processed/canonical poses, coordinates
    are in a normalized space (e.g., wrist at origin, palm width = 1).
    """

    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")
    z: float = Field(description="Z/depth coordinate")


class HandLandmark(BaseModel):
    """A single hand landmark with position and identification.

    MediaPipe Hand Landmarker detects 21 landmarks per hand,
    each with a unique index and name.
    """

    point: Point3D = Field(description="3D position of the landmark")
    name: str = Field(description="Landmark name (e.g., 'WRIST', 'THUMB_TIP')")
    index: Annotated[int, Field(ge=0, le=20, description="Landmark index (0-20)")]


class HandLandmarks(BaseModel):
    """Complete set of 21 hand landmarks.

    Represents all landmarks detected on a single hand.
    Must contain exactly 21 landmarks in the correct order.
    """

    landmarks: list[HandLandmark] = Field(
        description="List of 21 hand landmarks", min_length=21, max_length=21
    )

    @model_validator(mode="after")
    def validate_landmark_indices(self) -> Self:
        """Ensure landmarks are in correct order with proper indices."""
        indices = [lm.index for lm in self.landmarks]
        expected = list(range(21))
        if sorted(indices) != expected:
            raise ValueError(
                f"Landmarks must have indices 0-20, got: {sorted(indices)}"
            )
        return self


Handedness = Literal["Left", "Right"]
"""Which hand was detected - Left or Right from the subject's perspective."""


class DetectedHand(BaseModel):
    """A detected hand with landmarks and metadata.

    Represents a single hand detection from MediaPipe, including
    the landmarks, which hand it is, and detection confidence.
    """

    landmarks: HandLandmarks = Field(description="21 hand landmarks")
    handedness: Handedness = Field(description="Left or Right hand")
    confidence: Annotated[
        float, Field(ge=0.0, le=1.0, description="Detection confidence score")
    ]


# =============================================================================
# Recording Types
# =============================================================================


class RecordingFrame(BaseModel):
    """A single frame from a sign recording.

    Captures the hand landmark state at a specific moment in time.
    Either or both hands may be detected in a frame.
    """

    timestamp_ms: Annotated[
        int, Field(ge=0, description="Milliseconds since recording start")
    ]
    left_hand: Optional[DetectedHand] = Field(
        default=None, description="Left hand if detected"
    )
    right_hand: Optional[DetectedHand] = Field(
        default=None, description="Right hand if detected"
    )
    frame_confidence: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Overall frame detection confidence"),
    ] = 1.0

    @model_validator(mode="after")
    def validate_at_least_one_hand(self) -> Self:
        """Warn if no hands detected (but don't fail - may be intentional gap)."""
        # We allow frames with no hands for gaps in signing
        return self


class Recording(BaseModel):
    """A complete recording of a single sign.

    Contains all frames captured during one sign performance,
    with metadata about what sign was being performed.
    """

    frames: list[RecordingFrame] = Field(
        description="Sequence of recorded frames", min_length=1
    )
    sign_id: str = Field(description="ID of the sign being performed")
    start_time: datetime = Field(description="When recording started")
    end_time: datetime = Field(description="When recording ended")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional recording metadata"
    )

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        """Ensure end_time is after start_time."""
        if self.end_time < self.start_time:
            raise ValueError("end_time must be after start_time")
        return self


class RecordingSession(BaseModel):
    """A session containing multiple sign recordings.

    Groups recordings made during one capture session,
    typically by a single person at one sitting.
    """

    recordings: list[Recording] = Field(description="List of recordings in session")
    session_id: str = Field(description="Unique session identifier")
    created_at: datetime = Field(description="When session was created")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Session metadata (performer info, etc.)"
    )


# =============================================================================
# Sign Types
# =============================================================================

SignCategory = Literal[
    "alphabet",
    "numbers",
    "greetings",
    "questions",
    "common_phrases",
    "actions",
    "objects",
    "people",
    "places",
    "time",
    "colors",
    "emotions",
    "food_drink",
    "animals",
    "weather",
    "other",
]
"""Categories for organizing BSL signs."""


class SignDefinition(BaseModel):
    """Definition and metadata for a BSL sign.

    Contains the linguistic and categorical information about a sign,
    separate from the actual landmark data.
    """

    id: str = Field(description="Unique sign identifier")
    name: str = Field(description="English name/gloss for the sign")
    category: SignCategory = Field(description="Sign category")
    difficulty: Annotated[
        int, Field(ge=1, le=5, description="Difficulty level 1-5")
    ] = 3
    description: str = Field(default="", description="Description of how to perform")
    two_handed: bool = Field(
        default=False, description="Whether sign requires both hands"
    )
    regional_variants: list[str] = Field(
        default_factory=list, description="Regional variant names"
    )


class FingerAngles(BaseModel):
    """Joint angles for a single finger.

    Contains the angles at each joint of one finger,
    measured in degrees.
    """

    mcp: Annotated[
        float, Field(ge=-180.0, le=180.0, description="Metacarpophalangeal joint angle")
    ]
    pip: Annotated[
        float, Field(ge=-180.0, le=180.0, description="Proximal interphalangeal angle")
    ]
    dip: Annotated[
        float, Field(ge=-180.0, le=180.0, description="Distal interphalangeal angle")
    ]


class JointAngles(BaseModel):
    """All joint angles for one hand.

    Contains angles for each finger plus wrist orientation.
    Thumb has a slightly different joint structure.
    """

    thumb: FingerAngles = Field(description="Thumb joint angles (CMC/MCP/IP)")
    index: FingerAngles = Field(description="Index finger joint angles")
    middle: FingerAngles = Field(description="Middle finger joint angles")
    ring: FingerAngles = Field(description="Ring finger joint angles")
    pinky: FingerAngles = Field(description="Pinky finger joint angles")
    wrist_rotation: Annotated[
        float, Field(ge=-180.0, le=180.0, description="Wrist rotation angle")
    ] = 0.0
    wrist_flexion: Annotated[
        float, Field(ge=-180.0, le=180.0, description="Wrist flexion angle")
    ] = 0.0


class SignPose(BaseModel):
    """A canonical pose for a sign with normalized data.

    Represents the "ideal" hand position(s) for a sign,
    with normalized landmarks and computed joint angles.
    """

    left_hand_landmarks: Optional[list[Point3D]] = Field(
        default=None, description="Normalized left hand landmarks (21 points)"
    )
    right_hand_landmarks: Optional[list[Point3D]] = Field(
        default=None, description="Normalized right hand landmarks (21 points)"
    )
    left_hand_angles: Optional[JointAngles] = Field(
        default=None, description="Left hand joint angles"
    )
    right_hand_angles: Optional[JointAngles] = Field(
        default=None, description="Right hand joint angles"
    )
    tolerances: dict[str, float] = Field(
        default_factory=dict,
        description="Per-landmark tolerance values for matching",
    )

    @model_validator(mode="after")
    def validate_landmark_counts(self) -> Self:
        """Ensure landmark lists have exactly 21 points if present."""
        if self.left_hand_landmarks and len(self.left_hand_landmarks) != 21:
            raise ValueError("left_hand_landmarks must have exactly 21 points")
        if self.right_hand_landmarks and len(self.right_hand_landmarks) != 21:
            raise ValueError("right_hand_landmarks must have exactly 21 points")
        return self


class SignDictionaryEntry(BaseModel):
    """Complete dictionary entry for a BSL sign.

    Combines sign definition, canonical poses, and processing metadata
    into a single entry for the dictionary.
    """

    definition: SignDefinition = Field(description="Sign metadata and definition")
    poses: list[SignPose] = Field(
        description="Canonical poses for this sign (may have variants)",
        min_length=1,
    )
    sample_count: Annotated[
        int, Field(ge=1, description="Number of samples used to create poses")
    ]
    quality_score: Annotated[
        float, Field(ge=0.0, le=1.0, description="Overall quality score")
    ]
    created_at: datetime = Field(description="When entry was created")
    updated_at: datetime = Field(description="When entry was last updated")
    source: Literal["kaggle", "recorded", "merged"] = Field(
        description="Data source for this entry"
    )


class SignDictionary(BaseModel):
    """The complete BSL sign dictionary.

    Top-level container for all sign entries, with version
    information and metadata.
    """

    version: str = Field(description="Dictionary version (semver)")
    created_at: datetime = Field(description="When dictionary was created")
    updated_at: datetime = Field(description="When dictionary was last updated")
    entries: dict[str, SignDictionaryEntry] = Field(
        description="Sign entries keyed by sign ID"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional dictionary metadata"
    )

    @property
    def sign_count(self) -> int:
        """Return the number of signs in the dictionary."""
        return len(self.entries)


# =============================================================================
# Processing Types
# =============================================================================


class NormalizationParams(BaseModel):
    """Parameters used for landmark normalization.

    Stores the transformation applied to normalize hand landmarks
    to a canonical coordinate system.
    """

    scale_factor: Annotated[
        float, Field(gt=0.0, description="Scale factor applied to landmarks")
    ]
    rotation_matrix: list[list[float]] = Field(
        description="3x3 rotation matrix as nested list"
    )
    translation_vector: list[float] = Field(
        description="3D translation vector [x, y, z]"
    )

    @model_validator(mode="after")
    def validate_matrix_dimensions(self) -> Self:
        """Ensure rotation matrix is 3x3 and translation is 3D."""
        if len(self.rotation_matrix) != 3:
            raise ValueError("rotation_matrix must have 3 rows")
        for row in self.rotation_matrix:
            if len(row) != 3:
                raise ValueError("rotation_matrix must have 3 columns per row")
        if len(self.translation_vector) != 3:
            raise ValueError("translation_vector must have 3 elements")
        return self


class ProcessingResult(BaseModel):
    """Result from processing a set of recordings for one sign.

    Contains the canonical pose derived from samples,
    quality metrics, and any warnings.
    """

    canonical_pose: SignPose = Field(description="Computed canonical pose")
    quality_score: Annotated[
        float, Field(ge=0.0, le=1.0, description="Quality score for the result")
    ]
    sample_count: Annotated[
        int, Field(ge=1, description="Number of samples processed")
    ]
    warnings: list[str] = Field(
        default_factory=list, description="Processing warnings"
    )
    normalization_params: Optional[NormalizationParams] = Field(
        default=None, description="Normalization parameters used"
    )


class ValidationResult(BaseModel):
    """Result from validating sign data.

    Indicates whether data is valid and provides details
    about any errors or warnings found.
    """

    is_valid: bool = Field(description="Whether data passed validation")
    score: Annotated[
        float, Field(ge=0.0, le=1.0, description="Validation score")
    ]
    errors: list[str] = Field(
        default_factory=list, description="Validation errors (fail validation)"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Validation warnings (informational)"
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Ensure is_valid is False if there are errors."""
        if self.errors and self.is_valid:
            raise ValueError("is_valid must be False when errors are present")
        return self
