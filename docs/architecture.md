# Architecture

This document describes the system design and data flow of the BSL Data Extraction Tool.

## System Overview

The tool is designed as a modular pipeline that transforms raw hand landmark data into a standardized dictionary format suitable for the BSL Learning App.

```
┌──────────────────────────────────────────────────────────────────┐
│                        Data Sources                               │
├─────────────────────────────┬────────────────────────────────────┤
│      Kaggle Dataset         │         Recording Tool             │
│    (CSV/Parquet files)      │      (Web-based capture)           │
└─────────────────────────────┴────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Import Layer                               │
│  ┌───────────────────┐      ┌────────────────────────────────┐   │
│  │  import_kaggle.py │      │  JSON file import (manual)     │   │
│  │  - Parse CSV      │      │  - Validate structure          │   │
│  │  - Extract signs  │      │  - Copy to raw-recordings/     │   │
│  └───────────────────┘      └────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Processing Layer                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  normalize.py                                                │ │
│  │  - Translate to origin (wrist or centroid)                   │ │
│  │  - Scale to unit size (palm width or bbox)                   │ │
│  │  - Rotate to canonical orientation                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  process_recordings.py                                       │ │
│  │  - Filter by confidence                                      │ │
│  │  - Remove outliers (z-score or IQR)                          │ │
│  │  - Average samples (mean or trimmed mean)                    │ │
│  │  - Calculate per-landmark tolerances                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  angles.py                                                   │ │
│  │  - Calculate joint angles from landmarks                     │ │
│  │  - Convert to Three.js Euler rotations                       │ │
│  │  - Validate against anatomical limits                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Validation Layer                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  validation.py                                               │ │
│  │  - Check landmark completeness (21 points)                   │ │
│  │  - Verify coordinate ranges                                  │ │
│  │  - Check anatomical plausibility                             │ │
│  │  - Calculate quality scores                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Generation Layer                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  generate_dictionary.py                                      │ │
│  │  - Combine processed poses                                   │ │
│  │  - Add sign definitions and metadata                         │ │
│  │  - Export JSON (pretty and minified)                         │ │
│  │  - Generate TypeScript types                                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Output                                    │
│  ┌────────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ bsl-dictionary.json│  │ bsl-dictionary.  │  │ bsl-dictionary│  │
│  │  (pretty, 2-space) │  │    min.json      │  │     .d.ts    │  │
│  └────────────────────┘  └──────────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### Core Library (`src/`)

| Module | Purpose |
|--------|---------|
| `types.py` | Pydantic models for all data structures |
| `landmarks.py` | MediaPipe landmark constants and mappings |
| `normalize.py` | Landmark normalization transformations |
| `angles.py` | Joint angle calculations |
| `math_utils.py` | Vector/matrix math utilities |
| `validation.py` | Data validation functions |
| `sign_definitions.py` | BSL sign metadata definitions |

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `cli.py` | Unified CLI entry point |
| `import_kaggle.py` | Kaggle dataset importer |
| `process_recordings.py` | Recording processor |
| `generate_dictionary.py` | Dictionary generator |

### Recording Tool (`recording-tool/`)

| File | Purpose |
|------|---------|
| `index.html` | Main UI layout |
| `app.js` | Camera, detection, recording logic |
| `styles.css` | UI styling |
| `sign-definitions.js` | Sign list for dropdown |

## Data Flow Details

### 1. Import Stage

**Kaggle Import:**
```
train.csv → parse rows → extract landmarks → RecordingFrame → save JSON
```

**Recording Import:**
```
downloaded.json → validate structure → copy to raw-recordings/
```

### 2. Normalization Pipeline

```
Raw landmarks (image coordinates)
        │
        ▼
┌───────────────────┐
│    Translation    │  Subtract wrist position (or centroid)
│   Wrist → Origin  │  Result: hand centered at origin
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│      Scaling      │  Divide by palm width (index MCP to pinky MCP)
│  Palm width = 1.0 │  Result: consistent hand size
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│     Rotation      │  Align palm normal to Z-axis
│  Canonical orient │  Align fingers to Y-axis
└─────────┬─────────┘
          │
          ▼
Normalized landmarks (canonical space)
```

### 3. Processing Pipeline

```
Multiple recordings for sign X
        │
        ▼
┌───────────────────┐
│ Confidence Filter │  Remove frames with confidence < threshold
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Outlier Removal  │  Z-score or IQR method
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    Averaging      │  Mean or trimmed mean across samples
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Tolerance Calc    │  Standard deviation × multiplier
└─────────┬─────────┘
          │
          ▼
Canonical pose + tolerances
```

### 4. Angle Calculation

```
Normalized landmarks
        │
        ▼
┌───────────────────┐
│  Joint Vectors    │  Calculate bone vectors between landmarks
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Angle Calc       │  Dot product → angle between vectors
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Euler Convert    │  Convert to Three.js rotation format
└─────────┬─────────┘
          │
          ▼
JointAngles (degrees) + ThreeJS rotations (radians)
```

## Coordinate Systems

### MediaPipe (Input)
- Origin: Top-left of image
- X: Left to right (0-1)
- Y: Top to bottom (0-1)
- Z: Depth from camera (relative)

### Normalized (Internal)
- Origin: Wrist position
- X: Right (palm facing camera)
- Y: Up (towards fingers)
- Z: Forward (away from palm)
- Scale: Palm width = 1.0

### Three.js (Output)
- Right-handed coordinate system
- Y-up convention
- Euler angles in XYZ order

## Quality Scoring

The quality score (0.0-1.0) is calculated from:

| Component | Weight | Description |
|-----------|--------|-------------|
| Completeness | 30% | All 21 landmarks present and valid |
| Stability | 25% | Low frame-to-frame jitter |
| Anatomical | 25% | Reasonable bone ratios and angles |
| Coherence | 20% | Landmarks form coherent hand shape |

## Extension Points

### Adding New Data Sources
1. Create importer in `scripts/`
2. Convert to `RecordingFrame` format
3. Save to `raw-recordings/`

### Adding New Normalization Methods
1. Add function to `src/normalize.py`
2. Update `NormalizationResult` if needed
3. Add tests to `tests/test_normalize.py`

### Adding New Signs
1. Add definition to `src/sign_definitions.py`
2. Update `recording-tool/sign-definitions.js`
3. Record samples using the recording tool

## Performance Considerations

- **Batch processing**: Process signs in groups to reduce I/O
- **Caching**: Normalization results cached in processed/
- **Parallel processing**: Independent signs can be processed in parallel
- **Minified output**: Use `.min.json` for production to reduce size
