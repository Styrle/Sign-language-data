# Processing Parameters

This document explains all configurable parameters in the BSL data processing pipeline.

## Overview

Processing parameters control how raw landmark data is transformed into canonical poses. The right parameters depend on your data quality and use case.

## Configuration File

Parameters can be set in `bsl-tool.config.json`:

```json
{
  "min_confidence": 0.5,
  "outlier_method": "zscore",
  "averaging_method": "trimmed",
  "tolerance_multiplier": 2.0,
  "normalization_anchor": "wrist",
  "normalization_scale": "palm_width"
}
```

Or passed via CLI:

```bash
bsl-tool process batch --min-confidence 0.6
```

---

## Confidence Filtering

### `min_confidence`

**Type**: float (0.0 - 1.0)
**Default**: 0.5
**Description**: Minimum detection confidence to include a frame

MediaPipe returns a confidence score (0-1) for each hand detection. Low-confidence detections often have inaccurate landmarks.

| Value | Effect |
|-------|--------|
| 0.3 | Include most detections (more data, lower quality) |
| 0.5 | Balanced (default) |
| 0.7 | High quality only (less data) |
| 0.9 | Very strict (may lose valid frames) |

**Recommendation**: Start with 0.5, increase if you see noisy data.

---

## Outlier Removal

### `outlier_method`

**Type**: string
**Default**: `zscore`
**Options**: `zscore`, `iqr`, `none`

Outliers are landmark positions that deviate significantly from the group. These can result from tracking errors or unusual hand positions.

#### Z-Score Method (`zscore`)

Removes points more than N standard deviations from the mean.

```
outlier if: |x - mean| > threshold × std_dev
```

**Parameters**:
- `zscore_threshold`: Number of standard deviations (default: 2.5)

**Best for**: Normally distributed data, small datasets

#### IQR Method (`iqr`)

Uses interquartile range to identify outliers.

```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
outlier if: x < Q1 - 1.5×IQR or x > Q3 + 1.5×IQR
```

**Parameters**:
- `iqr_multiplier`: IQR multiplier (default: 1.5)

**Best for**: Non-normal distributions, data with extreme values

#### No Outlier Removal (`none`)

Skip outlier detection entirely.

**Best for**: Very clean data, when you want to preserve all samples

---

## Averaging Method

### `averaging_method`

**Type**: string
**Default**: `trimmed`
**Options**: `mean`, `trimmed`, `median`

How to combine multiple samples into a canonical pose.

#### Mean (`mean`)

Simple arithmetic mean of all samples.

```
canonical = sum(samples) / count(samples)
```

**Pros**: Simple, uses all data
**Cons**: Sensitive to outliers

#### Trimmed Mean (`trimmed`)

Mean after removing the top and bottom N% of values.

```
canonical = mean(samples[trim:-trim])
```

**Parameters**:
- `trim_percent`: Percentage to trim from each end (default: 10%)

**Pros**: Robust to outliers
**Cons**: Discards some valid data

#### Median (`median`)

Middle value when samples are sorted.

**Pros**: Very robust to outliers
**Cons**: Ignores distribution shape

---

## Tolerance Calculation

### `tolerance_multiplier`

**Type**: float
**Default**: 2.0
**Description**: Multiplier for standard deviation in tolerance calculation

Tolerances define how much variation is acceptable when matching a user's pose to the canonical pose.

```
tolerance[landmark] = std_dev[landmark] × tolerance_multiplier
```

| Value | Effect |
|-------|--------|
| 1.0 | Strict matching (~68% of natural variation) |
| 2.0 | Standard matching (~95% of natural variation) |
| 3.0 | Lenient matching (~99% of natural variation) |

**Recommendation**:
- 2.0 for learning app (forgiving)
- 1.5 for assessment (stricter)
- 2.5 for accessibility (very forgiving)

### `min_tolerance`

**Type**: float
**Default**: 0.02
**Description**: Minimum tolerance value (prevents zero tolerance)

Ensures all landmarks have some tolerance, even if samples are very consistent.

### `max_tolerance`

**Type**: float
**Default**: 0.2
**Description**: Maximum tolerance value (caps extreme variation)

Prevents excessive tolerance from highly variable samples.

---

## Normalization Parameters

### `normalization_anchor`

**Type**: string
**Default**: `wrist`
**Options**: `wrist`, `centroid`, `palm_center`

The point used as the origin for normalized coordinates.

#### Wrist (`wrist`)

Wrist landmark becomes origin (0, 0, 0).

**Pros**: Anatomically meaningful, stable
**Cons**: Palm position affects coordinates

#### Centroid (`centroid`)

Center of mass of all landmarks becomes origin.

**Pros**: Balanced representation
**Cons**: Changes with finger position

#### Palm Center (`palm_center`)

Center of palm landmarks (wrist, MCP joints) becomes origin.

**Pros**: More stable than full centroid
**Cons**: Slightly more complex

### `normalization_scale`

**Type**: string
**Default**: `palm_width`
**Options**: `palm_width`, `bounding_box`, `fixed`

How to normalize hand size.

#### Palm Width (`palm_width`)

Scale so distance from index MCP to pinky MCP = 1.0.

```
scale_factor = palm_width / 1.0
```

**Pros**: Consistent for different hand sizes, anatomically meaningful
**Cons**: Finger spread affects result

#### Bounding Box (`bounding_box`)

Scale so bounding box diagonal = 1.0.

```
scale_factor = bbox_diagonal / 1.0
```

**Pros**: Independent of pose
**Cons**: Finger extension affects result

#### Fixed (`fixed`)

No scaling applied.

**Pros**: Preserves absolute size
**Cons**: Different hand sizes won't match

### `apply_rotation`

**Type**: boolean
**Default**: true
**Description**: Whether to rotate landmarks to canonical orientation

When true:
- Palm normal aligned to +Z axis
- Finger direction aligned to +Y axis

When false:
- Landmarks keep their original orientation
- Useful for preserving specific hand angles

---

## Angle Calculation Parameters

### `angle_format`

**Type**: string
**Default**: `degrees`
**Options**: `degrees`, `radians`

Unit for joint angle output.

### `threejs_euler_order`

**Type**: string
**Default**: `XYZ`
**Options**: `XYZ`, `XZY`, `YXZ`, `YZX`, `ZXY`, `ZYX`

Euler angle order for Three.js bone rotations.

### `validate_anatomical_limits`

**Type**: boolean
**Default**: true
**Description**: Check angles against anatomical constraints

Flags angles that exceed normal human range of motion:

| Joint | Min | Max |
|-------|-----|-----|
| Finger MCP | 0° | 100° |
| Finger PIP | 0° | 110° |
| Finger DIP | 0° | 80° |
| Thumb CMC | 0° | 90° |

---

## Quality Scoring Parameters

### Quality Component Weights

```json
{
  "quality_weights": {
    "completeness": 0.30,
    "stability": 0.25,
    "anatomical": 0.25,
    "coherence": 0.20
  }
}
```

### Quality Thresholds

```json
{
  "quality_thresholds": {
    "excellent": 0.90,
    "good": 0.75,
    "acceptable": 0.60,
    "poor": 0.40
  }
}
```

---

## Processing Pipeline Order

The order of operations matters:

1. **Confidence filtering** - Remove low-confidence frames
2. **Normalization** - Transform to canonical space
3. **Outlier removal** - Remove statistical outliers
4. **Averaging** - Combine samples
5. **Tolerance calculation** - Compute variation bounds
6. **Angle calculation** - Derive joint angles
7. **Quality scoring** - Assess result quality

---

## Performance Parameters

### `batch_size`

**Type**: integer
**Default**: 100
**Description**: Number of recordings to process per batch

Affects memory usage and progress reporting.

### `parallel_workers`

**Type**: integer
**Default**: 4
**Description**: Number of parallel processing threads

Higher values speed up batch processing on multi-core systems.

---

## Recommended Presets

### High Quality (Production)

```json
{
  "min_confidence": 0.6,
  "outlier_method": "zscore",
  "zscore_threshold": 2.0,
  "averaging_method": "trimmed",
  "trim_percent": 15,
  "tolerance_multiplier": 1.8,
  "validate_anatomical_limits": true
}
```

### Fast Processing (Development)

```json
{
  "min_confidence": 0.4,
  "outlier_method": "none",
  "averaging_method": "mean",
  "tolerance_multiplier": 2.5,
  "validate_anatomical_limits": false
}
```

### Strict Matching (Assessment)

```json
{
  "min_confidence": 0.7,
  "outlier_method": "iqr",
  "averaging_method": "median",
  "tolerance_multiplier": 1.5,
  "validate_anatomical_limits": true
}
```

### Accessibility (Forgiving)

```json
{
  "min_confidence": 0.3,
  "outlier_method": "zscore",
  "zscore_threshold": 3.0,
  "averaging_method": "trimmed",
  "tolerance_multiplier": 3.0
}
```
