# Data Format Specification

This document describes all JSON schemas used in the BSL Data Extraction Tool.

## Table of Contents

1. [Recording Format](#recording-format)
2. [Processed Pose Format](#processed-pose-format)
3. [Dictionary Format](#dictionary-format)
4. [Configuration Format](#configuration-format)

---

## Recording Format

Raw recordings captured from the web tool or imported from Kaggle.

### Location
`data/raw-recordings/*.json`

### Schema

```json
{
  "sign_id": "string",
  "start_time": "ISO 8601 datetime",
  "end_time": "ISO 8601 datetime",
  "frames": [
    {
      "timestamp_ms": 0,
      "frame_confidence": 0.95,
      "left_hand": {
        "landmarks": [
          {"x": 0.0, "y": 0.0, "z": 0.0}
        ],
        "handedness": "Left",
        "confidence": 0.92
      },
      "right_hand": {
        "landmarks": [
          {"x": 0.0, "y": 0.0, "z": 0.0}
        ],
        "handedness": "Right",
        "confidence": 0.95
      }
    }
  ],
  "metadata": {
    "source": "recording-tool",
    "version": "1.0.0",
    "performer": "optional",
    "notes": "optional"
  }
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sign_id` | string | Yes | Unique identifier for the sign |
| `start_time` | datetime | Yes | When recording started |
| `end_time` | datetime | Yes | When recording ended |
| `frames` | array | Yes | List of captured frames |
| `frames[].timestamp_ms` | integer | Yes | Milliseconds since start |
| `frames[].frame_confidence` | float | No | Overall detection confidence (0-1) |
| `frames[].left_hand` | object | No | Left hand detection (if present) |
| `frames[].right_hand` | object | No | Right hand detection (if present) |
| `metadata` | object | No | Additional recording information |

### Landmark Array

Each hand contains exactly 21 landmarks in MediaPipe order:

| Index | Name | Description |
|-------|------|-------------|
| 0 | WRIST | Wrist base |
| 1 | THUMB_CMC | Thumb carpometacarpal |
| 2 | THUMB_MCP | Thumb metacarpophalangeal |
| 3 | THUMB_IP | Thumb interphalangeal |
| 4 | THUMB_TIP | Thumb tip |
| 5 | INDEX_FINGER_MCP | Index finger base |
| 6 | INDEX_FINGER_PIP | Index finger proximal |
| 7 | INDEX_FINGER_DIP | Index finger distal |
| 8 | INDEX_FINGER_TIP | Index finger tip |
| 9-12 | MIDDLE_FINGER_* | Middle finger joints |
| 13-16 | RING_FINGER_* | Ring finger joints |
| 17-20 | PINKY_* | Pinky finger joints |

### Example

```json
{
  "sign_id": "hello",
  "start_time": "2024-01-15T10:30:00.000Z",
  "end_time": "2024-01-15T10:30:03.000Z",
  "frames": [
    {
      "timestamp_ms": 0,
      "right_hand": {
        "landmarks": [
          {"x": 0.523, "y": 0.612, "z": -0.023},
          {"x": 0.498, "y": 0.589, "z": -0.041},
          {"x": 0.471, "y": 0.534, "z": -0.056}
        ],
        "handedness": "Right",
        "confidence": 0.94
      }
    }
  ],
  "metadata": {
    "source": "recording-tool",
    "version": "1.0.0"
  }
}
```

---

## Processed Pose Format

Normalized canonical poses output by the processing pipeline.

### Location
`data/processed/*.json`

### Schema

```json
{
  "sign_id": "string",
  "canonical_pose": {
    "left_hand_landmarks": [
      {"x": 0.0, "y": 0.0, "z": 0.0}
    ],
    "right_hand_landmarks": [
      {"x": 0.0, "y": 0.0, "z": 0.0}
    ],
    "left_hand_angles": {
      "thumb": {"mcp": 0.0, "pip": 0.0, "dip": 0.0},
      "index": {"mcp": 0.0, "pip": 0.0, "dip": 0.0},
      "middle": {"mcp": 0.0, "pip": 0.0, "dip": 0.0},
      "ring": {"mcp": 0.0, "pip": 0.0, "dip": 0.0},
      "pinky": {"mcp": 0.0, "pip": 0.0, "dip": 0.0},
      "wrist_rotation": 0.0,
      "wrist_flexion": 0.0
    },
    "right_hand_angles": {},
    "tolerances": {
      "wrist": 0.05,
      "thumb_tip": 0.08
    }
  },
  "quality_score": 0.92,
  "sample_count": 5,
  "warnings": ["string"],
  "normalization_params": {
    "scale_factor": 1.234,
    "rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]],
    "translation_vector": [0.0, 0.0, 0.0]
  }
}
```

### Normalized Coordinate Space

After normalization:
- **Origin**: Wrist position
- **Scale**: Palm width (index MCP to pinky MCP) = 1.0
- **Orientation**: Palm facing +Z, fingers pointing +Y

### Tolerances

Per-landmark tolerance values calculated from sample variance:

```
tolerance = std_dev × multiplier
```

Default multiplier: 2.0 (captures ~95% of natural variation)

---

## Dictionary Format

The final output dictionary for the learning app.

### Location
`data/output/bsl-dictionary.json`

### Schema

```json
{
  "version": "1.0.0",
  "created_at": "ISO 8601 datetime",
  "updated_at": "ISO 8601 datetime",
  "entries": {
    "sign_id": {
      "definition": {
        "id": "string",
        "name": "string",
        "category": "category_enum",
        "difficulty": 1,
        "description": "string",
        "two_handed": false,
        "regional_variants": ["string"]
      },
      "poses": [
        {
          "left_hand_landmarks": [],
          "right_hand_landmarks": [],
          "left_hand_angles": {},
          "right_hand_angles": {},
          "tolerances": {}
        }
      ],
      "sample_count": 5,
      "quality_score": 0.92,
      "created_at": "ISO 8601 datetime",
      "updated_at": "ISO 8601 datetime",
      "source": "kaggle|recorded|merged"
    }
  },
  "metadata": {
    "total_signs": 100,
    "signs_with_data": 85,
    "coverage_percent": 85.0,
    "average_quality": 0.88
  }
}
```

### Categories

Valid category values:

| Category | Description |
|----------|-------------|
| `alphabet` | Fingerspelling letters A-Z |
| `numbers` | Number signs 0-100+ |
| `greetings` | Hello, goodbye, etc. |
| `questions` | What, where, when, etc. |
| `common_phrases` | Thank you, please, etc. |
| `actions` | Verbs and activities |
| `objects` | Common nouns |
| `people` | Family, roles |
| `places` | Locations |
| `time` | Days, time expressions |
| `colors` | Color signs |
| `emotions` | Feelings |
| `food_drink` | Food and beverages |
| `animals` | Animal signs |
| `weather` | Weather expressions |
| `other` | Uncategorized |

### Difficulty Levels

| Level | Description |
|-------|-------------|
| 1 | Very easy - simple hand shapes |
| 2 | Easy - clear movements |
| 3 | Medium - moderate complexity |
| 4 | Hard - complex movements |
| 5 | Very hard - requires practice |

### Source Values

| Source | Description |
|--------|-------------|
| `kaggle` | Imported from Kaggle dataset |
| `recorded` | Captured with recording tool |
| `merged` | Combined from multiple sources |

### Complete Example

```json
{
  "version": "1.0.0",
  "created_at": "2024-01-15T10:30:00.000Z",
  "updated_at": "2024-01-15T10:30:00.000Z",
  "entries": {
    "hello": {
      "definition": {
        "id": "hello",
        "name": "Hello",
        "category": "greetings",
        "difficulty": 1,
        "description": "Wave hand with fingers together, palm facing outward",
        "two_handed": false,
        "regional_variants": []
      },
      "poses": [
        {
          "right_hand_landmarks": [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": -0.12, "y": 0.08, "z": 0.02},
            {"x": -0.18, "y": 0.22, "z": 0.01},
            {"x": -0.21, "y": 0.35, "z": -0.01},
            {"x": -0.23, "y": 0.45, "z": -0.02}
          ],
          "right_hand_angles": {
            "thumb": {"mcp": 15.2, "pip": 8.5, "dip": 5.1},
            "index": {"mcp": 5.0, "pip": 3.2, "dip": 2.1},
            "middle": {"mcp": 4.8, "pip": 3.0, "dip": 2.0},
            "ring": {"mcp": 5.1, "pip": 3.5, "dip": 2.3},
            "pinky": {"mcp": 6.2, "pip": 4.1, "dip": 2.8},
            "wrist_rotation": 0.0,
            "wrist_flexion": 5.5
          },
          "tolerances": {
            "wrist": 0.05,
            "thumb_cmc": 0.06,
            "thumb_tip": 0.08,
            "index_finger_tip": 0.07
          }
        }
      ],
      "sample_count": 8,
      "quality_score": 0.94,
      "created_at": "2024-01-15T10:30:00.000Z",
      "updated_at": "2024-01-15T10:30:00.000Z",
      "source": "recorded"
    }
  },
  "metadata": {
    "total_signs": 150,
    "signs_with_data": 127,
    "coverage_percent": 84.7,
    "average_quality": 0.89
  }
}
```

---

## Configuration Format

Tool configuration file.

### Location
`bsl-tool.config.json` (project root)

### Schema

```json
{
  "data_dir": "data",
  "kaggle_dir": "data/kaggle-bsl",
  "recordings_dir": "data/raw-recordings",
  "processed_dir": "data/processed",
  "output_dir": "data/output",
  "dictionary_file": "bsl-dictionary.json",
  "min_confidence": 0.5,
  "outlier_method": "zscore",
  "averaging_method": "trimmed",
  "tolerance_multiplier": 2.0
}
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `data_dir` | string | `data` | Root data directory |
| `kaggle_dir` | string | `data/kaggle-bsl` | Kaggle source directory |
| `recordings_dir` | string | `data/raw-recordings` | Raw recordings directory |
| `processed_dir` | string | `data/processed` | Processed poses directory |
| `output_dir` | string | `data/output` | Output directory |
| `dictionary_file` | string | `bsl-dictionary.json` | Dictionary filename |
| `min_confidence` | float | `0.5` | Minimum detection confidence |
| `outlier_method` | string | `zscore` | Outlier detection (`zscore` or `iqr`) |
| `averaging_method` | string | `trimmed` | Averaging method (`mean` or `trimmed`) |
| `tolerance_multiplier` | float | `2.0` | Tolerance calculation multiplier |

---

## TypeScript Types

Generated TypeScript definitions for the dictionary.

### Location
`data/output/bsl-dictionary.d.ts`

### Generated Types

```typescript
export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface FingerAngles {
  mcp: number;
  pip: number;
  dip: number;
}

export interface JointAngles {
  thumb: FingerAngles;
  index: FingerAngles;
  middle: FingerAngles;
  ring: FingerAngles;
  pinky: FingerAngles;
  wrist_rotation: number;
  wrist_flexion: number;
}

export interface SignPose {
  left_hand_landmarks?: Point3D[];
  right_hand_landmarks?: Point3D[];
  left_hand_angles?: JointAngles;
  right_hand_angles?: JointAngles;
  tolerances: Record<string, number>;
}

export interface SignDefinition {
  id: string;
  name: string;
  category: SignCategory;
  difficulty: 1 | 2 | 3 | 4 | 5;
  description: string;
  two_handed: boolean;
  regional_variants: string[];
}

export interface SignDictionaryEntry {
  definition: SignDefinition;
  poses: SignPose[];
  sample_count: number;
  quality_score: number;
  created_at: string;
  updated_at: string;
  source: 'kaggle' | 'recorded' | 'merged';
}

export interface SignDictionary {
  version: string;
  created_at: string;
  updated_at: string;
  entries: Record<string, SignDictionaryEntry>;
  metadata: DictionaryMetadata;
}

export type SignCategory =
  | 'alphabet'
  | 'numbers'
  | 'greetings'
  | 'questions'
  | 'common_phrases'
  | 'actions'
  | 'objects'
  | 'people'
  | 'places'
  | 'time'
  | 'colors'
  | 'emotions'
  | 'food_drink'
  | 'animals'
  | 'weather'
  | 'other';
```
