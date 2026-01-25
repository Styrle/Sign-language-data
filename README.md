# BSL Data Extraction Tool

Internal tool for creating the BSL sign dictionary used by the BSL Learning App.

## Overview

This tool provides a complete pipeline for extracting, processing, and generating British Sign Language (BSL) dictionary data from various sources. It handles:

- Importing hand landmark data from Kaggle datasets
- Capturing new sign recordings via a web-based recording tool
- Normalizing and processing landmark data
- Calculating joint angles for 3D visualization
- Generating dictionary JSON for the learning app

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e .

# Initialize project structure
bsl-tool init

# Check pipeline status
bsl-tool pipeline status

# Run full pipeline
bsl-tool pipeline run
```

## Project Structure

```
Sign-language-data/
├── src/                          # Core library modules
│   ├── types.py                  # Pydantic data models
│   ├── landmarks.py              # Landmark constants and mappings
│   ├── normalize.py              # Landmark normalization
│   ├── angles.py                 # Joint angle calculations
│   ├── math_utils.py             # Math utilities (vectors, matrices)
│   ├── validation.py             # Data validation functions
│   └── sign_definitions.py       # BSL sign definitions
│
├── scripts/                      # Pipeline scripts
│   ├── cli.py                    # Unified CLI (bsl-tool)
│   ├── import_kaggle.py          # Kaggle dataset importer
│   ├── process_recordings.py     # Recording processor
│   └── generate_dictionary.py    # Dictionary generator
│
├── recording-tool/               # Web-based sign recorder
│   ├── index.html                # Main page
│   ├── app.js                    # Application logic
│   ├── styles.css                # Styling
│   └── sign-definitions.js       # Sign list for UI
│
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── 01_explore_kaggle_data.ipynb
│   ├── 02_normalization_experiments.ipynb
│   ├── 03_tolerance_analysis.ipynb
│   ├── 04_angle_calculations.ipynb
│   ├── 05_quality_metrics.ipynb
│   └── notebook_utils.py         # Shared visualization helpers
│
├── tests/                        # Test suite
│   ├── test_normalize.py
│   ├── test_angles.py
│   ├── test_validation.py
│   ├── test_process_recordings.py
│   └── test_generate_dictionary.py
│
├── data/                         # Data directory (created by init)
│   ├── kaggle-bsl/               # Kaggle source data
│   ├── raw-recordings/           # Recordings from web tool
│   ├── processed/                # Processed canonical poses
│   └── output/                   # Generated dictionary files
│
├── docs/                         # Documentation
│   ├── architecture.md
│   ├── data-format.md
│   ├── recording-guide.md
│   ├── processing-params.md
│   └── troubleshooting.md
│
├── pyproject.toml                # Project configuration
├── CHANGELOG.md                  # Version history
└── README.md                     # This file
```

## Data Flow

```
┌─────────────────┐     ┌──────────────────┐
│  Kaggle Dataset │     │  Recording Tool  │
│   (CSV/Parquet) │     │  (Web Interface) │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│              Import Stage               │
│   - Parse raw landmark data             │
│   - Convert to standard format          │
│   - Initial quality filtering           │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│            Processing Stage             │
│   - Normalize landmarks                 │
│   - Remove outliers                     │
│   - Average multiple samples            │
│   - Calculate tolerances                │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│          Angle Calculation              │
│   - Compute joint angles                │
│   - Convert to Three.js format          │
│   - Validate anatomical limits          │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│          Dictionary Generation          │
│   - Combine all sign data               │
│   - Generate JSON + TypeScript types    │
│   - Create minified version             │
└────────────────────┬────────────────────┘
                     │
                     ▼
              bsl-dictionary.json
```

## Commands

### Main Commands

```bash
# Show tool information and configuration
bsl-tool info

# Initialize project directory structure
bsl-tool init [--force]

# Show pipeline status
bsl-tool pipeline status

# Run complete pipeline
bsl-tool pipeline run [--skip-import] [--skip-process] [--skip-validate] [--version VERSION]
```

### Analyze Commands

```bash
# Analyze Kaggle dataset
bsl-tool analyze kaggle /path/to/kaggle-bsl/

# Analyze raw recordings
bsl-tool analyze recordings /path/to/recordings/ [--detailed]

# Analyze generated dictionary
bsl-tool analyze dictionary /path/to/bsl-dictionary.json
```

### Import Commands

```bash
# Import Kaggle CSV data
bsl-tool import kaggle /path/to/data.csv [--output DIR] [--limit N]

# Import JSON recording files
bsl-tool import json file1.json file2.json [--output DIR]
```

### Process Commands

```bash
# Process single sign from recordings
bsl-tool process single recording1.json recording2.json [--output FILE]

# Batch process all recordings in directory
bsl-tool process batch /path/to/recordings/ [--output DIR] [--pattern "*.json"]

# Process all data sources
bsl-tool process all [--kaggle-dir DIR] [--recordings-dir DIR] [--output DIR]
```

### Validate Commands

```bash
# Validate landmarks in a file
bsl-tool validate landmarks /path/to/file.json

# Validate a recording session
bsl-tool validate recording /path/to/recording.json

# Validate dictionary file
bsl-tool validate dictionary /path/to/bsl-dictionary.json [--strict]

# Validate all data
bsl-tool validate all [--data-dir DIR]
```

### Generate Commands

```bash
# Generate dictionary from processed poses
bsl-tool generate dictionary [--version VERSION] [--processed-dir DIR] [--output-dir DIR]

# Generate TypeScript types
bsl-tool generate types --output /path/to/types.d.ts [--input DICT_FILE]

# Generate coverage/quality report
bsl-tool generate report [--input DICT_FILE] [--output FILE] [--format text|json|markdown]
```

### Global Options

```bash
# Increase verbosity (-v, -vv, -vvv)
bsl-tool -v pipeline run

# Dry run mode (show what would happen)
bsl-tool --dry-run pipeline run

# Use custom config file
bsl-tool --config myconfig.json pipeline run
```

## Recording Tool

The web-based recording tool captures hand landmarks using MediaPipe:

1. Open `recording-tool/index.html` in a modern browser
2. Allow camera access when prompted
3. Select a sign from the dropdown
4. Click "Record" and perform the sign
5. Review and save or discard the recording

Recordings are downloaded as JSON files that can be imported into the pipeline.

See [docs/recording-guide.md](docs/recording-guide.md) for detailed instructions.

## Output Format

The generated dictionary (`bsl-dictionary.json`) contains:

```json
{
  "version": "1.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "entries": {
    "hello": {
      "definition": {
        "id": "hello",
        "name": "Hello",
        "category": "greetings",
        "difficulty": 1,
        "description": "Wave hand side to side",
        "two_handed": false
      },
      "poses": [{
        "right_hand_landmarks": [...],
        "right_hand_angles": {...},
        "tolerances": {...}
      }],
      "sample_count": 5,
      "quality_score": 0.92,
      "source": "recorded"
    }
  }
}
```

See [docs/data-format.md](docs/data-format.md) for complete schema documentation.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=scripts

# Run specific test file
pytest tests/test_normalize.py -v
```

### Code Quality

```bash
# Format code
ruff format src scripts tests

# Lint
ruff check src scripts tests

# Type checking
mypy src scripts
```

### Using Notebooks

```bash
# Install Jupyter
pip install jupyter

# Start notebook server
jupyter notebook notebooks/
```

The notebooks provide interactive exploration of:
- Kaggle dataset structure and quality
- Normalization parameter tuning
- Tolerance threshold analysis
- Joint angle verification
- Quality metric development

### Contributing

1. Create a feature branch
2. Make changes with tests
3. Run linting and tests
4. Submit for review (do not auto-commit)

## Configuration

Create `bsl-tool.config.json` in the project root:

```json
{
  "data_dir": "data",
  "kaggle_dir": "data/kaggle-bsl",
  "recordings_dir": "data/raw-recordings",
  "processed_dir": "data/processed",
  "output_dir": "data/output",
  "min_confidence": 0.5,
  "outlier_method": "zscore",
  "averaging_method": "trimmed"
}
```

## Dependencies

- Python 3.10+
- NumPy - Array operations
- SciPy - Statistical functions
- Pydantic - Data validation
- Click - CLI framework

Recording tool requires:
- Modern browser with WebGL
- Camera access
- TensorFlow.js (loaded from CDN)

## License

See [LICENSE](LICENSE) for details.
