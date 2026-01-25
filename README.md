# BSL Data Extraction Tool

Internal tooling for extracting, processing, and normalizing British Sign Language (BSL) landmark data from the Kaggle BSL dataset and custom recordings.

## Overview

This project provides:
- **Python processing scripts** for landmark normalization, angle calculation, and validation
- **A minimal web recording tool** for capturing hand landmarks via webcam
- **A unified dictionary format** for BSL signs with normalized landmark data

## Project Structure

```
bsl-data-extraction/
├── /data                    # Data files (mostly gitignored)
│   ├── /kaggle-bsl          # Downloaded Kaggle BSL dataset
│   ├── /raw-recordings      # JSON exports from recording tool
│   ├── /processed           # Intermediate processing results
│   └── /output              # Final dictionary (committed)
│       └── bsl-dictionary.json
│
├── /recording-tool          # Web app for capturing landmarks
├── /scripts                 # Python processing scripts
├── /src                     # Shared Python modules
├── /tests                   # Python tests
└── /notebooks               # Jupyter exploration notebooks
```

## Setup

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
make install-dev

# Or using pip directly
pip install -e ".[dev,notebooks]"
```

## Usage

### Processing Pipeline

```bash
# Run the full processing pipeline
make process

# Or run individual scripts
python -m scripts.import_kaggle
python -m scripts.normalize
python -m scripts.calculate_angles
python -m scripts.validate
python -m scripts.generate_dictionary
```

### Development

```bash
# Run tests
make test

# Run linter
make lint

# Format code
make format

# Clean cache files
make clean
```

## Data Sources

1. **Kaggle BSL Dataset**: Pre-existing landmark data from Kaggle
2. **Custom Recordings**: Captured via the web recording tool

## Output Format

The final output is `data/output/bsl-dictionary.json`, containing normalized BSL signs with:
- Normalized landmark positions
- Joint angles
- Metadata and validation scores

## License

See [LICENSE](LICENSE) for details.
