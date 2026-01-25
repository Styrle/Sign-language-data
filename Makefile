.PHONY: install install-dev test lint format process clean help

# Default target
help:
	@echo "BSL Data Extraction Tool"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install package in editable mode"
	@echo "  make install-dev  Install with dev dependencies"
	@echo "  make test         Run pytest"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Run ruff formatter"
	@echo "  make process      Run full processing pipeline"
	@echo "  make clean        Remove cache files"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebooks]"

# Testing
test:
	pytest

# Linting
lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

# Processing pipeline
process:
	python -m scripts.cli process-all

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
