.PHONY: help setup test lint format clean train evaluate docker-up docker-down

PYTHON = python
PIP = pip

help:
	@echo "SpaceNetra Management Commands:"
	@echo "  setup        Install project dependencies and setup package"
	@echo "  test         Run unit and integration tests"
	@echo "  lint         Check code formatting and style"
	@echo "  format       Auto-format Python code using black and isort"
	@echo "  clean        Remove temporary files, cache, and build artifacts"
	@echo "  train        Run model training pipeline"
	@echo "  evaluate     Run evaluation protocol"

setup:
	$(PIP) install -e .
	$(PIP) install -r requirements.txt -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ tests/ models/
	black --check src/ tests/ models/
	isort --check-only src/ tests/ models/

format:
	black src/ tests/ models/
	isort src/ tests/ models/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py
