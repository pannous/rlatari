.PHONY: help install install-dev format lint type-check test clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install package dependencies"
	@echo "  make install-dev  - Install package with dev dependencies"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with ruff"
	@echo "  make type-check   - Type check with mypy"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Remove build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

format:
	black .
	ruff check --fix .

lint:
	ruff check .

type-check:
	mypy algorithms/ utils/ train.py

test:
	pytest

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
