.PHONY: help install install-dev test test-fast lint format type-check clean docs build upload

help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  test         Run all tests"
	@echo "  test-fast    Run tests excluding slow ones"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and ruff"
	@echo "  type-check   Run mypy type checking"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"
	@echo "  build        Build distribution packages"
	@echo "  upload       Upload to PyPI"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-fast:
	pytest -m "not slow"

lint:
	ruff check src tests
	black --check src tests
	mypy src

format:
	black src tests
	ruff check --fix src tests

type-check:
	mypy src

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."