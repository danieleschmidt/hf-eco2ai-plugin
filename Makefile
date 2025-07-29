# HF Eco2AI Plugin Makefile
# Provides convenient development commands

.PHONY: help install install-dev clean test test-all lint format type-check security docs build release

# Default target
help: ## Show this help message
	@echo "HF Eco2AI Plugin Development Commands"
	@echo "====================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install: ## Install package in production mode
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

# Cleaning
clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Testing
test: ## Run unit tests
	pytest tests/ -v

test-unit: ## Run only unit tests
	pytest tests/ -m "not integration and not slow" -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-performance: ## Run performance tests
	pytest tests/performance/ --benchmark-only -v

test-all: ## Run all tests including slow ones
	pytest tests/ -v --slow

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=hf_eco2ai --cov-report=html --cov-report=term-missing

# Code Quality
lint: ## Run linting checks
	ruff check src tests
	black --check src tests

format: ## Format code with black and ruff
	black src tests
	ruff check --fix src tests

type-check: ## Run type checking with mypy
	mypy src

quality: lint type-check ## Run all code quality checks

# Security
security: ## Run security scans
	bandit -r src/
	safety check
	pip-audit

# Documentation
docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs && python -m http.server 8000 --directory _build/html

# Building
build: clean ## Build package
	python -m build

build-docker: ## Build Docker image
	docker build -t hf-eco2ai-plugin:latest .

# Development Environment
dev-setup: install-dev ## Complete development setup
	@echo "✅ Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

# Validation
validate: quality test security ## Run all validation checks
	@echo "✅ All validation checks passed!"

# Utilities
version: ## Show current version
	@python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

info: ## Show development environment info
	@echo "Python: $$(python --version)"
	@echo "Project Version: $$(make version)"