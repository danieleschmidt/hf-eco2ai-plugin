# Development Guide

This guide covers development setup, architecture, and best practices for contributing to HF Eco2AI Plugin.

## Quick Setup

```bash
git clone https://github.com/terragonlabs/hf-eco2ai-plugin.git
cd hf-eco2ai-plugin
make dev-setup
```

## Architecture Overview

```
src/hf_eco2ai/
├── __init__.py          # Public API exports
├── callback.py          # Main Eco2AICallback class
├── config.py            # Configuration management
├── metrics.py           # Metric collection and calculation
├── exporters/           # Prometheus, MLflow, etc.
├── lightning/           # PyTorch Lightning integration
└── utils/               # Utilities and helpers
```

## Core Components

### Eco2AICallback
- Integrates with Hugging Face Trainer lifecycle
- Starts/stops energy tracking on training events
- Collects metrics per epoch/step
- Exports to various monitoring systems

### CarbonConfig
- Centralized configuration management
- Validation and defaults
- Regional carbon intensity data
- Export and monitoring settings

### Metrics Collection
- CPU/GPU power monitoring via `pynvml` and `psutil`
- Grid carbon intensity from APIs
- Training efficiency calculations
- Real-time metric aggregation

## Development Workflow

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Running Tests
```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests (requires GPU)
pytest tests/integration/ -m gpu

# With coverage
pytest --cov=hf_eco2ai --cov-report=html
```

### 3. Code Quality
```bash
# Format code
black src tests
ruff check --fix src tests

# Type checking
mypy src

# Security scan
bandit -r src

# Dependency check
safety check
```

### 4. Documentation
```bash
# Build docs
cd docs && make html

# Serve locally
python -m http.server 8000 -d docs/_build/html
```

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Mock external dependencies (Eco2AI, NVML)
- Test callback lifecycle methods
- Validate configuration handling
- Test metric calculations

### Integration Tests (`tests/integration/`)
- Real GPU monitoring (requires CUDA)
- End-to-end training scenarios
- Prometheus export validation
- File I/O operations

### Performance Tests (`tests/performance/`)
- Callback overhead measurement
- Memory usage monitoring
- Large dataset scenarios

## Debugging

### Energy Tracking Issues
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test GPU access
import pynvml
pynvml.nvmlInit()
print(f"GPUs: {pynvml.nvmlDeviceGetCount()}")
```

### Callback Integration
```python
# Minimal test case
from hf_eco2ai import Eco2AICallback

callback = Eco2AICallback()
# Check callback state after trainer events
```

## Release Process

### Version Management
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `pyproject.toml` and `__init__.py`
- Tag releases: `git tag v0.1.0`

### Pre-release Checklist
- [ ] All tests pass: `pytest`
- [ ] Code quality checks: `make lint`
- [ ] Documentation builds: `make docs`
- [ ] Security scan clean: `bandit -r src`
- [ ] Changelog updated
- [ ] Version bumped

### Release Steps
```bash
# Build distribution
make build

# Test on TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*

# Create GitHub release
gh release create v0.1.0 --notes "Release notes"
```

## Performance Considerations

### Callback Overhead
- Energy monitoring adds ~1-5ms per step
- GPU queries are cached to reduce NVML calls
- Async export for Prometheus metrics

### Memory Usage
- Bounded metric storage (configurable history)
- Efficient numpy arrays for time series
- Optional metric compression

### Scaling
- Multi-GPU support via device enumeration
- Distributed training compatibility
- Cloud instance auto-detection

## Contributing Guidelines

### Code Style
- Follow PEP 8 via `black` formatting
- Use type hints for all public APIs
- Descriptive variable names
- Minimal external dependencies

### Documentation
- Docstrings for all public functions
- Type information in docstrings
- Usage examples in docstrings
- Update README for new features

### Testing
- Test coverage >90%
- Both positive and negative test cases
- Mock external dependencies
- Include performance regression tests

### Pull Request Process
1. Fork and create feature branch
2. Implement changes with tests
3. Ensure all checks pass
4. Update documentation
5. Submit PR with detailed description

## Troubleshooting

### Common Issues

**Import Error: No module named 'eco2ai'**
```bash
pip install eco2ai>=2.0.0
```

**NVML Error: GPU not accessible**
```bash
# Check GPU driver
nvidia-smi

# Check permissions
sudo usermod -a -G video $USER
```

**Prometheus Export Not Working**
```python
# Verify port availability
netstat -tlnp | grep 9091

# Check firewall rules
sudo ufw status
```

### Debug Environment
```bash
# Python environment info
python --version
pip list | grep -E "(eco2ai|transformers|torch)"

# GPU information
nvidia-smi
lspci | grep -i nvidia

# System resources
htop
df -h
```