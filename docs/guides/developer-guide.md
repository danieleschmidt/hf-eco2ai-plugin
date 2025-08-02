# HF Eco2AI Plugin Developer Guide

## Table of Contents

1. [Development Setup](#development-setup)
2. [Architecture Overview](#architecture-overview)
3. [Contributing Guidelines](#contributing-guidelines)
4. [Testing](#testing)
5. [API Reference](#api-reference)
6. [Extension Development](#extension-development)
7. [Performance Optimization](#performance-optimization)
8. [Release Process](#release-process)

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- NVIDIA GPU (for testing GPU tracking)
- Docker and Docker Compose (for integration testing)

### Local Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/hf-eco2ai-plugin.git
cd hf-eco2ai-plugin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -m pytest tests/ -v
```

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=1.2.0",
    "coverage>=7.0.0"
]
```

### IDE Configuration

#### VS Code Settings

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.sortImports.args": ["--profile", "black"]
}
```

#### PyCharm Configuration

1. Set interpreter to `./venv/bin/python`
2. Configure test runner to pytest
3. Enable Black formatter
4. Configure isort with Black profile

## Architecture Overview

### Core Components

```
hf_eco2ai/
├── __init__.py              # Public API exports
├── callback.py              # Main Eco2AICallback class
├── config.py                # Configuration management
├── metrics.py               # Metrics collection and calculation
├── energy/                  # Energy measurement modules
│   ├── __init__.py
│   ├── base.py              # Abstract energy monitor
│   ├── gpu.py               # GPU energy tracking
│   ├── cpu.py               # CPU energy tracking
│   └── system.py            # System-wide energy tracking
├── carbon/                  # Carbon intensity and calculations
│   ├── __init__.py
│   ├── regions.py           # Regional carbon data
│   ├── realtime.py          # Real-time grid data APIs
│   └── calculator.py        # CO₂ calculations
├── export/                  # Data export functionality
│   ├── __init__.py
│   ├── prometheus.py        # Prometheus metrics export
│   ├── grafana.py           # Grafana dashboard generation
│   └── reports.py           # Report generation
├── integrations/            # Third-party integrations
│   ├── __init__.py
│   ├── mlflow.py            # MLflow integration
│   ├── wandb.py             # Weights & Biases integration
│   └── lightning.py         # PyTorch Lightning integration
└── utils/                   # Utility functions
    ├── __init__.py
    ├── hardware.py          # Hardware detection
    ├── logging.py           # Logging utilities
    └── validation.py        # Input validation
```

### Key Classes

#### Eco2AICallback

Main callback class that integrates with Hugging Face Trainer:

```python
class Eco2AICallback(TrainerCallback):
    def __init__(self, config: Optional[CarbonConfig] = None):
        self.config = config or CarbonConfig()
        self.energy_monitor = EnergyMonitor(self.config)
        self.carbon_calculator = CarbonCalculator(self.config)
        self.metrics_exporter = MetricsExporter(self.config)
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize tracking when training starts"""
        
    def on_step_end(self, args, state, control, **kwargs):
        """Collect metrics at each training step"""
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Process metrics at epoch end"""
```

#### EnergyMonitor

Handles energy measurement across different hardware:

```python
class EnergyMonitor:
    def __init__(self, config: CarbonConfig):
        self.gpu_monitor = GPUEnergyMonitor(config.gpu_ids)
        self.cpu_monitor = CPUEnergyMonitor()
        self.system_monitor = SystemEnergyMonitor()
    
    def start_monitoring(self):
        """Start energy measurement"""
        
    def get_current_consumption(self) -> EnergyMetrics:
        """Get current energy consumption"""
        
    def stop_monitoring(self) -> EnergyReport:
        """Stop monitoring and return summary"""
```

### Data Flow

```
Training Loop
     |
     v
Eco2AICallback
     |
     v
EnergyMonitor -----> CarbonCalculator
     |                       |
     v                       v
MetricsCollector -----> MetricsExporter
     |                       |
     v                       v
InternalStorage -----> ExternalSystems
                            |
                            v
                   [Prometheus, MLflow, etc.]
```

## Contributing Guidelines

### Code Style

We use Black for code formatting and isort for import sorting:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

### Type Hints

All code must include type hints:

```python
from typing import Dict, List, Optional, Union

def calculate_carbon_intensity(
    energy_kwh: float,
    grid_intensity: Optional[float] = None,
    region: str = "global"
) -> Dict[str, Union[float, str]]:
    """Calculate carbon intensity for energy consumption.
    
    Args:
        energy_kwh: Energy consumption in kWh
        grid_intensity: Grid carbon intensity in g CO₂/kWh
        region: Geographic region for grid data
        
    Returns:
        Dictionary containing carbon metrics
    """
    pass
```

### Documentation

All public functions and classes must have docstrings:

```python
class CarbonConfig:
    """Configuration for carbon tracking.
    
    This class manages all configuration options for the Eco2AI callback,
    including energy measurement settings, carbon intensity data, and
    export configurations.
    
    Attributes:
        project_name: Name of the project for tracking
        gpu_ids: List of GPU IDs to monitor
        measurement_interval: Seconds between measurements
        
    Example:
        >>> config = CarbonConfig(
        ...     project_name="my-experiment",
        ...     gpu_ids=[0, 1],
        ...     measurement_interval=5
        ... )
    """
```

### Testing Guidelines

#### Unit Tests

Test individual components in isolation:

```python
import pytest
from unittest.mock import Mock, patch
from hf_eco2ai.energy.gpu import GPUEnergyMonitor

class TestGPUEnergyMonitor:
    @patch('pynvml.nvmlDeviceGetPowerUsage')
    def test_get_power_usage(self, mock_power):
        """Test GPU power usage measurement."""
        mock_power.return_value = 250000  # 250W in milliwatts
        
        monitor = GPUEnergyMonitor([0])
        power = monitor.get_current_power()
        
        assert power == 250.0
        mock_power.assert_called_once()
    
    def test_invalid_gpu_id(self):
        """Test handling of invalid GPU IDs."""
        with pytest.raises(ValueError):
            GPUEnergyMonitor([999])
```

#### Integration Tests

Test component interactions:

```python
from hf_eco2ai import Eco2AICallback, CarbonConfig
from transformers import Trainer, TrainingArguments

class TestEco2AIIntegration:
    def test_trainer_integration(self, mock_model, mock_dataset):
        """Test callback integration with Hugging Face Trainer."""
        config = CarbonConfig(project_name="test")
        callback = Eco2AICallback(config)
        
        args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=2
        )
        
        trainer = Trainer(
            model=mock_model,
            args=args,
            train_dataset=mock_dataset,
            callbacks=[callback]
        )
        
        # Should not raise exceptions
        trainer.train()
        
        # Should have collected metrics
        assert callback.total_energy > 0
        assert callback.total_co2 > 0
```

#### Performance Tests

Ensure minimal overhead:

```python
import time
import pytest
from hf_eco2ai import Eco2AICallback

class TestPerformance:
    def test_callback_overhead(self, benchmark_trainer):
        """Test that callback adds minimal overhead."""
        # Baseline timing without callback
        start_time = time.time()
        benchmark_trainer.train()
        baseline_time = time.time() - start_time
        
        # Timing with callback
        benchmark_trainer.add_callback(Eco2AICallback())
        start_time = time.time()
        benchmark_trainer.train()
        callback_time = time.time() - start_time
        
        # Overhead should be less than 5%
        overhead = (callback_time - baseline_time) / baseline_time
        assert overhead < 0.05
```

### Benchmarking

Run performance benchmarks:

```bash
# Run benchmark suite
python -m pytest benchmarks/ --benchmark-only

# Compare with baseline
python -m pytest benchmarks/ --benchmark-compare=baseline.json
```

## API Reference

### Core Classes

#### Eco2AICallback

```python
class Eco2AICallback(TrainerCallback):
    """Main callback for carbon tracking in Hugging Face Trainer."""
    
    def __init__(
        self,
        config: Optional[CarbonConfig] = None,
        integrations: Optional[List[Integration]] = None
    ) -> None: ...
    
    def get_current_metrics(self) -> CarbonMetrics: ...
    def generate_report(self) -> CarbonReport: ...
    def export_metrics(self, format: str = "json") -> str: ...
```

#### CarbonConfig

```python
@dataclass
class CarbonConfig:
    """Configuration for carbon tracking."""
    
    project_name: str = "hf-training"
    gpu_ids: Union[List[int], str] = "auto"
    measurement_interval: float = 1.0
    log_level: str = "EPOCH"
    export_prometheus: bool = False
    prometheus_port: int = 9091
    save_report: bool = True
    report_path: str = "carbon_report.json"
    country: Optional[str] = None
    region: Optional[str] = None
    grid_carbon_intensity: Optional[float] = None
    use_real_time_carbon: bool = False
    auto_detect_location: bool = False
```

## Extension Development

### Creating Custom Integrations

```python
from hf_eco2ai.integrations.base import BaseIntegration
from hf_eco2ai.types import CarbonMetrics

class CustomIntegration(BaseIntegration):
    """Example custom integration."""
    
    def __init__(self, config: dict):
        self.config = config
        self.client = CustomClient(config)
    
    def log_metrics(self, metrics: CarbonMetrics) -> None:
        """Log metrics to custom system."""
        self.client.send_metrics({
            "energy_kwh": metrics.energy_kwh,
            "co2_kg": metrics.co2_kg,
            "timestamp": metrics.timestamp
        })
    
    def on_training_end(self, report: CarbonReport) -> None:
        """Handle training completion."""
        self.client.send_report(report.to_dict())

# Usage
custom_integration = CustomIntegration({"api_key": "your-key"})
callback = Eco2AICallback(
    integrations=[custom_integration]
)
```

### Custom Energy Monitors

```python
from hf_eco2ai.energy.base import BaseEnergyMonitor
from hf_eco2ai.types import EnergyMetrics

class CustomEnergyMonitor(BaseEnergyMonitor):
    """Custom energy monitoring implementation."""
    
    def start_monitoring(self) -> None:
        """Start energy measurement."""
        # Initialize your monitoring hardware/API
        pass
    
    def get_current_consumption(self) -> EnergyMetrics:
        """Get current energy consumption."""
        # Read from your energy measurement system
        return EnergyMetrics(
            total_energy_kwh=self.read_energy(),
            power_watts=self.read_power(),
            timestamp=time.time()
        )
    
    def stop_monitoring(self) -> EnergyMetrics:
        """Stop monitoring and return final metrics."""
        # Clean up and return final readings
        pass

# Register custom monitor
from hf_eco2ai.energy import register_monitor
register_monitor("custom", CustomEnergyMonitor)

# Use in config
config = CarbonConfig(energy_monitor="custom")
```

## Performance Optimization

### Profiling

```bash
# Profile callback performance
python -m cProfile -o profile.stats train_with_callback.py

# Analyze results
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(10)
"
```

### Memory Usage

```python
# Monitor memory usage
from memory_profiler import profile

@profile
def train_with_monitoring():
    callback = Eco2AICallback()
    trainer = Trainer(
        model=model,
        callbacks=[callback]
    )
    trainer.train()

# Run with: python -m memory_profiler train_script.py
```

### Optimization Tips

1. **Reduce measurement frequency** for long training runs
2. **Use aggregated metrics** instead of per-GPU details
3. **Batch metric exports** to reduce I/O overhead
4. **Cache carbon intensity data** to avoid repeated API calls

## Release Process

### Version Management

We use semantic versioning (SemVer):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite**: `python -m pytest tests/`
4. **Run benchmarks**: `python -m pytest benchmarks/`
5. **Build documentation**: `make docs`
6. **Create release branch**: `git checkout -b release/v1.2.0`
7. **Tag release**: `git tag -a v1.2.0 -m "Release v1.2.0"`
8. **Build and test package**: `python -m build && twine check dist/*`
9. **Upload to PyPI**: `twine upload dist/*`
10. **Create GitHub release** with changelog

### Automated Release

We use GitHub Actions for automated releases:

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

---

For more detailed information, see our [API Documentation](https://hf-eco2ai.readthedocs.io) and [Contributing Guidelines](CONTRIBUTING.md).
