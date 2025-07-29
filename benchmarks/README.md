# Performance Benchmarks

This directory contains performance benchmarks for the HF Eco2AI Plugin to ensure tracking accuracy and minimal overhead.

## Overview

Performance benchmarking focuses on:
- **Tracking Overhead**: Measuring the impact on training performance
- **Memory Usage**: Monitoring additional memory consumption
- **Accuracy**: Validating carbon tracking precision
- **Scalability**: Testing with different model sizes and GPU configurations

## Benchmark Categories

### 1. Overhead Benchmarks
Test the performance impact of enabling carbon tracking:

```python
# benchmarks/test_overhead.py
import time
import pytest
from transformers import Trainer
from hf_eco2ai import Eco2AICallback

def benchmark_training_overhead():
    """Measure training time with and without carbon tracking."""
    # Test with baseline (no callback)
    baseline_time = run_training_benchmark(callbacks=[])
    
    # Test with Eco2AI callback
    callback_time = run_training_benchmark(callbacks=[Eco2AICallback()])
    
    overhead_percent = ((callback_time - baseline_time) / baseline_time) * 100
    assert overhead_percent < 5.0  # Less than 5% overhead
```

### 2. Memory Benchmarks
Monitor memory consumption during tracking:

```python
# benchmarks/test_memory.py
import psutil
import pytest
from hf_eco2ai import Eco2AICallback

def benchmark_memory_usage():
    """Measure memory usage of carbon tracking."""
    initial_memory = psutil.virtual_memory().used
    
    callback = Eco2AICallback()
    # Run training simulation
    
    peak_memory = psutil.virtual_memory().used
    memory_overhead = peak_memory - initial_memory
    
    assert memory_overhead < 100 * 1024 * 1024  # Less than 100MB
```

### 3. Accuracy Benchmarks
Validate tracking accuracy against known baselines:

```python
# benchmarks/test_accuracy.py
import pytest
from hf_eco2ai import Eco2AICallback

def benchmark_energy_tracking_accuracy():
    """Compare against reference energy measurements."""
    callback = Eco2AICallback()
    
    # Run controlled training with known energy consumption
    measured_energy = callback.get_total_energy()
    expected_energy = 2.5  # kWh from reference measurement
    
    accuracy_error = abs(measured_energy - expected_energy) / expected_energy
    assert accuracy_error < 0.1  # Within 10% accuracy
```

## Running Benchmarks

### Prerequisites
```bash
pip install -e ".[dev]"
pip install pytest-benchmark memory-profiler
```

### Execute Benchmarks
```bash
# Run all benchmarks
pytest benchmarks/ -v

# Run specific benchmark category
pytest benchmarks/test_overhead.py -v

# Run with performance reporting
pytest benchmarks/ --benchmark-only --benchmark-sort=mean
```

### Continuous Benchmarking
```bash
# Store benchmark results
pytest benchmarks/ --benchmark-json=results/benchmark_results.json

# Compare with previous results
pytest benchmarks/ --benchmark-compare=results/baseline.json
```

## Benchmark Results

### Target Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Training Overhead | < 5% | Additional time added to training |
| Memory Overhead | < 100MB | Additional memory consumption |
| Tracking Accuracy | < 10% error | Carbon tracking precision |
| GPU Monitoring | < 1ms latency | GPU metrics collection speed |

### Historical Results

Results are stored in `benchmarks/results/` with timestamps for tracking performance over time.

## Benchmark Configuration

### Environment Variables
```bash
export BENCHMARK_GPU_COUNT=4
export BENCHMARK_MODEL_SIZE=large
export BENCHMARK_DURATION=300  # seconds
```

### Custom Configurations
```python
# benchmarks/config.py
BENCHMARK_CONFIGS = {
    "small_model": {
        "model_name": "distilbert-base-uncased",
        "batch_size": 16,
        "max_steps": 100
    },
    "large_model": {
        "model_name": "bert-large-uncased", 
        "batch_size": 8,
        "max_steps": 50
    }
}
```

## Integration with CI/CD

Benchmarks run automatically in CI to detect performance regressions:

```yaml
# .github/workflows/benchmark.yml
- name: Run Performance Benchmarks
  run: |
    pytest benchmarks/ --benchmark-json=benchmark_results.json
    
- name: Store Benchmark Results
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'pytest'
    output_file: benchmark_results.json
```

## Profiling Tools

### Memory Profiling
```bash
# Line-by-line memory usage
python -m memory_profiler benchmarks/profile_memory.py

# Memory usage over time
mprof run benchmarks/profile_memory.py
mprof plot
```

### CPU Profiling
```bash
# CPU performance profiling
python -m cProfile -o profile_output benchmarks/profile_cpu.py
python -m pstats profile_output
```

### GPU Profiling
```bash
# NVIDIA profiling (if available)
nvprof python benchmarks/profile_gpu.py

# PyTorch profiler
python benchmarks/profile_pytorch.py
```

## Best Practices

1. **Consistent Environment**: Run benchmarks in isolated, consistent environments
2. **Statistical Significance**: Run multiple iterations and report confidence intervals
3. **Baseline Comparison**: Always compare against baseline performance
4. **Automated Regression Detection**: Fail CI if performance degrades significantly
5. **Resource Monitoring**: Monitor CPU, memory, and GPU usage during benchmarks

## Contributing Benchmarks

When adding new benchmarks:

1. Follow the naming convention: `test_[category]_[specific_test].py`
2. Include docstrings explaining what is being measured
3. Set appropriate performance targets
4. Add any new dependencies to `pyproject.toml`
5. Document the benchmark in this README