# HF Eco2AI Plugin User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Configuration Options](#configuration-options)
4. [Advanced Features](#advanced-features)
5. [Monitoring and Dashboards](#monitoring-and-dashboards)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended)
- Hugging Face Transformers 4.40.0+
- Eco2AI 2.0.0+

### Installation

#### From PyPI (Recommended)

```bash
pip install hf-eco2ai-plugin
```

#### From Source

```bash
git clone https://github.com/yourusername/hf-eco2ai-plugin.git
cd hf-eco2ai-plugin
pip install -e .
```

#### Development Installation

```bash
git clone https://github.com/yourusername/hf-eco2ai-plugin.git
cd hf-eco2ai-plugin
pip install -e ".[dev]"
pre-commit install
```

### Verification

Verify your installation:

```python
from hf_eco2ai import Eco2AICallback
print("Installation successful!")
```

## Basic Usage

### Simple Integration

Add carbon tracking to your existing Hugging Face training script:

```python
from transformers import Trainer, TrainingArguments
from hf_eco2ai import Eco2AICallback

# Your existing model and data setup
model = ...
train_dataset = ...
eval_dataset = ...

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    # ... other arguments
)

# Create trainer with Eco2AI callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[Eco2AICallback()]  # Add this line!
)

# Train as usual
trainer.train()

# Access carbon report
print(trainer.carbon_report.summary())
```

### Quick Results

After training, you'll see output like:

```
Training Carbon Impact Summary
==============================
Total Energy: 12.4 kWh
Total CO₂: 5.1 kg CO₂eq
Grid Intensity: 412 g CO₂/kWh
Training Duration: 1h 23m
Efficiency: 2,847 samples/kWh

Carbon Impact Equivalent:
- 21 km driven by car
- 0.6 trees needed to offset
- $1.02 in carbon credits
```

## Configuration Options

### Basic Configuration

```python
from hf_eco2ai import Eco2AICallback, CarbonConfig

# Create configuration
config = CarbonConfig(
    project_name="my-model-training",
    country="USA",
    region="California",
    log_level="EPOCH",  # or "STEP"
    save_report=True,
    report_path="carbon_report.json"
)

# Use with callback
callback = Eco2AICallback(config=config)
```

### GPU Configuration

```python
# Track specific GPUs
config = CarbonConfig(
    gpu_ids=[0, 1, 2, 3],  # Track these GPU IDs
    per_gpu_metrics=True,   # Report per-GPU metrics
    aggregate_gpus=True     # Also provide aggregated metrics
)

# Auto-detect all GPUs
config = CarbonConfig(
    gpu_ids="auto",         # Automatically detect all GPUs
    track_gpu_energy=True   # Enable GPU energy tracking
)
```

### Regional Carbon Intensity

```python
# Manual region specification
config = CarbonConfig(
    country="Germany",
    region="Bavaria",
    grid_carbon_intensity=411  # g CO₂/kWh
)

# Coordinate-based
config = CarbonConfig(
    latitude=48.1351,
    longitude=11.5820,
    use_real_time_carbon=True  # Fetch live grid data
)

# Auto-detection
config = CarbonConfig(
    auto_detect_location=True,
    use_real_time_carbon=True
)
```

## Advanced Features

### Prometheus Integration

```python
config = CarbonConfig(
    export_prometheus=True,
    prometheus_port=9091,
    prometheus_prefix="hf_training"
)

callback = Eco2AICallback(config=config)
```

Metrics exported:
- `hf_training_energy_kwh_total`
- `hf_training_co2_kg_total`
- `hf_training_samples_per_kwh`
- `hf_training_gpu_power_watts`

### Custom Metrics

```python
from hf_eco2ai import Eco2AICallback

class CustomEco2AICallback(Eco2AICallback):
    def compute_additional_metrics(self, logs):
        # Add custom efficiency metrics
        if "eval_loss" in logs and self.current_energy > 0:
            logs["eval_loss_per_kwh"] = logs["eval_loss"] / self.current_energy
            logs["carbon_efficiency"] = len(self.trainer.train_dataset) / self.total_co2
        
        return logs

# Use custom callback
trainer = Trainer(
    model=model,
    args=args,
    callbacks=[CustomEco2AICallback()]
)
```

### Carbon Budget Enforcement

```python
from hf_eco2ai import CarbonBudgetCallback

# Stop training if budget exceeded
budget_callback = CarbonBudgetCallback(
    max_co2_kg=5.0,        # 5kg CO₂ budget
    action="stop",         # or "warn"
    check_frequency=100,   # Check every 100 steps
    grace_period=10        # Allow 10 steps to finish
)

trainer = Trainer(
    model=model,
    args=args,
    callbacks=[budget_callback]
)
```

### PyTorch Lightning Integration

```python
from pytorch_lightning import Trainer
from hf_eco2ai.lightning import Eco2AILightningCallback

# Configure callback
eco_callback = Eco2AILightningCallback(
    project_name="lightning-training",
    save_report=True
)

# Use with Lightning trainer
trainer = Trainer(
    callbacks=[eco_callback],
    accelerator="gpu",
    devices=4,
    max_epochs=10
)

trainer.fit(model, datamodule)
```

## Monitoring and Dashboards

### Grafana Setup

1. **Start Grafana and Prometheus:**

```bash
# Using Docker Compose
docker-compose up -d grafana prometheus
```

2. **Import Dashboard:**

```bash
# Import pre-built dashboard
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GRAFANA_TOKEN" \
  -d @dashboards/hf-carbon-tracking.json
```

3. **Access Dashboard:**
   - URL: http://localhost:3000
   - Dashboard: "HF Eco2AI Carbon Tracking"

### Real-time Monitoring

```python
# Monitor training in real-time
def monitor_training(trainer, callback):
    while trainer.is_training:
        metrics = callback.get_current_metrics()
        print(f"Current energy: {metrics.energy_kwh:.2f} kWh")
        print(f"Current CO₂: {metrics.co2_kg:.2f} kg")
        time.sleep(60)  # Update every minute

# Start monitoring in separate thread
import threading
monitor_thread = threading.Thread(
    target=monitor_training,
    args=(trainer, eco_callback)
)
monitor_thread.start()
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Verify pynvml installation
try:
    import pynvml
    pynvml.nvmlInit()
    print("NVIDIA ML library working")
except Exception as e:
    print(f"NVIDIA ML error: {e}")
```

**Solution:** Install NVIDIA drivers and CUDA toolkit

#### 2. Permission Errors

```bash
# Linux: Add user to appropriate groups
sudo usermod -a -G nvidia $USER
sudo usermod -a -G video $USER

# Restart session
logout
```

#### 3. High Tracking Overhead

```python
# Reduce measurement frequency
config = CarbonConfig(
    measurement_interval=10,  # Measure every 10 seconds (default: 1)
    log_level="EPOCH"         # Log per epoch, not per step
)
```

#### 4. Inaccurate Energy Readings

```python
# Enable detailed logging
import logging
logging.getLogger('hf_eco2ai').setLevel(logging.DEBUG)

# Verify hardware support
from hf_eco2ai.diagnostics import run_hardware_check
run_hardware_check()
```

### Debug Mode

```python
# Enable debug mode
config = CarbonConfig(
    debug=True,
    verbose_logging=True,
    save_raw_metrics=True
)

callback = Eco2AICallback(config=config)
```

## Best Practices

### 1. Minimize Overhead

```python
# Optimal configuration for minimal overhead
config = CarbonConfig(
    measurement_interval=5,    # Balance accuracy vs overhead
    log_level="EPOCH",        # Reduce logging frequency
    aggregate_gpus=True,      # Single aggregated metric
    per_gpu_metrics=False     # Disable per-GPU details
)
```

### 2. Reproducible Tracking

```python
# Ensure reproducible results
config = CarbonConfig(
    project_name="experiment-v1.2",
    experiment_id="run-001",
    save_report=True,
    report_path=f"carbon_reports/{timestamp}.json"
)
```

### 3. Integration with Experiment Tracking

```python
import mlflow
from hf_eco2ai.integrations import MLflowIntegration

# MLflow integration
with mlflow.start_run():
    callback = Eco2AICallback(
        config=config,
        integrations=[MLflowIntegration()]
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        callbacks=[callback]
    )
    
    trainer.train()
    # Carbon metrics automatically logged to MLflow
```

### 4. Carbon Optimization

```python
from hf_eco2ai.optimization import suggest_optimizations

# Get optimization suggestions
optimizations = suggest_optimizations(
    model=model,
    dataset_size=len(train_dataset),
    target_performance=0.95,
    current_config=training_args
)

print("Optimization suggestions:")
for opt in optimizations:
    print(f"- {opt.description}: {opt.estimated_savings}% energy reduction")
```

### 5. Continuous Monitoring

```python
# Setup alerts for high carbon usage
config = CarbonConfig(
    carbon_alerts=True,
    max_co2_per_hour=2.0,  # Alert if >2kg CO₂/hour
    alert_webhook="https://hooks.slack.com/..."
)
```

---

For more examples and detailed API documentation, visit our [Examples Repository](https://github.com/hf-eco2ai/examples) and [API Documentation](https://hf-eco2ai.readthedocs.io).
