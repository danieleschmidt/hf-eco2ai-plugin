# HF Eco2AI Plugin

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/HF-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Carbon](https://img.shields.io/badge/Carbon-Tracking-green.svg)](https://github.com/danieleschmidt/hf-eco2ai-plugin)
[![CI/CD](https://github.com/danieleschmidt/hf-eco2ai-plugin/workflows/Comprehensive%20CI%2FCD/badge.svg)](https://github.com/danieleschmidt/hf-eco2ai-plugin/actions)
[![Security](https://github.com/danieleschmidt/hf-eco2ai-plugin/workflows/Security/badge.svg)](https://github.com/danieleschmidt/hf-eco2ai-plugin/actions)
[![Carbon Tracking](https://github.com/danieleschmidt/hf-eco2ai-plugin/workflows/Carbon%20Tracking/badge.svg)](https://github.com/danieleschmidt/hf-eco2ai-plugin/actions)

A comprehensive Hugging Face Trainer callback that logs CO₂, kWh, and regional grid intensity for every epoch. Built on Eco2AI's energy tracking with enterprise-grade monitoring, automation, and sustainability features.

## 🌱 Overview

Eco2AI hit HackerNews for accurate energy tracking, but lacks integration with popular ML frameworks. This plugin provides:

- **Seamless HF integration** - Just add one callback
- **Real-time carbon tracking** - CO₂ emissions per epoch/step
- **Regional grid data** - Accurate carbon intensity by location
- **Enterprise monitoring** - Prometheus/Grafana stack with alerting
- **Automation & CI/CD** - Comprehensive workflows and maintenance
- **Security & compliance** - Multi-tool security scanning and reporting
- **Sustainability optimization** - Carbon budget enforcement and optimization recommendations

## 🏗️ Enterprise Features

### Comprehensive SDLC Implementation
- **Project Foundation**: ADRs, roadmaps, charters, and comprehensive documentation
- **Development Environment**: Pre-commit hooks, version consistency, configuration validation
- **Testing Infrastructure**: Unit, integration, performance, and E2E test suites
- **Build & Containerization**: Multi-stage builds, security scanning, SBOM generation
- **Monitoring & Observability**: Prometheus/Grafana stack with custom dashboards and alerts
- **Workflow Automation**: GitHub Actions templates for CI/CD, security, and carbon tracking
- **Metrics & Automation**: Real-time metrics collection, automated maintenance, and reporting

## ⚡ Key Metrics

- **Energy Consumption** (kWh)
- **CO₂ Emissions** (kg CO₂eq)
- **Grid Carbon Intensity** (g CO₂/kWh)
- **GPU Power Draw** (Watts)
- **Training Efficiency** (samples/kWh)

## 📋 Requirements

```bash
# Core dependencies
python>=3.10
transformers>=4.40.0
pytorch-lightning>=2.2.0  # Optional
eco2ai>=2.0.0
pynvml>=11.5.0  # NVIDIA GPU monitoring

# Monitoring
prometheus-client>=0.20.0
grafana-api>=1.0.3
pandas>=2.0.0
plotly>=5.20.0

# Cloud carbon data
carbontracker>=1.5.0
codecarbon>=2.3.0
```

## 🛠️ Installation

### Quick Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/danieleschmidt/hf-eco2ai-plugin.git
cd hf-eco2ai-plugin

# Run automated setup (includes environment, monitoring, and validation)
python scripts/final-integration.py --task all

# Verify installation
python scripts/validate-setup.py --category all
```

### Manual Installation

```bash
# From PyPI (when published)
pip install hf-eco2ai-plugin

# From source
git clone https://github.com/danieleschmidt/hf-eco2ai-plugin.git
cd hf-eco2ai-plugin
pip install -e .[dev,all]

# Set up development environment
pre-commit install
cp .env.example .env
```

## 🚀 Quick Start

### Basic Usage

```python
from transformers import Trainer, TrainingArguments
from hf_eco2ai import Eco2AICallback

# Add callback to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[Eco2AICallback()]  # That's it!
)

# Train as normal - carbon tracking happens automatically
trainer.train()

# View carbon report
print(trainer.carbon_report)
```

### Detailed Configuration

```python
from hf_eco2ai import Eco2AICallback, CarbonConfig

# Configure tracking
carbon_config = CarbonConfig(
    project_name="llama-finetune",
    country="USA",
    region="CA",  # California
    gpu_ids=[0, 1, 2, 3],
    log_level="STEP",  # or "EPOCH"
    export_prometheus=True,
    prometheus_port=9091,
    save_report=True,
    report_path="carbon_impact.json"
)

# Initialize callback
eco_callback = Eco2AICallback(config=carbon_config)

# Use with trainer
trainer = Trainer(
    model=model,
    args=args,
    callbacks=[eco_callback]
)
```

## 📊 Features

### Real-time Monitoring

```python
# Access metrics during training
def on_epoch_end(trainer, eco_callback):
    metrics = eco_callback.get_current_metrics()
    print(f"Epoch {trainer.epoch}")
    print(f"Energy used: {metrics.energy_kwh:.2f} kWh")
    print(f"CO₂ emitted: {metrics.co2_kg:.2f} kg")
    print(f"Efficiency: {metrics.samples_per_kwh:.0f} samples/kWh")
```

### PyTorch Lightning Integration

```python
from pytorch_lightning import Trainer
from hf_eco2ai.lightning import Eco2AILightningCallback

# Works with Lightning too!
trainer = Trainer(
    callbacks=[Eco2AILightningCallback()],
    accelerator="gpu",
    devices=4
)

trainer.fit(model, datamodule)
```

### Multi-GPU Tracking

```python
# Automatically tracks all GPUs
eco_callback = Eco2AICallback(
    track_gpu_energy=True,
    gpu_ids="auto",  # Detects all available GPUs
    aggregate_gpus=True  # Sum energy across GPUs
)

# Or track specific GPUs
eco_callback = Eco2AICallback(
    gpu_ids=[0, 2, 4, 6],  # Track only these GPUs
    per_gpu_metrics=True   # Report per-GPU metrics
)
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  HF Trainer     │────▶│ Eco2AI       │────▶│ Energy Monitor  │
│                 │     │ Callback     │     │ (CPU/GPU)       │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Training Loop   │     │ Grid Carbon  │     │ Metrics Export  │
│                 │     │ Intensity    │     │ (Prometheus)    │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## 📈 Dashboards

### Grafana Setup

```bash
# Import dashboard
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboards/hf-carbon-tracking.json

# Or use Docker
docker run -d \
  -p 3000:3000 \
  -v $(pwd)/dashboards:/var/lib/grafana/dashboards \
  hf-eco2ai/grafana-carbon
```

### Dashboard Features
- Real-time power consumption
- Cumulative CO₂ emissions
- Training efficiency trends
- Regional grid intensity
- Cost estimation (optional)

## 🌍 Regional Carbon Data

### Automatic Region Detection

```python
# Auto-detect location and grid carbon intensity
eco_callback = Eco2AICallback(
    auto_detect_location=True,
    use_real_time_carbon=True  # Live grid data
)
```

### Manual Region Configuration

```python
# Specify exact location for accurate carbon data
from hf_eco2ai import regions

eco_callback = Eco2AICallback(
    country="Germany",
    region="Bavaria",
    grid_carbon_intensity=regions.GERMANY.BAVARIA  # 411 g CO₂/kWh
)

# Or use coordinates
eco_callback = Eco2AICallback(
    latitude=48.1351,
    longitude=11.5820,
    use_real_time_carbon=True
)
```

## 🔧 Advanced Features

### Custom Metrics

```python
from hf_eco2ai import Eco2AICallback, MetricCollector

class CustomEco2AICallback(Eco2AICallback):
    def compute_additional_metrics(self, logs):
        # Add custom efficiency metrics
        if "loss" in logs and self.current_energy > 0:
            logs["loss_per_kwh"] = logs["loss"] / self.current_energy
            logs["carbon_per_sample"] = self.total_co2 / self.samples_seen
        
        return logs
```

### Experiment Comparison

```python
from hf_eco2ai.analysis import CarbonComparison

# Compare different training runs
comparison = CarbonComparison()

# Add experiments
comparison.add_experiment("baseline", "carbon_reports/baseline.json")
comparison.add_experiment("efficient", "carbon_reports/efficient.json")
comparison.add_experiment("quantized", "carbon_reports/quantized.json")

# Generate comparison report
comparison.plot_comparison(
    metrics=["total_co2", "samples_per_kwh", "cost"],
    save_path="comparison.html"
)
```

### Carbon Budget Enforcement

```python
from hf_eco2ai import CarbonBudgetCallback

# Stop training if carbon budget exceeded
budget_callback = CarbonBudgetCallback(
    max_co2_kg=10.0,  # 10kg CO₂ budget
    action="stop",    # or "warn"
    check_frequency=100  # Check every 100 steps
)

trainer = Trainer(
    model=model,
    args=args,
    callbacks=[budget_callback]
)
```

## 📊 Reports

### Generate Carbon Report

```python
# After training
carbon_report = eco_callback.generate_report()

print(carbon_report.summary())
"""
Training Carbon Impact Report
============================
Total Energy: 45.3 kWh
Total CO₂: 18.7 kg CO₂eq
Grid Intensity: 412 g CO₂/kWh
Duration: 3h 24m
Efficiency: 1,847 samples/kWh

Equivalent to:
- 78 km driven by car
- 2.3 trees needed to offset
- $4.53 in carbon credits
"""

# Export detailed report
carbon_report.to_json("carbon_impact.json")
carbon_report.to_pdf("carbon_impact.pdf")
carbon_report.to_csv("carbon_metrics.csv")
```

### MLflow Integration

```python
import mlflow
from hf_eco2ai.mlflow import log_carbon_metrics

# Automatically log to MLflow
with mlflow.start_run():
    trainer = Trainer(
        model=model,
        args=args,
        callbacks=[
            Eco2AICallback(mlflow_tracking=True)
        ]
    )
    
    trainer.train()
    
    # Metrics automatically logged to MLflow
```

## 🚦 CI/CD Integration

### Comprehensive GitHub Actions Workflows

The project includes enterprise-grade workflow templates:

```bash
# Set up workflows (requires repository admin)
cp docs/workflows/comprehensive-ci.yml.template .github/workflows/ci.yml
cp docs/workflows/security.yml.template .github/workflows/security.yml
cp docs/workflows/carbon-tracking.yml.template .github/workflows/carbon-tracking.yml
cp docs/workflows/dependency-update.yml.template .github/workflows/dependency-update.yml

# Configure repository secrets (in GitHub Settings → Secrets)
# CODECOV_TOKEN, PYPI_API_TOKEN, SLACK_WEBHOOK_URL
```

### Features

- **Multi-OS Testing**: Ubuntu, Windows, macOS with Python 3.10-3.12
- **Security Scanning**: Bandit, Safety, pip-audit, CodeQL, TruffleHog
- **Carbon Tracking**: Automated CI/CD footprint monitoring with budget enforcement
- **Dependency Management**: Automated security updates and dependency maintenance
- **Quality Gates**: Pre-commit hooks, test coverage, security validation
- **Automated Releases**: Semantic versioning, changelog generation, PyPI publishing

### Example Training with Carbon Tracking

```yaml
# .github/workflows/train.yml
name: Train with Carbon Tracking

on: [push]

jobs:
  train:
    runs-on: gpu-runner
    steps:
    - uses: actions/checkout@v4
    
    - name: Train model
      run: |
        python train.py --carbon-tracking
    
    - name: Check carbon budget
      run: |
        python scripts/collect-metrics.py --update
        python -m hf_eco2ai check-budget \
          --report carbon_impact.json \
          --max-co2 5.0
    
    - name: Upload carbon report
      uses: actions/upload-artifact@v4
      with:
        name: carbon-report
        path: carbon_impact.json
```

## 🌱 Best Practices

### Reduce Carbon Impact

```python
from hf_eco2ai.optimization import CarbonOptimizer

optimizer = CarbonOptimizer()

# Get recommendations
recommendations = optimizer.analyze_training(
    model=model,
    dataset_size=len(train_dataset),
    target_accuracy=0.95
)

print(recommendations)
"""
Recommendations to reduce carbon impact:
1. Use mixed precision training (-40% energy)
2. Enable gradient checkpointing (-25% memory)
3. Train during low-carbon hours (23:00-06:00)
4. Use renewable energy regions (Norway: 20g CO₂/kWh)
5. Consider model pruning after training
"""
```

### Schedule Low-Carbon Training

```python
from hf_eco2ai.scheduling import LowCarbonScheduler

scheduler = LowCarbonScheduler(
    region="California",
    flexibility_hours=12  # Can wait up to 12 hours
)

# Find optimal training window
best_time = scheduler.find_low_carbon_window(
    estimated_duration_hours=4,
    start_after=datetime.now()
)

print(f"Train at {best_time} for {scheduler.carbon_reduction:.1%} less CO₂")
```

## 🤝 Contributing

We welcome contributions! Priority areas:
- Additional cloud provider regions
- More efficient energy measurement
- Integration with other frameworks
- Carbon offset integrations
- Visualization improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

```bibtex
@software{hf_eco2ai_plugin,
  title={HF Eco2AI Plugin: Carbon Tracking for Transformers Training},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/hf-eco2ai-plugin}
}

@article{eco2ai,
  title={Eco2AI: Carbon Emissions Tracking for AI},
  author={Eco2AI Team},
  year={2024}
}
```

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 📊 Enterprise Management

### Automated Metrics Collection

```bash
# Real-time metrics collection
python scripts/collect-metrics.py --update --format summary

# Maintenance automation
python scripts/maintenance.py --task full --dry-run

# Integration management
python scripts/integration-tools.py --action all
```

### Monitoring Stack

```bash
# Start monitoring services
docker-compose up -d

# Access dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
open http://localhost:9093  # Alertmanager
```

### Validation and Setup

```bash
# Comprehensive validation
python scripts/validate-setup.py --category all

# Setup verification
python scripts/final-integration.py --task validation
```

## 🔗 Resources

### Documentation
- [Setup Guide](docs/SETUP_GUIDE.md) - Complete environment setup
- [Operations Manual](docs/OPERATIONS_MANUAL.md) - Daily operations and maintenance
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [API Reference](docs/api/API_REFERENCE.md) - Comprehensive API documentation
- [Workflow Setup Guide](docs/workflows/WORKFLOW_SETUP_GUIDE.md) - GitHub Actions configuration

### Architecture Documents
- [ADR Template](docs/adr/0001-architecture-decision-template.md) - Decision documentation standard
- [Project Charter](PROJECT_CHARTER.md) - Project scope and objectives
- [Roadmap](docs/ROADMAP.md) - Development roadmap and milestones

### Monitoring & Automation
- [Monitoring Guide](docs/monitoring/monitoring-guide.md) - Complete monitoring setup
- [Carbon Runbooks](docs/runbooks/carbon-emissions-runbook.md) - Emergency procedures
- [Deployment Guide](docs/deployment/deployment-guide.md) - Multi-environment deployment

## 📧 Contact

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **Repository**: [danieleschmidt/hf-eco2ai-plugin](https://github.com/danieleschmidt/hf-eco2ai-plugin)
