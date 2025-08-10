# Complete Setup Guide

This guide provides step-by-step instructions for setting up the HF Eco2AI Plugin development environment.

## Prerequisites

- Python 3.10+
- Git
- Docker (optional)
- GitHub account with repository access

## Quick Start

```bash
# Clone repository
git clone https://github.com/danieleschmidt/hf-eco2ai-plugin.git
cd hf-eco2ai-plugin

# Run automated setup
python scripts/final-integration.py --setup

# Validate installation
python scripts/validate-setup.py
```

## Manual Setup

### 1. Development Environment

```bash
# Install dependencies
pip install -e .[dev,all]

# Set up pre-commit hooks
pre-commit install

# Create environment file
cp .env.example .env
```

### 2. Monitoring Setup

```bash
# Start monitoring stack
docker-compose up -d

# Verify services
curl http://localhost:9090/api/v1/status/config  # Prometheus
curl http://localhost:3000/api/health           # Grafana
```

### 3. Validation

```bash
# Run comprehensive validation
python scripts/validate-setup.py --category all

# Test metrics collection
python scripts/collect-metrics.py --format summary

# Test automation
python scripts/maintenance.py --task health
```

## Configuration

See [Operations Manual](OPERATIONS_MANUAL.md) for detailed configuration options.

## Troubleshooting

See [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues and solutions.
