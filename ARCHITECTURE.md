# Architecture Overview

This document describes the architecture and design decisions for the HF Eco2AI Plugin.

## System Overview

The HF Eco2AI Plugin provides a seamless integration between Hugging Face Transformers and Eco2AI for real-time carbon footprint tracking during model training.

```
┌─────────────────────────────────────────────────────────────────┐
│                    HF Eco2AI Plugin Architecture                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  HF Trainer     │────▶│ Eco2AI       │────▶│ Energy Monitor  │
│  Callback API   │     │ Callback     │     │ (CPU/GPU)       │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Training Loop   │     │ Grid Carbon  │     │ Metrics Export  │
│ Integration     │     │ Intensity    │     │ (Prometheus)    │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Report          │     │ Regional     │     │ Visualization   │
│ Generation      │     │ Data APIs    │     │ (Grafana)       │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## Core Components

### 1. Callback System

#### Eco2AICallback
- **Purpose**: Main callback class integrating with HF Trainer
- **Responsibilities**:
  - Initialize energy monitoring
  - Hook into training lifecycle events
  - Aggregate and report metrics
  - Export data to external systems

#### Eco2AILightningCallback  
- **Purpose**: PyTorch Lightning integration
- **Responsibilities**:
  - Lightning-specific training hooks
  - Multi-device coordination
  - Distributed training support

### 2. Energy Monitoring

#### GPU Monitoring
- **Technology**: NVIDIA pynvml library
- **Metrics**: Power draw, utilization, temperature
- **Frequency**: Per-step or per-epoch configurable
- **Multi-GPU**: Automatic detection and aggregation

#### CPU Monitoring
- **Technology**: Eco2AI integration
- **Metrics**: CPU power consumption, utilization
- **Regional Data**: Grid carbon intensity by location

### 3. Data Management

#### Configuration System
```python
@dataclass
class CarbonConfig:
    project_name: str
    country: str
    region: str
    gpu_ids: List[int]
    log_level: str
    export_prometheus: bool
    prometheus_port: int
    save_report: bool
    report_path: str
```

#### Metrics Collection
- **Real-time**: Live monitoring during training
- **Aggregation**: Per-epoch and training totals
- **Storage**: JSON, CSV, database export options

### 4. External Integrations

#### Prometheus Export
- **Endpoint**: `/metrics` on configurable port
- **Metrics**: Gauges and counters for energy/carbon
- **Labels**: Training metadata, GPU IDs, regions

#### MLflow Integration
- **Auto-logging**: Automatic metric logging
- **Experiments**: Integration with existing MLflow runs
- **Artifacts**: Carbon reports as run artifacts

## Design Principles

### 1. Non-Intrusive Integration
- **Zero Code Changes**: Works with existing training scripts
- **Minimal Overhead**: <1% performance impact
- **Optional Features**: All advanced features are opt-in

### 2. Accuracy First
- **Hardware Monitoring**: Direct GPU power measurement
- **Regional Data**: Real-time grid carbon intensity
- **Validated Metrics**: Cross-validated with external tools

### 3. Extensibility
- **Plugin Architecture**: Support for multiple ML frameworks
- **Custom Metrics**: Extensible metric collection system
- **External APIs**: Pluggable carbon data sources

### 4. Production Ready
- **Error Handling**: Graceful degradation on failures
- **Resource Management**: Minimal memory footprint
- **Security**: No sensitive data collection or transmission

## Data Flow

### Training Lifecycle

```
Training Start
      │
      ▼
┌─────────────┐
│ Initialize  │
│ Monitoring  │
└─────────────┘
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Epoch Start │───▶│ Step Start  │───▶│ Step End    │
└─────────────┘    └─────────────┘    └─────────────┘
      │                   │                   │
      │                   ▼                   ▼
      │            ┌─────────────┐    ┌─────────────┐
      │            │ GPU Sample  │    │ Accumulate  │
      │            │ Power Draw  │    │ Metrics     │
      │            └─────────────┘    └─────────────┘
      ▼
┌─────────────┐
│ Epoch End   │
│ - Aggregate │
│ - Export    │
│ - Report    │
└─────────────┘
      │
      ▼
┌─────────────┐
│ Training    │
│ Complete    │
│ - Final     │
│   Report    │
└─────────────┘
```

### Metric Collection Strategy

1. **High-Frequency Sampling**: GPU power every 100ms
2. **Training Hooks**: Aggregate at epoch boundaries  
3. **Efficient Storage**: In-memory ring buffers
4. **Export Options**: Real-time or batch export

## Performance Considerations

### Memory Usage
- **Ring Buffers**: Fixed-size metric storage
- **Lazy Evaluation**: Compute aggregations on-demand
- **Memory Pool**: Reuse objects to reduce GC pressure

### CPU Overhead
- **Background Sampling**: Separate thread for monitoring
- **Efficient APIs**: Direct NVML calls, no shell commands
- **Caching**: Cache grid intensity data per region

### GPU Impact
- **Minimal Queries**: Only essential NVML calls
- **Batched Operations**: Group multiple metric reads
- **No Compute Impact**: Monitoring doesn't affect training

## Security Architecture

### Data Privacy
- **Local Processing**: All metrics processed locally
- **No PII**: No personally identifiable information collected
- **Configurable Export**: User controls all external data sharing

### Network Security
- **HTTPS Only**: All external API calls use TLS
- **API Key Management**: Secure credential handling
- **Rate Limiting**: Respect external API limits

### Supply Chain Security
- **SBOM Generation**: Complete dependency tracking
- **Vulnerability Scanning**: Continuous security monitoring
- **Dependency Pinning**: Reproducible builds

## Deployment Patterns

### Development
```yaml
# docker-compose.yml development profile
services:
  dev:
    build:
      target: development
    volumes:
      - .:/app:rw
    command: bash
```

### Production
```yaml
# docker-compose.yml production profile
services:
  app:
    build:
      target: production
    environment:
      - PROMETHEUS_ENABLED=true
    networks:
      - monitoring
```

### Monitoring Stack
```yaml
# Complete observability stack
services:
  app: # HF Eco2AI Plugin
  prometheus: # Metrics collection
  grafana: # Visualization
  alertmanager: # Alerting
```

## Future Architecture Considerations

### Scalability
- **Distributed Training**: Multi-node carbon tracking
- **Cloud Integration**: AWS, GCP, Azure carbon APIs
- **Database Backend**: PostgreSQL/InfluxDB for large deployments

### Framework Support
- **JAX Integration**: Eco2AI for JAX/Flax training
- **TensorFlow**: TF 2.x callback implementation
- **LightGBM**: Tree model training carbon tracking

### Advanced Features
- **Carbon Optimization**: Training schedule optimization
- **Predictive Models**: Energy usage prediction
- **Cost Tracking**: Cloud compute cost correlation