# Monitoring Guide for HF Eco2AI Plugin

## Table of Contents

1. [Overview](#overview)
2. [Metrics Reference](#metrics-reference)
3. [Dashboard Setup](#dashboard-setup)
4. [Alerting Configuration](#alerting-configuration)
5. [Observability Best Practices](#observability-best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

## Overview

The HF Eco2AI Plugin provides comprehensive monitoring and observability for machine learning training with a focus on carbon emissions, energy consumption, and sustainability metrics.

### Monitoring Stack

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notifications
- **Node Exporter**: System metrics
- **GPU Exporter**: GPU-specific metrics

### Key Monitoring Areas

1. **Carbon Tracking**: CO₂ emissions, carbon intensity, sustainability goals
2. **Energy Monitoring**: Power consumption, efficiency metrics
3. **Training Performance**: Speed, loss, convergence
4. **System Health**: GPU, CPU, memory, storage
5. **Cost Analysis**: Training costs, resource utilization

## Metrics Reference

### Carbon Emissions Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `hf_training_co2_kg_total` | Counter | Total CO₂ emissions in kg | `project_name`, `model_name` |
| `hf_training_co2_rate_kg_per_hour` | Gauge | Current CO₂ emission rate | `project_name` |
| `hf_training_grid_carbon_intensity_g_per_kwh` | Gauge | Grid carbon intensity | `country`, `region` |
| `hf_training_carbon_budget_kg` | Gauge | Configured carbon budget | `project_name` |
| `hf_training_carbon_budget_remaining_kg` | Gauge | Remaining carbon budget | `project_name` |

### Energy Consumption Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `hf_training_energy_kwh_total` | Counter | Total energy consumption | `project_name`, `model_name` |
| `hf_training_energy_rate_kwh_per_hour` | Gauge | Current energy consumption rate | `project_name` |
| `hf_training_gpu_power_watts` | Gauge | GPU power consumption | `gpu_id`, `gpu_model` |
| `hf_training_cpu_power_watts` | Gauge | CPU power consumption | `instance` |
| `hf_training_samples_per_kwh` | Gauge | Training efficiency | `project_name` |

### Training Performance Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `hf_training_loss` | Gauge | Current training loss | `project_name`, `model_name` |
| `hf_training_eval_loss` | Gauge | Current evaluation loss | `project_name`, `model_name` |
| `hf_training_learning_rate` | Gauge | Current learning rate | `project_name` |
| `hf_training_step_count` | Counter | Total training steps | `project_name` |
| `hf_training_epoch_count` | Counter | Total training epochs | `project_name` |
| `hf_training_samples_per_second` | Gauge | Training throughput | `project_name` |
| `hf_training_efficiency_score` | Gauge | Overall efficiency score (0-1) | `project_name` |

### GPU Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `hf_training_gpu_temperature_celsius` | Gauge | GPU temperature | `gpu_id`, `gpu_model` |
| `hf_training_gpu_memory_usage_bytes` | Gauge | GPU memory usage | `gpu_id`, `gpu_model` |
| `hf_training_gpu_memory_total_bytes` | Gauge | Total GPU memory | `gpu_id`, `gpu_model` |
| `hf_training_gpu_memory_usage_percent` | Gauge | GPU memory usage percentage | `gpu_id`, `gpu_model` |
| `hf_training_gpu_utilization_percent` | Gauge | GPU utilization percentage | `gpu_id`, `gpu_model` |
| `hf_training_gpu_count` | Gauge | Number of available GPUs | `instance` |

### Cost Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `hf_training_estimated_cost_usd` | Gauge | Estimated training cost | `project_name`, `currency` |
| `hf_training_cost_rate_usd_per_hour` | Gauge | Current cost rate | `project_name` |
| `hf_training_energy_cost_usd` | Gauge | Energy-related costs | `project_name` |
| `hf_training_compute_cost_usd` | Gauge | Compute-related costs | `project_name` |

### System Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `hf_training_last_update_timestamp` | Gauge | Last metric update timestamp | `instance` |
| `hf_training_plugin_version` | Info | Plugin version information | `version`, `commit` |
| `hf_training_monitoring_errors_total` | Counter | Total monitoring errors | `error_type` |
| `hf_training_network_latency_ms` | Gauge | Network latency | `target` |

## Dashboard Setup

### Quick Start Dashboard

1. **Import Pre-built Dashboard**

```bash
# Copy dashboard configuration
cp dashboards/hf-carbon-tracking.json /var/lib/grafana/dashboards/

# Or import via Grafana UI
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboards/hf-carbon-tracking.json
```

2. **Access Dashboard**

- URL: http://localhost:3000
- Username: admin
- Password: admin123
- Dashboard: "HF Eco2AI Carbon Tracking"

### Custom Dashboard Panels

#### Carbon Emissions Panel

```json
{
  "title": "Total CO₂ Emissions",
  "type": "stat",
  "targets": [
    {
      "expr": "hf_training_co2_kg_total",
      "legendFormat": "{{ project_name }}"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "kg",
      "color": {
        "mode": "thresholds"
      },
      "thresholds": {
        "steps": [
          {"color": "green", "value": null},
          {"color": "yellow", "value": 10},
          {"color": "red", "value": 25}
        ]
      }
    }
  }
}
```

#### Energy Consumption Over Time

```json
{
  "title": "Energy Consumption Over Time",
  "type": "graph",
  "targets": [
    {
      "expr": "rate(hf_training_energy_kwh_total[5m])",
      "legendFormat": "{{ project_name }} - Rate"
    },
    {
      "expr": "hf_training_energy_kwh_total",
      "legendFormat": "{{ project_name }} - Total"
    }
  ],
  "yAxes": [
    {
      "label": "Energy (kWh)",
      "min": 0
    }
  ]
}
```

#### Training Efficiency Heatmap

```json
{
  "title": "Training Efficiency by Hour",
  "type": "heatmap",
  "targets": [
    {
      "expr": "avg_over_time(hf_training_samples_per_kwh[1h])",
      "format": "time_series",
      "intervalFactor": 1
    }
  ],
  "heatmap": {
    "xBucketSize": "1h",
    "yBucketSize": "100",
    "colorMode": "spectrum"
  }
}
```

### Real-time Monitoring Dashboard

```json
{
  "dashboard": {
    "title": "Real-time Carbon Monitoring",
    "refresh": "5s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "title": "Live CO₂ Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "rate(hf_training_co2_kg_total[1m]) * 60",
            "legendFormat": "kg CO₂/hour"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 20,
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 5},
                {"color": "red", "value": 10}
              ]
            }
          }
        }
      }
    ]
  }
}
```

## Alerting Configuration

### Alert Severity Levels

- **Critical**: Immediate action required (carbon budget exceeded, system failure)
- **Warning**: Attention needed (high consumption, inefficiency)
- **Info**: Informational (optimization opportunities, status updates)

### Carbon Emission Alerts

```yaml
# High carbon emissions
- alert: HighCarbonEmissions
  expr: hf_training_co2_kg_total > 20
  for: 2m
  labels:
    severity: warning
    category: carbon
  annotations:
    summary: "High carbon emissions detected"
    description: "Training has emitted {{ $value }}kg of CO₂"
    runbook_url: "https://docs.terragonlabs.com/runbooks/carbon"

# Carbon budget exceeded
- alert: CarbonBudgetExceeded
  expr: hf_training_co2_kg_total > hf_training_carbon_budget_kg
  for: 0s
  labels:
    severity: critical
    category: carbon
  annotations:
    summary: "Carbon budget exceeded"
    description: "Project {{ $labels.project_name }} exceeded carbon budget"
    action: "Stop training immediately"
```

### Notification Channels

#### Slack Integration

```yaml
slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#sustainability-alerts'
    title: 'Carbon Alert: {{ .GroupLabels.alertname }}'
    text: |
      {{ range .Alerts }}
      *Project:* {{ .Labels.project_name }}
      *CO₂ Emissions:* {{ .Labels.co2_kg }}kg
      *Description:* {{ .Annotations.description }}
      {{ end }}
    color: '{{ if eq .Status "firing" }}danger{{ else }}good{{ end }}'
```

#### Email Notifications

```yaml
email_configs:
  - to: 'sustainability@terragonlabs.com'
    subject: 'Carbon Alert: {{ .GroupLabels.alertname }}'
    body: |
      Carbon Alert Details:
      
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Project: {{ .Labels.project_name }}
      CO₂ Emissions: {{ .Labels.co2_kg }}kg
      Time: {{ .StartsAt }}
      
      {{ .Annotations.description }}
      {{ end }}
```

#### PagerDuty Integration

```yaml
pagerduty_configs:
  - routing_key: 'YOUR_PAGERDUTY_ROUTING_KEY'
    description: 'Critical Carbon Alert: {{ .GroupLabels.alertname }}'
    severity: '{{ .CommonLabels.severity }}'
    details:
      project: '{{ .CommonLabels.project_name }}'
      co2_emissions: '{{ .CommonLabels.co2_kg }}kg'
      energy_consumption: '{{ .CommonLabels.energy_kwh }}kWh'
```

## Observability Best Practices

### Monitoring Strategy

1. **Four Golden Signals for ML Training**
   - **Latency**: Training speed and step time
   - **Traffic**: Throughput and samples/second
   - **Errors**: Failed steps and monitoring errors
   - **Saturation**: Resource utilization (GPU, memory)

2. **Carbon-Specific Metrics**
   - **Total Emissions**: Cumulative CO₂ output
   - **Emission Rate**: Real-time carbon generation
   - **Efficiency**: Samples per kWh
   - **Grid Intensity**: Regional carbon factors

### Alerting Best Practices

1. **Alert Fatigue Prevention**
   - Use appropriate thresholds
   - Implement alert inhibition
   - Group related alerts
   - Set proper escalation paths

2. **Actionable Alerts**
   - Include runbook links
   - Provide context and impact
   - Suggest remediation steps
   - Link to relevant dashboards

### Dashboard Design

1. **Hierarchical Structure**
   - Overview dashboard for executives
   - Operational dashboard for ML engineers
   - Detailed dashboard for performance tuning

2. **Visual Design Principles**
   - Use appropriate chart types
   - Implement consistent color schemes
   - Include trend indicators
   - Show targets and thresholds

### Data Retention

```yaml
# Prometheus configuration
storage:
  tsdb:
    retention.time: 90d  # Keep detailed metrics for 90 days
    retention.size: 50GB # Limit storage size
    
# Long-term storage (optional)
remote_write:
  - url: "http://thanos:9090/api/v1/receive"
    queue_config:
      capacity: 10000
      max_samples_per_send: 1000
```

## Troubleshooting

### Common Issues

#### Metrics Not Appearing

1. **Check Prometheus targets**

```bash
# View target status
curl http://localhost:9090/api/v1/targets

# Check service discovery
curl http://localhost:9090/api/v1/config
```

2. **Verify plugin configuration**

```python
# Check if Prometheus export is enabled
from hf_eco2ai import CarbonConfig

config = CarbonConfig()
print(f"Prometheus enabled: {config.export_prometheus}")
print(f"Prometheus port: {config.prometheus_port}")
```

3. **Test metrics endpoint**

```bash
# Check if metrics are exposed
curl http://localhost:9091/metrics | grep hf_training
```

#### High Cardinality Issues

1. **Identify high-cardinality metrics**

```promql
# Check series count by metric
topk(10, count by (__name__)({__name__=~"hf_training.*"}))

# Identify problematic labels
topk(10, count by (project_name)({__name__=~"hf_training.*"}))
```

2. **Reduce cardinality**

```python
# Configure label limits
config = CarbonConfig(
    max_projects=100,
    aggregate_similar_models=True,
    label_cleanup_enabled=True
)
```

#### Missing Alerts

1. **Check alert rules**

```bash
# Validate alert rules
prometheus --check-config=false --check-rules=true --rules-file=alerts.yml

# View active alerts
curl http://localhost:9090/api/v1/alerts
```

2. **Test alert expressions**

```promql
# Test in Prometheus UI
hf_training_co2_kg_total > 20

# Check for data availability
count(hf_training_co2_kg_total)
```

### Performance Optimization

#### Reduce Monitoring Overhead

```python
# Optimize collection frequency
config = CarbonConfig(
    measurement_interval=10.0,  # Collect every 10 seconds
    export_batch_size=100,      # Batch metrics export
    compression_enabled=True    # Compress metrics data
)
```

#### Optimize Prometheus Configuration

```yaml
# Reduce resource usage
global:
  scrape_interval: 30s        # Increase scrape interval
  evaluation_interval: 30s    # Increase evaluation interval
  
scrape_configs:
  - job_name: 'hf-eco2ai'
    scrape_interval: 15s       # Specific interval for critical metrics
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'hf_training_debug_.*'
        action: drop            # Drop debug metrics in production
```

## Advanced Configuration

### Custom Metrics

```python
# Add custom business metrics
from prometheus_client import Gauge, Counter

# Custom sustainability score
sustainability_score = Gauge(
    'hf_training_sustainability_score',
    'Composite sustainability score',
    ['project_name', 'model_type']
)

# Carbon offset credits
carbon_offset_credits = Counter(
    'hf_training_carbon_offset_credits_total',
    'Total carbon offset credits purchased',
    ['project_name', 'credit_type']
)
```

### Federation Setup

```yaml
# Multi-cluster monitoring
scrape_configs:
  - job_name: 'federate'
    scrape_interval: 15s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{job=~"hf-eco2ai.*"}'
        - '{__name__=~"hf_training.*"}'
    static_configs:
      - targets:
        - 'prometheus-cluster-1:9090'
        - 'prometheus-cluster-2:9090'
```

### Integration with External Systems

#### Carbon Accounting Integration

```python
# Export to carbon accounting system
from hf_eco2ai.integrations import CarbonAccountingAPI

api = CarbonAccountingAPI(
    endpoint="https://carbon-api.company.com",
    api_key="your-api-key"
)

# Automatic carbon report submission
eco_callback = Eco2AICallback(
    config=config,
    integrations=[api]
)
```

#### Cost Management Integration

```python
# AWS Cost Explorer integration
from hf_eco2ai.integrations import AWSCostExplorer

cost_integration = AWSCostExplorer(
    access_key_id="your-access-key",
    secret_access_key="your-secret-key",
    region="us-east-1"
)

eco_callback = Eco2AICallback(
    integrations=[cost_integration]
)
```

## Support and Resources

- **Documentation**: https://hf-eco2ai.readthedocs.io/monitoring
- **Example Dashboards**: https://github.com/terragonlabs/hf-eco2ai-dashboards
- **Community Forum**: https://forum.terragonlabs.com/hf-eco2ai
- **Issue Tracker**: https://github.com/terragonlabs/hf-eco2ai-plugin/issues
