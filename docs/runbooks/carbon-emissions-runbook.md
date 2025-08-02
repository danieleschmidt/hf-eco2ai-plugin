# Carbon Emissions Incident Response Runbook

## Alert: Critical Carbon Emissions

### Severity: Critical
### Response Time: Immediate (< 5 minutes)

## Overview

This runbook provides step-by-step procedures for responding to critical carbon emissions alerts from ML training workloads.

## Alert Triggers

- **Alert**: `CriticalCarbonEmissions`
- **Condition**: `hf_training_co2_kg_total > 50`
- **Duration**: 1 minute
- **Impact**: High environmental impact, potential regulatory issues

## Immediate Response (0-5 minutes)

### 1. Assess Alert Severity

```bash
# Check current carbon emissions
curl -s "http://prometheus:9090/api/v1/query?query=hf_training_co2_kg_total" | jq '.data.result[0].value[1]'

# Check emission rate
curl -s "http://prometheus:9090/api/v1/query?query=rate(hf_training_co2_kg_total[5m])*3600" | jq '.data.result[0].value[1]'
```

### 2. Identify Affected Training Jobs

```bash
# List all active training projects
curl -s "http://prometheus:9090/api/v1/label/project_name/values" | jq '.data[]'

# Find highest emitting projects
curl -s "http://prometheus:9090/api/v1/query?query=topk(5, hf_training_co2_kg_total)" | jq '.data.result[]'
```

### 3. Check Carbon Budget Status

```bash
# Check budget vs actual emissions
curl -s "http://prometheus:9090/api/v1/query?query=hf_training_co2_kg_total - hf_training_carbon_budget_kg" | jq '.data.result[]'
```

## Investigation (5-15 minutes)

### 4. Analyze Root Cause

#### Check Training Parameters

```bash
# Review training configuration
kubectl get configmap training-config -o yaml

# Check model size and complexity
kubectl logs deployment/training-job | grep -E "(model|parameters|batch_size)"
```

#### Examine Resource Usage

```bash
# GPU utilization
curl -s "http://prometheus:9090/api/v1/query?query=hf_training_gpu_utilization_percent" | jq '.data.result[]'

# Power consumption
curl -s "http://prometheus:9090/api/v1/query?query=hf_training_gpu_power_watts" | jq '.data.result[]'
```

#### Review Grid Carbon Intensity

```bash
# Current grid carbon intensity
curl -s "http://prometheus:9090/api/v1/query?query=hf_training_grid_carbon_intensity_g_per_kwh" | jq '.data.result[0].value[1]'

# Historical comparison
curl -s "http://prometheus:9090/api/v1/query_range?query=hf_training_grid_carbon_intensity_g_per_kwh&start=$(date -d '24 hours ago' -u +%s)&end=$(date -u +%s)&step=3600" | jq '.data.result[0].values[]'
```

### 5. Assess Training Progress

```bash
# Check training loss and convergence
curl -s "http://prometheus:9090/api/v1/query?query=hf_training_loss" | jq '.data.result[]'

# Training efficiency
curl -s "http://prometheus:9090/api/v1/query?query=hf_training_samples_per_kwh" | jq '.data.result[]'
```

## Mitigation Strategies

### Option 1: Immediate Training Suspension (Critical Cases)

```bash
# Stop training job
kubectl scale deployment/training-job --replicas=0

# Verify suspension
kubectl get pods -l app=training-job

# Document reason
kubectl annotate deployment/training-job emergency-stop="carbon-emissions-critical-$(date -u +%s)"
```

### Option 2: Parameter Optimization (High Cases)

#### Reduce Batch Size

```yaml
# Update training configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config
data:
  batch_size: "16"  # Reduced from 32
  gradient_accumulation_steps: "4"  # Compensate for smaller batch
```

#### Enable Mixed Precision

```yaml
# Enable FP16 training
data:
  fp16: "true"
  fp16_opt_level: "O1"
```

#### Implement Gradient Checkpointing

```yaml
# Reduce memory usage
data:
  gradient_checkpointing: "true"
  dataloader_num_workers: "2"  # Reduce CPU usage
```

### Option 3: Infrastructure Changes

#### Scale Down Resources

```bash
# Reduce GPU allocation
kubectl patch deployment training-job -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "trainer",
          "resources": {
            "limits": {
              "nvidia.com/gpu": "2"  # Reduced from 4
            }
          }
        }]
      }
    }
  }
}'
```

#### Migrate to Lower Carbon Region

```bash
# Check available regions with lower carbon intensity
curl -s "https://api.carbonintensity.org.uk/regional" | jq '.data[] | select(.intensity.actual < 300)'

# Initiate migration (example with kubectl)
kubectl patch deployment training-job -p '{
  "spec": {
    "template": {
      "spec": {
        "nodeSelector": {
          "zone": "us-west1-a"  # Lower carbon intensity zone
        }
      }
    }
  }
}'
```

## Recovery and Optimization

### 6. Implement Carbon Budget Controls

```python
# Add carbon budget callback
from hf_eco2ai import CarbonBudgetCallback

budget_callback = CarbonBudgetCallback(
    max_co2_kg=30.0,  # Reduced budget
    action="stop",
    check_frequency=50,  # More frequent checks
    grace_period=5
)

trainer.add_callback(budget_callback)
```

### 7. Enable Real-time Monitoring

```python
# Enhanced monitoring configuration
from hf_eco2ai import CarbonConfig

config = CarbonConfig(
    measurement_interval=2.0,  # More frequent measurements
    alert_thresholds={
        "co2_rate_kg_per_hour": 5.0,
        "energy_rate_kwh_per_hour": 15.0
    },
    auto_optimization=True
)
```

### 8. Schedule Training During Low-Carbon Hours

```python
# Use carbon-aware scheduling
from hf_eco2ai.scheduling import LowCarbonScheduler

scheduler = LowCarbonScheduler(
    region="us-west1",
    max_carbon_intensity=300,  # g CO2/kWh
    flexibility_hours=8
)

# Find optimal training window
best_time = scheduler.find_low_carbon_window(
    duration_hours=6,
    start_after=datetime.now()
)

print(f"Schedule training at {best_time} for {scheduler.carbon_reduction:.1%} less CO2")
```

## Post-Incident Actions

### 9. Document Incident

```bash
# Create incident report
cat > incident-report-$(date +%Y%m%d-%H%M).md << EOF
# Carbon Emissions Incident Report

**Date**: $(date)
**Alert**: CriticalCarbonEmissions
**Peak CO2**: $(curl -s "http://prometheus:9090/api/v1/query?query=max_over_time(hf_training_co2_kg_total[1h])" | jq -r '.data.result[0].value[1]')kg
**Duration**: [TO BE FILLED]
**Projects Affected**: [TO BE FILLED]

## Root Cause
[TO BE FILLED]

## Actions Taken
- [ ] Training suspended
- [ ] Parameters optimized
- [ ] Budget controls implemented
- [ ] Monitoring enhanced

## Lessons Learned
[TO BE FILLED]

## Prevention Measures
[TO BE FILLED]
EOF
```

### 10. Update Monitoring and Alerts

```yaml
# Enhanced alert rules
- alert: CarbonEmissionsWarning
  expr: hf_training_co2_kg_total > 25  # Lower threshold
  for: 30s  # Faster detection
  labels:
    severity: warning
  annotations:
    summary: "Carbon emissions approaching critical levels"
    action: "Review training parameters and consider optimization"

- alert: CarbonRateHigh
  expr: rate(hf_training_co2_kg_total[1m]) * 3600 > 8  # kg/hour
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "High carbon emission rate detected"
    recommendation: "Consider reducing training intensity"
```

### 11. Implement Preventive Measures

#### Carbon Budget Enforcement

```yaml
# Kubernetes resource quota with carbon limits
apiVersion: v1
kind: ResourceQuota
metadata:
  name: carbon-quota
spec:
  hard:
    limits.carbon.terragonlabs.com/co2-kg: "50"  # 50kg CO2 limit
    limits.carbon.terragonlabs.com/energy-kwh: "100"  # 100kWh limit
```

#### Automatic Training Optimization

```python
# Auto-optimization callback
from hf_eco2ai.optimization import AutoCarbonOptimizer

optimizer = AutoCarbonOptimizer(
    target_co2_rate=5.0,  # kg/hour
    optimization_strategies=[
        "batch_size_reduction",
        "mixed_precision",
        "gradient_checkpointing",
        "learning_rate_adjustment"
    ],
    max_performance_loss=0.05  # 5% max performance degradation
)

trainer.add_callback(optimizer)
```

## Communication Templates

### Internal Notification

```
SUBJECT: [URGENT] Critical Carbon Emissions Alert - Immediate Action Required

Team,

We have detected critical carbon emissions from our ML training workloads:

• Current CO2 Emissions: [VALUE] kg
• Emission Rate: [VALUE] kg/hour
• Projects Affected: [LIST]
• Carbon Budget Status: [EXCEEDED/APPROACHING]

Immediate actions taken:
• [LIST ACTIONS]

Next steps:
• [LIST NEXT STEPS]

Please review your training configurations and implement carbon optimization measures.

Dashboard: http://grafana:3000/d/carbon-tracking
Runbook: https://docs.terragonlabs.com/runbooks/carbon-emissions

Sustainability Team
```

### Executive Summary

```
SUBJECT: Carbon Emissions Incident Summary - [DATE]

Executive Summary:

We experienced a critical carbon emissions event on [DATE] during ML training operations. Peak emissions reached [VALUE] kg CO2, exceeding our sustainability targets.

Business Impact:
• Environmental: [IMPACT]
• Regulatory: [RISK ASSESSMENT]
• Operational: [DOWNTIME/DELAYS]
• Financial: [COST IMPLICATIONS]

Resolution:
• Incident duration: [TIME]
• Root cause: [SUMMARY]
• Mitigation: [ACTIONS TAKEN]

Prevention:
• Enhanced monitoring implemented
• Stricter carbon budgets enforced
• Team training scheduled

Next Review: [DATE]

Sustainability Officer
```

## Escalation Procedures

### Level 1: Team Lead (0-5 minutes)
- Initial assessment and immediate mitigation
- Decision on training suspension
- Implementation of quick fixes

### Level 2: Engineering Manager (5-15 minutes)
- Resource reallocation decisions
- Infrastructure changes approval
- Cross-team coordination

### Level 3: Sustainability Officer (15-30 minutes)
- Regulatory compliance assessment
- External communication decisions
- Long-term strategy adjustments

### Level 4: Executive Team (30+ minutes)
- Business impact assessment
- Policy changes
- External stakeholder communication

## Tools and Resources

- **Grafana Dashboard**: http://grafana:3000/d/carbon-tracking
- **Prometheus Queries**: http://prometheus:9090/graph
- **Alert Manager**: http://alertmanager:9093
- **Carbon API**: https://api.carbonintensity.org.uk
- **Documentation**: https://docs.terragonlabs.com/sustainability
- **Incident Template**: https://templates.terragonlabs.com/carbon-incident

## Contact Information

- **On-call Engineer**: +1-555-ONCALL
- **Sustainability Team**: sustainability@terragonlabs.com
- **Engineering Manager**: engineering-mgr@terragonlabs.com
- **Emergency Escalation**: +1-555-EMERGENCY

---

**Last Updated**: 2025-08-02  
**Version**: 1.0  
**Next Review**: 2025-11-02
