# Deployment Guide for HF Eco2AI Plugin

## Table of Contents

1. [Quick Start](#quick-start)
2. [Production Deployment](#production-deployment)
3. [Container Deployment](#container-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring Setup](#monitoring-setup)
6. [Security Configuration](#security-configuration)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Local Installation

```bash
# Install from PyPI
pip install hf-eco2ai-plugin

# Or install from source
git clone https://github.com/terragonlabs/hf-eco2ai-plugin.git
cd hf-eco2ai-plugin
pip install -e .

# Verify installation
python -c "import hf_eco2ai; print('Installation successful!')"
```

### Basic Usage

```python
from transformers import Trainer, TrainingArguments
from hf_eco2ai import Eco2AICallback

# Add to your existing training script
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[Eco2AICallback()]  # Just add this line!
)

trainer.train()
```

## Production Deployment

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended)
- 4GB+ RAM
- 10GB+ disk space

### Installation Steps

1. **Create virtual environment**

```bash
python -m venv hf-eco2ai-env
source hf-eco2ai-env/bin/activate  # On Windows: hf-eco2ai-env\Scripts\activate
```

2. **Install production dependencies**

```bash
pip install --upgrade pip
pip install hf-eco2ai-plugin[production]
```

3. **Configure environment**

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
vim .env  # Set your specific values
```

4. **Set up monitoring (optional)**

```bash
# Install monitoring components
pip install hf-eco2ai-plugin[monitoring]

# Start Prometheus and Grafana
docker-compose --profile monitoring up -d
```

### Configuration

Create a production configuration file:

```python
# config/production.py
from hf_eco2ai import CarbonConfig

config = CarbonConfig(
    project_name="production-training",
    country="USA",
    region="California",
    gpu_ids="auto",
    measurement_interval=5.0,
    export_prometheus=True,
    prometheus_port=9091,
    save_report=True,
    report_path="/app/reports/carbon_report.json",
    log_level="EPOCH"
)
```

### Service Configuration

Create a systemd service for background monitoring:

```ini
# /etc/systemd/system/hf-eco2ai.service
[Unit]
Description=HF Eco2AI Plugin Service
After=network.target

[Service]
Type=simple
User=hf-eco2ai
Group=hf-eco2ai
WorkingDirectory=/opt/hf-eco2ai
Environment=PATH=/opt/hf-eco2ai/venv/bin
EnvironmentFile=/opt/hf-eco2ai/.env
ExecStart=/opt/hf-eco2ai/venv/bin/python -m hf_eco2ai.service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable hf-eco2ai
sudo systemctl start hf-eco2ai
sudo systemctl status hf-eco2ai
```

## Container Deployment

### Docker

1. **Build image**

```bash
# Build production image
docker build --target production -t hf-eco2ai-plugin:latest .

# Or use pre-built image
docker pull ghcr.io/terragonlabs/hf-eco2ai-plugin:latest
```

2. **Run container**

```bash
docker run -d \
  --name hf-eco2ai \
  --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  -e CARBON_COUNTRY=USA \
  -e CARBON_REGION=California \
  hf-eco2ai-plugin:latest
```

### Docker Compose

1. **Production setup**

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  hf-eco2ai:
    image: ghcr.io/terragonlabs/hf-eco2ai-plugin:latest
    restart: unless-stopped
    environment:
      - CARBON_COUNTRY=USA
      - CARBON_REGION=California
      - PROMETHEUS_ENABLED=true
    volumes:
      - ./data:/app/data
      - ./reports:/app/reports
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus:v2.45.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana:10.0.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  prometheus-data:
  grafana-data:
```

2. **Deploy**

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes

1. **Deployment manifest**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hf-eco2ai-plugin
  labels:
    app: hf-eco2ai-plugin
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hf-eco2ai-plugin
  template:
    metadata:
      labels:
        app: hf-eco2ai-plugin
    spec:
      containers:
      - name: hf-eco2ai-plugin
        image: ghcr.io/terragonlabs/hf-eco2ai-plugin:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
        env:
        - name: CARBON_COUNTRY
          value: "USA"
        - name: CARBON_REGION
          value: "California"
        - name: PROMETHEUS_ENABLED
          value: "true"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: reports-volume
          mountPath: /app/reports
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: hf-eco2ai-data-pvc
      - name: reports-volume
        persistentVolumeClaim:
          claimName: hf-eco2ai-reports-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: hf-eco2ai-service
spec:
  selector:
    app: hf-eco2ai-plugin
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
```

2. **Deploy to Kubernetes**

```bash
kubectl apply -f k8s/
kubectl get pods -l app=hf-eco2ai-plugin
```

## Cloud Deployment

### AWS

#### ECS Deployment

1. **Create task definition**

```json
{
  "family": "hf-eco2ai-plugin",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "hf-eco2ai-plugin",
      "image": "ghcr.io/terragonlabs/hf-eco2ai-plugin:latest",
      "memory": 8192,
      "cpu": 2048,
      "essential": true,
      "environment": [
        {"name": "CARBON_COUNTRY", "value": "USA"},
        {"name": "CARBON_REGION", "value": "us-east-1"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hf-eco2ai-plugin",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

2. **Deploy with CDK/Terraform**

```typescript
// CDK example
import * as ecs from '@aws-cdk/aws-ecs';
import * as ec2 from '@aws-cdk/aws-ec2';

const cluster = new ecs.Cluster(this, 'HFEco2AICluster', {
  vpc: vpc,
  capacityProviders: ['FARGATE'],
});

const service = new ecs.FargateService(this, 'HFEco2AIService', {
  cluster: cluster,
  taskDefinition: taskDefinition,
  desiredCount: 1,
});
```

#### SageMaker Integration

```python
# sagemaker_training.py
import sagemaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='src',
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    framework_version='1.12.0',
    py_version='py38',
    hyperparameters={
        'enable_carbon_tracking': True,
        'carbon_country': 'USA',
        'carbon_region': 'us-east-1'
    }
)

estimator.fit({'training': training_data_path})
```

### Google Cloud Platform

#### GKE Deployment

```yaml
# gke-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hf-eco2ai-plugin
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hf-eco2ai-plugin
  template:
    metadata:
      labels:
        app: hf-eco2ai-plugin
    spec:
      containers:
      - name: hf-eco2ai-plugin
        image: gcr.io/project-id/hf-eco2ai-plugin:latest
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        env:
        - name: CARBON_COUNTRY
          value: "USA"
        - name: CARBON_REGION
          value: "us-central1"
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

#### Vertex AI Integration

```python
# vertex_ai_training.py
from google.cloud import aiplatform

job = aiplatform.CustomContainerTrainingJob(
    display_name="hf-eco2ai-training",
    container_uri="gcr.io/project-id/hf-eco2ai-plugin:latest",
    model_serving_container_image_uri="gcr.io/project-id/model:latest",
)

job.run(
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    environment_variables={
        "CARBON_COUNTRY": "USA",
        "CARBON_REGION": "us-central1"
    }
)
```

### Azure

#### Container Instances

```yaml
# azure-container-instance.yaml
apiVersion: 2019-12-01
location: eastus
name: hf-eco2ai-plugin
properties:
  containers:
  - name: hf-eco2ai
    properties:
      image: ghcr.io/terragonlabs/hf-eco2ai-plugin:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 8
      environmentVariables:
      - name: CARBON_COUNTRY
        value: USA
      - name: CARBON_REGION
        value: eastus
  osType: Linux
  restartPolicy: Always
type: Microsoft.ContainerInstance/containerGroups
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'hf-eco2ai'
    static_configs:
      - targets: ['localhost:9091']
    metrics_path: '/metrics'
    scrape_interval: 10s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "HF Eco2AI Carbon Tracking",
    "tags": ["carbon", "ml", "sustainability"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Total Energy Consumption",
        "type": "stat",
        "targets": [
          {
            "expr": "hf_training_energy_kwh_total",
            "refId": "A"
          }
        ]
      },
      {
        "id": 2,
        "title": "CO2 Emissions",
        "type": "stat",
        "targets": [
          {
            "expr": "hf_training_co2_kg_total",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# alert_rules.yml
groups:
- name: hf_eco2ai_alerts
  rules:
  - alert: HighCarbonEmissions
    expr: hf_training_co2_kg_total > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High carbon emissions detected"
      description: "Training has emitted {{ $value }}kg of CO2"

  - alert: ExcessiveEnergyConsumption
    expr: rate(hf_training_energy_kwh_total[5m]) > 5
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Excessive energy consumption"
      description: "Energy consumption rate is {{ $value }}kWh/hour"
```

## Security Configuration

### SSL/TLS Setup

```bash
# Generate self-signed certificate (for development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Or use Let's Encrypt for production
certbot certonly --standalone -d your-domain.com
```

### API Authentication

```python
# config/security.py
from hf_eco2ai import CarbonConfig

config = CarbonConfig(
    api_auth_enabled=True,
    api_auth_token="your-secure-token",
    prometheus_auth_enabled=True,
    ssl_enabled=True,
    ssl_cert_path="/path/to/cert.pem",
    ssl_key_path="/path/to/key.pem"
)
```

### Network Security

```yaml
# docker-compose with network security
version: '3.8'

services:
  hf-eco2ai:
    networks:
      - internal
    ports:
      - "127.0.0.1:8080:8080"  # Bind to localhost only

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    networks:
      - internal
      - external

networks:
  internal:
    driver: bridge
    internal: true
  external:
    driver: bridge
```

## Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check container GPU access
docker exec hf-eco2ai nvidia-smi
```

#### Permission Errors

```bash
# Fix file permissions
chown -R 1000:1000 /app/data /app/reports

# Fix Docker socket permissions
sudo usermod -aG docker $USER
```

#### Memory Issues

```bash
# Check memory usage
docker stats hf-eco2ai

# Increase container memory limit
docker run --memory="8g" hf-eco2ai-plugin:latest
```

### Logs and Debugging

```bash
# View container logs
docker logs hf-eco2ai -f

# Enable debug logging
docker run -e DEBUG_ENABLED=true hf-eco2ai-plugin:latest

# Access container shell
docker exec -it hf-eco2ai bash
```

### Performance Tuning

```python
# Optimize for performance
config = CarbonConfig(
    measurement_interval=10.0,  # Reduce measurement frequency
    log_level="EPOCH",          # Log less frequently
    per_gpu_metrics=False,      # Disable per-GPU details
    export_prometheus=False     # Disable if not needed
)
```

### Health Checks

```bash
# Check service health
curl http://localhost:8080/health

# Check metrics endpoint
curl http://localhost:9091/metrics

# Validate configuration
python -m hf_eco2ai.cli validate-config
```

## Support

For deployment issues:

1. Check the [troubleshooting guide](troubleshooting.md)
2. Search [GitHub issues](https://github.com/terragonlabs/hf-eco2ai-plugin/issues)
3. Join our [Discord community](https://discord.gg/green-ai)
4. Email support: hf-eco2ai@terragonlabs.com
