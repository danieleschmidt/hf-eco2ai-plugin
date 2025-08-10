# HF Eco2AI Plugin - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the HF Eco2AI Plugin in production environments.

## Prerequisites

### Required Software
- Docker >= 20.10
- Kubernetes >= 1.24
- Helm >= 3.8
- Terraform >= 1.4 (for infrastructure provisioning)

### Required Credentials
- Container registry access
- Cloud provider credentials (AWS/GCP/Azure)
- Kubernetes cluster access

## Quick Start

### 1. Container Deployment

```bash
# Build and run with Docker Compose
cd deployment/containers
docker-compose up -d

# Check service status
docker-compose ps
docker-compose logs hf-eco2ai
```

### 2. Kubernetes Deployment

```bash
# Apply Kubernetes manifests
cd deployment/kubernetes
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n hf-eco2ai
kubectl logs -f deployment/hf-eco2ai -n hf-eco2ai
```

### 3. Helm Deployment

```bash
# Deploy using Helm chart
cd deployment/helm
helm upgrade --install hf-eco2ai ./hf-eco2ai \
  --namespace hf-eco2ai \
  --create-namespace \
  --values values-production.yaml

# Check deployment
helm status hf-eco2ai -n hf-eco2ai
```

## Infrastructure Provisioning

### AWS Deployment with Terraform

```bash
cd deployment/terraform

# Initialize Terraform
terraform init

# Plan infrastructure changes
terraform plan -var="environment=production"

# Apply infrastructure
terraform apply -var="environment=production"

# Get outputs
terraform output
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| HF_ECO2AI_ENV | Environment (development/staging/production) | Yes | production |
| HF_ECO2AI_LOG_LEVEL | Logging level | No | INFO |
| HF_ECO2AI_CONFIG_PATH | Configuration file path | No | /app/config |

### Configuration Files

Production configuration is located in `deployment/config/production.json`.

Key configuration sections:
- **Logging**: Structured JSON logging to files and stdout
- **Monitoring**: Prometheus metrics and health checks
- **Carbon Tracking**: Real-time grid carbon intensity data
- **Performance**: Multi-threading and caching optimizations
- **Security**: Encryption, audit logging, and rate limiting

## Monitoring and Observability

### Health Checks

The application exposes the following health endpoints:
- `/health` - Application health status
- `/ready` - Readiness for traffic
- `/metrics` - Prometheus metrics

### Monitoring Stack

The deployment includes:
- **Prometheus** - Metrics collection
- **Grafana** - Visualization dashboards
- **Alertmanager** - Alert management

Access dashboards:
- Grafana: http://localhost:3000 (admin/eco2ai_admin)
- Prometheus: http://localhost:9090

## Security Considerations

### Container Security
- Non-root user execution
- Read-only root filesystem
- Minimal attack surface
- Security scanning in CI/CD

### Network Security
- TLS encryption for all external communication
- Network policies for pod-to-pod communication
- Ingress with SSL termination

### Data Security
- Encryption at rest and in transit
- Secret management with Kubernetes secrets
- Regular security audits

## Scaling and Performance

### Horizontal Pod Autoscaling

```yaml
# Automatically scale based on CPU/memory usage
minReplicas: 2
maxReplicas: 10
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80
```

### Performance Optimizations
- Multi-threaded processing
- Intelligent caching with TTL
- Batch processing for large datasets
- Asynchronous operations

## Troubleshooting

### Common Issues

1. **Pod startup failures**
   ```bash
   kubectl describe pod <pod-name> -n hf-eco2ai
   kubectl logs <pod-name> -n hf-eco2ai
   ```

2. **Configuration issues**
   ```bash
   kubectl get configmap hf-eco2ai-config -n hf-eco2ai -o yaml
   ```

3. **Network connectivity**
   ```bash
   kubectl port-forward service/hf-eco2ai-service 8080:80 -n hf-eco2ai
   curl http://localhost:8080/health
   ```

### Support

For additional support:
- GitHub Issues: https://github.com/terragonlabs/hf-eco2ai-plugin/issues
- Documentation: https://hf-eco2ai.readthedocs.io
- Email: daniel@terragonlabs.com

## License

MIT License - see LICENSE file for details.
