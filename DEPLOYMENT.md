# Deployment Guide

This guide covers deployment strategies and release automation for the HF Eco2AI Plugin.

## Release Process

### Automated Release Pipeline

1. **Version Tagging**
   ```bash
   # Create and push a version tag
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```

2. **Automated Steps** (via GitHub Actions)
   - Build and test the package
   - Generate changelog from git history
   - Create GitHub release with assets  
   - Publish to PyPI
   - Build and push Docker images
   - Update documentation

### Manual Release Steps

If automated release fails, follow these manual steps:

1. **Build Package**
   ```bash
   make clean
   make build
   twine check dist/*
   ```

2. **Test Package**
   ```bash
   # Test in clean environment
   python -m venv test_env
   source test_env/bin/activate
   pip install dist/*.whl
   python -c "import hf_eco2ai; print('OK')"
   deactivate && rm -rf test_env
   ```

3. **Publish to PyPI**
   ```bash
   twine upload dist/*
   ```

## Deployment Environments

### Development Environment

```bash
# Using Docker Compose
docker-compose up dev

# Or local installation
make install-dev
make test
```

### Testing Environment

```bash
# Run comprehensive tests
docker-compose --profile testing up test

# Or with tox
tox -e py310,py311,py312,coverage,lint
```

### Production Environment

```bash
# Production container
docker-compose up hf-eco2ai

# Or PyPI installation
pip install hf-eco2ai-plugin
```

## Container Deployment

### Basic Container Usage

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/terragonlabs/hf-eco2ai-plugin:latest

# Run container
docker run -it --rm ghcr.io/terragonlabs/hf-eco2ai-plugin:latest
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hf-eco2ai-plugin
spec:
  replicas: 3
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
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
```

### Docker Swarm Deployment

```yaml
# docker-stack.yml
version: '3.8'
services:
  hf-eco2ai:
    image: ghcr.io/terragonlabs/hf-eco2ai-plugin:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    networks:
      - eco2ai-network

networks:
  eco2ai-network:
    driver: overlay
```

## Monitoring and Observability

### Prometheus Metrics

The plugin exposes metrics on port 9091 (configurable):

```python
from hf_eco2ai import Eco2AICallback

callback = Eco2AICallback(
    export_prometheus=True,
    prometheus_port=9091
)
```

### Grafana Dashboard

Import the provided dashboard:

```bash
# Using Grafana CLI
grafana-cli plugins install grafana-piechart-panel
curl -X POST http://admin:admin123@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboards/hf-carbon-tracking.json
```

### Log Aggregation

Configure log forwarding for centralized monitoring:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/carbon-tracking.log'),
        logging.StreamHandler()
    ]
)
```

## Security Considerations

### Container Security

- Use non-root user (appuser:1000)
- Read-only root filesystem where possible
- Minimal base image (python:3.10-slim)
- Regular security scanning with Trivy/Snyk

### Network Security

- Expose only necessary ports
- Use TLS for external communications
- Implement proper firewall rules
- Network policies in Kubernetes

### Secrets Management

```bash
# Using Docker secrets
echo "your-api-key" | docker secret create eco2ai_api_key -

# Using Kubernetes secrets
kubectl create secret generic eco2ai-secrets \
  --from-literal=api-key=your-api-key
```

## Scaling Considerations

### Horizontal Scaling

- Stateless design allows easy horizontal scaling
- Use load balancer for multiple instances
- Consider resource limits and requests

### Performance Optimization

- Enable GPU support for ML workloads
- Use appropriate resource allocations
- Monitor memory usage and optimize

### Data Persistence

```yaml
# Persistent volume for carbon reports
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: carbon-reports-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

## Rollback Procedures

### Quick Rollback

```bash
# Docker rollback
docker-compose down
docker-compose up -d --scale hf-eco2ai=3

# Kubernetes rollback
kubectl rollout undo deployment/hf-eco2ai-plugin
kubectl rollout status deployment/hf-eco2ai-plugin
```

### PyPI Package Rollback

```bash
# Install previous version
pip install hf-eco2ai-plugin==0.1.0

# Or use version constraints
pip install "hf-eco2ai-plugin<0.2.0"
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade hf-eco2ai-plugin
   python -c "import hf_eco2ai; print(hf_eco2ai.__version__)"
   ```

2. **GPU Detection Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   nvidia-smi
   ```

3. **Permission Issues**
   ```bash
   # Fix container permissions
   docker run --user $(id -u):$(id -g) ...
   ```

### Health Checks

```bash
# Container health check
docker exec container_name python -c "import hf_eco2ai; print('OK')"

# Kubernetes health check
kubectl get pods -l app=hf-eco2ai-plugin
kubectl logs -l app=hf-eco2ai-plugin
```

## Support and Maintenance

### Regular Maintenance

- Weekly dependency updates
- Monthly security scans
- Quarterly performance reviews
- Annual architecture reviews

### Support Channels

- GitHub Issues: Bug reports and feature requests
- Documentation: Comprehensive guides and examples
- Community: Discord server for discussions
- Professional: Enterprise support available

### Monitoring Metrics

Track these key metrics:
- Deployment success rate
- Container startup time
- Memory and CPU usage
- Error rates and response times
- Security scan results