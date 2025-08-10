# Troubleshooting Guide

## Common Issues

### Setup Issues

#### Import Error: No module named 'hf_eco2ai'
**Solution:**
```bash
pip install -e .[dev,all]
```

#### Pre-commit hooks failing
**Solution:**
```bash
pre-commit clean
pre-commit install
pre-commit run --all-files
```

### Monitoring Issues

#### Prometheus not starting
**Possible causes:**
- Port 9090 already in use
- Configuration syntax error

**Solution:**
```bash
docker-compose logs prometheus
docker-compose restart prometheus
```

#### Grafana dashboards not loading
**Solution:**
```bash
# Reset Grafana data
docker-compose down
docker volume rm hf-eco2ai-plugin_grafana-data
docker-compose up -d
```

### Metrics Collection Issues

#### GitHub API rate limiting
**Solution:**
- Set GITHUB_TOKEN environment variable
- Reduce collection frequency

#### Carbon tracking data missing
**Solution:**
- Check for carbon_report.json files
- Verify eco2ai library installation
- Run training with carbon tracking enabled

### CI/CD Issues

#### Workflow not triggering
**Possible causes:**
- Workflow file syntax error
- Trigger conditions not met
- GitHub permissions

**Solution:**
```bash
# Validate workflow syntax
yamllint .github/workflows/*.yml

# Check trigger conditions
git log --oneline -5
```

#### Test failures
**Solution:**
```bash
# Run tests locally
pytest tests/ -v

# Check test coverage
pytest --cov=hf_eco2ai tests/

# Debug specific test
pytest tests/unit/test_callback.py::TestEco2AICallback::test_init -v -s
```

## Performance Issues

### Slow CI/CD builds
**Optimizations:**
- Use dependency caching
- Reduce test matrix
- Optimize Docker layers
- Use faster runners

### High memory usage
**Solutions:**
- Monitor process memory with psutil
- Implement memory cleanup in callbacks
- Reduce batch sizes for testing

## Getting Help

### Documentation
- [Setup Guide](SETUP_GUIDE.md)
- [Operations Manual](OPERATIONS_MANUAL.md)
- [API Reference](api/API_REFERENCE.md)

### Support Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Questions and community support
- Slack: Real-time team communication

### Debugging Tools

#### Validation Script
```bash
python scripts/validate-setup.py --category all --format summary
```

#### Health Check
```bash
python scripts/maintenance.py --task health --output health-report.json
```

#### Metrics Analysis
```bash
python scripts/collect-metrics.py --format summary --output metrics.txt
```
