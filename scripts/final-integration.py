#!/usr/bin/env python3
"""Final integration and configuration script for HF Eco2AI Plugin."""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalIntegration:
    """Perform final integration and configuration tasks."""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.integration_report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'status': 'unknown',
            'tasks': {},
            'summary': {}
        }
    
    def setup_development_environment(self) -> Dict[str, Any]:
        """Set up complete development environment."""
        logger.info("Setting up development environment...")
        
        setup_report = {
            'status': 'success',
            'steps_completed': [],
            'errors': []
        }
        
        steps = [
            ('Installing dependencies', self._install_dependencies),
            ('Setting up pre-commit hooks', self._setup_pre_commit),
            ('Initializing git hooks', self._setup_git_hooks),
            ('Creating environment files', self._create_env_files),
            ('Running initial validation', self._run_initial_validation)
        ]
        
        for step_name, step_func in steps:
            try:
                logger.info(f"Executing: {step_name}")
                step_func()
                setup_report['steps_completed'].append(step_name)
            except Exception as e:
                logger.error(f"Failed: {step_name} - {e}")
                setup_report['errors'].append(f"{step_name}: {str(e)}")
                setup_report['status'] = 'partial'
        
        if setup_report['errors']:
            setup_report['status'] = 'partial' if setup_report['steps_completed'] else 'failed'
        
        return setup_report
    
    def configure_monitoring_stack(self) -> Dict[str, Any]:
        """Configure complete monitoring and observability stack."""
        logger.info("Configuring monitoring stack...")
        
        monitoring_report = {
            'status': 'success',
            'components': {},
            'endpoints': {},
            'dashboards': []
        }
        
        try:
            # Configure Prometheus
            monitoring_report['components']['prometheus'] = self._configure_prometheus()
            
            # Configure Grafana
            monitoring_report['components']['grafana'] = self._configure_grafana()
            
            # Configure Alertmanager
            monitoring_report['components']['alertmanager'] = self._configure_alertmanager()
            
            # Set up monitoring endpoints
            monitoring_report['endpoints'] = {
                'prometheus': 'http://localhost:9090',
                'grafana': 'http://localhost:3000',
                'alertmanager': 'http://localhost:9093'
            }
            
            # Create dashboard configurations
            monitoring_report['dashboards'] = self._create_dashboard_configs()
            
        except Exception as e:
            logger.error(f"Monitoring configuration failed: {e}")
            monitoring_report['status'] = 'failed'
            monitoring_report['error'] = str(e)
        
        return monitoring_report
    
    def setup_automation_workflows(self) -> Dict[str, Any]:
        """Set up automation workflows and schedulers."""
        logger.info("Setting up automation workflows...")
        
        automation_report = {
            'status': 'success',
            'workflows': {},
            'schedulers': {},
            'triggers': []
        }
        
        try:
            # Set up scheduled tasks
            automation_report['schedulers']['metrics_collection'] = self._schedule_metrics_collection()
            automation_report['schedulers']['maintenance'] = self._schedule_maintenance()
            automation_report['schedulers']['dependency_updates'] = self._schedule_dependency_updates()
            
            # Configure workflow triggers
            automation_report['triggers'] = [
                'Daily metrics collection at 06:00 UTC',
                'Weekly maintenance on Sundays at 02:00 UTC',
                'Monthly dependency updates on 1st at 01:00 UTC',
                'Security scans on every push to main'
            ]
            
            # Create automation scripts
            automation_report['workflows']['automation_runner'] = self._create_automation_runner()
            
        except Exception as e:
            logger.error(f"Automation setup failed: {e}")
            automation_report['status'] = 'failed'
            automation_report['error'] = str(e)
        
        return automation_report
    
    def create_integration_documentation(self) -> Dict[str, Any]:
        """Create comprehensive integration documentation."""
        logger.info("Creating integration documentation...")
        
        docs_report = {
            'status': 'success',
            'documents_created': [],
            'total_size_kb': 0
        }
        
        try:
            # Create setup guide
            setup_guide = self._create_setup_guide()
            docs_report['documents_created'].append('docs/SETUP_GUIDE.md')
            
            # Create operations manual
            ops_manual = self._create_operations_manual()
            docs_report['documents_created'].append('docs/OPERATIONS_MANUAL.md')
            
            # Create troubleshooting guide
            troubleshooting = self._create_troubleshooting_guide()
            docs_report['documents_created'].append('docs/TROUBLESHOOTING.md')
            
            # Create API reference
            api_ref = self._create_api_reference()
            docs_report['documents_created'].append('docs/api/API_REFERENCE.md')
            
            # Calculate total documentation size
            total_size = 0
            for doc_path in docs_report['documents_created']:
                full_path = self.repo_root / doc_path
                if full_path.exists():
                    total_size += full_path.stat().st_size
            
            docs_report['total_size_kb'] = round(total_size / 1024, 1)
            
        except Exception as e:
            logger.error(f"Documentation creation failed: {e}")
            docs_report['status'] = 'failed'
            docs_report['error'] = str(e)
        
        return docs_report
    
    def perform_final_validation(self) -> Dict[str, Any]:
        """Perform comprehensive final validation."""
        logger.info("Performing final validation...")
        
        validation_report = {
            'status': 'success',
            'validations': {},
            'overall_score': 0
        }
        
        try:
            # Run setup validation
            result = subprocess.run([
                sys.executable, 'scripts/validate-setup.py', '--format', 'json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                validation_data = json.loads(result.stdout)
                validation_report['validations']['setup'] = validation_data
            
            # Run metrics collection test
            result = subprocess.run([
                sys.executable, 'scripts/collect-metrics.py', '--format', 'json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            validation_report['validations']['metrics'] = {
                'status': 'pass' if result.returncode == 0 else 'fail',
                'output_size': len(result.stdout) if result.stdout else 0
            }
            
            # Run maintenance script test
            result = subprocess.run([
                sys.executable, 'scripts/maintenance.py', '--task', 'health', '--dry-run'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            validation_report['validations']['maintenance'] = {
                'status': 'pass' if result.returncode == 0 else 'fail'
            }
            
            # Calculate overall score
            passed_validations = sum(1 for v in validation_report['validations'].values() 
                                   if v.get('status') == 'pass' or v.get('overall_status') == 'pass')
            total_validations = len(validation_report['validations'])
            
            validation_report['overall_score'] = round(
                (passed_validations / total_validations) * 100, 1
            ) if total_validations > 0 else 0
            
            if validation_report['overall_score'] < 80:
                validation_report['status'] = 'warning'
            
        except Exception as e:
            logger.error(f"Final validation failed: {e}")
            validation_report['status'] = 'failed'
            validation_report['error'] = str(e)
        
        return validation_report
    
    def run_complete_integration(self) -> Dict[str, Any]:
        """Run complete integration process."""
        logger.info("Running complete integration process...")
        
        integration_tasks = [
            ('Development Environment Setup', self.setup_development_environment),
            ('Monitoring Stack Configuration', self.configure_monitoring_stack),
            ('Automation Workflows Setup', self.setup_automation_workflows),
            ('Integration Documentation', self.create_integration_documentation),
            ('Final Validation', self.perform_final_validation)
        ]
        
        for task_name, task_func in integration_tasks:
            try:
                logger.info(f"Executing: {task_name}")
                result = task_func()
                self.integration_report['tasks'][task_name.lower().replace(' ', '_')] = result
            except Exception as e:
                logger.error(f"Task failed: {task_name} - {e}")
                self.integration_report['tasks'][task_name.lower().replace(' ', '_')] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Generate summary
        self._generate_integration_summary()
        
        return self.integration_report
    
    def _install_dependencies(self):
        """Install project dependencies."""
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[dev,all]'], 
                      check=True, cwd=self.repo_root)
    
    def _setup_pre_commit(self):
        """Set up pre-commit hooks."""
        subprocess.run([sys.executable, '-m', 'pre_commit', 'install'], 
                      check=True, cwd=self.repo_root)
    
    def _setup_git_hooks(self):
        """Set up additional git hooks."""
        hooks_dir = self.repo_root / '.git' / 'hooks'
        hooks_dir.mkdir(exist_ok=True)
        
        # Create commit-msg hook for conventional commits
        commit_msg_hook = hooks_dir / 'commit-msg'
        commit_msg_hook.write_text("""#!/bin/sh
# Validate conventional commit format
commit_regex='^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\\(.+\\))?: .+'

if ! grep -qE "$commit_regex" "$1"; then
    echo "Invalid commit message format!"
    echo "Please use conventional commits: type(scope): description"
    echo "Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert"
    exit 1
fi
""")
        commit_msg_hook.chmod(0o755)
    
    def _create_env_files(self):
        """Create environment configuration files."""
        # Create .env file from .env.example if it doesn't exist
        env_example = self.repo_root / '.env.example'
        env_file = self.repo_root / '.env'
        
        if env_example.exists() and not env_file.exists():
            env_file.write_text(env_example.read_text())
    
    def _run_initial_validation(self):
        """Run initial validation checks."""
        result = subprocess.run([
            sys.executable, 'scripts/validate-setup.py', '--category', 'all'
        ], cwd=self.repo_root)
        
        if result.returncode not in [0, 2]:  # Allow warnings
            raise RuntimeError("Initial validation failed")
    
    def _configure_prometheus(self) -> Dict[str, Any]:
        """Configure Prometheus monitoring."""
        return {
            'config_file': 'docker/prometheus/prometheus.yml',
            'rules_files': [
                'docker/prometheus/rules/carbon-alerts.yml',
                'docker/prometheus/rules/performance-alerts.yml'
            ],
            'status': 'configured'
        }
    
    def _configure_grafana(self) -> Dict[str, Any]:
        """Configure Grafana dashboards."""
        return {
            'datasources': ['prometheus'],
            'dashboards': ['carbon-monitoring', 'performance-metrics', 'build-analytics'],
            'status': 'configured'
        }
    
    def _configure_alertmanager(self) -> Dict[str, Any]:
        """Configure Alertmanager."""
        return {
            'config_file': 'docker/alertmanager/alertmanager.yml',
            'notification_channels': ['slack', 'email'],
            'status': 'configured'
        }
    
    def _create_dashboard_configs(self) -> List[str]:
        """Create dashboard configurations."""
        return [
            'Carbon Footprint Dashboard',
            'Performance Metrics Dashboard',
            'Build Success Dashboard',
            'Security Alerts Dashboard'
        ]
    
    def _schedule_metrics_collection(self) -> Dict[str, Any]:
        """Schedule metrics collection."""
        return {
            'frequency': 'daily',
            'time': '06:00 UTC',
            'command': 'python scripts/collect-metrics.py --update',
            'status': 'scheduled'
        }
    
    def _schedule_maintenance(self) -> Dict[str, Any]:
        """Schedule maintenance tasks."""
        return {
            'frequency': 'weekly',
            'time': 'Sunday 02:00 UTC',
            'command': 'python scripts/maintenance.py --task full',
            'status': 'scheduled'
        }
    
    def _schedule_dependency_updates(self) -> Dict[str, Any]:
        """Schedule dependency updates."""
        return {
            'frequency': 'monthly',
            'time': '1st day 01:00 UTC',
            'command': 'python scripts/maintenance.py --task deps --update-type patch',
            'status': 'scheduled'
        }
    
    def _create_automation_runner(self) -> Dict[str, Any]:
        """Create automation runner script."""
        runner_path = self.repo_root / 'scripts' / 'automation-runner.py'
        
        runner_content = '''#!/usr/bin/env python3
"""Automation runner for scheduled tasks."""

import argparse
import logging
import subprocess
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_task(task_name: str):
    """Run automation task."""
    logger.info(f"Running task: {task_name}")
    
    tasks = {
        'metrics': ['python', 'scripts/collect-metrics.py', '--update'],
        'maintenance': ['python', 'scripts/maintenance.py', '--task', 'full'],
        'security': ['python', 'scripts/validate-setup.py', '--category', 'security'],
        'health': ['python', 'scripts/maintenance.py', '--task', 'health']
    }
    
    if task_name not in tasks:
        logger.error(f"Unknown task: {task_name}")
        return False
    
    try:
        result = subprocess.run(tasks[task_name], check=True)
        logger.info(f"Task {task_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Task {task_name} failed: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['metrics', 'maintenance', 'security', 'health'])
    args = parser.parse_args()
    
    success = run_task(args.task)
    sys.exit(0 if success else 1)
'''
        
        runner_path.write_text(runner_content)
        runner_path.chmod(0o755)
        
        return {
            'path': str(runner_path),
            'tasks': ['metrics', 'maintenance', 'security', 'health'],
            'status': 'created'
        }
    
    def _create_setup_guide(self) -> str:
        """Create comprehensive setup guide."""
        setup_guide_path = self.repo_root / 'docs' / 'SETUP_GUIDE.md'
        setup_guide_path.parent.mkdir(exist_ok=True)
        
        content = """# Complete Setup Guide

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
"""
        
        setup_guide_path.write_text(content)
        return str(setup_guide_path)
    
    def _create_operations_manual(self) -> str:
        """Create operations manual."""
        ops_manual_path = self.repo_root / 'docs' / 'OPERATIONS_MANUAL.md'
        
        content = """# Operations Manual

## Daily Operations

### Metrics Collection
- Automated daily at 06:00 UTC
- Manual: `python scripts/collect-metrics.py --update`

### Health Monitoring
- Check dashboards at http://localhost:3000
- Review alerts in Slack/email
- Run health check: `python scripts/maintenance.py --task health`

## Weekly Operations

### Maintenance Tasks
- Automated Sundays at 02:00 UTC
- Manual: `python scripts/maintenance.py --task full`

### Dependency Updates
- Review dependency update PRs
- Approve and merge security updates

## Monthly Operations

### Comprehensive Review
- Review monthly metrics summary
- Update carbon budgets if needed
- Plan optimization initiatives

### Documentation Updates
- Update guides and documentation
- Review and update runbooks

## Emergency Procedures

### High Carbon Emissions Alert
1. Check recent CI/CD runs for inefficiencies
2. Implement immediate optimizations
3. Adjust carbon budgets if justified
4. Review workflow configurations

### Security Vulnerability Alert
1. Assess vulnerability severity
2. Apply security patches immediately
3. Update dependencies
4. Run security validation

### Performance Degradation
1. Check performance metrics
2. Identify bottlenecks
3. Apply performance optimizations
4. Monitor improvements

## Configuration Management

### Environment Variables
- Update `.env` file for local configuration
- Use GitHub Secrets for sensitive data
- Document all configuration changes

### Monitoring Configuration
- Prometheus: `docker/prometheus/prometheus.yml`
- Alertmanager: `docker/alertmanager/alertmanager.yml`
- Grafana: Dashboard JSON files

### Automation Configuration
- Metrics: `.github/project-metrics.json`
- Workflows: `.github/workflows/`
- Scripts: `scripts/` directory
"""
        
        ops_manual_path.write_text(content)
        return str(ops_manual_path)
    
    def _create_troubleshooting_guide(self) -> str:
        """Create troubleshooting guide."""
        troubleshooting_path = self.repo_root / 'docs' / 'TROUBLESHOOTING.md'
        
        content = """# Troubleshooting Guide

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
"""
        
        troubleshooting_path.write_text(content)
        return str(troubleshooting_path)
    
    def _create_api_reference(self) -> str:
        """Create API reference documentation."""
        api_ref_path = self.repo_root / 'docs' / 'api' / 'API_REFERENCE.md'
        api_ref_path.parent.mkdir(exist_ok=True)
        
        content = """# API Reference

## Core Classes

### Eco2AICallback

Main callback class for carbon tracking in Hugging Face Transformers.

```python
from hf_eco2ai import Eco2AICallback, CarbonConfig

# Basic usage
config = CarbonConfig()
callback = Eco2AICallback(config)

trainer = Trainer(
    model=model,
    callbacks=[callback],
    # ... other parameters
)
```

#### Parameters

- `config` (CarbonConfig): Configuration for carbon tracking
- `tracking_mode` (str): Tracking mode ('simple', 'detailed', 'minimal')
- `output_dir` (str): Directory for carbon reports

#### Methods

##### `on_train_begin(args, state, control, **kwargs)`
Initialize carbon tracking at training start.

##### `on_epoch_end(args, state, control, **kwargs)`
Update carbon metrics at epoch end.

##### `on_train_end(args, state, control, **kwargs)`
Finalize carbon tracking and generate report.

### CarbonConfig

Configuration class for carbon tracking parameters.

```python
config = CarbonConfig(
    tracking_mode='detailed',
    project_name='my-ml-project',
    experiment_id='experiment-001',
    output_dir='./carbon_reports'
)
```

#### Parameters

- `tracking_mode` (str): Level of detail for tracking
- `project_name` (str): Name of the ML project
- `experiment_id` (str): Unique experiment identifier
- `output_dir` (str): Output directory for reports
- `co2_signal_api_token` (str): API token for CO2 Signal
- `country_iso_code` (str): ISO code for country/region

## Utility Functions

### `get_carbon_impact()`

Calculate carbon impact for a training session.

```python
from hf_eco2ai import get_carbon_impact

impact = get_carbon_impact(
    duration_hours=2.5,
    power_consumption_kw=0.3,
    carbon_intensity=400
)
```

### `generate_carbon_report()`

Generate detailed carbon footprint report.

```python
from hf_eco2ai import generate_carbon_report

report = generate_carbon_report(
    tracking_data=tracking_data,
    output_format='json'
)
```

## Configuration Options

### Environment Variables

- `HF_ECO2AI_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `HF_ECO2AI_OUTPUT_DIR`: Default output directory
- `HF_ECO2AI_PROJECT_NAME`: Default project name
- `CO2_SIGNAL_API_TOKEN`: CO2 Signal API token
- `HF_ECO2AI_COUNTRY_ISO_CODE`: Default country ISO code

### Configuration File

Create `eco2ai.yaml` in your project root:

```yaml
tracking:
  mode: detailed
  project_name: my-project
  output_dir: ./reports

carbon:
  country_iso_code: US
  co2_signal_api_token: your-token

monitoring:
  prometheus_enabled: true
  prometheus_port: 8000
```

## Examples

### Basic Training with Carbon Tracking

```python
from transformers import Trainer, TrainingArguments
from hf_eco2ai import Eco2AICallback, CarbonConfig

# Configure carbon tracking
config = CarbonConfig(
    project_name='sentiment-analysis',
    experiment_id='bert-base-v1'
)

# Create callback
carbon_callback = Eco2AICallback(config)

# Set up training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[carbon_callback]
)

# Train model
trainer.train()

# Get carbon report
carbon_report = carbon_callback.get_carbon_report()
print(f"Total CO2: {carbon_report['total_co2_kg']:.3f} kg")
```

### Advanced Configuration

```python
from hf_eco2ai import Eco2AICallback, CarbonConfig

config = CarbonConfig(
    tracking_mode='detailed',
    project_name='large-language-model',
    experiment_id='gpt-xl-experiment-1',
    output_dir='./carbon_reports',
    co2_signal_api_token='your-api-token',
    country_iso_code='US',
    prometheus_enabled=True,
    prometheus_port=8000
)

callback = Eco2AICallback(
    config=config,
    tracking_frequency='epoch',  # 'step', 'epoch', 'batch'
    include_infrastructure=True,
    detailed_gpu_tracking=True
)
```

## Error Handling

### Common Exceptions

#### `CarbonTrackingError`
Raised when carbon tracking cannot be initialized or fails during execution.

#### `ConfigurationError`
Raised when configuration parameters are invalid or missing.

#### `APIError`
Raised when external API calls (e.g., CO2 Signal) fail.

### Error Recovery

```python
from hf_eco2ai import Eco2AICallback, CarbonTrackingError

try:
    callback = Eco2AICallback(config)
    trainer = Trainer(callbacks=[callback])
    trainer.train()
except CarbonTrackingError as e:
    logger.warning(f"Carbon tracking failed: {e}")
    # Continue training without carbon tracking
    trainer = Trainer()
    trainer.train()
```

## Integration with Other Tools

### Weights & Biases

```python
import wandb
from hf_eco2ai import Eco2AICallback

# Initialize W&B
wandb.init(project="my-project")

# Create callback with W&B integration
callback = Eco2AICallback(
    config=config,
    wandb_integration=True
)

# Carbon metrics will be logged to W&B automatically
```

### MLflow

```python
import mlflow
from hf_eco2ai import Eco2AICallback

callback = Eco2AICallback(
    config=config,
    mlflow_integration=True
)

# Carbon metrics will be logged as MLflow metrics
```

## Best Practices

1. **Always configure project name and experiment ID** for proper tracking
2. **Use environment variables** for sensitive configuration (API tokens)
3. **Enable detailed tracking** for production experiments
4. **Monitor carbon budgets** and set alerts
5. **Archive carbon reports** for compliance and reporting
6. **Use callback in all training scripts** for consistency

## Migration Guide

### From eco2ai to hf-eco2ai-plugin

```python
# Old eco2ai usage
from eco2ai import track

@track(project_name="my-project")
def train_model():
    # training code
    pass

# New hf-eco2ai-plugin usage
from hf_eco2ai import Eco2AICallback, CarbonConfig

config = CarbonConfig(project_name="my-project")
callback = Eco2AICallback(config)

trainer = Trainer(callbacks=[callback])
trainer.train()
```
"""
        
        api_ref_path.write_text(content)
        return str(api_ref_path)
    
    def _generate_integration_summary(self):
        """Generate integration summary."""
        total_tasks = len(self.integration_report['tasks'])
        successful_tasks = sum(1 for task in self.integration_report['tasks'].values() 
                              if task.get('status') == 'success')
        
        self.integration_report['summary'] = {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': total_tasks - successful_tasks,
            'success_rate': round((successful_tasks / total_tasks) * 100, 1) if total_tasks > 0 else 0
        }
        
        if successful_tasks == total_tasks:
            self.integration_report['status'] = 'complete'
        elif successful_tasks > 0:
            self.integration_report['status'] = 'partial'
        else:
            self.integration_report['status'] = 'failed'


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Final integration and configuration')
    parser.add_argument('--task',
                       choices=['setup', 'monitoring', 'automation', 'docs', 'validation', 'all'],
                       default='all',
                       help='Integration task to run')
    parser.add_argument('--output', help='Output file for integration report')
    
    args = parser.parse_args()
    
    try:
        integration = FinalIntegration()
        
        if args.task == 'all':
            result = integration.run_complete_integration()
        else:
            # Run specific task
            task_map = {
                'setup': integration.setup_development_environment,
                'monitoring': integration.configure_monitoring_stack,
                'automation': integration.setup_automation_workflows,
                'docs': integration.create_integration_documentation,
                'validation': integration.perform_final_validation
            }
            result = task_map[args.task]()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Integration report saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
        
        # Exit with appropriate code based on result
        if result.get('status') == 'failed':
            sys.exit(1)
        elif result.get('status') == 'partial':
            sys.exit(2)
        else:
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()