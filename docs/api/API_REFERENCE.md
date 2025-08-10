# API Reference

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
