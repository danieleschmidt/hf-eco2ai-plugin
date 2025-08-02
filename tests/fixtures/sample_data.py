"""Sample data fixtures for testing."""

from typing import Dict, List, Any
import tempfile
import json
from pathlib import Path


def create_sample_carbon_report() -> Dict[str, Any]:
    """Create a sample carbon report for testing."""
    return {
        "project_name": "test-training",
        "start_time": "2025-08-02T10:00:00Z",
        "end_time": "2025-08-02T13:30:00Z",
        "duration_seconds": 12600,
        "total_energy_kwh": 15.2,
        "total_co2_kg": 6.3,
        "grid_carbon_intensity": 415,
        "country": "USA",
        "region": "California",
        "gpu_info": {
            "gpu_count": 4,
            "gpu_models": ["NVIDIA RTX 4090"] * 4,
            "total_gpu_memory_gb": 96
        },
        "training_info": {
            "model_name": "bert-base-uncased",
            "dataset_size": 50000,
            "num_epochs": 3,
            "batch_size": 32,
            "learning_rate": 2e-5
        },
        "efficiency_metrics": {
            "samples_per_kwh": 3289,
            "co2_per_sample_mg": 126,
            "energy_per_epoch_kwh": 5.07,
            "time_per_epoch_minutes": 70
        },
        "environmental_impact": {
            "car_km_equivalent": 26.2,
            "trees_to_offset": 0.78,
            "carbon_cost_usd": 1.26
        }
    }


def create_sample_prometheus_metrics() -> List[str]:
    """Create sample Prometheus metrics for testing."""
    return [
        'hf_training_energy_kwh_total{project="test"} 15.2',
        'hf_training_co2_kg_total{project="test"} 6.3',
        'hf_training_gpu_power_watts{gpu_id="0"} 250.0',
        'hf_training_gpu_power_watts{gpu_id="1"} 240.0',
        'hf_training_samples_per_kwh{project="test"} 3289',
        'hf_training_efficiency_score{project="test"} 0.85'
    ]


def create_sample_training_config() -> Dict[str, Any]:
    """Create sample training configuration."""
    return {
        "model_name_or_path": "bert-base-uncased",
        "task_name": "text-classification",
        "do_train": True,
        "do_eval": True,
        "max_seq_length": 128,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "warmup_steps": 500,
        "logging_steps": 100,
        "eval_steps": 500,
        "save_steps": 1000,
        "output_dir": "./test_output",
        "overwrite_output_dir": True,
        "seed": 42
    }


def create_sample_dataset() -> List[Dict[str, Any]]:
    """Create sample dataset for testing."""
    return [
        {
            "text": "This is a positive example.",
            "label": 1,
            "input_ids": [101, 2023, 2003, 1037, 3893, 2742, 1012, 102],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1]
        },
        {
            "text": "This is a negative example.",
            "label": 0,
            "input_ids": [101, 2023, 2003, 1037, 4997, 2742, 1012, 102],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1]
        },
        {
            "text": "Another positive case.",
            "label": 1,
            "input_ids": [101, 2178, 3893, 2553, 1012, 102],
            "attention_mask": [1, 1, 1, 1, 1, 1]
        }
    ]


def create_temp_config_file(config: Dict[str, Any]) -> str:
    """Create a temporary configuration file."""
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    )
    json.dump(config, temp_file, indent=2)
    temp_file.close()
    return temp_file.name


def create_temp_dataset_file(dataset: List[Dict[str, Any]]) -> str:
    """Create a temporary dataset file."""
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.jsonl', delete=False
    )
    for item in dataset:
        json.dump(item, temp_file)
        temp_file.write('\n')
    temp_file.close()
    return temp_file.name


class MockGPUMonitor:
    """Mock GPU monitor for testing."""
    
    def __init__(self, gpu_count: int = 2):
        self.gpu_count = gpu_count
        self.monitoring = False
        self.power_readings = [250.0, 240.0, 230.0, 220.0][:gpu_count]
    
    def start_monitoring(self):
        """Start GPU monitoring."""
        self.monitoring = True
    
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.monitoring = False
    
    def get_power_draw(self, gpu_id: int) -> float:
        """Get power draw for specific GPU."""
        if 0 <= gpu_id < self.gpu_count:
            return self.power_readings[gpu_id]
        return 0.0
    
    def get_memory_usage(self, gpu_id: int) -> Dict[str, int]:
        """Get memory usage for specific GPU."""
        return {
            "used": 8192 - (gpu_id * 512),
            "total": 24576,
            "free": 16384 + (gpu_id * 512)
        }
    
    def get_temperature(self, gpu_id: int) -> int:
        """Get temperature for specific GPU."""
        return 75 - (gpu_id * 2)


class MockCarbonAPI:
    """Mock carbon intensity API for testing."""
    
    def __init__(self):
        self.carbon_data = {
            "USA": {"California": 411, "Texas": 465, "New York": 289},
            "Germany": {"Bavaria": 411, "Berlin": 398},
            "France": {"Île-de-France": 58, "Provence": 89},
        }
    
    def get_carbon_intensity(self, country: str, region: str = None) -> float:
        """Get carbon intensity for location."""
        if country in self.carbon_data:
            if region and region in self.carbon_data[country]:
                return self.carbon_data[country][region]
            # Return average for country if no region specified
            return sum(self.carbon_data[country].values()) / len(self.carbon_data[country])
        return 500.0  # Default global average
    
    def get_real_time_intensity(self, latitude: float, longitude: float) -> float:
        """Get real-time carbon intensity for coordinates."""
        # Simple mock based on latitude (northern regions tend to be cleaner)
        if latitude > 50:  # Northern regions
            return 200.0
        elif latitude > 30:  # Temperate regions
            return 400.0
        else:  # Tropical/southern regions
            return 600.0


class MockPrometheusClient:
    """Mock Prometheus client for testing."""
    
    def __init__(self):
        self.metrics = {}
        self.pushgateway_url = None
    
    def set_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a metric value."""
        key = f"{name}_{labels or {}}"
        self.metrics[key] = value
    
    def push_to_gateway(self, gateway_url: str, job: str):
        """Push metrics to Prometheus pushgateway."""
        self.pushgateway_url = gateway_url
        return True
    
    def get_metric_value(self, name: str, labels: Dict[str, str] = None) -> float:
        """Get a metric value for testing."""
        key = f"{name}_{labels or {}}"
        return self.metrics.get(key, 0.0)
