"""Pytest configuration and fixtures for HF Eco2AI Plugin tests."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any


@pytest.fixture
def mock_eco2ai():
    """Mock Eco2AI tracker."""
    mock_tracker = Mock()
    mock_tracker.start.return_value = None
    mock_tracker.stop.return_value = None
    mock_tracker.get_info.return_value = {
        "duration": 3600,
        "kWh": 2.5,
        "CO2": 1.2,
        "coal_equivalent": 0.5,
        "tree_equivalent": 2.3,
    }
    return mock_tracker


@pytest.fixture
def mock_trainer():
    """Mock Hugging Face Trainer."""
    trainer = Mock()
    trainer.model = Mock()
    trainer.args = Mock()
    trainer.args.output_dir = "/tmp/test_output"
    trainer.train_dataset = Mock()
    trainer.eval_dataset = Mock()
    return trainer


@pytest.fixture
def sample_training_logs() -> Dict[str, Any]:
    """Sample training logs for testing."""
    return {
        "epoch": 1.0,
        "train_loss": 0.5,
        "eval_loss": 0.45,
        "learning_rate": 5e-5,
        "step": 100,
    }


@pytest.fixture
def mock_gpu_stats():
    """Mock GPU statistics."""
    return {
        "gpu_count": 2,
        "gpu_0_power": 250.0,
        "gpu_1_power": 240.0,
        "gpu_0_memory_used": 8192,
        "gpu_1_memory_used": 7680,
        "gpu_0_temperature": 75,
        "gpu_1_temperature": 73,
    }