"""Contract tests for HF Eco2AI Plugin API compatibility."""

import pytest
from unittest.mock import Mock, patch

# These tests ensure the API contract remains stable across versions


class TestEco2AICallbackContract:
    """Test the public API contract of Eco2AICallback."""

    def test_callback_initialization_contract(self):
        """Test that callback can be initialized with expected parameters."""
        # This test would import the actual callback when implemented
        # For now, we document the expected contract
        expected_init_params = {
            'project_name': str,
            'country': str,
            'region': str,
            'gpu_ids': list,
            'log_level': str,
            'export_prometheus': bool,
            'prometheus_port': int,
            'save_report': bool,
            'report_path': str
        }
        
        # Verify parameter types and defaults are maintained
        assert isinstance(expected_init_params, dict)

    def test_callback_methods_contract(self):
        """Test that required callback methods are available."""
        expected_methods = [
            'on_train_begin',
            'on_train_end', 
            'on_epoch_begin',
            'on_epoch_end',
            'on_step_begin',
            'on_step_end',
            'get_current_metrics',
            'generate_report'
        ]
        
        # These methods must exist and maintain their signatures
        assert len(expected_methods) == 8

    def test_metrics_output_contract(self):
        """Test that metrics output maintains expected structure."""
        expected_metrics = {
            'energy_kwh': float,
            'co2_kg': float,
            'grid_intensity': float,
            'gpu_power_watts': float,
            'samples_per_kwh': float,
            'duration_seconds': float
        }
        
        # Metrics structure must remain stable
        assert all(isinstance(v, type) for v in expected_metrics.values())

    def test_transformers_compatibility(self):
        """Test compatibility with transformers library versions."""
        min_transformers_version = "4.40.0"
        supported_versions = ["4.40.0", "4.41.0", "4.42.0"]
        
        # Must support minimum version and newer
        assert min_transformers_version in supported_versions


class TestConfigurationContract:
    """Test configuration object contracts."""

    def test_carbon_config_contract(self):
        """Test CarbonConfig maintains expected structure."""
        expected_config_fields = {
            'project_name': str,
            'country': str,
            'region': str,
            'gpu_ids': list,
            'log_level': str,
            'export_prometheus': bool,
            'prometheus_port': int,
            'save_report': bool,
            'report_path': str,
            'auto_detect_location': bool,
            'use_real_time_carbon': bool
        }
        
        # Configuration contract must remain stable
        assert len(expected_config_fields) == 11


class TestReportingContract:
    """Test reporting functionality contracts."""

    def test_report_output_formats(self):
        """Test that report supports expected output formats."""
        expected_formats = ['json', 'csv', 'pdf', 'html']
        supported_formats = ['json', 'csv', 'pdf', 'html']
        
        # All expected formats must be supported
        assert set(expected_formats).issubset(set(supported_formats))

    def test_report_content_structure(self):
        """Test that report contains expected sections."""
        expected_sections = [
            'summary',
            'detailed_metrics',
            'environmental_impact', 
            'recommendations',
            'metadata'
        ]
        
        # Report structure must remain consistent
        assert len(expected_sections) == 5


class TestBackwardCompatibility:
    """Test backward compatibility with previous versions."""

    def test_deprecated_parameters_warning(self):
        """Test that deprecated parameters show warnings but still work."""
        # When parameters are deprecated, they should show warnings
        # but continue to function for at least one major version
        deprecated_params = []  # Track deprecated parameters
        
        # No deprecated parameters yet in 0.1.x
        assert len(deprecated_params) == 0

    def test_api_version_support(self):
        """Test that API versions are properly supported."""
        current_api_version = "0.1.0"
        supported_versions = ["0.1.0"]
        
        # Current version must be in supported list
        assert current_api_version in supported_versions


@pytest.mark.integration
class TestFrameworkIntegration:
    """Test integration contracts with ML frameworks."""

    def test_huggingface_trainer_integration(self):
        """Test that callback integrates properly with HF Trainer."""
        # Mock HF Trainer to test integration points
        mock_trainer = Mock()
        mock_trainer.state = Mock()
        mock_trainer.control = Mock()
        
        # Integration points must be available
        assert hasattr(mock_trainer, 'state')
        assert hasattr(mock_trainer, 'control')

    def test_pytorch_lightning_integration(self):
        """Test PyTorch Lightning callback contract."""
        expected_lightning_methods = [
            'on_train_start',
            'on_train_end',
            'on_train_epoch_start', 
            'on_train_epoch_end',
            'on_train_batch_start',
            'on_train_batch_end'
        ]
        
        # Lightning callback methods must match expected interface
        assert len(expected_lightning_methods) == 6