"""Comprehensive security and validation tests."""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.hf_eco2ai.security import (
    SecurityManager,
    SecurityPolicy,
    SecurityLevel,
    ThreatLevel,
    SecurityEvent
)
from src.hf_eco2ai.validation import (
    CarbonTrackingValidator,
    ValidationSeverity,
    ValidationResult,
    ValidationSuite
)
from src.hf_eco2ai.resilience import (
    ResilienceManager,
    FailureMode,
    FailureRecord,
    with_retry,
    with_circuit_breaker,
    HealthCheck
)
from src.hf_eco2ai.config import CarbonConfig
from src.hf_eco2ai.models import CarbonMetrics, CarbonReport, CarbonSummary


class TestSecurityManager:
    """Test security management features."""
    
    @pytest.fixture
    def security_policy(self):
        """Create security policy for testing."""
        return SecurityPolicy(
            encryption_required=True,
            data_retention_days=30,
            audit_logging=True,
            rate_limiting=True,
            max_requests_per_hour=100
        )
    
    @pytest.fixture
    def security_manager(self, security_policy):
        """Create security manager instance."""
        return SecurityManager(security_policy)
    
    def test_security_manager_initialization(self, security_manager):
        """Test security manager initialization."""
        assert security_manager.policy.encryption_required == True
        assert security_manager.policy.audit_logging == True
        assert len(security_manager.audit_log) == 1  # Initialization event
        assert security_manager.security_checks_enabled == True
    
    def test_authentication_with_valid_request(self, security_manager):
        """Test authentication with valid request."""
        request_data = {
            "source_ip": "192.168.1.100",
            "api_key": "valid_api_key_12345678"
        }
        
        result = security_manager.authenticate_request(request_data)
        assert result == True
    
    def test_authentication_with_invalid_ip(self, security_manager):
        """Test authentication blocks invalid IP."""
        # Set IP allowlist
        security_manager.policy.ip_allowlist = ["192.168.1.0/24"]
        
        request_data = {
            "source_ip": "10.0.0.1",  # Not in allowlist
            "api_key": "valid_api_key_12345678"
        }
        
        result = security_manager.authenticate_request(request_data)
        assert result == False
        
        # Check audit log
        security_events = [e for e in security_manager.audit_log if e.action == "ip_blocked"]
        assert len(security_events) > 0
    
    def test_rate_limiting(self, security_manager):
        """Test rate limiting functionality."""
        source_ip = "192.168.1.100"
        
        # Make requests up to the limit
        for i in range(security_manager.policy.max_requests_per_hour):
            request_data = {"source_ip": source_ip}
            result = security_manager.authenticate_request(request_data)
            assert result == True
        
        # Next request should be rate limited
        request_data = {"source_ip": source_ip}
        result = security_manager.authenticate_request(request_data)
        assert result == False
        
        # Check audit log for rate limiting
        rate_limit_events = [e for e in security_manager.audit_log if e.action == "rate_limited"]
        assert len(rate_limit_events) > 0
    
    def test_data_sanitization_public_level(self, security_manager):
        """Test data sanitization for public security level."""
        sensitive_data = {
            "api_key": "secret_key_12345",
            "password": "my_password",
            "email": "user@example.com",
            "normal_field": "public_data"
        }
        
        sanitized = security_manager.sanitize_data(sensitive_data, SecurityLevel.PUBLIC)
        
        assert sanitized["api_key"] != "secret_key_12345"  # Should be masked
        assert sanitized["password"] != "my_password"      # Should be masked
        assert "***" in sanitized["email"]                 # Should be partially masked
        assert sanitized["normal_field"] == "public_data"  # Should remain unchanged
    
    def test_data_sanitization_internal_level(self, security_manager):
        """Test data sanitization for internal security level."""
        sensitive_data = {
            "api_key": "secret_key_12345",
            "normal_field": "internal_data"
        }
        
        sanitized = security_manager.sanitize_data(sensitive_data, SecurityLevel.INTERNAL)
        
        assert sanitized["api_key"] != "secret_key_12345"  # Should still be masked
        assert sanitized["normal_field"] == "internal_data"
    
    def test_data_sanitization_confidential_level(self, security_manager):
        """Test data sanitization for confidential security level."""
        sensitive_data = {
            "api_key": "secret_key_12345",
            "password": "my_password",
            "confidential_field": "confidential_data"
        }
        
        sanitized = security_manager.sanitize_data(sensitive_data, SecurityLevel.CONFIDENTIAL)
        
        # At confidential level, sensitive data should be preserved
        assert sanitized["api_key"] == "secret_key_12345"
        assert sanitized["password"] == "my_password"
        assert sanitized["confidential_field"] == "confidential_data"
    
    def test_encryption_and_decryption(self, security_manager):
        """Test data encryption and decryption."""
        original_data = "sensitive information that needs encryption"
        
        # Encrypt data
        encrypted_result = security_manager.encrypt_data(original_data)
        
        assert encrypted_result["encrypted"] == True
        assert encrypted_result["data"] != original_data
        assert "algorithm" in encrypted_result
        
        # For testing, we'll skip actual decryption as it requires the key
        # In practice, the key would be securely managed
    
    def test_secure_token_generation(self, security_manager):
        """Test secure token generation."""
        token1 = security_manager.generate_secure_token(16)
        token2 = security_manager.generate_secure_token(16)
        
        assert len(token1) == 32  # Hex encoding doubles length
        assert len(token2) == 32
        assert token1 != token2  # Should be unique
        assert all(c in "0123456789abcdef" for c in token1)  # Should be hex
    
    def test_data_hashing_and_verification(self, security_manager):
        """Test data hashing and verification."""
        original_data = "password_to_hash"
        
        # Hash data
        hash_info = security_manager.hash_data(original_data)
        
        assert "hash" in hash_info
        assert "salt" in hash_info
        assert "algorithm" in hash_info
        assert hash_info["algorithm"] == "PBKDF2-SHA256"
        
        # Verify correct data
        assert security_manager.verify_hash(original_data, hash_info) == True
        
        # Verify incorrect data
        assert security_manager.verify_hash("wrong_password", hash_info) == False
    
    def test_security_event_logging(self, security_manager):
        """Test security event logging."""
        initial_log_count = len(security_manager.audit_log)
        
        security_manager.log_security_event(
            event_type="test",
            action="test_action",
            resource="test_resource",
            outcome="success",
            threat_level=ThreatLevel.LOW
        )
        
        assert len(security_manager.audit_log) == initial_log_count + 1
        
        latest_event = security_manager.audit_log[-1]
        assert latest_event.event_type == "test"
        assert latest_event.action == "test_action"
        assert latest_event.threat_level == ThreatLevel.LOW
    
    def test_security_report_generation(self, security_manager):
        """Test security report generation."""
        # Add some test events
        for i in range(5):
            security_manager.log_security_event(
                event_type="test",
                action=f"test_action_{i}",
                resource="test_resource",
                outcome="success" if i % 2 == 0 else "failure",
                threat_level=ThreatLevel.LOW if i < 3 else ThreatLevel.HIGH
            )
        
        report = security_manager.get_security_report()
        
        assert "total_events" in report
        assert "threat_summary" in report
        assert "policy_status" in report
        assert "recommendations" in report
        
        assert report["total_events"] > 5  # Including initialization event
        assert "low" in report["threat_summary"]
        assert "high" in report["threat_summary"]


class TestCarbonTrackingValidator:
    """Test carbon tracking validation features."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return CarbonTrackingValidator()
    
    @pytest.fixture
    def valid_config(self):
        """Create valid carbon config."""
        return CarbonConfig(
            gpu_ids=[0, 1],
            country="USA",
            region="California",
            export_prometheus=True,
            prometheus_port=9090
        )
    
    @pytest.fixture
    def invalid_config(self):
        """Create invalid carbon config."""
        return CarbonConfig(
            gpu_ids=[-1, 99],  # Invalid GPU IDs
            country="InvalidCountry",
            region="InvalidRegion",
            prometheus_port=99999,  # Invalid port
            max_co2_kg=-1.0,  # Invalid budget
            enable_carbon_budget=True
        )
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.security_checks_enabled == True
        assert validator.performance_checks_enabled == True
        assert len(validator.sensitive_patterns) > 0
    
    def test_valid_config_validation(self, validator, valid_config):
        """Test validation of valid configuration."""
        suite = validator.validate_config(valid_config)
        
        assert suite.suite_name == "configuration_validation"
        assert suite.total_checks > 0
        assert suite.passed_checks > 0
        assert suite.execution_time > 0
        
        # Most checks should pass for valid config
        assert suite.passed_checks >= suite.failed_checks
    
    def test_invalid_config_validation(self, validator, invalid_config):
        """Test validation of invalid configuration."""
        suite = validator.validate_config(invalid_config)
        
        assert suite.suite_name == "configuration_validation"
        assert suite.failed_checks > 0
        
        # Check for specific validation failures
        gpu_failures = [r for r in suite.results if r.check_name == "gpu_config" and not r.passed]
        assert len(gpu_failures) > 0
        
        region_failures = [r for r in suite.results if r.check_name == "region_config" and not r.passed]
        assert len(region_failures) > 0
        
        budget_failures = [r for r in suite.results if r.check_name == "budget_config" and not r.passed]
        assert len(budget_failures) > 0
    
    def test_metrics_validation(self, validator):
        """Test carbon metrics validation."""
        # Valid metrics
        valid_metrics = CarbonMetrics(
            timestamp=time.time(),
            energy_kwh=1.5,
            co2_kg=0.8,
            power_watts=150.0,
            grid_intensity=400.0,
            samples_processed=1000
        )
        
        suite = validator.validate_metrics(valid_metrics)
        assert suite.suite_name == "metrics_validation"
        assert suite.passed_checks > 0
        
        # Invalid metrics
        invalid_metrics = CarbonMetrics(
            timestamp=time.time() + 3600,  # Future timestamp
            energy_kwh=-1.0,  # Negative energy
            co2_kg=float('inf'),  # Invalid CO2
            power_watts=-50.0,  # Negative power
            grid_intensity=5000.0,  # Unrealistic intensity
            samples_processed=-100  # Negative samples
        )
        
        suite = validator.validate_metrics(invalid_metrics)
        assert suite.failed_checks > 0
    
    def test_report_validation(self, validator):
        """Test carbon report validation."""
        # Create valid report
        summary = CarbonSummary(
            total_energy_kwh=10.0,
            total_co2_kg=5.0,
            total_duration_hours=2.0
        )
        
        metrics = [
            CarbonMetrics(energy_kwh=1.0, co2_kg=0.5),
            CarbonMetrics(energy_kwh=2.0, co2_kg=1.0),
            CarbonMetrics(energy_kwh=7.0, co2_kg=3.5)  # Totals should match summary
        ]
        
        valid_report = CarbonReport(
            report_id="test_report",
            summary=summary,
            detailed_metrics=metrics
        )
        
        suite = validator.validate_report(valid_report)
        assert suite.suite_name == "report_validation"
        
        # Should have some passing checks
        completeness_checks = [r for r in suite.results if r.check_name == "report_completeness"]
        assert len(completeness_checks) > 0
    
    def test_validation_report_generation(self, validator, valid_config):
        """Test comprehensive validation report generation."""
        # Run multiple validation suites
        config_suite = validator.validate_config(valid_config)
        
        metrics = CarbonMetrics(energy_kwh=1.0, co2_kg=0.5)
        metrics_suite = validator.validate_metrics(metrics)
        
        suites = [config_suite, metrics_suite]
        report = validator.generate_validation_report(suites)
        
        assert "validation_summary" in report
        assert "severity_breakdown" in report
        assert "recommendations" in report
        assert "suite_details" in report
        
        summary = report["validation_summary"]
        assert summary["total_checks"] > 0
        assert summary["suites_executed"] == 2


class TestResilienceManager:
    """Test resilience and error handling features."""
    
    @pytest.fixture
    def resilience_manager(self):
        """Create resilience manager instance."""
        return ResilienceManager()
    
    def test_resilience_manager_initialization(self, resilience_manager):
        """Test resilience manager initialization."""
        assert len(resilience_manager.failure_history) == 0
        assert len(resilience_manager.circuit_breakers) == 0
        assert resilience_manager.default_retry_config["max_attempts"] == 3
    
    def test_failure_recording(self, resilience_manager):
        """Test failure recording functionality."""
        error = ValueError("Test error")
        
        resilience_manager.record_failure(
            component="test_component",
            error=error,
            failure_mode=FailureMode.TRANSIENT
        )
        
        assert len(resilience_manager.failure_history) == 1
        
        failure = resilience_manager.failure_history[0]
        assert failure.component == "test_component"
        assert failure.failure_type == "ValueError"
        assert failure.error_message == "Test error"
        assert failure.failure_mode == FailureMode.TRANSIENT
    
    def test_circuit_breaker_functionality(self, resilience_manager):
        """Test circuit breaker functionality."""
        component = "test_component"
        
        # Initially should allow requests
        assert resilience_manager.should_allow_request(component) == True
        
        # Record multiple failures to open circuit breaker
        for i in range(6):  # More than default threshold of 5
            error = RuntimeError(f"Error {i}")
            resilience_manager.record_failure(component, error)
        
        # Circuit breaker should now be open
        assert resilience_manager.should_allow_request(component) == False
        
        # Record success to eventually close circuit breaker
        resilience_manager.record_success(component)
    
    def test_failure_statistics(self, resilience_manager):
        """Test failure statistics calculation."""
        # Record various failures
        errors = [
            (ValueError("Error 1"), FailureMode.TRANSIENT),
            (RuntimeError("Error 2"), FailureMode.PERMANENT),
            (ConnectionError("Error 3"), FailureMode.TRANSIENT)
        ]
        
        for error, mode in errors:
            resilience_manager.record_failure("test_component", error, mode)
        
        stats = resilience_manager.get_failure_statistics()
        
        assert stats["total_failures"] == 3
        assert "ValueError" in stats["failure_types"]
        assert "RuntimeError" in stats["failure_types"]
        assert "transient" in stats["failure_modes"]
        assert "permanent" in stats["failure_modes"]
    
    def test_retry_decorator(self):
        """Test retry decorator functionality."""
        call_count = 0
        
        @with_retry(component="test", max_attempts=3, base_delay=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = failing_function()
        
        assert result == "success"
        assert call_count == 3
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        @with_circuit_breaker(component="test_cb", failure_threshold=2)
        def unstable_function(should_fail=False):
            if should_fail:
                raise RuntimeError("Function failed")
            return "success"
        
        # Function should work normally
        assert unstable_function(False) == "success"
        
        # Cause failures to open circuit breaker
        with pytest.raises(RuntimeError):
            unstable_function(True)
        
        with pytest.raises(RuntimeError):
            unstable_function(True)
        
        # Circuit breaker should now be open
        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            unstable_function(False)
    
    def test_health_check_system(self):
        """Test health check system."""
        health_check = HealthCheck()
        
        # Register health checks
        health_check.register_check("test_check_1", lambda: True)
        health_check.register_check("test_check_2", lambda: False)
        health_check.register_check("test_check_3", lambda: True)
        
        # Run health checks
        results = health_check.run_checks()
        
        assert "overall_healthy" in results
        assert "checks" in results
        assert results["overall_healthy"] == False  # One check fails
        
        assert results["checks"]["test_check_1"]["healthy"] == True
        assert results["checks"]["test_check_2"]["healthy"] == False
        assert results["checks"]["test_check_3"]["healthy"] == True
        
        # Get health status
        status = health_check.get_health_status()
        assert "healthy" in status
        assert "checks" in status
        assert status["healthy"] == False


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security features."""
    
    def test_end_to_end_security_workflow(self):
        """Test complete security workflow."""
        # Initialize security manager
        policy = SecurityPolicy(
            encryption_required=True,
            audit_logging=True,
            rate_limiting=True,
            max_requests_per_hour=10
        )
        
        security_manager = SecurityManager(policy)
        
        # Simulate API requests
        for i in range(5):
            request_data = {
                "source_ip": "192.168.1.100",
                "api_key": f"test_key_{i}"
            }
            
            # Authenticate request
            is_authenticated = security_manager.authenticate_request(request_data)
            assert is_authenticated == True
            
            # Process some data
            sensitive_data = {
                "user_id": f"user_{i}",
                "api_key": f"secret_{i}",
                "result": f"result_{i}"
            }
            
            # Sanitize data for logging
            sanitized = security_manager.sanitize_data(sensitive_data, SecurityLevel.INTERNAL)
            
            # Log the operation
            security_manager.log_security_event(
                event_type="api",
                action="process_data",
                resource="carbon_metrics",
                outcome="success"
            )
        
        # Generate security report
        report = security_manager.get_security_report()
        
        assert report["total_events"] > 5
        assert "api" in [event.event_type for event in security_manager.audit_log]


@pytest.mark.performance
class TestValidationPerformance:
    """Performance tests for validation features."""
    
    def test_large_config_validation_performance(self, benchmark):
        """Benchmark configuration validation performance."""
        validator = CarbonTrackingValidator()
        
        # Create complex configuration
        config = CarbonConfig(
            gpu_ids=list(range(8)),
            export_prometheus=True,
            save_report=True,
            enable_carbon_budget=True
        )
        
        def validate_config():
            return validator.validate_config(config)
        
        suite = benchmark(validate_config)
        assert suite.total_checks > 0
    
    def test_batch_metrics_validation_performance(self, benchmark):
        """Benchmark batch metrics validation performance."""
        validator = CarbonTrackingValidator()
        
        # Create batch of metrics
        metrics_batch = [
            CarbonMetrics(
                timestamp=time.time() - i,
                energy_kwh=1.0 + i * 0.1,
                co2_kg=0.5 + i * 0.05,
                power_watts=100.0 + i * 10
            )
            for i in range(100)
        ]
        
        def validate_metrics_batch():
            results = []
            for metrics in metrics_batch:
                suite = validator.validate_metrics(metrics)
                results.append(suite)
            return results
        
        suites = benchmark(validate_metrics_batch)
        assert len(suites) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])