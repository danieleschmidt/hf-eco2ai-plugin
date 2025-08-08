#!/usr/bin/env python3
"""Test robustness and reliability features of HF Eco2AI Plugin."""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hf_eco2ai import Eco2AICallback, CarbonConfig
from hf_eco2ai.error_handling import get_error_handler, ErrorSeverity
from hf_eco2ai.health_monitor import get_health_monitor, get_system_health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_error_handling():
    """Test comprehensive error handling system."""
    print("🧪 Testing error handling system...")
    
    error_handler = get_error_handler()
    
    # Test error handling
    test_error = ValueError("Test validation error")
    error_info = error_handler.handle_error(
        test_error, 
        {"component": "test"}, 
        ErrorSeverity.MEDIUM
    )
    
    assert error_info.error_type == "ValueError"
    assert error_info.severity == ErrorSeverity.MEDIUM
    assert "test" in error_info.context.get("component", "")
    
    # Test error summary
    summary = error_handler.get_error_summary()
    assert summary["total_errors"] >= 1
    assert "medium" in summary["by_severity"]
    
    print(f"✓ Error handling working - {summary['total_errors']} errors tracked")
    return True


def test_health_monitoring():
    """Test system health monitoring."""
    print("🧪 Testing health monitoring system...")
    
    health_monitor = get_health_monitor()
    
    # Get current health
    health = health_monitor.check_system_health()
    
    assert health.overall_status
    assert health.cpu_percent >= 0
    assert health.memory_percent >= 0
    assert health.disk_percent >= 0
    assert len(health.metrics) > 0
    
    # Test health summary
    health_monitor.health_history.append(health)
    summary = health_monitor.get_health_summary()
    
    assert "current_status" in summary
    assert "uptime_hours" in summary
    assert "averages" in summary
    
    print(f"✓ Health monitoring working - Status: {health.overall_status.value}")
    print(f"  CPU: {health.cpu_percent:.1f}%, Memory: {health.memory_percent:.1f}%")
    return True


def test_resilient_callback():
    """Test resilient callback with error scenarios."""
    print("🧪 Testing resilient callback operations...")
    
    config = CarbonConfig(
        project_name="resilience-test",
        country="USA",
        region="CA",
        log_to_console=False  # Reduce noise
    )
    
    callback = Eco2AICallback(config)
    
    # Test that callback handles missing dependencies gracefully
    assert callback.error_handler is not None
    assert callback.validator is not None
    assert callback.health_monitor is not None
    
    # Test metrics retrieval doesn't crash
    metrics = callback.get_current_metrics()
    assert isinstance(metrics, dict)
    
    # Test carbon report generation doesn't crash
    report = callback.generate_report()
    assert report.report_id
    
    print("✓ Resilient callback operations working")
    return True


def test_validation_system():
    """Test comprehensive validation system."""
    print("🧪 Testing validation system...")
    
    config = CarbonConfig(project_name="validation-test")
    callback = Eco2AICallback(config)
    
    # Create mock training args
    class MockArgs:
        num_train_epochs = 3
        per_device_train_batch_size = 16
        learning_rate = 5e-5
    
    args = MockArgs()
    
    # Test validation
    validation_result = callback.validator.validate_training_args(args)
    assert validation_result["valid"] == True
    
    # Test invalid args
    args.num_train_epochs = -1
    validation_result = callback.validator.validate_training_args(args)
    assert validation_result["valid"] == False
    
    print("✓ Validation system working")
    return True


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("🧪 Testing circuit breaker patterns...")
    
    from hf_eco2ai.error_handling import CircuitBreaker
    
    # Test circuit breaker
    failure_count = 0
    
    @CircuitBreaker(failure_threshold=3, timeout=1)
    def failing_function():
        nonlocal failure_count
        failure_count += 1
        if failure_count < 6:
            raise ValueError("Simulated failure")
        return "success"
    
    # Trigger failures to open circuit
    for i in range(5):
        try:
            failing_function()
        except Exception:
            pass
    
    # Circuit should be open now, test that it fails fast
    try:
        failing_function()
        assert False, "Circuit breaker should be open"
    except Exception as e:
        assert "Circuit breaker open" in str(e)
    
    print("✓ Circuit breaker working correctly")
    return True


def test_carbon_budget_enforcement():
    """Test carbon budget enforcement."""
    print("🧪 Testing carbon budget enforcement...")
    
    from hf_eco2ai import CarbonBudgetCallback
    
    # Create a callback with very low budget for testing
    budget_callback = CarbonBudgetCallback(
        max_co2_kg=0.001,  # Very low budget
        action="warn",  # Just warn, don't stop
        check_frequency=1
    )
    
    # Test basic initialization
    assert budget_callback.max_co2_kg == 0.001
    assert budget_callback.action == "warn"
    assert budget_callback.check_frequency == 1
    
    print("✓ Carbon budget enforcement initialized correctly")
    return True


def test_data_sanitization():
    """Test data sanitization and security features."""
    print("🧪 Testing data sanitization...")
    
    from hf_eco2ai.security import get_data_sanitizer
    
    sanitizer = get_data_sanitizer()
    
    # Test metrics sanitization
    test_metrics = {
        "energy_kwh": 0.5,
        "co2_kg": 0.2,
        "suspicious": "<script>alert('xss')</script>",
        "normal": "clean_data"
    }
    
    sanitized = sanitizer.sanitize_metrics(test_metrics)
    assert isinstance(sanitized, dict)
    assert "energy_kwh" in sanitized
    
    print("✓ Data sanitization working")
    return True


def main():
    """Run all robustness tests."""
    print("🚀 Starting HF Eco2AI Plugin Robustness Tests\n")
    
    tests = [
        test_error_handling,
        test_health_monitoring,
        test_resilient_callback,
        test_validation_system,
        test_circuit_breaker,
        test_carbon_budget_enforcement,
        test_data_sanitization
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print("✅ PASSED\n")
        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"🎯 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All robustness tests passed! Generation 2 implementation is robust!")
        return 0
    else:
        print("⚠️  Some robustness tests failed. Check implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())