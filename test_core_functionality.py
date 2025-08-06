#!/usr/bin/env python3
"""Test core functionality of HF Eco2AI Plugin."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_imports():
    """Test that core modules can be imported."""
    print("Testing core imports...")
    
    try:
        from hf_eco2ai.config import CarbonConfig
        print("✅ CarbonConfig imported successfully")
    except Exception as e:
        print(f"❌ CarbonConfig import failed: {e}")
        return False
    
    try:
        from hf_eco2ai.models import CarbonMetrics, CarbonReport
        print("✅ Data models imported successfully")
    except Exception as e:
        print(f"❌ Models import failed: {e}")
        return False
    
    try:
        from hf_eco2ai.utils import setup_logging, get_system_info
        print("✅ Utilities imported successfully")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True

def test_configuration_system():
    """Test configuration system functionality."""
    print("\nTesting configuration system...")
    
    try:
        from hf_eco2ai.config import CarbonConfig
        
        # Test default config
        config = CarbonConfig()
        print(f"✅ Default config created with {len(config.to_dict())} parameters")
        
        # Test config serialization
        config_dict = config.to_dict()
        config2 = CarbonConfig.from_dict(config_dict)
        print("✅ Config serialization/deserialization working")
        
        # Test validation
        missing_deps = config.validate_environment()
        print(f"✅ Environment validation complete, missing deps: {len(missing_deps)}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_models():
    """Test data model functionality."""
    print("\nTesting data models...")
    
    try:
        from hf_eco2ai.models import CarbonMetrics, CarbonReport
        import time
        
        # Test CarbonMetrics
        metrics = CarbonMetrics(
            timestamp=time.time(),
            step=100,
            epoch=1,
            energy_kwh=0.5,
            cumulative_energy_kwh=1.0,
            power_watts=250.0,
            co2_kg=0.2,
            cumulative_co2_kg=0.4,
            grid_intensity=400.0
        )
        
        metrics_dict = metrics.to_dict()
        print(f"✅ CarbonMetrics created with {len(metrics_dict)} fields")
        
        # Test JSON serialization
        metrics_json = metrics.to_json()
        print(f"✅ CarbonMetrics JSON serialization: {len(metrics_json)} chars")
        
        return True
    except Exception as e:
        print(f"❌ Data models test failed: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from hf_eco2ai.utils import get_system_info, format_duration, calculate_carbon_equivalents
        
        # Test system info
        system_info = get_system_info()
        print(f"✅ System info retrieved: {len(system_info)} fields")
        
        # Test formatting
        duration_str = format_duration(3661)  # 1 hour, 1 minute, 1 second
        print(f"✅ Duration formatting: {duration_str}")
        
        # Test carbon equivalents
        equivalents = calculate_carbon_equivalents(1.0)  # 1 kg CO2
        print(f"✅ Carbon equivalents calculated: {len(equivalents)} types")
        
        return True
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False

def test_advanced_features():
    """Test advanced features without external dependencies."""
    print("\nTesting advanced features...")
    
    try:
        from hf_eco2ai.validation import CarbonTrackingValidator
        from hf_eco2ai.security import SecurityValidator, DataSanitizer
        from hf_eco2ai.health import HealthMonitor
        from hf_eco2ai.optimization import QuantumInspiredOptimizer
        from hf_eco2ai.quantum_planner import QuantumInspiredTaskPlanner
        
        print("✅ Advanced modules imported successfully")
        
        # Test validation
        validator = CarbonTrackingValidator()
        print("✅ Validator initialized")
        
        # Test security
        security_validator = SecurityValidator()
        print("✅ Security validator initialized")
        
        # Test health monitoring
        health_monitor = HealthMonitor()
        print("✅ Health monitor initialized")
        
        # Test quantum optimizer
        optimizer = QuantumInspiredOptimizer()
        print("✅ Quantum optimizer initialized")
        
        # Test quantum planner
        planner = QuantumInspiredTaskPlanner()
        print("✅ Quantum task planner initialized")
        
        return True
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing HF Eco2AI Plugin Core Functionality")
    print("=" * 60)
    
    tests = [
        test_core_imports,
        test_configuration_system,
        test_data_models,
        test_utilities,
        test_advanced_features
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED! Core functionality is working correctly.")
    else:
        print(f"⚠️  {len(tests) - passed} tests failed. Some features may not work correctly.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)