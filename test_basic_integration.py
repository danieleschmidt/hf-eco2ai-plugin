#!/usr/bin/env python3
"""Basic integration test for HF Eco2AI Plugin."""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hf_eco2ai import Eco2AICallback, CarbonConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_callback():
    """Test basic callback functionality."""
    print("🧪 Testing basic callback functionality...")
    
    # Initialize callback with mock configuration
    config = CarbonConfig(
        project_name="test-training",
        country="USA", 
        region="CA",
        log_to_console=True,
        save_report=False
    )
    
    callback = Eco2AICallback(config)
    
    # Test current metrics
    metrics = callback.get_current_metrics()
    assert isinstance(metrics, dict)
    assert "power_watts" in metrics
    assert "energy_kwh" in metrics
    assert "co2_kg" in metrics
    
    print(f"✓ Current metrics: {metrics}")
    
    # Test energy tracker availability
    print(f"✓ Energy tracking available: {callback.energy_tracker.is_available()}")
    
    # Test carbon intensity
    carbon_intensity = callback.energy_tracker.carbon_provider.get_carbon_intensity()
    print(f"✓ Carbon intensity: {carbon_intensity} g CO₂/kWh")
    
    return True


def test_mock_training_simulation():
    """Test simulated training scenario."""
    print("\n🧪 Testing mock training simulation...")
    
    config = CarbonConfig(
        project_name="mock-bert-training",
        country="USA",
        region="CA", 
        log_to_console=True
    )
    
    callback = Eco2AICallback(config)
    
    # Simulate training start
    callback.energy_tracker.start_tracking()
    print("✓ Started energy tracking")
    
    # Simulate some training steps
    for step in range(3):
        time.sleep(0.1)  # Brief pause to simulate computation
        
        # Simulate sample processing
        callback._samples_processed += 100
        callback._current_step = step + 1
        
        # Get current metrics
        metrics = callback.get_current_metrics()
        print(f"  Step {step + 1}: {metrics['energy_kwh']:.4f} kWh, {metrics['co2_kg']:.4f} kg CO₂")
    
    # Stop tracking
    callback.energy_tracker.stop_tracking()
    print("✓ Stopped energy tracking")
    
    # Generate final report
    final_metrics = callback.get_current_metrics()
    print(f"✓ Final metrics: Energy={final_metrics['energy_kwh']:.4f} kWh, CO₂={final_metrics['co2_kg']:.4f} kg")
    
    return True


def test_carbon_report():
    """Test carbon report generation."""
    print("\n🧪 Testing carbon report generation...")
    
    config = CarbonConfig(project_name="report-test")
    callback = Eco2AICallback(config)
    
    # Add some mock metrics
    callback._samples_processed = 1000
    callback._current_step = 10
    callback._current_epoch = 1
    
    report = callback.generate_report()
    print(f"✓ Generated report with ID: {report.report_id}")
    
    summary_text = callback.carbon_summary
    print(f"✓ Summary preview: {summary_text[:100]}...")
    
    return True


def main():
    """Run all basic tests."""
    print("🚀 Starting HF Eco2AI Plugin Basic Integration Tests\n")
    
    tests = [
        test_basic_callback,
        test_mock_training_simulation, 
        test_carbon_report
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
        print("🎉 All tests passed! Generation 1 implementation is working!")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())