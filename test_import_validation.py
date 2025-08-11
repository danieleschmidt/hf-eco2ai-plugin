"""Import validation for advanced features."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all new module imports."""
    results = []
    
    try:
        from hf_eco2ai.carbon_intelligence import CarbonIntelligenceEngine, CarbonInsight
        results.append("✅ Carbon Intelligence Engine imported successfully")
    except Exception as e:
        results.append(f"❌ Carbon Intelligence Engine failed: {e}")
    
    try:
        from hf_eco2ai.sustainability_optimizer import SustainabilityOptimizer, SustainabilityGoal
        results.append("✅ Sustainability Optimizer imported successfully")
    except Exception as e:
        results.append(f"❌ Sustainability Optimizer failed: {e}")
    
    try:
        from hf_eco2ai.enterprise_monitoring import EnterpriseMonitor, MetricsCollector
        results.append("✅ Enterprise Monitoring imported successfully")
    except Exception as e:
        results.append(f"❌ Enterprise Monitoring failed: {e}")
    
    try:
        from hf_eco2ai.fault_tolerance import FaultToleranceManager, resilient
        results.append("✅ Fault Tolerance imported successfully")
    except Exception as e:
        results.append(f"❌ Fault Tolerance failed: {e}")
    
    try:
        from hf_eco2ai.quantum_optimizer import QuantumOptimizationOrchestrator
        results.append("✅ Quantum Optimizer imported successfully")
    except Exception as e:
        results.append(f"❌ Quantum Optimizer failed: {e}")
    
    try:
        from hf_eco2ai.adaptive_scaling import AdaptiveScaler, get_adaptive_scaler
        results.append("✅ Adaptive Scaling imported successfully")
    except Exception as e:
        results.append(f"❌ Adaptive Scaling failed: {e}")
    
    return results

def test_basic_functionality():
    """Test basic functionality of new features."""
    results = []
    
    try:
        from hf_eco2ai.carbon_intelligence import CarbonIntelligenceEngine
        engine = CarbonIntelligenceEngine()
        report = engine.generate_intelligence_report()
        assert "report_id" in report
        results.append("✅ Carbon Intelligence basic functionality works")
    except Exception as e:
        results.append(f"❌ Carbon Intelligence functionality failed: {e}")
    
    try:
        from hf_eco2ai.sustainability_optimizer import SustainabilityOptimizer
        optimizer = SustainabilityOptimizer()
        dashboard = optimizer.get_sustainability_dashboard()
        assert "dashboard_id" in dashboard
        results.append("✅ Sustainability Optimizer basic functionality works")
    except Exception as e:
        results.append(f"❌ Sustainability Optimizer functionality failed: {e}")
    
    try:
        from hf_eco2ai.quantum_optimizer import QuantumOptimizationOrchestrator
        quantum = QuantumOptimizationOrchestrator()
        summary = quantum.get_optimization_summary()
        assert isinstance(summary, dict)
        results.append("✅ Quantum Optimizer basic functionality works")
    except Exception as e:
        results.append(f"❌ Quantum Optimizer functionality failed: {e}")
    
    return results

if __name__ == "__main__":
    print("🧪 Testing Advanced Feature Imports...")
    print("="*50)
    
    import_results = test_imports()
    for result in import_results:
        print(result)
    
    print("\n🔬 Testing Basic Functionality...")
    print("="*50)
    
    func_results = test_basic_functionality()
    for result in func_results:
        print(result)
    
    print("\n📊 Summary:")
    print("="*50)
    
    total_tests = len(import_results) + len(func_results)
    passed_tests = sum(1 for r in import_results + func_results if r.startswith("✅"))
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Advanced features are ready!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Review implementation.")