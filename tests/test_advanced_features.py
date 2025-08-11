"""Comprehensive tests for advanced carbon intelligence features."""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from hf_eco2ai.carbon_intelligence import (
    CarbonIntelligenceEngine, CarbonIntelligenceCallback,
    CarbonInsight, CarbonPrediction, TrainingOptimization
)
from hf_eco2ai.sustainability_optimizer import (
    SustainabilityOptimizer, SustainabilityCallback,
    SustainabilityGoal, OptimizationStrategy, CarbonBudget
)
from hf_eco2ai.enterprise_monitoring import (
    EnterpriseMonitor, MetricsCollector, AlertManager, HealthChecker,
    AlertRule, Alert, MetricSnapshot
)
from hf_eco2ai.fault_tolerance import (
    FaultToleranceManager, RetryManager, CircuitBreakerManager,
    FailureMode, RecoveryStrategy, resilient
)
from hf_eco2ai.quantum_optimizer import (
    QuantumOptimizationOrchestrator, QuantumAnnealingOptimizer,
    QuantumInspiredGeneticAlgorithm
)
from hf_eco2ai.adaptive_scaling import (
    AdaptiveScaler, ResourceMonitor, ScalingDecisionEngine,
    AdaptiveConfig
)


class TestCarbonIntelligenceEngine:
    """Test carbon intelligence engine functionality."""
    
    @pytest.fixture
    def intelligence_engine(self):
        return CarbonIntelligenceEngine()
    
    @pytest.fixture
    def sample_metrics(self):
        return {
            "energy_kwh": 5.2,
            "co2_kg": 2.1,
            "samples_processed": 10000,
            "grid_intensity": 400,
            "gpu_utilization": 0.85,
            "training_duration_hours": 2.5
        }
    
    @pytest.fixture
    def sample_training_config(self):
        return {
            "batch_size": 32,
            "learning_rate": 5e-5,
            "num_epochs": 10,
            "fp16": False,
            "gradient_accumulation_steps": 1,
            "num_gpus": 1,
            "model_name": "bert-base-uncased"
        }
    
    @pytest.mark.asyncio
    async def test_analyze_training_session(self, intelligence_engine, sample_metrics, sample_training_config):
        """Test training session analysis."""
        insights = await intelligence_engine.analyze_training_session(
            sample_metrics, sample_training_config
        )
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Check insight structure
        for insight in insights:
            assert isinstance(insight, CarbonInsight)
            assert insight.insight_id
            assert insight.timestamp
            assert insight.category in ["efficiency", "optimization", "environmental", "cost"]
            assert insight.severity in ["low", "medium", "high", "critical"]
            assert isinstance(insight.impact_estimate, float)
            assert 0 <= insight.confidence_score <= 1
    
    @pytest.mark.asyncio
    async def test_predict_training_impact(self, intelligence_engine, sample_training_config):
        """Test training impact prediction."""
        prediction = await intelligence_engine.predict_training_impact(
            sample_training_config, estimated_duration_hours=3.0
        )
        
        assert isinstance(prediction, CarbonPrediction)
        assert prediction.prediction_id
        assert prediction.timestamp
        assert prediction.training_duration_hours == 3.0
        assert prediction.predicted_co2_kg > 0
        assert prediction.predicted_energy_kwh > 0
        assert len(prediction.confidence_interval) == 2
        assert prediction.confidence_interval[0] < prediction.confidence_interval[1]
        assert isinstance(prediction.optimization_recommendations, list)
    
    def test_generate_intelligence_report(self, intelligence_engine):
        """Test intelligence report generation."""
        # Add some mock insights
        intelligence_engine.insights = [
            CarbonInsight(
                insight_id="test_1",
                timestamp=datetime.now(),
                category="efficiency",
                severity="high",
                title="Test Insight",
                description="Test description",
                impact_estimate=2.5,
                implementation_effort="medium",
                action_items=["Test action"],
                confidence_score=0.8
            )
        ]
        
        report = intelligence_engine.generate_intelligence_report()
        
        assert "report_id" in report
        assert "generated_at" in report
        assert "summary" in report
        assert report["summary"]["total_insights"] == 1
        assert report["summary"]["total_potential_co2_savings_kg"] == 2.5
        assert "insights_by_category" in report
        assert "top_recommendations" in report


class TestSustainabilityOptimizer:
    """Test sustainability optimizer functionality."""
    
    @pytest.fixture
    def sustainability_optimizer(self):
        return SustainabilityOptimizer()
    
    @pytest.fixture
    def sample_metrics(self):
        return {
            "co2_kg": 5.0,
            "energy_kwh": 12.0,
            "cost_usd": 2.5,
            "gpu_utilization": 0.75,
            "samples_processed": 15000
        }
    
    @pytest.fixture
    def sample_config(self):
        return {
            "per_device_train_batch_size": 16,
            "learning_rate": 3e-5,
            "num_train_epochs": 5,
            "fp16": False,
            "gradient_accumulation_steps": 1,
            "num_gpus": 2
        }
    
    @pytest.mark.asyncio
    async def test_analyze_sustainability_opportunities(self, sustainability_optimizer, sample_metrics, sample_config):
        """Test sustainability opportunity analysis."""
        strategies = await sustainability_optimizer.analyze_sustainability_opportunities(
            sample_metrics, sample_config
        )
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        
        # Check strategy structure
        for strategy in strategies:
            assert isinstance(strategy, OptimizationStrategy)
            assert strategy.strategy_id
            assert strategy.name
            assert strategy.category in ["model", "hardware", "scheduling", "environmental", "behavioral"]
            assert strategy.co2_reduction_kg >= 0
            assert strategy.implementation_effort in ["low", "medium", "high"]
            assert 0 <= strategy.confidence_score <= 1
            assert isinstance(strategy.implementation_steps, list)
    
    def test_create_carbon_budget(self, sustainability_optimizer):
        """Test carbon budget creation."""
        budget = sustainability_optimizer.create_carbon_budget(
            name="Test Budget",
            budget_kg=20.0,
            period_days=30
        )
        
        assert isinstance(budget, CarbonBudget)
        assert budget.name == "Test Budget"
        assert budget.total_budget_kg == 20.0
        assert budget.used_budget_kg == 0.0
        assert budget.remaining_budget_kg == 20.0
        assert budget.utilization_percentage == 0.0
    
    def test_update_carbon_budget(self, sustainability_optimizer):
        """Test carbon budget updates."""
        budget = sustainability_optimizer.create_carbon_budget(
            name="Test Budget",
            budget_kg=10.0,
            period_days=30
        )
        
        # Update with usage within budget
        success = sustainability_optimizer.update_carbon_budget(budget.budget_id, 3.0)
        assert success is True
        assert budget.used_budget_kg == 3.0
        assert budget.utilization_percentage == 30.0
        
        # Update to exceed budget
        success = sustainability_optimizer.update_carbon_budget(budget.budget_id, 8.0)
        assert success is False
        assert budget.utilization_percentage > 100
    
    def test_get_sustainability_dashboard(self, sustainability_optimizer):
        """Test sustainability dashboard generation."""
        # Add some test data
        sustainability_optimizer.create_carbon_budget("Test", 50.0, 30)
        
        dashboard = sustainability_optimizer.get_sustainability_dashboard()
        
        assert "dashboard_id" in dashboard
        assert "generated_at" in dashboard
        assert "overview" in dashboard
        assert "goals_status" in dashboard
        assert "budget_status" in dashboard
        assert "sustainability_score" in dashboard


class TestEnterpriseMonitoring:
    """Test enterprise monitoring functionality."""
    
    @pytest.fixture
    def enterprise_monitor(self):
        return EnterpriseMonitor()
    
    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector(max_history=100)
    
    @pytest.fixture
    def alert_manager(self):
        return AlertManager()
    
    @pytest.fixture
    def sample_metrics(self):
        return {
            "co2_kg": 2.5,
            "energy_kwh": 6.0,
            "gpu_utilization": 85.0,
            "training_throughput": 120.0,
            "samples_per_kwh": 2000.0
        }
    
    def test_metrics_collection(self, metrics_collector, sample_metrics):
        """Test metrics collection and storage."""
        labels = {"session_id": "test_session", "model_name": "test_model"}
        
        metrics_collector.record_metric(sample_metrics, labels, "test_session")
        
        assert len(metrics_collector.metrics_history) == 1
        snapshot = metrics_collector.metrics_history[0]
        assert isinstance(snapshot, MetricSnapshot)
        assert snapshot.session_id == "test_session"
        assert snapshot.metrics == sample_metrics
        assert snapshot.labels == labels
    
    def test_alert_rule_evaluation(self, alert_manager, sample_metrics):
        """Test alert rule evaluation."""
        # Should trigger high carbon emissions alert
        high_carbon_metrics = sample_metrics.copy()
        high_carbon_metrics["co2_kg"] = 15.0  # Exceeds 10kg threshold
        
        triggered_alerts = alert_manager.evaluate_metrics(high_carbon_metrics)
        
        assert len(triggered_alerts) > 0
        alert = triggered_alerts[0]
        assert isinstance(alert, Alert)
        assert alert.severity in ["warning", "critical"]
        assert alert.metric_name == "co2_kg"
        assert alert.current_value == 15.0
    
    def test_health_checker(self):
        """Test health checking functionality."""
        health_checker = HealthChecker()
        
        # Register a test health check
        def test_check():
            return True
        
        health_checker.register_check("test_check", test_check)
        
        # Run health checks
        results = health_checker.run_health_checks()
        
        assert "test_check" in results
        assert results["test_check"] is True
        
        # Get health status
        status = health_checker.get_health_status()
        assert status["overall_healthy"] is True
        assert "test_check" in status["checks"]
    
    def test_monitoring_dashboard(self, enterprise_monitor):
        """Test monitoring dashboard generation."""
        dashboard = enterprise_monitor.get_monitoring_dashboard()
        
        assert "monitoring_status" in dashboard
        assert "health_status" in dashboard
        assert "metrics_summary" in dashboard
        assert "alert_summary" in dashboard
        assert "active_alerts" in dashboard


class TestFaultTolerance:
    """Test fault tolerance functionality."""
    
    @pytest.fixture
    def fault_manager(self):
        return FaultToleranceManager()
    
    @pytest.fixture
    def retry_manager(self):
        return RetryManager()
    
    @pytest.fixture
    def circuit_breaker_manager(self):
        return CircuitBreakerManager()
    
    def test_failure_classification(self, fault_manager):
        """Test failure classification."""
        # Network timeout error
        network_error = ConnectionError("Connection timeout")
        failure_mode = fault_manager.classify_failure(network_error, {})
        assert failure_mode == FailureMode.NETWORK_TIMEOUT
        
        # Memory error
        memory_error = MemoryError("Out of memory")
        failure_mode = fault_manager.classify_failure(memory_error, {})
        assert failure_mode == FailureMode.MEMORY_ERROR
        
        # Unknown error
        unknown_error = ValueError("Some unknown error")
        failure_mode = fault_manager.classify_failure(unknown_error, {})
        assert failure_mode == FailureMode.UNKNOWN_ERROR
    
    def test_retry_manager_sync(self, retry_manager):
        """Test synchronous retry functionality."""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        # Should succeed after retries
        result = retry_manager.retry_sync(
            failing_function,
            strategy=RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF,
            config_name="network"
        )
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_manager_async(self, retry_manager):
        """Test asynchronous retry functionality."""
        call_count = 0
        
        async def failing_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            return "async_success"
        
        # Should succeed after retries
        result = await retry_manager.retry_async(
            failing_async_function,
            strategy=RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF
        )
        
        assert result == "async_success"
        assert call_count == 2
    
    def test_circuit_breaker(self, circuit_breaker_manager):
        """Test circuit breaker functionality."""
        call_count = 0
        
        def unreliable_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 5:  # Fail first 5 calls
                raise RuntimeError("Service unavailable")
            return "success"
        
        # Should fail and open circuit breaker
        for i in range(6):
            try:
                circuit_breaker_manager.execute_with_circuit_breaker(
                    "test_service",
                    unreliable_function,
                    failure_threshold=3,
                    timeout_seconds=1
                )
            except RuntimeError:
                pass  # Expected failures
        
        # Circuit should be open now
        breaker_status = circuit_breaker_manager.get_circuit_breaker_status()
        assert "test_service" in breaker_status
        assert breaker_status["test_service"]["state"] == "OPEN"
    
    def test_resilient_decorator(self, fault_manager):
        """Test resilient function decorator."""
        call_count = 0
        
        @resilient("test_component", recovery_strategy=RecoveryStrategy.IGNORE_AND_CONTINUE)
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return "success"
        
        # First call should be handled gracefully
        result = sometimes_failing_function()
        assert result is None  # IGNORE_AND_CONTINUE returns None
        
        # Second call should succeed
        result = sometimes_failing_function()
        assert result == "success"
    
    def test_get_resilience_dashboard(self, fault_manager):
        """Test resilience dashboard generation."""
        # Create some failure records
        fault_manager.handle_failure(
            ValueError("Test error"),
            "test_component",
            {"test": "context"}
        )
        
        dashboard = fault_manager.get_resilience_dashboard()
        
        assert "resilience_overview" in dashboard
        assert "failure_breakdown" in dashboard
        assert "recovery_strategies" in dashboard
        assert "circuit_breakers" in dashboard


class TestQuantumOptimizer:
    """Test quantum optimization functionality."""
    
    @pytest.fixture
    def quantum_orchestrator(self):
        return QuantumOptimizationOrchestrator()
    
    @pytest.fixture
    def quantum_annealer(self):
        return QuantumAnnealingOptimizer(problem_size=8)
    
    @pytest.fixture
    def quantum_genetic(self):
        return QuantumInspiredGeneticAlgorithm(
            population_size=20, num_parameters=5, mutation_rate=0.1
        )
    
    @pytest.fixture
    def sample_training_config(self):
        return {
            "dataset_size": 10000,
            "base_gpu_power": 250,
            "num_gpus": 1,
            "grid_carbon_intensity": 400,
            "base_samples_per_second": 100
        }
    
    @pytest.mark.asyncio
    async def test_quantum_annealing_optimization(self, quantum_annealer, sample_training_config):
        """Test quantum annealing optimization."""
        result = await quantum_annealer.optimize(
            sample_training_config,
            max_iterations=50  # Reduced for testing
        )
        
        assert "optimized_config" in result
        assert "best_energy" in result
        assert "total_iterations" in result
        assert "energy_history" in result
        assert "convergence_rate" in result
        assert "optimization_summary" in result
        
        # Check that energy improved
        initial_energy = result["energy_history"][0]
        final_energy = result["best_energy"]
        assert final_energy <= initial_energy
    
    @pytest.mark.asyncio
    async def test_quantum_genetic_optimization(self, quantum_genetic, sample_training_config):
        """Test quantum genetic algorithm optimization."""
        result = await quantum_genetic.optimize(
            sample_training_config,
            max_generations=10  # Reduced for testing
        )
        
        assert "optimized_config" in result
        assert "best_fitness" in result
        assert "total_generations" in result
        assert "generation_stats" in result
        assert "convergence_analysis" in result
        assert "optimization_summary" in result
        
        # Check that fitness improved over generations
        generation_stats = result["generation_stats"]
        if len(generation_stats) > 1:
            initial_fitness = generation_stats[0]["best_fitness"]
            final_fitness = generation_stats[-1]["best_fitness"]
            assert final_fitness <= initial_fitness
    
    @pytest.mark.asyncio
    async def test_optimization_orchestrator(self, quantum_orchestrator, sample_training_config):
        """Test quantum optimization orchestrator."""
        # Test quantum annealing
        result = await quantum_orchestrator.optimize_training_config(
            sample_training_config,
            optimization_method="quantum_annealing",
            max_iterations=30
        )
        
        assert "optimized_config" in result
        assert "optimization_method" in result
        assert "optimization_time_seconds" in result
        assert "quantum_advantages" in result
        
        # Check optimization history
        assert len(quantum_orchestrator.optimization_history) == 1
        
        # Test summary generation
        summary = quantum_orchestrator.get_optimization_summary()
        assert "total_optimizations" in summary
        assert "best_overall_result" in summary
        assert "quantum_advantages_summary" in summary


class TestAdaptiveScaling:
    """Test adaptive scaling functionality."""
    
    @pytest.fixture
    def adaptive_config(self):
        return AdaptiveConfig(
            min_batch_size=8,
            max_batch_size=128,
            adaptation_interval_seconds=1  # Fast for testing
        )
    
    @pytest.fixture
    def resource_monitor(self):
        return ResourceMonitor(monitoring_interval=0.1)
    
    @pytest.fixture
    def adaptive_scaler(self, adaptive_config):
        return AdaptiveScaler(adaptive_config)
    
    def test_resource_monitor_metrics_collection(self, resource_monitor):
        """Test resource monitoring metrics collection."""
        # Start monitoring briefly
        resource_monitor.start_monitoring()
        time.sleep(0.5)  # Let it collect a few metrics
        resource_monitor.stop_monitoring()
        
        assert resource_monitor.current_metrics is not None
        assert len(resource_monitor.metrics_buffer) > 0
        
        # Check metrics structure
        metrics = resource_monitor.current_metrics
        assert 0 <= metrics.cpu_utilization <= 1
        assert 0 <= metrics.memory_utilization <= 1
        assert 0 <= metrics.gpu_utilization <= 1
        assert metrics.training_throughput > 0
        assert metrics.carbon_efficiency > 0
    
    def test_scaling_decision_logic(self, adaptive_scaler):
        """Test scaling decision logic."""
        # Mock high GPU utilization scenario
        from hf_eco2ai.adaptive_scaling import ScalingMetrics
        
        high_util_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=0.6,
            memory_utilization=0.7,
            gpu_utilization=0.95,  # High utilization
            gpu_memory_utilization=0.6,
            training_throughput=100.0,
            carbon_efficiency=800.0,
            cost_efficiency=120.0,
            queue_length=2,
            response_time_ms=150.0
        )
        
        # Simulate metrics update
        actions = adaptive_scaler.decision_engine._analyze_and_decide(high_util_metrics)
        
        # Should generate scale-up actions
        assert len(actions) > 0
        scale_up_actions = [a for a in actions if a.action_type == "scale_up"]
        assert len(scale_up_actions) > 0
    
    def test_adaptive_config_validation(self, adaptive_config):
        """Test adaptive configuration validation."""
        assert adaptive_config.min_batch_size < adaptive_config.max_batch_size
        assert adaptive_config.min_workers < adaptive_config.max_workers
        assert 0 < adaptive_config.target_gpu_utilization < 1
        assert 0 < adaptive_config.scale_up_threshold < 1
        assert 0 < adaptive_config.scale_down_threshold < adaptive_config.scale_up_threshold
    
    def test_scaling_dashboard(self, adaptive_scaler):
        """Test scaling dashboard generation."""
        dashboard = adaptive_scaler.get_scaling_dashboard()
        
        assert "scaling_status" in dashboard
        assert "current_metrics" in dashboard
        assert "recent_actions" in dashboard
        assert "optimization_insights" in dashboard
        assert "performance_trends" in dashboard


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for advanced features working together."""
    
    @pytest.mark.asyncio
    async def test_full_intelligence_pipeline(self):
        """Test complete carbon intelligence pipeline."""
        # Setup components
        intelligence_engine = CarbonIntelligenceEngine()
        sustainability_optimizer = SustainabilityOptimizer()
        enterprise_monitor = EnterpriseMonitor()
        
        # Simulate training session
        training_config = {
            "batch_size": 32,
            "learning_rate": 5e-5,
            "num_epochs": 5,
            "fp16": False,
            "model_name": "test-model"
        }
        
        training_metrics = {
            "energy_kwh": 8.5,
            "co2_kg": 3.4,
            "samples_processed": 20000,
            "gpu_utilization": 0.82,
            "training_duration_hours": 3.2
        }
        
        # Generate insights
        insights = await intelligence_engine.analyze_training_session(
            training_metrics, training_config
        )
        assert len(insights) > 0
        
        # Generate optimization strategies
        strategies = await sustainability_optimizer.analyze_sustainability_opportunities(
            training_metrics, training_config
        )
        assert len(strategies) > 0
        
        # Record metrics for monitoring
        enterprise_monitor.record_training_metrics(training_metrics)
        
        # Generate reports
        intelligence_report = intelligence_engine.generate_intelligence_report()
        sustainability_dashboard = sustainability_optimizer.get_sustainability_dashboard()
        monitoring_dashboard = enterprise_monitor.get_monitoring_dashboard()
        
        # Verify integration
        assert intelligence_report["summary"]["total_insights"] > 0
        assert sustainability_dashboard["overview"]["total_strategies"] > 0
        assert "monitoring_status" in monitoring_dashboard
    
    @pytest.mark.asyncio
    async def test_fault_tolerant_optimization(self):
        """Test optimization with fault tolerance."""
        fault_manager = FaultToleranceManager()
        quantum_optimizer = QuantumOptimizationOrchestrator()
        
        # Create a resilient optimization function
        @resilient("quantum_optimization", recovery_strategy=RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF)
        async def resilient_optimize(config):
            return await quantum_optimizer.optimize_training_config(
                config, "quantum_annealing", max_iterations=20
            )
        
        training_config = {
            "dataset_size": 5000,
            "base_gpu_power": 200,
            "num_gpus": 1,
            "grid_carbon_intensity": 350
        }
        
        # Should complete successfully with fault tolerance
        result = await resilient_optimize(training_config)
        
        assert "optimized_config" in result
        assert "quantum_advantages" in result
    
    def test_comprehensive_monitoring_with_alerts(self):
        """Test comprehensive monitoring with alerting."""
        enterprise_monitor = EnterpriseMonitor()
        enterprise_monitor.start_monitoring()
        
        try:
            # Simulate high carbon emissions
            high_carbon_metrics = {
                "co2_kg": 25.0,  # Exceeds alert threshold
                "energy_kwh": 60.0,
                "gpu_utilization": 95.0,
                "samples_per_kwh": 800.0  # Below efficiency threshold
            }
            
            enterprise_monitor.record_training_metrics(high_carbon_metrics)
            
            # Check that alerts were triggered
            active_alerts = enterprise_monitor.alert_manager.get_active_alerts()
            assert len(active_alerts) > 0
            
            # Verify alert types
            alert_severities = [alert.severity for alert in active_alerts]
            assert "critical" in alert_severities or "warning" in alert_severities
            
        finally:
            enterprise_monitor.stop_monitoring()


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for advanced features."""
    
    @pytest.mark.asyncio
    async def test_carbon_intelligence_performance(self):
        """Benchmark carbon intelligence engine performance."""
        engine = CarbonIntelligenceEngine()
        
        metrics = {
            "energy_kwh": 10.0,
            "co2_kg": 4.0,
            "samples_processed": 25000,
            "gpu_utilization": 0.88
        }
        
        config = {"batch_size": 64, "learning_rate": 3e-5, "num_epochs": 8}
        
        start_time = time.time()
        insights = await engine.analyze_training_session(metrics, config)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Should complete analysis in reasonable time
        assert analysis_time < 5.0  # Less than 5 seconds
        assert len(insights) > 0
    
    @pytest.mark.asyncio
    async def test_quantum_optimization_performance(self):
        """Benchmark quantum optimization performance."""
        optimizer = QuantumOptimizationOrchestrator()
        
        config = {
            "dataset_size": 15000,
            "base_gpu_power": 300,
            "num_gpus": 2,
            "grid_carbon_intensity": 450
        }
        
        start_time = time.time()
        result = await optimizer.optimize_training_config(
            config, "quantum_annealing", max_iterations=100
        )
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        # Should complete optimization in reasonable time
        assert optimization_time < 30.0  # Less than 30 seconds
        assert "optimized_config" in result
        assert result["best_energy"] < float('inf')
    
    def test_metrics_collection_throughput(self):
        """Benchmark metrics collection throughput."""
        collector = MetricsCollector(max_history=1000)
        
        sample_metrics = {
            "co2_kg": 1.5,
            "energy_kwh": 3.8,
            "throughput": 150.0
        }
        
        labels = {"session": "perf_test", "model": "benchmark"}
        
        start_time = time.time()
        
        # Record many metrics quickly
        for i in range(1000):
            collector.record_metric(sample_metrics, labels, f"session_{i}")
        
        end_time = time.time()
        collection_time = end_time - start_time
        
        # Should handle high-throughput metrics collection
        throughput = 1000 / collection_time
        assert throughput > 100  # At least 100 metrics per second
        assert len(collector.metrics_history) == 1000


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == "intelligence":
            pytest.main(["-v", "TestCarbonIntelligenceEngine"])
        elif test_category == "sustainability":
            pytest.main(["-v", "TestSustainabilityOptimizer"])
        elif test_category == "monitoring":
            pytest.main(["-v", "TestEnterpriseMonitoring"])
        elif test_category == "fault_tolerance":
            pytest.main(["-v", "TestFaultTolerance"])
        elif test_category == "quantum":
            pytest.main(["-v", "TestQuantumOptimizer"])
        elif test_category == "scaling":
            pytest.main(["-v", "TestAdaptiveScaling"])
        elif test_category == "integration":
            pytest.main(["-v", "-m", "integration"])
        elif test_category == "performance":
            pytest.main(["-v", "-m", "performance"])
        else:
            print("Unknown test category. Running all tests.")
            pytest.main(["-v"])
    else:
        # Run all tests
        pytest.main(["-v", "--tb=short"])