"""Comprehensive tests for quantum optimization features."""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.hf_eco2ai.quantum_planner import (
    QuantumInspiredTaskPlanner,
    QuantumState,
    QuantumCircuit,
    QuantumInspiredTaskPlanner
)
from src.hf_eco2ai.config import CarbonConfig
from src.hf_eco2ai.advanced_optimization import AdvancedModelOptimizer


class TestQuantumState:
    """Test quantum state functionality."""
    
    def test_quantum_state_creation(self):
        """Test quantum state creation and properties."""
        config = {"batch_size": 32, "learning_rate": 1e-4}
        state = QuantumState(
            configuration=config,
            amplitude=complex(0.7, 0.3),
            energy=1.5,
            carbon_score=0.8,
            performance_score=0.9
        )
        
        assert state.configuration == config
        assert abs(state.amplitude - complex(0.7, 0.3)) < 1e-10
        assert state.energy == 1.5
        assert state.carbon_score == 0.8
        assert state.performance_score == 0.9
    
    def test_probability_calculation(self):
        """Test quantum probability calculation."""
        state = QuantumState(
            configuration={},
            amplitude=complex(0.6, 0.8),  # |amplitude|^2 = 0.36 + 0.64 = 1.0
            energy=1.0,
            carbon_score=0.5,
            performance_score=0.5
        )
        
        probability = state.probability()
        assert abs(probability - 1.0) < 1e-10
    
    def test_total_score_calculation(self):
        """Test total optimization score calculation."""
        state = QuantumState(
            configuration={},
            amplitude=complex(1.0, 0.0),
            energy=2.0,
            carbon_score=0.8,
            performance_score=0.6
        )
        
        expected_score = (0.8 * 0.4 + 0.6 * 0.3 + (1.0 / 2.0) * 0.3)
        assert abs(state.total_score() - expected_score) < 1e-10


class TestQuantumCircuit:
    """Test quantum circuit functionality."""
    
    def test_circuit_creation(self):
        """Test quantum circuit creation."""
        states = [
            QuantumState({}, complex(0.7, 0.0), 1.0, 0.5, 0.5),
            QuantumState({}, complex(0.7, 0.0), 2.0, 0.6, 0.4)
        ]
        
        entanglement_matrix = np.eye(2)
        circuit = QuantumCircuit(states=states, entanglement_matrix=entanglement_matrix)
        
        assert len(circuit.states) == 2
        assert circuit.entanglement_matrix.shape == (2, 2)
    
    def test_normalization(self):
        """Test quantum state normalization."""
        states = [
            QuantumState({}, complex(2.0, 0.0), 1.0, 0.5, 0.5),
            QuantumState({}, complex(2.0, 0.0), 2.0, 0.6, 0.4)
        ]
        
        circuit = QuantumCircuit(states=states, entanglement_matrix=np.eye(2))
        circuit.normalize()
        
        total_probability = sum(state.probability() for state in circuit.states)
        assert abs(total_probability - 1.0) < 1e-10
    
    def test_interference(self):
        """Test quantum interference application."""
        states = [
            QuantumState({}, complex(1.0, 0.0), 1.0, 0.9, 0.5),  # High carbon score
            QuantumState({}, complex(1.0, 0.0), 2.0, 0.1, 0.4)   # Low carbon score
        ]
        
        circuit = QuantumCircuit(states=states, entanglement_matrix=np.eye(2))
        original_amplitudes = [abs(state.amplitude) for state in circuit.states]
        
        circuit.apply_interference("carbon_score")
        
        new_amplitudes = [abs(state.amplitude) for state in circuit.states]
        
        # High carbon score state should have increased amplitude
        assert new_amplitudes[0] > original_amplitudes[0]
    
    def test_measurement(self):
        """Test quantum measurement."""
        states = [
            QuantumState({"config": "a"}, complex(0.8, 0.0), 1.0, 0.5, 0.5),
            QuantumState({"config": "b"}, complex(0.6, 0.0), 2.0, 0.6, 0.4)
        ]
        
        circuit = QuantumCircuit(states=states, entanglement_matrix=np.eye(2))
        circuit.normalize()
        
        measured_states = circuit.measure(num_measurements=1)
        
        assert len(measured_states) == 1
        assert measured_states[0] in states
        assert len(circuit.measurement_history) == 1


class TestQuantumPlanner:
    """Test quantum-inspired task planner."""
    
    @pytest.fixture
    def planner(self):
        """Create quantum planner instance."""
        config = CarbonConfig()
        return QuantumInspiredTaskPlanner(config)
    
    @pytest.fixture
    def sample_requirements(self):
        """Sample task requirements."""
        return {
            "model_type": "transformer",
            "model_parameters": 100_000_000,
            "dataset_size": 10000,
            "target_accuracy": 0.95,
            "estimated_duration_hours": 2.0
        }
    
    @pytest.fixture
    def sample_constraints(self):
        """Sample task constraints."""
        return {
            "max_co2_kg": 5.0,
            "max_gpus": 4,
            "max_batch_size": 64
        }
    
    def test_planner_initialization(self, planner):
        """Test quantum planner initialization."""
        assert planner.config is not None
        assert planner.quantum_circuit is None
        assert planner.optimization_history == []
        assert planner.num_qubits == 8
        assert planner.max_iterations == 100
    
    def test_optimization_task(self, planner, sample_requirements, sample_constraints):
        """Test complete optimization task."""
        plan = planner.optimize_task(sample_requirements, sample_constraints)
        
        assert "optimal_configuration" in plan
        assert "quantum_optimization_score" in plan
        assert "energy_savings_percent" in plan
        assert "carbon_savings_percent" in plan
        assert "quantum_metrics" in plan
        
        # Check quantum metrics
        quantum_metrics = plan["quantum_metrics"]
        assert "amplitude" in quantum_metrics
        assert "probability" in quantum_metrics
        assert "coherence_score" in quantum_metrics
        assert "entanglement_score" in quantum_metrics
        
        # Verify optimization history is updated
        assert len(planner.optimization_history) == 1
    
    def test_quantum_sampling(self, planner, sample_requirements):
        """Test advanced quantum sampling."""
        parameter_spaces = {
            "batch_size": [8, 16, 32, 64],
            "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4],
            "precision": ["fp32", "fp16", "bf16"]
        }
        
        configs = planner._quantum_sampling(parameter_spaces, 20, sample_requirements)
        
        assert len(configs) == 20
        
        # Check all configs have valid values
        for config in configs:
            assert config["batch_size"] in parameter_spaces["batch_size"]
            assert config["learning_rate"] in parameter_spaces["learning_rate"]
            assert config["precision"] in parameter_spaces["precision"]
    
    def test_known_good_patterns(self, planner, sample_requirements):
        """Test known good pattern generation."""
        patterns = planner._get_known_good_patterns(sample_requirements)
        
        assert len(patterns) > 0
        
        # Should have patterns based on model size
        model_patterns = [p for p in patterns if "batch_size" in p and "learning_rate" in p]
        assert len(model_patterns) > 0
        
        # Should have carbon-optimal patterns
        carbon_patterns = [p for p in patterns if "start_time_hour" in p]
        assert len(carbon_patterns) > 0
    
    def test_energy_score_calculation(self, planner, sample_requirements):
        """Test energy score calculation."""
        config = {
            "batch_size": 32,
            "precision": "fp16",
            "gpu_count": 2
        }
        
        energy_score = planner._calculate_energy_score(config, sample_requirements)
        
        assert isinstance(energy_score, float)
        assert energy_score > 0
    
    def test_carbon_score_calculation(self, planner, sample_requirements):
        """Test carbon score calculation."""
        config = {
            "start_time_hour": 2,  # Low carbon hour
            "precision": "fp16",   # Energy efficient
            "batch_size": 64      # Efficient batch size
        }
        
        carbon_score = planner._calculate_carbon_score(config, sample_requirements)
        
        assert isinstance(carbon_score, float)
        assert 0 <= carbon_score <= 1
    
    def test_performance_score_calculation(self, planner, sample_requirements):
        """Test performance score calculation."""
        config = {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "gpu_count": 2
        }
        
        performance_score = planner._calculate_performance_score(config, sample_requirements)
        
        assert isinstance(performance_score, float)
        assert 0 <= performance_score <= 1
    
    def test_coherence_calculation(self, planner):
        """Test quantum coherence calculation."""
        # Initialize a quantum circuit first
        planner.quantum_circuit = QuantumCircuit(
            states=[
                QuantumState({}, complex(0.7, 0.0), 1.0, 0.5, 0.5),
                QuantumState({}, complex(0.7, 0.0), 2.0, 0.6, 0.4)
            ],
            entanglement_matrix=np.eye(2)
        )
        
        coherence = planner._calculate_quantum_coherence()
        
        assert isinstance(coherence, float)
        assert 0 <= coherence <= 1
    
    def test_entanglement_calculation(self, planner):
        """Test quantum entanglement calculation."""
        # Initialize a quantum circuit first
        planner.quantum_circuit = QuantumCircuit(
            states=[
                QuantumState({"batch_size": 32, "learning_rate": 1e-4}, complex(0.7, 0.0), 1.0, 0.5, 0.5),
                QuantumState({"batch_size": 64, "learning_rate": 2e-4}, complex(0.7, 0.0), 2.0, 0.6, 0.4)
            ],
            entanglement_matrix=np.array([[1.0, 0.1], [0.1, 1.0]])
        )
        
        entanglement = planner._calculate_entanglement_measure()
        
        assert isinstance(entanglement, float)
        assert entanglement >= 0


class TestAdvancedOptimization:
    """Test advanced model optimization features."""
    
    @pytest.fixture
    def optimizer(self):
        """Create model optimizer instance."""
        config = CarbonConfig()
        return AdvancedModelOptimizer(config)
    
    @pytest.fixture
    def sample_model_info(self):
        """Sample model information."""
        return {
            "parameters": 100_000_000,
            "type": "transformer"
        }
    
    @pytest.fixture
    def sample_performance_requirements(self):
        """Sample performance requirements."""
        return {
            "min_accuracy": 0.9,
            "max_latency_ms": 100
        }
    
    @pytest.mark.asyncio
    async def test_optimize_model_architecture(self, optimizer, sample_model_info, sample_performance_requirements):
        """Test model architecture optimization."""
        result = await optimizer.optimize_model_architecture(
            sample_model_info,
            sample_performance_requirements,
            carbon_budget=10.0
        )
        
        assert result.original_model_size == 100_000_000
        assert result.optimized_model_size < result.original_model_size
        assert result.compression_ratio > 1.0
        assert result.estimated_speedup >= 1.0
        assert 0 <= result.estimated_energy_savings <= 1.0
        assert -1.0 <= result.accuracy_impact <= 1.0
        assert len(result.optimization_techniques) > 0
        assert result.implementation_complexity in ["easy", "medium", "hard"]
    
    def test_quantization_analysis(self, optimizer, sample_model_info):
        """Test quantization analysis."""
        results = optimizer._analyze_quantization(sample_model_info, 0.9)
        
        assert len(results) > 0
        
        for result in results:
            assert "quantization" in result["name"]
            assert result["technique"] == "quantization"
            assert result["speedup"] >= 1.0
            assert 0 <= result["energy_savings"] <= 1.0
            assert result["feasible"] == True
    
    def test_pruning_analysis(self, optimizer, sample_model_info):
        """Test pruning analysis."""
        results = optimizer._analyze_pruning(sample_model_info, 0.9)
        
        assert len(results) > 0
        
        for result in results:
            assert "pruning" in result["name"]
            assert result["technique"] == "pruning"
            assert result["speedup"] >= 1.0
            assert 0 <= result["energy_savings"] <= 1.0
    
    def test_distillation_analysis(self, optimizer, sample_model_info):
        """Test distillation analysis."""
        # Test with large model (distillation is most effective for large models)
        large_model_info = {"parameters": 1_000_000_000, "type": "transformer"}
        
        results = optimizer._analyze_distillation(large_model_info, 0.9)
        
        assert len(results) > 0
        
        for result in results:
            assert "distillation" in result["name"]
            assert result["technique"] == "distillation"
            assert result["speedup"] >= 1.0
            assert 0 <= result["energy_savings"] <= 1.0
    
    def test_implementation_guide_generation(self, optimizer):
        """Test implementation guide generation."""
        from src.hf_eco2ai.advanced_optimization import ModelOptimizationResult
        
        result = ModelOptimizationResult(
            original_model_size=100_000_000,
            optimized_model_size=50_000_000,
            compression_ratio=2.0,
            estimated_speedup=1.8,
            estimated_energy_savings=0.4,
            accuracy_impact=-0.02,
            optimization_techniques=["quantization_int8", "pruning_structured"],
            implementation_complexity="medium"
        )
        
        guide = optimizer.generate_implementation_guide(result)
        
        assert "overview" in guide
        assert "implementation_steps" in guide
        assert "code_examples" in guide
        assert "validation_checklist" in guide
        assert "monitoring_recommendations" in guide
        
        # Check overview
        overview = guide["overview"]
        assert overview["compression_ratio"] == 2.0
        assert overview["estimated_speedup"] == 1.8
        
        # Check implementation steps
        assert len(guide["implementation_steps"]) > 0
        
        # Check code examples
        assert "quantization" in guide["code_examples"]
        assert "pruning" in guide["code_examples"]
        
        # Check validation checklist
        assert len(guide["validation_checklist"]) > 0
        
        # Check monitoring recommendations
        assert len(guide["monitoring_recommendations"]) > 0


@pytest.mark.benchmark
class TestQuantumPerformance:
    """Performance benchmarks for quantum optimization."""
    
    def test_quantum_state_creation_benchmark(self, benchmark):
        """Benchmark quantum state creation."""
        def create_quantum_state():
            return QuantumState(
                configuration={"batch_size": 32, "learning_rate": 1e-4},
                amplitude=complex(0.7, 0.3),
                energy=1.5,
                carbon_score=0.8,
                performance_score=0.9
            )
        
        result = benchmark(create_quantum_state)
        assert result is not None
    
    def test_quantum_circuit_optimization_benchmark(self, benchmark):
        """Benchmark quantum circuit optimization."""
        def optimize_circuit():
            states = [
                QuantumState({f"param_{i}": i}, complex(np.random.random(), np.random.random()), 
                           np.random.random(), np.random.random(), np.random.random())
                for i in range(100)
            ]
            
            circuit = QuantumCircuit(states=states, entanglement_matrix=np.random.random((100, 100)))
            
            # Run optimization steps
            for _ in range(10):
                circuit.apply_interference("carbon_score")
                circuit.normalize()
            
            return circuit.measure(num_measurements=5)
        
        results = benchmark(optimize_circuit)
        assert len(results) == 5
    
    def test_large_scale_optimization_benchmark(self, benchmark):
        """Benchmark large-scale optimization task."""
        def large_optimization():
            config = CarbonConfig()
            planner = QuantumInspiredTaskPlanner(config)
            
            requirements = {
                "model_type": "transformer",
                "model_parameters": 1_000_000_000,  # Large model
                "dataset_size": 100000,
                "target_accuracy": 0.95,
                "estimated_duration_hours": 8.0
            }
            
            constraints = {
                "max_co2_kg": 50.0,
                "max_gpus": 8,
                "max_batch_size": 128
            }
            
            return planner.optimize_task(requirements, constraints)
        
        plan = benchmark(large_optimization)
        assert "optimal_configuration" in plan


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])