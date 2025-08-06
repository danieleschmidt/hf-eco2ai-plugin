"""Quantum-Inspired Task Planner for Carbon-Efficient ML Training.

This module implements quantum-inspired algorithms for optimizing ML training
workloads across multiple dimensions: energy efficiency, carbon footprint,
performance, and cost.

Key quantum concepts used:
- Superposition: Evaluating multiple configurations simultaneously
- Entanglement: Coupling related optimization parameters
- Quantum Annealing: Finding global optima in complex solution spaces
- Interference: Amplifying optimal solutions while canceling suboptimal ones
"""

import time
import logging
import numpy as np
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import math
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue
import threading

from .config import CarbonConfig
from .models import CarbonMetrics, OptimizationRecommendation
from .utils import get_carbon_intensity_by_time, estimate_training_time

logger = logging.getLogger(__name__)


@dataclass
class QuantumState:
    """Represents a quantum state in the optimization space."""
    
    configuration: Dict[str, Any]
    amplitude: complex  # Quantum amplitude
    energy: float  # Energy level of this state
    carbon_score: float  # Carbon efficiency score
    performance_score: float  # Performance score
    entangled_params: Set[str] = field(default_factory=set)
    
    def probability(self) -> float:
        """Calculate probability of measuring this state."""
        return abs(self.amplitude) ** 2
    
    def total_score(self) -> float:
        """Calculate total optimization score."""
        return (self.carbon_score * 0.4 + 
                self.performance_score * 0.3 + 
                (1.0 / max(self.energy, 0.001)) * 0.3)


@dataclass
class QuantumCircuit:
    """Quantum circuit for optimization operations."""
    
    states: List[QuantumState]
    entanglement_matrix: np.ndarray
    measurement_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def normalize(self):
        """Normalize quantum amplitudes."""
        total_prob = sum(state.probability() for state in self.states)
        if total_prob > 0:
            normalization_factor = 1.0 / math.sqrt(total_prob)
            for state in self.states:
                state.amplitude *= normalization_factor
    
    def apply_interference(self, target_metric: str = "carbon_score"):
        """Apply quantum interference to amplify optimal states."""
        # Sort states by target metric
        sorted_states = sorted(self.states, 
                             key=lambda s: getattr(s, target_metric), 
                             reverse=True)
        
        # Apply constructive interference to top states
        top_third = len(sorted_states) // 3
        
        for i, state in enumerate(sorted_states):
            if i < top_third:
                # Constructive interference - amplify good states
                phase_boost = math.pi / 4
                state.amplitude *= complex(math.cos(phase_boost), math.sin(phase_boost))
            elif i > 2 * top_third:
                # Destructive interference - diminish poor states
                phase_reduction = -math.pi / 4
                state.amplitude *= complex(math.cos(phase_reduction), math.sin(phase_reduction))
        
        self.normalize()
    
    def measure(self, num_measurements: int = 1) -> List[QuantumState]:
        """Perform quantum measurement to collapse to classical states."""
        probabilities = [state.probability() for state in self.states]
        
        if sum(probabilities) == 0:
            return self.states[:num_measurements]  # Fallback
        
        # Weighted random selection based on quantum probabilities
        selected_indices = np.random.choice(
            len(self.states), 
            size=num_measurements, 
            p=np.array(probabilities) / sum(probabilities),
            replace=False
        )
        
        measured_states = [self.states[i] for i in selected_indices]
        
        # Record measurement
        self.measurement_history.append({
            "timestamp": time.time(),
            "num_measurements": num_measurements,
            "selected_indices": selected_indices.tolist(),
            "probabilities": probabilities
        })
        
        return measured_states


class QuantumInspiredTaskPlanner:
    """Quantum-inspired planner for optimizing ML training tasks."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize quantum-inspired task planner.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        self.quantum_circuit: Optional[QuantumCircuit] = None
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Quantum parameters
        self.num_qubits = 8  # Configuration space dimensions
        self.max_iterations = 100
        self.convergence_threshold = 0.01
        
        # Entanglement relationships between parameters
        self.entanglement_map = {
            "batch_size": ["learning_rate", "gradient_accumulation"],
            "learning_rate": ["warmup_steps", "weight_decay"],
            "sequence_length": ["batch_size", "memory_usage"],
            "precision": ["batch_size", "memory_usage"],
            "carbon_intensity": ["start_time", "duration"],
            "gpu_count": ["batch_size", "communication_overhead"]
        }
        
        logger.info("Initialized quantum-inspired task planner")
    
    def plan_optimal_training(self, 
                            task_requirements: Dict[str, Any],
                            constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Plan optimal training configuration using quantum-inspired optimization.
        
        Args:
            task_requirements: Training task requirements
            constraints: Optimization constraints (budget, time, etc.)
            
        Returns:
            Optimal training plan with quantum-optimized configuration
        """
        logger.info("Starting quantum-inspired training optimization")
        
        constraints = constraints or {}
        
        # Initialize quantum superposition of configurations
        self._initialize_quantum_superposition(task_requirements, constraints)
        
        # Apply quantum optimization
        optimal_states = self._quantum_optimize()
        
        # Measure final quantum state
        final_configuration = self.quantum_circuit.measure(num_measurements=3)
        
        # Select best configuration
        best_config = max(final_configuration, key=lambda s: s.total_score())
        
        # Generate comprehensive plan
        plan = self._generate_training_plan(best_config, task_requirements, constraints)
        
        # Record optimization
        self.optimization_history.append({
            "timestamp": time.time(),
            "task_requirements": task_requirements,
            "constraints": constraints,
            "optimization_score": best_config.total_score(),
            "carbon_savings_estimate": plan["carbon_savings_percent"],
            "energy_savings_estimate": plan["energy_savings_percent"]
        })
        
        logger.info(f"Quantum optimization complete. Carbon savings: {plan['carbon_savings_percent']:.1f}%")
        
        return plan
    
    def _initialize_quantum_superposition(self, 
                                        requirements: Dict[str, Any],
                                        constraints: Dict[str, Any]):
        """Initialize quantum superposition of training configurations."""
        logger.debug("Initializing quantum superposition")
        
        # Define parameter spaces for superposition
        parameter_spaces = {
            "batch_size": [8, 16, 32, 64, 128, 256],
            "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
            "precision": ["fp32", "fp16", "bf16"],
            "gradient_accumulation_steps": [1, 2, 4, 8],
            "warmup_steps": [100, 500, 1000, 2000],
            "weight_decay": [0.0, 0.01, 0.1],
            "start_time_hour": list(range(24)),
            "gpu_count": [1, 2, 4, 8] if constraints.get("max_gpus", 1) > 1 else [1]
        }
        
        # Filter by constraints
        if "max_batch_size" in constraints:
            parameter_spaces["batch_size"] = [
                b for b in parameter_spaces["batch_size"] 
                if b <= constraints["max_batch_size"]
            ]
        
        if "max_gpus" in constraints:
            parameter_spaces["gpu_count"] = [
                g for g in parameter_spaces["gpu_count"]
                if g <= constraints["max_gpus"]
            ]
        
        # Generate quantum states using tensor product
        states = []
        
        # Sample configurations from parameter space (quantum superposition)
        num_states = min(64, np.prod([len(space) for space in parameter_spaces.values()]))
        
        for _ in range(num_states):
            config = {}
            for param, space in parameter_spaces.items():
                config[param] = np.random.choice(space)
            
            # Calculate state properties
            energy = self._calculate_energy_score(config, requirements)
            carbon_score = self._calculate_carbon_score(config, requirements)
            performance_score = self._calculate_performance_score(config, requirements)
            
            # Initial quantum amplitude (uniform superposition)
            amplitude = complex(1.0 / math.sqrt(num_states), 0)
            
            state = QuantumState(
                configuration=config,
                amplitude=amplitude,
                energy=energy,
                carbon_score=carbon_score,
                performance_score=performance_score
            )
            
            states.append(state)
        
        # Create entanglement matrix
        entanglement_matrix = self._create_entanglement_matrix(len(states))
        
        self.quantum_circuit = QuantumCircuit(
            states=states,
            entanglement_matrix=entanglement_matrix
        )
        
        logger.debug(f"Initialized {len(states)} quantum states")
    
    def _calculate_energy_score(self, config: Dict[str, Any], 
                              requirements: Dict[str, Any]) -> float:
        """Calculate energy efficiency score for configuration."""
        base_energy = 1.0
        
        # Batch size impact
        batch_size = config.get("batch_size", 32)
        base_energy *= (batch_size / 32) ** 0.7  # Larger batches more efficient
        
        # Precision impact
        precision = config.get("precision", "fp32")
        precision_multiplier = {"fp32": 1.0, "fp16": 0.6, "bf16": 0.65}
        base_energy *= precision_multiplier.get(precision, 1.0)
        
        # GPU scaling
        gpu_count = config.get("gpu_count", 1)
        if gpu_count > 1:
            # Communication overhead
            efficiency = 0.9 ** (gpu_count - 1)
            base_energy *= efficiency
        
        # Gradient accumulation efficiency
        grad_acc = config.get("gradient_accumulation_steps", 1)
        if grad_acc > 1:
            base_energy *= 0.95  # Slight efficiency gain
        
        return base_energy
    
    def _calculate_carbon_score(self, config: Dict[str, Any], 
                               requirements: Dict[str, Any]) -> float:
        """Calculate carbon efficiency score for configuration."""
        start_hour = config.get("start_time_hour", 12)
        duration_hours = requirements.get("estimated_duration_hours", 2.0)
        region = requirements.get("region", "USA")
        
        # Calculate average carbon intensity
        total_intensity = 0
        for hour_offset in range(int(duration_hours) + 1):
            hour = (start_hour + hour_offset) % 24
            intensity = get_carbon_intensity_by_time(region, hour)
            total_intensity += intensity
        
        avg_intensity = total_intensity / (int(duration_hours) + 1)
        
        # Lower carbon intensity = higher score
        carbon_score = 1000 / max(avg_intensity, 100)
        
        # Apply configuration-specific carbon multipliers
        energy_score = self._calculate_energy_score(config, requirements)
        carbon_score /= energy_score  # More efficient configs have better carbon scores
        
        return carbon_score
    
    def _calculate_performance_score(self, config: Dict[str, Any], 
                                    requirements: Dict[str, Any]) -> float:
        """Calculate performance score for configuration."""
        base_performance = 1.0
        
        # Batch size vs memory trade-off
        batch_size = config.get("batch_size", 32)
        optimal_batch = 64  # Assume optimal around 64
        batch_efficiency = 1.0 - abs(batch_size - optimal_batch) / optimal_batch * 0.3
        base_performance *= max(batch_efficiency, 0.3)
        
        # Learning rate impact
        learning_rate = config.get("learning_rate", 5e-5)
        optimal_lr = 5e-5
        lr_ratio = learning_rate / optimal_lr
        if lr_ratio > 1:
            lr_efficiency = 1.0 / lr_ratio
        else:
            lr_efficiency = lr_ratio
        base_performance *= max(lr_efficiency, 0.1)
        
        # GPU scaling efficiency
        gpu_count = config.get("gpu_count", 1)
        if gpu_count > 1:
            # Assume diminishing returns
            scaling_efficiency = math.log2(gpu_count) / gpu_count * 0.8 + 0.2
            base_performance *= scaling_efficiency
        
        # Precision vs accuracy trade-off
        precision = config.get("precision", "fp32")
        precision_performance = {"fp32": 1.0, "fp16": 0.95, "bf16": 0.98}
        base_performance *= precision_performance.get(precision, 1.0)
        
        return base_performance
    
    def _create_entanglement_matrix(self, num_states: int) -> np.ndarray:
        """Create quantum entanglement matrix for parameter coupling."""
        matrix = np.eye(num_states, dtype=complex)
        
        # Add entanglement between related parameters
        for i in range(num_states):
            for j in range(i + 1, num_states):
                state_i = self.quantum_circuit.states[i] if self.quantum_circuit else None
                state_j = self.quantum_circuit.states[j] if self.quantum_circuit else None
                
                if state_i and state_j:
                    entanglement_strength = self._calculate_entanglement_strength(
                        state_i.configuration, state_j.configuration
                    )
                    
                    if entanglement_strength > 0.1:
                        phase = np.random.uniform(0, 2 * np.pi)
                        matrix[i, j] = entanglement_strength * complex(np.cos(phase), np.sin(phase))
                        matrix[j, i] = np.conj(matrix[i, j])
        
        return matrix
    
    def _calculate_entanglement_strength(self, config1: Dict[str, Any], 
                                       config2: Dict[str, Any]) -> float:
        """Calculate entanglement strength between two configurations."""
        entanglement = 0.0
        
        for param, related_params in self.entanglement_map.items():
            if param in config1 and param in config2:
                # Check if related parameters are also similar
                param_similarity = self._parameter_similarity(config1[param], config2[param])
                
                related_similarity = 0.0
                for related_param in related_params:
                    if related_param in config1 and related_param in config2:
                        related_similarity += self._parameter_similarity(
                            config1[related_param], config2[related_param]
                        )
                
                if related_params:
                    related_similarity /= len(related_params)
                
                # High entanglement when main param is different but related params are similar
                entanglement += (1.0 - param_similarity) * related_similarity
        
        return min(entanglement / len(self.entanglement_map), 1.0)
    
    def _parameter_similarity(self, val1: Any, val2: Any) -> float:
        """Calculate similarity between two parameter values."""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Numeric parameters
            if val1 == 0 and val2 == 0:
                return 1.0
            max_val = max(abs(val1), abs(val2), 1e-8)
            return 1.0 - abs(val1 - val2) / max_val
        elif isinstance(val1, str) and isinstance(val2, str):
            # Categorical parameters
            return 1.0 if val1 == val2 else 0.0
        else:
            # Mixed or unknown types
            return 1.0 if val1 == val2 else 0.0
    
    def _quantum_optimize(self) -> List[QuantumState]:
        """Apply quantum optimization algorithm."""
        logger.debug("Applying quantum optimization")
        
        for iteration in range(self.max_iterations):
            # Apply quantum gates
            self._apply_quantum_evolution()
            
            # Apply interference to amplify good solutions
            self.quantum_circuit.apply_interference("carbon_score")
            self.quantum_circuit.apply_interference("performance_score")
            
            # Check convergence
            if self._check_convergence():
                logger.debug(f"Quantum optimization converged at iteration {iteration}")
                break
        
        return self.quantum_circuit.states
    
    def _apply_quantum_evolution(self):
        """Apply quantum evolution operator."""
        for state in self.quantum_circuit.states:
            # Rotate amplitude based on state quality
            total_score = state.total_score()
            rotation_angle = total_score * np.pi / 10  # Scale rotation
            
            # Apply rotation
            state.amplitude *= complex(np.cos(rotation_angle), np.sin(rotation_angle))
        
        # Apply entanglement operations
        self._apply_entanglement()
        
        # Normalize after evolution
        self.quantum_circuit.normalize()
    
    def _apply_entanglement(self):
        """Apply entanglement operations between quantum states."""
        states = self.quantum_circuit.states
        matrix = self.quantum_circuit.entanglement_matrix
        
        # Apply entanglement matrix transformation
        amplitudes = np.array([state.amplitude for state in states])
        new_amplitudes = matrix @ amplitudes
        
        for i, state in enumerate(states):
            state.amplitude = new_amplitudes[i]
    
    def _check_convergence(self) -> bool:
        """Check if quantum optimization has converged."""
        if len(self.quantum_circuit.measurement_history) < 2:
            return False
        
        # Check if probabilities have stabilized
        current_probs = [state.probability() for state in self.quantum_circuit.states]
        
        # Simple convergence check - could be more sophisticated
        max_prob_change = 0.0
        if hasattr(self, '_last_probabilities'):
            for curr, last in zip(current_probs, self._last_probabilities):
                max_prob_change = max(max_prob_change, abs(curr - last))
        
        self._last_probabilities = current_probs
        
        return max_prob_change < self.convergence_threshold
    
    def _generate_training_plan(self, 
                               best_state: QuantumState,
                               requirements: Dict[str, Any],
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training plan from optimal quantum state."""
        config = best_state.configuration.copy()
        
        # Calculate baseline metrics for comparison
        baseline_config = self._get_baseline_configuration(requirements)
        baseline_energy = self._calculate_energy_score(baseline_config, requirements)
        baseline_carbon = self._calculate_carbon_score(baseline_config, requirements)
        
        # Calculate savings
        energy_savings = ((baseline_energy - best_state.energy) / baseline_energy) * 100
        carbon_savings = ((baseline_carbon - best_state.carbon_score) / baseline_carbon) * 100
        
        # Generate implementation recommendations
        recommendations = self._generate_implementation_recommendations(config, requirements)
        
        # Estimate training metrics
        estimated_duration = requirements.get("estimated_duration_hours", 2.0)
        estimated_samples = requirements.get("dataset_size", 10000)
        
        if config.get("batch_size"):
            # Adjust duration based on batch size efficiency
            batch_efficiency = config["batch_size"] / 32  # Baseline batch size
            estimated_duration /= batch_efficiency ** 0.5
        
        return {
            "optimal_configuration": config,
            "baseline_configuration": baseline_config,
            "quantum_optimization_score": best_state.total_score(),
            "energy_savings_percent": max(0, energy_savings),
            "carbon_savings_percent": max(0, carbon_savings),
            "estimated_duration_hours": estimated_duration,
            "estimated_energy_kwh": estimated_duration * 0.5 / best_state.energy,  # Rough estimate
            "start_time_recommendation": self._format_start_time(config.get("start_time_hour", 12)),
            "implementation_recommendations": recommendations,
            "quantum_metrics": {
                "amplitude": abs(best_state.amplitude),
                "probability": best_state.probability(),
                "energy_level": best_state.energy,
                "carbon_score": best_state.carbon_score,
                "performance_score": best_state.performance_score
            },
            "entanglement_summary": self._summarize_entanglements(best_state),
            "confidence_level": min(best_state.probability() * 100, 95),
            "optimization_metadata": {
                "quantum_states_evaluated": len(self.quantum_circuit.states),
                "optimization_iterations": len(self.quantum_circuit.measurement_history),
                "convergence_achieved": self._check_convergence()
            }
        }
    
    def _get_baseline_configuration(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Get baseline configuration for comparison."""
        return {
            "batch_size": 32,
            "learning_rate": 5e-5,
            "precision": "fp32",
            "gradient_accumulation_steps": 1,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "start_time_hour": time.localtime().tm_hour,
            "gpu_count": 1
        }
    
    def _generate_implementation_recommendations(self, 
                                               config: Dict[str, Any],
                                               requirements: Dict[str, Any]) -> List[str]:
        """Generate specific implementation recommendations."""
        recommendations = []
        
        # Batch size recommendations
        batch_size = config.get("batch_size", 32)
        if batch_size > 64:
            recommendations.append(
                f"Use large batch size ({batch_size}) with gradient accumulation for memory efficiency"
            )
        elif batch_size < 16:
            recommendations.append(
                f"Small batch size ({batch_size}) may require learning rate adjustment"
            )
        
        # Precision recommendations
        precision = config.get("precision", "fp32")
        if precision in ["fp16", "bf16"]:
            recommendations.append(
                f"Enable {precision} mixed precision training for energy savings"
            )
            recommendations.append("Monitor for numerical instabilities with reduced precision")
        
        # Scheduling recommendations
        start_hour = config.get("start_time_hour", 12)
        current_hour = time.localtime().tm_hour
        if abs(start_hour - current_hour) > 2:
            recommendations.append(
                f"Schedule training to start at {start_hour:02d}:00 for optimal carbon efficiency"
            )
        
        # GPU scaling recommendations
        gpu_count = config.get("gpu_count", 1)
        if gpu_count > 1:
            recommendations.append(
                f"Configure distributed training across {gpu_count} GPUs"
            )
            recommendations.append("Monitor communication overhead and adjust batch size accordingly")
        
        # Learning rate recommendations
        lr = config.get("learning_rate", 5e-5)
        if lr > 1e-4:
            recommendations.append("Use learning rate scheduling to prevent overshooting")
        elif lr < 1e-5:
            recommendations.append("Consider warming up learning rate for better convergence")
        
        return recommendations
    
    def _format_start_time(self, hour: int) -> str:
        """Format start time recommendation."""
        return f"{hour:02d}:00 ({time.strftime('%A', time.struct_time((2023, 1, 1, hour, 0, 0, 0, 1, 0)))})"
    
    def _summarize_entanglements(self, state: QuantumState) -> Dict[str, List[str]]:
        """Summarize quantum entanglements in the optimal state."""
        entanglements = {}
        config = state.configuration
        
        for param, related_params in self.entanglement_map.items():
            if param in config:
                entangled_with = []
                for related in related_params:
                    if related in config:
                        entangled_with.append(f"{related}={config[related]}")
                if entangled_with:
                    entanglements[f"{param}={config[param]}"] = entangled_with
        
        return entanglements
    
    def optimize_multi_objective(self, 
                               objectives: List[str],
                               weights: List[float],
                               requirements: Dict[str, Any],
                               constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize for multiple objectives using quantum superposition.
        
        Args:
            objectives: List of objectives to optimize (e.g., ['carbon', 'performance', 'cost'])
            weights: Weight for each objective (must sum to 1.0)
            requirements: Training task requirements
            constraints: Optimization constraints
            
        Returns:
            Multi-objective optimization result
        """
        if len(objectives) != len(weights) or abs(sum(weights) - 1.0) > 0.01:
            raise ValueError("Objectives and weights must have same length and weights must sum to 1.0")
        
        logger.info(f"Starting multi-objective optimization: {objectives} with weights {weights}")
        
        # Initialize quantum states
        self._initialize_quantum_superposition(requirements, constraints or {})
        
        # Custom multi-objective evolution
        for iteration in range(self.max_iterations):
            # Evolve based on multiple objectives
            self._apply_multi_objective_evolution(objectives, weights)
            
            if self._check_convergence():
                break
        
        # Select Pareto optimal solutions
        pareto_optimal = self._find_pareto_optimal_states(objectives)
        
        # Measure best states
        final_states = self.quantum_circuit.measure(num_measurements=min(5, len(pareto_optimal)))
        
        # Generate comprehensive multi-objective report
        return self._generate_multi_objective_report(
            final_states, objectives, weights, requirements, constraints or {}
        )
    
    def _apply_multi_objective_evolution(self, objectives: List[str], weights: List[float]):
        """Apply quantum evolution for multi-objective optimization."""
        for state in self.quantum_circuit.states:
            # Calculate weighted objective score
            objective_scores = []
            
            for obj in objectives:
                if obj == "carbon":
                    objective_scores.append(state.carbon_score)
                elif obj == "performance":
                    objective_scores.append(state.performance_score)
                elif obj == "energy":
                    objective_scores.append(1.0 / max(state.energy, 0.001))
                else:
                    objective_scores.append(1.0)  # Default
            
            # Weighted combination
            combined_score = sum(score * weight for score, weight in zip(objective_scores, weights))
            
            # Apply rotation based on combined score
            rotation_angle = combined_score * np.pi / 10
            state.amplitude *= complex(np.cos(rotation_angle), np.sin(rotation_angle))
        
        self.quantum_circuit.normalize()
    
    def _find_pareto_optimal_states(self, objectives: List[str]) -> List[QuantumState]:
        """Find Pareto optimal states for multi-objective optimization."""
        pareto_optimal = []
        
        for state in self.quantum_circuit.states:
            is_dominated = False
            
            for other_state in self.quantum_circuit.states:
                if other_state == state:
                    continue
                
                # Check if other_state dominates state
                dominates = True
                for obj in objectives:
                    state_score = getattr(state, f"{obj}_score", 0)
                    other_score = getattr(other_state, f"{obj}_score", 0)
                    
                    if obj == "energy":  # Lower is better for energy
                        if state.energy >= other_state.energy:
                            dominates = False
                            break
                    else:  # Higher is better for carbon and performance
                        if state_score >= other_score:
                            dominates = False
                            break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(state)
        
        return pareto_optimal
    
    def _generate_multi_objective_report(self, 
                                       final_states: List[QuantumState],
                                       objectives: List[str],
                                       weights: List[float],
                                       requirements: Dict[str, Any],
                                       constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multi-objective optimization report."""
        best_state = max(final_states, key=lambda s: s.total_score())
        
        # Calculate objective-specific improvements
        baseline = self._get_baseline_configuration(requirements)
        baseline_scores = {
            "carbon": self._calculate_carbon_score(baseline, requirements),
            "performance": self._calculate_performance_score(baseline, requirements),
            "energy": self._calculate_energy_score(baseline, requirements)
        }
        
        improvements = {}
        for obj in objectives:
            if obj == "energy":
                improvements[obj] = ((baseline_scores[obj] - best_state.energy) / baseline_scores[obj]) * 100
            elif obj == "carbon":
                improvements[obj] = ((best_state.carbon_score - baseline_scores[obj]) / baseline_scores[obj]) * 100
            elif obj == "performance":
                improvements[obj] = ((best_state.performance_score - baseline_scores[obj]) / baseline_scores[obj]) * 100
        
        return {
            "multi_objective_results": {
                "objectives": objectives,
                "weights": weights,
                "best_configuration": best_state.configuration,
                "objective_improvements": improvements,
                "pareto_frontier_size": len(self._find_pareto_optimal_states(objectives)),
                "quantum_confidence": best_state.probability()
            },
            "alternative_solutions": [
                {
                    "configuration": state.configuration,
                    "scores": {
                        "carbon": state.carbon_score,
                        "performance": state.performance_score,
                        "energy": state.energy
                    },
                    "probability": state.probability()
                }
                for state in final_states[:3]  # Top 3 alternatives
            ],
            "implementation_plan": self._generate_training_plan(best_state, requirements, constraints)
        }
    
    def export_quantum_state(self, output_path: str):
        """Export current quantum state for analysis.
        
        Args:
            output_path: Output file path
        """
        if not self.quantum_circuit:
            logger.warning("No quantum circuit to export")
            return
        
        export_data = {
            "timestamp": time.time(),
            "num_states": len(self.quantum_circuit.states),
            "quantum_states": [
                {
                    "configuration": state.configuration,
                    "amplitude_real": state.amplitude.real,
                    "amplitude_imag": state.amplitude.imag,
                    "probability": state.probability(),
                    "energy": state.energy,
                    "carbon_score": state.carbon_score,
                    "performance_score": state.performance_score,
                    "total_score": state.total_score()
                }
                for state in self.quantum_circuit.states
            ],
            "measurement_history": self.quantum_circuit.measurement_history,
            "optimization_history": self.optimization_history,
            "entanglement_map": self.entanglement_map
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported quantum state to {output_path}")


# Global quantum task planner instance
_quantum_planner = QuantumInspiredTaskPlanner()

def get_quantum_planner() -> QuantumInspiredTaskPlanner:
    """Get global quantum-inspired task planner instance."""
    return _quantum_planner