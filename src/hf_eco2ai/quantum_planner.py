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
        
        # Weighted random sampling based on quantum probabilities
        measured_states = []
        for _ in range(num_measurements):
            cumulative_prob = 0
            random_value = np.random.random()
            
            for state in self.states:
                cumulative_prob += state.probability()
                if random_value <= cumulative_prob:
                    measured_states.append(state)
                    break
        
        # Store measurement in history
        self.measurement_history.append({
            "timestamp": time.time(),
            "measured_configs": [s.configuration for s in measured_states],
            "total_probability": sum(probabilities)
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
        
        # Generate quantum states using advanced superposition sampling
        num_states = min(128, np.prod([len(space) for space in parameter_spaces.values()]))
        
        # Use quantum-inspired sampling with controlled exploration
        states_configs = self._quantum_sampling(parameter_spaces, num_states, requirements)
        
        for config in states_configs:
            
            # Calculate state properties with advanced scoring
            energy = self._calculate_energy_score(config, requirements)
            carbon_score = self._calculate_carbon_score(config, requirements)
            performance_score = self._calculate_performance_score(config, requirements)
            
            # Quantum amplitude based on initial assessment (non-uniform)
            initial_score = (carbon_score * 0.4 + performance_score * 0.3 + (1.0/max(energy, 0.001)) * 0.3)
            amplitude_magnitude = math.sqrt(initial_score / num_states) if initial_score > 0 else 1.0 / math.sqrt(num_states)
            quantum_phase = 2 * math.pi * np.random.random()  # Random phase
            amplitude = complex(amplitude_magnitude * math.cos(quantum_phase), 
                              amplitude_magnitude * math.sin(quantum_phase))
            
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
        
        logger.debug(f"Initialized {len(states)} quantum states with advanced superposition")
    
    def _quantum_sampling(self, parameter_spaces: Dict[str, List], 
                         num_states: int, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Advanced quantum-inspired sampling of configuration space."""
        configs = []
        
        # Phase 1: Pure random sampling (exploration)
        exploration_states = num_states // 3
        for _ in range(exploration_states):
            config = {}
            for param, space in parameter_spaces.items():
                config[param] = np.random.choice(space)
            configs.append(config)
        
        # Phase 2: Targeted sampling based on known good patterns (exploitation)
        exploitation_states = num_states // 3
        good_patterns = self._get_known_good_patterns(requirements)
        
        for pattern in good_patterns[:exploitation_states]:
            config = {}
            for param, space in parameter_spaces.items():
                if param in pattern:
                    # Use pattern value with small perturbation
                    target_val = pattern[param]
                    if target_val in space:
                        config[param] = target_val
                    else:
                        # Find closest value in space
                        config[param] = min(space, key=lambda x: abs(x - target_val) if isinstance(x, (int, float)) else float('inf'))
                else:
                    config[param] = np.random.choice(space)
            configs.append(config)
        
        # Phase 3: Quantum tunneling - explore unlikely but potentially optimal regions
        tunneling_states = num_states - exploration_states - exploitation_states
        for _ in range(tunneling_states):
            config = {}
            for param, space in parameter_spaces.items():
                # Bias toward extreme values (quantum tunneling effect)
                if np.random.random() < 0.3:  # 30% chance of extreme value
                    config[param] = np.random.choice([space[0], space[-1]])
                else:
                    config[param] = np.random.choice(space)
            configs.append(config)
        
        return configs
    
    def _get_known_good_patterns(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get known good configuration patterns based on model type and requirements."""
        patterns = []
        
        model_type = requirements.get("model_type", "transformer")
        model_size = requirements.get("model_parameters", 100_000_000)
        
        if model_type == "transformer":
            if model_size < 1_000_000:  # Small model
                patterns.extend([
                    {"batch_size": 64, "learning_rate": 5e-4, "precision": "fp16"},
                    {"batch_size": 32, "learning_rate": 1e-4, "precision": "bf16"},
                ])
            elif model_size < 100_000_000:  # Medium model
                patterns.extend([
                    {"batch_size": 32, "learning_rate": 2e-4, "precision": "fp16"},
                    {"batch_size": 16, "learning_rate": 1e-4, "precision": "bf16"},
                ])
            else:  # Large model
                patterns.extend([
                    {"batch_size": 16, "learning_rate": 1e-4, "precision": "fp16"},
                    {"batch_size": 8, "learning_rate": 5e-5, "precision": "bf16"},
                ])
        
        # Add carbon-optimal patterns
        patterns.extend([
            {"start_time_hour": 2, "precision": "fp16"},  # Low carbon hour + efficiency
            {"start_time_hour": 14, "precision": "bf16"},  # Solar peak + modern precision
        ])
        
        return patterns
    
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
        """Apply advanced quantum annealing optimization algorithm."""
        logger.debug("Applying quantum annealing optimization")
        
        # Initialize annealing parameters
        initial_temperature = 10.0
        final_temperature = 0.01
        cooling_rate = 0.95
        current_temperature = initial_temperature
        
        best_energy = float('inf')
        best_states = []
        stagnation_count = 0
        max_stagnation = 20
        
        for iteration in range(self.max_iterations):
            # Apply quantum annealing step
            self._apply_quantum_annealing_step(current_temperature)
            
            # Apply multi-objective interference
            self._apply_multi_objective_interference()
            
            # Track best solutions (Pareto optimal)
            current_best_energy = min(state.energy for state in self.quantum_circuit.states)
            if current_best_energy < best_energy:
                best_energy = current_best_energy
                best_states = self._get_pareto_optimal_states()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Adaptive cooling with restart mechanism
            if stagnation_count > max_stagnation:
                # Restart with higher temperature
                current_temperature = initial_temperature * 0.5
                stagnation_count = 0
                logger.debug(f"Quantum annealing restart at iteration {iteration}")
            else:
                current_temperature *= cooling_rate
            
            # Check convergence
            if self._check_advanced_convergence(iteration):
                logger.debug(f"Quantum optimization converged at iteration {iteration}")
                break
            
            # Stop if temperature too low
            if current_temperature < final_temperature:
                logger.debug(f"Quantum annealing completed at iteration {iteration}")
                break
        
        return self.quantum_circuit.states
    
    def _apply_quantum_annealing_step(self, temperature: float):
        """Apply quantum annealing evolution step with temperature control."""
        for state in self.quantum_circuit.states:
            # Calculate energy-based rotation with thermal fluctuation
            energy_factor = math.exp(-state.energy / max(temperature, 0.001))
            total_score = state.total_score()
            
            # Quantum rotation with thermal noise
            base_rotation = total_score * np.pi / 8
            thermal_noise = np.random.normal(0, temperature / 10)
            rotation_angle = base_rotation + thermal_noise
            
            # Apply rotation with temperature-dependent amplitude
            rotation_amplitude = energy_factor * 0.8 + 0.2  # Ensure some minimum rotation
            state.amplitude *= complex(
                rotation_amplitude * np.cos(rotation_angle), 
                rotation_amplitude * np.sin(rotation_angle)
            )
        
        # Apply temperature-dependent entanglement
        self._apply_thermal_entanglement(temperature)
        
        # Normalize after evolution
        self.quantum_circuit.normalize()
    
    def _apply_thermal_entanglement(self, temperature: float):
        """Apply temperature-dependent entanglement operations."""
        states = self.quantum_circuit.states
        matrix = self.quantum_circuit.entanglement_matrix
        
        # Scale entanglement strength by temperature
        entanglement_strength = min(1.0, temperature / 5.0)
        
        # Apply entanglement matrix transformation with thermal scaling
        amplitudes = np.array([state.amplitude for state in states])
        identity = np.eye(len(states))
        thermal_matrix = (1 - entanglement_strength) * identity + entanglement_strength * matrix
        
        new_amplitudes = thermal_matrix @ amplitudes
        
        for i, state in enumerate(states):
            state.amplitude = new_amplitudes[i]
    
    def _apply_multi_objective_interference(self):
        """Apply interference targeting multiple objectives simultaneously."""
        # Apply interference for each objective with different weights
        objectives = [
            ("carbon_score", 0.4),
            ("performance_score", 0.3), 
            ("energy", 0.3)  # Lower energy is better
        ]
        
        for objective, weight in objectives:
            if objective == "energy":
                # For energy, lower is better, so reverse sort
                sorted_states = sorted(self.quantum_circuit.states, 
                                     key=lambda s: s.energy, reverse=False)
            else:
                sorted_states = sorted(self.quantum_circuit.states, 
                                     key=lambda s: getattr(s, objective), reverse=True)
            
            # Apply weighted interference
            top_third = len(sorted_states) // 3
            
            for i, state in enumerate(sorted_states):
                if i < top_third:
                    # Constructive interference - weighted by objective importance
                    phase_boost = weight * math.pi / 3
                    state.amplitude *= complex(math.cos(phase_boost), math.sin(phase_boost))
                elif i > 2 * top_third:
                    # Destructive interference
                    phase_reduction = -weight * math.pi / 4
                    state.amplitude *= complex(math.cos(phase_reduction), math.sin(phase_reduction))
        
        self.quantum_circuit.normalize()
    
    def _get_pareto_optimal_states(self) -> List[QuantumState]:
        """Find Pareto optimal solutions in multi-objective space."""
        pareto_states = []
        
        for state in self.quantum_circuit.states:
            is_dominated = False
            
            for other_state in self.quantum_circuit.states:
                if state == other_state:
                    continue
                
                # Check if other_state dominates state
                # (better or equal in all objectives, strictly better in at least one)
                carbon_better = other_state.carbon_score >= state.carbon_score
                performance_better = other_state.performance_score >= state.performance_score
                energy_better = other_state.energy <= state.energy  # Lower energy is better
                
                # Check strict dominance
                carbon_strictly_better = other_state.carbon_score > state.carbon_score
                performance_strictly_better = other_state.performance_score > state.performance_score
                energy_strictly_better = other_state.energy < state.energy
                
                if (carbon_better and performance_better and energy_better and 
                    (carbon_strictly_better or performance_strictly_better or energy_strictly_better)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_states.append(state)
        
        return pareto_states
    
    def _check_advanced_convergence(self, iteration: int) -> bool:
        """Advanced convergence checking with multiple criteria."""
        if iteration < 10:  # Need minimum iterations
            return False
        
        # Check probability stabilization
        current_probs = [state.probability() for state in self.quantum_circuit.states]
        
        prob_convergence = False
        if hasattr(self, '_last_probabilities'):
            max_prob_change = max(abs(curr - last) 
                                for curr, last in zip(current_probs, self._last_probabilities))
            prob_convergence = max_prob_change < self.convergence_threshold
        
        # Check energy convergence
        current_energies = [state.energy for state in self.quantum_circuit.states]
        energy_convergence = False
        if hasattr(self, '_last_energies'):
            max_energy_change = max(abs(curr - last) 
                                  for curr, last in zip(current_energies, self._last_energies))
            energy_convergence = max_energy_change < (self.convergence_threshold * 10)
        
        # Check amplitude convergence
        current_amplitudes = [abs(state.amplitude) for state in self.quantum_circuit.states]
        amplitude_convergence = False
        if hasattr(self, '_last_amplitudes'):
            max_amp_change = max(abs(curr - last) 
                               for curr, last in zip(current_amplitudes, self._last_amplitudes))
            amplitude_convergence = max_amp_change < self.convergence_threshold
        
        # Store for next iteration
        self._last_probabilities = current_probs
        self._last_energies = current_energies
        self._last_amplitudes = current_amplitudes
        
        # Require convergence in at least 2 out of 3 criteria
        convergence_count = sum([prob_convergence, energy_convergence, amplitude_convergence])
        return convergence_count >= 2
    
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
        
        # Calculate quantum coherence metrics
        coherence_score = self._calculate_quantum_coherence()
        entanglement_score = self._calculate_entanglement_measure()
        
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
                "performance_score": best_state.performance_score,
                "coherence_score": coherence_score,
                "entanglement_score": entanglement_score
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
    
    def _calculate_quantum_coherence(self) -> float:
        """Calculate quantum coherence measure for the current state ensemble."""
        if not self.quantum_circuit or not self.quantum_circuit.states:
            return 0.0
        
        # Calculate von Neumann entropy as coherence measure
        amplitudes = np.array([state.amplitude for state in self.quantum_circuit.states])
        probabilities = np.abs(amplitudes) ** 2
        
        # Normalize probabilities
        total_prob = np.sum(probabilities)
        if total_prob > 0:
            probabilities = probabilities / total_prob
        
        # Calculate von Neumann entropy
        entropy = 0.0
        for p in probabilities:
            if p > 1e-10:  # Avoid log(0)
                entropy -= p * np.log2(p)
        
        # Normalize to [0, 1] scale
        max_entropy = np.log2(len(probabilities))
        coherence = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(coherence)
    
    def _calculate_entanglement_measure(self) -> float:
        """Calculate quantum entanglement measure between states."""
        if not self.quantum_circuit or len(self.quantum_circuit.states) < 2:
            return 0.0
        
        # Calculate pairwise entanglement using quantum mutual information
        states = self.quantum_circuit.states
        n_states = len(states)
        
        total_entanglement = 0.0
        pairs_counted = 0
        
        for i in range(n_states):
            for j in range(i + 1, n_states):
                state_i = states[i]
                state_j = states[j]
                
                # Calculate entanglement between parameter configurations
                entanglement = self._calculate_configuration_entanglement(
                    state_i.configuration, state_j.configuration
                )
                
                # Weight by quantum amplitudes
                amplitude_weight = abs(state_i.amplitude * np.conj(state_j.amplitude))
                weighted_entanglement = entanglement * amplitude_weight
                
                total_entanglement += weighted_entanglement
                pairs_counted += 1
        
        # Normalize by number of pairs
        avg_entanglement = total_entanglement / pairs_counted if pairs_counted > 0 else 0.0
        
        return float(avg_entanglement)
    
    def _calculate_configuration_entanglement(self, config1: Dict[str, Any], 
                                            config2: Dict[str, Any]) -> float:
        """Calculate entanglement between two parameter configurations."""
        entanglement_score = 0.0
        total_comparisons = 0
        
        for param, related_params in self.entanglement_map.items():
            if param in config1 and param in config2:
                # Check how many related parameters are also entangled
                related_entangled = 0
                for related in related_params:
                    if related in config1 and related in config2:
                        # Calculate correlation between parameters
                        val1 = config1[related]
                        val2 = config2[related]
                        
                        # Normalize values for comparison
                        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                            correlation = 1.0 - abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-6)
                        else:
                            correlation = 1.0 if val1 == val2 else 0.0
                        
                        related_entangled += correlation
                        total_comparisons += 1
                
                # Add to entanglement score
                if len(related_params) > 0:
                    param_entanglement = related_entangled / len(related_params)
                    entanglement_score += param_entanglement
        
        # Normalize by total possible entanglements
        if total_comparisons > 0:
            return entanglement_score / len(self.entanglement_map)
        else:
            return 0.0
    
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