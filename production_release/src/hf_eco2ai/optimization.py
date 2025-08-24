"""Advanced optimization algorithms for carbon-efficient ML training."""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, Empty
import threading

from .config import CarbonConfig
from .models import CarbonMetrics, OptimizationRecommendation
from .monitoring import EnergyTracker
from .utils import get_carbon_intensity_by_time, find_optimal_training_window

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStrategy:
    """Optimization strategy configuration."""
    
    name: str
    description: str
    priority: int  # 1-10, higher is more important
    estimated_savings_percent: float  # Expected energy/carbon savings
    implementation_complexity: str  # "low", "medium", "high"
    parameters: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "estimated_savings_percent": self.estimated_savings_percent,
            "implementation_complexity": self.implementation_complexity,
            "parameters": self.parameters,
            "prerequisites": self.prerequisites
        }


@dataclass
class OptimizationResult:
    """Result of optimization analysis."""
    
    baseline_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    strategies_applied: List[OptimizationStrategy]
    total_energy_savings_percent: float
    total_co2_savings_percent: float
    performance_impact_percent: float
    recommendations: List[OptimizationRecommendation]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_metrics": self.baseline_metrics,
            "optimized_metrics": self.optimized_metrics,
            "strategies_applied": [s.to_dict() for s in self.strategies_applied],
            "total_energy_savings_percent": self.total_energy_savings_percent,
            "total_co2_savings_percent": self.total_co2_savings_percent,
            "performance_impact_percent": self.performance_impact_percent,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "timestamp": self.timestamp
        }


class QuantumInspiredOptimizer:
    """Quantum-inspired optimization for carbon-efficient training.
    
    Uses quantum-inspired algorithms like quantum annealing concepts
    to find optimal training configurations for minimal carbon footprint.
    """
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize quantum-inspired optimizer.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        self.optimization_history: List[OptimizationResult] = []
        
        # Quantum-inspired parameters
        self.temperature = 1.0  # Annealing temperature
        self.cooling_rate = 0.95
        self.min_temperature = 0.001
        self.max_iterations = 1000
        
        logger.info("Initialized quantum-inspired optimizer")
    
    def optimize_training_schedule(self, 
                                 training_duration_hours: float,
                                 flexibility_hours: int = 24,
                                 region: str = "USA") -> Dict[str, Any]:
        """Optimize training schedule using quantum-inspired search.
        
        Args:
            training_duration_hours: Expected training duration
            flexibility_hours: Scheduling flexibility window
            region: Region for carbon intensity data
            
        Returns:
            Optimized schedule with minimal carbon footprint
        """
        logger.info(f"Optimizing training schedule for {training_duration_hours}h training")
        
        # Initialize solution space
        best_schedule = None
        best_carbon_score = float('inf')
        current_temperature = self.temperature
        
        for iteration in range(self.max_iterations):
            # Generate candidate schedule using quantum superposition concept
            candidate_schedule = self._generate_candidate_schedule(
                training_duration_hours, flexibility_hours, region
            )
            
            carbon_score = self._evaluate_schedule_carbon_impact(candidate_schedule, region)
            
            # Quantum annealing acceptance criterion
            if (carbon_score < best_carbon_score or 
                self._quantum_accept(carbon_score, best_carbon_score, current_temperature)):
                
                best_schedule = candidate_schedule
                best_carbon_score = carbon_score
            
            # Cool down temperature
            current_temperature *= self.cooling_rate
            
            if current_temperature < self.min_temperature:
                break
        
        # Calculate carbon savings
        baseline_schedule = self._generate_baseline_schedule(training_duration_hours)
        baseline_carbon = self._evaluate_schedule_carbon_impact(baseline_schedule, region)
        carbon_savings = ((baseline_carbon - best_carbon_score) / baseline_carbon) * 100
        
        return {
            "optimal_schedule": best_schedule,
            "carbon_savings_percent": carbon_savings,
            "baseline_carbon_score": baseline_carbon,
            "optimized_carbon_score": best_carbon_score,
            "iterations_used": iteration + 1,
            "recommendations": self._generate_schedule_recommendations(best_schedule)
        }
    
    def _generate_candidate_schedule(self, duration_hours: float, 
                                   flexibility_hours: int, region: str) -> Dict[str, Any]:
        """Generate candidate training schedule using quantum superposition."""
        # Quantum-inspired: consider multiple possible start times simultaneously
        possible_start_hours = np.arange(0, 24, 0.5)  # 30-minute intervals
        
        # Apply quantum interference - favor times with low carbon intensity
        carbon_weights = []
        for start_hour in possible_start_hours:
            avg_intensity = self._calculate_avg_carbon_intensity(
                start_hour, duration_hours, region
            )
            # Higher weight for lower carbon intensity (quantum tunneling to better solutions)
            weight = 1.0 / (1.0 + avg_intensity / 100)
            carbon_weights.append(weight)
        
        # Quantum sampling based on weights
        carbon_weights = np.array(carbon_weights)
        probabilities = carbon_weights / np.sum(carbon_weights)
        
        start_hour = np.random.choice(possible_start_hours, p=probabilities)
        
        # Generate dynamic configuration based on time
        config = self._generate_time_adaptive_config(start_hour, region)
        
        return {
            "start_hour": start_hour,
            "duration_hours": duration_hours,
            "dynamic_config": config,
            "region": region
        }
    
    def _quantum_accept(self, new_score: float, current_score: float, temperature: float) -> bool:
        """Quantum annealing acceptance criterion."""
        if temperature <= 0:
            return False
        
        # Quantum tunneling probability
        delta = new_score - current_score
        if delta <= 0:
            return True  # Always accept better solutions
        
        # Quantum tunneling through energy barriers
        acceptance_probability = np.exp(-delta / temperature)
        return np.random.random() < acceptance_probability
    
    def _calculate_avg_carbon_intensity(self, start_hour: float, 
                                      duration_hours: float, region: str) -> float:
        """Calculate average carbon intensity over training period."""
        total_intensity = 0
        num_samples = int(duration_hours * 2)  # 30-minute samples
        
        for i in range(num_samples):
            hour = (start_hour + i * 0.5) % 24
            intensity = get_carbon_intensity_by_time(region, int(hour))
            total_intensity += intensity
        
        return total_intensity / num_samples if num_samples > 0 else 500
    
    def _generate_time_adaptive_config(self, start_hour: float, region: str) -> Dict[str, Any]:
        """Generate time-adaptive configuration using quantum entanglement concepts."""
        carbon_intensity = get_carbon_intensity_by_time(region, int(start_hour))
        
        # Quantum entanglement: couple batch size with carbon intensity
        if carbon_intensity < 200:  # Low carbon time
            batch_size_multiplier = 1.2  # Larger batches when grid is clean
            precision = "fp16"  # Faster training when clean
        elif carbon_intensity < 400:  # Medium carbon time
            batch_size_multiplier = 1.0
            precision = "fp16"
        else:  # High carbon time
            batch_size_multiplier = 0.8  # Smaller batches to reduce power
            precision = "fp32"  # More conservative
        
        # Quantum coherence: optimize learning rate with power efficiency
        base_lr = 5e-5
        if carbon_intensity < 300:
            learning_rate_multiplier = 1.0
        else:
            learning_rate_multiplier = 0.8  # Slower but more efficient
        
        return {
            "batch_size_multiplier": batch_size_multiplier,
            "precision": precision,
            "learning_rate_multiplier": learning_rate_multiplier,
            "gradient_checkpointing": carbon_intensity > 400,
            "dynamic_batch_size": True,
            "carbon_aware_scheduling": True
        }
    
    def _evaluate_schedule_carbon_impact(self, schedule: Dict[str, Any], region: str) -> float:
        """Evaluate carbon impact of training schedule."""
        start_hour = schedule["start_hour"]
        duration_hours = schedule["duration_hours"]
        config = schedule["dynamic_config"]
        
        # Base carbon intensity over time period
        avg_carbon_intensity = self._calculate_avg_carbon_intensity(
            start_hour, duration_hours, region
        )
        
        # Apply configuration multipliers
        efficiency_multiplier = 1.0
        
        if config.get("precision") == "fp16":
            efficiency_multiplier *= 0.7  # 30% less energy
        
        if config.get("gradient_checkpointing"):
            efficiency_multiplier *= 1.1  # 10% more energy but saves memory
        
        batch_mult = config.get("batch_size_multiplier", 1.0)
        if batch_mult > 1.0:
            efficiency_multiplier *= (1.0 / batch_mult) * 0.9  # Larger batches are more efficient
        
        return avg_carbon_intensity * efficiency_multiplier
    
    def _generate_baseline_schedule(self, duration_hours: float) -> Dict[str, Any]:
        """Generate baseline schedule (immediate start, default config)."""
        return {
            "start_hour": time.localtime().tm_hour,
            "duration_hours": duration_hours,
            "dynamic_config": {
                "batch_size_multiplier": 1.0,
                "precision": "fp32",
                "learning_rate_multiplier": 1.0,
                "gradient_checkpointing": False,
                "dynamic_batch_size": False,
                "carbon_aware_scheduling": False
            },
            "region": "USA"
        }
    
    def _generate_schedule_recommendations(self, schedule: Dict[str, Any]) -> List[str]:
        """Generate recommendations for optimal schedule."""
        recommendations = []
        config = schedule["dynamic_config"]
        
        start_time = time.strftime("%H:%M", time.gmtime(schedule["start_hour"] * 3600))
        recommendations.append(f"Start training at {start_time} for optimal carbon efficiency")
        
        if config.get("precision") == "fp16":
            recommendations.append("Use FP16 precision for energy efficiency")
        
        if config.get("gradient_checkpointing"):
            recommendations.append("Enable gradient checkpointing to reduce memory usage")
        
        batch_mult = config.get("batch_size_multiplier", 1.0)
        if batch_mult != 1.0:
            change = "increase" if batch_mult > 1.0 else "decrease"
            recommendations.append(f"Consider {change} batch size by {abs(batch_mult - 1.0) * 100:.0f}%")
        
        if config.get("dynamic_batch_size"):
            recommendations.append("Implement dynamic batch sizing based on grid carbon intensity")
        
        return recommendations


class CarbonAwareScheduler:
    """Intelligent scheduler for carbon-aware ML training."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize carbon-aware scheduler.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        self.scheduling_queue = Queue()
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.scheduler_thread = None
        self.stop_scheduler = threading.Event()
        
        logger.info("Initialized carbon-aware scheduler")
    
    def submit_training_job(self, job_config: Dict[str, Any]) -> str:
        """Submit training job for carbon-aware scheduling.
        
        Args:
            job_config: Training job configuration
            
        Returns:
            Job ID for tracking
        """
        job_id = f"job_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        job_data = {
            "job_id": job_id,
            "config": job_config,
            "submission_time": time.time(),
            "status": "queued",
            "priority": job_config.get("priority", 5),
            "carbon_budget_kg": job_config.get("carbon_budget_kg"),
            "deadline": job_config.get("deadline"),
            "estimated_duration_hours": job_config.get("estimated_duration_hours", 1.0)
        }
        
        self.scheduling_queue.put(job_data)
        logger.info(f"Submitted training job {job_id} for carbon-aware scheduling")
        
        return job_id
    
    def start_scheduler(self):
        """Start the carbon-aware scheduler."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Scheduler already running")
            return
        
        self.stop_scheduler.clear()
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="CarbonAwareScheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        
        logger.info("Started carbon-aware scheduler")
    
    def stop_scheduler(self):
        """Stop the carbon-aware scheduler."""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            logger.warning("Scheduler not running")
            return
        
        self.stop_scheduler.set()
        self.scheduler_thread.join(timeout=10)
        
        logger.info("Stopped carbon-aware scheduler")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while not self.stop_scheduler.wait(60):  # Check every minute
            try:
                self._process_scheduling_queue()
                self._update_active_jobs()
                self._optimize_running_jobs()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    def _process_scheduling_queue(self):
        """Process queued jobs for optimal scheduling."""
        queued_jobs = []
        
        # Collect all queued jobs
        while True:
            try:
                job = self.scheduling_queue.get_nowait()
                queued_jobs.append(job)
            except Empty:
                break
        
        if not queued_jobs:
            return
        
        # Sort by carbon optimization potential
        queued_jobs.sort(key=lambda j: self._calculate_carbon_optimization_score(j), reverse=True)
        
        current_time = time.time()
        current_hour = time.localtime(current_time).tm_hour
        
        for job in queued_jobs:
            # Check if it's a good time to start this job
            if self._should_start_job_now(job, current_hour):
                self._start_job(job)
            else:
                # Put back in queue for later
                self.scheduling_queue.put(job)
    
    def _calculate_carbon_optimization_score(self, job: Dict[str, Any]) -> float:
        """Calculate carbon optimization score for job prioritization."""
        config = job["config"]
        duration = job["estimated_duration_hours"]
        
        # Base score from duration (longer jobs benefit more from optimization)
        score = duration * 10
        
        # Priority multiplier
        priority = job["priority"]
        score *= (priority / 5.0)  # Normalize around 5
        
        # Carbon budget constraint (urgent jobs get higher score)
        if job.get("carbon_budget_kg"):
            # Tighter budgets need better scheduling
            budget_factor = 10.0 / max(job["carbon_budget_kg"], 0.1)
            score *= budget_factor
        
        # Deadline urgency
        if job.get("deadline"):
            deadline = job["deadline"]
            time_remaining = deadline - time.time()
            if time_remaining > 0:
                urgency = max(1.0, duration * 3600 / time_remaining)
                score *= urgency
        
        return score
    
    def _should_start_job_now(self, job: Dict[str, Any], current_hour: int) -> bool:
        """Determine if job should start now based on carbon optimization."""
        region = job["config"].get("region", "USA")
        duration_hours = job["estimated_duration_hours"]
        
        # Get current carbon intensity
        current_intensity = get_carbon_intensity_by_time(region, current_hour)
        
        # Find optimal time in next 24 hours
        optimal_hour, savings_percent = find_optimal_training_window(duration_hours, region)
        
        # Start now if:
        # 1. We're at or near optimal time
        # 2. Deadline pressure
        # 3. Very low carbon intensity
        
        time_diff = abs(current_hour - optimal_hour)
        if time_diff > 12:  # Handle wrap-around
            time_diff = 24 - time_diff
        
        if time_diff <= 1:  # Within 1 hour of optimal
            return True
        
        if current_intensity < 200:  # Very clean grid
            return True
        
        # Check deadline pressure
        if job.get("deadline"):
            time_remaining = job["deadline"] - time.time()
            min_time_needed = duration_hours * 3600 * 1.2  # 20% buffer
            if time_remaining <= min_time_needed:
                return True
        
        return False
    
    def _start_job(self, job: Dict[str, Any]):
        """Start a training job."""
        job_id = job["job_id"]
        job["status"] = "running"
        job["start_time"] = time.time()
        
        self.active_jobs[job_id] = job
        
        logger.info(f"Started carbon-optimized training job {job_id}")
        
        # Here you would integrate with your actual training system
        # For now, we simulate job completion
        threading.Timer(
            job["estimated_duration_hours"] * 60,  # Convert to minutes for simulation
            self._complete_job,
            args=[job_id]
        ).start()
    
    def _complete_job(self, job_id: str):
        """Mark job as completed."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job["status"] = "completed"
            job["end_time"] = time.time()
            
            # Calculate actual carbon savings
            duration = job["end_time"] - job["start_time"]
            
            logger.info(f"Completed training job {job_id} in {duration/3600:.1f} hours")
            
            # Move to completed jobs (you might want to store this differently)
            del self.active_jobs[job_id]
    
    def _update_active_jobs(self):
        """Update status of active jobs."""
        current_time = time.time()
        
        for job_id, job in list(self.active_jobs.items()):
            if job["status"] == "running":
                # Check if job should be paused due to high carbon intensity
                if self._should_pause_job(job):
                    self._pause_job(job_id)
                elif job["status"] == "paused" and self._should_resume_job(job):
                    self._resume_job(job_id)
    
    def _should_pause_job(self, job: Dict[str, Any]) -> bool:
        """Check if job should be paused due to carbon intensity."""
        config = job["config"]
        
        # Only pause if job has flexible carbon settings
        if not config.get("carbon_aware_pausing", False):
            return False
        
        region = config.get("region", "USA")
        current_hour = time.localtime().tm_hour
        current_intensity = get_carbon_intensity_by_time(region, current_hour)
        
        # Pause if carbon intensity is very high
        return current_intensity > 600
    
    def _should_resume_job(self, job: Dict[str, Any]) -> bool:
        """Check if paused job should be resumed."""
        region = job["config"].get("region", "USA")
        current_hour = time.localtime().tm_hour
        current_intensity = get_carbon_intensity_by_time(region, current_hour)
        
        # Resume if carbon intensity has decreased
        return current_intensity < 450
    
    def _pause_job(self, job_id: str):
        """Pause a running job."""
        if job_id in self.active_jobs:
            self.active_jobs[job_id]["status"] = "paused"
            self.active_jobs[job_id]["pause_time"] = time.time()
            logger.info(f"Paused job {job_id} due to high carbon intensity")
    
    def _resume_job(self, job_id: str):
        """Resume a paused job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job["status"] = "running"
            
            # Add pause time to estimated completion
            if "pause_time" in job:
                pause_duration = time.time() - job["pause_time"]
                job["estimated_duration_hours"] += pause_duration / 3600
                del job["pause_time"]
            
            logger.info(f"Resumed job {job_id} - carbon intensity improved")
    
    def _optimize_running_jobs(self):
        """Dynamically optimize running jobs."""
        for job_id, job in self.active_jobs.items():
            if job["status"] == "running":
                self._apply_dynamic_optimizations(job)
    
    def _apply_dynamic_optimizations(self, job: Dict[str, Any]):
        """Apply dynamic optimizations to running job."""
        config = job["config"]
        
        # Dynamic batch size adjustment based on current carbon intensity
        if config.get("dynamic_batch_size", False):
            region = config.get("region", "USA")
            current_hour = time.localtime().tm_hour
            current_intensity = get_carbon_intensity_by_time(region, current_hour)
            
            # Adjust batch size inversely to carbon intensity
            if current_intensity < 300:
                suggested_batch_multiplier = 1.2
            elif current_intensity > 500:
                suggested_batch_multiplier = 0.8
            else:
                suggested_batch_multiplier = 1.0
            
            # Here you would communicate this to the training process
            logger.debug(f"Job {job['job_id']}: suggested batch multiplier {suggested_batch_multiplier:.2f}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a submitted job.
        
        Args:
            job_id: Job ID to query
            
        Returns:
            Job status dictionary or None if not found
        """
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].copy()
        return None
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics.
        
        Returns:
            Scheduler statistics
        """
        return {
            "active_jobs": len(self.active_jobs),
            "queued_jobs": self.scheduling_queue.qsize(),
            "running_jobs": len([j for j in self.active_jobs.values() if j["status"] == "running"]),
            "paused_jobs": len([j for j in self.active_jobs.values() if j["status"] == "paused"]),
            "scheduler_uptime": time.time() - (self.scheduler_thread.ident if self.scheduler_thread else time.time())
        }


class AdaptiveOptimizer:
    """Adaptive optimization that learns from training patterns."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize adaptive optimizer.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        self.optimization_history: List[Dict[str, Any]] = []
        self.learned_patterns: Dict[str, Any] = {}
        self.performance_model = None
        
        logger.info("Initialized adaptive optimizer")
    
    def learn_from_training_run(self, training_metrics: Dict[str, Any]):
        """Learn optimization patterns from completed training run.
        
        Args:
            training_metrics: Metrics from completed training
        """
        # Extract key patterns
        pattern = {
            "timestamp": time.time(),
            "model_size": training_metrics.get("model_parameters", 0),
            "batch_size": training_metrics.get("batch_size", 32),
            "sequence_length": training_metrics.get("sequence_length", 512),
            "precision": training_metrics.get("precision", "fp32"),
            "energy_kwh": training_metrics.get("energy_kwh", 0),
            "duration_hours": training_metrics.get("duration_hours", 1),
            "final_loss": training_metrics.get("final_loss", 1.0),
            "carbon_intensity": training_metrics.get("avg_carbon_intensity", 400),
            "gpu_utilization": training_metrics.get("avg_gpu_utilization", 80)
        }
        
        self.optimization_history.append(pattern)
        
        # Update learned patterns
        self._update_learned_patterns(pattern)
        
        logger.info(f"Learned from training run: {pattern['energy_kwh']:.3f} kWh, {pattern['final_loss']:.3f} loss")
    
    def _update_learned_patterns(self, pattern: Dict[str, Any]):
        """Update learned optimization patterns."""
        # Group by model size ranges
        model_size = pattern["model_size"]
        size_category = self._get_model_size_category(model_size)
        
        if size_category not in self.learned_patterns:
            self.learned_patterns[size_category] = {
                "optimal_batch_sizes": [],
                "energy_efficiency_scores": [],
                "convergence_patterns": [],
                "carbon_efficiency_scores": []
            }
        
        category_data = self.learned_patterns[size_category]
        
        # Calculate efficiency scores
        samples_processed = pattern["batch_size"] * 1000  # Estimate
        energy_efficiency = samples_processed / max(pattern["energy_kwh"], 0.001)
        carbon_efficiency = samples_processed / (pattern["energy_kwh"] * pattern["carbon_intensity"])
        
        # Store learned data
        category_data["optimal_batch_sizes"].append({
            "batch_size": pattern["batch_size"],
            "efficiency": energy_efficiency,
            "loss": pattern["final_loss"]
        })
        
        category_data["energy_efficiency_scores"].append(energy_efficiency)
        category_data["carbon_efficiency_scores"].append(carbon_efficiency)
        
        # Keep only recent data (last 100 runs per category)
        for key in category_data:
            if isinstance(category_data[key], list) and len(category_data[key]) > 100:
                category_data[key] = category_data[key][-100:]
    
    def _get_model_size_category(self, model_size: int) -> str:
        """Categorize model by size."""
        if model_size < 1e6:  # < 1M parameters
            return "small"
        elif model_size < 1e8:  # < 100M parameters
            return "medium"
        elif model_size < 1e9:  # < 1B parameters
            return "large"
        else:
            return "extra_large"
    
    def suggest_optimizations(self, training_config: Dict[str, Any]) -> List[OptimizationStrategy]:
        """Suggest optimizations based on learned patterns.
        
        Args:
            training_config: Proposed training configuration
            
        Returns:
            List of optimization strategies
        """
        suggestions = []
        
        model_size = training_config.get("model_parameters", 0)
        size_category = self._get_model_size_category(model_size)
        
        if size_category in self.learned_patterns:
            patterns = self.learned_patterns[size_category]
            
            # Suggest optimal batch size
            batch_suggestion = self._suggest_optimal_batch_size(patterns, training_config)
            if batch_suggestion:
                suggestions.append(batch_suggestion)
            
            # Suggest precision optimization
            precision_suggestion = self._suggest_precision_optimization(patterns, training_config)
            if precision_suggestion:
                suggestions.append(precision_suggestion)
            
            # Suggest scheduling optimization
            schedule_suggestion = self._suggest_schedule_optimization(training_config)
            if schedule_suggestion:
                suggestions.append(schedule_suggestion)
        
        return suggestions
    
    def _suggest_optimal_batch_size(self, patterns: Dict[str, Any], 
                                  config: Dict[str, Any]) -> Optional[OptimizationStrategy]:
        """Suggest optimal batch size based on learned patterns."""
        if not patterns["optimal_batch_sizes"]:
            return None
        
        # Find batch sizes with best efficiency/loss trade-off
        batch_data = patterns["optimal_batch_sizes"]
        
        # Score each batch size by efficiency and loss
        scored_batches = []
        for data in batch_data:
            score = data["efficiency"] / max(data["loss"], 0.1)  # Higher efficiency, lower loss is better
            scored_batches.append((data["batch_size"], score))
        
        # Get top 3 batch sizes
        scored_batches.sort(key=lambda x: x[1], reverse=True)
        top_batches = scored_batches[:3]
        
        if top_batches:
            recommended_batch = int(np.mean([b[0] for b in top_batches]))
            current_batch = config.get("batch_size", 32)
            
            if abs(recommended_batch - current_batch) / current_batch > 0.1:  # >10% difference
                return OptimizationStrategy(
                    name="adaptive_batch_size",
                    description=f"Use batch size {recommended_batch} (learned optimal for this model size)",
                    priority=7,
                    estimated_savings_percent=15,
                    implementation_complexity="low",
                    parameters={"recommended_batch_size": recommended_batch}
                )
        
        return None
    
    def _suggest_precision_optimization(self, patterns: Dict[str, Any], 
                                      config: Dict[str, Any]) -> Optional[OptimizationStrategy]:
        """Suggest precision optimization."""
        current_precision = config.get("precision", "fp32")
        
        if current_precision == "fp32":
            # Suggest FP16 for energy savings
            return OptimizationStrategy(
                name="fp16_precision",
                description="Use FP16 precision for significant energy savings",
                priority=8,
                estimated_savings_percent=30,
                implementation_complexity="low",
                parameters={"precision": "fp16"},
                prerequisites=["mixed_precision_support"]
            )
        
        return None
    
    def _suggest_schedule_optimization(self, config: Dict[str, Any]) -> Optional[OptimizationStrategy]:
        """Suggest training schedule optimization."""
        duration_hours = config.get("estimated_duration_hours", 1.0)
        region = config.get("region", "USA")
        
        if duration_hours > 2:  # Only suggest for longer training runs
            optimal_hour, savings_percent = find_optimal_training_window(duration_hours, region)
            current_hour = time.localtime().tm_hour
            
            if abs(optimal_hour - current_hour) > 2 and savings_percent > 10:
                return OptimizationStrategy(
                    name="carbon_aware_scheduling",
                    description=f"Delay training to start at {optimal_hour:02d}:00 for {savings_percent:.1f}% carbon savings",
                    priority=6,
                    estimated_savings_percent=savings_percent,
                    implementation_complexity="medium",
                    parameters={
                        "optimal_start_hour": optimal_hour,
                        "estimated_savings": savings_percent
                    }
                )
        
        return None
    
    def export_learned_patterns(self, output_path: str):
        """Export learned patterns to file.
        
        Args:
            output_path: Output file path
        """
        export_data = {
            "learned_patterns": self.learned_patterns,
            "optimization_history_count": len(self.optimization_history),
            "export_timestamp": time.time(),
            "categories": list(self.learned_patterns.keys())
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported learned patterns to {output_path}")


# Global optimizer instances
_quantum_optimizer = QuantumInspiredOptimizer()
_carbon_scheduler = CarbonAwareScheduler()
_adaptive_optimizer = AdaptiveOptimizer()

def get_quantum_optimizer() -> QuantumInspiredOptimizer:
    """Get global quantum-inspired optimizer instance."""
    return _quantum_optimizer

def get_carbon_scheduler() -> CarbonAwareScheduler:
    """Get global carbon-aware scheduler instance."""
    return _carbon_scheduler

def get_adaptive_optimizer() -> AdaptiveOptimizer:
    """Get global adaptive optimizer instance."""
    return _adaptive_optimizer