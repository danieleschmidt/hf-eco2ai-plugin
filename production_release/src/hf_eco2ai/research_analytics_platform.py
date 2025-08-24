"""Advanced Research Analytics Platform for Carbon Intelligence.

This module provides comprehensive research analytics capabilities including:
- Comparative carbon analysis across model architectures
- Carbon-aware hyperparameter optimization
- Carbon efficiency leaderboards
- Research publication metrics generation
- Advanced statistical analysis and visualization
- Meta-analysis of carbon optimization techniques
"""

import time
import logging
import json
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import statistics
from collections import defaultdict, Counter
from enum import Enum
import warnings

# Scientific computing and visualization
try:
    import scipy.stats as stats
    from scipy.optimize import minimize, differential_evolution
    from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import existing components
try:
    from .autonomous_research_engine import ResearchHypothesis, ExperimentResult, ResearchPublication
    from .performance_optimizer import optimized, get_performance_optimizer
    from .error_handling import handle_gracefully, ErrorSeverity, resilient_operation
    from .models import CarbonReport, CarbonMetrics
except ImportError:
    # Fallback classes and decorators
    @dataclass
    class ResearchHypothesis:
        hypothesis_id: str
        title: str
        description: str
    
    @dataclass 
    class ExperimentResult:
        result_id: str
        hypothesis_id: str
        conclusion: str
    
    def optimized(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def handle_gracefully(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class ErrorSeverity:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ArchitectureFamily(Enum):
    """Model architecture families for comparative analysis."""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    HYBRID = "hybrid"
    GRAPH_NN = "graph_nn"
    DIFFUSION = "diffusion"
    GAN = "gan"
    AUTOENCODER = "autoencoder"
    ENSEMBLE = "ensemble"


class OptimizationObjective(Enum):
    """Optimization objectives for carbon-aware training."""
    MINIMIZE_CARBON = "minimize_carbon"
    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCE_ACCURACY_CARBON = "balance_accuracy_carbon"
    MINIMIZE_TIME_TO_TARGET = "minimize_time_to_target"


@dataclass
class ModelArchitectureProfile:
    """Profile of a model architecture for carbon analysis."""
    architecture_id: str
    family: ArchitectureFamily
    name: str
    parameters: int
    layers: int
    computational_complexity: str  # O(n²), O(n log n), etc.
    memory_complexity: str
    typical_batch_size: int
    hardware_requirements: List[str]
    carbon_efficiency_score: float = 0.0
    typical_training_time_hours: float = 0.0
    energy_per_parameter: float = 0.0  # kWh per parameter
    co2_per_sample: float = 0.0  # kg CO₂ per training sample


@dataclass
class CarbonBenchmarkResult:
    """Results from carbon benchmarking experiments."""
    benchmark_id: str
    architecture_profile: ModelArchitectureProfile
    dataset_name: str
    dataset_size: int
    hyperparameters: Dict[str, Any]
    hardware_config: Dict[str, Any]
    training_metrics: Dict[str, float]
    carbon_metrics: Dict[str, float]
    accuracy_metrics: Dict[str, float]
    efficiency_score: float
    carbon_accuracy_pareto_position: Tuple[float, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HyperparameterOptimizationResult:
    """Results from carbon-aware hyperparameter optimization."""
    optimization_id: str
    objective: OptimizationObjective
    search_space: Dict[str, Any]
    best_params: Dict[str, Any]
    best_score: float
    carbon_cost: float
    optimization_history: List[Dict[str, Any]]
    convergence_metrics: Dict[str, float]
    recommendations: List[str]
    statistical_significance: Dict[str, float]


@dataclass
class ResearchPublication:
    """Enhanced research publication with metrics."""
    publication_id: str
    title: str
    abstract: str
    carbon_impact_factor: float = 0.0
    replication_studies: int = 0
    citation_count: int = 0
    real_world_implementations: int = 0
    co2_reduction_potential_kg: float = 0.0
    confidence_score: float = 0.0


class ModelArchitectureAnalyzer:
    """Analyze and compare carbon efficiency across model architectures."""
    
    def __init__(self):
        self.architecture_profiles: Dict[str, ModelArchitectureProfile] = {}
        self.benchmark_results: List[CarbonBenchmarkResult] = []
        self.comparative_studies: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with known architecture profiles
        self._initialize_architecture_database()
    
    def _initialize_architecture_database(self):
        """Initialize database with known model architectures."""
        architectures = [
            {
                "architecture_id": "bert_base",
                "family": ArchitectureFamily.TRANSFORMER,
                "name": "BERT Base",
                "parameters": 110_000_000,
                "layers": 12,
                "computational_complexity": "O(n²)",
                "memory_complexity": "O(n²)",
                "typical_batch_size": 32,
                "hardware_requirements": ["16GB GPU", "mixed_precision"],
                "typical_training_time_hours": 24.0
            },
            {
                "architecture_id": "gpt2_medium",
                "family": ArchitectureFamily.TRANSFORMER,
                "name": "GPT-2 Medium",
                "parameters": 345_000_000,
                "layers": 24,
                "computational_complexity": "O(n²)",
                "memory_complexity": "O(n²)",
                "typical_batch_size": 16,
                "hardware_requirements": ["24GB GPU", "gradient_checkpointing"],
                "typical_training_time_hours": 72.0
            },
            {
                "architecture_id": "resnet50",
                "family": ArchitectureFamily.CNN,
                "name": "ResNet-50",
                "parameters": 25_600_000,
                "layers": 50,
                "computational_complexity": "O(n)",
                "memory_complexity": "O(1)",
                "typical_batch_size": 64,
                "hardware_requirements": ["8GB GPU"],
                "typical_training_time_hours": 12.0
            },
            {
                "architecture_id": "efficientnet_b3",
                "family": ArchitectureFamily.CNN,
                "name": "EfficientNet-B3",
                "parameters": 12_000_000,
                "layers": 30,
                "computational_complexity": "O(n)",
                "memory_complexity": "O(1)",
                "typical_batch_size": 128,
                "hardware_requirements": ["6GB GPU"],
                "typical_training_time_hours": 8.0
            }
        ]
        
        for arch_data in architectures:
            profile = ModelArchitectureProfile(**arch_data)
            self.architecture_profiles[profile.architecture_id] = profile
            
        logger.info(f"Initialized {len(architectures)} architecture profiles")
    
    @optimized(cache_ttl=300.0)
    def analyze_architecture_carbon_efficiency(self, architecture_id: str, 
                                             training_data: List[CarbonMetrics]) -> Dict[str, float]:
        """Analyze carbon efficiency of a specific architecture."""
        if not training_data:
            return {}
        
        # Calculate efficiency metrics
        total_energy = sum(m.cumulative_energy_kwh for m in training_data)
        total_co2 = sum(m.cumulative_co2_kg for m in training_data)
        total_samples = sum(m.samples_processed for m in training_data)
        training_time = training_data[-1].timestamp - training_data[0].timestamp
        
        profile = self.architecture_profiles.get(architecture_id)
        if not profile:
            return {}
        
        # Calculate comprehensive efficiency metrics
        efficiency_metrics = {
            "samples_per_kwh": total_samples / total_energy if total_energy > 0 else 0,
            "co2_per_sample": total_co2 / total_samples if total_samples > 0 else 0,
            "energy_per_parameter": total_energy / profile.parameters if profile.parameters > 0 else 0,
            "carbon_intensity": total_co2 / total_energy if total_energy > 0 else 0,
            "training_efficiency": total_samples / training_time if training_time > 0 else 0,
            "parameter_efficiency": total_samples / profile.parameters if profile.parameters > 0 else 0,
            "power_utilization": total_energy / (training_time / 3600) if training_time > 0 else 0
        }
        
        # Calculate relative efficiency scores (0-100)
        benchmarks = {
            "samples_per_kwh": 10000,  # Benchmark values
            "co2_per_sample": 0.0001,
            "energy_per_parameter": 1e-9
        }
        
        efficiency_scores = {}
        for metric, value in efficiency_metrics.items():
            if metric in benchmarks:
                if metric == "co2_per_sample":  # Lower is better
                    score = max(0, 100 * (1 - value / benchmarks[metric]))
                else:  # Higher is better
                    score = min(100, 100 * (value / benchmarks[metric]))
                efficiency_scores[f"{metric}_score"] = score
        
        # Overall efficiency score
        scores = list(efficiency_scores.values())
        efficiency_metrics["overall_efficiency_score"] = statistics.mean(scores) if scores else 0
        efficiency_metrics.update(efficiency_scores)
        
        return efficiency_metrics
    
    @handle_gracefully(severity=ErrorSeverity.MEDIUM, fallback_value={})
    def compare_architectures(self, architecture_ids: List[str], 
                            metric: str = "carbon_efficiency") -> Dict[str, Any]:
        """Compare multiple architectures on specified metrics."""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for architecture comparison")
            return {}
        
        comparison_data = []
        
        for arch_id in architecture_ids:
            profile = self.architecture_profiles.get(arch_id)
            if not profile:
                continue
            
            # Get benchmark results for this architecture
            arch_results = [r for r in self.benchmark_results if r.architecture_profile.architecture_id == arch_id]
            
            if arch_results:
                avg_efficiency = statistics.mean([r.efficiency_score for r in arch_results])
                avg_carbon = statistics.mean([r.carbon_metrics.get("total_co2_kg", 0) for r in arch_results])
                avg_accuracy = statistics.mean([r.accuracy_metrics.get("accuracy", 0) for r in arch_results])
            else:
                # Use profile estimates
                avg_efficiency = profile.carbon_efficiency_score
                avg_carbon = profile.co2_per_sample * 1000  # Estimate for 1000 samples
                avg_accuracy = 85.0  # Default estimate
            
            comparison_data.append({
                "architecture_id": arch_id,
                "family": profile.family.value,
                "parameters": profile.parameters,
                "efficiency_score": avg_efficiency,
                "carbon_cost": avg_carbon,
                "accuracy": avg_accuracy,
                "complexity": profile.computational_complexity,
                "memory_requirement": profile.layers * 100  # Simplified
            })
        
        if not comparison_data:
            return {}
        
        df = pd.DataFrame(comparison_data)
        
        # Statistical analysis
        analysis_results = {
            "summary_statistics": df.describe().to_dict(),
            "correlations": df.select_dtypes(include=[np.number]).corr().to_dict(),
            "rankings": {}
        }
        
        # Rank architectures by different metrics
        for col in ["efficiency_score", "carbon_cost", "accuracy", "parameters"]:
            if col in df.columns:
                ascending = col == "carbon_cost"  # Lower carbon cost is better
                ranked = df.sort_values(col, ascending=ascending)
                analysis_results["rankings"][col] = ranked["architecture_id"].tolist()
        
        # Pareto frontier analysis (efficiency vs accuracy)
        if len(comparison_data) > 2:
            pareto_frontier = self._calculate_pareto_frontier(
                df, "efficiency_score", "accuracy"
            )
            analysis_results["pareto_frontier"] = pareto_frontier
        
        # Statistical significance tests
        if len(comparison_data) > 2:
            significance_tests = self._perform_significance_tests(df)
            analysis_results["statistical_tests"] = significance_tests
        
        return analysis_results
    
    def _calculate_pareto_frontier(self, df: pd.DataFrame, x_col: str, y_col: str) -> List[str]:
        """Calculate Pareto frontier for two objectives."""
        pareto_optimal = []
        
        for i, row in df.iterrows():
            is_dominated = False
            
            for j, other_row in df.iterrows():
                if i == j:
                    continue
                
                # Check if this point is dominated
                if (other_row[x_col] >= row[x_col] and other_row[y_col] >= row[y_col] and
                    (other_row[x_col] > row[x_col] or other_row[y_col] > row[y_col])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(row["architecture_id"])
        
        return pareto_optimal
    
    def _perform_significance_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests on architecture performance."""
        tests = {}
        
        # Group by architecture family
        if "family" in df.columns and len(df["family"].unique()) > 1:
            families = df["family"].unique()
            efficiency_groups = [df[df["family"] == family]["efficiency_score"].values 
                               for family in families if len(df[df["family"] == family]) > 1]
            
            if len(efficiency_groups) > 1 and all(len(group) > 0 for group in efficiency_groups):
                try:
                    # ANOVA test for efficiency differences between families
                    f_stat, p_value = stats.f_oneway(*efficiency_groups)
                    tests["anova_efficiency"] = {
                        "f_statistic": f_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
                except Exception as e:
                    logger.debug(f"ANOVA test failed: {e}")
        
        # Correlation significance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    try:
                        corr_coef, p_value = stats.pearsonr(df[col1], df[col2])
                        tests[f"correlation_{col1}_{col2}"] = {
                            "correlation": corr_coef,
                            "p_value": p_value,
                            "significant": p_value < 0.05
                        }
                    except Exception as e:
                        logger.debug(f"Correlation test failed for {col1}-{col2}: {e}")
        
        return tests
    
    def generate_architecture_insights(self) -> List[str]:
        """Generate insights from architecture analysis."""
        insights = []
        
        if len(self.architecture_profiles) < 2:
            return ["Insufficient data for architecture insights"]
        
        # Efficiency by family
        family_efficiency = defaultdict(list)
        for profile in self.architecture_profiles.values():
            family_efficiency[profile.family.value].append(profile.carbon_efficiency_score)
        
        if family_efficiency:
            best_family = max(family_efficiency.keys(), 
                            key=lambda k: statistics.mean(family_efficiency[k]) if family_efficiency[k] else 0)
            insights.append(f"Most carbon-efficient architecture family: {best_family}")
        
        # Parameter efficiency analysis
        param_efficiency = [(p.parameters, p.carbon_efficiency_score) for p in self.architecture_profiles.values()]
        if len(param_efficiency) > 1:
            # Check for efficiency plateau
            sorted_by_params = sorted(param_efficiency)
            efficiency_gains = []
            for i in range(1, len(sorted_by_params)):
                param_ratio = sorted_by_params[i][0] / sorted_by_params[i-1][0]
                efficiency_ratio = sorted_by_params[i][1] / max(sorted_by_params[i-1][1], 0.01)
                efficiency_gains.append(efficiency_ratio / param_ratio)
            
            if efficiency_gains:
                avg_gain = statistics.mean(efficiency_gains)
                if avg_gain < 0.5:
                    insights.append("Diminishing returns observed: Larger models show decreasing carbon efficiency per parameter")
                elif avg_gain > 1.5:
                    insights.append("Scaling benefits: Larger models show improving carbon efficiency per parameter")
        
        # Hardware requirement insights
        hw_reqs = Counter()
        for profile in self.architecture_profiles.values():
            for req in profile.hardware_requirements:
                hw_reqs[req] += 1
        
        if hw_reqs:
            most_common_req = hw_reqs.most_common(1)[0][0]
            insights.append(f"Most common hardware requirement: {most_common_req}")
        
        return insights


class CarbonAwareHyperparameterOptimizer:
    """Optimize hyperparameters considering both performance and carbon cost."""
    
    def __init__(self):
        self.optimization_history: List[HyperparameterOptimizationResult] = []
        self.carbon_models: Dict[str, Any] = {}  # Predictive models for carbon cost
        
    @resilient_operation(max_attempts=3)
    def optimize_hyperparameters(self, 
                                objective: OptimizationObjective,
                                search_space: Dict[str, Any],
                                carbon_predictor: Callable[[Dict[str, Any]], float],
                                performance_predictor: Callable[[Dict[str, Any]], float],
                                budget_kwh: float = 10.0,
                                max_iterations: int = 100) -> HyperparameterOptimizationResult:
        """Perform carbon-aware hyperparameter optimization."""
        
        optimization_id = str(uuid.uuid4())
        logger.info(f"Starting carbon-aware hyperparameter optimization: {optimization_id}")
        
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for hyperparameter optimization")
            return self._create_fallback_result(optimization_id, objective, search_space)
        
        # Generate candidate configurations
        candidates = self._generate_candidates(search_space, max_iterations)
        
        # Evaluate candidates
        evaluation_results = []
        total_carbon_cost = 0.0
        
        for i, candidate in enumerate(candidates):
            if total_carbon_cost > budget_kwh:
                logger.info(f"Carbon budget exceeded at iteration {i}")
                break
            
            # Predict carbon cost and performance
            try:
                carbon_cost = carbon_predictor(candidate)
                performance = performance_predictor(candidate)
                
                # Multi-objective scoring
                score = self._calculate_multi_objective_score(
                    carbon_cost, performance, objective
                )
                
                evaluation_results.append({
                    "candidate": candidate,
                    "carbon_cost": carbon_cost,
                    "performance": performance,
                    "score": score,
                    "iteration": i
                })
                
                total_carbon_cost += carbon_cost
                
            except Exception as e:
                logger.warning(f"Failed to evaluate candidate {i}: {e}")
                continue
        
        if not evaluation_results:
            return self._create_fallback_result(optimization_id, objective, search_space)
        
        # Find best configuration
        best_result = max(evaluation_results, key=lambda x: x["score"])
        
        # Statistical analysis
        convergence_metrics = self._analyze_convergence(evaluation_results)
        recommendations = self._generate_optimization_recommendations(
            evaluation_results, objective
        )
        
        # Significance testing
        significance_tests = self._test_optimization_significance(evaluation_results)
        
        result = HyperparameterOptimizationResult(
            optimization_id=optimization_id,
            objective=objective,
            search_space=search_space,
            best_params=best_result["candidate"],
            best_score=best_result["score"],
            carbon_cost=total_carbon_cost,
            optimization_history=evaluation_results,
            convergence_metrics=convergence_metrics,
            recommendations=recommendations,
            statistical_significance=significance_tests
        )
        
        self.optimization_history.append(result)
        
        logger.info(f"Optimization completed: {len(evaluation_results)} evaluations, "
                   f"best score: {best_result['score']:.4f}")
        
        return result
    
    def _generate_candidates(self, search_space: Dict[str, Any], max_iterations: int) -> List[Dict[str, Any]]:
        """Generate candidate hyperparameter configurations."""
        candidates = []
        
        # Convert search space to sklearn format
        param_grid = {}
        for param, values in search_space.items():
            if isinstance(values, dict) and "type" in values:
                if values["type"] == "continuous":
                    # Generate continuous values
                    low, high = values["low"], values["high"]
                    param_values = np.linspace(low, high, min(10, max_iterations // len(search_space)))
                elif values["type"] == "discrete":
                    param_values = values["values"]
                else:
                    param_values = [values.get("default", 1.0)]
            elif isinstance(values, list):
                param_values = values
            else:
                param_values = [values]
            
            param_grid[param] = param_values
        
        # Generate parameter combinations
        if SKLEARN_AVAILABLE:
            # Use sklearn's ParameterGrid for systematic exploration
            grid = ParameterGrid(param_grid)
            candidates = list(grid)[:max_iterations]
        else:
            # Fallback: random sampling
            import random
            for _ in range(min(max_iterations, 50)):
                candidate = {}
                for param, values in param_grid.items():
                    candidate[param] = random.choice(values)
                candidates.append(candidate)
        
        # Add adaptive sampling based on previous results
        if len(self.optimization_history) > 0:
            adaptive_candidates = self._generate_adaptive_candidates(search_space, 10)
            candidates.extend(adaptive_candidates)
        
        return candidates[:max_iterations]
    
    def _generate_adaptive_candidates(self, search_space: Dict[str, Any], 
                                    num_candidates: int) -> List[Dict[str, Any]]:
        """Generate candidates based on previous optimization results."""
        if not self.optimization_history:
            return []
        
        # Get best configurations from previous runs
        best_configs = []
        for result in self.optimization_history[-5:]:  # Last 5 optimizations
            if result.optimization_history:
                best_from_run = max(result.optimization_history, key=lambda x: x["score"])
                best_configs.append(best_from_run["candidate"])
        
        if not best_configs:
            return []
        
        # Generate variations of best configurations
        adaptive_candidates = []
        for _ in range(num_candidates):
            base_config = best_configs[len(adaptive_candidates) % len(best_configs)]
            
            # Add noise to create variations
            varied_config = base_config.copy()
            for param, value in varied_config.items():
                if isinstance(value, (int, float)):
                    noise_factor = 0.1  # 10% variation
                    noise = np.random.normal(0, abs(value) * noise_factor)
                    varied_config[param] = type(value)(value + noise)
            
            adaptive_candidates.append(varied_config)
        
        return adaptive_candidates
    
    def _calculate_multi_objective_score(self, carbon_cost: float, performance: float, 
                                       objective: OptimizationObjective) -> float:
        """Calculate multi-objective score based on optimization objective."""
        
        if objective == OptimizationObjective.MINIMIZE_CARBON:
            # Minimize carbon cost primarily
            return -carbon_cost + 0.1 * performance
        
        elif objective == OptimizationObjective.MAXIMIZE_EFFICIENCY:
            # Maximize performance per unit carbon
            return performance / max(carbon_cost, 0.001)
        
        elif objective == OptimizationObjective.BALANCE_ACCURACY_CARBON:
            # Balance both objectives equally
            normalized_performance = performance / 100.0  # Assume performance is percentage
            normalized_carbon = carbon_cost / 10.0  # Normalize carbon cost
            return normalized_performance - normalized_carbon
        
        elif objective == OptimizationObjective.MINIMIZE_ENERGY:
            # Similar to carbon but focus on energy
            return -carbon_cost * 0.5 + 0.1 * performance  # Assuming carbon correlates with energy
        
        else:
            # Default: balance performance and carbon
            return performance - carbon_cost
    
    def _analyze_convergence(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze optimization convergence."""
        if len(results) < 5:
            return {}
        
        scores = [r["score"] for r in results]
        iterations = list(range(len(scores)))
        
        # Calculate convergence metrics
        metrics = {}
        
        # Best score improvement over time
        best_scores = []
        current_best = float('-inf')
        for score in scores:
            current_best = max(current_best, score)
            best_scores.append(current_best)
        
        # Convergence rate (how quickly we approach best score)
        final_best = best_scores[-1]
        convergence_points = []
        for i, score in enumerate(best_scores):
            if final_best > 0:
                convergence_points.append(score / final_best)
        
        if convergence_points:
            # Calculate when we reached 90% of final performance
            target_ratio = 0.9
            convergence_iteration = len(convergence_points)
            for i, ratio in enumerate(convergence_points):
                if ratio >= target_ratio:
                    convergence_iteration = i
                    break
            
            metrics["convergence_iteration"] = convergence_iteration
            metrics["convergence_rate"] = convergence_iteration / len(scores)
        
        # Score variance (stability)
        if len(scores) > 1:
            metrics["score_variance"] = np.var(scores)
            metrics["score_improvement"] = (scores[-1] - scores[0]) / max(abs(scores[0]), 0.001)
        
        # Exploration vs exploitation balance
        carbon_costs = [r["carbon_cost"] for r in results]
        performances = [r["performance"] for r in results]
        
        if len(set(carbon_costs)) > 1:
            metrics["carbon_diversity"] = np.std(carbon_costs) / max(np.mean(carbon_costs), 0.001)
        if len(set(performances)) > 1:
            metrics["performance_diversity"] = np.std(performances) / max(np.mean(performances), 0.001)
        
        return metrics
    
    def _generate_optimization_recommendations(self, results: List[Dict[str, Any]], 
                                             objective: OptimizationObjective) -> List[str]:
        """Generate recommendations based on optimization results."""
        recommendations = []
        
        if not results:
            return ["Insufficient data for recommendations"]
        
        # Analyze parameter importance
        best_results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]
        worst_results = sorted(results, key=lambda x: x["score"])[:5]
        
        # Find parameters that consistently appear in best results
        param_importance = defaultdict(list)
        for result in best_results:
            for param, value in result["candidate"].items():
                param_importance[param].append(value)
        
        for param, values in param_importance.items():
            if len(set(values)) == 1:  # Same value in all best results
                recommendations.append(f"Parameter '{param}' shows strong preference for value {values[0]}")
            elif len(values) > 1:
                avg_val = statistics.mean(values) if all(isinstance(v, (int, float)) for v in values) else None
                if avg_val is not None:
                    recommendations.append(f"Parameter '{param}' optimal range around {avg_val:.3f}")
        
        # Carbon efficiency recommendations
        carbon_costs = [r["carbon_cost"] for r in results]
        performances = [r["performance"] for r in results]
        
        if len(carbon_costs) > 1 and len(performances) > 1:
            # Find configurations with good carbon efficiency
            efficiency_ratios = [p/max(c, 0.001) for p, c in zip(performances, carbon_costs)]
            best_efficiency_idx = np.argmax(efficiency_ratios)
            best_efficiency_config = results[best_efficiency_idx]["candidate"]
            
            recommendations.append(f"Most carbon-efficient configuration: {best_efficiency_config}")
        
        # Objective-specific recommendations
        if objective == OptimizationObjective.MINIMIZE_CARBON:
            min_carbon_result = min(results, key=lambda x: x["carbon_cost"])
            recommendations.append(f"Minimum carbon configuration achieves {min_carbon_result['carbon_cost']:.4f} kWh")
        
        elif objective == OptimizationObjective.BALANCE_ACCURACY_CARBON:
            # Find Pareto optimal points
            pareto_configs = self._find_pareto_optimal_configs(results)
            recommendations.append(f"Found {len(pareto_configs)} Pareto optimal configurations")
        
        return recommendations
    
    def _find_pareto_optimal_configs(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find Pareto optimal configurations in carbon-performance space."""
        pareto_optimal = []
        
        for result in results:
            is_dominated = False
            performance = result["performance"]
            carbon_cost = result["carbon_cost"]
            
            for other in results:
                if result == other:
                    continue
                
                other_performance = other["performance"]
                other_carbon = other["carbon_cost"]
                
                # Check if this point is dominated (worse in both objectives)
                if (other_performance >= performance and other_carbon <= carbon_cost and 
                    (other_performance > performance or other_carbon < carbon_cost)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(result)
        
        return pareto_optimal
    
    def _test_optimization_significance(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Test statistical significance of optimization results."""
        if len(results) < 10:
            return {"insufficient_data": True}
        
        scores = [r["score"] for r in results]
        carbon_costs = [r["carbon_cost"] for r in results]
        performances = [r["performance"] for r in results]
        
        tests = {}
        
        # Test if performance and carbon cost are significantly correlated
        if len(set(performances)) > 1 and len(set(carbon_costs)) > 1:
            try:
                corr_coef, p_value = stats.pearsonr(performances, carbon_costs)
                tests["performance_carbon_correlation"] = corr_coef
                tests["performance_carbon_p_value"] = p_value
                tests["correlation_significant"] = p_value < 0.05
            except Exception:
                pass
        
        # Test improvement over random baseline
        if len(scores) > 5:
            # Compare top 20% vs bottom 20%
            sorted_scores = sorted(scores)
            top_20_pct = sorted_scores[int(0.8 * len(sorted_scores)):]
            bottom_20_pct = sorted_scores[:int(0.2 * len(sorted_scores))]
            
            if len(top_20_pct) > 1 and len(bottom_20_pct) > 1:
                try:
                    t_stat, p_value = stats.ttest_ind(top_20_pct, bottom_20_pct)
                    tests["improvement_t_statistic"] = t_stat
                    tests["improvement_p_value"] = p_value
                    tests["significant_improvement"] = p_value < 0.05
                except Exception:
                    pass
        
        return tests
    
    def _create_fallback_result(self, optimization_id: str, objective: OptimizationObjective, 
                               search_space: Dict[str, Any]) -> HyperparameterOptimizationResult:
        """Create fallback result when optimization fails."""
        # Use default/middle values from search space
        default_params = {}
        for param, values in search_space.items():
            if isinstance(values, dict):
                if "default" in values:
                    default_params[param] = values["default"]
                elif "low" in values and "high" in values:
                    default_params[param] = (values["low"] + values["high"]) / 2
            elif isinstance(values, list) and values:
                default_params[param] = values[0]
        
        return HyperparameterOptimizationResult(
            optimization_id=optimization_id,
            objective=objective,
            search_space=search_space,
            best_params=default_params,
            best_score=0.0,
            carbon_cost=0.0,
            optimization_history=[],
            convergence_metrics={},
            recommendations=["Optimization failed - using default parameters"],
            statistical_significance={}
        )


class CarbonEfficiencyLeaderboard:
    """Maintain leaderboards for carbon efficiency across different categories."""
    
    def __init__(self):
        self.leaderboards: Dict[str, List[Dict[str, Any]]] = {
            "overall": [],
            "by_architecture": defaultdict(list),
            "by_dataset": defaultdict(list),
            "by_task": defaultdict(list),
            "by_organization": defaultdict(list)
        }
        self.submission_history: List[Dict[str, Any]] = []
        
    def submit_result(self, submission: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a result to the leaderboard."""
        submission_id = str(uuid.uuid4())
        submission["submission_id"] = submission_id
        submission["submission_time"] = datetime.now()
        
        # Validate submission
        validation_result = self._validate_submission(submission)
        if not validation_result["valid"]:
            return validation_result
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(submission)
        submission["efficiency_score"] = efficiency_score
        
        # Add to appropriate leaderboards
        self._add_to_leaderboards(submission)
        
        # Record submission
        self.submission_history.append(submission)
        
        # Calculate ranking
        ranking_info = self._get_ranking_info(submission)
        
        logger.info(f"Leaderboard submission accepted: {submission_id}, score: {efficiency_score:.4f}")
        
        return {
            "valid": True,
            "submission_id": submission_id,
            "efficiency_score": efficiency_score,
            "ranking_info": ranking_info,
            "leaderboard_position": ranking_info.get("overall_rank", 0)
        }
    
    def _validate_submission(self, submission: Dict[str, Any]) -> Dict[str, bool]:
        """Validate a leaderboard submission."""
        required_fields = [
            "model_name", "architecture", "dataset", "task_type",
            "carbon_emissions_kg", "energy_consumption_kwh", "training_time_hours",
            "accuracy_metric", "accuracy_value", "hardware_used"
        ]
        
        validation = {"valid": True, "errors": []}
        
        for field in required_fields:
            if field not in submission:
                validation["errors"].append(f"Missing required field: {field}")
                validation["valid"] = False
        
        # Validate numeric fields
        numeric_fields = ["carbon_emissions_kg", "energy_consumption_kwh", "training_time_hours", "accuracy_value"]
        for field in numeric_fields:
            if field in submission:
                try:
                    float(submission[field])
                    if submission[field] < 0:
                        validation["errors"].append(f"Field {field} must be non-negative")
                        validation["valid"] = False
                except (ValueError, TypeError):
                    validation["errors"].append(f"Field {field} must be numeric")
                    validation["valid"] = False
        
        # Validate accuracy value is reasonable (0-100 for percentage metrics)
        if "accuracy_value" in submission:
            acc_value = submission["accuracy_value"]
            if submission.get("accuracy_metric", "").endswith("%") and (acc_value < 0 or acc_value > 100):
                validation["errors"].append("Accuracy percentage must be between 0 and 100")
                validation["valid"] = False
        
        return validation
    
    def _calculate_efficiency_score(self, submission: Dict[str, Any]) -> float:
        """Calculate carbon efficiency score for leaderboard ranking."""
        carbon_kg = submission["carbon_emissions_kg"]
        accuracy = submission["accuracy_value"]
        energy_kwh = submission["energy_consumption_kwh"]
        
        # Multi-factor efficiency score
        # Higher accuracy and lower carbon/energy = higher score
        
        # Normalize accuracy (assume 0-100 scale)
        normalized_accuracy = min(accuracy / 100.0, 1.0)
        
        # Carbon penalty (lower carbon = higher score)
        carbon_penalty = 1.0 / (1.0 + carbon_kg)
        
        # Energy penalty (lower energy = higher score)
        energy_penalty = 1.0 / (1.0 + energy_kwh)
        
        # Combined efficiency score
        efficiency_score = normalized_accuracy * carbon_penalty * energy_penalty * 1000
        
        # Bonus for exceptional performance
        if normalized_accuracy > 0.95 and carbon_kg < 0.1:
            efficiency_score *= 1.2  # 20% bonus for high accuracy + low carbon
        
        return efficiency_score
    
    def _add_to_leaderboards(self, submission: Dict[str, Any]):
        """Add submission to appropriate leaderboard categories."""
        # Overall leaderboard
        self.leaderboards["overall"].append(submission)
        self.leaderboards["overall"].sort(key=lambda x: x["efficiency_score"], reverse=True)
        self.leaderboards["overall"] = self.leaderboards["overall"][:100]  # Keep top 100
        
        # Architecture-specific leaderboard
        arch = submission["architecture"]
        self.leaderboards["by_architecture"][arch].append(submission)
        self.leaderboards["by_architecture"][arch].sort(key=lambda x: x["efficiency_score"], reverse=True)
        self.leaderboards["by_architecture"][arch] = self.leaderboards["by_architecture"][arch][:50]
        
        # Dataset-specific leaderboard
        dataset = submission["dataset"]
        self.leaderboards["by_dataset"][dataset].append(submission)
        self.leaderboards["by_dataset"][dataset].sort(key=lambda x: x["efficiency_score"], reverse=True)
        self.leaderboards["by_dataset"][dataset] = self.leaderboards["by_dataset"][dataset][:50]
        
        # Organization-specific (if provided)
        if "organization" in submission:
            org = submission["organization"]
            self.leaderboards["by_organization"][org].append(submission)
            self.leaderboards["by_organization"][org].sort(key=lambda x: x["efficiency_score"], reverse=True)
    
    def _get_ranking_info(self, submission: Dict[str, Any]) -> Dict[str, Any]:
        """Get ranking information for a submission."""
        ranking_info = {}
        
        # Overall ranking
        overall_rank = 1
        for entry in self.leaderboards["overall"]:
            if entry["submission_id"] == submission["submission_id"]:
                ranking_info["overall_rank"] = overall_rank
                break
            overall_rank += 1
        
        # Architecture ranking
        arch = submission["architecture"]
        arch_rank = 1
        for entry in self.leaderboards["by_architecture"][arch]:
            if entry["submission_id"] == submission["submission_id"]:
                ranking_info["architecture_rank"] = arch_rank
                break
            arch_rank += 1
        
        # Dataset ranking
        dataset = submission["dataset"]
        dataset_rank = 1
        for entry in self.leaderboards["by_dataset"][dataset]:
            if entry["submission_id"] == submission["submission_id"]:
                ranking_info["dataset_rank"] = dataset_rank
                break
            dataset_rank += 1
        
        ranking_info["total_submissions"] = len(self.submission_history)
        ranking_info["percentile"] = (1 - (ranking_info.get("overall_rank", 1) - 1) / max(len(self.leaderboards["overall"]), 1)) * 100
        
        return ranking_info
    
    def get_leaderboard(self, category: str = "overall", limit: int = 10) -> List[Dict[str, Any]]:
        """Get leaderboard for specified category."""
        if category == "overall":
            return self.leaderboards["overall"][:limit]
        elif category.startswith("by_"):
            category_name = category[3:]  # Remove "by_" prefix
            if category_name in self.leaderboards:
                # Return top entries across all subcategories
                all_entries = []
                for subcategory_entries in self.leaderboards[category_name].values():
                    all_entries.extend(subcategory_entries)
                all_entries.sort(key=lambda x: x["efficiency_score"], reverse=True)
                return all_entries[:limit]
        
        return []
    
    def get_leaderboard_statistics(self) -> Dict[str, Any]:
        """Get comprehensive leaderboard statistics."""
        stats = {
            "total_submissions": len(self.submission_history),
            "unique_architectures": len(self.leaderboards["by_architecture"]),
            "unique_datasets": len(self.leaderboards["by_dataset"]),
            "unique_organizations": len(self.leaderboards["by_organization"])
        }
        
        if self.submission_history:
            efficiency_scores = [s["efficiency_score"] for s in self.submission_history]
            carbon_emissions = [s["carbon_emissions_kg"] for s in self.submission_history]
            accuracy_values = [s["accuracy_value"] for s in self.submission_history]
            
            stats.update({
                "efficiency_score_statistics": {
                    "mean": statistics.mean(efficiency_scores),
                    "median": statistics.median(efficiency_scores),
                    "max": max(efficiency_scores),
                    "min": min(efficiency_scores),
                    "std": statistics.stdev(efficiency_scores) if len(efficiency_scores) > 1 else 0
                },
                "carbon_statistics": {
                    "mean_kg": statistics.mean(carbon_emissions),
                    "median_kg": statistics.median(carbon_emissions),
                    "min_kg": min(carbon_emissions),
                    "max_kg": max(carbon_emissions)
                },
                "accuracy_statistics": {
                    "mean": statistics.mean(accuracy_values),
                    "median": statistics.median(accuracy_values),
                    "max": max(accuracy_values),
                    "min": min(accuracy_values)
                }
            })
        
        # Trending analysis
        if len(self.submission_history) > 10:
            recent_submissions = self.submission_history[-30:]  # Last 30 submissions
            older_submissions = self.submission_history[-60:-30] if len(self.submission_history) > 60 else []
            
            if older_submissions:
                recent_avg_efficiency = statistics.mean([s["efficiency_score"] for s in recent_submissions])
                older_avg_efficiency = statistics.mean([s["efficiency_score"] for s in older_submissions])
                
                trend = "improving" if recent_avg_efficiency > older_avg_efficiency else "declining"
                stats["efficiency_trend"] = trend
                stats["efficiency_improvement_rate"] = (recent_avg_efficiency - older_avg_efficiency) / older_avg_efficiency
        
        return stats


class ResearchPublicationMetricsGenerator:
    """Generate metrics and analysis for research publications."""
    
    def __init__(self):
        self.publications: List[ResearchPublication] = []
        self.citation_network: Dict[str, List[str]] = {}  # pub_id -> [citing_pub_ids]
        self.implementation_tracking: Dict[str, List[Dict[str, Any]]] = {}
        
    def add_publication(self, publication: ResearchPublication) -> str:
        """Add a publication to the tracking system."""
        self.publications.append(publication)
        logger.info(f"Added publication: {publication.title}")
        return publication.publication_id
    
    def calculate_carbon_impact_factor(self, publication_id: str) -> float:
        """Calculate carbon impact factor for a publication."""
        publication = self._get_publication(publication_id)
        if not publication:
            return 0.0
        
        # Factors contributing to carbon impact:
        # 1. Direct carbon reduction potential
        # 2. Citation count (influence)
        # 3. Real-world implementations
        # 4. Replication studies (validation)
        
        base_impact = publication.co2_reduction_potential_kg
        
        # Citation multiplier (more citations = higher impact)
        citation_multiplier = 1 + (publication.citation_count * 0.1)
        
        # Implementation multiplier (real-world usage)
        impl_multiplier = 1 + (publication.real_world_implementations * 0.2)
        
        # Replication multiplier (scientific validation)
        replication_multiplier = 1 + (publication.replication_studies * 0.15)
        
        # Network effect (citations to this paper's citations)
        network_effect = self._calculate_citation_network_effect(publication_id)
        
        carbon_impact_factor = (base_impact * citation_multiplier * 
                              impl_multiplier * replication_multiplier * 
                              (1 + network_effect))
        
        publication.carbon_impact_factor = carbon_impact_factor
        return carbon_impact_factor
    
    def _get_publication(self, publication_id: str) -> Optional[ResearchPublication]:
        """Get publication by ID."""
        for pub in self.publications:
            if pub.publication_id == publication_id:
                return pub
        return None
    
    def _calculate_citation_network_effect(self, publication_id: str) -> float:
        """Calculate network effect based on citation patterns."""
        if publication_id not in self.citation_network:
            return 0.0
        
        citing_papers = self.citation_network[publication_id]
        
        # Calculate influence of papers that cite this paper
        total_influence = 0.0
        for citing_id in citing_papers:
            citing_pub = self._get_publication(citing_id)
            if citing_pub:
                # Weight by the citing paper's own citation count
                citing_influence = 1 + (citing_pub.citation_count * 0.05)
                total_influence += citing_influence
        
        # Normalize by number of citing papers
        if len(citing_papers) > 0:
            return total_influence / len(citing_papers) * 0.1
        
        return 0.0
    
    def track_implementation(self, publication_id: str, implementation_data: Dict[str, Any]):
        """Track real-world implementation of research."""
        if publication_id not in self.implementation_tracking:
            self.implementation_tracking[publication_id] = []
        
        implementation_data["tracked_at"] = datetime.now()
        implementation_data["implementation_id"] = str(uuid.uuid4())
        
        self.implementation_tracking[publication_id].append(implementation_data)
        
        # Update publication metrics
        publication = self._get_publication(publication_id)
        if publication:
            publication.real_world_implementations = len(self.implementation_tracking[publication_id])
        
        logger.info(f"Tracked implementation for publication {publication_id}")
    
    def generate_research_impact_report(self, publication_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive research impact report."""
        if publication_id:
            publications = [self._get_publication(publication_id)]
            publications = [p for p in publications if p is not None]
        else:
            publications = self.publications
        
        if not publications:
            return {"error": "No publications found"}
        
        # Calculate metrics for each publication
        publication_metrics = []
        for pub in publications:
            impact_factor = self.calculate_carbon_impact_factor(pub.publication_id)
            
            implementations = self.implementation_tracking.get(pub.publication_id, [])
            impl_details = self._analyze_implementations(implementations)
            
            metrics = {
                "publication_id": pub.publication_id,
                "title": pub.title,
                "carbon_impact_factor": impact_factor,
                "citation_count": pub.citation_count,
                "replication_studies": pub.replication_studies,
                "real_world_implementations": pub.real_world_implementations,
                "co2_reduction_potential_kg": pub.co2_reduction_potential_kg,
                "confidence_score": pub.confidence_score,
                "implementation_analysis": impl_details
            }
            
            publication_metrics.append(metrics)
        
        # Aggregate statistics
        if len(publication_metrics) > 1:
            impact_factors = [p["carbon_impact_factor"] for p in publication_metrics]
            aggregate_stats = {
                "total_publications": len(publication_metrics),
                "mean_impact_factor": statistics.mean(impact_factors),
                "median_impact_factor": statistics.median(impact_factors),
                "max_impact_factor": max(impact_factors),
                "total_co2_reduction_potential": sum(p["co2_reduction_potential_kg"] for p in publication_metrics),
                "total_implementations": sum(p["real_world_implementations"] for p in publication_metrics),
                "total_citations": sum(p["citation_count"] for p in publication_metrics)
            }
        else:
            aggregate_stats = publication_metrics[0] if publication_metrics else {}
        
        # Research trends analysis
        trends = self._analyze_research_trends()
        
        return {
            "publication_metrics": publication_metrics,
            "aggregate_statistics": aggregate_stats,
            "research_trends": trends,
            "top_impact_publications": sorted(publication_metrics, 
                                            key=lambda x: x["carbon_impact_factor"], 
                                            reverse=True)[:10],
            "generated_at": datetime.now()
        }
    
    def _analyze_implementations(self, implementations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze implementation patterns and success."""
        if not implementations:
            return {"total": 0, "analysis": "No implementations tracked"}
        
        # Categorize implementations
        categories = defaultdict(int)
        success_rates = []
        carbon_savings = []
        
        for impl in implementations:
            category = impl.get("category", "unknown")
            categories[category] += 1
            
            if "success_rate" in impl:
                success_rates.append(impl["success_rate"])
            
            if "carbon_savings_kg" in impl:
                carbon_savings.append(impl["carbon_savings_kg"])
        
        analysis = {
            "total": len(implementations),
            "categories": dict(categories),
            "most_common_category": max(categories.keys(), key=categories.get) if categories else "none"
        }
        
        if success_rates:
            analysis["average_success_rate"] = statistics.mean(success_rates)
            analysis["success_rate_std"] = statistics.stdev(success_rates) if len(success_rates) > 1 else 0
        
        if carbon_savings:
            analysis["total_carbon_savings_kg"] = sum(carbon_savings)
            analysis["average_carbon_savings_kg"] = statistics.mean(carbon_savings)
        
        return analysis
    
    def _analyze_research_trends(self) -> Dict[str, Any]:
        """Analyze trends in research publications."""
        if len(self.publications) < 2:
            return {"insufficient_data": True}
        
        # Sort publications by publication date (using generated_at as proxy)
        sorted_pubs = sorted(self.publications, key=lambda x: x.generated_at)
        
        # Analyze trends over time
        trends = {}
        
        # Publication rate trend
        if len(sorted_pubs) > 5:
            recent_pubs = sorted_pubs[-12:]  # Last 12 publications
            older_pubs = sorted_pubs[-24:-12] if len(sorted_pubs) > 24 else sorted_pubs[:-12]
            
            if older_pubs:
                recent_rate = len(recent_pubs)
                older_rate = len(older_pubs)
                trends["publication_rate_change"] = (recent_rate - older_rate) / older_rate
        
        # Impact factor trends
        impact_factors = []
        for pub in sorted_pubs:
            if pub.carbon_impact_factor > 0:
                impact_factors.append(pub.carbon_impact_factor)
        
        if len(impact_factors) > 5:
            recent_impacts = impact_factors[-6:]
            older_impacts = impact_factors[-12:-6] if len(impact_factors) > 12 else impact_factors[:-6]
            
            if older_impacts:
                recent_avg = statistics.mean(recent_impacts)
                older_avg = statistics.mean(older_impacts)
                trends["impact_factor_trend"] = (recent_avg - older_avg) / older_avg
        
        # Research focus trends (based on keywords)
        keyword_trends = defaultdict(int)
        for pub in sorted_pubs[-20:]:  # Recent publications
            for keyword in pub.keywords:
                keyword_trends[keyword] += 1
        
        if keyword_trends:
            trends["trending_keywords"] = sorted(keyword_trends.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]
        
        return trends


class ResearchAnalyticsPlatform:
    """Main platform coordinating all research analytics capabilities."""
    
    def __init__(self):
        self.architecture_analyzer = ModelArchitectureAnalyzer()
        self.hyperparameter_optimizer = CarbonAwareHyperparameterOptimizer()
        self.leaderboard = CarbonEfficiencyLeaderboard()
        self.publication_metrics = ResearchPublicationMetricsGenerator()
        
        logger.info("Research Analytics Platform initialized")
    
    def run_comprehensive_analysis(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive research analysis across all modules."""
        results = {
            "analysis_id": str(uuid.uuid4()),
            "started_at": datetime.now(),
            "config": analysis_config
        }
        
        # Architecture analysis
        if analysis_config.get("analyze_architectures", True):
            logger.info("Running architecture analysis...")
            arch_ids = analysis_config.get("architecture_ids", list(self.architecture_analyzer.architecture_profiles.keys()))
            if arch_ids:
                arch_comparison = self.architecture_analyzer.compare_architectures(arch_ids)
                arch_insights = self.architecture_analyzer.generate_architecture_insights()
                results["architecture_analysis"] = {
                    "comparison": arch_comparison,
                    "insights": arch_insights
                }
        
        # Hyperparameter optimization analysis
        if analysis_config.get("analyze_hyperparameters", True):
            logger.info("Analyzing hyperparameter optimization trends...")
            if self.hyperparameter_optimizer.optimization_history:
                opt_analysis = self._analyze_optimization_trends()
                results["hyperparameter_analysis"] = opt_analysis
        
        # Leaderboard analysis
        if analysis_config.get("analyze_leaderboard", True):
            logger.info("Analyzing leaderboard trends...")
            leaderboard_stats = self.leaderboard.get_leaderboard_statistics()
            results["leaderboard_analysis"] = leaderboard_stats
        
        # Publication impact analysis
        if analysis_config.get("analyze_publications", True):
            logger.info("Analyzing research publications...")
            publication_report = self.publication_metrics.generate_research_impact_report()
            results["publication_analysis"] = publication_report
        
        # Cross-module insights
        if analysis_config.get("generate_insights", True):
            logger.info("Generating cross-module insights...")
            insights = self._generate_cross_module_insights(results)
            results["cross_module_insights"] = insights
        
        results["completed_at"] = datetime.now()
        results["duration_seconds"] = (results["completed_at"] - results["started_at"]).total_seconds()
        
        logger.info(f"Comprehensive analysis completed in {results['duration_seconds']:.2f} seconds")
        return results
    
    def _analyze_optimization_trends(self) -> Dict[str, Any]:
        """Analyze trends in hyperparameter optimization."""
        history = self.hyperparameter_optimizer.optimization_history
        
        if not history:
            return {"no_data": True}
        
        # Analyze optimization objectives
        objective_counts = defaultdict(int)
        objective_performance = defaultdict(list)
        
        for result in history:
            obj = result.objective.value
            objective_counts[obj] += 1
            objective_performance[obj].append(result.best_score)
        
        # Convergence analysis
        convergence_data = []
        for result in history:
            if result.convergence_metrics:
                convergence_data.append(result.convergence_metrics)
        
        trends = {
            "total_optimizations": len(history),
            "objective_distribution": dict(objective_counts),
            "objective_performance": {obj: {
                "mean_score": statistics.mean(scores),
                "max_score": max(scores),
                "score_variance": statistics.variance(scores) if len(scores) > 1 else 0
            } for obj, scores in objective_performance.items()},
            "convergence_trends": self._analyze_convergence_trends(convergence_data)
        }
        
        return trends
    
    def _analyze_convergence_trends(self, convergence_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze convergence trends across optimizations."""
        if not convergence_data:
            return {}
        
        # Extract convergence metrics
        convergence_rates = []
        score_improvements = []
        
        for data in convergence_data:
            if "convergence_rate" in data:
                convergence_rates.append(data["convergence_rate"])
            if "score_improvement" in data:
                score_improvements.append(data["score_improvement"])
        
        trends = {}
        
        if convergence_rates:
            trends["average_convergence_rate"] = statistics.mean(convergence_rates)
            trends["convergence_consistency"] = 1 - (statistics.stdev(convergence_rates) / statistics.mean(convergence_rates))
        
        if score_improvements:
            trends["average_improvement"] = statistics.mean(score_improvements)
            trends["improvement_reliability"] = len([x for x in score_improvements if x > 0]) / len(score_improvements)
        
        return trends
    
    def _generate_cross_module_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights that span multiple analysis modules."""
        insights = []
        
        # Architecture-Leaderboard insights
        arch_analysis = analysis_results.get("architecture_analysis", {})
        leaderboard_analysis = analysis_results.get("leaderboard_analysis", {})
        
        if arch_analysis and leaderboard_analysis:
            arch_rankings = arch_analysis.get("comparison", {}).get("rankings", {})
            if "efficiency_score" in arch_rankings and leaderboard_analysis.get("unique_architectures", 0) > 0:
                top_arch = arch_rankings["efficiency_score"][0] if arch_rankings["efficiency_score"] else "unknown"
                insights.append(f"Top performing architecture ({top_arch}) correlates with leaderboard submissions")
        
        # Optimization-Publication insights
        opt_analysis = analysis_results.get("hyperparameter_analysis", {})
        pub_analysis = analysis_results.get("publication_analysis", {})
        
        if opt_analysis and pub_analysis:
            opt_count = opt_analysis.get("total_optimizations", 0)
            pub_count = pub_analysis.get("aggregate_statistics", {}).get("total_publications", 0)
            
            if opt_count > 0 and pub_count > 0:
                ratio = opt_count / pub_count
                if ratio > 10:
                    insights.append("High optimization-to-publication ratio suggests thorough experimental validation")
                elif ratio < 2:
                    insights.append("Low optimization activity relative to publications may indicate research gaps")
        
        # Carbon efficiency trends
        if leaderboard_analysis.get("efficiency_trend") == "improving":
            insights.append("Community carbon efficiency is improving over time")
        elif leaderboard_analysis.get("efficiency_trend") == "declining":
            insights.append("Carbon efficiency appears to be declining - investigate causes")
        
        # Publication impact insights
        pub_stats = pub_analysis.get("aggregate_statistics", {})
        if pub_stats.get("total_co2_reduction_potential", 0) > 1000:
            insights.append(f"Research has significant carbon reduction potential: {pub_stats['total_co2_reduction_potential']:.1f} kg CO₂")
        
        # Implementation gap analysis
        total_pubs = pub_stats.get("total_publications", 0)
        total_impls = pub_stats.get("total_implementations", 0)
        if total_pubs > 0:
            impl_rate = total_impls / total_pubs
            if impl_rate < 0.1:
                insights.append("Low implementation rate suggests need for better research-to-practice transfer")
            elif impl_rate > 0.5:
                insights.append("High implementation rate indicates strong research-practice collaboration")
        
        return insights
    
    def export_research_dashboard_data(self) -> Dict[str, Any]:
        """Export data for research dashboard visualization."""
        dashboard_data = {
            "export_timestamp": datetime.now(),
            "architecture_profiles": len(self.architecture_analyzer.architecture_profiles),
            "optimization_experiments": len(self.hyperparameter_optimizer.optimization_history),
            "leaderboard_submissions": len(self.leaderboard.submission_history),
            "research_publications": len(self.publication_metrics.publications),
            
            # Quick stats for dashboard
            "quick_stats": {
                "top_architectures": self.architecture_analyzer.compare_architectures(
                    list(self.architecture_analyzer.architecture_profiles.keys())[:5]
                ).get("rankings", {}).get("efficiency_score", [])[:3],
                
                "recent_leaderboard": self.leaderboard.get_leaderboard("overall", 5),
                
                "top_publications": self.publication_metrics.generate_research_impact_report().get(
                    "top_impact_publications", []
                )[:3]
            }
        }
        
        return dashboard_data


# Convenience functions for easy integration
def create_research_platform() -> ResearchAnalyticsPlatform:
    """Create a research analytics platform with default configuration."""
    return ResearchAnalyticsPlatform()


def run_quick_analysis(architectures: List[str] = None) -> Dict[str, Any]:
    """Run quick research analysis with minimal configuration."""
    platform = create_research_platform()
    
    config = {
        "analyze_architectures": True,
        "analyze_hyperparameters": True,
        "analyze_leaderboard": True,
        "analyze_publications": True,
        "generate_insights": True
    }
    
    if architectures:
        config["architecture_ids"] = architectures
    
    return platform.run_comprehensive_analysis(config)