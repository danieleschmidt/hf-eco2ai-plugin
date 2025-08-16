"""Enterprise Auto-Scaling Engine for Dynamic Carbon Tracking Infrastructure.

This module implements intelligent auto-scaling capabilities for carbon tracking
infrastructure, supporting adaptive monitoring frequency, predictive scaling,
and global load distribution across data centers.
"""

import asyncio
import logging
import time
import json
import math
import numpy as np
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import statistics
import psutil
import socket
from pathlib import Path

try:
    import kubernetes
    from kubernetes import client, config
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

try:
    import boto3
    import botocore
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import monitoring_v3
    from google.cloud import compute_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.monitor import MonitorManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction for infrastructure resources."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"      # Horizontal scaling (more instances)
    SCALE_IN = "scale_in"        # Horizontal scaling (fewer instances)
    MAINTAIN = "maintain"


class LoadLevel(Enum):
    """System load levels for auto-scaling decisions."""
    CRITICAL = "critical"        # > 90% utilization
    HIGH = "high"               # 70-90% utilization
    MEDIUM = "medium"           # 30-70% utilization
    LOW = "low"                 # 10-30% utilization
    IDLE = "idle"               # < 10% utilization


class ScalingPolicy(Enum):
    """Auto-scaling policies for different scenarios."""
    AGGRESSIVE = "aggressive"     # Quick response, high cost
    BALANCED = "balanced"        # Moderate response, balanced cost
    CONSERVATIVE = "conservative" # Slow response, low cost
    PREDICTIVE = "predictive"    # ML-based proactive scaling
    ADAPTIVE = "adaptive"        # Self-learning optimization


@dataclass
class GeographicRegion:
    """Geographic region configuration for global scaling."""
    
    region_id: str
    region_name: str
    cloud_provider: str  # "aws", "gcp", "azure", "on_premise"
    carbon_intensity_gco2_kwh: float
    electricity_cost_per_kwh: float
    compute_cost_multiplier: float
    
    # Network connectivity
    latency_to_primary_ms: float
    bandwidth_gbps: float
    reliability_score: float  # 0.0 - 1.0
    
    # Capacity and scaling
    max_cpu_cores: int
    max_memory_gb: int
    max_storage_tb: int
    auto_scaling_enabled: bool
    current_utilization: float = 0.0
    
    # Geographic coordinates for optimization
    latitude: float = 0.0
    longitude: float = 0.0
    timezone: str = "UTC"


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    
    timestamp: float
    region_id: str
    
    # Resource utilization
    cpu_utilization: float
    memory_utilization: float
    storage_utilization: float
    network_utilization: float
    
    # Carbon tracking specific metrics
    tracking_requests_per_second: float
    gpu_clusters_monitored: int
    metrics_collection_latency_ms: float
    aggregation_latency_ms: float
    
    # Predictive indicators
    predicted_load_5min: float
    predicted_load_15min: float
    predicted_load_1hour: float
    
    # Cost and carbon metrics
    current_cost_per_hour: float
    carbon_emissions_kg_per_hour: float
    efficiency_score: float  # metrics/cost ratio


@dataclass
class ScalingDecision:
    """Auto-scaling decision with rationale."""
    
    decision_id: str
    timestamp: float
    region_id: str
    
    # Scaling decision
    direction: ScalingDirection
    resource_type: str  # "cpu", "memory", "instances", "storage"
    current_capacity: float
    target_capacity: float
    scaling_factor: float
    
    # Decision rationale
    trigger_metric: str
    trigger_value: float
    threshold_breached: float
    confidence_score: float
    
    # Impact estimates
    estimated_cost_impact: float
    estimated_carbon_impact: float
    estimated_performance_impact: float
    estimated_completion_time: float
    
    # Approval and execution
    auto_approved: bool
    requires_human_approval: bool
    execution_status: str = "pending"
    execution_start_time: Optional[float] = None
    execution_completion_time: Optional[float] = None


class TrainingLoadPredictor:
    """ML training load prediction for proactive scaling."""
    
    def __init__(self, history_window_hours: int = 168):  # 1 week
        """Initialize load predictor.
        
        Args:
            history_window_hours: Historical data window for predictions
        """
        self.history_window_hours = history_window_hours
        self.max_history_points = history_window_hours * 60  # 1 point per minute
        
        # Historical data storage
        self.load_history: deque = deque(maxlen=self.max_history_points)
        self.pattern_cache: Dict[str, List[float]] = {}
        
        # Prediction models (simplified)
        self.daily_patterns: Dict[int, List[float]] = {}  # hour -> avg_load
        self.weekly_patterns: Dict[int, List[float]] = {}  # day_of_week -> avg_load
        self.seasonal_factors: Dict[str, float] = {}
        
        # Model accuracy tracking
        self.prediction_accuracy: Dict[str, deque] = {
            "5min": deque(maxlen=288),  # 24 hours of 5-min predictions
            "15min": deque(maxlen=96),  # 24 hours of 15-min predictions
            "1hour": deque(maxlen=168)  # 1 week of hourly predictions
        }
        
        logger.info("Training load predictor initialized")
    
    def add_load_measurement(self, 
                           timestamp: float,
                           load_metrics: Dict[str, float]):
        """Add load measurement to historical data.
        
        Args:
            timestamp: Measurement timestamp
            load_metrics: Load metrics (CPU, memory, requests/sec, etc.)
        """
        measurement = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp),
            **load_metrics
        }
        
        self.load_history.append(measurement)
        
        # Update patterns periodically
        if len(self.load_history) % 60 == 0:  # Every hour
            self._update_patterns()
    
    def predict_load(self, 
                    horizon_minutes: int,
                    metric_name: str = "cpu_utilization") -> Tuple[float, float]:
        """Predict future load with confidence interval.
        
        Args:
            horizon_minutes: Prediction horizon in minutes
            metric_name: Metric to predict
            
        Returns:
            Tuple of (predicted_value, confidence_score)
        """
        if len(self.load_history) < 10:
            # Not enough data for prediction
            return 0.5, 0.0
        
        try:
            # Get recent measurements
            recent_values = [
                m[metric_name] for m in list(self.load_history)[-60:]  # Last hour
                if metric_name in m
            ]
            
            if not recent_values:
                return 0.5, 0.0
            
            # Current timestamp and target time
            current_time = time.time()
            target_time = current_time + (horizon_minutes * 60)
            target_dt = datetime.fromtimestamp(target_time)
            
            # Multiple prediction approaches
            predictions = []
            
            # 1. Trend-based prediction
            trend_pred = self._predict_trend(recent_values, horizon_minutes)
            predictions.append(trend_pred)
            
            # 2. Seasonal pattern prediction
            seasonal_pred = self._predict_seasonal(target_dt, metric_name)
            predictions.append(seasonal_pred)
            
            # 3. Moving average prediction
            ma_pred = self._predict_moving_average(recent_values, horizon_minutes)
            predictions.append(ma_pred)
            
            # 4. Pattern matching prediction
            pattern_pred = self._predict_pattern_matching(recent_values, horizon_minutes)
            predictions.append(pattern_pred)
            
            # Ensemble prediction
            valid_predictions = [p for p in predictions if not math.isnan(p)]
            
            if not valid_predictions:
                return np.mean(recent_values), 0.1
            
            # Weighted ensemble
            weights = [0.3, 0.4, 0.2, 0.1][:len(valid_predictions)]
            weights = [w / sum(weights) for w in weights]  # Normalize
            
            ensemble_pred = sum(p * w for p, w in zip(valid_predictions, weights))
            
            # Calculate confidence based on prediction variance
            prediction_variance = np.var(valid_predictions) if len(valid_predictions) > 1 else 0.1
            confidence = max(0.1, 1.0 - prediction_variance / max(ensemble_pred, 0.1))
            
            return max(0.0, min(1.0, ensemble_pred)), min(1.0, confidence)
            
        except Exception as e:
            logger.warning(f"Load prediction failed: {e}")
            return np.mean(recent_values) if recent_values else 0.5, 0.1
    
    def _predict_trend(self, values: List[float], horizon_minutes: int) -> float:
        """Predict based on linear trend."""
        if len(values) < 3:
            return np.mean(values)
        
        # Linear regression on recent values
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate trend
        slope = np.polyfit(x, y, 1)[0]
        
        # Project trend forward
        future_x = len(values) + (horizon_minutes / 60.0)  # Convert to same scale
        prediction = values[-1] + slope * (horizon_minutes / 60.0)
        
        return max(0.0, min(1.0, prediction))
    
    def _predict_seasonal(self, target_dt: datetime, metric_name: str) -> float:
        """Predict based on seasonal patterns."""
        hour = target_dt.hour
        day_of_week = target_dt.weekday()
        
        # Daily pattern
        if hour in self.daily_patterns:
            daily_pred = np.mean(self.daily_patterns[hour])
        else:
            daily_pred = 0.5
        
        # Weekly pattern
        if day_of_week in self.weekly_patterns:
            weekly_pred = np.mean(self.weekly_patterns[day_of_week])
        else:
            weekly_pred = 0.5
        
        # Combine patterns
        return (daily_pred + weekly_pred) / 2.0
    
    def _predict_moving_average(self, values: List[float], horizon_minutes: int) -> float:
        """Predict using moving average."""
        # Exponential moving average with decay
        alpha = 0.3  # Smoothing factor
        
        if not values:
            return 0.5
        
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        return ema
    
    def _predict_pattern_matching(self, values: List[float], horizon_minutes: int) -> float:
        """Predict using historical pattern matching."""
        if len(values) < 10:
            return np.mean(values)
        
        # Find similar historical patterns
        pattern_length = min(10, len(values))
        current_pattern = values[-pattern_length:]
        
        # Search for similar patterns in history
        similar_patterns = []
        
        for i in range(len(self.load_history) - pattern_length - 5):
            historical_pattern = [
                self.load_history[i + j].get("cpu_utilization", 0.5)
                for j in range(pattern_length)
            ]
            
            # Calculate pattern similarity
            similarity = 1.0 / (1.0 + np.mean(np.abs(np.array(current_pattern) - np.array(historical_pattern))))
            
            if similarity > 0.7:  # Threshold for similarity
                # Get the following values
                future_values = [
                    self.load_history[i + pattern_length + j].get("cpu_utilization", 0.5)
                    for j in range(min(5, len(self.load_history) - i - pattern_length))
                ]
                
                if future_values:
                    similar_patterns.append((similarity, future_values[0]))
        
        if similar_patterns:
            # Weighted average of similar patterns
            total_weight = sum(sim for sim, _ in similar_patterns)
            weighted_pred = sum(sim * val for sim, val in similar_patterns) / total_weight
            return weighted_pred
        
        return np.mean(values)
    
    def _update_patterns(self):
        """Update daily and weekly patterns from historical data."""
        if len(self.load_history) < 100:
            return
        
        # Update daily patterns
        hourly_data = defaultdict(list)
        daily_data = defaultdict(list)
        
        for measurement in self.load_history:
            dt = measurement["datetime"]
            hour = dt.hour
            day_of_week = dt.weekday()
            
            cpu_util = measurement.get("cpu_utilization", 0.5)
            
            hourly_data[hour].append(cpu_util)
            daily_data[day_of_week].append(cpu_util)
        
        # Calculate average patterns
        self.daily_patterns = {
            hour: values for hour, values in hourly_data.items()
            if len(values) >= 3
        }
        
        self.weekly_patterns = {
            day: values for day, values in daily_data.items()
            if len(values) >= 10
        }
        
        logger.debug(f"Updated patterns: {len(self.daily_patterns)} hourly, {len(self.weekly_patterns)} daily")
    
    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Get prediction accuracy statistics."""
        accuracy_stats = {}
        
        for horizon, accuracies in self.prediction_accuracy.items():
            if accuracies:
                accuracy_stats[horizon] = {
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                    "min_accuracy": np.min(accuracies),
                    "max_accuracy": np.max(accuracies),
                    "sample_count": len(accuracies)
                }
            else:
                accuracy_stats[horizon] = {"mean_accuracy": 0.0, "sample_count": 0}
        
        return accuracy_stats


class LoadBalancer:
    """Geographic load balancer for carbon tracking infrastructure."""
    
    def __init__(self):
        """Initialize load balancer."""
        self.regions: Dict[str, GeographicRegion] = {}
        self.routing_table: Dict[str, str] = {}  # request_source -> region_id
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Load balancing strategies
        self.strategies = {
            "round_robin": self._route_round_robin,
            "least_loaded": self._route_least_loaded,
            "carbon_optimal": self._route_carbon_optimal,
            "cost_optimal": self._route_cost_optimal,
            "latency_optimal": self._route_latency_optimal,
            "adaptive": self._route_adaptive
        }
        
        self.current_strategy = "adaptive"
        self._round_robin_counter = 0
        
        logger.info("Load balancer initialized")
    
    def register_region(self, region: GeographicRegion) -> bool:
        """Register a geographic region.
        
        Args:
            region: Geographic region configuration
            
        Returns:
            True if registration successful
        """
        try:
            self.regions[region.region_id] = region
            logger.info(f"Region registered: {region.region_id} ({region.region_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to register region {region.region_id}: {e}")
            return False
    
    def route_request(self, 
                     request_metadata: Dict[str, Any],
                     strategy: str = None) -> Optional[str]:
        """Route request to optimal region.
        
        Args:
            request_metadata: Request metadata for routing decision
            strategy: Load balancing strategy to use
            
        Returns:
            Region ID for routing or None if no suitable region
        """
        if not self.regions:
            return None
        
        strategy = strategy or self.current_strategy
        
        if strategy not in self.strategies:
            logger.warning(f"Unknown strategy {strategy}, using adaptive")
            strategy = "adaptive"
        
        try:
            return self.strategies[strategy](request_metadata)
        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            return list(self.regions.keys())[0]  # Fallback to first region
    
    def _route_round_robin(self, request_metadata: Dict[str, Any]) -> str:
        """Round-robin routing strategy."""
        available_regions = [
            region_id for region_id, region in self.regions.items()
            if region.auto_scaling_enabled and region.current_utilization < 0.9
        ]
        
        if not available_regions:
            available_regions = list(self.regions.keys())
        
        self._round_robin_counter = (self._round_robin_counter + 1) % len(available_regions)
        return available_regions[self._round_robin_counter]
    
    def _route_least_loaded(self, request_metadata: Dict[str, Any]) -> str:
        """Route to least loaded region."""
        available_regions = [
            (region_id, region) for region_id, region in self.regions.items()
            if region.auto_scaling_enabled
        ]
        
        if not available_regions:
            return list(self.regions.keys())[0]
        
        # Find region with lowest utilization
        best_region = min(available_regions, key=lambda x: x[1].current_utilization)
        return best_region[0]
    
    def _route_carbon_optimal(self, request_metadata: Dict[str, Any]) -> str:
        """Route to region with lowest carbon intensity."""
        available_regions = [
            (region_id, region) for region_id, region in self.regions.items()
            if region.auto_scaling_enabled and region.current_utilization < 0.8
        ]
        
        if not available_regions:
            return self._route_least_loaded(request_metadata)
        
        # Find region with lowest carbon intensity
        best_region = min(available_regions, key=lambda x: x[1].carbon_intensity_gco2_kwh)
        return best_region[0]
    
    def _route_cost_optimal(self, request_metadata: Dict[str, Any]) -> str:
        """Route to region with lowest operational cost."""
        available_regions = [
            (region_id, region) for region_id, region in self.regions.items()
            if region.auto_scaling_enabled and region.current_utilization < 0.8
        ]
        
        if not available_regions:
            return self._route_least_loaded(request_metadata)
        
        # Calculate effective cost (base cost * utilization factor)
        def effective_cost(region):
            base_cost = region.electricity_cost_per_kwh * region.compute_cost_multiplier
            utilization_penalty = 1.0 + (region.current_utilization * 0.5)
            return base_cost * utilization_penalty
        
        best_region = min(available_regions, key=lambda x: effective_cost(x[1]))
        return best_region[0]
    
    def _route_latency_optimal(self, request_metadata: Dict[str, Any]) -> str:
        """Route to region with lowest latency."""
        source_location = request_metadata.get("source_location", {})
        source_lat = source_location.get("latitude", 0.0)
        source_lon = source_location.get("longitude", 0.0)
        
        available_regions = [
            (region_id, region) for region_id, region in self.regions.items()
            if region.auto_scaling_enabled and region.current_utilization < 0.8
        ]
        
        if not available_regions:
            return self._route_least_loaded(request_metadata)
        
        # Calculate distance-based latency estimation
        def estimated_latency(region):
            if source_lat == 0.0 and source_lon == 0.0:
                return region.latency_to_primary_ms
            
            # Haversine distance calculation
            lat_dist = math.radians(region.latitude - source_lat)
            lon_dist = math.radians(region.longitude - source_lon)
            
            a = (math.sin(lat_dist / 2) ** 2 + 
                 math.cos(math.radians(source_lat)) * math.cos(math.radians(region.latitude)) *
                 math.sin(lon_dist / 2) ** 2)
            
            distance_km = 2 * math.asin(math.sqrt(a)) * 6371  # Earth radius
            estimated_latency_ms = region.latency_to_primary_ms + (distance_km / 200)  # Rough estimate
            
            return estimated_latency_ms
        
        best_region = min(available_regions, key=lambda x: estimated_latency(x[1]))
        return best_region[0]
    
    def _route_adaptive(self, request_metadata: Dict[str, Any]) -> str:
        """Adaptive routing based on multiple factors."""
        available_regions = [
            (region_id, region) for region_id, region in self.regions.items()
            if region.auto_scaling_enabled
        ]
        
        if not available_regions:
            return list(self.regions.keys())[0]
        
        # Multi-criteria scoring
        best_score = float('-inf')
        best_region = available_regions[0][0]
        
        for region_id, region in available_regions:
            score = 0.0
            
            # Load factor (lower is better)
            load_score = (1.0 - region.current_utilization) * 30
            
            # Carbon factor (lower is better)
            max_carbon = max(r.carbon_intensity_gco2_kwh for _, r in available_regions)
            carbon_score = (1.0 - region.carbon_intensity_gco2_kwh / max_carbon) * 25
            
            # Cost factor (lower is better)
            max_cost = max(r.compute_cost_multiplier for _, r in available_regions)
            cost_score = (1.0 - region.compute_cost_multiplier / max_cost) * 20
            
            # Reliability factor (higher is better)
            reliability_score = region.reliability_score * 15
            
            # Latency factor (lower is better)
            max_latency = max(r.latency_to_primary_ms for _, r in available_regions)
            latency_score = (1.0 - region.latency_to_primary_ms / max_latency) * 10
            
            total_score = load_score + carbon_score + cost_score + reliability_score + latency_score
            
            if total_score > best_score:
                best_score = total_score
                best_region = region_id
        
        return best_region
    
    def update_region_load(self, region_id: str, load_metrics: Dict[str, float]):
        """Update region load metrics.
        
        Args:
            region_id: Region identifier
            load_metrics: Current load metrics
        """
        if region_id in self.regions:
            region = self.regions[region_id]
            
            # Update current utilization
            region.current_utilization = load_metrics.get("cpu_utilization", 0.0)
            
            # Store historical load
            self.load_history[region_id].append({
                "timestamp": time.time(),
                **load_metrics
            })
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across regions."""
        distribution = {}
        
        total_load = sum(region.current_utilization for region in self.regions.values())
        
        for region_id, region in self.regions.items():
            distribution[region_id] = {
                "current_utilization": region.current_utilization,
                "load_percentage": (region.current_utilization / max(total_load, 1)) * 100,
                "carbon_intensity": region.carbon_intensity_gco2_kwh,
                "cost_multiplier": region.compute_cost_multiplier,
                "reliability": region.reliability_score,
                "auto_scaling_enabled": region.auto_scaling_enabled
            }
        
        return {
            "regions": distribution,
            "total_regions": len(self.regions),
            "active_regions": sum(1 for r in self.regions.values() if r.auto_scaling_enabled),
            "average_utilization": total_load / max(len(self.regions), 1),
            "load_balance_score": 1.0 - np.std([r.current_utilization for r in self.regions.values()]) if self.regions else 0.0
        }


class EnterpriseAutoScaler:
    """Enterprise auto-scaler for carbon tracking infrastructure."""
    
    def __init__(self,
                 scaling_policy: ScalingPolicy = ScalingPolicy.BALANCED,
                 prediction_horizon_minutes: int = 15,
                 scaling_cooldown_seconds: int = 300):
        """Initialize enterprise auto-scaler.
        
        Args:
            scaling_policy: Default scaling policy
            prediction_horizon_minutes: Prediction horizon for proactive scaling
            scaling_cooldown_seconds: Cooldown period between scaling actions
        """
        self.scaling_policy = scaling_policy
        self.prediction_horizon_minutes = prediction_horizon_minutes
        self.scaling_cooldown_seconds = scaling_cooldown_seconds
        
        # Core components
        self.load_predictor = TrainingLoadPredictor()
        self.load_balancer = LoadBalancer()
        
        # Scaling state
        self.current_metrics: Dict[str, ScalingMetrics] = {}
        self.scaling_decisions: Dict[str, ScalingDecision] = {}
        self.last_scaling_time: Dict[str, float] = {}
        
        # Thresholds for different policies
        self.scaling_thresholds = {
            ScalingPolicy.AGGRESSIVE: {
                "scale_up_cpu": 0.6,
                "scale_down_cpu": 0.3,
                "scale_up_memory": 0.7,
                "scale_down_memory": 0.4,
                "response_time_ms": 30,
                "confidence_threshold": 0.6
            },
            ScalingPolicy.BALANCED: {
                "scale_up_cpu": 0.7,
                "scale_down_cpu": 0.2,
                "scale_up_memory": 0.8,
                "scale_down_memory": 0.3,
                "response_time_ms": 60,
                "confidence_threshold": 0.7
            },
            ScalingPolicy.CONSERVATIVE: {
                "scale_up_cpu": 0.8,
                "scale_down_cpu": 0.1,
                "scale_up_memory": 0.85,
                "scale_down_memory": 0.2,
                "response_time_ms": 120,
                "confidence_threshold": 0.8
            },
            ScalingPolicy.PREDICTIVE: {
                "scale_up_cpu": 0.6,
                "scale_down_cpu": 0.25,
                "scale_up_memory": 0.7,
                "scale_down_memory": 0.35,
                "response_time_ms": 45,
                "confidence_threshold": 0.75
            },
            ScalingPolicy.ADAPTIVE: {
                "scale_up_cpu": 0.65,
                "scale_down_cpu": 0.25,
                "scale_up_memory": 0.75,
                "scale_down_memory": 0.35,
                "response_time_ms": 45,
                "confidence_threshold": 0.7
            }
        }
        
        # Cloud provider integrations
        self.cloud_integrations = {
            "kubernetes": K8S_AVAILABLE,
            "aws": AWS_AVAILABLE,
            "gcp": GCP_AVAILABLE,
            "azure": AZURE_AVAILABLE
        }
        
        # Statistics
        self.stats = {
            "scaling_actions_total": 0,
            "scaling_actions_successful": 0,
            "average_response_time_ms": 0.0,
            "cost_savings_percentage": 0.0,
            "carbon_reduction_percentage": 0.0,
            "prediction_accuracy": 0.0
        }
        
        logger.info(f"Enterprise auto-scaler initialized with {scaling_policy.value} policy")
    
    async def add_metrics(self, region_id: str, metrics: ScalingMetrics):
        """Add metrics for scaling analysis.
        
        Args:
            region_id: Region identifier
            metrics: Current scaling metrics
        """
        self.current_metrics[region_id] = metrics
        
        # Update load predictor
        load_data = {
            "cpu_utilization": metrics.cpu_utilization,
            "memory_utilization": metrics.memory_utilization,
            "requests_per_second": metrics.tracking_requests_per_second,
            "latency_ms": metrics.metrics_collection_latency_ms
        }
        
        self.load_predictor.add_load_measurement(metrics.timestamp, load_data)
        
        # Update load balancer
        self.load_balancer.update_region_load(region_id, load_data)
        
        # Analyze for scaling opportunities
        await self._analyze_scaling_opportunity(region_id, metrics)
    
    async def _analyze_scaling_opportunity(self, region_id: str, metrics: ScalingMetrics):
        """Analyze current metrics for scaling opportunities.
        
        Args:
            region_id: Region identifier
            metrics: Current metrics
        """
        try:
            # Check cooldown period
            last_scaling = self.last_scaling_time.get(region_id, 0)
            if time.time() - last_scaling < self.scaling_cooldown_seconds:
                return  # Still in cooldown
            
            thresholds = self.scaling_thresholds[self.scaling_policy]
            
            # Predictive analysis
            cpu_prediction, cpu_confidence = self.load_predictor.predict_load(
                self.prediction_horizon_minutes, "cpu_utilization"
            )
            
            memory_prediction, memory_confidence = self.load_predictor.predict_load(
                self.prediction_horizon_minutes, "memory_utilization"
            )
            
            # Scaling decisions based on current and predicted load
            decisions = []
            
            # CPU scaling analysis
            if metrics.cpu_utilization > thresholds["scale_up_cpu"]:
                decisions.append(await self._create_scaling_decision(
                    region_id, "cpu", ScalingDirection.SCALE_UP,
                    metrics.cpu_utilization, thresholds["scale_up_cpu"],
                    cpu_confidence
                ))
            elif (metrics.cpu_utilization < thresholds["scale_down_cpu"] and 
                  cpu_prediction < thresholds["scale_down_cpu"]):
                decisions.append(await self._create_scaling_decision(
                    region_id, "cpu", ScalingDirection.SCALE_DOWN,
                    metrics.cpu_utilization, thresholds["scale_down_cpu"],
                    cpu_confidence
                ))
            
            # Memory scaling analysis
            if metrics.memory_utilization > thresholds["scale_up_memory"]:
                decisions.append(await self._create_scaling_decision(
                    region_id, "memory", ScalingDirection.SCALE_UP,
                    metrics.memory_utilization, thresholds["scale_up_memory"],
                    memory_confidence
                ))
            elif (metrics.memory_utilization < thresholds["scale_down_memory"] and 
                  memory_prediction < thresholds["scale_down_memory"]):
                decisions.append(await self._create_scaling_decision(
                    region_id, "memory", ScalingDirection.SCALE_DOWN,
                    metrics.memory_utilization, thresholds["scale_down_memory"],
                    memory_confidence
                ))
            
            # Latency-based instance scaling
            if metrics.metrics_collection_latency_ms > 1000:  # 1 second
                decisions.append(await self._create_scaling_decision(
                    region_id, "instances", ScalingDirection.SCALE_OUT,
                    metrics.metrics_collection_latency_ms, 1000,
                    0.8
                ))
            
            # Execute approved decisions
            for decision in decisions:
                if decision.confidence_score >= thresholds["confidence_threshold"]:
                    await self._execute_scaling_decision(decision)
                
        except Exception as e:
            logger.error(f"Scaling analysis failed for {region_id}: {e}")
    
    async def _create_scaling_decision(self,
                                     region_id: str,
                                     resource_type: str,
                                     direction: ScalingDirection,
                                     current_value: float,
                                     threshold: float,
                                     confidence: float) -> ScalingDecision:
        """Create scaling decision with impact analysis.
        
        Args:
            region_id: Region identifier
            resource_type: Type of resource to scale
            direction: Scaling direction
            current_value: Current resource utilization
            threshold: Threshold that triggered scaling
            confidence: Confidence in the decision
            
        Returns:
            Scaling decision with impact estimates
        """
        decision_id = f"{region_id}_{resource_type}_{direction.value}_{int(time.time())}"
        
        # Calculate scaling factor
        if direction in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT]:
            scaling_factor = 1.5  # 50% increase
        else:
            scaling_factor = 0.8  # 20% decrease
        
        # Estimate impacts
        current_capacity = current_value
        target_capacity = current_capacity * scaling_factor
        
        # Cost impact estimation
        cost_multiplier = scaling_factor if direction in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT] else scaling_factor
        estimated_cost_impact = (cost_multiplier - 1.0) * 100  # Percentage change
        
        # Carbon impact estimation
        estimated_carbon_impact = estimated_cost_impact * 0.8  # Assume 80% correlation
        
        # Performance impact estimation
        if direction in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT]:
            estimated_performance_impact = -20  # 20% latency reduction
        else:
            estimated_performance_impact = 10   # 10% latency increase
        
        # Auto-approval logic
        auto_approved = (
            confidence > 0.8 and
            abs(estimated_cost_impact) < 50 and  # Less than 50% cost change
            direction in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT]  # Prefer scaling up
        )
        
        decision = ScalingDecision(
            decision_id=decision_id,
            timestamp=time.time(),
            region_id=region_id,
            direction=direction,
            resource_type=resource_type,
            current_capacity=current_capacity,
            target_capacity=target_capacity,
            scaling_factor=scaling_factor,
            trigger_metric=resource_type,
            trigger_value=current_value,
            threshold_breached=threshold,
            confidence_score=confidence,
            estimated_cost_impact=estimated_cost_impact,
            estimated_carbon_impact=estimated_carbon_impact,
            estimated_performance_impact=estimated_performance_impact,
            estimated_completion_time=300.0,  # 5 minutes
            auto_approved=auto_approved,
            requires_human_approval=not auto_approved
        )
        
        self.scaling_decisions[decision_id] = decision
        
        logger.info(f"Scaling decision created: {decision_id} - {direction.value} {resource_type} in {region_id}")
        
        return decision
    
    async def _execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision.
        
        Args:
            decision: Scaling decision to execute
            
        Returns:
            True if execution successful
        """
        if not decision.auto_approved and decision.requires_human_approval:
            logger.info(f"Scaling decision {decision.decision_id} requires human approval")
            return False
        
        try:
            decision.execution_status = "executing"
            decision.execution_start_time = time.time()
            
            # Execute based on resource type and cloud provider
            success = await self._execute_cloud_scaling(decision)
            
            if success:
                decision.execution_status = "completed"
                decision.execution_completion_time = time.time()
                self.last_scaling_time[decision.region_id] = time.time()
                
                # Update statistics
                self.stats["scaling_actions_total"] += 1
                self.stats["scaling_actions_successful"] += 1
                
                logger.info(f"Scaling decision {decision.decision_id} executed successfully")
                return True
            else:
                decision.execution_status = "failed"
                logger.error(f"Scaling decision {decision.decision_id} execution failed")
                return False
                
        except Exception as e:
            decision.execution_status = "failed"
            logger.error(f"Scaling execution failed for {decision.decision_id}: {e}")
            return False
    
    async def _execute_cloud_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling on cloud infrastructure.
        
        Args:
            decision: Scaling decision
            
        Returns:
            True if successful
        """
        # This is a placeholder for actual cloud scaling implementation
        # In practice, this would integrate with cloud APIs
        
        if decision.resource_type == "instances":
            return await self._scale_instances(decision)
        elif decision.resource_type in ["cpu", "memory"]:
            return await self._scale_resources(decision)
        else:
            logger.warning(f"Unknown resource type for scaling: {decision.resource_type}")
            return False
    
    async def _scale_instances(self, decision: ScalingDecision) -> bool:
        """Scale instances (horizontal scaling).
        
        Args:
            decision: Scaling decision
            
        Returns:
            True if successful
        """
        # Kubernetes scaling
        if self.cloud_integrations["kubernetes"]:
            try:
                # Placeholder for Kubernetes HPA scaling
                logger.info(f"Scaling Kubernetes deployment in {decision.region_id}")
                await asyncio.sleep(1)  # Simulate API call
                return True
            except Exception as e:
                logger.error(f"Kubernetes scaling failed: {e}")
        
        # AWS Auto Scaling
        if self.cloud_integrations["aws"]:
            try:
                # Placeholder for AWS Auto Scaling Group
                logger.info(f"Scaling AWS ASG in {decision.region_id}")
                await asyncio.sleep(1)  # Simulate API call
                return True
            except Exception as e:
                logger.error(f"AWS scaling failed: {e}")
        
        # Simulate successful scaling
        await asyncio.sleep(2)  # Simulate scaling time
        return True
    
    async def _scale_resources(self, decision: ScalingDecision) -> bool:
        """Scale resources (vertical scaling).
        
        Args:
            decision: Scaling decision
            
        Returns:
            True if successful
        """
        # Vertical scaling simulation
        logger.info(f"Vertical scaling {decision.resource_type} in {decision.region_id}")
        await asyncio.sleep(3)  # Simulate resource allocation time
        return True
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status.
        
        Returns:
            Scaling status information
        """
        # Recent decisions
        recent_decisions = [
            decision for decision in self.scaling_decisions.values()
            if time.time() - decision.timestamp < 3600  # Last hour
        ]
        
        # Decision statistics
        pending_decisions = [d for d in recent_decisions if d.execution_status == "pending"]
        executing_decisions = [d for d in recent_decisions if d.execution_status == "executing"]
        completed_decisions = [d for d in recent_decisions if d.execution_status == "completed"]
        failed_decisions = [d for d in recent_decisions if d.execution_status == "failed"]
        
        # Load distribution
        load_distribution = self.load_balancer.get_load_distribution()
        
        # Prediction accuracy
        prediction_accuracy = self.load_predictor.get_prediction_accuracy()
        
        return {
            "scaling_policy": self.scaling_policy.value,
            "prediction_horizon_minutes": self.prediction_horizon_minutes,
            "cooldown_seconds": self.scaling_cooldown_seconds,
            "decisions": {
                "total_recent": len(recent_decisions),
                "pending": len(pending_decisions),
                "executing": len(executing_decisions),
                "completed": len(completed_decisions),
                "failed": len(failed_decisions),
                "success_rate": len(completed_decisions) / max(len(recent_decisions), 1) * 100
            },
            "load_distribution": load_distribution,
            "prediction_accuracy": prediction_accuracy,
            "cloud_integrations": self.cloud_integrations,
            "statistics": self.stats,
            "regions_monitored": len(self.current_metrics),
            "active_regions": sum(1 for region in self.load_balancer.regions.values() 
                                if region.auto_scaling_enabled),
            "timestamp": time.time()
        }
    
    async def optimize_global_distribution(self) -> Dict[str, Any]:
        """Optimize global load distribution for carbon efficiency.
        
        Returns:
            Optimization results
        """
        if not self.load_balancer.regions:
            return {"status": "no_regions_configured"}
        
        optimization_start = time.time()
        
        # Analyze current distribution
        current_distribution = self.load_balancer.get_load_distribution()
        
        # Find optimal distribution
        total_load = sum(
            region["current_utilization"] for region in current_distribution["regions"].values()
        )
        
        # Carbon-optimal distribution
        carbon_weights = {}
        total_carbon_weight = 0
        
        for region_id, region_data in current_distribution["regions"].items():
            region = self.load_balancer.regions[region_id]
            if region.auto_scaling_enabled:
                # Lower carbon intensity = higher weight
                weight = 1.0 / max(region.carbon_intensity_gco2_kwh, 1)
                carbon_weights[region_id] = weight
                total_carbon_weight += weight
        
        # Calculate optimal load distribution
        optimal_distribution = {}
        migration_recommendations = []
        
        for region_id, weight in carbon_weights.items():
            optimal_load = (weight / total_carbon_weight) * total_load
            current_load = current_distribution["regions"][region_id]["current_utilization"]
            
            load_difference = optimal_load - current_load
            
            optimal_distribution[region_id] = {
                "current_load": current_load,
                "optimal_load": optimal_load,
                "load_difference": load_difference,
                "migration_needed": abs(load_difference) > 0.1  # 10% threshold
            }
            
            if abs(load_difference) > 0.1:
                migration_recommendations.append({
                    "region_id": region_id,
                    "action": "scale_up" if load_difference > 0 else "scale_down",
                    "magnitude": abs(load_difference),
                    "carbon_benefit": self._calculate_carbon_benefit(region_id, load_difference)
                })
        
        optimization_time = time.time() - optimization_start
        
        return {
            "current_distribution": current_distribution,
            "optimal_distribution": optimal_distribution,
            "migration_recommendations": migration_recommendations,
            "optimization_metrics": {
                "carbon_reduction_potential": sum(
                    rec["carbon_benefit"] for rec in migration_recommendations
                ),
                "regions_requiring_migration": len(migration_recommendations),
                "optimization_time_ms": optimization_time * 1000
            },
            "timestamp": time.time()
        }
    
    def _calculate_carbon_benefit(self, region_id: str, load_change: float) -> float:
        """Calculate carbon benefit of load migration.
        
        Args:
            region_id: Region identifier
            load_change: Change in load (positive = increase)
            
        Returns:
            Carbon benefit in kg CO2/hr
        """
        if region_id not in self.load_balancer.regions:
            return 0.0
        
        region = self.load_balancer.regions[region_id]
        
        # Estimate power consumption change
        power_change_kw = abs(load_change) * 10  # Rough estimate: 10kW per 10% load
        
        # Calculate carbon impact
        carbon_change_kg_hr = power_change_kw * (region.carbon_intensity_gco2_kwh / 1000)
        
        # Benefit is reduction in carbon (negative load change in high-carbon region)
        return -carbon_change_kg_hr if load_change < 0 else 0.0
    
    async def shutdown(self):
        """Shutdown auto-scaler and cleanup resources."""
        logger.info("Enterprise auto-scaler shutting down")


# Global auto-scaler instance
_enterprise_autoscaler: Optional[EnterpriseAutoScaler] = None


def get_enterprise_autoscaler(
    scaling_policy: ScalingPolicy = ScalingPolicy.BALANCED,
    **kwargs
) -> EnterpriseAutoScaler:
    """Get global enterprise auto-scaler instance."""
    global _enterprise_autoscaler
    
    if _enterprise_autoscaler is None:
        _enterprise_autoscaler = EnterpriseAutoScaler(
            scaling_policy=scaling_policy,
            **kwargs
        )
    
    return _enterprise_autoscaler


async def initialize_enterprise_autoscaling():
    """Initialize enterprise auto-scaling with optimal configuration."""
    autoscaler = get_enterprise_autoscaler(
        scaling_policy=ScalingPolicy.ADAPTIVE,
        prediction_horizon_minutes=15,
        scaling_cooldown_seconds=300
    )
    
    # Register example regions
    regions = [
        GeographicRegion(
            region_id="us-west-2",
            region_name="US West (Oregon)",
            cloud_provider="aws",
            carbon_intensity_gco2_kwh=250.0,
            electricity_cost_per_kwh=0.08,
            compute_cost_multiplier=1.0,
            latency_to_primary_ms=10.0,
            bandwidth_gbps=100.0,
            reliability_score=0.99,
            max_cpu_cores=10000,
            max_memory_gb=100000,
            max_storage_tb=1000,
            auto_scaling_enabled=True,
            latitude=45.5152,
            longitude=-122.6784
        ),
        GeographicRegion(
            region_id="eu-central-1",
            region_name="Europe (Frankfurt)",
            cloud_provider="aws",
            carbon_intensity_gco2_kwh=350.0,
            electricity_cost_per_kwh=0.15,
            compute_cost_multiplier=1.2,
            latency_to_primary_ms=150.0,
            bandwidth_gbps=80.0,
            reliability_score=0.98,
            max_cpu_cores=8000,
            max_memory_gb=80000,
            max_storage_tb=800,
            auto_scaling_enabled=True,
            latitude=50.1109,
            longitude=8.6821
        ),
        GeographicRegion(
            region_id="ap-southeast-1",
            region_name="Asia Pacific (Singapore)",
            cloud_provider="aws",
            carbon_intensity_gco2_kwh=420.0,
            electricity_cost_per_kwh=0.12,
            compute_cost_multiplier=1.1,
            latency_to_primary_ms=200.0,
            bandwidth_gbps=90.0,
            reliability_score=0.97,
            max_cpu_cores=6000,
            max_memory_gb=60000,
            max_storage_tb=600,
            auto_scaling_enabled=True,
            latitude=1.3521,
            longitude=103.8198
        )
    ]
    
    for region in regions:
        autoscaler.load_balancer.register_region(region)
    
    logger.info("Enterprise auto-scaling initialized with global regions")
    return autoscaler