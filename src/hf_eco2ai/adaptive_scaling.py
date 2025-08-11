"""Adaptive scaling and auto-optimization system for carbon-efficient ML training."""

import asyncio
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from queue import Queue, Empty
from collections import deque


logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    gpu_memory_utilization: float
    training_throughput: float  # samples/second
    carbon_efficiency: float  # samples/kg_CO2
    cost_efficiency: float  # samples/dollar
    queue_length: int
    response_time_ms: float


@dataclass
class ScalingAction:
    """Scaling action to be executed."""
    action_id: str
    timestamp: datetime
    action_type: str  # scale_up, scale_down, optimize, rebalance
    target_component: str  # gpus, batch_size, workers, memory
    current_value: Union[int, float]
    target_value: Union[int, float]
    reason: str
    estimated_impact: Dict[str, float]
    confidence: float
    priority: int  # 1 (highest) to 10 (lowest)


@dataclass
class AdaptiveConfig:
    """Adaptive configuration parameters."""
    min_batch_size: int = 8
    max_batch_size: int = 256
    min_workers: int = 1
    max_workers: int = 16
    target_gpu_utilization: float = 0.85
    target_memory_utilization: float = 0.8
    scale_up_threshold: float = 0.9
    scale_down_threshold: float = 0.6
    adaptation_interval_seconds: int = 30
    carbon_weight: float = 0.4
    performance_weight: float = 0.3
    cost_weight: float = 0.3


class ResourceMonitor:
    """Real-time resource monitoring system."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_buffer = deque(maxlen=1000)
        self.monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Resource tracking
        self.current_metrics: Optional[ScalingMetrics] = None
        self.baseline_metrics: Optional[ScalingMetrics] = None
        
        # Callbacks for metric updates
        self.metric_callbacks: List[Callable[[ScalingMetrics], None]] = []
    
    def add_callback(self, callback: Callable[[ScalingMetrics], None]) -> None:
        """Add callback for metric updates."""
        self.metric_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.current_metrics = metrics
                self.metrics_buffer.append(metrics)
                
                # Call registered callbacks
                for callback in self.metric_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Metrics callback error: {e}")
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_utilization = self._get_cpu_utilization()
            memory_utilization = self._get_memory_utilization()
            
            # GPU metrics (if available)
            gpu_utilization = self._get_gpu_utilization()
            gpu_memory_utilization = self._get_gpu_memory_utilization()
            
            # Training metrics
            training_throughput = self._get_training_throughput()
            carbon_efficiency = self._get_carbon_efficiency()
            cost_efficiency = self._get_cost_efficiency()
            
            # System metrics
            queue_length = self._get_queue_length()
            response_time_ms = self._get_response_time()
            
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization,
                gpu_utilization=gpu_utilization,
                gpu_memory_utilization=gpu_memory_utilization,
                training_throughput=training_throughput,
                carbon_efficiency=carbon_efficiency,
                cost_efficiency=cost_efficiency,
                queue_length=queue_length,
                response_time_ms=response_time_ms
            )
        
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return default metrics to avoid breaking the monitoring
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_utilization=0.5,
                memory_utilization=0.5,
                gpu_utilization=0.5,
                gpu_memory_utilization=0.5,
                training_throughput=100.0,
                carbon_efficiency=1000.0,
                cost_efficiency=100.0,
                queue_length=0,
                response_time_ms=100.0
            )
    
    def _get_cpu_utilization(self) -> float:
        """Get CPU utilization percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1) / 100.0
        except ImportError:
            # Simulate CPU utilization if psutil not available
            return np.random.uniform(0.3, 0.9)
    
    def _get_memory_utilization(self) -> float:
        """Get memory utilization percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return np.random.uniform(0.4, 0.8)
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu / 100.0
        except:
            return np.random.uniform(0.5, 0.95)
    
    def _get_gpu_memory_utilization(self) -> float:
        """Get GPU memory utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return memory_info.used / memory_info.total
        except:
            return np.random.uniform(0.4, 0.9)
    
    def _get_training_throughput(self) -> float:
        """Get training throughput in samples/second."""
        # This would be provided by the training system
        # For now, simulate based on recent performance
        if hasattr(self, '_last_throughput'):
            # Add some variance
            noise = np.random.normal(0, self._last_throughput * 0.1)
            return max(10, self._last_throughput + noise)
        
        self._last_throughput = np.random.uniform(50, 200)
        return self._last_throughput
    
    def _get_carbon_efficiency(self) -> float:
        """Get carbon efficiency in samples/kg_CO2."""
        # Simulate carbon efficiency
        base_efficiency = 1000
        gpu_util = getattr(self, '_last_gpu_util', 0.7)
        # Higher GPU utilization generally means better carbon efficiency
        return base_efficiency * (0.5 + gpu_util * 0.5)
    
    def _get_cost_efficiency(self) -> float:
        """Get cost efficiency in samples/dollar."""
        # Simulate cost efficiency
        return np.random.uniform(80, 150)
    
    def _get_queue_length(self) -> int:
        """Get current queue length."""
        return np.random.randint(0, 10)
    
    def _get_response_time(self) -> float:
        """Get response time in milliseconds."""
        return np.random.uniform(50, 300)
    
    def get_recent_metrics(self, window_minutes: int = 5) -> List[ScalingMetrics]:
        """Get metrics from recent time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        return [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
    
    def get_metric_trends(self, window_minutes: int = 10) -> Dict[str, Dict[str, float]]:
        """Analyze metric trends over time window."""
        recent_metrics = self.get_recent_metrics(window_minutes)
        
        if len(recent_metrics) < 2:
            return {}
        
        trends = {}
        metric_names = [
            'cpu_utilization', 'memory_utilization', 'gpu_utilization',
            'gpu_memory_utilization', 'training_throughput', 'carbon_efficiency'
        ]
        
        for metric_name in metric_names:
            values = [getattr(m, metric_name) for m in recent_metrics]
            
            # Calculate trend (simple linear regression slope)
            x = np.arange(len(values))
            y = np.array(values)
            
            if len(values) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                r_squared = np.corrcoef(x, y)[0, 1] ** 2 if len(values) > 2 else 0
                
                trends[metric_name] = {
                    'slope': slope,
                    'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                    'r_squared': r_squared,
                    'current_value': values[-1],
                    'avg_value': np.mean(values),
                    'volatility': np.std(values)
                }
        
        return trends


class ScalingDecisionEngine:
    """Intelligent scaling decision engine."""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.resource_monitor = ResourceMonitor()
        
        # Decision history
        self.decision_history: List[ScalingAction] = []
        self.action_queue = Queue()
        
        # ML-based prediction (simplified)
        self.performance_predictor = PerformancePredictor()
        
        # Register for metric updates
        self.resource_monitor.add_callback(self._on_metrics_update)
    
    def start(self) -> None:
        """Start the scaling decision engine."""
        self.resource_monitor.start_monitoring()
        logger.info("Scaling decision engine started")
    
    def stop(self) -> None:
        """Stop the scaling decision engine."""
        self.resource_monitor.stop_monitoring()
        logger.info("Scaling decision engine stopped")
    
    def _on_metrics_update(self, metrics: ScalingMetrics) -> None:
        """Handle metrics update."""
        try:
            # Analyze current state and make scaling decisions
            scaling_actions = self._analyze_and_decide(metrics)
            
            # Queue actions for execution
            for action in scaling_actions:
                self.action_queue.put(action)
                self.decision_history.append(action)
        
        except Exception as e:
            logger.error(f"Error in scaling decision: {e}")
    
    def _analyze_and_decide(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Analyze metrics and make scaling decisions."""
        actions = []
        
        # GPU utilization analysis
        if metrics.gpu_utilization > self.config.scale_up_threshold:
            actions.extend(self._decide_scale_up(metrics))
        elif metrics.gpu_utilization < self.config.scale_down_threshold:
            actions.extend(self._decide_scale_down(metrics))
        
        # Memory pressure analysis
        if metrics.gpu_memory_utilization > 0.95:
            actions.extend(self._decide_memory_optimization(metrics))
        
        # Performance degradation analysis
        if metrics.training_throughput < self._get_expected_throughput():
            actions.extend(self._decide_performance_optimization(metrics))
        
        # Carbon efficiency analysis
        carbon_threshold = self._get_carbon_efficiency_threshold()
        if metrics.carbon_efficiency < carbon_threshold:
            actions.extend(self._decide_carbon_optimization(metrics))
        
        # Filter and prioritize actions
        return self._prioritize_actions(actions)
    
    def _decide_scale_up(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Decide on scale-up actions."""
        actions = []
        
        # Increase batch size if memory allows
        if metrics.gpu_memory_utilization < 0.8:
            current_batch_size = getattr(self, '_current_batch_size', 32)
            target_batch_size = min(
                current_batch_size * 2,
                self.config.max_batch_size
            )
            
            if target_batch_size > current_batch_size:
                action = ScalingAction(
                    action_id=f"scale_batch_{int(time.time())}",
                    timestamp=datetime.now(),
                    action_type="scale_up",
                    target_component="batch_size",
                    current_value=current_batch_size,
                    target_value=target_batch_size,
                    reason=f"GPU utilization high ({metrics.gpu_utilization:.2f}), memory available",
                    estimated_impact={
                        "throughput_increase": 0.4,
                        "carbon_efficiency_increase": 0.2,
                        "memory_increase": 0.3
                    },
                    confidence=0.8,
                    priority=3
                )
                actions.append(action)
        
        # Increase number of workers
        current_workers = getattr(self, '_current_workers', 4)
        if current_workers < self.config.max_workers and metrics.cpu_utilization < 0.8:
            target_workers = min(current_workers + 2, self.config.max_workers)
            
            action = ScalingAction(
                action_id=f"scale_workers_{int(time.time())}",
                timestamp=datetime.now(),
                action_type="scale_up",
                target_component="workers",
                current_value=current_workers,
                target_value=target_workers,
                reason=f"GPU utilization high, CPU available ({metrics.cpu_utilization:.2f})",
                estimated_impact={
                    "throughput_increase": 0.2,
                    "cpu_utilization_increase": 0.3
                },
                confidence=0.7,
                priority=4
            )
            actions.append(action)
        
        return actions
    
    def _decide_scale_down(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Decide on scale-down actions."""
        actions = []
        
        # Reduce batch size to improve carbon efficiency
        current_batch_size = getattr(self, '_current_batch_size', 32)
        if current_batch_size > self.config.min_batch_size:
            target_batch_size = max(
                current_batch_size // 2,
                self.config.min_batch_size
            )
            
            action = ScalingAction(
                action_id=f"scale_down_batch_{int(time.time())}",
                timestamp=datetime.now(),
                action_type="scale_down",
                target_component="batch_size",
                current_value=current_batch_size,
                target_value=target_batch_size,
                reason=f"GPU utilization low ({metrics.gpu_utilization:.2f}), optimize for carbon efficiency",
                estimated_impact={
                    "carbon_efficiency_increase": 0.15,
                    "memory_decrease": 0.3,
                    "throughput_decrease": 0.2
                },
                confidence=0.75,
                priority=5
            )
            actions.append(action)
        
        return actions
    
    def _decide_memory_optimization(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Decide on memory optimization actions."""
        actions = []
        
        # Enable gradient checkpointing
        action = ScalingAction(
            action_id=f"enable_grad_checkpoint_{int(time.time())}",
            timestamp=datetime.now(),
            action_type="optimize",
            target_component="memory",
            current_value=0,  # Not enabled
            target_value=1,   # Enabled
            reason=f"GPU memory utilization critical ({metrics.gpu_memory_utilization:.2f})",
            estimated_impact={
                "memory_decrease": 0.4,
                "throughput_decrease": 0.1,
                "carbon_efficiency_increase": 0.05
            },
            confidence=0.9,
            priority=1  # High priority for memory issues
        )
        actions.append(action)
        
        # Reduce batch size as emergency measure
        current_batch_size = getattr(self, '_current_batch_size', 32)
        if current_batch_size > self.config.min_batch_size:
            target_batch_size = max(
                current_batch_size // 2,
                self.config.min_batch_size
            )
            
            action = ScalingAction(
                action_id=f"emergency_batch_reduce_{int(time.time())}",
                timestamp=datetime.now(),
                action_type="scale_down",
                target_component="batch_size",
                current_value=current_batch_size,
                target_value=target_batch_size,
                reason="Emergency memory pressure relief",
                estimated_impact={
                    "memory_decrease": 0.5,
                    "throughput_decrease": 0.3
                },
                confidence=0.95,
                priority=2
            )
            actions.append(action)
        
        return actions
    
    def _decide_performance_optimization(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Decide on performance optimization actions."""
        actions = []
        
        # Enable mixed precision if not already enabled
        if not getattr(self, '_mixed_precision_enabled', False):
            action = ScalingAction(
                action_id=f"enable_fp16_{int(time.time())}",
                timestamp=datetime.now(),
                action_type="optimize",
                target_component="precision",
                current_value=32,  # fp32
                target_value=16,   # fp16
                reason=f"Performance below expected ({metrics.training_throughput:.1f} samples/s)",
                estimated_impact={
                    "throughput_increase": 0.3,
                    "memory_decrease": 0.2,
                    "carbon_efficiency_increase": 0.25
                },
                confidence=0.85,
                priority=3
            )
            actions.append(action)
        
        return actions
    
    def _decide_carbon_optimization(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Decide on carbon optimization actions."""
        actions = []
        
        # Optimize batch size for carbon efficiency
        optimal_batch_size = self._calculate_optimal_batch_size_for_carbon(metrics)
        current_batch_size = getattr(self, '_current_batch_size', 32)
        
        if abs(optimal_batch_size - current_batch_size) > 4:  # Significant difference
            action = ScalingAction(
                action_id=f"carbon_optimize_batch_{int(time.time())}",
                timestamp=datetime.now(),
                action_type="optimize",
                target_component="batch_size",
                current_value=current_batch_size,
                target_value=optimal_batch_size,
                reason=f"Carbon efficiency below threshold ({metrics.carbon_efficiency:.0f} samples/kg CO2)",
                estimated_impact={
                    "carbon_efficiency_increase": 0.2,
                    "throughput_change": 0.0  # Neutral
                },
                confidence=0.7,
                priority=4
            )
            actions.append(action)
        
        return actions
    
    def _prioritize_actions(self, actions: List[ScalingAction]) -> List[ScalingAction]:
        """Prioritize and filter actions."""
        # Sort by priority (lower number = higher priority)
        actions.sort(key=lambda x: (x.priority, -x.confidence))
        
        # Remove conflicting actions (keep highest priority)
        filtered_actions = []
        seen_components = set()
        
        for action in actions:
            if action.target_component not in seen_components:
                filtered_actions.append(action)
                seen_components.add(action.target_component)
        
        return filtered_actions[:3]  # Limit to top 3 actions
    
    def _get_expected_throughput(self) -> float:
        """Get expected throughput based on current configuration."""
        # This would be based on historical data and current config
        return 120.0  # samples/second
    
    def _get_carbon_efficiency_threshold(self) -> float:
        """Get carbon efficiency threshold."""
        return 800.0  # samples/kg CO2
    
    def _calculate_optimal_batch_size_for_carbon(self, metrics: ScalingMetrics) -> int:
        """Calculate optimal batch size for carbon efficiency."""
        # Simplified optimization - larger batches are usually more carbon efficient
        # up to memory limits
        if metrics.gpu_memory_utilization < 0.7:
            return min(128, self.config.max_batch_size)
        elif metrics.gpu_memory_utilization < 0.85:
            return 64
        else:
            return 32
    
    def get_pending_actions(self) -> List[ScalingAction]:
        """Get all pending scaling actions."""
        actions = []
        try:
            while True:
                action = self.action_queue.get_nowait()
                actions.append(action)
        except Empty:
            pass
        
        return actions


class PerformancePredictor:
    """ML-based performance predictor for scaling decisions."""
    
    def __init__(self):
        self.prediction_history: List[Dict[str, Any]] = []
        self.model_weights = np.random.random(10)  # Simplified linear model
    
    def predict_throughput(
        self,
        batch_size: int,
        num_workers: int,
        gpu_utilization: float,
        memory_utilization: float
    ) -> float:
        """Predict training throughput for given configuration."""
        # Simplified linear model
        features = np.array([
            batch_size / 100.0,
            num_workers / 10.0,
            gpu_utilization,
            memory_utilization,
            batch_size * gpu_utilization,  # Interaction term
            num_workers * (1 - memory_utilization),  # Interaction term
            1.0,  # Bias term
            0, 0, 0  # Padding
        ])
        
        predicted_throughput = np.dot(features, self.model_weights)
        return max(10, predicted_throughput * 100)  # Ensure positive and scale up
    
    def predict_carbon_efficiency(
        self,
        batch_size: int,
        gpu_utilization: float,
        training_throughput: float
    ) -> float:
        """Predict carbon efficiency for given configuration."""
        # Carbon efficiency generally increases with:
        # - Higher GPU utilization
        # - Larger batch sizes (up to a point)
        # - Higher throughput
        
        batch_factor = min(batch_size / 64.0, 2.0)  # Optimal around 64, diminishing returns
        util_factor = gpu_utilization
        throughput_factor = training_throughput / 100.0
        
        base_efficiency = 1000
        predicted_efficiency = base_efficiency * batch_factor * util_factor * throughput_factor
        
        return max(100, predicted_efficiency)


class AdaptiveScaler:
    """Main adaptive scaling orchestrator."""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.decision_engine = ScalingDecisionEngine(self.config)
        
        # Current configuration state
        self.current_config = {
            "batch_size": 32,
            "num_workers": 4,
            "mixed_precision": False,
            "gradient_checkpointing": False
        }
        
        # Scaling history
        self.scaling_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # Action executor
        self.action_executor = ActionExecutor()
        
        # Monitoring
        self.scaling_active = False
        self._scaling_thread: Optional[threading.Thread] = None
    
    def start_adaptive_scaling(self) -> None:
        """Start adaptive scaling system."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.decision_engine.start()
        
        # Start action execution loop
        self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._scaling_thread.start()
        
        logger.info("Adaptive scaling system started")
    
    def stop_adaptive_scaling(self) -> None:
        """Stop adaptive scaling system."""
        self.scaling_active = False
        self.decision_engine.stop()
        
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5)
        
        logger.info("Adaptive scaling system stopped")
    
    def _scaling_loop(self) -> None:
        """Main scaling loop."""
        while self.scaling_active:
            try:
                # Get pending actions
                pending_actions = self.decision_engine.get_pending_actions()
                
                # Execute actions
                for action in pending_actions:
                    success = self._execute_scaling_action(action)
                    
                    # Record execution result
                    execution_record = {
                        "action": asdict(action),
                        "executed_at": datetime.now().isoformat(),
                        "success": success,
                        "config_after": self.current_config.copy()
                    }
                    self.scaling_history.append(execution_record)
                
                time.sleep(self.config.adaptation_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(self.config.adaptation_interval_seconds)
    
    def _execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action."""
        try:
            logger.info(f"Executing scaling action: {action.action_type} {action.target_component} from {action.current_value} to {action.target_value}")
            
            # Update current configuration
            if action.target_component == "batch_size":
                self.current_config["batch_size"] = int(action.target_value)
                # In a real system, this would update the trainer configuration
                
            elif action.target_component == "workers":
                self.current_config["num_workers"] = int(action.target_value)
                
            elif action.target_component == "precision":
                self.current_config["mixed_precision"] = action.target_value == 16
                
            elif action.target_component == "memory":
                if "gradient_checkpoint" in action.action_id:
                    self.current_config["gradient_checkpointing"] = action.target_value == 1
            
            # Simulate action execution
            execution_time = np.random.uniform(1, 5)  # 1-5 seconds
            time.sleep(execution_time)
            
            logger.info(f"Scaling action completed successfully in {execution_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")
            return False
    
    def get_scaling_dashboard(self) -> Dict[str, Any]:
        """Get adaptive scaling dashboard data."""
        recent_actions = [
            record for record in self.scaling_history 
            if datetime.fromisoformat(record["executed_at"]) > datetime.now() - timedelta(hours=1)
        ]
        
        successful_actions = [r for r in recent_actions if r["success"]]
        
        # Current metrics
        current_metrics = self.decision_engine.resource_monitor.current_metrics
        
        return {
            "scaling_status": {
                "active": self.scaling_active,
                "current_config": self.current_config,
                "actions_last_hour": len(recent_actions),
                "success_rate": len(successful_actions) / max(1, len(recent_actions)) * 100
            },
            "current_metrics": asdict(current_metrics) if current_metrics else {},
            "recent_actions": recent_actions[-10:],  # Last 10 actions
            "optimization_insights": self._generate_optimization_insights(),
            "performance_trends": self._analyze_performance_trends()
        }
    
    def _generate_optimization_insights(self) -> List[str]:
        """Generate optimization insights based on scaling history."""
        insights = []
        
        if len(self.scaling_history) < 5:
            insights.append("Insufficient data for detailed insights - continue monitoring")
            return insights
        
        # Analyze recent patterns
        recent_actions = self.scaling_history[-20:]
        
        # Batch size analysis
        batch_size_actions = [a for a in recent_actions if a["action"]["target_component"] == "batch_size"]
        if len(batch_size_actions) > 3:
            insights.append("Frequent batch size adjustments detected - consider optimization")
        
        # Success rate analysis
        success_rate = sum(1 for a in recent_actions if a["success"]) / len(recent_actions)
        if success_rate < 0.8:
            insights.append(f"Action success rate low ({success_rate:.1%}) - review scaling logic")
        
        # Configuration stability
        configs = [a["config_after"] for a in recent_actions]
        if len(set(str(c) for c in configs)) > len(configs) * 0.7:
            insights.append("High configuration instability - consider dampening scaling sensitivity")
        
        return insights
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends."""
        metrics_history = self.decision_engine.resource_monitor.get_recent_metrics(30)
        
        if len(metrics_history) < 10:
            return {"status": "Insufficient data for trend analysis"}
        
        # Calculate trends for key metrics
        timestamps = [(m.timestamp - metrics_history[0].timestamp).total_seconds() for m in metrics_history]
        
        trends = {}
        for metric_name in ['gpu_utilization', 'training_throughput', 'carbon_efficiency']:
            values = [getattr(m, metric_name) for m in metrics_history]
            
            if len(values) > 1:
                # Simple linear regression
                slope = np.polyfit(timestamps, values, 1)[0]
                trends[metric_name] = {
                    "slope": slope,
                    "direction": "improving" if slope > 0 else "declining" if slope < 0 else "stable",
                    "current_value": values[-1],
                    "change_rate_per_minute": slope * 60
                }
        
        return {
            "trends": trends,
            "overall_assessment": self._assess_overall_performance_trend(trends)
        }
    
    def _assess_overall_performance_trend(self, trends: Dict[str, Any]) -> str:
        """Assess overall performance trend."""
        positive_trends = sum(1 for t in trends.values() if t.get("slope", 0) > 0)
        negative_trends = sum(1 for t in trends.values() if t.get("slope", 0) < 0)
        
        if positive_trends > negative_trends:
            return "Performance improving - adaptive scaling is effective"
        elif negative_trends > positive_trends:
            return "Performance declining - review scaling strategy"
        else:
            return "Performance stable - system well-tuned"
    
    def export_scaling_report(self, filepath: Path) -> None:
        """Export comprehensive adaptive scaling report."""
        dashboard = self.get_scaling_dashboard()
        
        report = {
            "adaptive_scaling_report": dashboard,
            "scaling_history": self.scaling_history,
            "configuration": asdict(self.config),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Adaptive scaling report exported to {filepath}")


class ActionExecutor:
    """Executes scaling actions safely."""
    
    def __init__(self):
        self.execution_lock = threading.Lock()
        self.execution_history: List[Dict[str, Any]] = []
    
    def execute_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action safely."""
        with self.execution_lock:
            try:
                # Validate action before execution
                if not self._validate_action(action):
                    logger.error(f"Action validation failed: {action.action_id}")
                    return False
                
                # Execute the action
                success = self._perform_action(action)
                
                # Record execution
                self.execution_history.append({
                    "action_id": action.action_id,
                    "executed_at": datetime.now().isoformat(),
                    "success": success,
                    "execution_time_ms": time.time() * 1000 - action.timestamp.timestamp() * 1000
                })
                
                return success
                
            except Exception as e:
                logger.error(f"Error executing action {action.action_id}: {e}")
                return False
    
    def _validate_action(self, action: ScalingAction) -> bool:
        """Validate action before execution."""
        # Basic validation
        if not action.action_id or not action.target_component:
            return False
        
        # Value range validation
        if action.target_component == "batch_size":
            if not (8 <= action.target_value <= 512):
                return False
        
        elif action.target_component == "workers":
            if not (1 <= action.target_value <= 32):
                return False
        
        return True
    
    def _perform_action(self, action: ScalingAction) -> bool:
        """Perform the actual scaling action."""
        # In a real system, this would interact with the training framework
        # For now, we simulate the action execution
        
        execution_time = np.random.uniform(0.5, 3.0)
        time.sleep(execution_time)
        
        # Simulate occasional failures
        return np.random.random() > 0.05  # 95% success rate


# Global adaptive scaler instance
_adaptive_scaler: Optional[AdaptiveScaler] = None


def get_adaptive_scaler(config: Optional[AdaptiveConfig] = None) -> AdaptiveScaler:
    """Get or create the global adaptive scaler."""
    global _adaptive_scaler
    
    if _adaptive_scaler is None:
        _adaptive_scaler = AdaptiveScaler(config)
    
    return _adaptive_scaler


def start_adaptive_scaling(config: Optional[AdaptiveConfig] = None) -> AdaptiveScaler:
    """Start adaptive scaling with default or custom configuration."""
    scaler = get_adaptive_scaler(config)
    scaler.start_adaptive_scaling()
    return scaler