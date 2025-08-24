"""Enhanced enterprise health monitoring and system diagnostics for carbon tracking."""

import time
import logging
import threading
import psutil
import os
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue
import gc
import socket
import subprocess
import asyncio
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import pickle
import uuid

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Enhanced health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"
    DEGRADED = "degraded"
    RECOVERING = "recovering"


class AlertSeverity(Enum):
    """Alert severity levels for escalation."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of health metrics."""
    SYSTEM = "system"
    APPLICATION = "application"
    NETWORK = "network"
    STORAGE = "storage"
    CUSTOM = "custom"


@dataclass
class EnhancedHealthMetric:
    """Enhanced individual health metric with predictive capabilities."""
    name: str
    value: float
    status: HealthStatus
    metric_type: MetricType = MetricType.SYSTEM
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: float = None
    unit: str = ""
    description: str = ""
    source: str = "system"
    tags: Dict[str, str] = None
    predicted_value: Optional[float] = None
    trend: Optional[str] = None  # "increasing", "decreasing", "stable"
    confidence_score: float = 1.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.tags is None:
            self.tags = {}


@dataclass
class AlertPolicy:
    """Alert escalation policy configuration."""
    name: str
    severity: AlertSeverity
    conditions: Dict[str, Any]
    escalation_delay_minutes: int = 15
    max_escalations: int = 3
    notification_channels: List[str] = None
    auto_resolve: bool = False
    cooldown_minutes: int = 60
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ["log"]


@dataclass
class HealthAlert:
    """Health alert with comprehensive tracking."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_name: str
    message: str
    value: float
    threshold: float
    escalation_level: int = 0
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    last_escalation: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SystemHealth:
    """Enhanced system health summary."""
    overall_status: HealthStatus
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    process_count: int
    uptime_seconds: float
    timestamp: float
    metrics: List[EnhancedHealthMetric]
    alert_count: int = 0
    degraded_services: List[str] = None
    predictive_warnings: List[str] = None
    
    def __post_init__(self):
        if self.degraded_services is None:
            self.degraded_services = []
        if self.predictive_warnings is None:
            self.predictive_warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enhanced data."""
        result = asdict(self)
        # Convert enum values to strings
        result["overall_status"] = self.overall_status.value
        for i, metric in enumerate(result["metrics"]):
            result["metrics"][i]["status"] = self.metrics[i].status.value
            result["metrics"][i]["metric_type"] = self.metrics[i].metric_type.value
        return result


class NetworkHealthChecker:
    """Advanced network connectivity health checker."""
    
    def __init__(self):
        self.test_endpoints = [
            ("google.com", 80),
            ("8.8.8.8", 53),
            ("1.1.1.1", 53),
            ("github.com", 443),
            ("api.openai.com", 443)
        ]
        self.latency_history = deque(maxlen=50)
    
    def check_connectivity(self) -> EnhancedHealthMetric:
        """Comprehensive network connectivity check."""
        successful_connections = 0
        total_latency = 0
        failed_endpoints = []
        
        for host, port in self.test_endpoints:
            try:
                start_time = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                
                latency = (time.time() - start_time) * 1000
                
                if result == 0:
                    successful_connections += 1
                    total_latency += latency
                else:
                    failed_endpoints.append(f"{host}:{port}")
                    
            except Exception as e:
                failed_endpoints.append(f"{host}:{port} ({str(e)[:20]})")
        
        connectivity_percent = (successful_connections / len(self.test_endpoints)) * 100
        avg_latency = total_latency / max(successful_connections, 1)
        
        # Store latency history for trend analysis
        if successful_connections > 0:
            self.latency_history.append(avg_latency)
        
        # Determine status based on connectivity and latency
        status = HealthStatus.HEALTHY
        if connectivity_percent < 100:
            status = HealthStatus.WARNING
        if connectivity_percent < 70:
            status = HealthStatus.CRITICAL
        if connectivity_percent == 0:
            status = HealthStatus.FAILURE
        
        # Check for high latency even if connected
        if avg_latency > 1000 and status == HealthStatus.HEALTHY:
            status = HealthStatus.WARNING
        
        # Calculate trend
        trend = "stable"
        if len(self.latency_history) >= 5:
            recent_avg = np.mean(list(self.latency_history)[-5:])
            older_avg = np.mean(list(self.latency_history)[:-5])
            if recent_avg > older_avg * 1.2:
                trend = "degrading"
            elif recent_avg < older_avg * 0.8:
                trend = "improving"
        
        return EnhancedHealthMetric(
            name="network_connectivity",
            value=connectivity_percent,
            status=status,
            metric_type=MetricType.NETWORK,
            threshold_warning=80,
            threshold_critical=50,
            unit="%",
            description=f"Network connectivity ({successful_connections}/{len(self.test_endpoints)} endpoints)",
            trend=trend,
            tags={
                "avg_latency_ms": str(round(avg_latency, 2)),
                "failed_endpoints": ",".join(failed_endpoints[:3]) if failed_endpoints else "none",
                "latency_trend": trend
            }
        )
    
    def check_dns_resolution(self) -> EnhancedHealthMetric:
        """Check DNS resolution performance."""
        test_domains = ["google.com", "github.com", "stackoverflow.com"]
        successful_resolutions = 0
        total_time = 0
        
        for domain in test_domains:
            try:
                start_time = time.time()
                socket.gethostbyname(domain)
                total_time += (time.time() - start_time) * 1000
                successful_resolutions += 1
            except Exception:
                pass
        
        success_rate = (successful_resolutions / len(test_domains)) * 100
        avg_resolution_time = total_time / max(successful_resolutions, 1)
        
        status = HealthStatus.HEALTHY
        if success_rate < 100:
            status = HealthStatus.WARNING
        if success_rate < 50:
            status = HealthStatus.CRITICAL
        if avg_resolution_time > 500:
            status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
        
        return EnhancedHealthMetric(
            name="dns_resolution",
            value=success_rate,
            status=status,
            metric_type=MetricType.NETWORK,
            threshold_warning=80,
            threshold_critical=50,
            unit="%",
            description="DNS resolution success rate",
            tags={
                "avg_resolution_time_ms": str(round(avg_resolution_time, 2)),
                "successful_resolutions": str(successful_resolutions)
            }
        )


class MemoryLeakDetector:
    """Advanced memory leak detection and analysis."""
    
    def __init__(self, window_size: int = 30):
        self.memory_history = deque(maxlen=window_size)
        self.window_size = window_size
        self.leak_threshold = 0.5  # 0.5% per check
        self.critical_leak_threshold = 1.0  # 1% per check
    
    def analyze_memory_trend(self) -> Tuple[bool, float, str, float]:
        """Advanced memory usage trend analysis."""
        current_memory = psutil.virtual_memory().percent
        current_time = time.time()
        
        self.memory_history.append({
            "memory_percent": current_memory,
            "timestamp": current_time
        })
        
        if len(self.memory_history) < 10:
            return False, current_memory, "insufficient_data", 0.0
        
        # Extract data for analysis
        timestamps = [item["timestamp"] for item in self.memory_history]
        memory_values = [item["memory_percent"] for item in self.memory_history]
        
        # Normalize timestamps to start from 0
        timestamps = np.array(timestamps) - timestamps[0]
        memory_values = np.array(memory_values)
        
        # Linear regression for trend analysis
        n = len(timestamps)
        sum_x = np.sum(timestamps)
        sum_y = np.sum(memory_values)
        sum_xy = np.sum(timestamps * memory_values)
        sum_x2 = np.sum(timestamps * timestamps)
        
        # Calculate slope (memory change per second)
        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Convert to percentage per check interval (assuming 30 second intervals)
        slope_per_check = slope * 30
        
        # Calculate R-squared for trend confidence
        mean_y = np.mean(memory_values)
        ss_tot = np.sum((memory_values - mean_y) ** 2)
        predicted_y = slope * timestamps + (sum_y - slope * sum_x) / n
        ss_res = np.sum((memory_values - predicted_y) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0, min(1, r_squared))
        
        # Determine leak status and trend
        leak_detected = False
        trend_description = "stable"
        
        if slope_per_check > self.critical_leak_threshold:
            leak_detected = True
            trend_description = "critical_leak"
        elif slope_per_check > self.leak_threshold:
            leak_detected = True
            trend_description = "potential_leak"
        elif slope_per_check > 0.1:
            trend_description = "slowly_increasing"
        elif slope_per_check < -0.1:
            trend_description = "decreasing"
        
        return leak_detected, slope_per_check, trend_description, confidence
    
    def get_memory_metrics(self) -> List[EnhancedHealthMetric]:
        """Comprehensive memory health metrics."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        leak_detected, slope, trend, confidence = self.analyze_memory_trend()
        
        metrics = []
        
        # Physical memory usage
        memory_status = HealthStatus.HEALTHY
        if memory.percent > 95:
            memory_status = HealthStatus.CRITICAL
        elif memory.percent > 85:
            memory_status = HealthStatus.WARNING
        
        metrics.append(EnhancedHealthMetric(
            name="memory_usage",
            value=memory.percent,
            status=memory_status,
            metric_type=MetricType.SYSTEM,
            threshold_warning=85,
            threshold_critical=95,
            unit="%",
            description="Physical memory utilization",
            trend=trend,
            confidence_score=confidence,
            tags={
                "available_gb": str(round(memory.available / (1024**3), 2)),
                "used_gb": str(round(memory.used / (1024**3), 2)),
                "total_gb": str(round(memory.total / (1024**3), 2)),
                "buffers_gb": str(round(getattr(memory, 'buffers', 0) / (1024**3), 2)),
                "cached_gb": str(round(getattr(memory, 'cached', 0) / (1024**3), 2))
            }
        ))
        
        # Memory leak detection
        leak_status = HealthStatus.HEALTHY
        if leak_detected and "critical" in trend:
            leak_status = HealthStatus.CRITICAL
        elif leak_detected:
            leak_status = HealthStatus.WARNING
        
        metrics.append(EnhancedHealthMetric(
            name="memory_leak_indicator",
            value=abs(slope),
            status=leak_status,
            metric_type=MetricType.APPLICATION,
            threshold_warning=self.leak_threshold,
            threshold_critical=self.critical_leak_threshold,
            unit="%/check",
            description="Memory usage trend analysis for leak detection",
            trend=trend,
            confidence_score=confidence,
            tags={
                "leak_detected": str(leak_detected),
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            }
        ))
        
        # Swap usage
        swap_status = HealthStatus.HEALTHY
        if swap.percent > 80:
            swap_status = HealthStatus.CRITICAL
        elif swap.percent > 50:
            swap_status = HealthStatus.WARNING
        
        metrics.append(EnhancedHealthMetric(
            name="swap_usage",
            value=swap.percent,
            status=swap_status,
            metric_type=MetricType.SYSTEM,
            threshold_warning=50,
            threshold_critical=80,
            unit="%",
            description="Virtual memory (swap) utilization",
            tags={
                "used_gb": str(round(swap.used / (1024**3), 2)),
                "total_gb": str(round(swap.total / (1024**3), 2))
            }
        ))
        
        return metrics


class DiskSpaceMonitor:
    """Enhanced disk space monitoring with predictive analysis."""
    
    def __init__(self):
        self.monitored_paths = ["/", "/tmp", "./reports", "./checkpoints", "./logs"]
        self.usage_history = {}
    
    def get_disk_metrics(self) -> List[EnhancedHealthMetric]:
        """Comprehensive disk space metrics."""
        metrics = []
        current_time = time.time()
        
        for path in self.monitored_paths:
            try:
                if not Path(path).exists():
                    continue
                
                disk = psutil.disk_usage(path)
                usage_percent = (disk.used / disk.total) * 100
                
                # Track usage history for trend analysis
                if path not in self.usage_history:
                    self.usage_history[path] = deque(maxlen=20)
                
                self.usage_history[path].append({
                    "usage_percent": usage_percent,
                    "timestamp": current_time
                })
                
                # Calculate trend
                trend = self._calculate_disk_trend(path)
                
                # Predict when disk might fill up
                predicted_full_days = self._predict_disk_full_time(path)
                
                # Determine status
                status = HealthStatus.HEALTHY
                if usage_percent > 95:
                    status = HealthStatus.CRITICAL
                elif usage_percent > 90:
                    status = HealthStatus.WARNING
                elif predicted_full_days is not None and predicted_full_days < 7:
                    status = HealthStatus.WARNING
                
                # Generate metric name safe for all systems
                safe_path = path.replace('/', '_').replace('.', 'dot').replace('\\', '_')
                if safe_path.startswith('_'):
                    safe_path = 'root' + safe_path
                
                metrics.append(EnhancedHealthMetric(
                    name=f"disk_usage_{safe_path}",
                    value=usage_percent,
                    status=status,
                    metric_type=MetricType.STORAGE,
                    threshold_warning=90,
                    threshold_critical=95,
                    unit="%",
                    description=f"Disk usage for {path}",
                    trend=trend,
                    tags={
                        "path": path,
                        "free_gb": str(round(disk.free / (1024**3), 2)),
                        "used_gb": str(round(disk.used / (1024**3), 2)),
                        "total_gb": str(round(disk.total / (1024**3), 2)),
                        "predicted_full_days": str(predicted_full_days) if predicted_full_days else "never"
                    }
                ))
                
                # Add IO metrics if available
                try:
                    disk_io = psutil.disk_io_counters(perdisk=False)
                    if disk_io:
                        # Calculate IO utilization (simplified)
                        io_percent = min(100, (disk_io.read_bytes + disk_io.write_bytes) / (1024**3) * 10)
                        
                        io_status = HealthStatus.HEALTHY
                        if io_percent > 80:
                            io_status = HealthStatus.WARNING
                        if io_percent > 95:
                            io_status = HealthStatus.CRITICAL
                        
                        metrics.append(EnhancedHealthMetric(
                            name=f"disk_io_{safe_path}",
                            value=io_percent,
                            status=io_status,
                            metric_type=MetricType.STORAGE,
                            threshold_warning=80,
                            threshold_critical=95,
                            unit="%",
                            description=f"Disk I/O utilization estimate for {path}",
                            tags={
                                "read_gb": str(round(disk_io.read_bytes / (1024**3), 2)),
                                "write_gb": str(round(disk_io.write_bytes / (1024**3), 2))
                            }
                        ))
                except Exception:
                    pass  # IO metrics not critical
                    
            except Exception as e:
                logger.error(f"Error checking disk usage for {path}: {e}")
        
        return metrics
    
    def _calculate_disk_trend(self, path: str) -> str:
        """Calculate disk usage trend for a path."""
        if path not in self.usage_history or len(self.usage_history[path]) < 3:
            return "stable"
        
        history = list(self.usage_history[path])
        recent_usage = [item["usage_percent"] for item in history[-3:]]
        older_usage = [item["usage_percent"] for item in history[:-3]] if len(history) > 3 else recent_usage
        
        if not older_usage:
            return "stable"
        
        recent_avg = np.mean(recent_usage)
        older_avg = np.mean(older_usage)
        
        if recent_avg > older_avg + 1:
            return "increasing"
        elif recent_avg < older_avg - 1:
            return "decreasing"
        else:
            return "stable"
    
    def _predict_disk_full_time(self, path: str) -> Optional[float]:
        """Predict when disk might become full (in days)."""
        if path not in self.usage_history or len(self.usage_history[path]) < 5:
            return None
        
        history = list(self.usage_history[path])
        if len(history) < 5:
            return None
        
        # Calculate growth rate
        timestamps = [item["timestamp"] for item in history]
        usage_values = [item["usage_percent"] for item in history]
        
        # Simple linear regression
        n = len(timestamps)
        if n < 2:
            return None
        
        sum_x = sum(timestamps)
        sum_y = sum(usage_values)
        sum_xy = sum(t * u for t, u in zip(timestamps, usage_values))
        sum_x2 = sum(t * t for t in timestamps)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return None
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope <= 0:
            return None  # Not growing
        
        # Calculate days until 100% (assuming current trend continues)
        current_usage = usage_values[-1]
        remaining_percent = 100 - current_usage
        
        if remaining_percent <= 0:
            return 0
        
        # Convert slope from percent per second to percent per day
        slope_per_day = slope * 86400  # 86400 seconds in a day
        
        if slope_per_day <= 0:
            return None
        
        days_until_full = remaining_percent / slope_per_day
        
        # Return only if prediction is reasonable (between 1 day and 1 year)
        if 1 <= days_until_full <= 365:
            return round(days_until_full, 1)
        
        return None


class PredictiveHealthAnalyzer:
    """Advanced predictive analysis for failure detection."""
    
    def __init__(self):
        self.metric_history: Dict[str, deque] = {}
        self.prediction_models = {}
        self.prediction_accuracy = {}
    
    def add_metric_data(self, metric: EnhancedHealthMetric):
        """Add metric data for predictive analysis."""
        if metric.name not in self.metric_history:
            self.metric_history[metric.name] = deque(maxlen=100)
        
        self.metric_history[metric.name].append({
            "timestamp": metric.timestamp,
            "value": metric.value,
            "status": metric.status.value,
            "trend": metric.trend
        })
    
    def predict_failure(self, metric_name: str, horizon_minutes: int = 30) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Advanced failure prediction with confidence scoring."""
        if metric_name not in self.metric_history:
            return False, 0.0, "no_data", {}
        
        history = list(self.metric_history[metric_name])
        if len(history) < 10:
            return False, 0.0, "insufficient_data", {}
        
        # Extract recent data for analysis
        recent_values = [item["value"] for item in history[-20:]]
        timestamps = [item["timestamp"] for item in history[-20:]]
        
        # Normalize timestamps
        base_time = timestamps[0]
        normalized_times = [(t - base_time) / 60 for t in timestamps]  # Convert to minutes
        
        # Multiple prediction approaches
        predictions = {}
        
        # 1. Linear trend prediction
        linear_pred = self._linear_trend_prediction(normalized_times, recent_values, horizon_minutes)
        predictions["linear"] = linear_pred
        
        # 2. Exponential smoothing prediction
        exp_pred = self._exponential_smoothing_prediction(recent_values, horizon_minutes)
        predictions["exponential"] = exp_pred
        
        # 3. Seasonal/cyclical analysis (if enough data)
        if len(history) >= 50:
            seasonal_pred = self._seasonal_prediction(history, horizon_minutes)
            predictions["seasonal"] = seasonal_pred
        
        # Combine predictions
        combined_prediction = self._combine_predictions(predictions, metric_name)
        
        # Determine failure likelihood
        failure_likely, confidence, reason = self._evaluate_failure_risk(
            combined_prediction, metric_name, history
        )
        
        metadata = {
            "predicted_value": combined_prediction.get("value", 0),
            "prediction_methods": list(predictions.keys()),
            "historical_accuracy": self.prediction_accuracy.get(metric_name, 0.0),
            "data_points_used": len(recent_values),
            "prediction_horizon_minutes": horizon_minutes
        }
        
        return failure_likely, confidence, reason, metadata
    
    def _linear_trend_prediction(self, times: List[float], values: List[float], horizon: int) -> Dict[str, Any]:
        """Linear trend-based prediction."""
        if len(times) < 2:
            return {"value": values[-1] if values else 0, "confidence": 0.0}
        
        # Linear regression
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(values)
        sum_xy = sum(t * v for t, v in zip(times, values))
        sum_x2 = sum(t * t for t in times)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return {"value": values[-1], "confidence": 0.0}
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict future value
        future_time = times[-1] + horizon
        predicted_value = slope * future_time + intercept
        
        # Calculate confidence based on R-squared
        predicted_values = [slope * t + intercept for t in times]
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(values, predicted_values))
        ss_tot = sum((v - sum_y / n) ** 2 for v in values)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0, min(1, r_squared))
        
        return {
            "value": predicted_value,
            "confidence": confidence,
            "slope": slope,
            "r_squared": r_squared
        }
    
    def _exponential_smoothing_prediction(self, values: List[float], horizon: int) -> Dict[str, Any]:
        """Exponential smoothing prediction."""
        if len(values) < 3:
            return {"value": values[-1] if values else 0, "confidence": 0.0}
        
        alpha = 0.3  # Smoothing parameter
        smoothed_values = [values[0]]
        
        # Calculate smoothed values
        for i in range(1, len(values)):
            smoothed = alpha * values[i] + (1 - alpha) * smoothed_values[-1]
            smoothed_values.append(smoothed)
        
        # Simple trend estimation
        if len(smoothed_values) >= 2:
            trend = smoothed_values[-1] - smoothed_values[-2]
        else:
            trend = 0
        
        # Predict future value
        predicted_value = smoothed_values[-1] + trend * (horizon / 5)  # Assume 5-minute intervals
        
        # Calculate confidence based on recent prediction accuracy
        recent_errors = []
        for i in range(max(1, len(values) - 10), len(values)):
            if i < len(smoothed_values):
                error = abs(values[i] - smoothed_values[i])
                recent_errors.append(error)
        
        avg_error = np.mean(recent_errors) if recent_errors else 0
        max_possible_error = max(values) - min(values)
        confidence = max(0, 1 - (avg_error / max_possible_error)) if max_possible_error > 0 else 0.5
        
        return {
            "value": predicted_value,
            "confidence": confidence,
            "trend": trend,
            "avg_error": avg_error
        }
    
    def _seasonal_prediction(self, history: List[Dict], horizon: int) -> Dict[str, Any]:
        """Seasonal/cyclical pattern prediction."""
        # Look for daily patterns (assuming metrics every 30 seconds, 2880 points per day)
        values = [item["value"] for item in history]
        
        if len(values) < 100:
            return {"value": values[-1], "confidence": 0.0}
        
        # Simple seasonal decomposition - look for patterns
        # This is a simplified approach; production would use more sophisticated methods
        
        # Try to detect 24-hour cycle (if data spans multiple days)
        cycle_length = min(96, len(values) // 3)  # Assume 15-minute intervals for daily cycle
        
        if len(values) < cycle_length * 2:
            return {"value": values[-1], "confidence": 0.0}
        
        # Calculate seasonal component
        seasonal_avg = []
        for i in range(cycle_length):
            cycle_values = []
            j = i
            while j < len(values):
                cycle_values.append(values[j])
                j += cycle_length
            
            if cycle_values:
                seasonal_avg.append(np.mean(cycle_values))
        
        # Predict based on seasonal pattern
        position_in_cycle = (len(values) + horizon // 5) % cycle_length  # Assuming 5-min intervals
        predicted_value = seasonal_avg[position_in_cycle] if seasonal_avg else values[-1]
        
        # Calculate confidence based on seasonal pattern strength
        if len(seasonal_avg) > 1:
            seasonal_variance = np.var(seasonal_avg)
            total_variance = np.var(values)
            confidence = min(1, seasonal_variance / total_variance) if total_variance > 0 else 0.5
        else:
            confidence = 0.0
        
        return {
            "value": predicted_value,
            "confidence": confidence,
            "cycle_length": cycle_length,
            "seasonal_variance": seasonal_variance if 'seasonal_variance' in locals() else 0
        }
    
    def _combine_predictions(self, predictions: Dict[str, Dict], metric_name: str) -> Dict[str, Any]:
        """Combine multiple predictions using weighted average."""
        if not predictions:
            return {"value": 0, "confidence": 0.0}
        
        total_weight = 0
        weighted_value = 0
        max_confidence = 0
        
        # Weight predictions by their confidence
        for method, pred in predictions.items():
            confidence = pred.get("confidence", 0.0)
            value = pred.get("value", 0)
            
            # Apply method-specific weights
            method_weight = {
                "linear": 1.0,
                "exponential": 0.8,
                "seasonal": 0.6
            }.get(method, 0.5)
            
            weight = confidence * method_weight
            total_weight += weight
            weighted_value += value * weight
            max_confidence = max(max_confidence, confidence)
        
        if total_weight > 0:
            combined_value = weighted_value / total_weight
        else:
            combined_value = list(predictions.values())[0].get("value", 0)
        
        return {
            "value": combined_value,
            "confidence": max_confidence,
            "methods_used": len(predictions)
        }
    
    def _evaluate_failure_risk(self, prediction: Dict, metric_name: str, 
                              history: List[Dict]) -> Tuple[bool, float, str]:
        """Evaluate failure risk based on prediction."""
        predicted_value = prediction.get("value", 0)
        confidence = prediction.get("confidence", 0.0)
        
        # Define thresholds based on metric type
        thresholds = {
            "cpu_usage": {"warning": 85, "critical": 95},
            "memory_usage": {"warning": 85, "critical": 95},
            "disk_usage": {"warning": 90, "critical": 95},
            "network_connectivity": {"warning": 80, "critical": 50, "inverted": True},
            "gpu_temperature": {"warning": 80, "critical": 90},
            "memory_leak_indicator": {"warning": 0.5, "critical": 1.0}
        }
        
        # Extract base metric name (remove prefixes/suffixes)
        base_metric = metric_name
        for key in thresholds.keys():
            if key in metric_name:
                base_metric = key
                break
        
        if base_metric not in thresholds:
            return False, 0.0, "unknown_metric"
        
        threshold_config = thresholds[base_metric]
        warning_threshold = threshold_config["warning"]
        critical_threshold = threshold_config["critical"]
        inverted = threshold_config.get("inverted", False)
        
        # Check if prediction indicates failure
        failure_likely = False
        risk_level = 0.0
        reason = "normal"
        
        if inverted:
            # For metrics where lower is worse (e.g., network connectivity)
            if predicted_value < critical_threshold:
                failure_likely = True
                risk_level = 0.9
                reason = "critical_threshold_predicted"
            elif predicted_value < warning_threshold:
                risk_level = 0.6
                reason = "warning_threshold_predicted"
        else:
            # For metrics where higher is worse
            if predicted_value > critical_threshold:
                failure_likely = True
                risk_level = 0.9
                reason = "critical_threshold_predicted"
            elif predicted_value > warning_threshold:
                risk_level = 0.6
                reason = "warning_threshold_predicted"
        
        # Adjust risk based on confidence and recent trend
        if len(history) >= 5:
            recent_trend = [item["value"] for item in history[-5:]]
            trend_direction = "stable"
            
            if len(recent_trend) >= 2:
                if recent_trend[-1] > recent_trend[0] * 1.1:
                    trend_direction = "increasing"
                elif recent_trend[-1] < recent_trend[0] * 0.9:
                    trend_direction = "decreasing"
            
            # Increase risk if trend is worsening
            if trend_direction == "increasing" and not inverted:
                risk_level *= 1.2
            elif trend_direction == "decreasing" and inverted:
                risk_level *= 1.2
        
        # Combine with prediction confidence
        final_confidence = min(1.0, risk_level * confidence)
        
        return failure_likely, final_confidence, reason
    
    def get_predictive_metrics(self) -> List[EnhancedHealthMetric]:
        """Generate predictive health metrics."""
        metrics = []
        
        for metric_name in self.metric_history.keys():
            if len(self.metric_history[metric_name]) < 5:
                continue
            
            # Generate predictions for different time horizons
            for horizon in [15, 30, 60]:  # 15 minutes, 30 minutes, 1 hour
                failure_likely, confidence, reason, metadata = self.predict_failure(metric_name, horizon)
                
                status = HealthStatus.HEALTHY
                if failure_likely and confidence > 0.8:
                    status = HealthStatus.CRITICAL
                elif failure_likely and confidence > 0.5:
                    status = HealthStatus.WARNING
                elif confidence > 0.3:
                    status = HealthStatus.WARNING
                
                metrics.append(EnhancedHealthMetric(
                    name=f"predictive_{metric_name}_{horizon}min",
                    value=confidence * 100,
                    status=status,
                    metric_type=MetricType.APPLICATION,
                    threshold_warning=50,
                    threshold_critical=80,
                    unit="%",
                    description=f"Failure prediction for {metric_name} in {horizon} minutes",
                    confidence_score=confidence,
                    predicted_value=metadata.get("predicted_value"),
                    tags={
                        "reason": reason,
                        "base_metric": metric_name,
                        "horizon_minutes": str(horizon),
                        "prediction_methods": ",".join(metadata.get("prediction_methods", [])),
                        "data_points": str(metadata.get("data_points_used", 0))
                    }
                ))
        
        return metrics


class AlertManager:
    """Enterprise alert management with advanced escalation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_policies: Dict[str, AlertPolicy] = self._init_default_policies()
        self.notification_channels = self._init_notification_channels()
        self.alert_history: List[HealthAlert] = []
        self._lock = threading.Lock()
        self._escalation_thread = None
        self._start_escalation_monitor()
    
    def _init_default_policies(self) -> Dict[str, AlertPolicy]:
        """Initialize comprehensive alert policies."""
        return {
            "system_critical": AlertPolicy(
                name="system_critical",
                severity=AlertSeverity.CRITICAL,
                conditions={"status": ["critical", "failure"], "metric_type": "system"},
                escalation_delay_minutes=5,
                max_escalations=3,
                notification_channels=["log", "email", "webhook"],
                cooldown_minutes=30
            ),
            "memory_leak": AlertPolicy(
                name="memory_leak",
                severity=AlertSeverity.WARNING,
                conditions={"metric_name": "memory_leak_indicator", "value": ">0.5"},
                escalation_delay_minutes=15,
                notification_channels=["log", "email"],
                auto_resolve=True,
                cooldown_minutes=60
            ),
            "disk_critical": AlertPolicy(
                name="disk_critical",
                severity=AlertSeverity.CRITICAL,
                conditions={"metric_name": "disk_usage*", "value": ">95"},
                escalation_delay_minutes=10,
                notification_channels=["log", "email", "webhook", "slack"],
                cooldown_minutes=120
            ),
            "network_failure": AlertPolicy(
                name="network_failure",
                severity=AlertSeverity.CRITICAL,
                conditions={"metric_name": "network_connectivity", "value": "<50"},
                escalation_delay_minutes=5,
                notification_channels=["log", "webhook"],
                cooldown_minutes=30
            ),
            "predictive_warning": AlertPolicy(
                name="predictive_warning",
                severity=AlertSeverity.WARNING,
                conditions={"metric_name": "predictive_*", "value": ">70"},
                escalation_delay_minutes=30,
                notification_channels=["log"],
                auto_resolve=True,
                cooldown_minutes=180
            ),
            "gpu_overheating": AlertPolicy(
                name="gpu_overheating",
                severity=AlertSeverity.CRITICAL,
                conditions={"metric_name": "gpu_*_temperature", "value": ">85"},
                escalation_delay_minutes=5,
                notification_channels=["log", "email", "webhook"],
                cooldown_minutes=60
            )
        }
    
    def _init_notification_channels(self) -> Dict[str, Callable]:
        """Initialize notification channels."""
        return {
            "log": self._notify_log,
            "email": self._notify_email,
            "webhook": self._notify_webhook,
            "slack": self._notify_slack,
            "sms": self._notify_sms
        }
    
    def _start_escalation_monitor(self):
        """Start the escalation monitoring thread."""
        if self._escalation_thread is None or not self._escalation_thread.is_alive():
            self._escalation_thread = threading.Thread(
                target=self._escalation_monitor_loop,
                daemon=True,
                name="alert-escalation"
            )
            self._escalation_thread.start()
    
    def _escalation_monitor_loop(self):
        """Monitor alerts for escalation."""
        while True:
            try:
                current_time = datetime.now()
                
                with self._lock:
                    for alert in list(self.active_alerts.values()):
                        if alert.acknowledged or alert.resolved:
                            continue
                        
                        policy = self.alert_policies.get(alert.metadata.get("policy"))
                        if not policy:
                            continue
                        
                        # Check if escalation is needed
                        time_since_alert = current_time - alert.timestamp
                        time_since_escalation = current_time - (alert.last_escalation or alert.timestamp)
                        
                        should_escalate = (
                            alert.escalation_level < policy.max_escalations and
                            time_since_escalation.total_seconds() >= policy.escalation_delay_minutes * 60
                        )
                        
                        if should_escalate:
                            self._escalate_alert(alert, policy)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in escalation monitor: {e}")
                time.sleep(60)
    
    def _escalate_alert(self, alert: HealthAlert, policy: AlertPolicy):
        """Escalate an alert to the next level."""
        alert.escalation_level += 1
        alert.last_escalation = datetime.now()
        
        # Increase severity for higher escalation levels
        if alert.escalation_level >= 2:
            if alert.severity == AlertSeverity.WARNING:
                alert.severity = AlertSeverity.CRITICAL
            elif alert.severity == AlertSeverity.CRITICAL:
                alert.severity = AlertSeverity.EMERGENCY
        
        # Send escalated notifications
        escalated_message = f"ESCALATED (Level {alert.escalation_level}): {alert.message}"
        
        for channel in policy.notification_channels:
            if channel in self.notification_channels:
                try:
                    # Create escalated alert copy
                    escalated_alert = HealthAlert(
                        alert_id=alert.alert_id,
                        timestamp=alert.timestamp,
                        severity=alert.severity,
                        metric_name=alert.metric_name,
                        message=escalated_message,
                        value=alert.value,
                        threshold=alert.threshold,
                        escalation_level=alert.escalation_level,
                        metadata={**alert.metadata, "escalated": True}
                    )
                    
                    self.notification_channels[channel](escalated_alert, policy)
                except Exception as e:
                    logger.error(f"Failed to send escalated notification via {channel}: {e}")
        
        logger.warning(f"Alert {alert.alert_id} escalated to level {alert.escalation_level}")
    
    def process_metric(self, metric: EnhancedHealthMetric):
        """Process metric and trigger alerts if needed."""
        for policy_name, policy in self.alert_policies.items():
            if self._matches_policy(metric, policy):
                self._trigger_alert(metric, policy)
    
    def _matches_policy(self, metric: EnhancedHealthMetric, policy: AlertPolicy) -> bool:
        """Enhanced policy matching with multiple conditions."""
        conditions = policy.conditions
        
        # Check status condition (can be list or single value)
        if "status" in conditions:
            expected_statuses = conditions["status"]
            if isinstance(expected_statuses, str):
                expected_statuses = [expected_statuses]
            
            if metric.status.value not in expected_statuses:
                return False
        
        # Check metric type
        if "metric_type" in conditions:
            if metric.metric_type.value != conditions["metric_type"]:
                return False
        
        # Check metric name (supports wildcards and patterns)
        if "metric_name" in conditions:
            pattern = conditions["metric_name"]
            if isinstance(pattern, list):
                # Multiple patterns - match any
                if not any(self._match_metric_name(metric.name, p) for p in pattern):
                    return False
            else:
                if not self._match_metric_name(metric.name, pattern):
                    return False
        
        # Check value conditions
        if "value" in conditions:
            condition = conditions["value"]
            if not self._check_value_condition(metric.value, condition):
                return False
        
        # Check confidence score condition
        if "confidence" in conditions:
            condition = conditions["confidence"]
            if not self._check_value_condition(metric.confidence_score, condition):
                return False
        
        # Check trend condition
        if "trend" in conditions:
            expected_trends = conditions["trend"]
            if isinstance(expected_trends, str):
                expected_trends = [expected_trends]
            
            if metric.trend not in expected_trends:
                return False
        
        return True
    
    def _match_metric_name(self, metric_name: str, pattern: str) -> bool:
        """Match metric name against pattern with wildcard support."""
        if "*" in pattern:
            # Simple wildcard matching
            pattern_parts = pattern.split("*")
            if len(pattern_parts) == 2:
                prefix, suffix = pattern_parts
                return metric_name.startswith(prefix) and metric_name.endswith(suffix)
            else:
                # Multiple wildcards - more complex matching
                import re
                regex_pattern = pattern.replace("*", ".*")
                return bool(re.match(f"^{regex_pattern}$", metric_name))
        else:
            return metric_name == pattern
    
    def _check_value_condition(self, value: float, condition: str) -> bool:
        """Check if value meets condition."""
        if condition.startswith(">="):
            threshold = float(condition[2:])
            return value >= threshold
        elif condition.startswith("<="):
            threshold = float(condition[2:])
            return value <= threshold
        elif condition.startswith(">"):
            threshold = float(condition[1:])
            return value > threshold
        elif condition.startswith("<"):
            threshold = float(condition[1:])
            return value < threshold
        elif condition.startswith("=="):
            threshold = float(condition[2:])
            return abs(value - threshold) < 0.001  # Float comparison
        else:
            # Default to greater than
            try:
                threshold = float(condition)
                return value > threshold
            except ValueError:
                return False
    
    def _trigger_alert(self, metric: EnhancedHealthMetric, policy: AlertPolicy):
        """Trigger alert with cooldown and deduplication."""
        alert_key = f"{metric.name}_{policy.name}"
        current_time = datetime.now()
        
        with self._lock:
            # Check for existing alert in cooldown
            existing_alert = self.active_alerts.get(alert_key)
            if existing_alert and not existing_alert.resolved:
                # Update existing alert
                existing_alert.value = metric.value
                existing_alert.timestamp = current_time
                return
            
            # Check cooldown from history
            recent_alerts = [
                alert for alert in self.alert_history
                if (alert.metric_name == metric.name and 
                    alert.metadata.get("policy") == policy.name and
                    current_time - alert.timestamp < timedelta(minutes=policy.cooldown_minutes))
            ]
            
            if recent_alerts:
                logger.debug(f"Alert {alert_key} in cooldown period")
                return
            
            # Create new alert
            alert = HealthAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=current_time,
                severity=policy.severity,
                metric_name=metric.name,
                message=f"{metric.name}: {metric.description} (Value: {metric.value}{metric.unit})",
                value=metric.value,
                threshold=metric.threshold_critical or metric.threshold_warning or 0,
                metadata={
                    "policy": policy.name,
                    "metric_type": metric.metric_type.value,
                    "status": metric.status.value,
                    "tags": metric.tags,
                    "confidence": metric.confidence_score,
                    "trend": metric.trend,
                    "predicted_value": metric.predicted_value
                }
            )
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            # Send initial notifications
            self._send_notifications(alert, policy)
            
            logger.info(f"Alert triggered: {alert.alert_id} for {metric.name}")
    
    def _send_notifications(self, alert: HealthAlert, policy: AlertPolicy):
        """Send notifications through all configured channels."""
        for channel in policy.notification_channels:
            if channel in self.notification_channels:
                try:
                    self.notification_channels[channel](alert, policy)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel}: {e}")
    
    def _notify_log(self, alert: HealthAlert, policy: AlertPolicy):
        """Enhanced log notification."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }[alert.severity]
        
        escalation_info = f" (Escalation {alert.escalation_level})" if alert.escalation_level > 0 else ""
        confidence_info = f" (Confidence: {alert.metadata.get('confidence', 0):.2f})" if alert.metadata.get('confidence') else ""
        
        logger.log(
            log_level, 
            f"HEALTH ALERT [{alert.severity.value.upper()}]{escalation_info}: "
            f"{alert.message}{confidence_info}"
        )
    
    def _notify_email(self, alert: HealthAlert, policy: AlertPolicy):
        """Send email notification."""
        email_config = self.config.get("email_config", {})
        if not email_config.get("smtp_server"):
            logger.debug("Email notification skipped - no SMTP configuration")
            return
        
        try:
            # Create email content
            subject = f"Health Alert [{alert.severity.value.upper()}]: {alert.metric_name}"
            
            body = f"""
Health Alert Details:
- Alert ID: {alert.alert_id}
- Metric: {alert.metric_name}
- Current Value: {alert.value}
- Threshold: {alert.threshold}
- Status: {alert.metadata.get('status', 'unknown')}
- Severity: {alert.severity.value}
- Timestamp: {alert.timestamp.isoformat()}

Description: {alert.message}

Additional Information:
- Confidence: {alert.metadata.get('confidence', 'N/A')}
- Trend: {alert.metadata.get('trend', 'N/A')}
- Tags: {alert.metadata.get('tags', {})}

This is an automated alert from the HF Eco2AI Health Monitoring System.
"""
            
            # Send email (placeholder implementation)
            logger.info(f"EMAIL ALERT sent for {alert.alert_id}: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _notify_webhook(self, alert: HealthAlert, policy: AlertPolicy):
        """Send webhook notification."""
        webhook_url = self.config.get("webhook_url")
        if not webhook_url:
            logger.debug("Webhook notification skipped - no URL configured")
            return
        
        try:
            payload = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "metric_name": alert.metric_name,
                "message": alert.message,
                "value": alert.value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat(),
                "escalation_level": alert.escalation_level,
                "metadata": alert.metadata,
                "policy": policy.name
            }
            
            response = requests.post(
                webhook_url, 
                json=payload, 
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook notification sent successfully for alert {alert.alert_id}")
            else:
                logger.warning(f"Webhook notification failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
    
    def _notify_slack(self, alert: HealthAlert, policy: AlertPolicy):
        """Send Slack notification."""
        slack_webhook = self.config.get("slack_webhook_url")
        if not slack_webhook:
            logger.debug("Slack notification skipped - no webhook URL configured")
            return
        
        try:
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "#ff0000"
            }
            
            payload = {
                "text": f"Health Alert: {alert.metric_name}",
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "fields": [
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Value", "value": f"{alert.value}", "short": True},
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Status", "value": alert.metadata.get('status', 'unknown'), "short": True},
                        {"title": "Message", "value": alert.message, "short": False}
                    ],
                    "footer": "HF Eco2AI Health Monitor",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(slack_webhook, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Slack notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
    
    def _notify_sms(self, alert: HealthAlert, policy: AlertPolicy):
        """Send SMS notification (placeholder)."""
        # This would integrate with SMS service like Twilio
        logger.info(f"SMS ALERT (placeholder): {alert.message}")
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self.active_alerts.values():
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.metadata["acknowledged_by"] = user
                    alert.metadata["acknowledged_at"] = datetime.now().isoformat()
                    logger.info(f"Alert {alert_id} acknowledged by {user}")
                    return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert."""
        with self._lock:
            for key, alert in list(self.active_alerts.items()):
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    alert.metadata["resolved_by"] = user
                    del self.active_alerts[key]
                    logger.info(f"Alert {alert_id} resolved by {user}")
                    return True
        return False
    
    def get_active_alerts(self) -> List[HealthAlert]:
        """Get all active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics."""
        with self._lock:
            active_alerts = list(self.active_alerts.values())
            
            # Count by severity
            severity_counts = {}
            for alert in active_alerts:
                severity = alert.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by policy
            policy_counts = {}
            for alert in active_alerts:
                policy = alert.metadata.get("policy", "unknown")
                policy_counts[policy] = policy_counts.get(policy, 0) + 1
            
            # Calculate resolution statistics from history
            recent_history = [
                alert for alert in self.alert_history
                if datetime.now() - alert.timestamp < timedelta(days=7)
            ]
            
            total_alerts_week = len(recent_history)
            resolved_alerts_week = len([a for a in recent_history if a.resolved])
            avg_resolution_time = 0
            
            if resolved_alerts_week > 0:
                resolution_times = [
                    (a.resolution_time - a.timestamp).total_seconds() / 60
                    for a in recent_history 
                    if a.resolved and a.resolution_time
                ]
                if resolution_times:
                    avg_resolution_time = sum(resolution_times) / len(resolution_times)
            
            return {
                "active_alerts": {
                    "total": len(active_alerts),
                    "by_severity": severity_counts,
                    "by_policy": policy_counts
                },
                "weekly_statistics": {
                    "total_alerts": total_alerts_week,
                    "resolved_alerts": resolved_alerts_week,
                    "resolution_rate": (resolved_alerts_week / total_alerts_week * 100) if total_alerts_week > 0 else 0,
                    "avg_resolution_time_minutes": round(avg_resolution_time, 2)
                },
                "escalation_statistics": {
                    "escalated_alerts": len([a for a in active_alerts if a.escalation_level > 0]),
                    "max_escalation_level": max([a.escalation_level for a in active_alerts], default=0)
                },
                "policy_statistics": {
                    "total_policies": len(self.alert_policies),
                    "policies_with_active_alerts": len(set(a.metadata.get("policy") for a in active_alerts))
                }
            }


class EnterpriseHealthMonitor:
    """Enterprise-grade health monitoring with comprehensive features."""
    
    def __init__(self, check_interval: int = 30, 
                 history_size: int = 1000,
                 alert_config: Optional[Dict[str, Any]] = None):
        """Initialize enterprise health monitor."""
        self.check_interval = check_interval
        self.history_size = history_size
        
        # Health history
        self.health_history: queue.deque = queue.deque(maxlen=history_size)
        
        # Monitoring state
        self.is_monitoring = False
        self._monitor_thread = None
        self._start_time = time.time()
        
        # Enhanced components
        self.network_checker = NetworkHealthChecker()
        self.memory_detector = MemoryLeakDetector()
        self.disk_monitor = DiskSpaceMonitor()
        self.predictive_analyzer = PredictiveHealthAnalyzer()
        self.alert_manager = AlertManager(alert_config)
        
        # Enhanced thresholds
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 85.0, 
            "memory_critical": 95.0,
            "disk_warning": 85.0,
            "disk_critical": 95.0,
            "network_warning": 80.0,
            "network_critical": 50.0,
            "gpu_temperature_warning": 80.0,
            "gpu_temperature_critical": 90.0,
            "cpu_load_warning": 80.0,
            "cpu_load_critical": 120.0
        }
        
        # Thread pool for async health checks
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="health-check")
        
        # Performance tracking
        self.performance_metrics = {
            "checks_completed": 0,
            "checks_failed": 0,
            "avg_check_duration": 0.0,
            "last_check_duration": 0.0
        }
        
        logger.info("Enterprise health monitor initialized with advanced capabilities")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring is already running")
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            daemon=True,
            name="health-monitor"
        )
        self._monitor_thread.start()
        logger.info("Enterprise health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.executor.shutdown(wait=False)
        logger.info("Enterprise health monitoring stopped")
    
    def _monitor_loop(self):
        """Enhanced monitoring loop with error recovery."""
        consecutive_failures = 0
        max_failures = 5
        
        while self.is_monitoring:
            check_start_time = time.time()
            
            try:
                health = self.check_system_health()
                self.health_history.append(health)
                
                # Update performance metrics
                check_duration = time.time() - check_start_time
                self.performance_metrics["checks_completed"] += 1
                self.performance_metrics["last_check_duration"] = check_duration
                
                # Update average duration
                total_checks = self.performance_metrics["checks_completed"]
                current_avg = self.performance_metrics["avg_check_duration"]
                self.performance_metrics["avg_check_duration"] = (
                    (current_avg * (total_checks - 1) + check_duration) / total_checks
                )
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Trigger alerts if needed
                if health.overall_status in [HealthStatus.CRITICAL, HealthStatus.FAILURE]:
                    self._trigger_system_alerts(health)
                
            except Exception as e:
                consecutive_failures += 1
                self.performance_metrics["checks_failed"] += 1
                
                logger.error(f"Error in health monitoring loop (failure {consecutive_failures}): {e}")
                
                # If too many consecutive failures, increase check interval
                if consecutive_failures >= max_failures:
                    logger.critical(f"Too many consecutive health check failures ({max_failures}). Increasing check interval.")
                    time.sleep(self.check_interval * 2)  # Double the interval
                    consecutive_failures = 0  # Reset counter
            
            # Adaptive sleep based on system load
            sleep_duration = self._calculate_adaptive_sleep()
            time.sleep(sleep_duration)
    
    def _calculate_adaptive_sleep(self) -> float:
        """Calculate adaptive sleep duration based on system load."""
        try:
            # Get current CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Adjust sleep duration based on load
            if cpu_percent > 90:
                return self.check_interval * 1.5  # Slow down when system is overloaded
            elif cpu_percent < 30:
                return self.check_interval * 0.8  # Speed up when system is idle
            else:
                return self.check_interval
                
        except Exception:
            return self.check_interval
    
    def _trigger_system_alerts(self, health: SystemHealth):
        """Trigger system-level alerts for critical conditions."""
        try:
            # System-wide critical alert
            if health.overall_status == HealthStatus.FAILURE:
                logger.critical(f"SYSTEM FAILURE detected: {len(health.degraded_services)} services degraded")
            
            # Log predictive warnings
            if health.predictive_warnings:
                for warning in health.predictive_warnings:
                    logger.warning(f"Predictive alert: {warning}")
            
        except Exception as e:
            logger.error(f"Error triggering system alerts: {e}")
    
    def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check."""
        metrics = []
        degraded_services = []
        predictive_warnings = []
        
        try:
            # Collect all metrics
            metrics.extend(self._get_cpu_metrics())
            metrics.extend(self.memory_detector.get_memory_metrics())
            metrics.extend(self.disk_monitor.get_disk_metrics())
            
            # Network metrics
            network_metrics = [
                self.network_checker.check_connectivity(),
                self.network_checker.check_dns_resolution()
            ]
            metrics.extend(network_metrics)
            
            # GPU metrics (if available)
            gpu_metrics = self._get_gpu_metrics()
            metrics.extend(gpu_metrics)
            
            # Process metrics through predictive analyzer
            for metric in metrics:
                self.predictive_analyzer.add_metric_data(metric)
                self.alert_manager.process_metric(metric)
            
            # Get predictive metrics
            predictive_metrics = self.predictive_analyzer.get_predictive_metrics()
            metrics.extend(predictive_metrics)
            
            # Identify degraded services and warnings
            for metric in metrics:
                if metric.status in [HealthStatus.CRITICAL, HealthStatus.FAILURE]:
                    degraded_services.append(f"{metric.name}: {metric.value}{metric.unit}")
                
                if metric.name.startswith("predictive_") and metric.status != HealthStatus.HEALTHY:
                    predictive_warnings.append(
                        f"{metric.tags.get('base_metric', metric.name)}: {metric.tags.get('reason', 'unknown')}"
                    )
            
            # Calculate overall system status
            overall_status = self._calculate_overall_status(metrics)
            
            # Get basic system info for compatibility
            cpu_percent = next((m.value for m in metrics if m.name == "cpu_usage"), 0)
            memory_percent = next((m.value for m in metrics if m.name == "memory_usage"), 0)
            
            # Find primary disk usage
            disk_metrics = [m for m in metrics if m.name.startswith("disk_usage") and not m.name.endswith("_io")]
            disk_percent = disk_metrics[0].value if disk_metrics else 0
            
            # Network I/O
            try:
                network = psutil.net_io_counters()
                network_io = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            except Exception:
                network_io = {"bytes_sent": 0, "bytes_recv": 0, "packets_sent": 0, "packets_recv": 0}
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime
            uptime_seconds = time.time() - self._start_time
            
            # Alert count
            alert_count = len(self.alert_manager.get_active_alerts())
            
            return SystemHealth(
                overall_status=overall_status,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                process_count=process_count,
                uptime_seconds=uptime_seconds,
                timestamp=time.time(),
                metrics=metrics,
                alert_count=alert_count,
                degraded_services=degraded_services,
                predictive_warnings=predictive_warnings
            )
            
        except Exception as e:
            logger.error(f"Error during system health check: {e}")
            # Return minimal health status in case of error
            return SystemHealth(
                overall_status=HealthStatus.FAILURE,
                cpu_percent=0,
                memory_percent=0,
                disk_percent=0,
                network_io={},
                process_count=0,
                uptime_seconds=time.time() - self._start_time,
                timestamp=time.time(),
                metrics=[],
                alert_count=0,
                degraded_services=["health_check_system: monitoring failure"],
                predictive_warnings=[]
            )
    
    def _get_cpu_metrics(self) -> List[EnhancedHealthMetric]:
        """Get comprehensive CPU metrics."""
        metrics = []
        
        try:
            # Overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._evaluate_threshold(
                cpu_percent, 
                self.thresholds["cpu_warning"], 
                self.thresholds["cpu_critical"]
            )
            
            metrics.append(EnhancedHealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                status=cpu_status,
                metric_type=MetricType.SYSTEM,
                threshold_warning=self.thresholds["cpu_warning"],
                threshold_critical=self.thresholds["cpu_critical"],
                unit="%",
                description="Overall CPU utilization percentage"
            ))
            
            # Per-core CPU usage
            try:
                cpu_per_core = psutil.cpu_percent(percpu=True)
                for i, core_percent in enumerate(cpu_per_core):
                    core_status = self._evaluate_threshold(
                        core_percent,
                        self.thresholds["cpu_warning"],
                        self.thresholds["cpu_critical"]
                    )
                    metrics.append(EnhancedHealthMetric(
                        name=f"cpu_core_{i}_usage",
                        value=core_percent,
                        status=core_status,
                        metric_type=MetricType.SYSTEM,
                        threshold_warning=self.thresholds["cpu_warning"],
                        threshold_critical=self.thresholds["cpu_critical"],
                        unit="%",
                        description=f"CPU core {i} utilization",
                        tags={"core": str(i)}
                    ))
            except Exception:
                pass
            
            # CPU load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                cpu_count = psutil.cpu_count()
                
                for i, (period, load) in enumerate(zip(["1min", "5min", "15min"], load_avg)):
                    load_percent = (load / cpu_count) * 100
                    load_status = self._evaluate_threshold(
                        load_percent,
                        self.thresholds["cpu_load_warning"],
                        self.thresholds["cpu_load_critical"]
                    )
                    metrics.append(EnhancedHealthMetric(
                        name=f"cpu_load_avg_{period}",
                        value=load_percent,
                        status=load_status,
                        metric_type=MetricType.SYSTEM,
                        threshold_warning=self.thresholds["cpu_load_warning"],
                        threshold_critical=self.thresholds["cpu_load_critical"],
                        unit="%",
                        description=f"CPU load average ({period}) as percentage of available cores",
                        tags={"period": period, "cpu_count": str(cpu_count)}
                    ))
            except Exception:
                pass  # Load average not available on all systems
            
            # CPU frequency (if available)
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    # Calculate frequency utilization
                    freq_percent = (cpu_freq.current / cpu_freq.max) * 100 if cpu_freq.max > 0 else 0
                    
                    metrics.append(EnhancedHealthMetric(
                        name="cpu_frequency_utilization",
                        value=freq_percent,
                        status=HealthStatus.HEALTHY,  # Frequency scaling is normal
                        metric_type=MetricType.SYSTEM,
                        unit="%",
                        description="CPU frequency utilization (current/max)",
                        tags={
                            "current_mhz": str(round(cpu_freq.current, 2)),
                            "max_mhz": str(round(cpu_freq.max, 2)),
                            "min_mhz": str(round(cpu_freq.min, 2))
                        }
                    ))
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error getting CPU metrics: {e}")
        
        return metrics
    
    def _get_gpu_metrics(self) -> List[EnhancedHealthMetric]:
        """Get comprehensive GPU health metrics."""
        metrics = []
        
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    memory_util = utilization.memory
                    
                    gpu_status = self._evaluate_threshold(gpu_util, 80, 95)
                    metrics.append(EnhancedHealthMetric(
                        name=f"gpu_{i}_utilization",
                        value=gpu_util,
                        status=gpu_status,
                        metric_type=MetricType.SYSTEM,
                        threshold_warning=80,
                        threshold_critical=95,
                        unit="%",
                        description=f"GPU {i} compute utilization",
                        tags={"gpu_id": str(i), "memory_util": str(memory_util)}
                    ))
                except Exception:
                    pass
                
                # GPU temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temp_status = self._evaluate_threshold(
                        temp,
                        self.thresholds["gpu_temperature_warning"],
                        self.thresholds["gpu_temperature_critical"]
                    )
                    metrics.append(EnhancedHealthMetric(
                        name=f"gpu_{i}_temperature",
                        value=temp,
                        status=temp_status,
                        metric_type=MetricType.SYSTEM,
                        threshold_warning=self.thresholds["gpu_temperature_warning"],
                        threshold_critical=self.thresholds["gpu_temperature_critical"],
                        unit="°C",
                        description=f"GPU {i} temperature",
                        tags={"gpu_id": str(i)}
                    ))
                except Exception:
                    pass
                
                # GPU memory
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_percent = (mem_info.used / mem_info.total) * 100
                    
                    mem_status = self._evaluate_threshold(mem_percent, 80, 95)
                    metrics.append(EnhancedHealthMetric(
                        name=f"gpu_{i}_memory",
                        value=mem_percent,
                        status=mem_status,
                        metric_type=MetricType.SYSTEM,
                        threshold_warning=80,
                        threshold_critical=95,
                        unit="%",
                        description=f"GPU {i} memory utilization",
                        tags={
                            "gpu_id": str(i),
                            "used_mb": str(round(mem_info.used / (1024**2))),
                            "total_mb": str(round(mem_info.total / (1024**2))),
                            "free_mb": str(round(mem_info.free / (1024**2)))
                        }
                    ))
                except Exception:
                    pass
                
                # GPU power consumption (if available)
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitDefault(handle) / 1000.0
                    
                    power_percent = (power_draw / power_limit) * 100 if power_limit > 0 else 0
                    
                    power_status = self._evaluate_threshold(power_percent, 80, 95)
                    metrics.append(EnhancedHealthMetric(
                        name=f"gpu_{i}_power",
                        value=power_percent,
                        status=power_status,
                        metric_type=MetricType.SYSTEM,
                        threshold_warning=80,
                        threshold_critical=95,
                        unit="%",
                        description=f"GPU {i} power utilization",
                        tags={
                            "gpu_id": str(i),
                            "power_draw_watts": str(round(power_draw, 1)),
                            "power_limit_watts": str(round(power_limit, 1))
                        }
                    ))
                except Exception:
                    pass
                    
        except ImportError:
            # pynvml not available
            pass
        except Exception as e:
            logger.debug(f"GPU monitoring not available: {e}")
        
        return metrics
    
    def _evaluate_threshold(self, value: float, warning: float, critical: float) -> HealthStatus:
        """Evaluate metric against thresholds."""
        if value >= critical:
            return HealthStatus.CRITICAL
        elif value >= warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _calculate_overall_status(self, metrics: List[EnhancedHealthMetric]) -> HealthStatus:
        """Calculate overall system status from individual metrics."""
        status_priority = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 1, 
            HealthStatus.DEGRADED: 2,
            HealthStatus.CRITICAL: 3,
            HealthStatus.FAILURE: 4,
            HealthStatus.RECOVERING: 1  # Treat as warning level
        }
        
        overall_status = HealthStatus.HEALTHY
        critical_count = 0
        warning_count = 0
        failure_count = 0
        
        for metric in metrics:
            if metric.status == HealthStatus.FAILURE:
                failure_count += 1
            elif metric.status == HealthStatus.CRITICAL:
                critical_count += 1
            elif metric.status == HealthStatus.WARNING:
                warning_count += 1
            
            if status_priority[metric.status] > status_priority[overall_status]:
                overall_status = metric.status
        
        # Apply business logic for status determination
        if failure_count >= 1:
            overall_status = HealthStatus.FAILURE
        elif critical_count >= 3:
            overall_status = HealthStatus.FAILURE
        elif critical_count >= 1:
            overall_status = HealthStatus.CRITICAL
        elif warning_count >= 5:
            overall_status = HealthStatus.DEGRADED
        elif warning_count >= 2:
            overall_status = HealthStatus.WARNING
        
        return overall_status
    
    def get_current_health(self) -> Optional[SystemHealth]:
        """Get current health status."""
        if not self.health_history:
            return self.force_health_check()
        return self.health_history[-1]
    
    def get_health_history(self, limit: Optional[int] = None) -> List[SystemHealth]:
        """Get health history."""
        history = list(self.health_history)
        if limit:
            return history[-limit:]
        return history
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary with insights."""
        if not self.health_history:
            return {"status": "no_data"}
        
        recent_health = list(self.health_history)[-10:]
        
        # Calculate averages
        avg_cpu = sum(h.cpu_percent for h in recent_health) / len(recent_health)
        avg_memory = sum(h.memory_percent for h in recent_health) / len(recent_health)
        avg_disk = sum(h.disk_percent for h in recent_health) / len(recent_health)
        
        # Status distribution
        status_counts = {}
        metric_counts = {}
        for health in recent_health:
            status = health.overall_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            for metric in health.metrics:
                metric_type = metric.metric_type.value
                metric_counts[metric_type] = metric_counts.get(metric_type, 0) + 1
        
        current = recent_health[-1] if recent_health else None
        
        # Alert statistics
        alert_stats = self.alert_manager.get_alert_statistics()
        
        # Predictive insights
        predictive_metrics = self.predictive_analyzer.get_predictive_metrics()
        high_risk_predictions = [
            m for m in predictive_metrics 
            if m.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]
        ]
        
        # Performance statistics
        check_success_rate = 0
        if self.performance_metrics["checks_completed"] + self.performance_metrics["checks_failed"] > 0:
            total_checks = self.performance_metrics["checks_completed"] + self.performance_metrics["checks_failed"]
            check_success_rate = (self.performance_metrics["checks_completed"] / total_checks) * 100
        
        return {
            "current_status": current.overall_status.value if current else "unknown",
            "uptime_hours": round((time.time() - self._start_time) / 3600, 2),
            "system_averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2),
                "disk_percent": round(avg_disk, 2)
            },
            "status_distribution": status_counts,
            "metric_type_counts": metric_counts,
            "monitoring_performance": {
                "checks_completed": self.performance_metrics["checks_completed"],
                "checks_failed": self.performance_metrics["checks_failed"],
                "success_rate_percent": round(check_success_rate, 2),
                "avg_check_duration_seconds": round(self.performance_metrics["avg_check_duration"], 3),
                "last_check_duration_seconds": round(self.performance_metrics["last_check_duration"], 3)
            },
            "alert_summary": alert_stats,
            "predictive_insights": {
                "total_predictions": len(predictive_metrics),
                "high_risk_count": len(high_risk_predictions),
                "predictions": [
                    {
                        "metric": m.name,
                        "confidence": round(m.value, 2),
                        "risk_level": m.status.value,
                        "horizon": m.tags.get("horizon_minutes", "unknown")
                    }
                    for m in high_risk_predictions[:5]
                ]
            },
            "system_capabilities": {
                "cpu_cores": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "gpu_count": self._get_gpu_count(),
                "network_monitoring": True,
                "predictive_analysis": True,
                "advanced_alerting": True
            },
            "degraded_services": current.degraded_services if current else [],
            "predictive_warnings": current.predictive_warnings if current else [],
            "total_checks": len(self.health_history),
            "monitoring_active": self.is_monitoring
        }
    
    def _get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetCount()
        except (ImportError, Exception):
            return 0
    
    def force_health_check(self) -> SystemHealth:
        """Force an immediate comprehensive health check."""
        logger.info("Forcing immediate health check")
        health = self.check_system_health()
        self.health_history.append(health)
        return health
    
    def export_health_data(self, file_path: Path, format: str = "json", 
                          include_predictions: bool = True):
        """Export comprehensive health data."""
        health_data = {
            "export_metadata": {
                "timestamp": time.time(),
                "format": format,
                "version": "1.0",
                "include_predictions": include_predictions
            },
            "summary": self.get_health_summary(),
            "current_health": self.get_current_health().to_dict() if self.get_current_health() else None,
            "history": [h.to_dict() for h in self.health_history],
            "alert_statistics": self.alert_manager.get_alert_statistics(),
            "active_alerts": [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
            "performance_metrics": self.performance_metrics
        }
        
        if include_predictions:
            predictive_metrics = self.predictive_analyzer.get_predictive_metrics()
            health_data["predictive_analysis"] = [
                {
                    "name": m.name,
                    "value": m.value,
                    "confidence": m.confidence_score,
                    "status": m.status.value,
                    "tags": m.tags
                }
                for m in predictive_metrics
            ]
        
        try:
            if format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(health_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Health data exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export health data: {e}")
            raise
    
    def cleanup_resources(self):
        """Enhanced resource cleanup with comprehensive maintenance."""
        try:
            # Memory cleanup
            before_gc = psutil.virtual_memory().percent
            gc.collect()
            after_gc = psutil.virtual_memory().percent
            
            # Clean up old alert data
            cutoff_time = datetime.now() - timedelta(days=7)
            old_alerts = [
                alert for alert in self.alert_manager.alert_history
                if alert.timestamp < cutoff_time
            ]
            
            # Keep only recent alert history
            self.alert_manager.alert_history = [
                alert for alert in self.alert_manager.alert_history
                if alert.timestamp >= cutoff_time
            ]
            
            # Clean up prediction history
            for metric_name in list(self.predictive_analyzer.metric_history.keys()):
                history = self.predictive_analyzer.metric_history[metric_name]
                logger.debug(f"Prediction history for {metric_name}: {len(history)} entries")
            
            # Log cleanup results
            memory = psutil.virtual_memory()
            logger.info(
                f"Resource cleanup completed - "
                f"Memory: {memory.percent:.1f}% used, "
                f"Available: {memory.available / (1024**3):.1f} GB, "
                f"GC freed: {before_gc - after_gc:.2f}%, "
                f"Old alerts cleaned: {len(old_alerts)}"
            )
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")


# Enhanced Carbon Tracking Health Monitor
class CarbonTrackingHealthMonitor(EnterpriseHealthMonitor):
    """Specialized health monitor for carbon tracking operations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Carbon-specific thresholds
        self.thresholds.update({
            "energy_rate_warning": 1000.0,  # Watts
            "energy_rate_critical": 2000.0,  # Watts
            "co2_rate_warning": 0.5,  # kg/hour
            "co2_rate_critical": 1.0,  # kg/hour
            "model_training_warning": 85.0,  # GPU utilization during training
            "model_training_critical": 95.0,
        })
        
        # Carbon-specific monitoring
        self.carbon_history = deque(maxlen=1000)
        
        logger.info("Carbon tracking health monitor initialized")
    
    def check_carbon_health(self, current_power: float = 0, 
                           current_co2_rate: float = 0) -> List[EnhancedHealthMetric]:
        """Check carbon-specific health metrics."""
        metrics = []
        
        # Power consumption rate
        power_status = self._evaluate_threshold(
            current_power,
            self.thresholds["energy_rate_warning"],
            self.thresholds["energy_rate_critical"]
        )
        metrics.append(EnhancedHealthMetric(
            name="power_consumption",
            value=current_power,
            status=power_status,
            metric_type=MetricType.APPLICATION,
            threshold_warning=self.thresholds["energy_rate_warning"],
            threshold_critical=self.thresholds["energy_rate_critical"],
            unit="W",
            description="Current power consumption rate",
            tags={"source": "carbon_tracking"}
        ))
        
        # CO2 emission rate
        co2_status = self._evaluate_threshold(
            current_co2_rate,
            self.thresholds["co2_rate_warning"],
            self.thresholds["co2_rate_critical"]
        )
        metrics.append(EnhancedHealthMetric(
            name="co2_emission_rate",
            value=current_co2_rate,
            status=co2_status,
            metric_type=MetricType.APPLICATION,
            threshold_warning=self.thresholds["co2_rate_warning"],
            threshold_critical=self.thresholds["co2_rate_critical"],
            unit="kg/hr",
            description="Current CO2 emission rate",
            tags={"source": "carbon_tracking"}
        ))
        
        # Training efficiency metric (if in training mode)
        gpu_metrics = [m for m in self.get_current_health().metrics if "gpu" in m.name and "utilization" in m.name]
        if gpu_metrics:
            avg_gpu_util = sum(m.value for m in gpu_metrics) / len(gpu_metrics)
            
            # Calculate training efficiency (higher GPU util with lower power is better)
            if current_power > 0:
                efficiency = avg_gpu_util / (current_power / 100)  # Normalize power to 0-100 scale
            else:
                efficiency = avg_gpu_util
            
            efficiency_status = HealthStatus.HEALTHY
            if efficiency < 0.5:
                efficiency_status = HealthStatus.WARNING
            if efficiency < 0.3:
                efficiency_status = HealthStatus.CRITICAL
            
            metrics.append(EnhancedHealthMetric(
                name="training_efficiency",
                value=efficiency,
                status=efficiency_status,
                metric_type=MetricType.APPLICATION,
                threshold_warning=0.5,
                threshold_critical=0.3,
                unit="ratio",
                description="Training efficiency (GPU utilization per unit power)",
                tags={
                    "avg_gpu_util": str(round(avg_gpu_util, 2)),
                    "power_consumption": str(round(current_power, 2))
                }
            ))
        
        return metrics
    
    def get_carbon_summary(self) -> Dict[str, Any]:
        """Get carbon-specific summary."""
        base_summary = self.get_health_summary()
        
        # Add carbon-specific metrics
        current_health = self.get_current_health()
        if current_health:
            carbon_metrics = [
                m for m in current_health.metrics 
                if m.tags.get("source") == "carbon_tracking"
            ]
            
            base_summary["carbon_tracking"] = {
                "power_metrics_count": len([m for m in carbon_metrics if "power" in m.name]),
                "co2_metrics_count": len([m for m in carbon_metrics if "co2" in m.name]),
                "training_active": any("training" in m.name for m in carbon_metrics),
                "carbon_alerts": len([
                    a for a in self.alert_manager.get_active_alerts()
                    if any("carbon" in tag for tag in a.metadata.get("tags", {}).values())
                ])
            }
        
        return base_summary


# Global instances
_enhanced_health_monitor = CarbonTrackingHealthMonitor(
    check_interval=30,
    history_size=1000,
    alert_config={
        "webhook_url": os.environ.get("HEALTH_WEBHOOK_URL"),
        "slack_webhook_url": os.environ.get("SLACK_WEBHOOK_URL"),
        "email_config": {
            "smtp_server": os.environ.get("SMTP_SERVER", "localhost"),
            "smtp_port": int(os.environ.get("SMTP_PORT", "587")),
            "username": os.environ.get("SMTP_USERNAME"),
            "password": os.environ.get("SMTP_PASSWORD")
        }
    }
)


def get_enhanced_health_monitor() -> CarbonTrackingHealthMonitor:
    """Get the enhanced global health monitor instance."""
    return _enhanced_health_monitor


def start_enhanced_health_monitoring():
    """Start enhanced global health monitoring."""
    _enhanced_health_monitor.start_monitoring()


def stop_enhanced_health_monitoring():
    """Stop enhanced health monitoring."""
    _enhanced_health_monitor.stop_monitoring()


def get_system_health() -> Optional[SystemHealth]:
    """Get current system health."""
    return _enhanced_health_monitor.get_current_health()


def get_health_dashboard() -> Dict[str, Any]:
    """Get comprehensive health dashboard data."""
    monitor = get_enhanced_health_monitor()
    return {
        "summary": monitor.get_health_summary(),
        "current_health": monitor.get_current_health().to_dict() if monitor.get_current_health() else None,
        "active_alerts": [asdict(alert) for alert in monitor.alert_manager.get_active_alerts()],
        "alert_statistics": monitor.alert_manager.get_alert_statistics(),
        "predictive_metrics": [
            {
                "name": m.name,
                "value": m.value,
                "confidence": m.confidence_score,
                "status": m.status.value
            }
            for m in monitor.predictive_analyzer.get_predictive_metrics()[:10]
        ]
    }


def acknowledge_health_alert(alert_id: str, user: str = "system") -> bool:
    """Acknowledge a health alert."""
    monitor = get_enhanced_health_monitor()
    return monitor.alert_manager.acknowledge_alert(alert_id, user)


def resolve_health_alert(alert_id: str, user: str = "system") -> bool:
    """Resolve a health alert."""
    monitor = get_enhanced_health_monitor()
    return monitor.alert_manager.resolve_alert(alert_id, user)


def health_alert_callback(health: SystemHealth):
    """Enhanced health alert callback with comprehensive logging."""
    logger.warning(f"System health alert: {health.overall_status.value}")
    logger.warning(f"CPU: {health.cpu_percent:.1f}%, Memory: {health.memory_percent:.1f}%, Disk: {health.disk_percent:.1f}%")
    
    # Log degraded services
    if health.degraded_services:
        logger.error(f"Degraded services: {len(health.degraded_services)}")
        for service in health.degraded_services[:3]:
            logger.error(f"  - {service}")
    
    # Log predictive warnings
    if health.predictive_warnings:
        logger.warning(f"Predictive warnings: {len(health.predictive_warnings)}")
        for warning in health.predictive_warnings[:3]:
            logger.warning(f"  - {warning}")
    
    # Log alert count
    if health.alert_count > 0:
        logger.warning(f"Active alerts: {health.alert_count}")