"""Health monitoring and system diagnostics for carbon tracking."""

import time
import logging
import threading
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue
import gc


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: float = None
    unit: str = ""
    description: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class SystemHealth:
    """System health summary."""
    overall_status: HealthStatus
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    process_count: int
    uptime_seconds: float
    timestamp: float
    metrics: List[HealthMetric]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, check_interval: int = 30, 
                 history_size: int = 1000,
                 alert_callbacks: Optional[List[Callable]] = None):
        """Initialize health monitor."""
        self.check_interval = check_interval
        self.history_size = history_size
        self.alert_callbacks = alert_callbacks or []
        
        # Health history
        self.health_history: queue.deque = queue.deque(maxlen=history_size)
        
        # Monitoring state
        self.is_monitoring = False
        self._monitor_thread = None
        self._start_time = time.time()
        
        # Thresholds (configurable)
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 85.0, 
            "memory_critical": 95.0,
            "disk_warning": 85.0,
            "disk_critical": 95.0,
        }
        
        # Thread pool for async health checks
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="health-check")
        
        logger.info("Health monitor initialized")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            daemon=True,
            name="health-monitor"
        )
        self._monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.executor.shutdown(wait=False)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                health = self.check_system_health()
                self.health_history.append(health)
                
                # Trigger alerts if needed
                if health.overall_status in [HealthStatus.CRITICAL, HealthStatus.FAILURE]:
                    self._trigger_alerts(health)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
            
            time.sleep(self.check_interval)
    
    def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check."""
        metrics = []
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = self._evaluate_threshold(
            cpu_percent, 
            self.thresholds["cpu_warning"], 
            self.thresholds["cpu_critical"]
        )
        metrics.append(HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            status=cpu_status,
            threshold_warning=self.thresholds["cpu_warning"],
            threshold_critical=self.thresholds["cpu_critical"],
            unit="%",
            description="CPU utilization percentage"
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_status = self._evaluate_threshold(
            memory.percent,
            self.thresholds["memory_warning"],
            self.thresholds["memory_critical"]
        )
        metrics.append(HealthMetric(
            name="memory_usage",
            value=memory.percent,
            status=memory_status,
            threshold_warning=self.thresholds["memory_warning"],
            threshold_critical=self.thresholds["memory_critical"],
            unit="%",
            description="Memory utilization percentage"
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_status = self._evaluate_threshold(
            disk_percent,
            self.thresholds["disk_warning"],
            self.thresholds["disk_critical"]
        )
        metrics.append(HealthMetric(
            name="disk_usage",
            value=disk_percent,
            status=disk_status,
            threshold_warning=self.thresholds["disk_warning"],
            threshold_critical=self.thresholds["disk_critical"],
            unit="%",
            description="Disk space utilization percentage"
        ))
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        # Uptime
        uptime_seconds = time.time() - self._start_time
        
        # Overall status
        status_priority = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 1, 
            HealthStatus.CRITICAL: 2,
            HealthStatus.FAILURE: 3
        }
        
        overall_status = HealthStatus.HEALTHY
        for metric in metrics:
            if status_priority[metric.status] > status_priority[overall_status]:
                overall_status = metric.status
        
        return SystemHealth(
            overall_status=overall_status,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk_percent,
            network_io=network_io,
            process_count=process_count,
            uptime_seconds=uptime_seconds,
            timestamp=time.time(),
            metrics=metrics
        )
    
    def _evaluate_threshold(self, value: float, warning: float, critical: float) -> HealthStatus:
        """Evaluate metric against thresholds."""
        if value >= critical:
            return HealthStatus.CRITICAL
        elif value >= warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _trigger_alerts(self, health: SystemHealth):
        """Trigger health alerts."""
        for callback in self.alert_callbacks:
            try:
                self.executor.submit(callback, health)
            except Exception as e:
                logger.error(f"Error triggering health alert: {e}")
    
    def add_alert_callback(self, callback: Callable[[SystemHealth], None]):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_current_health(self) -> Optional[SystemHealth]:
        """Get current health status."""
        if not self.health_history:
            return self.check_system_health()
        return self.health_history[-1]
    
    def get_health_history(self, limit: Optional[int] = None) -> List[SystemHealth]:
        """Get health history."""
        history = list(self.health_history)
        if limit:
            return history[-limit:]
        return history
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary statistics."""
        if not self.health_history:
            return {"status": "no_data"}
        
        recent_health = list(self.health_history)[-10:]  # Last 10 checks
        
        # Calculate averages
        avg_cpu = sum(h.cpu_percent for h in recent_health) / len(recent_health)
        avg_memory = sum(h.memory_percent for h in recent_health) / len(recent_health)
        avg_disk = sum(h.disk_percent for h in recent_health) / len(recent_health)
        
        # Count status occurrences
        status_counts = {}
        for health in recent_health:
            status = health.overall_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        current = recent_health[-1] if recent_health else None
        
        return {
            "current_status": current.overall_status.value if current else "unknown",
            "uptime_hours": (time.time() - self._start_time) / 3600,
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "disk_percent": avg_disk
            },
            "status_distribution": status_counts,
            "total_checks": len(self.health_history),
            "monitoring_active": self.is_monitoring
        }
    
    def export_health_data(self, file_path: Path, format: str = "json"):
        """Export health data to file."""
        health_data = {
            "export_timestamp": time.time(),
            "summary": self.get_health_summary(),
            "history": [h.to_dict() for h in self.health_history]
        }
        
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
        """Cleanup system resources."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            memory = psutil.virtual_memory()
            logger.info(f"Memory cleanup - Available: {memory.available / (1024**3):.1f} GB")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")


class CarbonTrackingHealthMonitor(HealthMonitor):
    """Specialized health monitor for carbon tracking operations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Carbon-specific thresholds
        self.thresholds.update({
            "energy_rate_warning": 1000.0,  # Watts
            "energy_rate_critical": 2000.0,  # Watts
            "co2_rate_warning": 0.5,  # kg/hour
            "co2_rate_critical": 1.0,  # kg/hour
        })
    
    def check_carbon_health(self, current_power: float, 
                           current_co2_rate: float) -> List[HealthMetric]:
        """Check carbon-specific health metrics."""
        metrics = []
        
        # Power consumption rate
        power_status = self._evaluate_threshold(
            current_power,
            self.thresholds["energy_rate_warning"],
            self.thresholds["energy_rate_critical"]
        )
        metrics.append(HealthMetric(
            name="power_consumption",
            value=current_power,
            status=power_status,
            threshold_warning=self.thresholds["energy_rate_warning"],
            threshold_critical=self.thresholds["energy_rate_critical"],
            unit="W",
            description="Current power consumption rate"
        ))
        
        # CO2 emission rate
        co2_status = self._evaluate_threshold(
            current_co2_rate,
            self.thresholds["co2_rate_warning"],
            self.thresholds["co2_rate_critical"]
        )
        metrics.append(HealthMetric(
            name="co2_emission_rate",
            value=current_co2_rate,
            status=co2_status,
            threshold_warning=self.thresholds["co2_rate_warning"],
            threshold_critical=self.thresholds["co2_rate_critical"],
            unit="kg/hr",
            description="Current CO2 emission rate"
        ))
        
        return metrics


# Global health monitor instance
_health_monitor = CarbonTrackingHealthMonitor()


def get_health_monitor() -> CarbonTrackingHealthMonitor:
    """Get global health monitor instance."""
    return _health_monitor


def start_health_monitoring():
    """Start global health monitoring."""
    _health_monitor.start_monitoring()


def stop_health_monitoring():
    """Stop global health monitoring."""
    _health_monitor.stop_monitoring()


def get_system_health() -> Optional[SystemHealth]:
    """Get current system health."""
    return _health_monitor.get_current_health()


def health_alert_callback(health: SystemHealth):
    """Default health alert callback."""
    logger.warning(f"System health alert: {health.overall_status.value}")
    logger.warning(f"CPU: {health.cpu_percent:.1f}%, Memory: {health.memory_percent:.1f}%, Disk: {health.disk_percent:.1f}%")