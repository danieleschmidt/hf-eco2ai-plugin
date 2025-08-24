"""Health monitoring and alerting for carbon tracking system."""

import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import psutil

from .config import CarbonConfig
from .models import CarbonMetrics
from .utils import get_system_info

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    threshold_breached: Optional[str] = None
    recovery_suggestion: Optional[str] = None
    
    def is_healthy(self) -> bool:
        """Check if status is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    def needs_attention(self) -> bool:
        """Check if status needs attention."""
        return self.status in (HealthStatus.WARNING, HealthStatus.CRITICAL)


@dataclass
class SystemHealth:
    """Overall system health status."""
    
    overall_status: HealthStatus
    checks: List[HealthCheck]
    timestamp: float
    uptime_seconds: float
    
    def get_critical_checks(self) -> List[HealthCheck]:
        """Get all critical health checks."""
        return [c for c in self.checks if c.status == HealthStatus.CRITICAL]
    
    def get_warning_checks(self) -> List[HealthCheck]:
        """Get all warning health checks."""
        return [c for c in self.checks if c.status == HealthStatus.WARNING]
    
    def get_healthy_checks(self) -> List[HealthCheck]:
        """Get all healthy checks."""
        return [c for c in self.checks if c.status == HealthStatus.HEALTHY]
    
    def summary_text(self) -> str:
        """Generate health summary text."""
        status_icon = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.UNKNOWN: "❓"
        }
        
        summary = [
            f"{status_icon[self.overall_status]} System Status: {self.overall_status.value.upper()}",
            f"Uptime: {self.uptime_seconds/3600:.1f} hours",
            f"Checks: {len(self.get_healthy_checks())} healthy, "
            f"{len(self.get_warning_checks())} warnings, "
            f"{len(self.get_critical_checks())} critical",
            ""
        ]
        
        if self.get_critical_checks():
            summary.append("🔴 CRITICAL ISSUES:")
            for check in self.get_critical_checks():
                summary.append(f"  - {check.name}: {check.message}")
                if check.recovery_suggestion:
                    summary.append(f"    → {check.recovery_suggestion}")
            summary.append("")
        
        if self.get_warning_checks():
            summary.append("⚠️ WARNINGS:")
            for check in self.get_warning_checks():
                summary.append(f"  - {check.name}: {check.message}")
            summary.append("")
        
        return "\n".join(summary)


class HealthMonitor:
    """Monitor system health for carbon tracking."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize health monitor.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        self.start_time = time.time()
        self.last_check_time = 0
        self.check_interval = 30  # seconds
        
        # Health thresholds
        self.thresholds = {
            "cpu_percent_warning": 80,
            "cpu_percent_critical": 95,
            "memory_percent_warning": 85,
            "memory_percent_critical": 95,
            "disk_percent_warning": 85,
            "disk_percent_critical": 95,
            "gpu_temp_warning": 80,  # Celsius
            "gpu_temp_critical": 90,
            "response_time_warning": 5.0,  # seconds
            "response_time_critical": 10.0,
            "error_rate_warning": 0.05,  # 5%
            "error_rate_critical": 0.10,  # 10%
        }
        
        # Health history
        self.health_history: List[SystemHealth] = []
        self.max_history_size = 1000
        
        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._health_queue = queue.Queue()
        
        logger.info("Initialized health monitor")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Health monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="HealthMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self._monitoring_thread or not self._monitoring_thread.is_alive():
            logger.warning("Health monitoring not running")
            return
        
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=5)
        
        logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.wait(self.check_interval):
            try:
                health = self.check_health()
                self._health_queue.put(health)
                
                # Keep history size manageable
                self.health_history.append(health)
                if len(self.health_history) > self.max_history_size:
                    self.health_history = self.health_history[-self.max_history_size:]
                
                # Log critical issues
                if health.overall_status == HealthStatus.CRITICAL:
                    logger.critical(f"System health critical: {len(health.get_critical_checks())} issues")
                elif health.overall_status == HealthStatus.WARNING:
                    logger.warning(f"System health warning: {len(health.get_warning_checks())} issues")
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
    
    def check_health(self) -> SystemHealth:
        """Perform comprehensive health check.
        
        Returns:
            Current system health status
        """
        checks = []
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # System resource checks
        checks.extend(self._check_system_resources())
        
        # GPU health checks
        checks.extend(self._check_gpu_health())
        
        # Storage health checks
        checks.extend(self._check_storage_health())
        
        # Application health checks
        checks.extend(self._check_application_health())
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        
        health = SystemHealth(
            overall_status=overall_status,
            checks=checks,
            timestamp=current_time,
            uptime_seconds=uptime
        )
        
        self.last_check_time = current_time
        return health
    
    def _check_system_resources(self) -> List[HealthCheck]:
        """Check system resource usage."""
        checks = []
        
        # CPU check
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent >= self.thresholds["cpu_percent_critical"]:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
                suggestion = "Check for runaway processes or reduce workload"
            elif cpu_percent >= self.thresholds["cpu_percent_warning"]:
                status = HealthStatus.WARNING
                message = f"CPU usage high: {cpu_percent:.1f}%"
                suggestion = "Monitor CPU usage and consider workload optimization"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
                suggestion = None
            
            checks.append(HealthCheck(
                name="cpu_usage",
                status=status,
                message=message,
                timestamp=time.time(),
                metrics={"cpu_percent": cpu_percent},
                recovery_suggestion=suggestion
            ))
        
        except Exception as e:
            checks.append(HealthCheck(
                name="cpu_usage",
                status=HealthStatus.UNKNOWN,
                message=f"CPU check failed: {e}",
                timestamp=time.time()
            ))
        
        # Memory check
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent >= self.thresholds["memory_percent_critical"]:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {memory_percent:.1f}%"
                suggestion = "Free memory or increase available RAM"
            elif memory_percent >= self.thresholds["memory_percent_warning"]:
                status = HealthStatus.WARNING
                message = f"Memory usage high: {memory_percent:.1f}%"
                suggestion = "Monitor memory usage and optimize memory consumption"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
                suggestion = None
            
            checks.append(HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                timestamp=time.time(),
                metrics={
                    "memory_percent": memory_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                },
                recovery_suggestion=suggestion
            ))
        
        except Exception as e:
            checks.append(HealthCheck(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {e}",
                timestamp=time.time()
            ))
        
        return checks
    
    def _check_gpu_health(self) -> List[HealthCheck]:
        """Check GPU health if available."""
        checks = []
        
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Temperature check
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    if temp >= self.thresholds["gpu_temp_critical"]:
                        status = HealthStatus.CRITICAL
                        message = f"GPU {i} temperature critical: {temp}°C"
                        suggestion = "Check cooling system and reduce GPU workload"
                    elif temp >= self.thresholds["gpu_temp_warning"]:
                        status = HealthStatus.WARNING
                        message = f"GPU {i} temperature high: {temp}°C"
                        suggestion = "Monitor GPU temperature and improve cooling"
                    else:
                        status = HealthStatus.HEALTHY
                        message = f"GPU {i} temperature normal: {temp}°C"
                        suggestion = None
                    
                    checks.append(HealthCheck(
                        name=f"gpu_{i}_temperature",
                        status=status,
                        message=message,
                        timestamp=time.time(),
                        metrics={"temperature_celsius": temp},
                        recovery_suggestion=suggestion
                    ))
                
                except Exception as e:
                    checks.append(HealthCheck(
                        name=f"gpu_{i}_temperature",
                        status=HealthStatus.UNKNOWN,
                        message=f"GPU {i} temperature check failed: {e}",
                        timestamp=time.time()
                    ))
                
                # Memory check
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_percent = (mem_info.used / mem_info.total) * 100
                    
                    if mem_percent >= 95:
                        status = HealthStatus.WARNING
                        message = f"GPU {i} memory high: {mem_percent:.1f}%"
                        suggestion = "Consider reducing batch size or model size"
                    else:
                        status = HealthStatus.HEALTHY
                        message = f"GPU {i} memory usage: {mem_percent:.1f}%"
                        suggestion = None
                    
                    checks.append(HealthCheck(
                        name=f"gpu_{i}_memory",
                        status=status,
                        message=message,
                        timestamp=time.time(),
                        metrics={
                            "memory_percent": mem_percent,
                            "used_mb": mem_info.used / (1024**2),
                            "total_mb": mem_info.total / (1024**2)
                        },
                        recovery_suggestion=suggestion
                    ))
                
                except Exception as e:
                    checks.append(HealthCheck(
                        name=f"gpu_{i}_memory",
                        status=HealthStatus.UNKNOWN,
                        message=f"GPU {i} memory check failed: {e}",
                        timestamp=time.time()
                    ))
        
        except ImportError:
            checks.append(HealthCheck(
                name="gpu_availability",
                status=HealthStatus.WARNING,
                message="GPU monitoring not available (pynvml not installed)",
                timestamp=time.time(),
                recovery_suggestion="Install pynvml for GPU monitoring: pip install pynvml"
            ))
        
        except Exception as e:
            checks.append(HealthCheck(
                name="gpu_check",
                status=HealthStatus.UNKNOWN,
                message=f"GPU health check failed: {e}",
                timestamp=time.time()
            ))
        
        return checks
    
    def _check_storage_health(self) -> List[HealthCheck]:
        """Check storage health."""
        checks = []
        
        # Check disk usage for output directories
        paths_to_check = [
            Path(self.config.report_path).parent if self.config.save_report else None,
            Path(self.config.csv_path).parent if self.config.export_csv else None,
            Path.cwd()  # Current working directory
        ]
        
        for path in filter(None, paths_to_check):
            try:
                disk_usage = psutil.disk_usage(str(path))
                disk_percent = (disk_usage.used / disk_usage.total) * 100
                
                if disk_percent >= self.thresholds["disk_percent_critical"]:
                    status = HealthStatus.CRITICAL
                    message = f"Disk usage critical at {path}: {disk_percent:.1f}%"
                    suggestion = "Free disk space immediately"
                elif disk_percent >= self.thresholds["disk_percent_warning"]:
                    status = HealthStatus.WARNING
                    message = f"Disk usage high at {path}: {disk_percent:.1f}%"
                    suggestion = "Clean up disk space soon"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Disk usage normal at {path}: {disk_percent:.1f}%"
                    suggestion = None
                
                checks.append(HealthCheck(
                    name=f"disk_usage_{path.name}",
                    status=status,
                    message=message,
                    timestamp=time.time(),
                    metrics={
                        "disk_percent": disk_percent,
                        "free_gb": disk_usage.free / (1024**3),
                        "total_gb": disk_usage.total / (1024**3)
                    },
                    recovery_suggestion=suggestion
                ))
            
            except Exception as e:
                checks.append(HealthCheck(
                    name=f"disk_check_{path.name}",
                    status=HealthStatus.UNKNOWN,
                    message=f"Disk check failed for {path}: {e}",
                    timestamp=time.time()
                ))
        
        return checks
    
    def _check_application_health(self) -> List[HealthCheck]:
        """Check application-specific health."""
        checks = []
        
        # Check if monitoring components are responsive
        try:
            from .monitoring import EnergyTracker
            start_time = time.time()
            
            tracker = EnergyTracker(
                gpu_ids=self.config.gpu_ids,
                country=self.config.country,
                region=self.config.region
            )
            
            # Test basic functionality
            tracker.start_tracking()
            power, energy, co2 = tracker.get_current_consumption()
            tracker.stop_tracking()
            
            response_time = time.time() - start_time
            
            if response_time >= self.thresholds["response_time_critical"]:
                status = HealthStatus.CRITICAL
                message = f"Energy tracker response time critical: {response_time:.2f}s"
                suggestion = "Check system resources and restart if needed"
            elif response_time >= self.thresholds["response_time_warning"]:
                status = HealthStatus.WARNING
                message = f"Energy tracker response slow: {response_time:.2f}s"
                suggestion = "Monitor system performance"
            else:
                status = HealthStatus.HEALTHY
                message = f"Energy tracker responsive: {response_time:.2f}s"
                suggestion = None
            
            checks.append(HealthCheck(
                name="energy_tracker_response",
                status=status,
                message=message,
                timestamp=time.time(),
                metrics={
                    "response_time_seconds": response_time,
                    "power_watts": power,
                    "energy_kwh": energy,
                    "co2_kg": co2
                },
                recovery_suggestion=suggestion
            ))
        
        except Exception as e:
            checks.append(HealthCheck(
                name="energy_tracker_check",
                status=HealthStatus.CRITICAL,
                message=f"Energy tracker failed: {e}",
                timestamp=time.time(),
                recovery_suggestion="Check dependencies and configuration"
            ))
        
        return checks
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system status from individual checks."""
        if any(c.status == HealthStatus.CRITICAL for c in checks):
            return HealthStatus.CRITICAL
        elif any(c.status == HealthStatus.WARNING for c in checks):
            return HealthStatus.WARNING
        elif any(c.status == HealthStatus.UNKNOWN for c in checks):
            return HealthStatus.WARNING  # Treat unknown as warning
        else:
            return HealthStatus.HEALTHY
    
    def get_latest_health(self) -> Optional[SystemHealth]:
        """Get latest health status.
        
        Returns:
            Latest health status or None if no checks performed
        """
        return self.health_history[-1] if self.health_history else None
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get health trends over time.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary of metric trends
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_health = [h for h in self.health_history if h.timestamp >= cutoff_time]
        
        trends = {}
        
        for health in recent_health:
            for check in health.checks:
                if check.metrics:
                    for metric_name, metric_value in check.metrics.items():
                        if isinstance(metric_value, (int, float)):
                            key = f"{check.name}_{metric_name}"
                            if key not in trends:
                                trends[key] = []
                            trends[key].append(metric_value)
        
        return trends
    
    def export_health_report(self, output_path: str):
        """Export health report to file.
        
        Args:
            output_path: Output file path
        """
        latest_health = self.get_latest_health()
        trends = self.get_health_trends()
        
        report_data = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "latest_health": {
                "overall_status": latest_health.overall_status.value if latest_health else "unknown",
                "checks": [
                    {
                        "name": c.name,
                        "status": c.status.value,
                        "message": c.message,
                        "timestamp": c.timestamp,
                        "metrics": c.metrics
                    }
                    for c in latest_health.checks
                ] if latest_health else []
            },
            "health_history_count": len(self.health_history),
            "trends_24h": trends,
            "thresholds": self.thresholds
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Health report exported to {output_path}")


class AlertManager:
    """Manage alerts for carbon tracking system."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize alert manager.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        self.alert_handlers: List[Callable[[HealthCheck], None]] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_cooldown = 300  # 5 minutes between similar alerts
        self.last_alerts: Dict[str, float] = {}
        
        logger.info("Initialized alert manager")
    
    def add_alert_handler(self, handler: Callable[[HealthCheck], None]):
        """Add an alert handler function.
        
        Args:
            handler: Function to handle alerts
        """
        self.alert_handlers.append(handler)
    
    def process_health_check(self, health: SystemHealth):
        """Process health check and trigger alerts if needed.
        
        Args:
            health: System health status
        """
        current_time = time.time()
        
        for check in health.checks:
            if check.needs_attention():
                alert_key = f"{check.name}_{check.status.value}"
                
                # Check cooldown
                last_alert_time = self.last_alerts.get(alert_key, 0)
                if current_time - last_alert_time < self.alert_cooldown:
                    continue
                
                # Trigger alert
                self._trigger_alert(check)
                self.last_alerts[alert_key] = current_time
    
    def _trigger_alert(self, check: HealthCheck):
        """Trigger alert for a health check.
        
        Args:
            check: Health check that triggered the alert
        """
        alert_data = {
            "timestamp": time.time(),
            "check_name": check.name,
            "status": check.status.value,
            "message": check.message,
            "metrics": check.metrics,
            "recovery_suggestion": check.recovery_suggestion
        }
        
        self.alert_history.append(alert_data)
        
        # Call all alert handlers
        for handler in self.alert_handlers:
            try:
                handler(check)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"ALERT triggered: {check.name} - {check.message}")


# Example alert handlers
def log_alert_handler(check: HealthCheck):
    """Simple alert handler that logs to system logger."""
    level = logging.CRITICAL if check.status == HealthStatus.CRITICAL else logging.WARNING
    logger.log(level, f"HEALTH ALERT: {check.name} - {check.message}")

def email_alert_handler(check: HealthCheck, smtp_config: Dict[str, str]):
    """Email alert handler (example implementation)."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        
        subject = f"Carbon Tracking Alert: {check.status.value.upper()}"
        body = f"""
        Health Check Alert
        
        Component: {check.name}
        Status: {check.status.value}
        Message: {check.message}
        Time: {time.ctime(check.timestamp)}
        
        Recovery Suggestion: {check.recovery_suggestion or 'None provided'}
        
        Metrics: {check.metrics}
        """
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = smtp_config['from']
        msg['To'] = smtp_config['to']
        
        with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
        
        logger.info(f"Alert email sent for {check.name}")
    
    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")


# Global health monitor instance
_health_monitor = HealthMonitor()
_alert_manager = AlertManager()

def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    return _health_monitor

def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    return _alert_manager