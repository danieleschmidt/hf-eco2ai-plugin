"""Enterprise-grade monitoring and observability for carbon tracking."""

import asyncio
import time
import json
import threading
import queue
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server, CollectorRegistry
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    severity: str  # info, warning, error, critical
    description: str
    cooldown_minutes: int = 15
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self, current_value: float) -> bool:
        """Check if alert should trigger."""
        if not self.enabled:
            return False
            
        # Check cooldown period
        if self.last_triggered:
            cooldown_expired = datetime.now() > self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if not cooldown_expired:
                return False
        
        # Evaluate condition
        if self.operator == '>':
            return current_value > self.threshold
        elif self.operator == '<':
            return current_value < self.threshold
        elif self.operator == '>=':
            return current_value >= self.threshold
        elif self.operator == '<=':
            return current_value <= self.threshold
        elif self.operator == '==':
            return abs(current_value - self.threshold) < 1e-6
        elif self.operator == '!=':
            return abs(current_value - self.threshold) >= 1e-6
        return False


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot."""
    timestamp: datetime
    metrics: Dict[str, float]
    labels: Dict[str, str]
    session_id: str


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_check_results: Dict[str, bool] = {}
        self.check_interval = 60  # seconds
        self.running = False
        self._thread: Optional[threading.Thread] = None
    
    def register_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def start(self) -> None:
        """Start the health monitoring thread."""
        if self.running:
            return
            
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Health monitoring started")
    
    def stop(self) -> None:
        """Stop health monitoring."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self.run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def run_health_checks(self) -> Dict[str, bool]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
                self.last_check_results[name] = result
                
                if not result:
                    logger.warning(f"Health check failed: {name}")
                    
            except Exception as e:
                logger.error(f"Health check '{name}' raised exception: {e}")
                results[name] = False
                self.last_check_results[name] = False
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "overall_healthy": all(self.last_check_results.values()),
            "checks": self.last_check_results.copy(),
            "last_check_time": datetime.now().isoformat(),
            "monitoring_active": self.running
        }


class MetricsCollector:
    """Advanced metrics collection and aggregation."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history: List[MetricSnapshot] = []
        self.aggregated_metrics: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
        
        # Prometheus integration
        self.prometheus_registry = None
        self.prometheus_metrics: Dict[str, Any] = {}
        
        if HAS_PROMETHEUS:
            self._setup_prometheus()
    
    def _setup_prometheus(self) -> None:
        """Set up Prometheus metrics."""
        self.prometheus_registry = CollectorRegistry()
        
        # Core carbon tracking metrics
        self.prometheus_metrics = {
            'carbon_emissions_total': Counter(
                'hf_eco2ai_carbon_emissions_kg_total',
                'Total carbon emissions in kg CO2',
                ['session_id', 'model_name', 'region'],
                registry=self.prometheus_registry
            ),
            'energy_consumption_total': Counter(
                'hf_eco2ai_energy_consumption_kwh_total',
                'Total energy consumption in kWh',
                ['session_id', 'gpu_id', 'region'],
                registry=self.prometheus_registry
            ),
            'training_efficiency': Gauge(
                'hf_eco2ai_training_efficiency_samples_per_kwh',
                'Training efficiency in samples per kWh',
                ['session_id', 'model_name'],
                registry=self.prometheus_registry
            ),
            'gpu_utilization': Gauge(
                'hf_eco2ai_gpu_utilization_percent',
                'GPU utilization percentage',
                ['gpu_id', 'session_id'],
                registry=self.prometheus_registry
            ),
            'carbon_intensity': Gauge(
                'hf_eco2ai_grid_carbon_intensity_g_per_kwh',
                'Grid carbon intensity in g CO2/kWh',
                ['region', 'provider'],
                registry=self.prometheus_registry
            ),
            'training_duration': Histogram(
                'hf_eco2ai_training_duration_seconds',
                'Training duration in seconds',
                ['model_name', 'dataset_size'],
                buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400],
                registry=self.prometheus_registry
            )
        }
        
        logger.info("Prometheus metrics initialized")
    
    def record_metric(
        self, 
        metrics: Dict[str, float], 
        labels: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Record a metric snapshot."""
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            metrics=metrics.copy(),
            labels=labels or {},
            session_id=session_id or f"session_{int(time.time())}"
        )
        
        with self._lock:
            self.metrics_history.append(snapshot)
            
            # Trim history if needed
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history:]
            
            # Update Prometheus metrics
            if HAS_PROMETHEUS and self.prometheus_metrics:
                self._update_prometheus_metrics(snapshot)
        
        # Update aggregated metrics
        self._update_aggregations(metrics)
    
    def _update_prometheus_metrics(self, snapshot: MetricSnapshot) -> None:
        """Update Prometheus metrics from snapshot."""
        try:
            labels = snapshot.labels
            session_id = snapshot.session_id
            
            # Update counters and gauges
            if 'co2_kg' in snapshot.metrics:
                self.prometheus_metrics['carbon_emissions_total'].labels(
                    session_id=session_id,
                    model_name=labels.get('model_name', 'unknown'),
                    region=labels.get('region', 'unknown')
                ).inc(snapshot.metrics['co2_kg'])
            
            if 'energy_kwh' in snapshot.metrics:
                self.prometheus_metrics['energy_consumption_total'].labels(
                    session_id=session_id,
                    gpu_id=labels.get('gpu_id', '0'),
                    region=labels.get('region', 'unknown')
                ).inc(snapshot.metrics['energy_kwh'])
            
            if 'samples_per_kwh' in snapshot.metrics:
                self.prometheus_metrics['training_efficiency'].labels(
                    session_id=session_id,
                    model_name=labels.get('model_name', 'unknown')
                ).set(snapshot.metrics['samples_per_kwh'])
            
            if 'gpu_utilization' in snapshot.metrics:
                self.prometheus_metrics['gpu_utilization'].labels(
                    gpu_id=labels.get('gpu_id', '0'),
                    session_id=session_id
                ).set(snapshot.metrics['gpu_utilization'])
            
            if 'grid_intensity' in snapshot.metrics:
                self.prometheus_metrics['carbon_intensity'].labels(
                    region=labels.get('region', 'unknown'),
                    provider=labels.get('provider', 'unknown')
                ).set(snapshot.metrics['grid_intensity'])
            
            if 'training_duration' in snapshot.metrics:
                self.prometheus_metrics['training_duration'].labels(
                    model_name=labels.get('model_name', 'unknown'),
                    dataset_size=labels.get('dataset_size', 'unknown')
                ).observe(snapshot.metrics['training_duration'])
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    def _update_aggregations(self, metrics: Dict[str, float]) -> None:
        """Update aggregated metrics."""
        for metric_name, value in metrics.items():
            if metric_name not in self.aggregated_metrics:
                self.aggregated_metrics[metric_name] = {
                    'count': 0,
                    'sum': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'avg': 0.0
                }
            
            agg = self.aggregated_metrics[metric_name]
            agg['count'] += 1
            agg['sum'] += value
            agg['min'] = min(agg['min'], value)
            agg['max'] = max(agg['max'], value)
            agg['avg'] = agg['sum'] / agg['count']
    
    def get_metrics_summary(self, time_window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """Get metrics summary for a time window."""
        cutoff_time = None
        if time_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            if cutoff_time:
                relevant_snapshots = [s for s in self.metrics_history if s.timestamp >= cutoff_time]
            else:
                relevant_snapshots = self.metrics_history.copy()
        
        if not relevant_snapshots:
            return {"summary": "No metrics available for the specified time window"}
        
        # Calculate aggregations for the time window
        window_aggregations = {}
        for snapshot in relevant_snapshots:
            for metric_name, value in snapshot.metrics.items():
                if metric_name not in window_aggregations:
                    window_aggregations[metric_name] = []
                window_aggregations[metric_name].append(value)
        
        summary = {
            "time_window": f"Last {time_window_minutes} minutes" if time_window_minutes else "All time",
            "snapshots_count": len(relevant_snapshots),
            "metrics": {}
        }
        
        for metric_name, values in window_aggregations.items():
            if values:
                summary["metrics"][metric_name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": np.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "std": np.std(values) if len(values) > 1 else 0.0
                }
        
        return summary
    
    def export_metrics_csv(self, filepath: Path, time_window_minutes: Optional[int] = None) -> None:
        """Export metrics to CSV file."""
        cutoff_time = None
        if time_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            if cutoff_time:
                relevant_snapshots = [s for s in self.metrics_history if s.timestamp >= cutoff_time]
            else:
                relevant_snapshots = self.metrics_history.copy()
        
        if not relevant_snapshots:
            logger.warning("No metrics to export")
            return
        
        # Get all unique metric names and label names
        all_metrics = set()
        all_labels = set()
        for snapshot in relevant_snapshots:
            all_metrics.update(snapshot.metrics.keys())
            all_labels.update(snapshot.labels.keys())
        
        # Create CSV content
        header = ['timestamp', 'session_id'] + list(all_labels) + list(all_metrics)
        rows = []
        
        for snapshot in relevant_snapshots:
            row = [
                snapshot.timestamp.isoformat(),
                snapshot.session_id
            ]
            
            # Add label values
            for label in all_labels:
                row.append(snapshot.labels.get(label, ''))
            
            # Add metric values
            for metric in all_metrics:
                row.append(snapshot.metrics.get(metric, ''))
            
            rows.append(row)
        
        # Write CSV file
        with open(filepath, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        
        logger.info(f"Exported {len(rows)} metric snapshots to {filepath}")


class AlertManager:
    """Advanced alerting system."""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.metrics_collector: Optional[MetricsCollector] = None
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Set up default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_carbon_emissions",
                name="High Carbon Emissions",
                metric_name="co2_kg",
                operator=">",
                threshold=10.0,  # 10 kg CO2
                severity="warning",
                description="Carbon emissions exceeded 10 kg CO2",
                cooldown_minutes=30
            ),
            AlertRule(
                rule_id="very_high_carbon_emissions",
                name="Very High Carbon Emissions",
                metric_name="co2_kg",
                operator=">",
                threshold=50.0,  # 50 kg CO2
                severity="critical",
                description="Carbon emissions exceeded 50 kg CO2 - immediate attention required",
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id="low_training_efficiency",
                name="Low Training Efficiency",
                metric_name="samples_per_kwh",
                operator="<",
                threshold=1000.0,  # 1000 samples per kWh
                severity="warning",
                description="Training efficiency is below 1000 samples/kWh",
                cooldown_minutes=60
            ),
            AlertRule(
                rule_id="high_gpu_temperature",
                name="High GPU Temperature",
                metric_name="gpu_temperature",
                operator=">",
                threshold=80.0,  # 80°C
                severity="error",
                description="GPU temperature exceeded 80°C",
                cooldown_minutes=10
            ),
            AlertRule(
                rule_id="carbon_budget_warning",
                name="Carbon Budget Warning",
                metric_name="budget_utilization",
                operator=">",
                threshold=80.0,  # 80% of budget
                severity="warning",
                description="Carbon budget utilization exceeded 80%",
                cooldown_minutes=120
            ),
            AlertRule(
                rule_id="carbon_budget_critical",
                name="Carbon Budget Critical",
                metric_name="budget_utilization", 
                operator=">",
                threshold=95.0,  # 95% of budget
                severity="critical",
                description="Carbon budget utilization exceeded 95% - stop training recommended",
                cooldown_minutes=60
            )
        ]
        
        self.rules.extend(default_rules)
        logger.info(f"Initialized {len(default_rules)} default alert rules")
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        removed = len(self.rules) < original_count
        
        if removed:
            logger.info(f"Removed alert rule: {rule_id}")
        
        return removed
    
    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info("Added alert handler")
    
    def set_metrics_collector(self, collector: MetricsCollector) -> None:
        """Set the metrics collector for automatic alerting."""
        self.metrics_collector = collector
    
    def evaluate_metrics(self, metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate metrics against alert rules."""
        triggered_alerts = []
        
        for rule in self.rules:
            if rule.metric_name in metrics:
                current_value = metrics[rule.metric_name]
                
                if rule.should_trigger(current_value):
                    alert = Alert(
                        alert_id=f"alert_{int(time.time())}_{rule.rule_id}",
                        rule_id=rule.rule_id,
                        timestamp=datetime.now(),
                        metric_name=rule.metric_name,
                        current_value=current_value,
                        threshold=rule.threshold,
                        severity=rule.severity,
                        message=f"{rule.description} - Current: {current_value:.2f}, Threshold: {rule.threshold:.2f}"
                    )
                    
                    triggered_alerts.append(alert)
                    self.active_alerts.append(alert)
                    rule.last_triggered = datetime.now()
                    
                    # Call alert handlers
                    for handler in self.alert_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logger.error(f"Alert handler error: {e}")
                    
                    logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")
        
        return triggered_alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)
        if alert:
            alert.acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)
        if alert:
            alert.resolved = True
            alert.resolved_at = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        active = [a for a in self.active_alerts if not a.resolved]
        
        if severity_filter:
            active = [a for a in active if a.severity == severity_filter]
        
        return active
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        active = self.get_active_alerts()
        
        severity_counts = {}
        for alert in active:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        return {
            "total_active_alerts": len(active),
            "severity_breakdown": severity_counts,
            "unacknowledged_alerts": len([a for a in active if not a.acknowledged]),
            "oldest_alert": min([a.timestamp for a in active], default=None),
            "most_recent_alert": max([a.timestamp for a in active], default=None)
        }


class EnterpriseMonitor:
    """Main enterprise monitoring orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enterprise monitoring."""
        self.config = config or {}
        
        # Components
        self.health_checker = HealthChecker()
        self.metrics_collector = MetricsCollector(
            max_history=self.config.get('max_metrics_history', 10000)
        )
        self.alert_manager = AlertManager()
        
        # Connect components
        self.alert_manager.set_metrics_collector(self.metrics_collector)
        
        # Structured logging
        if HAS_STRUCTLOG:
            self.logger = structlog.get_logger()
        else:
            self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.monitoring_active = False
        self.session_id = f"monitor_{int(time.time())}"
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Set up default alert handlers
        self._setup_default_alert_handlers()
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks."""
        
        def check_metrics_collection():
            """Check if metrics collection is working."""
            return len(self.metrics_collector.metrics_history) >= 0
        
        def check_alert_system():
            """Check if alert system is responsive."""
            return len(self.alert_manager.rules) > 0
        
        def check_disk_space():
            """Check available disk space."""
            try:
                import shutil
                _, _, free = shutil.disk_usage("/")
                free_gb = free / (1024**3)
                return free_gb > 1.0  # At least 1GB free
            except:
                return True  # Assume OK if can't check
        
        self.health_checker.register_check("metrics_collection", check_metrics_collection)
        self.health_checker.register_check("alert_system", check_alert_system)
        self.health_checker.register_check("disk_space", check_disk_space)
    
    def _setup_default_alert_handlers(self) -> None:
        """Set up default alert handlers."""
        
        def log_alert_handler(alert: Alert) -> None:
            """Log alerts to structured logging."""
            if HAS_STRUCTLOG:
                self.logger.warning(
                    "Carbon tracking alert",
                    alert_id=alert.alert_id,
                    severity=alert.severity,
                    metric_name=alert.metric_name,
                    current_value=alert.current_value,
                    threshold=alert.threshold,
                    message=alert.message
                )
            else:
                self.logger.warning(f"Alert: {alert.message}")
        
        self.alert_manager.add_handler(log_alert_handler)
    
    def start_monitoring(self, prometheus_port: Optional[int] = None) -> None:
        """Start enterprise monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start health monitoring
        self.health_checker.start()
        
        # Start Prometheus server if available and requested
        if HAS_PROMETHEUS and prometheus_port and self.metrics_collector.prometheus_registry:
            try:
                start_http_server(
                    prometheus_port, 
                    registry=self.metrics_collector.prometheus_registry
                )
                logger.info(f"Prometheus metrics server started on port {prometheus_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")
        
        logger.info("Enterprise monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop enterprise monitoring."""
        self.monitoring_active = False
        self.health_checker.stop()
        logger.info("Enterprise monitoring stopped")
    
    def record_training_metrics(
        self, 
        metrics: Dict[str, float], 
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record training metrics and evaluate alerts."""
        if not self.monitoring_active:
            return
        
        # Record metrics
        self.metrics_collector.record_metric(
            metrics, 
            labels or {}, 
            self.session_id
        )
        
        # Evaluate alerts
        self.alert_manager.evaluate_metrics(metrics)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        health_status = self.health_checker.get_health_status()
        metrics_summary = self.metrics_collector.get_metrics_summary(time_window_minutes=60)
        alert_summary = self.alert_manager.get_alert_summary()
        
        return {
            "monitoring_status": {
                "active": self.monitoring_active,
                "session_id": self.session_id,
                "uptime_minutes": (datetime.now() - datetime.fromisoformat(self.session_id.split('_')[1])).seconds // 60 if '_' in self.session_id else 0
            },
            "health_status": health_status,
            "metrics_summary": metrics_summary,
            "alert_summary": alert_summary,
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged
                }
                for alert in self.alert_manager.get_active_alerts()
            ]
        }
    
    def export_monitoring_report(self, filepath: Path) -> None:
        """Export comprehensive monitoring report."""
        dashboard = self.get_monitoring_dashboard()
        
        report = {
            "monitoring_report": dashboard,
            "generated_at": datetime.now().isoformat(),
            "configuration": self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report exported to {filepath}")


# Default monitoring instance
_default_monitor: Optional[EnterpriseMonitor] = None


def get_enterprise_monitor(config: Optional[Dict[str, Any]] = None) -> EnterpriseMonitor:
    """Get or create the default enterprise monitor."""
    global _default_monitor
    
    if _default_monitor is None:
        _default_monitor = EnterpriseMonitor(config)
    
    return _default_monitor


def start_enterprise_monitoring(
    prometheus_port: Optional[int] = 9091,
    config: Optional[Dict[str, Any]] = None
) -> EnterpriseMonitor:
    """Start enterprise monitoring with default configuration."""
    monitor = get_enterprise_monitor(config)
    monitor.start_monitoring(prometheus_port)
    return monitor