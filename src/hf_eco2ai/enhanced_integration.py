"""
Enhanced Integration Module for HF Eco2AI Carbon Tracking System
Integrates all enhanced security, monitoring, fault tolerance, error handling, and compliance components
for seamless operation in production environments.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import uuid

# Import enhanced components
from .security_enhanced import EnhancedSecurityManager
from .health_monitor_enhanced import EnhancedHealthMonitor
from .fault_tolerance_enhanced import EnhancedFaultToleranceManager
from .error_handling_enhanced import EnhancedErrorHandler
from .compliance import ComplianceFramework, AuditEventType, ComplianceLevel

# Import existing core components
from .callback import Eco2AICallback
from .config import CarbonConfig
from .monitoring import EnergyTracker, GPUMonitor
from .exporters import PrometheusExporter, ReportExporter


class SystemMode(Enum):
    """System operation modes"""
    NORMAL = "normal"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class IntegrationStatus(Enum):
    """Integration component status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class SystemStatus:
    """Overall system status"""
    timestamp: str
    mode: SystemMode
    components: Dict[str, IntegrationStatus]
    active_alerts: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    compliance_status: Dict[str, Any]
    risk_level: str


class EnhancedIntegrationManager:
    """Main integration manager that coordinates all enhanced components"""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        self.config = config or CarbonConfig()
        self._running = False
        self._monitor_thread = None
        self._initialization_complete = False
        
        # Component initialization
        self.security_manager = None
        self.health_monitor = None
        self.fault_tolerance = None
        self.error_handler = None
        self.compliance_framework = None
        
        # Core components
        self.energy_tracker = None
        self.gpu_monitor = None
        self.prometheus_exporter = None
        self.report_exporter = None
        
        # System state
        self.system_mode = SystemMode.NORMAL
        self.component_status = {}
        self.active_alerts = []
        self.startup_time = datetime.now()
        
        # Event handlers
        self._event_handlers = {
            'security_threat': [],
            'health_alert': [],
            'fault_detected': [],
            'error_occurred': [],
            'compliance_violation': [],
            'system_mode_change': []
        }
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup integrated logging"""
        self.logger = logging.getLogger("enhanced_integration")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def initialize_components(self) -> bool:
        """Initialize all enhanced components with proper error handling"""
        try:
            self.logger.info("Starting enhanced component initialization...")
            
            # Initialize compliance framework first (needed for audit logging)
            self.compliance_framework = ComplianceFramework()
            self.compliance_framework.start_compliance_monitoring()
            self.component_status['compliance'] = IntegrationStatus.HEALTHY
            
            # Log initialization start
            self.compliance_framework.log_audit_event(
                event_type=AuditEventType.SYSTEM_CONFIGURATION,
                user_id="system",
                resource="integration_manager",
                action="initialize_components",
                details={"initialization_start": datetime.now().isoformat()},
                compliance_level=ComplianceLevel.ISO27001,
                session_id="init_session",
                ip_address="localhost",
                user_agent="integration_manager",
                risk_level="low",
                data_classification="system",
                outcome="in_progress"
            )
            
            # Initialize error handler early
            self.error_handler = EnhancedErrorHandler()
            self.component_status['error_handler'] = IntegrationStatus.HEALTHY
            
            # Initialize security manager
            self.security_manager = EnhancedSecurityManager()
            self.component_status['security'] = IntegrationStatus.HEALTHY
            
            # Initialize health monitor
            self.health_monitor = EnhancedHealthMonitor()
            self.health_monitor.start_monitoring()
            self.component_status['health_monitor'] = IntegrationStatus.HEALTHY
            
            # Initialize fault tolerance manager
            self.fault_tolerance = EnhancedFaultToleranceManager()
            self.component_status['fault_tolerance'] = IntegrationStatus.HEALTHY
            
            # Initialize core carbon tracking components
            self.energy_tracker = EnergyTracker()
            self.gpu_monitor = GPUMonitor()
            self.prometheus_exporter = PrometheusExporter()
            self.report_exporter = ReportExporter()
            self.component_status['carbon_tracking'] = IntegrationStatus.HEALTHY
            
            # Setup component integrations
            self._setup_component_integrations()
            
            # Register event handlers
            self._register_event_handlers()
            
            self._initialization_complete = True
            
            # Log successful initialization
            self.compliance_framework.log_audit_event(
                event_type=AuditEventType.SYSTEM_CONFIGURATION,
                user_id="system",
                resource="integration_manager",
                action="initialize_components",
                details={
                    "initialization_complete": datetime.now().isoformat(),
                    "components_initialized": list(self.component_status.keys()),
                    "initialization_duration": (datetime.now() - self.startup_time).total_seconds()
                },
                compliance_level=ComplianceLevel.ISO27001,
                session_id="init_session",
                ip_address="localhost",
                user_agent="integration_manager",
                risk_level="low",
                data_classification="system",
                outcome="success"
            )
            
            self.logger.info("Enhanced component initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during component initialization: {e}")
            
            # Log initialization failure
            if self.compliance_framework:
                self.compliance_framework.log_audit_event(
                    event_type=AuditEventType.SYSTEM_CONFIGURATION,
                    user_id="system",
                    resource="integration_manager",
                    action="initialize_components",
                    details={
                        "initialization_failed": datetime.now().isoformat(),
                        "error": str(e),
                        "partial_components": list(self.component_status.keys())
                    },
                    compliance_level=ComplianceLevel.ISO27001,
                    session_id="init_session",
                    ip_address="localhost",
                    user_agent="integration_manager",
                    risk_level="high",
                    data_classification="system",
                    outcome="failure"
                )
                
            return False
            
    def _setup_component_integrations(self):
        """Setup cross-component integrations"""
        try:
            # Integrate error handler with all components
            if self.error_handler and self.health_monitor:
                # Connect health alerts to error handler
                def handle_health_alert(alert):
                    self.error_handler.handle_error(
                        Exception(f"Health alert: {alert.get('message', 'Unknown health issue')}"),
                        context={
                            "component": "health_monitor",
                            "alert_type": alert.get("alert_type"),
                            "severity": alert.get("severity"),
                            "timestamp": alert.get("timestamp")
                        }
                    )
                self.health_monitor.alert_manager.add_handler("integration", handle_health_alert)
                
            # Integrate security events with compliance logging
            if self.security_manager and self.compliance_framework:
                def handle_security_event(event_type, details):
                    self.compliance_framework.log_audit_event(
                        event_type=AuditEventType.SECURITY_EVENT,
                        user_id=details.get("user_id", "unknown"),
                        resource=details.get("resource", "unknown"),
                        action=event_type,
                        details=details,
                        compliance_level=ComplianceLevel.GDPR,
                        session_id=details.get("session_id", "unknown"),
                        ip_address=details.get("ip_address", "unknown"),
                        user_agent=details.get("user_agent", "unknown"),
                        risk_level=details.get("risk_level", "medium"),
                        data_classification="security",
                        outcome=details.get("outcome", "detected")
                    )
                # Note: Would need to modify security_manager to support callbacks
                
            # Integrate fault tolerance with health monitoring
            if self.fault_tolerance and self.health_monitor:
                # Configure circuit breakers based on health status
                def update_circuit_breaker_thresholds():
                    health_status = self.health_monitor.get_system_health()
                    if health_status.get("overall_health", 0) < 0.7:
                        # Reduce thresholds during poor health
                        pass  # Implementation would depend on circuit breaker API
                        
            self.logger.info("Component integrations setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up component integrations: {e}")
            
    def _register_event_handlers(self):
        """Register event handlers for cross-component communication"""
        
        # Security threat handler
        def handle_security_threat(threat_info):
            self.active_alerts.append({
                "type": "security_threat",
                "timestamp": datetime.now().isoformat(),
                "details": threat_info,
                "severity": threat_info.get("severity", "medium")
            })
            
            # Escalate to emergency mode if critical threat
            if threat_info.get("severity") == "critical":
                self.set_system_mode(SystemMode.EMERGENCY, "Critical security threat detected")
                
        self.register_event_handler('security_threat', handle_security_threat)
        
        # Health alert handler
        def handle_health_alert(alert_info):
            self.active_alerts.append({
                "type": "health_alert",
                "timestamp": datetime.now().isoformat(),
                "details": alert_info,
                "severity": alert_info.get("severity", "medium")
            })
            
            # Switch to degraded mode if critical health issue
            if alert_info.get("severity") == "critical":
                self.set_system_mode(SystemMode.DEGRADED, "Critical health issue detected")
                
        self.register_event_handler('health_alert', handle_health_alert)
        
        # Fault detection handler
        def handle_fault_detected(fault_info):
            self.active_alerts.append({
                "type": "fault_detected",
                "timestamp": datetime.now().isoformat(),
                "details": fault_info,
                "severity": fault_info.get("severity", "medium")
            })
            
            # Trigger recovery procedures
            if self.fault_tolerance:
                recovery_success = self.fault_tolerance.handle_fault(fault_info)
                if not recovery_success:
                    self.set_system_mode(SystemMode.DEGRADED, "Fault recovery failed")
                    
        self.register_event_handler('fault_detected', handle_fault_detected)
        
        # Error occurrence handler
        def handle_error_occurred(error_info):
            self.active_alerts.append({
                "type": "error_occurred",
                "timestamp": datetime.now().isoformat(),
                "details": error_info,
                "severity": error_info.get("severity", "medium")
            })
            
        self.register_event_handler('error_occurred', handle_error_occurred)
        
        # Compliance violation handler
        def handle_compliance_violation(violation_info):
            self.active_alerts.append({
                "type": "compliance_violation",
                "timestamp": datetime.now().isoformat(),
                "details": violation_info,
                "severity": "high"  # Compliance violations are always high severity
            })
            
            # Log violation
            if self.compliance_framework:
                self.compliance_framework.log_audit_event(
                    event_type=AuditEventType.COMPLIANCE_CHECK,
                    user_id="system",
                    resource="compliance_checker",
                    action="violation_detected",
                    details=violation_info,
                    compliance_level=ComplianceLevel.GDPR,
                    session_id="monitoring",
                    ip_address="localhost",
                    user_agent="integration_manager",
                    risk_level="high",
                    data_classification="compliance",
                    outcome="violation"
                )
                
        self.register_event_handler('compliance_violation', handle_compliance_violation)
        
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler for specific event types"""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].append(handler)
            
    def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event to all registered handlers"""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")
                    
    def set_system_mode(self, mode: SystemMode, reason: str = ""):
        """Change system operation mode"""
        previous_mode = self.system_mode
        self.system_mode = mode
        
        mode_change_info = {
            "previous_mode": previous_mode.value,
            "new_mode": mode.value,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log mode change
        self.logger.info(f"System mode changed from {previous_mode.value} to {mode.value}: {reason}")
        
        # Emit mode change event
        self.emit_event('system_mode_change', mode_change_info)
        
        # Log in compliance system
        if self.compliance_framework:
            self.compliance_framework.log_audit_event(
                event_type=AuditEventType.SYSTEM_CONFIGURATION,
                user_id="system",
                resource="integration_manager",
                action="mode_change",
                details=mode_change_info,
                compliance_level=ComplianceLevel.ISO27001,
                session_id="monitoring",
                ip_address="localhost",
                user_agent="integration_manager",
                risk_level="medium" if mode != SystemMode.EMERGENCY else "high",
                data_classification="system",
                outcome="success"
            )
            
        # Adjust component behavior based on mode
        self._adjust_components_for_mode(mode)
        
    def _adjust_components_for_mode(self, mode: SystemMode):
        """Adjust component behavior based on system mode"""
        try:
            if mode == SystemMode.DEGRADED:
                # Reduce monitoring frequency, enable fallback mechanisms
                if self.health_monitor:
                    self.health_monitor.set_monitoring_interval(30)  # Longer intervals
                if self.fault_tolerance:
                    self.fault_tolerance.enable_graceful_degradation()
                    
            elif mode == SystemMode.EMERGENCY:
                # Maximum alerting, minimal functionality
                if self.health_monitor:
                    self.health_monitor.set_monitoring_interval(5)  # Frequent monitoring
                # Could disable non-essential features
                
            elif mode == SystemMode.MAINTENANCE:
                # Reduce logging, pause some monitors
                if self.health_monitor:
                    self.health_monitor.pause_non_critical_monitors()
                    
            elif mode == SystemMode.NORMAL:
                # Restore normal operation
                if self.health_monitor:
                    self.health_monitor.set_monitoring_interval(10)  # Normal intervals
                    self.health_monitor.resume_all_monitors()
                if self.fault_tolerance:
                    self.fault_tolerance.disable_graceful_degradation()
                    
        except Exception as e:
            self.logger.error(f"Error adjusting components for mode {mode.value}: {e}")
            
    def start_monitoring(self):
        """Start integrated monitoring"""
        if self._running:
            self.logger.warning("Monitoring already running")
            return
            
        if not self._initialization_complete:
            if not self.initialize_components():
                raise RuntimeError("Failed to initialize components")
                
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Integrated monitoring started")
        
    def stop_monitoring(self):
        """Stop integrated monitoring"""
        self._running = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
            
        # Stop component monitoring
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
        if self.compliance_framework:
            self.compliance_framework.stop_compliance_monitoring()
            
        self.logger.info("Integrated monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Update component status
                self._update_component_status()
                
                # Check for expired alerts
                self._cleanup_expired_alerts()
                
                # Perform health checks
                self._perform_integrated_health_check()
                
                # Sleep for monitoring interval
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                if self.error_handler:
                    self.error_handler.handle_error(e, context={"component": "monitoring_loop"})
                    
    def _update_component_status(self):
        """Update status of all components"""
        try:
            # Check each component
            components = {
                'security': self.security_manager,
                'health_monitor': self.health_monitor,
                'fault_tolerance': self.fault_tolerance,
                'error_handler': self.error_handler,
                'compliance': self.compliance_framework
            }
            
            for name, component in components.items():
                if component is None:
                    self.component_status[name] = IntegrationStatus.OFFLINE
                else:
                    # Basic health check - would need component-specific methods
                    self.component_status[name] = IntegrationStatus.HEALTHY
                    
        except Exception as e:
            self.logger.error(f"Error updating component status: {e}")
            
    def _cleanup_expired_alerts(self):
        """Remove expired alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.active_alerts = [
                alert for alert in self.active_alerts
                if datetime.fromisoformat(alert['timestamp']) > cutoff_time
            ]
        except Exception as e:
            self.logger.error(f"Error cleaning up alerts: {e}")
            
    def _perform_integrated_health_check(self):
        """Perform integrated health check across all components"""
        try:
            health_issues = []
            
            # Check component connectivity
            for name, status in self.component_status.items():
                if status != IntegrationStatus.HEALTHY:
                    health_issues.append(f"Component {name} is {status.value}")
                    
            # Check alert levels
            critical_alerts = [a for a in self.active_alerts if a.get('severity') == 'critical']
            if len(critical_alerts) > 5:
                health_issues.append(f"High number of critical alerts: {len(critical_alerts)}")
                
            # Emit health status event if issues found
            if health_issues:
                self.emit_event('health_alert', {
                    "message": "Integrated health check found issues",
                    "issues": health_issues,
                    "severity": "warning",
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Error in integrated health check: {e}")
            
    def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        try:
            # Get performance metrics
            performance_metrics = {}
            if self.health_monitor:
                try:
                    health_data = self.health_monitor.get_system_health()
                    performance_metrics.update({
                        "cpu_usage": health_data.get("cpu_usage", 0),
                        "memory_usage": health_data.get("memory_usage", 0),
                        "disk_usage": health_data.get("disk_usage", 0),
                        "overall_health": health_data.get("overall_health", 0)
                    })
                except Exception:
                    pass
                    
            # Get compliance status
            compliance_status = {}
            if self.compliance_framework:
                try:
                    for level in ComplianceLevel:
                        status = self.compliance_framework.get_compliance_status(level)
                        compliance_status[level.value] = status.get("compliance_percentage", 0)
                except Exception:
                    pass
                    
            # Calculate risk level
            risk_level = "low"
            if self.system_mode == SystemMode.EMERGENCY:
                risk_level = "critical"
            elif self.system_mode == SystemMode.DEGRADED:
                risk_level = "high"
            elif len([a for a in self.active_alerts if a.get('severity') == 'critical']) > 0:
                risk_level = "medium"
                
            return SystemStatus(
                timestamp=datetime.now().isoformat(),
                mode=self.system_mode,
                components=self.component_status.copy(),
                active_alerts=self.active_alerts.copy(),
                performance_metrics=performance_metrics,
                compliance_status=compliance_status,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return SystemStatus(
                timestamp=datetime.now().isoformat(),
                mode=self.system_mode,
                components={},
                active_alerts=[],
                performance_metrics={},
                compliance_status={},
                risk_level="unknown"
            )
            
    def create_enhanced_callback(self, **callback_kwargs) -> 'EnhancedEco2AICallback':
        """Create an enhanced Eco2AI callback with integrated monitoring"""
        return EnhancedEco2AICallback(
            integration_manager=self,
            **callback_kwargs
        )
        
    def shutdown(self):
        """Graceful shutdown of all components"""
        try:
            self.logger.info("Starting graceful shutdown...")
            
            # Log shutdown initiation
            if self.compliance_framework:
                self.compliance_framework.log_audit_event(
                    event_type=AuditEventType.SYSTEM_CONFIGURATION,
                    user_id="system",
                    resource="integration_manager",
                    action="shutdown",
                    details={
                        "shutdown_start": datetime.now().isoformat(),
                        "uptime_seconds": (datetime.now() - self.startup_time).total_seconds()
                    },
                    compliance_level=ComplianceLevel.ISO27001,
                    session_id="shutdown",
                    ip_address="localhost",
                    user_agent="integration_manager",
                    risk_level="low",
                    data_classification="system",
                    outcome="in_progress"
                )
                
            # Stop monitoring
            self.stop_monitoring()
            
            # Shutdown components in reverse order
            components = [
                ('health_monitor', self.health_monitor),
                ('fault_tolerance', self.fault_tolerance),
                ('security_manager', self.security_manager),
                ('error_handler', self.error_handler),
                ('compliance_framework', self.compliance_framework)
            ]
            
            for name, component in components:
                if component and hasattr(component, 'shutdown'):
                    try:
                        component.shutdown()
                        self.logger.info(f"Component {name} shutdown completed")
                    except Exception as e:
                        self.logger.error(f"Error shutting down {name}: {e}")
                        
            self.logger.info("Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


class EnhancedEco2AICallback(Eco2AICallback):
    """Enhanced Eco2AI callback with integrated monitoring and security"""
    
    def __init__(self, integration_manager: EnhancedIntegrationManager, **kwargs):
        super().__init__(**kwargs)
        self.integration_manager = integration_manager
        self.session_id = str(uuid.uuid4())
        
    def on_train_begin(self, logs=None):
        """Enhanced training start with security and compliance logging"""
        super().on_train_begin(logs)
        
        # Log training start event
        if self.integration_manager.compliance_framework:
            self.integration_manager.compliance_framework.log_audit_event(
                event_type=AuditEventType.SYSTEM_CONFIGURATION,
                user_id="trainer",
                resource="ml_training",
                action="training_start",
                details={
                    "session_id": self.session_id,
                    "model_name": getattr(self, 'model_name', 'unknown'),
                    "dataset_size": getattr(self, 'dataset_size', 'unknown'),
                    "training_config": logs or {}
                },
                compliance_level=ComplianceLevel.ISO27001,
                session_id=self.session_id,
                ip_address="localhost",
                user_agent="eco2ai_callback",
                risk_level="low",
                data_classification="training",
                outcome="success"
            )
            
    def on_epoch_end(self, epoch, logs=None):
        """Enhanced epoch end with health monitoring integration"""
        super().on_epoch_end(epoch, logs)
        
        # Check system health during training
        if self.integration_manager.health_monitor:
            health_status = self.integration_manager.health_monitor.get_system_health()
            if health_status.get("overall_health", 1.0) < 0.5:
                self.integration_manager.emit_event('health_alert', {
                    "message": "Poor system health during training",
                    "epoch": epoch,
                    "health_score": health_status.get("overall_health"),
                    "severity": "warning",
                    "session_id": self.session_id
                })
                
    def on_train_end(self, logs=None):
        """Enhanced training end with comprehensive logging"""
        super().on_train_end(logs)
        
        # Log training completion
        if self.integration_manager.compliance_framework:
            self.integration_manager.compliance_framework.log_audit_event(
                event_type=AuditEventType.SYSTEM_CONFIGURATION,
                user_id="trainer",
                resource="ml_training",
                action="training_complete",
                details={
                    "session_id": self.session_id,
                    "training_results": logs or {},
                    "carbon_metrics": getattr(self, 'carbon_metrics', {})
                },
                compliance_level=ComplianceLevel.ISO27001,
                session_id=self.session_id,
                ip_address="localhost",
                user_agent="eco2ai_callback",
                risk_level="low",
                data_classification="training",
                outcome="success"
            )


# Global integration manager instance
_integration_manager = None


def get_integration_manager() -> EnhancedIntegrationManager:
    """Get or create global integration manager instance"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = EnhancedIntegrationManager()
    return _integration_manager


def initialize_enhanced_system(config: Optional[CarbonConfig] = None) -> EnhancedIntegrationManager:
    """Initialize the enhanced HF Eco2AI system with all components"""
    global _integration_manager
    _integration_manager = EnhancedIntegrationManager(config)
    
    if _integration_manager.initialize_components():
        _integration_manager.start_monitoring()
        return _integration_manager
    else:
        raise RuntimeError("Failed to initialize enhanced system")


def create_enhanced_callback(**kwargs) -> EnhancedEco2AICallback:
    """Create an enhanced callback with integrated monitoring"""
    manager = get_integration_manager()
    return manager.create_enhanced_callback(**kwargs)


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize the enhanced system
        manager = initialize_enhanced_system()
        
        # Get system status
        status = manager.get_system_status()
        print(f"System Status: {status.mode.value}")
        print(f"Active Alerts: {len(status.active_alerts)}")
        print(f"Component Status: {status.components}")
        
        # Create enhanced callback
        callback = create_enhanced_callback()
        print("Enhanced callback created successfully")
        
        # Run for a short time then shutdown
        time.sleep(5)
        manager.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()