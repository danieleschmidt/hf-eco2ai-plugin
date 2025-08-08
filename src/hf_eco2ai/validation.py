"""Comprehensive validation and error handling for carbon tracking."""

import logging
import time
import inspect
import traceback
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from functools import wraps
import asyncio
from pathlib import Path
import json
import hashlib
import secrets
from enum import Enum
from contextlib import contextmanager

from .config import CarbonConfig
from .models import CarbonMetrics, CarbonReport

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    check_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    remediation: Optional[str] = None


@dataclass 
class ValidationSuite:
    """Collection of validation results."""
    
    suite_name: str
    results: List[ValidationResult] = field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    execution_time: float = 0.0
    
    def add_result(self, result: ValidationResult):
        """Add validation result to suite."""
        self.results.append(result)
        self.total_checks += 1
        
        if result.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = sum(
                1 for r in self.results if r.severity == severity
            )
        
        return {
            "suite_name": self.suite_name,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "success_rate": self.passed_checks / max(self.total_checks, 1) * 100,
            "execution_time": self.execution_time,
            "severity_breakdown": severity_counts,
            "critical_failures": [
                r for r in self.results 
                if r.severity == ValidationSeverity.CRITICAL and not r.passed
            ]
        }
        return [r for r in self.results if r.is_passing()]
    
    def has_errors(self) -> bool:
        """Check if report has any errors."""
        return len(self.get_errors()) > 0
    
    def has_warnings(self) -> bool:
        """Check if report has any warnings."""
        return len(self.get_warnings()) > 0
    
    def is_valid(self) -> bool:
        """Check if overall validation is successful (no errors)."""
        return not self.has_errors()
    
    def summary_text(self) -> str:
        """Generate summary text of validation."""
        total = len(self.results)
        passing = len(self.get_passing())
        warnings = len(self.get_warnings())
        errors = len(self.get_errors())
        
        status_icon = "✅" if self.is_valid() else "❌"
        
        summary = [
            f"{status_icon} Validation Report",
            f"Total checks: {total}",
            f"Passing: {passing}",
            f"Warnings: {warnings}",
            f"Errors: {errors}",
            ""
        ]
        
        if errors > 0:
            summary.append("❌ ERRORS:")
            for result in self.get_errors():
                summary.append(f"  - {result.component}: {result.message}")
                if result.fix_suggestion:
                    summary.append(f"    Fix: {result.fix_suggestion}")
            summary.append("")
        
        if warnings > 0:
            summary.append("⚠️ WARNINGS:")
            for result in self.get_warnings():
                summary.append(f"  - {result.component}: {result.message}")
                if result.fix_suggestion:
                    summary.append(f"    Suggestion: {result.fix_suggestion}")
            summary.append("")
        
        return "\n".join(summary)


class CarbonTrackingValidator:
    """Comprehensive validator for carbon tracking setup."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize validator.
        
        Args:
            config: Carbon tracking configuration to validate
        """
        self.config = config or CarbonConfig()
        self.results: List[ValidationResult] = []
    
    def validate_all(self) -> ValidationReport:
        """Run all validation checks.
        
        Returns:
            Comprehensive validation report
        """
        logger.info("Starting comprehensive validation")
        
        self.results = []
        
        # Core validations
        self._validate_dependencies()
        self._validate_configuration()
        self._validate_system_requirements()
        self._validate_gpu_setup()
        self._validate_storage_access()
        self._validate_monitoring_capabilities()
        self._validate_export_settings()
        self._validate_integration_settings()
        
        report = ValidationReport(
            results=self.results.copy(),
            timestamp=time.time(),
            system_info=get_system_info()
        )
        
        logger.info(f"Validation completed: {len(report.get_passing())} passed, "
                   f"{len(report.get_warnings())} warnings, {len(report.get_errors())} errors")
        
        return report
    
    def _validate_dependencies(self):
        """Validate required and optional dependencies."""
        dependencies = check_dependencies()
        
        # Required dependencies
        required = ["transformers"]
        for dep in required:
            if not dependencies.get(dep, False):
                self.results.append(ValidationResult(
                    component="dependencies",
                    status="error",
                    message=f"Required dependency '{dep}' is not available",
                    fix_suggestion=f"Install with: pip install {dep}"
                ))
            else:
                self.results.append(ValidationResult(
                    component="dependencies",
                    status="pass",
                    message=f"Required dependency '{dep}' is available"
                ))
        
        # Optional but recommended dependencies
        recommended = ["pynvml", "eco2ai", "psutil", "prometheus_client"]
        for dep in recommended:
            if not dependencies.get(dep, False):
                self.results.append(ValidationResult(
                    component="dependencies",
                    status="warning",
                    message=f"Recommended dependency '{dep}' is not available",
                    fix_suggestion=f"Install with: pip install {dep}"
                ))
            else:
                self.results.append(ValidationResult(
                    component="dependencies",
                    status="pass",
                    message=f"Recommended dependency '{dep}' is available"
                ))
    
    def _validate_configuration(self):
        """Validate configuration settings."""
        try:
            # Test configuration creation and validation
            missing_deps = self.config.validate_environment()
            
            if not missing_deps:
                self.results.append(ValidationResult(
                    component="configuration",
                    status="pass",
                    message="Configuration is valid"
                ))
            else:
                self.results.append(ValidationResult(
                    component="configuration",
                    status="warning",
                    message=f"Configuration has missing dependencies: {missing_deps}",
                    fix_suggestion=f"Install missing dependencies: pip install {' '.join(missing_deps)}"
                ))
            
            # Validate specific settings
            if self.config.gpu_sampling_interval < 0.1:
                self.results.append(ValidationResult(
                    component="configuration",
                    status="warning",
                    message="GPU sampling interval is very low, may impact performance",
                    fix_suggestion="Consider increasing gpu_sampling_interval to >= 0.5"
                ))
            
            if self.config.enable_carbon_budget and not self.config.max_co2_kg:
                self.results.append(ValidationResult(
                    component="configuration",
                    status="error",
                    message="Carbon budget enabled but max_co2_kg not set",
                    fix_suggestion="Set max_co2_kg when enable_carbon_budget=True"
                ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                component="configuration",
                status="error",
                message=f"Configuration validation failed: {e}",
                fix_suggestion="Check configuration parameters and fix invalid values"
            ))
    
    def _validate_system_requirements(self):
        """Validate system requirements."""
        system_info = get_system_info()
        
        # Check Python version
        import sys
        if sys.version_info < (3, 10):
            self.results.append(ValidationResult(
                component="system",
                status="error",
                message=f"Python {sys.version} is too old, requires Python 3.10+",
                fix_suggestion="Upgrade to Python 3.10 or newer"
            ))
        else:
            self.results.append(ValidationResult(
                component="system",
                status="pass",
                message=f"Python version {sys.version_info.major}.{sys.version_info.minor} is supported"
            ))
        
        # Check available memory
        if system_info.get("memory_total_gb"):
            memory_gb = system_info["memory_total_gb"]
            if memory_gb < 4:
                self.results.append(ValidationResult(
                    component="system",
                    status="warning",
                    message=f"Low system memory: {memory_gb}GB (recommend 8GB+)",
                    fix_suggestion="Consider upgrading system memory for better performance"
                ))
            else:
                self.results.append(ValidationResult(
                    component="system",
                    status="pass",
                    message=f"System memory: {memory_gb}GB"
                ))
        
        # Check CPU count
        cpu_count = system_info.get("cpu_count", 0)
        if cpu_count < 2:
            self.results.append(ValidationResult(
                component="system",
                status="warning",
                message=f"Low CPU count: {cpu_count} (recommend 4+ cores)",
                fix_suggestion="Multi-core CPU recommended for better performance"
            ))
        else:
            self.results.append(ValidationResult(
                component="system",
                status="pass",
                message=f"CPU cores: {cpu_count}"
            ))
    
    def _validate_gpu_setup(self):
        """Validate GPU setup and monitoring."""
        system_info = get_system_info()
        gpu_count = system_info.get("gpu_count", 0)
        
        if gpu_count == 0:
            self.results.append(ValidationResult(
                component="gpu",
                status="warning",
                message="No GPUs detected - will use CPU-only monitoring",
                fix_suggestion="Install CUDA-capable GPU for better monitoring accuracy"
            ))
        else:
            self.results.append(ValidationResult(
                component="gpu",
                status="pass",
                message=f"Detected {gpu_count} GPU(s)",
                details={"gpus": system_info.get("gpus", [])}
            ))
        
        # Validate GPU IDs configuration
        if isinstance(self.config.gpu_ids, list):
            for gpu_id in self.config.gpu_ids:
                if gpu_id >= gpu_count:
                    self.results.append(ValidationResult(
                        component="gpu",
                        status="error",
                        message=f"GPU ID {gpu_id} not available (only {gpu_count} GPUs detected)",
                        fix_suggestion=f"Use GPU IDs 0-{gpu_count-1} or 'auto'"
                    ))
        
        # Test GPU monitoring
        try:
            import pynvml
            pynvml.nvmlInit()
            self.results.append(ValidationResult(
                component="gpu",
                status="pass",
                message="GPU monitoring (pynvml) is functional"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                component="gpu",
                status="warning",
                message=f"GPU monitoring not available: {e}",
                fix_suggestion="Install NVIDIA drivers and pynvml: pip install pynvml"
            ))
    
    def _validate_storage_access(self):
        """Validate storage and file access."""
        # Check report output directory
        if self.config.save_report:
            report_path = Path(self.config.report_path)
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Test write access
                test_file = report_path.parent / ".carbon_test"
                test_file.write_text("test")
                test_file.unlink()
                
                self.results.append(ValidationResult(
                    component="storage",
                    status="pass",
                    message=f"Report output directory is writable: {report_path.parent}"
                ))
            
            except Exception as e:
                self.results.append(ValidationResult(
                    component="storage",
                    status="error",
                    message=f"Cannot write to report directory: {e}",
                    fix_suggestion=f"Ensure {report_path.parent} exists and is writable"
                ))
        
        # Check CSV output if enabled
        if self.config.export_csv:
            csv_path = Path(self.config.csv_path)
            try:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                self.results.append(ValidationResult(
                    component="storage",
                    status="pass",
                    message=f"CSV output directory is accessible: {csv_path.parent}"
                ))
            except Exception as e:
                self.results.append(ValidationResult(
                    component="storage",
                    status="error",
                    message=f"Cannot access CSV directory: {e}",
                    fix_suggestion=f"Ensure {csv_path.parent} exists and is writable"
                ))
    
    def _validate_monitoring_capabilities(self):
        """Validate monitoring capabilities."""
        try:
            from .monitoring import EnergyTracker
            
            # Test basic energy tracker initialization
            tracker = EnergyTracker(
                gpu_ids=self.config.gpu_ids,
                country=self.config.country,
                region=self.config.region
            )
            
            if tracker.is_available():
                self.results.append(ValidationResult(
                    component="monitoring",
                    status="pass",
                    message="Energy tracking is available"
                ))
            else:
                self.results.append(ValidationResult(
                    component="monitoring",
                    status="warning",
                    message="Energy tracking using estimation mode",
                    fix_suggestion="Install pynvml and ensure NVIDIA drivers for accurate GPU tracking"
                ))
            
            # Test carbon intensity provider
            carbon_intensity = tracker.carbon_provider.get_carbon_intensity()
            if carbon_intensity > 0:
                self.results.append(ValidationResult(
                    component="monitoring",
                    status="pass",
                    message=f"Carbon intensity data available: {carbon_intensity:.0f} g CO₂/kWh"
                ))
            else:
                self.results.append(ValidationResult(
                    component="monitoring",
                    status="warning",
                    message="Carbon intensity data may not be available",
                    fix_suggestion="Check country/region settings in configuration"
                ))
        
        except Exception as e:
            self.results.append(ValidationResult(
                component="monitoring",
                status="error",
                message=f"Monitoring system initialization failed: {e}",
                fix_suggestion="Check dependencies and configuration"
            ))
    
    def _validate_export_settings(self):
        """Validate export and integration settings."""
        # Prometheus validation
        if self.config.export_prometheus:
            try:
                import prometheus_client
                self.results.append(ValidationResult(
                    component="export",
                    status="pass",
                    message="Prometheus client is available"
                ))
                
                # Test port availability
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((self.config.prometheus_host, self.config.prometheus_port))
                sock.close()
                
                if result == 0:
                    self.results.append(ValidationResult(
                        component="export",
                        status="warning",
                        message=f"Port {self.config.prometheus_port} appears to be in use",
                        fix_suggestion="Choose a different prometheus_port or stop existing service"
                    ))
                else:
                    self.results.append(ValidationResult(
                        component="export",
                        status="pass",
                        message=f"Prometheus port {self.config.prometheus_port} is available"
                    ))
                
            except ImportError:
                self.results.append(ValidationResult(
                    component="export",
                    status="error",
                    message="Prometheus export enabled but prometheus_client not available",
                    fix_suggestion="Install prometheus_client: pip install prometheus-client"
                ))
    
    def _validate_integration_settings(self):
        """Validate external integration settings."""
        # MLflow validation
        if self.config.mlflow_tracking:
            try:
                import mlflow
                self.results.append(ValidationResult(
                    component="integrations",
                    status="pass",
                    message="MLflow is available"
                ))
            except ImportError:
                self.results.append(ValidationResult(
                    component="integrations",
                    status="error",
                    message="MLflow tracking enabled but mlflow not available",
                    fix_suggestion="Install mlflow: pip install mlflow"
                ))
        
        # Weights & Biases validation
        if self.config.wandb_tracking:
            try:
                import wandb
                self.results.append(ValidationResult(
                    component="integrations",
                    status="pass",
                    message="Weights & Biases is available"
                ))
            except ImportError:
                self.results.append(ValidationResult(
                    component="integrations",
                    status="error",
                    message="wandb tracking enabled but wandb not available",
                    fix_suggestion="Install wandb: pip install wandb"
                ))


class ErrorHandler:
    """Centralized error handling for carbon tracking."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize error handler.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        self.error_log: List[Dict[str, Any]] = []
        self.error_count = 0
        self.last_error_time = 0
        self.recovery_strategies: Dict[str, Callable] = {
            "gpu_monitoring": self._recover_gpu_monitoring,
            "prometheus_export": self._recover_prometheus_export,
            "file_write": self._recover_file_write,
        }
    
    @contextmanager
    def handle_errors(self, context: str, fallback_value: Any = None, 
                     suppress_errors: bool = True):
        """Context manager for graceful error handling.
        
        Args:
            context: Description of the operation
            fallback_value: Value to return on error
            suppress_errors: Whether to suppress exceptions
        """
        try:
            yield
        except Exception as e:
            self._log_error(context, e)
            
            # Try recovery strategy if available
            recovery_strategy = self.recovery_strategies.get(context)
            if recovery_strategy:
                try:
                    return recovery_strategy(e)
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed for {context}: {recovery_error}")
            
            if suppress_errors:
                logger.error(f"Error in {context}: {e}")
                return fallback_value
            else:
                raise
    
    def _log_error(self, context: str, error: Exception):
        """Log an error with context."""
        self.error_count += 1
        self.last_error_time = time.time()
        
        error_record = {
            "timestamp": time.time(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "error_count": self.error_count
        }
        
        self.error_log.append(error_record)
        
        # Keep only last 100 errors to prevent memory issues
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
        
        logger.error(f"Error #{self.error_count} in {context}: {error}")
    
    def _recover_gpu_monitoring(self, error: Exception):
        """Recovery strategy for GPU monitoring failures."""
        logger.warning("GPU monitoring failed, switching to CPU-only mode")
        # Return mock GPU metrics
        return {
            "power_watts": 0.0,
            "energy_kwh": 0.0,
            "utilization": 0.0
        }
    
    def _recover_prometheus_export(self, error: Exception):
        """Recovery strategy for Prometheus export failures."""
        logger.warning("Prometheus export failed, continuing without metrics export")
        return None
    
    def _recover_file_write(self, error: Exception):
        """Recovery strategy for file write failures."""
        logger.warning(f"File write failed, attempting to use temporary directory")
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "carbon_tracking"
        temp_dir.mkdir(exist_ok=True)
        return temp_dir
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        if not self.error_log:
            return {"status": "no_errors", "total_errors": 0}
        
        # Group errors by context
        error_contexts = {}
        for error in self.error_log:
            context = error["context"]
            if context not in error_contexts:
                error_contexts[context] = []
            error_contexts[context].append(error)
        
        return {
            "status": "errors_encountered",
            "total_errors": self.error_count,
            "last_error_time": self.last_error_time,
            "contexts": {ctx: len(errors) for ctx, errors in error_contexts.items()},
            "recent_errors": self.error_log[-5:]  # Last 5 errors
        }
    
    def export_error_log(self, output_path: str):
        """Export error log to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "error_log": self.error_log,
                "summary": self.get_error_summary(),
                "exported_at": time.time()
            }, f, indent=2, default=str)
        
        logger.info(f"Error log exported to {output_path}")


# Global error handler instance
_error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _error_handler

def handle_errors(context: str, fallback_value: Any = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with _error_handler.handle_errors(context, fallback_value):
                return func(*args, **kwargs)
            return fallback_value
        return wrapper
    return decorator