"""Comprehensive error handling and resilience for carbon tracking."""

import time
import logging
import functools
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass
from enum import Enum
import traceback
import threading
import json
from pathlib import Path


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Error information container."""
    timestamp: float
    severity: ErrorSeverity
    error_type: str
    message: str
    traceback: str
    context: Dict[str, Any]
    resolved: bool = False
    retry_count: int = 0


class CircuitBreaker:
    """Circuit breaker pattern for resilient operations."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == "open":
                    if time.time() - self.last_failure_time > self.timeout:
                        self.state = "half_open"
                        logger.info(f"Circuit breaker entering half-open state for {func.__name__}")
                    else:
                        raise CircuitBreakerOpenError(f"Circuit breaker open for {func.__name__}")
                
                try:
                    result = func(*args, **kwargs)
                    if self.state == "half_open":
                        self.state = "closed"
                        self.failure_count = 0
                        logger.info(f"Circuit breaker closed for {func.__name__}")
                    return result
                    
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "open"
                        logger.warning(f"Circuit breaker opened for {func.__name__}")
                    
                    raise
        
        return wrapper


class RetryPolicy:
    """Configurable retry policy."""
    
    def __init__(self, max_attempts: int = 3, backoff_factor: float = 1.0,
                 retry_exceptions: tuple = (Exception,)):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.retry_exceptions = retry_exceptions
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except self.retry_exceptions as e:
                    last_exception = e
                    if attempt < self.max_attempts - 1:
                        delay = self.backoff_factor * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {self.max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.errors: List[ErrorInfo] = []
        self.log_file = log_file
        self._lock = threading.Lock()
        
        # Error handlers by type
        self.handlers: Dict[Type[Exception], Callable] = {
            ImportError: self._handle_import_error,
            ConnectionError: self._handle_connection_error,
            TimeoutError: self._handle_timeout_error,
            PermissionError: self._handle_permission_error,
            FileNotFoundError: self._handle_file_error,
            ValueError: self._handle_validation_error,
        }
        
        logger.info("Initialized comprehensive error handling system")
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> ErrorInfo:
        """Handle an error with appropriate response."""
        error_info = ErrorInfo(
            timestamp=time.time(),
            severity=severity,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            context=context or {}
        )
        
        with self._lock:
            self.errors.append(error_info)
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, f"Error handled: {error_info.error_type} - {error_info.message}")
        
        # Call specific handler if available
        handler = self.handlers.get(type(error))
        if handler:
            try:
                handler(error, error_info)
            except Exception as handler_error:
                logger.error(f"Error in error handler: {handler_error}")
        
        # Write to log file if configured
        if self.log_file:
            self._write_to_log_file(error_info)
        
        return error_info
    
    def _handle_import_error(self, error: ImportError, error_info: ErrorInfo):
        """Handle missing dependency errors."""
        missing_module = str(error).split("'")[1] if "'" in str(error) else "unknown"
        
        suggestions = {
            "eco2ai": "Install with: pip install eco2ai>=2.0.0",
            "pynvml": "Install with: pip install pynvml>=11.5.0", 
            "psutil": "Install with: pip install psutil>=5.9.0",
            "transformers": "Install with: pip install transformers>=4.40.0",
            "prometheus_client": "Install with: pip install prometheus-client>=0.20.0"
        }
        
        suggestion = suggestions.get(missing_module, f"Install missing module: {missing_module}")
        error_info.context["suggestion"] = suggestion
        error_info.context["missing_module"] = missing_module
        
        logger.warning(f"Missing dependency {missing_module}: {suggestion}")
    
    def _handle_connection_error(self, error: ConnectionError, error_info: ErrorInfo):
        """Handle network connectivity errors."""
        error_info.context["recovery_action"] = "retry_with_backoff"
        logger.warning("Network connectivity issue - will retry with backoff")
    
    def _handle_timeout_error(self, error: TimeoutError, error_info: ErrorInfo):
        """Handle timeout errors."""
        error_info.context["recovery_action"] = "increase_timeout"
        logger.warning("Operation timed out - consider increasing timeout values")
    
    def _handle_permission_error(self, error: PermissionError, error_info: ErrorInfo):
        """Handle permission errors."""
        error_info.context["recovery_action"] = "check_permissions"
        logger.error("Permission denied - check file/directory permissions")
    
    def _handle_file_error(self, error: FileNotFoundError, error_info: ErrorInfo):
        """Handle file not found errors."""
        error_info.context["recovery_action"] = "create_missing_file"
        logger.warning(f"File not found: {error.filename} - will attempt to create")
    
    def _handle_validation_error(self, error: ValueError, error_info: ErrorInfo):
        """Handle validation errors."""
        error_info.context["recovery_action"] = "validate_input"
        logger.warning(f"Validation error: {error} - check input parameters")
    
    def _write_to_log_file(self, error_info: ErrorInfo):
        """Write error to log file."""
        try:
            log_entry = {
                "timestamp": error_info.timestamp,
                "severity": error_info.severity.value,
                "error_type": error_info.error_type,
                "message": error_info.message,
                "context": error_info.context
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write to error log file: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        with self._lock:
            if not self.errors:
                return {"total_errors": 0, "by_severity": {}, "recent_errors": []}
            
            # Count by severity
            by_severity = {}
            for error in self.errors:
                severity = error.severity.value
                by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Recent errors (last 10)
            recent = self.errors[-10:]
            
            return {
                "total_errors": len(self.errors),
                "by_severity": by_severity,
                "recent_errors": [
                    {
                        "timestamp": error.timestamp,
                        "severity": error.severity.value,
                        "type": error.error_type,
                        "message": error.message[:100] + "..." if len(error.message) > 100 else error.message
                    }
                    for error in recent
                ]
            }
    
    def clear_errors(self):
        """Clear error history."""
        with self._lock:
            self.errors.clear()
        logger.info("Error history cleared")


class SafetyWrapper:
    """Safety wrapper for critical operations."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    def safe_execute(self, func: Callable, *args, 
                    fallback_value: Any = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Dict[str, Any] = None,
                    **kwargs) -> Any:
        """Safely execute a function with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.error_handler.handle_error(e, context, severity)
            logger.warning(f"Safe execution failed for {func.__name__}, returning fallback value")
            return fallback_value
    
    def safe_import(self, module_name: str, fallback=None):
        """Safely import a module."""
        try:
            import importlib
            return importlib.import_module(module_name)
        except ImportError as e:
            self.error_handler.handle_error(e, {"module": module_name}, ErrorSeverity.MEDIUM)
            return fallback


# Global error handler instance
_error_handler = ErrorHandler()
_safety_wrapper = SafetyWrapper(_error_handler)


def get_error_handler() -> ErrorHandler:
    """Get global error handler."""
    return _error_handler


def get_safety_wrapper() -> SafetyWrapper:
    """Get global safety wrapper."""
    return _safety_wrapper


def resilient_operation(max_attempts: int = 3, circuit_breaker: bool = True):
    """Decorator for resilient operations with retry and circuit breaker."""
    def decorator(func: Callable) -> Callable:
        # Apply retry policy
        func = RetryPolicy(max_attempts=max_attempts)(func)
        
        # Apply circuit breaker if requested
        if circuit_breaker:
            func = CircuitBreaker()(func)
        
        return func
    
    return decorator


def handle_gracefully(severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                     fallback_value: Any = None):
    """Decorator to handle errors gracefully."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _safety_wrapper.safe_execute(
                func, *args, **kwargs, 
                fallback_value=fallback_value, 
                severity=severity,
                context={"function": func.__name__}
            )
        return wrapper
    return decorator