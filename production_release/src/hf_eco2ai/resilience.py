"""Resilience and error handling for carbon tracking systems."""

import time
import logging
import asyncio
import threading
import queue
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager
from enum import Enum
import traceback
import signal
import sys
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of failure modes."""
    TRANSIENT = "transient"  # Temporary failures that may resolve
    PERMANENT = "permanent"  # Permanent failures requiring intervention
    DEGRADED = "degraded"    # Partial failures with reduced functionality
    CASCADING = "cascading"  # Failures that cause other failures


class RecoveryStrategy(Enum):
    """Recovery strategies for failures."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"


@dataclass
class FailureRecord:
    """Record of a system failure."""
    
    timestamp: float
    component: str
    failure_type: str
    error_message: str
    stack_trace: str
    failure_mode: FailureMode
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_time: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    
    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "closed"  # closed, open, half_open
    failure_threshold: int = 5
    timeout: float = 60.0
    success_threshold: int = 3
    consecutive_successes: int = 0


class ResilienceManager:
    """Centralized resilience and error handling manager."""
    
    def __init__(self):
        """Initialize resilience manager."""
        self.failure_history: List[FailureRecord] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.retry_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._shutdown_handlers: List[Callable] = []
        
        # Default retry configuration
        self.default_retry_config = {
            "max_attempts": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "exponential_backoff": True,
            "jitter": True
        }
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info("Resilience manager initialized")
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def record_failure(self, 
                      component: str,
                      error: Exception,
                      failure_mode: FailureMode = FailureMode.TRANSIENT,
                      context: Optional[Dict[str, Any]] = None):
        """Record a system failure.
        
        Args:
            component: Component that failed
            error: Exception that occurred
            failure_mode: Type of failure
            context: Additional context information
        """
        failure_record = FailureRecord(
            timestamp=time.time(),
            component=component,
            failure_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            failure_mode=failure_mode,
            context=context or {}
        )
        
        with self._lock:
            self.failure_history.append(failure_record)
            
            # Update circuit breaker state
            if component in self.circuit_breakers:
                cb = self.circuit_breakers[component]
                cb.failure_count += 1
                cb.last_failure_time = time.time()
                
                if cb.failure_count >= cb.failure_threshold:
                    cb.state = "open"
                    logger.warning(f"Circuit breaker opened for {component}")
        
        logger.error(f"Failure recorded for {component}: {error}")
    
    def record_success(self, component: str):
        """Record successful operation for circuit breaker management.
        
        Args:
            component: Component that succeeded
        """
        with self._lock:
            if component in self.circuit_breakers:
                cb = self.circuit_breakers[component]
                
                if cb.state == "half_open":
                    cb.consecutive_successes += 1
                    if cb.consecutive_successes >= cb.success_threshold:
                        cb.state = "closed"
                        cb.failure_count = 0
                        cb.consecutive_successes = 0
                        logger.info(f"Circuit breaker closed for {component}")
                elif cb.state == "closed":
                    # Reset failure count on success
                    cb.failure_count = max(0, cb.failure_count - 1)
    
    def should_allow_request(self, component: str) -> bool:
        """Check if request should be allowed based on circuit breaker state.
        
        Args:
            component: Component to check
            
        Returns:
            True if request should be allowed
        """
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreakerState()
            return True
        
        cb = self.circuit_breakers[component]
        current_time = time.time()
        
        if cb.state == "closed":
            return True
        elif cb.state == "open":
            if current_time - cb.last_failure_time >= cb.timeout:
                cb.state = "half_open"
                cb.consecutive_successes = 0
                logger.info(f"Circuit breaker half-opened for {component}")
                return True
            return False
        elif cb.state == "half_open":
            return True
        
        return False
    
    def get_failure_statistics(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get failure statistics for analysis.
        
        Args:
            component: Specific component to analyze (None for all)
            
        Returns:
            Failure statistics
        """
        with self._lock:
            failures = self.failure_history
            
            if component:
                failures = [f for f in failures if f.component == component]
            
            if not failures:
                return {"total_failures": 0}
            
            # Calculate statistics
            total_failures = len(failures)
            recent_failures = [
                f for f in failures 
                if time.time() - f.timestamp < 3600  # Last hour
            ]
            
            failure_types = {}
            failure_modes = {}
            components = {}
            
            for failure in failures:
                failure_types[failure.failure_type] = failure_types.get(failure.failure_type, 0) + 1
                failure_modes[failure.failure_mode.value] = failure_modes.get(failure.failure_mode.value, 0) + 1
                components[failure.component] = components.get(failure.component, 0) + 1
            
            return {
                "total_failures": total_failures,
                "recent_failures": len(recent_failures),
                "failure_types": failure_types,
                "failure_modes": failure_modes,
                "affected_components": components,
                "average_recovery_time": self._calculate_average_recovery_time(failures),
                "recovery_success_rate": self._calculate_recovery_success_rate(failures)
            }
    
    def _calculate_average_recovery_time(self, failures: List[FailureRecord]) -> Optional[float]:
        """Calculate average recovery time for failures."""
        recovery_times = [
            f.recovery_time for f in failures 
            if f.recovery_time is not None
        ]
        
        if recovery_times:
            return sum(recovery_times) / len(recovery_times)
        return None
    
    def _calculate_recovery_success_rate(self, failures: List[FailureRecord]) -> float:
        """Calculate recovery success rate."""
        recovery_attempts = [f for f in failures if f.recovery_attempted]
        
        if not recovery_attempts:
            return 0.0
        
        successful_recoveries = [f for f in recovery_attempts if f.recovery_successful]
        return len(successful_recoveries) / len(recovery_attempts)
    
    def add_shutdown_handler(self, handler: Callable):
        """Add handler for graceful shutdown.
        
        Args:
            handler: Function to call during shutdown
        """
        self._shutdown_handlers.append(handler)
    
    def shutdown(self):
        """Perform graceful shutdown."""
        logger.info("Starting graceful shutdown")
        
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")
        
        logger.info("Graceful shutdown completed")


def with_retry(component: str = "default",
               max_attempts: int = 3,
               base_delay: float = 1.0,
               max_delay: float = 60.0,
               exponential_backoff: bool = True,
               jitter: bool = True,
               exceptions: Tuple[Type[Exception], ...] = (Exception,),
               recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY):
    """Decorator to add retry logic to functions.
    
    Args:
        component: Component name for tracking
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential_backoff: Use exponential backoff
        jitter: Add random jitter to delays
        exceptions: Exception types to retry on
        recovery_strategy: Strategy for handling failures
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            
            # Check circuit breaker
            if not manager.should_allow_request(component):
                raise RuntimeError(f"Circuit breaker open for {component}")
            
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    manager.record_success(component)
                    return result
                
                except exceptions as e:
                    last_exception = e
                    failure_mode = FailureMode.TRANSIENT
                    
                    # Determine failure mode based on exception type
                    if isinstance(e, (ConnectionError, TimeoutError)):
                        failure_mode = FailureMode.TRANSIENT
                    elif isinstance(e, (PermissionError, FileNotFoundError)):
                        failure_mode = FailureMode.PERMANENT
                    
                    manager.record_failure(
                        component=component,
                        error=e,
                        failure_mode=failure_mode,
                        context={"attempt": attempt + 1, "function": func.__name__}
                    )
                    
                    if attempt == max_attempts - 1:
                        break
                    
                    # Calculate delay
                    if exponential_backoff:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    # Add jitter
                    if jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {component}, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
            
            # All attempts failed
            if recovery_strategy == RecoveryStrategy.FAIL_FAST:
                raise last_exception
            elif recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                logger.warning(f"All retries failed for {component}, returning degraded result")
                return None
            else:
                raise last_exception
        
        return wrapper
    return decorator


def with_circuit_breaker(component: str,
                        failure_threshold: int = 5,
                        timeout: float = 60.0,
                        success_threshold: int = 3):
    """Decorator to add circuit breaker pattern to functions.
    
    Args:
        component: Component name
        failure_threshold: Number of failures to open circuit
        timeout: Time to wait before half-opening circuit
        success_threshold: Successes needed to close circuit
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            
            # Configure circuit breaker if not exists
            if component not in manager.circuit_breakers:
                manager.circuit_breakers[component] = CircuitBreakerState(
                    failure_threshold=failure_threshold,
                    timeout=timeout,
                    success_threshold=success_threshold
                )
            
            # Check if request should be allowed
            if not manager.should_allow_request(component):
                raise RuntimeError(f"Circuit breaker open for {component}")
            
            try:
                result = func(*args, **kwargs)
                manager.record_success(component)
                return result
            except Exception as e:
                manager.record_failure(component, e)
                raise
        
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to functions.
    
    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")
            
            # Set up timeout signal
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


@contextmanager
def graceful_degradation(default_value: Any = None, 
                        log_error: bool = True):
    """Context manager for graceful degradation on errors.
    
    Args:
        default_value: Value to return on error
        log_error: Whether to log the error
    """
    try:
        yield
    except Exception as e:
        if log_error:
            logger.warning(f"Operation failed, using default value: {e}")
        return default_value


@contextmanager
def error_boundary(component: str, 
                  recovery_action: Optional[Callable] = None):
    """Context manager to catch and handle errors with recovery.
    
    Args:
        component: Component name for tracking
        recovery_action: Function to call for recovery
    """
    try:
        yield
    except Exception as e:
        manager = get_resilience_manager()
        manager.record_failure(component, e)
        
        if recovery_action:
            try:
                recovery_action()
                logger.info(f"Recovery action successful for {component}")
            except Exception as recovery_error:
                logger.error(f"Recovery action failed for {component}: {recovery_error}")
        
        raise


class HealthCheck:
    """Health check system for monitoring component status."""
    
    def __init__(self):
        """Initialize health check system."""
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.check_results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks.
        
        Returns:
            Dictionary with check results
        """
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                check_time = time.time() - start_time
                
                results[name] = {
                    "healthy": is_healthy,
                    "check_time": check_time,
                    "timestamp": time.time()
                }
                
                if not is_healthy:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
                overall_healthy = False
        
        with self._lock:
            self.check_results = results
        
        return {
            "overall_healthy": overall_healthy,
            "checks": results,
            "timestamp": time.time()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status.
        
        Returns:
            Current health status
        """
        with self._lock:
            return {
                "healthy": all(
                    result.get("healthy", False) 
                    for result in self.check_results.values()
                ),
                "checks": self.check_results.copy(),
                "last_check": max(
                    (result.get("timestamp", 0) for result in self.check_results.values()),
                    default=0
                )
            }


class BackupManager:
    """Manager for creating and restoring system backups."""
    
    def __init__(self, backup_dir: str = "backups"):
        """Initialize backup manager.
        
        Args:
            backup_dir: Directory to store backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
    def create_backup(self, data: Any, name: str) -> str:
        """Create a backup of data.
        
        Args:
            data: Data to backup
            name: Backup name
            
        Returns:
            Path to backup file
        """
        timestamp = int(time.time())
        backup_filename = f"{name}_{timestamp}.backup"
        backup_path = self.backup_dir / backup_filename
        
        with open(backup_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Backup created: {backup_path}")
        return str(backup_path)
    
    def restore_backup(self, backup_path: str) -> Any:
        """Restore data from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Restored data
        """
        with open(backup_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Backup restored: {backup_path}")
        return data
    
    def list_backups(self, name_filter: Optional[str] = None) -> List[str]:
        """List available backups.
        
        Args:
            name_filter: Filter backups by name
            
        Returns:
            List of backup file paths
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("*.backup"):
            if name_filter is None or name_filter in backup_file.name:
                backups.append(str(backup_file))
        
        return sorted(backups, reverse=True)  # Most recent first


# Global instances
_resilience_manager = None
_health_check = None
_backup_manager = None


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


def get_health_check() -> HealthCheck:
    """Get global health check instance."""
    global _health_check
    if _health_check is None:
        _health_check = HealthCheck()
    return _health_check


def get_backup_manager() -> BackupManager:
    """Get global backup manager instance."""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager


def initialize_resilience():
    """Initialize resilience systems."""
    # Initialize all global instances
    get_resilience_manager()
    get_health_check()
    get_backup_manager()
    
    logger.info("Resilience systems initialized")