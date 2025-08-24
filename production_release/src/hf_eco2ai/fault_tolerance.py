"""Enterprise fault tolerance and resilience systems."""

import asyncio
import time
import json
import logging
import threading
import functools
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import random
from concurrent.futures import ThreadPoolExecutor, Future


logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of failures the system can encounter."""
    NETWORK_TIMEOUT = "network_timeout"
    API_RATE_LIMIT = "api_rate_limit"
    DISK_FULL = "disk_full"
    MEMORY_ERROR = "memory_error"
    GPU_ERROR = "gpu_error"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_CORRUPTION = "data_corruption"
    PERMISSION_DENIED = "permission_denied"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    RETRY_EXPONENTIAL_BACKOFF = "retry_exponential_backoff"
    RETRY_LINEAR_BACKOFF = "retry_linear_backoff"
    FALLBACK_TO_ALTERNATIVE = "fallback_to_alternative"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    CHECKPOINT_RECOVERY = "checkpoint_recovery"
    FAIL_FAST = "fail_fast"
    IGNORE_AND_CONTINUE = "ignore_and_continue"


@dataclass
class FailureRecord:
    """Record of a failure event."""
    failure_id: str
    timestamp: datetime
    failure_mode: FailureMode
    component: str
    error_message: str
    context: Dict[str, Any]
    recovery_strategy: RecoveryStrategy
    recovery_successful: bool = False
    recovery_time_seconds: float = 0.0
    attempts: int = 1


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    name: str
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    timeout_seconds: int = 60
    success_count: int = 0
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed through circuit breaker."""
        now = datetime.now()
        
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and (now - self.last_failure_time).seconds >= self.timeout_seconds:
                self.state = "HALF_OPEN"
                self.success_count = 0
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to close
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"


class RetryManager:
    """Advanced retry logic with multiple strategies."""
    
    def __init__(self):
        self.retry_configs: Dict[str, Dict[str, Any]] = {
            "default": {
                "max_attempts": 3,
                "base_delay": 1.0,
                "max_delay": 60.0,
                "exponential_base": 2.0,
                "jitter": True
            },
            "network": {
                "max_attempts": 5,
                "base_delay": 0.5,
                "max_delay": 30.0,
                "exponential_base": 2.0,
                "jitter": True
            },
            "critical": {
                "max_attempts": 10,
                "base_delay": 0.1,
                "max_delay": 120.0,
                "exponential_base": 1.5,
                "jitter": True
            }
        }
    
    def calculate_delay(
        self, 
        attempt: int, 
        strategy: RecoveryStrategy,
        config_name: str = "default"
    ) -> float:
        """Calculate delay before next retry attempt."""
        config = self.retry_configs.get(config_name, self.retry_configs["default"])
        
        if strategy == RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF:
            delay = config["base_delay"] * (config["exponential_base"] ** (attempt - 1))
        elif strategy == RecoveryStrategy.RETRY_LINEAR_BACKOFF:
            delay = config["base_delay"] * attempt
        else:
            delay = config["base_delay"]
        
        # Apply max delay limit
        delay = min(delay, config["max_delay"])
        
        # Add jitter to prevent thundering herd
        if config.get("jitter", True):
            jitter = random.uniform(0.8, 1.2)
            delay *= jitter
        
        return delay
    
    async def retry_async(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        strategy: RecoveryStrategy = RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF,
        config_name: str = "default",
        failure_mode: Optional[FailureMode] = None
    ) -> Any:
        """Retry an async function with specified strategy."""
        kwargs = kwargs or {}
        config = self.retry_configs.get(config_name, self.retry_configs["default"])
        max_attempts = config["max_attempts"]
        
        for attempt in range(1, max_attempts + 1):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                if attempt == max_attempts:
                    logger.error(f"All {max_attempts} retry attempts failed: {e}")
                    raise
                
                delay = self.calculate_delay(attempt, strategy, config_name)
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        raise RuntimeError(f"Max retry attempts ({max_attempts}) exceeded")
    
    def retry_sync(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        strategy: RecoveryStrategy = RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF,
        config_name: str = "default",
        failure_mode: Optional[FailureMode] = None
    ) -> Any:
        """Retry a sync function with specified strategy."""
        kwargs = kwargs or {}
        config = self.retry_configs.get(config_name, self.retry_configs["default"])
        max_attempts = config["max_attempts"]
        
        for attempt in range(1, max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if attempt == max_attempts:
                    logger.error(f"All {max_attempts} retry attempts failed: {e}")
                    raise
                
                delay = self.calculate_delay(attempt, strategy, config_name)
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise RuntimeError(f"Max retry attempts ({max_attempts}) exceeded")


class CircuitBreakerManager:
    """Manages circuit breakers for different components."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self._lock = threading.Lock()
    
    def get_circuit_breaker(
        self, 
        name: str, 
        failure_threshold: int = 5,
        timeout_seconds: int = 60
    ) -> CircuitBreakerState:
        """Get or create a circuit breaker."""
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreakerState(
                    name=name,
                    failure_threshold=failure_threshold,
                    timeout_seconds=timeout_seconds
                )
            return self.circuit_breakers[name]
    
    def execute_with_circuit_breaker(
        self,
        name: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        failure_threshold: int = 5,
        timeout_seconds: int = 60
    ) -> Any:
        """Execute function with circuit breaker protection."""
        kwargs = kwargs or {}
        breaker = self.get_circuit_breaker(name, failure_threshold, timeout_seconds)
        
        if not breaker.should_allow_request():
            raise RuntimeError(f"Circuit breaker '{name}' is OPEN - request blocked")
        
        try:
            result = func(*args, **kwargs)
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            logger.warning(f"Circuit breaker '{name}' recorded failure: {e}")
            raise
    
    async def execute_async_with_circuit_breaker(
        self,
        name: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        failure_threshold: int = 5,
        timeout_seconds: int = 60
    ) -> Any:
        """Execute async function with circuit breaker protection."""
        kwargs = kwargs or {}
        breaker = self.get_circuit_breaker(name, failure_threshold, timeout_seconds)
        
        if not breaker.should_allow_request():
            raise RuntimeError(f"Circuit breaker '{name}' is OPEN - request blocked")
        
        try:
            result = await func(*args, **kwargs)
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            logger.warning(f"Circuit breaker '{name}' recorded failure: {e}")
            raise
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        with self._lock:
            return {
                name: {
                    "state": breaker.state,
                    "failure_count": breaker.failure_count,
                    "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
                    "success_count": breaker.success_count
                }
                for name, breaker in self.circuit_breakers.items()
            }


class CheckpointManager:
    """Manages training checkpoints for recovery."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
    
    def create_checkpoint(
        self,
        checkpoint_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Create a checkpoint."""
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "metadata": metadata or {}
        }
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        self.checkpoints[checkpoint_id] = checkpoint_data
        logger.info(f"Created checkpoint: {checkpoint_id} at {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.checkpoints[checkpoint_id] = checkpoint_data
            logger.info(f"Loaded checkpoint: {checkpoint_id}")
            
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                checkpoints.append({
                    "checkpoint_id": checkpoint_data.get("checkpoint_id"),
                    "timestamp": checkpoint_data.get("timestamp"),
                    "file_path": str(checkpoint_file),
                    "metadata": checkpoint_data.get("metadata", {})
                })
            except Exception as e:
                logger.error(f"Error reading checkpoint file {checkpoint_file}: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_count: int = 10) -> int:
        """Clean up old checkpoints, keeping only the most recent ones."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_count:
            return 0
        
        checkpoints_to_delete = checkpoints[keep_count:]
        deleted_count = 0
        
        for checkpoint in checkpoints_to_delete:
            try:
                checkpoint_path = Path(checkpoint["file_path"])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old checkpoint: {checkpoint['checkpoint_id']}")
            except Exception as e:
                logger.error(f"Failed to delete checkpoint {checkpoint['checkpoint_id']}: {e}")
        
        return deleted_count


class FaultToleranceManager:
    """Main fault tolerance orchestrator."""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.retry_manager = RetryManager()
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir or Path("./checkpoints")
        )
        
        self.failure_records: List[FailureRecord] = []
        self.recovery_strategies: Dict[FailureMode, RecoveryStrategy] = {
            FailureMode.NETWORK_TIMEOUT: RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF,
            FailureMode.API_RATE_LIMIT: RecoveryStrategy.RETRY_LINEAR_BACKOFF,
            FailureMode.DISK_FULL: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureMode.MEMORY_ERROR: RecoveryStrategy.CHECKPOINT_RECOVERY,
            FailureMode.GPU_ERROR: RecoveryStrategy.CIRCUIT_BREAKER,
            FailureMode.DEPENDENCY_UNAVAILABLE: RecoveryStrategy.FALLBACK_TO_ALTERNATIVE,
            FailureMode.CONFIGURATION_ERROR: RecoveryStrategy.FAIL_FAST,
            FailureMode.DATA_CORRUPTION: RecoveryStrategy.CHECKPOINT_RECOVERY,
            FailureMode.PERMISSION_DENIED: RecoveryStrategy.FAIL_FAST,
            FailureMode.UNKNOWN_ERROR: RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF
        }
        
        self._failure_detection_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
    
    def classify_failure(self, error: Exception, context: Dict[str, Any]) -> FailureMode:
        """Classify an error into a failure mode."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Network-related errors
        if any(term in error_str for term in ["timeout", "connection", "network", "unreachable"]):
            return FailureMode.NETWORK_TIMEOUT
        
        # Rate limiting
        if any(term in error_str for term in ["rate limit", "too many requests", "429"]):
            return FailureMode.API_RATE_LIMIT
        
        # Disk space
        if any(term in error_str for term in ["no space", "disk full", "storage"]):
            return FailureMode.DISK_FULL
        
        # Memory errors
        if error_type in ["MemoryError", "OutOfMemoryError"] or "memory" in error_str:
            return FailureMode.MEMORY_ERROR
        
        # GPU errors
        if any(term in error_str for term in ["cuda", "gpu", "device", "nvidia"]):
            return FailureMode.GPU_ERROR
        
        # Permission errors
        if error_type in ["PermissionError", "AccessDenied"] or "permission" in error_str:
            return FailureMode.PERMISSION_DENIED
        
        # Configuration errors
        if error_type in ["ConfigurationError", "ValueError"] and "config" in error_str:
            return FailureMode.CONFIGURATION_ERROR
        
        # Import/dependency errors
        if error_type in ["ImportError", "ModuleNotFoundError"]:
            return FailureMode.DEPENDENCY_UNAVAILABLE
        
        return FailureMode.UNKNOWN_ERROR
    
    def handle_failure(
        self,
        error: Exception,
        component: str,
        context: Dict[str, Any],
        custom_strategy: Optional[RecoveryStrategy] = None
    ) -> FailureRecord:
        """Handle a failure with appropriate recovery strategy."""
        failure_mode = self.classify_failure(error, context)
        recovery_strategy = custom_strategy or self.recovery_strategies[failure_mode]
        
        failure_record = FailureRecord(
            failure_id=f"failure_{int(time.time())}_{random.randint(1000, 9999)}",
            timestamp=datetime.now(),
            failure_mode=failure_mode,
            component=component,
            error_message=str(error),
            context=context.copy(),
            recovery_strategy=recovery_strategy
        )
        
        start_time = time.time()
        recovery_successful = False
        
        try:
            if recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                recovery_successful = self._graceful_degradation(error, context)
            elif recovery_strategy == RecoveryStrategy.FALLBACK_TO_ALTERNATIVE:
                recovery_successful = self._fallback_to_alternative(error, context)
            elif recovery_strategy == RecoveryStrategy.CHECKPOINT_RECOVERY:
                recovery_successful = self._checkpoint_recovery(error, context)
            elif recovery_strategy == RecoveryStrategy.IGNORE_AND_CONTINUE:
                recovery_successful = True  # By definition, we ignore and continue
            else:
                # For retry strategies, mark as successful (actual retry happens at call site)
                recovery_successful = True
            
        except Exception as recovery_error:
            logger.error(f"Recovery strategy failed: {recovery_error}")
            recovery_successful = False
        
        failure_record.recovery_successful = recovery_successful
        failure_record.recovery_time_seconds = time.time() - start_time
        
        self.failure_records.append(failure_record)
        
        logger.info(
            f"Handled failure {failure_record.failure_id}: "
            f"{failure_mode.value} -> {recovery_strategy.value} "
            f"(Success: {recovery_successful})"
        )
        
        return failure_record
    
    def _graceful_degradation(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Implement graceful degradation."""
        # Reduce functionality to core essentials
        degraded_features = context.get("degraded_features", [])
        
        if "prometheus_export" not in degraded_features:
            degraded_features.append("prometheus_export")
            logger.info("Degraded: Disabled Prometheus export")
            return True
        
        if "detailed_logging" not in degraded_features:
            degraded_features.append("detailed_logging")
            logger.info("Degraded: Reduced logging verbosity")
            return True
        
        if "advanced_analytics" not in degraded_features:
            degraded_features.append("advanced_analytics")
            logger.info("Degraded: Disabled advanced analytics")
            return True
        
        return False
    
    def _fallback_to_alternative(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Implement fallback to alternative systems."""
        component = context.get("component", "unknown")
        
        if component == "carbon_intensity_api":
            # Fall back to default carbon intensity values
            context["fallback_carbon_intensity"] = 400  # Global average
            logger.info("Fallback: Using default carbon intensity")
            return True
        
        if component == "gpu_monitoring":
            # Fall back to CPU-based estimates
            context["fallback_monitoring"] = "cpu_estimate"
            logger.info("Fallback: Using CPU-based power estimates")
            return True
        
        if component == "prometheus_export":
            # Fall back to file-based metrics
            context["fallback_export"] = "file_based"
            logger.info("Fallback: Using file-based metric export")
            return True
        
        return False
    
    def _checkpoint_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Implement checkpoint-based recovery."""
        session_id = context.get("session_id")
        if not session_id:
            logger.warning("No session ID for checkpoint recovery")
            return False
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        session_checkpoints = [
            cp for cp in checkpoints 
            if cp.get("metadata", {}).get("session_id") == session_id
        ]
        
        if session_checkpoints:
            latest_checkpoint = session_checkpoints[0]
            checkpoint_data = self.checkpoint_manager.load_checkpoint(
                latest_checkpoint["checkpoint_id"]
            )
            
            if checkpoint_data:
                context["recovered_from_checkpoint"] = latest_checkpoint["checkpoint_id"]
                context["recovery_data"] = checkpoint_data["data"]
                logger.info(f"Recovered from checkpoint: {latest_checkpoint['checkpoint_id']}")
                return True
        
        logger.warning("No suitable checkpoint found for recovery")
        return False
    
    def create_resilient_wrapper(
        self,
        component_name: str,
        failure_mode: Optional[FailureMode] = None,
        recovery_strategy: Optional[RecoveryStrategy] = None
    ):
        """Create a decorator for resilient function execution."""
        
        def decorator(func):
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        "component": component_name,
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                    
                    failure_record = self.handle_failure(
                        e, component_name, context, recovery_strategy
                    )
                    
                    # Implement recovery based on strategy
                    if failure_record.recovery_strategy in [
                        RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF,
                        RecoveryStrategy.RETRY_LINEAR_BACKOFF
                    ]:
                        return self.retry_manager.retry_sync(
                            func, args, kwargs, 
                            failure_record.recovery_strategy
                        )
                    elif failure_record.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                        return self.circuit_breaker_manager.execute_with_circuit_breaker(
                            component_name, func, args, kwargs
                        )
                    elif failure_record.recovery_strategy == RecoveryStrategy.IGNORE_AND_CONTINUE:
                        logger.warning(f"Ignoring error in {component_name}: {e}")
                        return None
                    else:
                        # Re-raise for strategies that don't auto-recover
                        raise
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context = {
                        "component": component_name,
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                    
                    failure_record = self.handle_failure(
                        e, component_name, context, recovery_strategy
                    )
                    
                    # Implement recovery based on strategy
                    if failure_record.recovery_strategy in [
                        RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF,
                        RecoveryStrategy.RETRY_LINEAR_BACKOFF
                    ]:
                        return await self.retry_manager.retry_async(
                            func, args, kwargs,
                            failure_record.recovery_strategy
                        )
                    elif failure_record.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                        return await self.circuit_breaker_manager.execute_async_with_circuit_breaker(
                            component_name, func, args, kwargs
                        )
                    elif failure_record.recovery_strategy == RecoveryStrategy.IGNORE_AND_CONTINUE:
                        logger.warning(f"Ignoring error in {component_name}: {e}")
                        return None
                    else:
                        # Re-raise for strategies that don't auto-recover
                        raise
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_resilience_dashboard(self) -> Dict[str, Any]:
        """Get resilience dashboard data."""
        recent_failures = [
            f for f in self.failure_records 
            if f.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        failure_mode_counts = {}
        recovery_success_counts = {}
        
        for failure in recent_failures:
            mode = failure.failure_mode.value
            failure_mode_counts[mode] = failure_mode_counts.get(mode, 0) + 1
            
            strategy = failure.recovery_strategy.value
            if strategy not in recovery_success_counts:
                recovery_success_counts[strategy] = {"success": 0, "failure": 0}
            
            if failure.recovery_successful:
                recovery_success_counts[strategy]["success"] += 1
            else:
                recovery_success_counts[strategy]["failure"] += 1
        
        circuit_breaker_status = self.circuit_breaker_manager.get_circuit_breaker_status()
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        return {
            "resilience_overview": {
                "total_failures_24h": len(recent_failures),
                "unique_failure_modes": len(failure_mode_counts),
                "recovery_success_rate": (
                    sum(1 for f in recent_failures if f.recovery_successful) / 
                    max(1, len(recent_failures))
                ) * 100,
                "active_circuit_breakers": sum(
                    1 for status in circuit_breaker_status.values() 
                    if status["state"] != "CLOSED"
                ),
                "available_checkpoints": len(checkpoints)
            },
            "failure_breakdown": failure_mode_counts,
            "recovery_strategies": recovery_success_counts,
            "circuit_breakers": circuit_breaker_status,
            "recent_checkpoints": checkpoints[:5]  # Most recent 5
        }
    
    def export_resilience_report(self, filepath: Path) -> None:
        """Export comprehensive resilience report."""
        dashboard = self.get_resilience_dashboard()
        
        report = {
            "resilience_report": dashboard,
            "failure_records": [asdict(f) for f in self.failure_records[-100:]],  # Last 100 failures
            "recovery_strategies_config": {
                mode.value: strategy.value 
                for mode, strategy in self.recovery_strategies.items()
            },
            "generated_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Resilience report exported to {filepath}")


# Global fault tolerance manager
_fault_tolerance_manager: Optional[FaultToleranceManager] = None


def get_fault_tolerance_manager(checkpoint_dir: Optional[Path] = None) -> FaultToleranceManager:
    """Get or create the global fault tolerance manager."""
    global _fault_tolerance_manager
    
    if _fault_tolerance_manager is None:
        _fault_tolerance_manager = FaultToleranceManager(checkpoint_dir)
    
    return _fault_tolerance_manager


def resilient(
    component_name: str,
    failure_mode: Optional[FailureMode] = None,
    recovery_strategy: Optional[RecoveryStrategy] = None
):
    """Decorator for resilient function execution."""
    manager = get_fault_tolerance_manager()
    return manager.create_resilient_wrapper(component_name, failure_mode, recovery_strategy)