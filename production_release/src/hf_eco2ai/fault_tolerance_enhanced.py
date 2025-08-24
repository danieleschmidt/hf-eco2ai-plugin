"""Enhanced enterprise fault tolerance and resilience systems for carbon tracking."""

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
import hashlib
import shutil
import subprocess
import requests
import socket
import psutil
import os
import gzip
import pickle
from collections import deque
import uuid
import schedule
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Enhanced types of failures the system can encounter."""
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
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATABASE_ERROR = "database_error"
    AUTHENTICATION_FAILURE = "authentication_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    EXTERNAL_API_ERROR = "external_api_error"
    BACKUP_FAILURE = "backup_failure"
    RECOVERY_FAILURE = "recovery_failure"


class RecoveryStrategy(Enum):
    """Enhanced recovery strategies for different failure modes."""
    RETRY_EXPONENTIAL_BACKOFF = "retry_exponential_backoff"
    RETRY_LINEAR_BACKOFF = "retry_linear_backoff"
    RETRY_ADAPTIVE_BACKOFF = "retry_adaptive_backoff"
    FALLBACK_TO_ALTERNATIVE = "fallback_to_alternative"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    CHECKPOINT_RECOVERY = "checkpoint_recovery"
    FAIL_FAST = "fail_fast"
    IGNORE_AND_CONTINUE = "ignore_and_continue"
    AUTOMATIC_RESTART = "automatic_restart"
    ROLLBACK_AND_RETRY = "rollback_and_retry"
    REDUNDANT_EXECUTION = "redundant_execution"
    STATE_RESTORATION = "state_restoration"
    EXTERNAL_FAILOVER = "external_failover"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class FailureRecord:
    """Enhanced record of a failure event."""
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
    severity: str = "medium"
    impact_assessment: str = "unknown"
    root_cause: Optional[str] = None
    mitigation_actions: List[str] = None
    lessons_learned: Optional[str] = None
    similar_failures_count: int = 0
    
    def __post_init__(self):
        if self.mitigation_actions is None:
            self.mitigation_actions = []


@dataclass
class CircuitBreakerState:
    """Enhanced circuit breaker state tracking."""
    name: str
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    timeout_seconds: int = 60
    success_count: int = 0
    failure_rate_threshold: float = 0.5  # 50% failure rate
    request_count: int = 0
    success_count_total: int = 0
    
    def should_allow_request(self) -> bool:
        """Enhanced request filtering with failure rate consideration."""
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
        """Record a successful operation with rate tracking."""
        self.request_count += 1
        self.success_count_total += 1
        
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
        """Record a failed operation with enhanced logic."""
        self.failure_count += 1
        self.request_count += 1
        self.last_failure_time = datetime.now()
        
        # Calculate failure rate
        if self.request_count >= 10:  # Minimum requests for rate calculation
            failure_rate = (self.request_count - self.success_count_total) / self.request_count
            
            if failure_rate >= self.failure_rate_threshold or self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
        elif self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"


@dataclass
class BackupMetadata:
    """Metadata for backup files."""
    backup_id: str
    timestamp: datetime
    component: str
    file_path: str
    file_size: int
    checksum: str
    compression: str
    encryption: bool
    retention_days: int
    backup_type: str  # "full", "incremental", "differential"


@dataclass
class RedundantMetric:
    """Redundant metric collection configuration."""
    metric_name: str
    primary_collector: str
    backup_collectors: List[str]
    consensus_threshold: float = 0.8
    max_deviation_percent: float = 10.0


class AutoBackupManager:
    """Automated backup and restore system with comprehensive features."""
    
    def __init__(self, backup_dir: Path, retention_days: int = 30):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.backup_metadata: Dict[str, BackupMetadata] = {}
        self.backup_schedule = {}
        self._lock = threading.Lock()
        
        # Start cleanup scheduler
        self._start_cleanup_scheduler()
        
        logger.info(f"AutoBackupManager initialized with backup dir: {backup_dir}")
    
    def _start_cleanup_scheduler(self):
        """Start automated cleanup of old backups."""
        def cleanup_old_backups():
            try:
                self._cleanup_old_backups()
            except Exception as e:
                logger.error(f"Backup cleanup failed: {e}")
        
        # Schedule cleanup daily at 2 AM
        schedule.every().day.at("02:00").do(cleanup_old_backups)
        
        # Start scheduler thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def create_backup(self, component: str, source_path: str, 
                     backup_type: str = "full", 
                     compression: bool = True, 
                     encryption: bool = False) -> str:
        """Create a backup of specified component."""
        backup_id = f"{component}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        timestamp = datetime.now()
        
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")
        
        # Determine backup file extension
        if compression:
            backup_filename = f"{backup_id}.tar.gz"
        else:
            backup_filename = f"{backup_id}.tar"
        
        backup_path = self.backup_dir / backup_filename
        
        try:
            # Create backup
            if source.is_file():
                self._backup_file(source, backup_path, compression)
            else:
                self._backup_directory(source, backup_path, compression)
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_path)
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                component=component,
                file_path=str(backup_path),
                file_size=backup_path.stat().st_size,
                checksum=checksum,
                compression="gzip" if compression else "none",
                encryption=encryption,
                retention_days=self.retention_days,
                backup_type=backup_type
            )
            
            with self._lock:
                self.backup_metadata[backup_id] = metadata
            
            # Save metadata to file
            self._save_metadata(backup_id, metadata)
            
            logger.info(f"Backup created: {backup_id} for component {component}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Backup creation failed for {component}: {e}")
            if backup_path.exists():
                backup_path.unlink()
            raise
    
    def _backup_file(self, source: Path, backup_path: Path, compression: bool):
        """Backup a single file."""
        if compression:
            with open(source, 'rb') as f_in:
                with gzip.open(backup_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(source, backup_path)
    
    def _backup_directory(self, source: Path, backup_path: Path, compression: bool):
        """Backup a directory."""
        if compression:
            # Use tar with gzip compression
            subprocess.run([
                'tar', '-czf', str(backup_path), '-C', str(source.parent), source.name
            ], check=True)
        else:
            # Use tar without compression
            subprocess.run([
                'tar', '-cf', str(backup_path), '-C', str(source.parent), source.name
            ], check=True)
    
    def restore_backup(self, backup_id: str, restore_path: str, 
                      verify_checksum: bool = True) -> bool:
        """Restore a backup to specified location."""
        with self._lock:
            metadata = self.backup_metadata.get(backup_id)
        
        if not metadata:
            logger.error(f"Backup metadata not found: {backup_id}")
            return False
        
        backup_path = Path(metadata.file_path)
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Verify checksum if requested
            if verify_checksum:
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum != metadata.checksum:
                    logger.error(f"Backup checksum mismatch for {backup_id}")
                    return False
            
            restore_destination = Path(restore_path)
            restore_destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Restore based on compression
            if metadata.compression == "gzip":
                if backup_path.suffix == '.gz':
                    # Single file backup
                    with gzip.open(backup_path, 'rb') as f_in:
                        with open(restore_destination, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    # Directory backup
                    subprocess.run([
                        'tar', '-xzf', str(backup_path), '-C', str(restore_destination.parent)
                    ], check=True)
            else:
                # No compression
                if backup_path.suffix == '.tar':
                    subprocess.run([
                        'tar', '-xf', str(backup_path), '-C', str(restore_destination.parent)
                    ], check=True)
                else:
                    shutil.copy2(backup_path, restore_destination)
            
            logger.info(f"Backup {backup_id} restored to {restore_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup restoration failed for {backup_id}: {e}")
            return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _save_metadata(self, backup_id: str, metadata: BackupMetadata):
        """Save backup metadata to file."""
        metadata_path = self.backup_dir / f"{backup_id}.metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
    
    def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0
        
        with self._lock:
            expired_backups = [
                backup_id for backup_id, metadata in self.backup_metadata.items()
                if metadata.timestamp < cutoff_date
            ]
        
        for backup_id in expired_backups:
            try:
                metadata = self.backup_metadata[backup_id]
                backup_path = Path(metadata.file_path)
                metadata_path = self.backup_dir / f"{backup_id}.metadata.json"
                
                # Delete files
                if backup_path.exists():
                    backup_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Remove from metadata
                del self.backup_metadata[backup_id]
                deleted_count += 1
                
            except Exception as e:
                logger.error(f"Failed to delete backup {backup_id}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} expired backups")
    
    def list_backups(self, component: Optional[str] = None) -> List[BackupMetadata]:
        """List available backups."""
        with self._lock:
            backups = list(self.backup_metadata.values())
        
        if component:
            backups = [b for b in backups if b.component == component]
        
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)


class RedundantMetricCollector:
    """Redundant metric collection with consensus and reliability scoring."""
    
    def __init__(self):
        self.collectors: Dict[str, RedundantMetric] = {}
        self.metric_history: Dict[str, deque] = {}
        self._lock = threading.Lock()
    
    def register_redundant_metric(self, metric_config: RedundantMetric):
        """Register a redundant metric configuration."""
        with self._lock:
            self.collectors[metric_config.metric_name] = metric_config
            self.metric_history[metric_config.metric_name] = deque(maxlen=100)
        
        logger.info(f"Registered redundant metric: {metric_config.metric_name}")
    
    def collect_metric(self, metric_name: str, 
                      collector_values: Dict[str, float]) -> Tuple[float, bool, str]:
        """Collect metric from multiple sources and determine consensus."""
        if metric_name not in self.collectors:
            raise ValueError(f"Metric {metric_name} not registered for redundant collection")
        
        config = self.collectors[metric_name]
        
        # Validate we have values from required collectors
        primary_value = collector_values.get(config.primary_collector)
        backup_values = {
            collector: collector_values.get(collector)
            for collector in config.backup_collectors
            if collector_values.get(collector) is not None
        }
        
        if primary_value is None and not backup_values:
            return 0.0, False, "No collector values available"
        
        # Determine consensus value
        all_values = [v for v in [primary_value] + list(backup_values.values()) if v is not None]
        
        if len(all_values) == 1:
            consensus_value = all_values[0]
            confidence = 0.5  # Low confidence with only one value
            status = "single_source"
        else:
            # Calculate consensus
            consensus_value, confidence, status = self._calculate_consensus(all_values, config)
        
        # Store in history
        with self._lock:
            self.metric_history[metric_name].append({
                "timestamp": time.time(),
                "value": consensus_value,
                "confidence": confidence,
                "sources": len(all_values)
            })
        
        is_reliable = confidence >= config.consensus_threshold
        
        return consensus_value, is_reliable, status
    
    def _calculate_consensus(self, values: List[float], 
                           config: RedundantMetric) -> Tuple[float, float, str]:
        """Calculate consensus value and confidence."""
        if len(values) < 2:
            return values[0], 0.5, "insufficient_data"
        
        # Calculate median as consensus
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            consensus = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            consensus = sorted_values[n//2]
        
        # Calculate confidence based on agreement
        max_deviation = max(abs(v - consensus) for v in values)
        max_allowed = consensus * (config.max_deviation_percent / 100)
        
        if max_allowed == 0:
            confidence = 1.0 if max_deviation == 0 else 0.0
        else:
            confidence = max(0.0, 1.0 - (max_deviation / max_allowed))
        
        # Determine status
        if confidence >= 0.9:
            status = "high_consensus"
        elif confidence >= 0.7:
            status = "moderate_consensus"
        elif confidence >= 0.5:
            status = "low_consensus"
        else:
            status = "no_consensus"
        
        return consensus, confidence, status


class ExternalAPIManager:
    """Enhanced external API management with intelligent backoff and failover."""
    
    def __init__(self):
        self.api_configs = {
            "carbon_intensity": {
                "base_url": "https://api.carbonintensity.org.uk",
                "timeout": 10,
                "rate_limit_per_minute": 60,
                "backoff_strategy": "exponential",
                "fallback_urls": ["https://api2.carbonintensity.org.uk"]
            },
            "weather_api": {
                "base_url": "https://api.openweathermap.org",
                "timeout": 15,
                "rate_limit_per_minute": 100,
                "backoff_strategy": "linear",
                "fallback_urls": []
            }
        }
        self.request_history: Dict[str, deque] = {}
        self.backoff_state: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def make_request(self, api_name: str, endpoint: str, 
                    params: Optional[Dict] = None, 
                    max_retries: int = 3) -> Tuple[bool, Any, str]:
        """Make API request with intelligent backoff and failover."""
        if api_name not in self.api_configs:
            return False, None, f"Unknown API: {api_name}"
        
        config = self.api_configs[api_name]
        
        # Check rate limiting
        if not self._check_rate_limit(api_name, config):
            return False, None, "Rate limit exceeded"
        
        # Check if in backoff period
        backoff_delay = self._get_backoff_delay(api_name)
        if backoff_delay > 0:
            return False, None, f"In backoff period: {backoff_delay:.1f}s remaining"
        
        # Try primary URL first, then fallbacks
        urls_to_try = [config["base_url"]] + config.get("fallback_urls", [])
        
        for url_base in urls_to_try:
            url = f"{url_base}/{endpoint.lstrip('/')}"
            
            for attempt in range(max_retries + 1):
                try:
                    response = requests.get(
                        url,
                        params=params or {},
                        timeout=config["timeout"]
                    )
                    
                    if response.status_code == 200:
                        # Success - reset backoff
                        self._reset_backoff(api_name)
                        self._record_request(api_name, True)
                        return True, response.json(), "Success"
                    
                    elif response.status_code == 429:  # Rate limited
                        self._apply_backoff(api_name, "rate_limit")
                        break  # Try next URL
                    
                    elif response.status_code >= 500:  # Server error
                        if attempt < max_retries:
                            delay = self._calculate_retry_delay(attempt, config["backoff_strategy"])
                            time.sleep(delay)
                            continue
                        else:
                            # Try next URL if available
                            break
                    
                    else:
                        break  # Try next URL
                        
                except requests.exceptions.Timeout:
                    if attempt < max_retries:
                        delay = self._calculate_retry_delay(attempt, "exponential")
                        time.sleep(delay)
                        continue
                    else:
                        # Try next URL
                        break
                
                except requests.exceptions.ConnectionError:
                    # Try next URL immediately
                    break
                
                except Exception as e:
                    logger.error(f"Unexpected error in API request: {e}")
                    break
        
        # All URLs and retries failed
        self._apply_backoff(api_name, "all_endpoints_failed")
        return False, None, "All API endpoints failed"
    
    def _check_rate_limit(self, api_name: str, config: Dict[str, Any]) -> bool:
        """Check if request is within rate limit."""
        with self._lock:
            if api_name not in self.request_history:
                self.request_history[api_name] = deque(maxlen=config["rate_limit_per_minute"])
            
            history = self.request_history[api_name]
            current_time = time.time()
            
            # Remove requests older than 1 minute
            while history and current_time - history[0] > 60:
                history.popleft()
            
            return len(history) < config["rate_limit_per_minute"]
    
    def _record_request(self, api_name: str, success: bool):
        """Record a request in history."""
        with self._lock:
            if api_name not in self.request_history:
                self.request_history[api_name] = deque(maxlen=100)
            
            self.request_history[api_name].append(time.time())
    
    def _get_backoff_delay(self, api_name: str) -> float:
        """Get remaining backoff delay."""
        with self._lock:
            backoff = self.backoff_state.get(api_name)
            if not backoff:
                return 0.0
            
            elapsed = time.time() - backoff["start_time"]
            remaining = backoff["duration"] - elapsed
            
            return max(0.0, remaining)
    
    def _apply_backoff(self, api_name: str, reason: str):
        """Apply backoff strategy with enhanced logic."""
        with self._lock:
            current_backoff = self.backoff_state.get(api_name, {"count": 0})
            count = current_backoff.get("count", 0) + 1
            
            # Calculate backoff duration based on reason and count
            if reason == "rate_limit":
                duration = min(300, 60 * count)  # 1-5 minutes
            elif reason == "server_error":
                duration = min(600, 30 * (2 ** count))  # Exponential up to 10 minutes
            elif reason == "timeout":
                duration = min(120, 15 * count)  # Linear up to 2 minutes
            elif reason == "all_endpoints_failed":
                duration = min(1800, 60 * (2 ** count))  # Exponential up to 30 minutes
            else:
                duration = min(180, 20 * count)  # Default backoff
            
            self.backoff_state[api_name] = {
                "count": count,
                "start_time": time.time(),
                "duration": duration,
                "reason": reason
            }
            
            logger.warning(f"Applied backoff to {api_name}: {duration}s due to {reason}")
    
    def _reset_backoff(self, api_name: str):
        """Reset backoff state after successful request."""
        with self._lock:
            if api_name in self.backoff_state:
                del self.backoff_state[api_name]
    
    def _calculate_retry_delay(self, attempt: int, strategy: str) -> float:
        """Calculate delay for retry attempt."""
        if strategy == "exponential":
            return min(30, 0.5 * (2 ** attempt))  # 0.5, 1, 2, 4, 8... up to 30s
        elif strategy == "linear":
            return min(10, 1 + attempt)  # 1, 2, 3, 4... up to 10s
        else:
            return 1.0  # Default 1 second


class CrashRecoveryManager:
    """Handles system crash recovery and state restoration."""
    
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_handlers: Dict[str, Callable] = {}
        self.state_checkpoints: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def register_recovery_handler(self, component: str, handler: Callable):
        """Register a recovery handler for a component."""
        self.recovery_handlers[component] = handler
        logger.info(f"Registered recovery handler for {component}")
    
    def save_state_checkpoint(self, component: str, state_data: Dict[str, Any]):
        """Save state checkpoint for a component."""
        checkpoint_file = self.state_dir / f"{component}_state.json"
        
        try:
            with self._lock:
                # Add timestamp and checksum
                state_data["_checkpoint_timestamp"] = time.time()
                state_data["_checkpoint_id"] = str(uuid.uuid4())
                
                # Calculate checksum
                state_json = json.dumps(state_data, sort_keys=True, default=str)
                state_data["_checksum"] = hashlib.sha256(state_json.encode()).hexdigest()
                
                # Save to file with atomic write
                temp_file = checkpoint_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state_data, f, indent=2, default=str)
                
                # Atomic move
                temp_file.rename(checkpoint_file)
                
                # Keep in memory for quick access
                self.state_checkpoints[component] = state_data
            
            logger.debug(f"State checkpoint saved for {component}")
            
        except Exception as e:
            logger.error(f"Failed to save state checkpoint for {component}: {e}")
    
    def restore_state_checkpoint(self, component: str) -> Optional[Dict[str, Any]]:
        """Restore state checkpoint for a component."""
        checkpoint_file = self.state_dir / f"{component}_state.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"No state checkpoint found for {component}")
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                state_data = json.load(f)
            
            # Verify checksum if present
            if "_checksum" in state_data:
                saved_checksum = state_data.pop("_checksum")
                state_json = json.dumps(state_data, sort_keys=True, default=str)
                calculated_checksum = hashlib.sha256(state_json.encode()).hexdigest()
                
                if saved_checksum != calculated_checksum:
                    logger.error(f"State checkpoint corrupted for {component}")
                    return None
            
            # Remove metadata
            state_data.pop("_checkpoint_timestamp", None)
            state_data.pop("_checkpoint_id", None)
            
            logger.info(f"State checkpoint restored for {component}")
            return state_data
            
        except Exception as e:
            logger.error(f"Failed to restore state checkpoint for {component}: {e}")
            return None
    
    def perform_crash_recovery(self) -> Dict[str, bool]:
        """Perform crash recovery for all registered components."""
        recovery_results = {}
        
        logger.info("Starting crash recovery process")
        
        for component, handler in self.recovery_handlers.items():
            try:
                # Restore state checkpoint
                state_data = self.restore_state_checkpoint(component)
                
                # Call recovery handler
                success = handler(state_data)
                recovery_results[component] = success
                
                if success:
                    logger.info(f"Successfully recovered {component}")
                else:
                    logger.error(f"Failed to recover {component}")
                    
            except Exception as e:
                logger.error(f"Recovery handler failed for {component}: {e}")
                recovery_results[component] = False
        
        successful_recoveries = sum(1 for success in recovery_results.values() if success)
        total_components = len(recovery_results)
        
        logger.info(f"Crash recovery completed: {successful_recoveries}/{total_components} components recovered")
        
        return recovery_results


class EnhancedRetryManager:
    """Advanced retry logic with adaptive behavior and pattern recognition."""
    
    def __init__(self):
        self.retry_configs: Dict[str, Dict[str, Any]] = {
            "default": {
                "max_attempts": 3,
                "base_delay": 1.0,
                "max_delay": 60.0,
                "exponential_base": 2.0,
                "jitter": True,
                "adaptive": False
            },
            "network": {
                "max_attempts": 5,
                "base_delay": 0.5,
                "max_delay": 30.0,
                "exponential_base": 2.0,
                "jitter": True,
                "adaptive": True
            },
            "critical": {
                "max_attempts": 10,
                "base_delay": 0.1,
                "max_delay": 120.0,
                "exponential_base": 1.5,
                "jitter": True,
                "adaptive": True
            },
            "external_api": {
                "max_attempts": 7,
                "base_delay": 2.0,
                "max_delay": 300.0,
                "exponential_base": 1.8,
                "jitter": True,
                "adaptive": True
            }
        }
        self.failure_history: Dict[str, deque] = {}
        self._lock = threading.Lock()
    
    def calculate_delay(
        self, 
        attempt: int, 
        strategy: RecoveryStrategy,
        config_name: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Enhanced delay calculation with adaptive behavior."""
        config = self.retry_configs.get(config_name, self.retry_configs["default"])
        
        if strategy == RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF:
            delay = config["base_delay"] * (config["exponential_base"] ** (attempt - 1))
        elif strategy == RecoveryStrategy.RETRY_LINEAR_BACKOFF:
            delay = config["base_delay"] * attempt
        elif strategy == RecoveryStrategy.RETRY_ADAPTIVE_BACKOFF:
            delay = self._calculate_adaptive_delay(attempt, config, context)
        else:
            delay = config["base_delay"]
        
        # Apply max delay limit
        delay = min(delay, config["max_delay"])
        
        # Add jitter to prevent thundering herd
        if config.get("jitter", True):
            jitter = random.uniform(0.8, 1.2)
            delay *= jitter
        
        return delay
    
    def _calculate_adaptive_delay(self, attempt: int, config: Dict[str, Any], 
                                 context: Optional[Dict[str, Any]]) -> float:
        """Calculate adaptive delay based on historical performance."""
        base_delay = config["base_delay"]
        
        # Get historical failure data for this context
        if context and "component" in context:
            component = context["component"]
            with self._lock:
                if component not in self.failure_history:
                    self.failure_history[component] = deque(maxlen=50)
                
                history = list(self.failure_history[component])
            
            if len(history) >= 3:
                # Calculate success rate in recent attempts
                recent_history = history[-10:]
                success_count = sum(1 for record in recent_history if record.get("success", False))
                success_rate = success_count / len(recent_history)
                
                # Adjust delay based on success rate
                if success_rate < 0.3:  # High failure rate
                    multiplier = 2.0
                elif success_rate < 0.6:  # Moderate failure rate
                    multiplier = 1.5
                else:  # Good success rate
                    multiplier = 1.0
                
                return base_delay * multiplier * (config["exponential_base"] ** (attempt - 1))
        
        # Default to exponential backoff
        return base_delay * (config["exponential_base"] ** (attempt - 1))
    
    def record_attempt(self, component: str, success: bool, attempt: int, 
                      error_type: Optional[str] = None):
        """Record retry attempt for adaptive behavior."""
        with self._lock:
            if component not in self.failure_history:
                self.failure_history[component] = deque(maxlen=50)
            
            self.failure_history[component].append({
                "timestamp": time.time(),
                "success": success,
                "attempt": attempt,
                "error_type": error_type
            })


class EnhancedCircuitBreakerManager:
    """Enhanced circuit breaker management with failure rate analysis."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self._lock = threading.Lock()
    
    def get_circuit_breaker(
        self, 
        name: str, 
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        failure_rate_threshold: float = 0.5
    ) -> CircuitBreakerState:
        """Get or create a circuit breaker with enhanced configuration."""
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreakerState(
                    name=name,
                    failure_threshold=failure_threshold,
                    timeout_seconds=timeout_seconds,
                    failure_rate_threshold=failure_rate_threshold
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
        """Execute function with enhanced circuit breaker protection."""
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


class EnhancedFaultToleranceManager:
    """Enhanced fault tolerance orchestrator with comprehensive recovery capabilities."""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None, 
                 backup_dir: Optional[Path] = None,
                 state_dir: Optional[Path] = None):
        # Core managers
        self.retry_manager = EnhancedRetryManager()
        self.circuit_breaker_manager = EnhancedCircuitBreakerManager()
        
        # Enhanced managers
        self.backup_manager = AutoBackupManager(
            backup_dir or Path("./backups")
        )
        self.crash_recovery_manager = CrashRecoveryManager(
            state_dir or Path("./state")
        )
        self.redundant_collector = RedundantMetricCollector()
        self.external_api_manager = ExternalAPIManager()
        
        # Failure tracking
        self.failure_records: List[FailureRecord] = []
        self.recovery_strategies: Dict[FailureMode, RecoveryStrategy] = {
            FailureMode.NETWORK_TIMEOUT: RecoveryStrategy.RETRY_ADAPTIVE_BACKOFF,
            FailureMode.API_RATE_LIMIT: RecoveryStrategy.RETRY_LINEAR_BACKOFF,
            FailureMode.DISK_FULL: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureMode.MEMORY_ERROR: RecoveryStrategy.CHECKPOINT_RECOVERY,
            FailureMode.GPU_ERROR: RecoveryStrategy.CIRCUIT_BREAKER,
            FailureMode.DEPENDENCY_UNAVAILABLE: RecoveryStrategy.FALLBACK_TO_ALTERNATIVE,
            FailureMode.CONFIGURATION_ERROR: RecoveryStrategy.FAIL_FAST,
            FailureMode.DATA_CORRUPTION: RecoveryStrategy.ROLLBACK_AND_RETRY,
            FailureMode.PERMISSION_DENIED: RecoveryStrategy.FAIL_FAST,
            FailureMode.UNKNOWN_ERROR: RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF,
            FailureMode.SERVICE_UNAVAILABLE: RecoveryStrategy.EXTERNAL_FAILOVER,
            FailureMode.DATABASE_ERROR: RecoveryStrategy.STATE_RESTORATION,
            FailureMode.AUTHENTICATION_FAILURE: RecoveryStrategy.FAIL_FAST,
            FailureMode.RESOURCE_EXHAUSTION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureMode.TIMEOUT_ERROR: RecoveryStrategy.RETRY_ADAPTIVE_BACKOFF,
            FailureMode.VALIDATION_ERROR: RecoveryStrategy.FAIL_FAST,
            FailureMode.EXTERNAL_API_ERROR: RecoveryStrategy.RETRY_ADAPTIVE_BACKOFF,
            FailureMode.BACKUP_FAILURE: RecoveryStrategy.FALLBACK_TO_ALTERNATIVE,
            FailureMode.RECOVERY_FAILURE: RecoveryStrategy.MANUAL_INTERVENTION
        }
        
        # Monitoring and alerting
        self._failure_detection_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._start_monitoring()
        
        logger.info("Enhanced fault tolerance manager initialized")
    
    def _start_monitoring(self):
        """Start fault tolerance monitoring."""
        self._failure_detection_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="fault-tolerance-monitor"
        )
        self._monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Monitor system for fault tolerance health."""
        while self._failure_detection_active:
            try:
                # Check circuit breaker health
                self._check_circuit_breaker_health()
                
                # Check backup system health  
                self._check_backup_system_health()
                
                # Check external API health
                self._check_external_api_health()
                
                # Analyze failure patterns
                self._analyze_failure_patterns()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in fault tolerance monitoring: {e}")
                time.sleep(60)
    
    def _check_circuit_breaker_health(self):
        """Check health of circuit breakers."""
        with self.circuit_breaker_manager._lock:
            status = {
                name: {
                    "state": breaker.state,
                    "failure_count": breaker.failure_count,
                    "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
                    "success_count": breaker.success_count
                }
                for name, breaker in self.circuit_breaker_manager.circuit_breakers.items()
            }
        
        open_breakers = [
            name for name, info in status.items() 
            if info["state"] == "OPEN"
        ]
        
        if len(open_breakers) > 3:  # Threshold for concern
            logger.warning(f"Multiple circuit breakers open: {open_breakers}")
    
    def _check_backup_system_health(self):
        """Check backup system health."""
        try:
            recent_backups = self.backup_manager.list_backups()[:10]
            if not recent_backups:
                logger.warning("No recent backups found")
            else:
                latest_backup = recent_backups[0]
                hours_since_backup = (datetime.now() - latest_backup.timestamp).total_seconds() / 3600
                
                if hours_since_backup > 24:
                    logger.warning(f"Latest backup is {hours_since_backup:.1f} hours old")
        except Exception as e:
            logger.error(f"Error checking backup system health: {e}")
    
    def _check_external_api_health(self):
        """Check external API health."""
        with self.external_api_manager._lock:
            api_status = {}
            for api_name in self.external_api_manager.api_configs.keys():
                backoff = self.external_api_manager.backoff_state.get(api_name)
                history = self.external_api_manager.request_history.get(api_name, deque())
                
                api_status[api_name] = {
                    "available": backoff is None,
                    "backoff_remaining": self.external_api_manager._get_backoff_delay(api_name),
                    "recent_requests": len(history),
                    "backoff_reason": backoff.get("reason") if backoff else None
                }
        
        unavailable_apis = [
            api for api, status in api_status.items()
            if not status["available"]
        ]
        
        if len(unavailable_apis) > 1:
            logger.warning(f"Multiple APIs unavailable: {unavailable_apis}")
    
    def _analyze_failure_patterns(self):
        """Analyze recent failure patterns for insights."""
        if len(self.failure_records) < 5:
            return
        
        recent_failures = [
            f for f in self.failure_records[-20:]
            if (datetime.now() - f.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        if len(recent_failures) >= 5:
            # Group by failure mode
            failure_counts = {}
            for failure in recent_failures:
                mode = failure.failure_mode
                failure_counts[mode] = failure_counts.get(mode, 0) + 1
            
            # Look for concerning patterns
            for mode, count in failure_counts.items():
                if count >= 3:
                    logger.warning(f"High frequency of {mode.value} failures: {count} in last hour")
    
    def classify_failure(self, error: Exception, context: Dict[str, Any]) -> FailureMode:
        """Enhanced failure classification with pattern recognition."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Enhanced pattern matching
        if any(term in error_str for term in ["timeout", "connection", "network", "unreachable"]):
            return FailureMode.NETWORK_TIMEOUT
        
        if any(term in error_str for term in ["rate limit", "too many requests", "429"]):
            return FailureMode.API_RATE_LIMIT
        
        if any(term in error_str for term in ["no space", "disk full", "storage"]):
            return FailureMode.DISK_FULL
        
        if error_type in ["MemoryError", "OutOfMemoryError"] or "memory" in error_str:
            return FailureMode.MEMORY_ERROR
        
        if any(term in error_str for term in ["cuda", "gpu", "device", "nvidia"]):
            return FailureMode.GPU_ERROR
        
        if error_type in ["PermissionError", "AccessDenied"] or "permission" in error_str:
            return FailureMode.PERMISSION_DENIED
        
        if error_type in ["ConfigurationError", "ValueError"] and "config" in error_str:
            return FailureMode.CONFIGURATION_ERROR
        
        if error_type in ["ImportError", "ModuleNotFoundError"]:
            return FailureMode.DEPENDENCY_UNAVAILABLE
        
        if any(term in error_str for term in ["service", "unavailable", "503"]):
            return FailureMode.SERVICE_UNAVAILABLE
        
        if any(term in error_str for term in ["database", "db", "connection"]):
            return FailureMode.DATABASE_ERROR
        
        if any(term in error_str for term in ["auth", "unauthorized", "401", "403"]):
            return FailureMode.AUTHENTICATION_FAILURE
        
        return FailureMode.UNKNOWN_ERROR
    
    def handle_failure(
        self,
        error: Exception,
        component: str,
        context: Dict[str, Any],
        custom_strategy: Optional[RecoveryStrategy] = None
    ) -> FailureRecord:
        """Enhanced failure handling with comprehensive recovery."""
        failure_mode = self.classify_failure(error, context)
        recovery_strategy = custom_strategy or self.recovery_strategies[failure_mode]
        
        failure_record = FailureRecord(
            failure_id=f"failure_{int(time.time())}_{random.randint(1000, 9999)}",
            timestamp=datetime.now(),
            failure_mode=failure_mode,
            component=component,
            error_message=str(error),
            context=context.copy(),
            recovery_strategy=recovery_strategy,
            severity=self._assess_failure_severity(failure_mode, context),
            impact_assessment=self._assess_failure_impact(failure_mode, component)
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
            elif recovery_strategy == RecoveryStrategy.STATE_RESTORATION:
                recovery_successful = self._state_restoration(error, context)
            elif recovery_strategy == RecoveryStrategy.ROLLBACK_AND_RETRY:
                recovery_successful = self._rollback_and_retry(error, context)
            elif recovery_strategy == RecoveryStrategy.IGNORE_AND_CONTINUE:
                recovery_successful = True
            else:
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
    
    def _assess_failure_severity(self, failure_mode: FailureMode, context: Dict[str, Any]) -> str:
        """Assess failure severity."""
        if failure_mode in [FailureMode.DATA_CORRUPTION, FailureMode.MEMORY_ERROR]:
            return "critical"
        elif failure_mode in [FailureMode.DISK_FULL, FailureMode.GPU_ERROR]:
            return "high"
        elif failure_mode in [FailureMode.NETWORK_TIMEOUT, FailureMode.API_RATE_LIMIT]:
            return "medium"
        else:
            return "low"
    
    def _assess_failure_impact(self, failure_mode: FailureMode, component: str) -> str:
        """Assess failure impact on system."""
        critical_components = ["carbon_tracker", "monitoring_system", "backup_manager"]
        
        if component in critical_components:
            return "high_impact"
        elif failure_mode in [FailureMode.DATA_CORRUPTION, FailureMode.MEMORY_ERROR]:
            return "high_impact"
        else:
            return "medium_impact"
    
    def _graceful_degradation(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Enhanced graceful degradation."""
        degraded_features = context.get("degraded_features", [])
        
        degradation_priorities = [
            ("detailed_logging", "Reduced logging verbosity"),
            ("prometheus_export", "Disabled Prometheus export"),
            ("advanced_analytics", "Disabled advanced analytics"),
            ("real_time_monitoring", "Reduced monitoring frequency"),
            ("backup_creation", "Disabled automatic backups"),
            ("external_api_calls", "Disabled non-essential API calls"),
            ("predictive_analysis", "Disabled predictive analysis"),
            ("redundant_collection", "Disabled redundant metrics"),
            ("audit_logging", "Reduced audit logging"),
            ("alert_notifications", "Reduced alert notifications")
        ]
        
        for feature, description in degradation_priorities:
            if feature not in degraded_features:
                degraded_features.append(feature)
                context["degraded_features"] = degraded_features
                
                self._apply_feature_degradation(feature, context)
                
                logger.warning(f"Graceful degradation: {description}")
                return True
        
        # Emergency mode if all features degraded
        if len(degraded_features) >= len(degradation_priorities):
            context["emergency_mode"] = True
            logger.critical("Entering emergency mode - minimal functionality only")
            return True
        
        return False
    
    def _apply_feature_degradation(self, feature: str, context: Dict[str, Any]):
        """Apply specific degradation for a feature."""
        if feature == "detailed_logging":
            logging.getLogger().setLevel(logging.WARNING)
        elif feature == "real_time_monitoring":
            context["monitoring_interval_multiplier"] = context.get("monitoring_interval_multiplier", 1) * 2
        elif feature == "external_api_calls":
            context["skip_external_apis"] = True
        elif feature == "backup_creation":
            context["disable_auto_backup"] = True
        elif feature == "redundant_collection":
            context["single_source_metrics"] = True
    
    def _fallback_to_alternative(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Enhanced fallback to alternative systems."""
        component = context.get("component", "unknown")
        
        fallback_strategies = {
            "carbon_intensity_api": {
                "primary_fallback": "regional_api",
                "secondary_fallback": "cached_values",
                "tertiary_fallback": "default_values",
                "default_value": 400
            },
            "gpu_monitoring": {
                "primary_fallback": "nvidia_ml",
                "secondary_fallback": "system_monitoring", 
                "tertiary_fallback": "cpu_estimates",
                "estimation_multiplier": 0.7
            },
            "prometheus_export": {
                "primary_fallback": "file_export",
                "secondary_fallback": "log_export",
                "tertiary_fallback": "memory_buffer",
                "buffer_size": 1000
            }
        }
        
        if component in fallback_strategies:
            strategy = fallback_strategies[component]
            
            # Try fallbacks in order
            for fallback_type in ["primary_fallback", "secondary_fallback", "tertiary_fallback"]:
                try:
                    if self._try_fallback(component, strategy[fallback_type], strategy, context):
                        return True
                except Exception as e:
                    logger.debug(f"{fallback_type} failed for {component}: {e}")
        
        # Generic fallback
        context["fallback_mode"] = True
        context["minimal_functionality"] = True
        logger.warning(f"Generic fallback activated for {component}")
        return True
    
    def _try_fallback(self, component: str, fallback_type: str, 
                     strategy: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Try a specific fallback strategy."""
        if fallback_type == "default_values":
            context["fallback_value"] = strategy.get("default_value", 0)
            return True
        elif fallback_type == "cpu_estimates":
            context["fallback_monitoring"] = "cpu_estimate"
            context["estimation_multiplier"] = strategy.get("estimation_multiplier", 1.0)
            return True
        elif fallback_type == "memory_buffer":
            context["buffer_mode"] = True
            context["buffer_size"] = strategy.get("buffer_size", 100)
            return True
        
        return False
    
    def _checkpoint_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Enhanced checkpoint recovery."""
        component = context.get("component")
        if not component:
            return False
        
        # Restore from state checkpoint
        state_data = self.crash_recovery_manager.restore_state_checkpoint(component)
        if state_data:
            context["recovered_state"] = state_data
            logger.info(f"Recovered state for {component}")
            return True
        
        return False
    
    def _state_restoration(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Restore system state from backup."""
        component = context.get("component")
        if not component:
            return False
        
        # Find recent backup for component
        recent_backups = self.backup_manager.list_backups(component)
        if recent_backups:
            backup = recent_backups[0]  # Most recent
            
            # Restore backup to temporary location for validation
            temp_path = f"/tmp/{component}_recovery"
            if self.backup_manager.restore_backup(backup.backup_id, temp_path):
                context["restored_backup"] = backup.backup_id
                context["restore_path"] = temp_path
                logger.info(f"State restored from backup {backup.backup_id}")
                return True
        
        return False
    
    def _rollback_and_retry(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Rollback operation and retry."""
        component = context.get("component")
        operation = context.get("operation")
        
        if component and operation:
            logger.info(f"Rolling back {operation} for {component}")
            
            # Restore previous state
            if self._state_restoration(error, context):
                context["rollback_completed"] = True
                return True
        
        return False
    
    def get_resilience_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive resilience dashboard."""
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
        
        with self.circuit_breaker_manager._lock:
            circuit_breaker_status = {
                name: {
                    "state": breaker.state,
                    "failure_count": breaker.failure_count,
                    "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
                }
                for name, breaker in self.circuit_breaker_manager.circuit_breakers.items()
            }
        
        backups = self.backup_manager.list_backups()
        
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
                "available_backups": len(backups)
            },
            "failure_breakdown": failure_mode_counts,
            "recovery_strategies": recovery_success_counts,
            "circuit_breakers": circuit_breaker_status,
            "recent_backups": [asdict(b) for b in backups[:5]],
            "external_apis": self.external_api_manager._lock and {
                api: {
                    "available": api not in self.external_api_manager.backoff_state,
                    "backoff_remaining": self.external_api_manager._get_backoff_delay(api)
                }
                for api in self.external_api_manager.api_configs.keys()
            },
            "redundant_metrics": {
                "registered": len(self.redundant_collector.collectors)
            }
        }


# Global enhanced fault tolerance manager
_fault_tolerance_manager: Optional[EnhancedFaultToleranceManager] = None


def get_fault_tolerance_manager(checkpoint_dir: Optional[Path] = None,
                               backup_dir: Optional[Path] = None,
                               state_dir: Optional[Path] = None) -> EnhancedFaultToleranceManager:
    """Get or create the global enhanced fault tolerance manager."""
    global _fault_tolerance_manager
    
    if _fault_tolerance_manager is None:
        _fault_tolerance_manager = EnhancedFaultToleranceManager(
            checkpoint_dir, backup_dir, state_dir
        )
    
    return _fault_tolerance_manager


def create_automatic_backup(component: str, source_path: str, 
                           backup_type: str = "incremental") -> str:
    """Create an automatic backup for a component."""
    manager = get_fault_tolerance_manager()
    return manager.backup_manager.create_backup(component, source_path, backup_type)


def restore_from_backup(backup_id: str, restore_path: str) -> bool:
    """Restore from a backup."""
    manager = get_fault_tolerance_manager()
    return manager.backup_manager.restore_backup(backup_id, restore_path)


def register_redundant_metric(metric_name: str, primary_collector: str, 
                             backup_collectors: List[str]) -> None:
    """Register a metric for redundant collection."""
    manager = get_fault_tolerance_manager()
    config = RedundantMetric(
        metric_name=metric_name,
        primary_collector=primary_collector,
        backup_collectors=backup_collectors
    )
    manager.redundant_collector.register_redundant_metric(config)


def collect_redundant_metric(metric_name: str, collector_values: Dict[str, float]) -> Tuple[float, bool, str]:
    """Collect a metric with redundancy."""
    manager = get_fault_tolerance_manager()
    return manager.redundant_collector.collect_metric(metric_name, collector_values)


def make_external_api_request(api_name: str, endpoint: str, 
                             params: Optional[Dict] = None) -> Tuple[bool, Any, str]:
    """Make an external API request with fault tolerance."""
    manager = get_fault_tolerance_manager()
    return manager.external_api_manager.make_request(api_name, endpoint, params)


def save_component_state(component: str, state_data: Dict[str, Any]) -> None:
    """Save component state for crash recovery."""
    manager = get_fault_tolerance_manager()
    manager.crash_recovery_manager.save_state_checkpoint(component, state_data)


def restore_component_state(component: str) -> Optional[Dict[str, Any]]:
    """Restore component state from checkpoint."""
    manager = get_fault_tolerance_manager()
    return manager.crash_recovery_manager.restore_state_checkpoint(component)


def register_crash_recovery_handler(component: str, handler: Callable) -> None:
    """Register a crash recovery handler."""
    manager = get_fault_tolerance_manager()
    manager.crash_recovery_manager.register_recovery_handler(component, handler)


def perform_system_recovery() -> Dict[str, bool]:
    """Perform system-wide crash recovery."""
    manager = get_fault_tolerance_manager()
    return manager.crash_recovery_manager.perform_crash_recovery()


def resilient(
    component_name: str,
    failure_mode: Optional[FailureMode] = None,
    recovery_strategy: Optional[RecoveryStrategy] = None,
    auto_backup: bool = False,
    save_state: bool = False
):
    """Enhanced decorator for resilient function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_fault_tolerance_manager()
            
            # Save state if requested
            if save_state:
                try:
                    state_data = {
                        "function": func.__name__,
                        "args": [str(arg) for arg in args],
                        "kwargs": {k: str(v) for k, v in kwargs.items()},
                        "timestamp": time.time()
                    }
                    save_component_state(component_name, state_data)
                except Exception as e:
                    logger.debug(f"Failed to save state for {component_name}: {e}")
            
            # Create backup if requested
            if auto_backup:
                try:
                    logger.debug(f"Auto-backup triggered for {component_name}")
                    # Implementation depends on what needs to be backed up
                except Exception as e:
                    logger.debug(f"Auto-backup failed for {component_name}: {e}")
            
            # Execute with fault tolerance
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "component": component_name,
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                
                failure_record = manager.handle_failure(
                    e, component_name, context, recovery_strategy
                )
                
                # Apply recovery strategy
                if failure_record.recovery_strategy in [
                    RecoveryStrategy.RETRY_EXPONENTIAL_BACKOFF,
                    RecoveryStrategy.RETRY_LINEAR_BACKOFF,
                    RecoveryStrategy.RETRY_ADAPTIVE_BACKOFF
                ]:
                    # Implement retry logic
                    max_attempts = 3
                    for attempt in range(1, max_attempts + 1):
                        try:
                            delay = manager.retry_manager.calculate_delay(
                                attempt, failure_record.recovery_strategy, "default", context
                            )
                            if attempt > 1:
                                time.sleep(delay)
                            
                            result = func(*args, **kwargs)
                            manager.retry_manager.record_attempt(component_name, True, attempt)
                            return result
                            
                        except Exception as retry_error:
                            manager.retry_manager.record_attempt(component_name, False, attempt, str(retry_error))
                            if attempt == max_attempts:
                                raise
                
                elif failure_record.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    return manager.circuit_breaker_manager.execute_with_circuit_breaker(
                        component_name, func, args, kwargs
                    )
                elif failure_record.recovery_strategy == RecoveryStrategy.IGNORE_AND_CONTINUE:
                    logger.warning(f"Ignoring error in {component_name}: {e}")
                    return None
                else:
                    raise
        
        return wrapper
    
    return decorator


def external_api_resilient(api_name: str, max_retries: int = 3):
    """Decorator for resilient external API calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_fault_tolerance_manager()
            
            # Extract endpoint and params from function arguments
            endpoint = kwargs.get('endpoint', args[0] if args else '')
            params = kwargs.get('params', args[1] if len(args) > 1 else None)
            
            # Use the external API manager
            success, result, message = manager.external_api_manager.make_request(
                api_name, endpoint, params, max_retries
            )
            
            if success:
                return result
            else:
                raise Exception(f"API call failed: {message}")
        
        return wrapper
    
    return decorator


def get_fault_tolerance_dashboard() -> Dict[str, Any]:
    """Get comprehensive fault tolerance dashboard."""
    manager = get_fault_tolerance_manager()
    return {
        "resilience_overview": manager.get_resilience_dashboard(),
        "backup_status": {
            "total_backups": len(manager.backup_manager.backup_metadata),
            "recent_backups": len(manager.backup_manager.list_backups()[:5])
        },
        "crash_recovery": {
            "registered_handlers": len(manager.crash_recovery_manager.recovery_handlers),
            "available_checkpoints": len(list(manager.crash_recovery_manager.state_dir.glob("*_state.json")))
        },
        "redundant_metrics": {
            "registered_metrics": len(manager.redundant_collector.collectors)
        }
    }