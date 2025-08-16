"""Enhanced comprehensive error handling and resilience for carbon tracking."""

import time
import logging
import functools
from typing import Dict, List, Optional, Any, Callable, Type, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import threading
import json
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import numpy as np
from collections import deque, Counter
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import pickle
import hashlib
import os
import re
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Enhanced error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ErrorCategory(Enum):
    """Categories of errors for better classification."""
    SYSTEM = "system"
    APPLICATION = "application"
    NETWORK = "network"
    DATA = "data"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    EXTERNAL = "external"
    USER = "user"


class ImpactLevel(Enum):
    """Impact levels for error assessment."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


@dataclass
class ErrorPattern:
    """Error pattern for trend analysis."""
    pattern_id: str
    error_type: str
    message_pattern: str
    frequency: int
    first_seen: datetime
    last_seen: datetime
    components_affected: List[str]
    resolution_rate: float
    trend: str  # "increasing", "decreasing", "stable"
    

@dataclass
class ErrorInfo:
    """Enhanced error information container."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    error_type: str
    message: str
    traceback: str
    context: Dict[str, Any]
    component: str
    function_name: Optional[str]
    line_number: Optional[int]
    file_path: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    resolved: bool = False
    retry_count: int = 0
    resolution_time: Optional[datetime] = None
    resolution_method: Optional[str] = None
    impact_level: ImpactLevel = ImpactLevel.LOW
    related_errors: List[str] = None
    root_cause: Optional[str] = None
    mitigation_actions: List[str] = None
    lessons_learned: Optional[str] = None
    
    def __post_init__(self):
        if self.related_errors is None:
            self.related_errors = []
        if self.mitigation_actions is None:
            self.mitigation_actions = []


@dataclass 
class ErrorTrend:
    """Error trend analysis data."""
    component: str
    error_type: str
    count_1h: int
    count_24h: int
    count_7d: int
    trend_direction: str  # "up", "down", "stable"
    severity_distribution: Dict[str, int]
    resolution_rate: float
    avg_resolution_time: float
    predicted_next_occurrence: Optional[datetime]


@dataclass
class ErrorAlertRule:
    """Rule for error alerting."""
    rule_id: str
    name: str
    conditions: Dict[str, Any]
    severity_threshold: ErrorSeverity
    frequency_threshold: int
    time_window_minutes: int
    notification_channels: List[str]
    escalation_rules: List[Dict[str, Any]]
    active: bool = True


class CircuitBreaker:
    """Enhanced circuit breaker with error-specific logic."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, 
                 error_types: Optional[List[str]] = None):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.error_types = error_types or []  # Specific error types to track
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self.error_history = deque(maxlen=100)
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
                    error_type = type(e).__name__
                    
                    # Only count specific error types if configured
                    if not self.error_types or error_type in self.error_types:
                        self.failure_count += 1
                        self.last_failure_time = time.time()
                        self.error_history.append({
                            "timestamp": time.time(),
                            "error_type": error_type,
                            "error_message": str(e)
                        })
                        
                        if self.failure_count >= self.failure_threshold:
                            self.state = "open"
                            logger.warning(f"Circuit breaker opened for {func.__name__}")
                    
                    raise
        
        return wrapper


class RetryPolicy:
    """Enhanced retry policy with error-specific configurations."""
    
    def __init__(self, max_attempts: int = 3, backoff_factor: float = 1.0,
                 retry_exceptions: tuple = (Exception,),
                 retry_conditions: Optional[Dict[str, Any]] = None):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.retry_exceptions = retry_exceptions
        self.retry_conditions = retry_conditions or {}
        self.attempt_history = deque(maxlen=1000)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry based on exception and conditions."""
        if attempt >= self.max_attempts:
            return False
        
        if not isinstance(exception, self.retry_exceptions):
            return False
        
        # Check specific retry conditions
        error_message = str(exception).lower()
        
        # Don't retry on certain conditions
        non_retryable_conditions = [
            "authentication", "authorization", "permission", "forbidden",
            "not found", "invalid", "malformed", "syntax error"
        ]
        
        if any(condition in error_message for condition in non_retryable_conditions):
            return False
        
        # Custom retry conditions
        for condition, should_retry in self.retry_conditions.items():
            if condition.lower() in error_message:
                return should_retry
        
        return True
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    # Record successful attempt
                    self.attempt_history.append({
                        "timestamp": time.time(),
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "success": True
                    })
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Record failed attempt
                    self.attempt_history.append({
                        "timestamp": time.time(),
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "success": False,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    })
                    
                    if not self.should_retry(e, attempt + 1):
                        break
                    
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


class ErrorPatternAnalyzer:
    """Analyzes error patterns and trends for predictive insights."""
    
    def __init__(self, analysis_window_hours: int = 24):
        self.analysis_window_hours = analysis_window_hours
        self.patterns: Dict[str, ErrorPattern] = {}
        self.error_cache = deque(maxlen=10000)
        self._lock = threading.Lock()
    
    def add_error(self, error_info: ErrorInfo):
        """Add error to pattern analysis."""
        with self._lock:
            self.error_cache.append(error_info)
            self._update_patterns(error_info)
    
    def _update_patterns(self, error_info: ErrorInfo):
        """Update error patterns with new error."""
        # Create pattern key
        pattern_key = f"{error_info.component}_{error_info.error_type}"
        
        # Extract message pattern (remove specific details)
        message_pattern = self._generalize_message(error_info.message)
        
        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            pattern.frequency += 1
            pattern.last_seen = error_info.timestamp
            
            if error_info.component not in pattern.components_affected:
                pattern.components_affected.append(error_info.component)
        else:
            self.patterns[pattern_key] = ErrorPattern(
                pattern_id=str(uuid.uuid4()),
                error_type=error_info.error_type,
                message_pattern=message_pattern,
                frequency=1,
                first_seen=error_info.timestamp,
                last_seen=error_info.timestamp,
                components_affected=[error_info.component],
                resolution_rate=0.0,
                trend="stable"
            )
    
    def _generalize_message(self, message: str) -> str:
        """Generalize error message for pattern matching."""
        # Replace specific numbers, paths, IDs with placeholders
        generalized = re.sub(r'\d+', '[NUMBER]', message)
        generalized = re.sub(r'/[^\s]+', '[PATH]', generalized)
        generalized = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '[UUID]', generalized)
        generalized = re.sub(r'\b\w+@\w+\.\w+\b', '[EMAIL]', generalized)
        
        return generalized
    
    def analyze_trends(self) -> List[ErrorTrend]:
        """Analyze error trends over time."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=7)
        
        # Filter recent errors
        recent_errors = [
            error for error in self.error_cache
            if error.timestamp >= cutoff_time
        ]
        
        # Group by component and error type
        trend_data = {}
        
        for error in recent_errors:
            key = f"{error.component}_{error.error_type}"
            
            if key not in trend_data:
                trend_data[key] = {
                    "component": error.component,
                    "error_type": error.error_type,
                    "errors_1h": [],
                    "errors_24h": [],
                    "errors_7d": [],
                    "resolved_count": 0,
                    "resolution_times": []
                }
            
            data = trend_data[key]
            
            # Categorize by time window
            hours_ago = (current_time - error.timestamp).total_seconds() / 3600
            
            if hours_ago <= 1:
                data["errors_1h"].append(error)
            if hours_ago <= 24:
                data["errors_24h"].append(error)
            if hours_ago <= 168:  # 7 days
                data["errors_7d"].append(error)
            
            # Track resolutions
            if error.resolved and error.resolution_time:
                data["resolved_count"] += 1
                resolution_time = (error.resolution_time - error.timestamp).total_seconds() / 60
                data["resolution_times"].append(resolution_time)
        
        # Create trend objects
        trends = []
        for key, data in trend_data.items():
            count_1h = len(data["errors_1h"])
            count_24h = len(data["errors_24h"])
            count_7d = len(data["errors_7d"])
            
            # Calculate trend direction
            if count_24h > 0:
                recent_rate = count_1h / 1  # errors per hour in last hour
                daily_rate = count_24h / 24  # average errors per hour in last day
                
                if recent_rate > daily_rate * 1.5:
                    trend_direction = "up"
                elif recent_rate < daily_rate * 0.5:
                    trend_direction = "down"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"
            
            # Calculate resolution metrics
            resolution_rate = data["resolved_count"] / max(count_7d, 1)
            avg_resolution_time = np.mean(data["resolution_times"]) if data["resolution_times"] else 0.0
            
            # Calculate severity distribution
            severity_dist = {}
            for error in data["errors_24h"]:
                severity = error.severity.value
                severity_dist[severity] = severity_dist.get(severity, 0) + 1
            
            trends.append(ErrorTrend(
                component=data["component"],
                error_type=data["error_type"],
                count_1h=count_1h,
                count_24h=count_24h,
                count_7d=count_7d,
                trend_direction=trend_direction,
                severity_distribution=severity_dist,
                resolution_rate=resolution_rate,
                avg_resolution_time=avg_resolution_time,
                predicted_next_occurrence=self._predict_next_occurrence(data["errors_24h"])
            ))
        
        return trends
    
    def _predict_next_occurrence(self, recent_errors: List[ErrorInfo]) -> Optional[datetime]:
        """Predict when next error might occur based on patterns."""
        if len(recent_errors) < 3:
            return None
        
        # Calculate average interval between errors
        timestamps = sorted([error.timestamp for error in recent_errors])
        intervals = []
        
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if intervals:
            avg_interval = np.mean(intervals)
            # Predict next occurrence
            last_error_time = timestamps[-1]
            predicted_time = last_error_time + timedelta(seconds=avg_interval)
            
            # Only return if prediction is reasonable (within next 7 days)
            if predicted_time <= datetime.now() + timedelta(days=7):
                return predicted_time
        
        return None
    
    def get_top_patterns(self, limit: int = 10) -> List[ErrorPattern]:
        """Get top error patterns by frequency."""
        with self._lock:
            sorted_patterns = sorted(
                self.patterns.values(),
                key=lambda p: p.frequency,
                reverse=True
            )
            return sorted_patterns[:limit]
    
    def get_emerging_patterns(self, hours_back: int = 24) -> List[ErrorPattern]:
        """Get patterns that have emerged recently."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            emerging = [
                pattern for pattern in self.patterns.values()
                if pattern.first_seen >= cutoff_time and pattern.frequency >= 3
            ]
            
            return sorted(emerging, key=lambda p: p.frequency, reverse=True)


class ErrorNotificationManager:
    """Manages error notifications and escalations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alert_rules: Dict[str, ErrorAlertRule] = {}
        self.notification_history = deque(maxlen=1000)
        self.escalation_state: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Initialize default rules
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialize default alert rules."""
        self.alert_rules["critical_errors"] = ErrorAlertRule(
            rule_id="critical_errors",
            name="Critical Error Alert",
            conditions={
                "severity": ["critical", "emergency"],
                "frequency": 1
            },
            severity_threshold=ErrorSeverity.CRITICAL,
            frequency_threshold=1,
            time_window_minutes=5,
            notification_channels=["email", "slack", "webhook"],
            escalation_rules=[
                {"delay_minutes": 15, "channels": ["email", "sms"]},
                {"delay_minutes": 30, "channels": ["phone", "email"]}
            ]
        )
        
        self.alert_rules["error_spike"] = ErrorAlertRule(
            rule_id="error_spike",
            name="Error Spike Detection",
            conditions={
                "frequency": 10,
                "component": "*"
            },
            severity_threshold=ErrorSeverity.MEDIUM,
            frequency_threshold=10,
            time_window_minutes=10,
            notification_channels=["slack", "email"],
            escalation_rules=[]
        )
        
        self.alert_rules["new_error_pattern"] = ErrorAlertRule(
            rule_id="new_error_pattern",
            name="New Error Pattern",
            conditions={
                "new_pattern": True,
                "frequency": 3
            },
            severity_threshold=ErrorSeverity.MEDIUM,
            frequency_threshold=3,
            time_window_minutes=60,
            notification_channels=["slack"],
            escalation_rules=[]
        )
    
    def add_alert_rule(self, rule: ErrorAlertRule):
        """Add a new alert rule."""
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def check_alert_conditions(self, error_info: ErrorInfo, recent_errors: List[ErrorInfo]):
        """Check if error matches any alert conditions."""
        for rule_id, rule in self.alert_rules.items():
            if not rule.active:
                continue
            
            if self._matches_rule(error_info, recent_errors, rule):
                self._trigger_alert(rule, error_info, recent_errors)
    
    def _matches_rule(self, error_info: ErrorInfo, recent_errors: List[ErrorInfo], 
                     rule: ErrorAlertRule) -> bool:
        """Check if error matches rule conditions."""
        conditions = rule.conditions
        
        # Check severity
        if "severity" in conditions:
            severity_list = conditions["severity"]
            if isinstance(severity_list, str):
                severity_list = [severity_list]
            
            if error_info.severity.value not in severity_list:
                return False
        
        # Check frequency within time window
        if "frequency" in conditions:
            time_window = timedelta(minutes=rule.time_window_minutes)
            cutoff_time = datetime.now() - time_window
            
            # Count matching errors in time window
            matching_errors = [
                err for err in recent_errors
                if err.timestamp >= cutoff_time and
                   err.component == error_info.component and
                   err.error_type == error_info.error_type
            ]
            
            if len(matching_errors) < conditions["frequency"]:
                return False
        
        # Check component
        if "component" in conditions:
            component_pattern = conditions["component"]
            if component_pattern != "*" and component_pattern != error_info.component:
                return False
        
        # Check for new patterns
        if conditions.get("new_pattern", False):
            # This would be set by the pattern analyzer
            if not error_info.context.get("is_new_pattern", False):
                return False
        
        return True
    
    def _trigger_alert(self, rule: ErrorAlertRule, error_info: ErrorInfo, 
                      recent_errors: List[ErrorInfo]):
        """Trigger alert for matching rule."""
        alert_key = f"{rule.rule_id}_{error_info.component}_{error_info.error_type}"
        
        # Check if already escalating
        with self._lock:
            if alert_key in self.escalation_state:
                return  # Already being handled
            
            # Initialize escalation state
            self.escalation_state[alert_key] = {
                "rule_id": rule.rule_id,
                "triggered_at": datetime.now(),
                "escalation_level": 0,
                "error_info": error_info
            }
        
        # Send initial notifications
        self._send_notifications(rule, error_info, rule.notification_channels)
        
        # Schedule escalations
        if rule.escalation_rules:
            self._schedule_escalations(alert_key, rule, error_info)
    
    def _send_notifications(self, rule: ErrorAlertRule, error_info: ErrorInfo, 
                           channels: List[str]):
        """Send notifications through specified channels."""
        for channel in channels:
            try:
                if channel == "email":
                    self._send_email_notification(rule, error_info)
                elif channel == "slack":
                    self._send_slack_notification(rule, error_info)
                elif channel == "webhook":
                    self._send_webhook_notification(rule, error_info)
                elif channel == "sms":
                    self._send_sms_notification(rule, error_info)
                
                # Record notification
                self.notification_history.append({
                    "timestamp": datetime.now(),
                    "rule_id": rule.rule_id,
                    "channel": channel,
                    "error_id": error_info.error_id,
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"Failed to send {channel} notification: {e}")
                self.notification_history.append({
                    "timestamp": datetime.now(),
                    "rule_id": rule.rule_id,
                    "channel": channel,
                    "error_id": error_info.error_id,
                    "success": False,
                    "error": str(e)
                })
    
    def _send_email_notification(self, rule: ErrorAlertRule, error_info: ErrorInfo):
        """Send email notification."""
        email_config = self.config.get("email", {})
        if not email_config.get("smtp_server"):
            logger.debug("Email notification skipped - no SMTP configuration")
            return
        
        subject = f"Error Alert: {rule.name} - {error_info.component}"
        
        body = f"""
Error Alert: {rule.name}

Component: {error_info.component}
Error Type: {error_info.error_type}
Severity: {error_info.severity.value}
Message: {error_info.message}
Timestamp: {error_info.timestamp}

Error ID: {error_info.error_id}

This is an automated alert from the HF Eco2AI Error Monitoring System.
"""
        
        # In a real implementation, this would send via SMTP
        logger.info(f"EMAIL ALERT: {subject}")
    
    def _send_slack_notification(self, rule: ErrorAlertRule, error_info: ErrorInfo):
        """Send Slack notification."""
        slack_webhook = self.config.get("slack_webhook_url")
        if not slack_webhook:
            logger.debug("Slack notification skipped - no webhook URL")
            return
        
        color_map = {
            ErrorSeverity.LOW: "good",
            ErrorSeverity.MEDIUM: "warning",
            ErrorSeverity.HIGH: "danger",
            ErrorSeverity.CRITICAL: "#ff0000",
            ErrorSeverity.EMERGENCY: "#ff0000"
        }
        
        payload = {
            "text": f"Error Alert: {rule.name}",
            "attachments": [{
                "color": color_map.get(error_info.severity, "warning"),
                "fields": [
                    {"title": "Component", "value": error_info.component, "short": True},
                    {"title": "Error Type", "value": error_info.error_type, "short": True},
                    {"title": "Severity", "value": error_info.severity.value.upper(), "short": True},
                    {"title": "Category", "value": error_info.category.value, "short": True},
                    {"title": "Message", "value": error_info.message[:200], "short": False}
                ],
                "footer": "HF Eco2AI Error Monitor",
                "ts": int(error_info.timestamp.timestamp())
            }]
        }
        
        try:
            response = requests.post(slack_webhook, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Slack notification sent for error {error_info.error_id}")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def _send_webhook_notification(self, rule: ErrorAlertRule, error_info: ErrorInfo):
        """Send webhook notification."""
        webhook_url = self.config.get("webhook_url")
        if not webhook_url:
            logger.debug("Webhook notification skipped - no URL")
            return
        
        payload = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "error_id": error_info.error_id,
            "component": error_info.component,
            "error_type": error_info.error_type,
            "severity": error_info.severity.value,
            "category": error_info.category.value,
            "message": error_info.message,
            "timestamp": error_info.timestamp.isoformat(),
            "context": error_info.context
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Webhook notification sent for error {error_info.error_id}")
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def _send_sms_notification(self, rule: ErrorAlertRule, error_info: ErrorInfo):
        """Send SMS notification (placeholder)."""
        # This would integrate with SMS service like Twilio
        logger.info(f"SMS ALERT: {rule.name} - {error_info.component}")
    
    def _schedule_escalations(self, alert_key: str, rule: ErrorAlertRule, 
                             error_info: ErrorInfo):
        """Schedule escalation notifications."""
        def escalate():
            time.sleep(60)  # Check every minute for escalations
            
            while alert_key in self.escalation_state:
                escalation_info = self.escalation_state[alert_key]
                current_level = escalation_info["escalation_level"]
                
                if current_level < len(rule.escalation_rules):
                    escalation_rule = rule.escalation_rules[current_level]
                    delay_minutes = escalation_rule["delay_minutes"]
                    
                    time_since_trigger = datetime.now() - escalation_info["triggered_at"]
                    if time_since_trigger.total_seconds() >= delay_minutes * 60:
                        # Send escalation
                        channels = escalation_rule["channels"]
                        self._send_notifications(rule, error_info, channels)
                        
                        # Update escalation level
                        self.escalation_state[alert_key]["escalation_level"] += 1
                        
                        logger.warning(f"Escalated alert {alert_key} to level {current_level + 1}")
                
                time.sleep(60)
        
        # Start escalation thread
        escalation_thread = threading.Thread(target=escalate, daemon=True)
        escalation_thread.start()
    
    def resolve_alert(self, alert_key: str):
        """Resolve an active alert."""
        with self._lock:
            if alert_key in self.escalation_state:
                del self.escalation_state[alert_key]
                logger.info(f"Resolved alert: {alert_key}")


class EnhancedErrorHandler:
    """Enhanced centralized error handling system with advanced analytics."""
    
    def __init__(self, log_file: Optional[Path] = None, 
                 enable_analytics: bool = True,
                 notification_config: Optional[Dict[str, Any]] = None):
        self.errors: List[ErrorInfo] = []
        self.log_file = log_file
        self.enable_analytics = enable_analytics
        self._lock = threading.Lock()
        
        # Enhanced components
        self.pattern_analyzer = ErrorPatternAnalyzer() if enable_analytics else None
        self.notification_manager = ErrorNotificationManager(notification_config)
        
        # Error handlers by type
        self.handlers: Dict[Type[Exception], Callable] = {
            ImportError: self._handle_import_error,
            ConnectionError: self._handle_connection_error,
            TimeoutError: self._handle_timeout_error,
            PermissionError: self._handle_permission_error,
            FileNotFoundError: self._handle_file_error,
            ValueError: self._handle_validation_error,
            MemoryError: self._handle_memory_error,
            OSError: self._handle_os_error,
        }
        
        # Custom recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {
            "retry_with_backoff": self._retry_with_backoff,
            "fallback_to_default": self._fallback_to_default,
            "graceful_degradation": self._graceful_degradation,
            "restart_component": self._restart_component,
            "clear_cache": self._clear_cache,
            "increase_resources": self._increase_resources
        }
        
        logger.info("Enhanced error handling system initialized")
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.APPLICATION,
                    component: str = "unknown") -> ErrorInfo:
        """Enhanced error handling with comprehensive analysis."""
        
        # Extract traceback information
        tb_info = traceback.extract_tb(error.__traceback__)
        file_path = None
        line_number = None
        function_name = None
        
        if tb_info:
            frame = tb_info[-1]  # Last frame (where error occurred)
            file_path = frame.filename
            line_number = frame.lineno
            function_name = frame.name
        
        error_info = ErrorInfo(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            context=context or {},
            component=component,
            function_name=function_name,
            line_number=line_number,
            file_path=file_path,
            user_id=context.get("user_id") if context else None,
            session_id=context.get("session_id") if context else None,
            impact_level=self._assess_impact(error, severity, category)
        )
        
        with self._lock:
            self.errors.append(error_info)
        
        # Log error with appropriate level
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.EMERGENCY: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, f"Error handled: {error_info.error_type} - {error_info.message}")
        
        # Add to pattern analysis
        if self.pattern_analyzer:
            self.pattern_analyzer.add_error(error_info)
            
            # Check if this is a new pattern
            patterns = self.pattern_analyzer.get_emerging_patterns(hours_back=1)
            if any(p.first_seen == error_info.timestamp for p in patterns):
                error_info.context["is_new_pattern"] = True
        
        # Check alert conditions
        recent_errors = self.get_recent_errors(hours=1)
        self.notification_manager.check_alert_conditions(error_info, recent_errors)
        
        # Call specific handler if available
        handler = self.handlers.get(type(error))
        if handler:
            try:
                recovery_strategy = handler(error, error_info)
                if recovery_strategy:
                    self._apply_recovery_strategy(recovery_strategy, error_info)
            except Exception as handler_error:
                logger.error(f"Error in error handler: {handler_error}")
        
        # Write to log file if configured
        if self.log_file:
            self._write_to_log_file(error_info)
        
        return error_info
    
    def _assess_impact(self, error: Exception, severity: ErrorSeverity, 
                      category: ErrorCategory) -> ImpactLevel:
        """Assess the impact level of an error."""
        # Critical errors always have high impact
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.EMERGENCY]:
            return ImpactLevel.CRITICAL
        
        # Security errors have high impact regardless of severity
        if category == ErrorCategory.SECURITY:
            return ImpactLevel.HIGH if severity == ErrorSeverity.HIGH else ImpactLevel.MEDIUM
        
        # Data corruption has high impact
        if "corruption" in str(error).lower() or "corrupt" in str(error).lower():
            return ImpactLevel.HIGH
        
        # Network errors typically have medium impact
        if category == ErrorCategory.NETWORK:
            return ImpactLevel.MEDIUM if severity >= ErrorSeverity.MEDIUM else ImpactLevel.LOW
        
        # Map severity to impact
        severity_to_impact = {
            ErrorSeverity.LOW: ImpactLevel.LOW,
            ErrorSeverity.MEDIUM: ImpactLevel.MEDIUM,
            ErrorSeverity.HIGH: ImpactLevel.HIGH
        }
        
        return severity_to_impact.get(severity, ImpactLevel.LOW)
    
    def _handle_import_error(self, error: ImportError, error_info: ErrorInfo) -> Optional[str]:
        """Enhanced import error handling."""
        missing_module = str(error).split("'")[1] if "'" in str(error) else "unknown"
        
        suggestions = {
            "eco2ai": "Install with: pip install eco2ai>=2.0.0",
            "pynvml": "Install with: pip install pynvml>=11.5.0", 
            "psutil": "Install with: pip install psutil>=5.9.0",
            "transformers": "Install with: pip install transformers>=4.40.0",
            "prometheus_client": "Install with: pip install prometheus-client>=0.20.0",
            "requests": "Install with: pip install requests>=2.25.0",
            "numpy": "Install with: pip install numpy>=1.21.0",
            "schedule": "Install with: pip install schedule>=1.1.0"
        }
        
        suggestion = suggestions.get(missing_module, f"Install missing module: {missing_module}")
        error_info.context["suggestion"] = suggestion
        error_info.context["missing_module"] = missing_module
        error_info.mitigation_actions.append(f"Install {missing_module}")
        
        logger.warning(f"Missing dependency {missing_module}: {suggestion}")
        
        # For critical modules, suggest fallback
        if missing_module in ["pynvml", "psutil"]:
            return "fallback_to_default"
        
        return None
    
    def _handle_connection_error(self, error: ConnectionError, error_info: ErrorInfo) -> str:
        """Enhanced connection error handling."""
        error_info.context["recovery_action"] = "retry_with_backoff"
        error_info.mitigation_actions.extend([
            "Check network connectivity",
            "Verify service availability",
            "Consider using fallback endpoint"
        ])
        
        logger.warning("Network connectivity issue - will retry with backoff")
        return "retry_with_backoff"
    
    def _handle_timeout_error(self, error: TimeoutError, error_info: ErrorInfo) -> str:
        """Enhanced timeout error handling."""
        error_info.context["recovery_action"] = "retry_with_backoff"
        error_info.mitigation_actions.extend([
            "Increase timeout values",
            "Check network latency",
            "Consider using faster endpoint"
        ])
        
        logger.warning("Operation timed out - will retry with increased timeout")
        return "retry_with_backoff"
    
    def _handle_permission_error(self, error: PermissionError, error_info: ErrorInfo) -> None:
        """Enhanced permission error handling."""
        error_info.context["recovery_action"] = "check_permissions"
        error_info.mitigation_actions.extend([
            "Check file/directory permissions",
            "Verify user has necessary access rights",
            "Consider running with elevated privileges"
        ])
        
        logger.error("Permission denied - check file/directory permissions")
        return None  # Cannot auto-recover from permission errors
    
    def _handle_file_error(self, error: FileNotFoundError, error_info: ErrorInfo) -> str:
        """Enhanced file error handling."""
        error_info.context["recovery_action"] = "fallback_to_default"
        error_info.mitigation_actions.extend([
            "Check if file path is correct",
            "Verify file exists",
            "Create missing directories if needed"
        ])
        
        logger.warning(f"File not found: {error.filename} - will use fallback")
        return "fallback_to_default"
    
    def _handle_validation_error(self, error: ValueError, error_info: ErrorInfo) -> str:
        """Enhanced validation error handling."""
        error_info.context["recovery_action"] = "fallback_to_default"
        error_info.mitigation_actions.extend([
            "Validate input parameters",
            "Check data format and types",
            "Use default values for invalid inputs"
        ])
        
        logger.warning(f"Validation error: {error} - using defaults")
        return "fallback_to_default"
    
    def _handle_memory_error(self, error: MemoryError, error_info: ErrorInfo) -> str:
        """Enhanced memory error handling."""
        error_info.context["recovery_action"] = "increase_resources"
        error_info.mitigation_actions.extend([
            "Free up memory",
            "Reduce batch size",
            "Use memory-efficient algorithms",
            "Consider adding swap space"
        ])
        
        logger.critical("Memory exhausted - attempting to free resources")
        return "increase_resources"
    
    def _handle_os_error(self, error: OSError, error_info: ErrorInfo) -> Optional[str]:
        """Enhanced OS error handling."""
        error_str = str(error).lower()
        
        if "disk" in error_str or "space" in error_str:
            error_info.mitigation_actions.extend([
                "Free up disk space",
                "Clean temporary files",
                "Move data to different location"
            ])
            return "clear_cache"
        
        elif "resource" in error_str:
            error_info.mitigation_actions.append("Wait and retry operation")
            return "retry_with_backoff"
        
        return None
    
    def _apply_recovery_strategy(self, strategy_name: str, error_info: ErrorInfo):
        """Apply recovery strategy."""
        if strategy_name in self.recovery_strategies:
            try:
                strategy_func = self.recovery_strategies[strategy_name]
                result = strategy_func(error_info)
                
                if result:
                    error_info.resolution_method = strategy_name
                    logger.info(f"Applied recovery strategy '{strategy_name}' for error {error_info.error_id}")
                else:
                    logger.warning(f"Recovery strategy '{strategy_name}' was not successful")
                    
            except Exception as e:
                logger.error(f"Failed to apply recovery strategy '{strategy_name}': {e}")
    
    def _retry_with_backoff(self, error_info: ErrorInfo) -> bool:
        """Retry operation with exponential backoff."""
        # This would be implemented by the calling code
        # Here we just mark the strategy as applied
        error_info.context["retry_scheduled"] = True
        return True
    
    def _fallback_to_default(self, error_info: ErrorInfo) -> bool:
        """Use default/fallback values."""
        error_info.context["using_fallback"] = True
        return True
    
    def _graceful_degradation(self, error_info: ErrorInfo) -> bool:
        """Enable graceful degradation mode."""
        error_info.context["degraded_mode"] = True
        return True
    
    def _restart_component(self, error_info: ErrorInfo) -> bool:
        """Restart affected component."""
        # This would trigger component restart
        error_info.context["restart_scheduled"] = True
        return True
    
    def _clear_cache(self, error_info: ErrorInfo) -> bool:
        """Clear caches to free up resources."""
        import gc
        gc.collect()
        error_info.context["cache_cleared"] = True
        return True
    
    def _increase_resources(self, error_info: ErrorInfo) -> bool:
        """Attempt to increase available resources."""
        # Free up memory
        import gc
        gc.collect()
        
        # Log current resource usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            logger.info(f"Memory usage after cleanup: {memory.percent}%")
        except ImportError:
            pass
        
        error_info.context["resources_freed"] = True
        return True
    
    def _write_to_log_file(self, error_info: ErrorInfo):
        """Write enhanced error information to log file."""
        try:
            log_entry = {
                "error_id": error_info.error_id,
                "timestamp": error_info.timestamp.isoformat(),
                "severity": error_info.severity.value,
                "category": error_info.category.value,
                "error_type": error_info.error_type,
                "message": error_info.message,
                "component": error_info.component,
                "function_name": error_info.function_name,
                "line_number": error_info.line_number,
                "file_path": error_info.file_path,
                "context": error_info.context,
                "impact_level": error_info.impact_level.value,
                "mitigation_actions": error_info.mitigation_actions
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write to error log file: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary with analytics."""
        with self._lock:
            if not self.errors:
                return {"total_errors": 0, "analytics_available": False}
            
            # Basic statistics
            by_severity = {}
            by_category = {}
            by_component = {}
            
            for error in self.errors:
                severity = error.severity.value
                category = error.category.value
                component = error.component
                
                by_severity[severity] = by_severity.get(severity, 0) + 1
                by_category[category] = by_category.get(category, 0) + 1
                by_component[component] = by_component.get(component, 0) + 1
            
            # Recent errors (last 10)
            recent = self.errors[-10:]
            
            summary = {
                "total_errors": len(self.errors),
                "by_severity": by_severity,
                "by_category": by_category,
                "by_component": by_component,
                "recent_errors": [
                    {
                        "error_id": error.error_id,
                        "timestamp": error.timestamp.isoformat(),
                        "severity": error.severity.value,
                        "category": error.category.value,
                        "type": error.error_type,
                        "component": error.component,
                        "message": error.message[:100] + "..." if len(error.message) > 100 else error.message,
                        "resolved": error.resolved
                    }
                    for error in recent
                ]
            }
            
            # Add analytics if available
            if self.pattern_analyzer:
                trends = self.pattern_analyzer.analyze_trends()
                top_patterns = self.pattern_analyzer.get_top_patterns(5)
                emerging_patterns = self.pattern_analyzer.get_emerging_patterns(24)
                
                summary["analytics"] = {
                    "trends_count": len(trends),
                    "top_error_patterns": [
                        {
                            "error_type": p.error_type,
                            "frequency": p.frequency,
                            "trend": p.trend,
                            "components_affected": len(p.components_affected)
                        }
                        for p in top_patterns
                    ],
                    "emerging_patterns_24h": len(emerging_patterns),
                    "trends_summary": [
                        {
                            "component": t.component,
                            "error_type": t.error_type,
                            "count_24h": t.count_24h,
                            "trend_direction": t.trend_direction,
                            "resolution_rate": round(t.resolution_rate, 2)
                        }
                        for t in trends[:5]  # Top 5 trends
                    ]
                }
            
            return summary
    
    def get_recent_errors(self, hours: int = 24) -> List[ErrorInfo]:
        """Get errors from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                error for error in self.errors
                if error.timestamp >= cutoff_time
            ]
    
    def resolve_error(self, error_id: str, resolution_method: str = "manual"):
        """Mark an error as resolved."""
        with self._lock:
            for error in self.errors:
                if error.error_id == error_id:
                    error.resolved = True
                    error.resolution_time = datetime.now()
                    error.resolution_method = resolution_method
                    
                    logger.info(f"Error {error_id} marked as resolved via {resolution_method}")
                    
                    # Resolve any related alerts
                    alert_key = f"*_{error.component}_{error.error_type}"
                    self.notification_manager.resolve_alert(alert_key)
                    
                    return True
        
        return False
    
    def clear_errors(self):
        """Clear error history."""
        with self._lock:
            self.errors.clear()
        
        if self.pattern_analyzer:
            self.pattern_analyzer.patterns.clear()
            self.pattern_analyzer.error_cache.clear()
        
        logger.info("Error history cleared")
    
    def export_error_analytics(self, file_path: Path):
        """Export comprehensive error analytics."""
        analytics_data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": self.get_error_summary(),
            "all_errors": [asdict(error) for error in self.errors],
        }
        
        if self.pattern_analyzer:
            analytics_data["patterns"] = [asdict(p) for p in self.pattern_analyzer.get_top_patterns(20)]
            analytics_data["trends"] = [asdict(t) for t in self.pattern_analyzer.analyze_trends()]
        
        analytics_data["notifications"] = list(self.notification_manager.notification_history)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(analytics_data, f, indent=2, default=str)
            
            logger.info(f"Error analytics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export error analytics: {e}")
            raise


class SafetyWrapper:
    """Enhanced safety wrapper for critical operations."""
    
    def __init__(self, error_handler: EnhancedErrorHandler):
        self.error_handler = error_handler
    
    def safe_execute(self, func: Callable, *args, 
                    fallback_value: Any = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.APPLICATION,
                    component: str = "unknown",
                    context: Dict[str, Any] = None,
                    **kwargs) -> Any:
        """Safely execute a function with enhanced error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.error_handler.handle_error(
                e, context, severity, category, component
            )
            logger.warning(f"Safe execution failed for {func.__name__}, returning fallback value")
            return fallback_value
    
    def safe_import(self, module_name: str, fallback=None, 
                   component: str = "import_system"):
        """Safely import a module with enhanced error handling."""
        try:
            import importlib
            return importlib.import_module(module_name)
        except ImportError as e:
            self.error_handler.handle_error(
                e, 
                {"module": module_name}, 
                ErrorSeverity.MEDIUM,
                ErrorCategory.SYSTEM,
                component
            )
            return fallback


# Global enhanced error handler instance
_error_handler = EnhancedErrorHandler()
_safety_wrapper = SafetyWrapper(_error_handler)


def get_error_handler() -> EnhancedErrorHandler:
    """Get global enhanced error handler."""
    return _error_handler


def get_safety_wrapper() -> SafetyWrapper:
    """Get global safety wrapper."""
    return _safety_wrapper


def resilient_operation(max_attempts: int = 3, circuit_breaker: bool = True,
                       error_types: Optional[List[str]] = None):
    """Enhanced decorator for resilient operations."""
    def decorator(func: Callable) -> Callable:
        # Apply retry policy
        retry_policy = RetryPolicy(
            max_attempts=max_attempts,
            retry_exceptions=(Exception,),
            retry_conditions={}
        )
        func = retry_policy(func)
        
        # Apply circuit breaker if requested
        if circuit_breaker:
            breaker = CircuitBreaker(error_types=error_types)
            func = breaker(func)
        
        return func
    
    return decorator


def handle_gracefully(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     category: ErrorCategory = ErrorCategory.APPLICATION,
                     component: str = "unknown",
                     fallback_value: Any = None):
    """Enhanced decorator to handle errors gracefully."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _safety_wrapper.safe_execute(
                func, *args, **kwargs, 
                fallback_value=fallback_value, 
                severity=severity,
                category=category,
                component=component,
                context={"function": func.__name__}
            )
        return wrapper
    return decorator


def error_analytics_dashboard() -> Dict[str, Any]:
    """Get comprehensive error analytics dashboard."""
    handler = get_error_handler()
    summary = handler.get_error_summary()
    
    # Add notification statistics
    notification_stats = {
        "total_notifications": len(handler.notification_manager.notification_history),
        "active_alerts": len(handler.notification_manager.escalation_state),
        "alert_rules": len(handler.notification_manager.alert_rules)
    }
    
    return {
        "error_summary": summary,
        "notification_stats": notification_stats,
        "system_health": {
            "error_rate_24h": len(handler.get_recent_errors(24)),
            "critical_errors_24h": len([
                e for e in handler.get_recent_errors(24)
                if e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.EMERGENCY]
            ]),
            "resolution_rate": len([
                e for e in handler.errors if e.resolved
            ]) / max(len(handler.errors), 1) * 100
        }
    }


def configure_error_notifications(email_config: Optional[Dict] = None,
                                 slack_webhook: Optional[str] = None,
                                 webhook_url: Optional[str] = None):
    """Configure error notification settings."""
    config = {}
    
    if email_config:
        config["email"] = email_config
    
    if slack_webhook:
        config["slack_webhook_url"] = slack_webhook
    
    if webhook_url:
        config["webhook_url"] = webhook_url
    
    # Update global error handler configuration
    handler = get_error_handler()
    handler.notification_manager.config.update(config)
    
    logger.info("Error notification configuration updated")


def add_custom_error_rule(rule: ErrorAlertRule):
    """Add a custom error alert rule."""
    handler = get_error_handler()
    handler.notification_manager.add_alert_rule(rule)


@contextmanager
def error_context(component: str, operation: str = None, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.APPLICATION):
    """Context manager for enhanced error handling."""
    context = {
        "component": component,
        "operation": operation,
        "start_time": time.time()
    }
    
    try:
        yield context
        context["success"] = True
        
    except Exception as e:
        context["success"] = False
        context["duration"] = time.time() - context["start_time"]
        
        handler = get_error_handler()
        handler.handle_error(e, context, severity, category, component)
        raise
    
    finally:
        if "duration" not in context:
            context["duration"] = time.time() - context["start_time"]