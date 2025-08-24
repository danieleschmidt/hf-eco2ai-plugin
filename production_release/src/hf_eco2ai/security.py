"""Security and privacy features for carbon tracking."""

import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from functools import wraps
from enum import Enum
import threading
import os
import inspect
import json
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration for carbon tracking."""
    
    enable_data_encryption: bool = False
    enable_audit_logging: bool = True
    allowed_export_paths: List[str] = None
    blocked_hostnames: List[str] = None
    max_file_size_mb: int = 100
    enable_pii_detection: bool = True
    require_secure_connections: bool = True
    
    def __post_init__(self):
        """Initialize default values."""
        if self.allowed_export_paths is None:
            self.allowed_export_paths = ["/tmp/carbon_reports", "./carbon_reports", "./reports"]
        
        if self.blocked_hostnames is None:
            self.blocked_hostnames = []


class SecurityValidator:
    """Validate security aspects of carbon tracking."""
    
    def __init__(self, config: SecurityConfig = None):
        """Initialize security validator.
        
        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()
        self.sensitive_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:\d{4}[- ]?){3}\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\bsk-[a-zA-Z0-9]{48}\b',  # API keys (OpenAI style)
            r'\b[A-Za-z0-9]{40}\b',  # Generic 40-char keys
            r'password[\s]*[:=][\s]*[\S]+',  # Password fields
            r'token[\s]*[:=][\s]*[\S]+',  # Token fields
        ]
    
    def validate_export_path(self, path: str) -> tuple[bool, str]:
        """Validate if export path is safe.
        
        Args:
            path: Path to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        path_obj = Path(path).resolve()
        
        # Check against allowed paths
        allowed = False
        for allowed_path in self.config.allowed_export_paths:
            allowed_path_obj = Path(allowed_path).resolve()
            try:
                path_obj.relative_to(allowed_path_obj)
                allowed = True
                break
            except ValueError:
                continue
        
        if not allowed:
            return False, f"Path not in allowed export locations: {self.config.allowed_export_paths}"
        
        # Check for path traversal attempts
        if ".." in str(path) or path.startswith("/"):
            return False, "Potential path traversal detected"
        
        # Check file extension
        allowed_extensions = {".json", ".csv", ".txt", ".html", ".md"}
        if path_obj.suffix.lower() not in allowed_extensions:
            return False, f"File extension not allowed: {path_obj.suffix}"
        
        return True, "Path is valid"
    
    def validate_url(self, url: str) -> tuple[bool, str]:
        """Validate if URL is safe for connections.
        
        Args:
            url: URL to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            parsed = urlparse(url)
        except Exception as e:
            return False, f"Invalid URL format: {e}"
        
        # Check protocol
        if self.config.require_secure_connections and parsed.scheme != "https":
            return False, "Only HTTPS connections allowed"
        
        if parsed.scheme not in {"http", "https"}:
            return False, f"Protocol not allowed: {parsed.scheme}"
        
        # Check hostname
        if parsed.hostname in self.config.blocked_hostnames:
            return False, f"Hostname blocked: {parsed.hostname}"
        
        # Check for private/local addresses
        import ipaddress
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_loopback:
                return False, "Private/loopback addresses not allowed"
        except ValueError:
            # Not an IP address, that's okay
            pass
        
        return True, "URL is valid"
    
    def scan_for_pii(self, data: str) -> List[Dict[str, Any]]:
        """Scan data for potential PII.
        
        Args:
            data: Data to scan
            
        Returns:
            List of potential PII findings
        """
        if not self.config.enable_pii_detection:
            return []
        
        findings = []
        
        for i, pattern in enumerate(self.sensitive_patterns):
            matches = re.finditer(pattern, data, re.IGNORECASE)
            for match in matches:
                finding = {
                    "pattern_id": i,
                    "pattern_type": self._get_pattern_type(i),
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "context": data[max(0, match.start()-20):match.end()+20],
                    "confidence": "high"
                }
                findings.append(finding)
        
        return findings
    
    def _get_pattern_type(self, pattern_id: int) -> str:
        """Get human-readable pattern type."""
        pattern_types = [
            "email", "credit_card", "ssn", "api_key", "generic_key", 
            "password", "token"
        ]
        return pattern_types[pattern_id] if pattern_id < len(pattern_types) else "unknown"
    
    def validate_file_size(self, file_path: str) -> tuple[bool, str]:
        """Validate file size is within limits.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            file_size = Path(file_path).stat().st_size
            max_size = self.config.max_file_size_mb * 1024 * 1024
            
            if file_size > max_size:
                return False, f"File size {file_size/1024/1024:.1f}MB exceeds limit of {self.config.max_file_size_mb}MB"
            
            return True, "File size is within limits"
        
        except FileNotFoundError:
            return True, "File not found (will be created)"
        except Exception as e:
            return False, f"Error checking file size: {e}"


class DataSanitizer:
    """Sanitize data before storage or export."""
    
    def __init__(self, security_config: SecurityConfig = None):
        """Initialize data sanitizer.
        
        Args:
            security_config: Security configuration
        """
        self.security_config = security_config or SecurityConfig()
        self.validator = SecurityValidator(security_config)
    
    def sanitize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metrics data for export.
        
        Args:
            metrics: Raw metrics data
            
        Returns:
            Sanitized metrics data
        """
        sanitized = {}
        
        for key, value in metrics.items():
            # Remove or mask sensitive keys
            if self._is_sensitive_key(key):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str):
                # Scan and sanitize string values
                sanitized[key] = self._sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_metrics(value)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_value(item) for item in value]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if key contains sensitive information."""
        sensitive_keys = {
            "password", "token", "secret", "key", "auth", "credential",
            "user", "username", "email", "hostname", "ip", "address"
        }
        
        key_lower = key.lower()
        return any(sensitive_key in key_lower for sensitive_key in sensitive_keys)
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string content."""
        # Check for PII
        findings = self.validator.scan_for_pii(text)
        
        if findings:
            sanitized_text = text
            # Replace PII with placeholders (in reverse order to maintain positions)
            for finding in reversed(findings):
                placeholder = f"[{finding['pattern_type'].upper()}_REDACTED]"
                sanitized_text = (
                    sanitized_text[:finding['start_pos']] +
                    placeholder +
                    sanitized_text[finding['end_pos']:]
                )
            return sanitized_text
        
        return text
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize individual value."""
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, dict):
            return self.sanitize_metrics(value)
        else:
            return value


class AuditLogger:
    """Audit logging for carbon tracking operations."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize audit logger.
        
        Args:
            log_file: Optional audit log file path
        """
        self.log_file = log_file
        self.session_id = secrets.token_hex(8)
        self.audit_entries: List[Dict[str, Any]] = []
        
        if log_file:
            self.log_path = Path(log_file)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event_type: str, details: Dict[str, Any], 
                  severity: str = "info"):
        """Log an audit event.
        
        Args:
            event_type: Type of event (e.g., 'export', 'access', 'error')
            details: Event details
            severity: Event severity ('info', 'warning', 'error', 'critical')
        """
        audit_entry = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "checksum": self._calculate_checksum(details)
        }
        
        self.audit_entries.append(audit_entry)
        
        # Write to file if configured
        if self.log_file:
            self._write_audit_entry(audit_entry)
        
        # Also log to standard logger
        log_level = getattr(logging, severity.upper(), logging.INFO)
        logger.log(log_level, f"AUDIT [{event_type}]: {details}")
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for audit entry integrity."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _write_audit_entry(self, entry: Dict[str, Any]):
        """Write audit entry to file."""
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def export_audit_log(self, output_path: str, 
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None):
        """Export audit log for a time range.
        
        Args:
            output_path: Output file path
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
        """
        filtered_entries = []
        
        for entry in self.audit_entries:
            timestamp = entry['timestamp']
            
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            
            filtered_entries.append(entry)
        
        output_data = {
            "session_id": self.session_id,
            "export_timestamp": time.time(),
            "entries": filtered_entries,
            "summary": {
                "total_entries": len(filtered_entries),
                "time_range": {
                    "start": start_time,
                    "end": end_time
                }
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        self.log_event("audit_export", {
            "export_path": str(output_path),
            "entries_exported": len(filtered_entries)
        })


class ComplianceChecker:
    """Check compliance with various regulations and standards."""
    
    def __init__(self):
        """Initialize compliance checker."""
        self.compliance_frameworks = {
            "gdpr": self._check_gdpr_compliance,
            "ccpa": self._check_ccpa_compliance,
            "sox": self._check_sox_compliance,
            "pci_dss": self._check_pci_compliance
        }
    
    def check_compliance(self, framework: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with specific framework.
        
        Args:
            framework: Compliance framework to check
            data: Data to evaluate
            
        Returns:
            Compliance check results
        """
        if framework not in self.compliance_frameworks:
            return {
                "framework": framework,
                "status": "unsupported",
                "message": f"Framework {framework} not supported"
            }
        
        checker = self.compliance_frameworks[framework]
        return checker(data)
    
    def _check_gdpr_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check GDPR compliance."""
        issues = []
        
        # Check for personal data
        validator = SecurityValidator()
        pii_findings = validator.scan_for_pii(json.dumps(data))
        
        if pii_findings:
            issues.append({
                "issue": "potential_personal_data",
                "description": "Data may contain personal information",
                "findings": len(pii_findings),
                "severity": "high"
            })
        
        # Check for data retention policy
        if "retention_policy" not in data:
            issues.append({
                "issue": "missing_retention_policy",
                "description": "No data retention policy specified",
                "severity": "medium"
            })
        
        return {
            "framework": "gdpr",
            "status": "compliant" if not issues else "issues_found",
            "issues": issues,
            "recommendations": self._get_gdpr_recommendations(issues)
        }
    
    def _check_ccpa_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check CCPA compliance."""
        issues = []
        
        # Similar checks as GDPR but with CCPA-specific requirements
        validator = SecurityValidator()
        pii_findings = validator.scan_for_pii(json.dumps(data))
        
        if pii_findings:
            issues.append({
                "issue": "personal_information_detected",
                "description": "Data contains personal information subject to CCPA",
                "severity": "high"
            })
        
        return {
            "framework": "ccpa",
            "status": "compliant" if not issues else "issues_found",
            "issues": issues
        }
    
    def _check_sox_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check SOX compliance for financial reporting."""
        issues = []
        
        # Check for audit trail
        if "audit_trail" not in data:
            issues.append({
                "issue": "missing_audit_trail",
                "description": "No audit trail for data changes",
                "severity": "high"
            })
        
        # Check for data integrity
        if "data_integrity_hash" not in data:
            issues.append({
                "issue": "missing_integrity_verification",
                "description": "No data integrity verification",
                "severity": "medium"
            })
        
        return {
            "framework": "sox",
            "status": "compliant" if not issues else "issues_found",
            "issues": issues
        }
    
    def _check_pci_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check PCI DSS compliance."""
        issues = []
        
        # Check for credit card data
        validator = SecurityValidator()
        data_str = json.dumps(data)
        
        if re.search(r'\b(?:\d{4}[- ]?){3}\d{4}\b', data_str):
            issues.append({
                "issue": "potential_card_data",
                "description": "Data may contain credit card numbers",
                "severity": "critical"
            })
        
        return {
            "framework": "pci_dss",
            "status": "compliant" if not issues else "issues_found",
            "issues": issues
        }
    
    def _get_gdpr_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Get GDPR compliance recommendations."""
        recommendations = []
        
        for issue in issues:
            if issue["issue"] == "potential_personal_data":
                recommendations.extend([
                    "Implement data anonymization or pseudonymization",
                    "Add explicit consent mechanisms",
                    "Implement data subject rights (access, portability, erasure)",
                    "Add privacy notice and data processing documentation"
                ])
            
            elif issue["issue"] == "missing_retention_policy":
                recommendations.extend([
                    "Define data retention periods",
                    "Implement automated data deletion",
                    "Document legal basis for data processing"
                ])
        
        return list(set(recommendations))  # Remove duplicates


# Global instances
_audit_logger = AuditLogger()
_security_validator = SecurityValidator()
_data_sanitizer = DataSanitizer()
_compliance_checker = ComplianceChecker()

def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    return _audit_logger

def get_security_validator() -> SecurityValidator:
    """Get global security validator instance."""
    return _security_validator

def get_data_sanitizer() -> DataSanitizer:
    """Get global data sanitizer instance."""
    return _data_sanitizer

def get_compliance_checker() -> ComplianceChecker:
    """Get global compliance checker instance."""
    return _compliance_checker

def secure_export(func):
    """Decorator for secure data export operations."""
    def wrapper(*args, **kwargs):
        # Log the export operation
        _audit_logger.log_event("data_export", {
            "function": func.__name__,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys())
        })
        
        try:
            result = func(*args, **kwargs)
            _audit_logger.log_event("export_success", {
                "function": func.__name__
            })
            return result
        except Exception as e:
            _audit_logger.log_event("export_error", {
                "function": func.__name__,
                "error": str(e)
            }, severity="error")
            raise
    
    return wrapper


class SecurityManager:
    """Comprehensive security manager for carbon tracking operations."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security manager."""
        self.config = config or SecurityConfig()
        self.validator = SecurityValidator()
        self.sanitizer = DataSanitizer()
        self.audit_logger = AuditLogger()
        self.compliance_checker = ComplianceChecker()
        self._lock = threading.Lock()
        
        logger.info("Initialized SecurityManager with robust protections")
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Comprehensive data validation."""
        with self._lock:
            try:
                # Input validation
                validation_result = self.validator.validate_input(data)
                if not validation_result["valid"]:
                    self.audit_logger.log_event("validation_failure", {
                        "reason": validation_result["message"],
                        "data_type": type(data).__name__
                    }, severity="warning")
                
                # PII scanning
                if isinstance(data, (str, dict)):
                    data_str = json.dumps(data) if isinstance(data, dict) else data
                    pii_findings = self.validator.scan_for_pii(data_str)
                    if pii_findings:
                        self.audit_logger.log_event("pii_detected", {
                            "findings_count": len(pii_findings)
                        }, severity="high")
                
                return validation_result
                
            except Exception as e:
                self.audit_logger.log_event("security_validation_error", {
                    "error": str(e)
                }, severity="error")
                raise
    
    def sanitize_data(self, data: Any) -> Any:
        """Sanitize data for safe processing."""
        try:
            if isinstance(data, dict):
                sanitized = {}
                for key, value in data.items():
                    sanitized[key] = self.sanitizer.sanitize_value(value)
                return sanitized
            elif isinstance(data, str):
                return self.sanitizer.sanitize_string(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Data sanitization failed: {e}")
            raise
    
    def check_compliance(self, framework: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with security frameworks."""
        return self.compliance_checker.check_compliance(framework, data)
    
    def audit_operation(self, operation: str, metadata: Dict[str, Any]):
        """Audit security-relevant operations."""
        self.audit_logger.log_event(operation, metadata)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary."""
        return {
            "config": {
                "encryption_enabled": self.config.enable_data_encryption,
                "audit_enabled": self.config.enable_audit_logging,
                "pii_detection": self.config.enable_pii_detection,
                "validation_strict": self.config.strict_validation
            },
            "status": "active",
            "components": {
                "validator": "active",
                "sanitizer": "active", 
                "audit_logger": "active",
                "compliance_checker": "active"
            }
        }


# Global instances
_security_manager = SecurityManager()