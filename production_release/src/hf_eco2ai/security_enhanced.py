"""Enhanced enterprise security and privacy features for carbon tracking."""

import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import re
from functools import wraps
from enum import Enum
import threading
import os
from urllib.parse import urlparse
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import ipaddress
from datetime import datetime, timedelta
import uuid
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UserRole(Enum):
    """User roles for RBAC."""
    GUEST = "guest"
    USER = "user"
    OPERATOR = "operator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class ThreatLevel(Enum):
    """Threat detection levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Enhanced audit event with digital signature."""
    event_id: str
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    event_type: str
    resource: str
    action: str
    outcome: str  # success, failure, blocked
    source_ip: str
    user_agent: str
    details: Dict[str, Any]
    digital_signature: Optional[str] = None
    
    def calculate_signature(self, secret_key: str) -> str:
        """Calculate digital signature for tamper detection."""
        data_to_sign = f"{self.event_id}{self.timestamp}{self.event_type}{self.outcome}"
        signature = hmac.new(
            secret_key.encode(),
            data_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_signature(self, secret_key: str) -> bool:
        """Verify digital signature for tamper detection."""
        if not self.digital_signature:
            return False
        expected_signature = self.calculate_signature(secret_key)
        return hmac.compare_digest(self.digital_signature, expected_signature)


@dataclass
class EnterpriseSecurityConfig:
    """Enhanced security configuration for enterprise carbon tracking."""
    
    # Encryption settings
    enable_data_encryption: bool = True
    enable_credential_encryption: bool = True
    encryption_key_rotation_days: int = 30
    
    # Access control
    enable_rbac: bool = True
    session_timeout_minutes: int = 60
    max_failed_login_attempts: int = 3
    lockout_duration_minutes: int = 30
    
    # Audit and compliance
    enable_audit_logging: bool = True
    enable_tamper_detection: bool = True
    audit_retention_days: int = 365
    enable_gdpr_compliance: bool = True
    
    # Threat detection
    enable_threat_detection: bool = True
    suspicious_pattern_threshold: int = 5
    rate_limit_requests_per_minute: int = 100
    enable_ip_blocking: bool = True
    
    # File and network security
    allowed_export_paths: List[str] = None
    blocked_hostnames: List[str] = None
    blocked_ip_ranges: List[str] = None
    max_file_size_mb: int = 100
    enable_file_integrity_checks: bool = True
    
    # Privacy and data protection
    enable_pii_detection: bool = True
    enable_data_anonymization: bool = True
    require_secure_connections: bool = True
    enable_data_retention_policies: bool = True
    
    def __post_init__(self):
        """Initialize default values."""
        if self.allowed_export_paths is None:
            self.allowed_export_paths = ["/tmp/carbon_reports", "./carbon_reports", "./reports"]
        
        if self.blocked_hostnames is None:
            self.blocked_hostnames = ["localhost", "127.0.0.1", "0.0.0.0"]
            
        if self.blocked_ip_ranges is None:
            self.blocked_ip_ranges = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]


@dataclass
class ThreatEvent:
    """Security threat event record."""
    event_id: str
    timestamp: datetime
    threat_level: ThreatLevel
    source_ip: str
    event_type: str
    description: str
    context: Dict[str, Any]
    mitigation_applied: bool = False
    resolved: bool = False


@dataclass
class UserSession:
    """User session management."""
    session_id: str
    user_id: str
    role: UserRole
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    permissions: Set[str]
    
    def is_expired(self, timeout_minutes: int) -> bool:
        """Check if session is expired."""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


class CredentialManager:
    """Secure credential management with encryption."""
    
    def __init__(self, config: EnterpriseSecurityConfig):
        self.config = config
        self._encryption_key = None
        self._salt = os.urandom(16)
        self._credentials_db = None
        self._init_encryption()
        self._init_storage()
    
    def _init_encryption(self):
        """Initialize encryption system."""
        # Use environment variable or generate key
        master_password = os.environ.get('HF_ECO2AI_MASTER_KEY', 'default-key-change-in-production')
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        self._fernet = Fernet(key)
    
    def _init_storage(self):
        """Initialize secure credential storage."""
        self._credentials_db = Path("./credentials.db")
        
        # Create database if it doesn't exist
        with sqlite3.connect(self._credentials_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS credentials (
                    id TEXT PRIMARY KEY,
                    service TEXT NOT NULL,
                    encrypted_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def store_credential(self, service: str, credential_data: Dict[str, Any]) -> str:
        """Store encrypted credential."""
        credential_id = str(uuid.uuid4())
        
        # Encrypt the credential data
        json_data = json.dumps(credential_data)
        encrypted_data = self._fernet.encrypt(json_data.encode())
        
        with sqlite3.connect(self._credentials_db) as conn:
            conn.execute(
                "INSERT INTO credentials (id, service, encrypted_data) VALUES (?, ?, ?)",
                (credential_id, service, base64.b64encode(encrypted_data).decode())
            )
            conn.commit()
        
        logger.info(f"Stored encrypted credential for service: {service}")
        return credential_id
    
    def retrieve_credential(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt credential."""
        with sqlite3.connect(self._credentials_db) as conn:
            cursor = conn.execute(
                "SELECT encrypted_data FROM credentials WHERE id = ?",
                (credential_id,)
            )
            result = cursor.fetchone()
        
        if not result:
            return None
        
        try:
            encrypted_data = base64.b64decode(result[0])
            decrypted_data = self._fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to decrypt credential: {e}")
            return None
    
    def rotate_encryption_key(self):
        """Rotate the encryption key for enhanced security."""
        # This would typically involve re-encrypting all stored credentials
        # with a new key - simplified implementation here
        logger.info("Encryption key rotation initiated")
        # Implementation would re-encrypt all credentials with new key


class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self, config: EnterpriseSecurityConfig):
        self.config = config
        self.sessions: Dict[str, UserSession] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()
        
        # Define role permissions
        self.role_permissions = {
            UserRole.GUEST: {"view_reports"},
            UserRole.USER: {"view_reports", "export_data", "view_metrics"},
            UserRole.OPERATOR: {"view_reports", "export_data", "view_metrics", "modify_settings"},
            UserRole.ADMIN: {"view_reports", "export_data", "view_metrics", "modify_settings", 
                           "manage_users", "view_audit_logs"},
            UserRole.SUPER_ADMIN: {"*"}  # All permissions
        }
    
    def authenticate_user(self, user_id: str, password: str, ip_address: str, 
                         user_agent: str) -> Optional[str]:
        """Authenticate user and create session."""
        # Check for too many failed attempts
        if self._is_locked_out(user_id):
            logger.warning(f"User {user_id} is locked out due to failed attempts")
            return None
        
        # Simplified authentication - in production, verify against secure storage
        if self._verify_credentials(user_id, password):
            session_id = str(uuid.uuid4())
            user_role = self._get_user_role(user_id)
            
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                role=user_role,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                ip_address=ip_address,
                user_agent=user_agent,
                permissions=self.role_permissions.get(user_role, set())
            )
            
            with self._lock:
                self.sessions[session_id] = session
            
            # Clear failed attempts on successful login
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            
            logger.info(f"User {user_id} authenticated successfully")
            return session_id
        else:
            self._record_failed_attempt(user_id)
            logger.warning(f"Failed authentication attempt for user {user_id}")
            return None
    
    def _verify_credentials(self, user_id: str, password: str) -> bool:
        """Verify user credentials - simplified implementation."""
        # In production, this would verify against encrypted password storage
        # For demo purposes, we'll use a simple check
        default_users = {
            "admin": "admin123",
            "operator": "operator123",
            "user": "user123"
        }
        return default_users.get(user_id) == password
    
    def _get_user_role(self, user_id: str) -> UserRole:
        """Get user role - simplified implementation."""
        role_mapping = {
            "admin": UserRole.ADMIN,
            "operator": UserRole.OPERATOR,
            "user": UserRole.USER
        }
        return role_mapping.get(user_id, UserRole.GUEST)
    
    def _is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        if user_id not in self.failed_attempts:
            return False
        
        recent_attempts = [
            attempt for attempt in self.failed_attempts[user_id]
            if datetime.now() - attempt < timedelta(minutes=self.config.lockout_duration_minutes)
        ]
        
        return len(recent_attempts) >= self.config.max_failed_login_attempts
    
    def _record_failed_attempt(self, user_id: str):
        """Record a failed login attempt."""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        self.failed_attempts[user_id].append(datetime.now())
        
        # Clean up old attempts
        cutoff = datetime.now() - timedelta(minutes=self.config.lockout_duration_minutes)
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > cutoff
        ]
    
    def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validate and refresh session."""
        with self._lock:
            session = self.sessions.get(session_id)
            
            if not session:
                return None
            
            if session.is_expired(self.config.session_timeout_minutes):
                del self.sessions[session_id]
                logger.info(f"Session {session_id} expired")
                return None
            
            session.update_activity()
            return session
    
    def check_permission(self, session_id: str, required_permission: str) -> bool:
        """Check if session has required permission."""
        session = self.validate_session(session_id)
        if not session:
            return False
        
        # Super admin has all permissions
        if "*" in session.permissions:
            return True
        
        return required_permission in session.permissions
    
    def logout(self, session_id: str):
        """Logout user and invalidate session."""
        with self._lock:
            if session_id in self.sessions:
                user_id = self.sessions[session_id].user_id
                del self.sessions[session_id]
                logger.info(f"User {user_id} logged out")


class ThreatDetectionEngine:
    """Advanced threat detection for suspicious patterns."""
    
    def __init__(self, config: EnterpriseSecurityConfig):
        self.config = config
        self.threat_events: List[ThreatEvent] = []
        self.request_history: Dict[str, List[datetime]] = {}
        self.blocked_ips: Set[str] = set()
        self._lock = threading.Lock()
        
        # Suspicious patterns
        self.suspicious_patterns = {
            "mass_data_export": r"export.*\b(all|\*|bulk)\b",
            "admin_privilege_escalation": r"(admin|root|sudo|privilege)",
            "sql_injection": r"(select|union|drop|delete|insert)\s*\w*\s*(from|into|where)",
            "path_traversal": r"\.\.[\/\\]",
            "credential_stuffing": r"(password|token|key)\s*[=:]",
        }
    
    def analyze_request(self, ip_address: str, user_agent: str, 
                       request_data: str, endpoint: str) -> ThreatLevel:
        """Analyze incoming request for threats."""
        threat_level = ThreatLevel.NONE
        threats_detected = []
        
        # Rate limiting check
        if self._check_rate_limit(ip_address):
            threats_detected.append("rate_limit_exceeded")
            threat_level = ThreatLevel.MEDIUM
        
        # Pattern analysis
        for pattern_name, pattern in self.suspicious_patterns.items():
            if re.search(pattern, request_data, re.IGNORECASE):
                threats_detected.append(pattern_name)
                threat_level = max(threat_level, ThreatLevel.HIGH, key=lambda x: x.value)
        
        # IP reputation check
        if self._check_ip_reputation(ip_address):
            threats_detected.append("malicious_ip")
            threat_level = ThreatLevel.CRITICAL
        
        # User agent analysis
        if self._analyze_user_agent(user_agent):
            threats_detected.append("suspicious_user_agent")
            threat_level = max(threat_level, ThreatLevel.LOW, key=lambda x: x.value)
        
        # Log threat event if detected
        if threats_detected:
            threat_event = ThreatEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                threat_level=threat_level,
                source_ip=ip_address,
                event_type="suspicious_request",
                description=f"Threats detected: {', '.join(threats_detected)}",
                context={
                    "endpoint": endpoint,
                    "user_agent": user_agent,
                    "threats": threats_detected,
                    "request_data_hash": hashlib.sha256(request_data.encode()).hexdigest()[:16]
                }
            )
            
            with self._lock:
                self.threat_events.append(threat_event)
            
            # Apply mitigation
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self._apply_mitigation(ip_address, threat_level)
        
        return threat_level
    
    def _check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP exceeds rate limit."""
        now = datetime.now()
        
        if ip_address not in self.request_history:
            self.request_history[ip_address] = []
        
        # Clean old requests (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.request_history[ip_address] = [
            req_time for req_time in self.request_history[ip_address]
            if req_time > cutoff
        ]
        
        # Add current request
        self.request_history[ip_address].append(now)
        
        # Check if rate limit exceeded
        return len(self.request_history[ip_address]) > self.config.rate_limit_requests_per_minute
    
    def _check_ip_reputation(self, ip_address: str) -> bool:
        """Check IP against reputation databases (simplified)."""
        # In production, this would check against threat intelligence feeds
        known_malicious_ips = {
            "192.168.1.100",  # Example malicious IP
            "10.0.0.100"
        }
        return ip_address in known_malicious_ips
    
    def _analyze_user_agent(self, user_agent: str) -> bool:
        """Analyze user agent for suspicious patterns."""
        suspicious_agents = [
            "sqlmap", "nmap", "masscan", "nikto", "dirb", "gobuster",
            "python-requests", "curl", "wget"
        ]
        
        user_agent_lower = user_agent.lower()
        return any(agent in user_agent_lower for agent in suspicious_agents)
    
    def _apply_mitigation(self, ip_address: str, threat_level: ThreatLevel):
        """Apply mitigation measures for detected threats."""
        if threat_level == ThreatLevel.CRITICAL:
            # Block IP immediately
            self.blocked_ips.add(ip_address)
            logger.critical(f"Blocked IP {ip_address} due to critical threat")
        elif threat_level == ThreatLevel.HIGH:
            # Temporary rate limiting
            logger.warning(f"Applied additional rate limiting to IP {ip_address}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked."""
        return ip_address in self.blocked_ips
    
    def unblock_ip(self, ip_address: str):
        """Manually unblock an IP."""
        self.blocked_ips.discard(ip_address)
        logger.info(f"Unblocked IP {ip_address}")
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get threat detection summary."""
        recent_threats = [
            event for event in self.threat_events
            if datetime.now() - event.timestamp < timedelta(hours=24)
        ]
        
        threat_counts = {}
        for event in recent_threats:
            level = event.threat_level.value
            threat_counts[level] = threat_counts.get(level, 0) + 1
        
        return {
            "total_threats_24h": len(recent_threats),
            "threat_breakdown": threat_counts,
            "blocked_ips": list(self.blocked_ips),
            "active_monitoring": self.config.enable_threat_detection
        }


class EnhancedSecurityValidator:
    """Enhanced security validator with advanced threat detection."""
    
    def __init__(self, config: EnterpriseSecurityConfig = None):
        """Initialize enhanced security validator."""
        self.config = config or EnterpriseSecurityConfig()
        self.credential_manager = CredentialManager(self.config)
        self.rbac_manager = RBACManager(self.config)
        self.threat_engine = ThreatDetectionEngine(self.config)
        
        # Enhanced sensitive patterns
        self.sensitive_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:\d{4}[- ]?){3}\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\bsk-[a-zA-Z0-9]{48}\b',  # API keys (OpenAI style)
            r'\b[A-Za-z0-9]{40}\b',  # Generic 40-char keys
            r'password[\s]*[:=][\s]*[\S]+',  # Password fields
            r'token[\s]*[:=][\s]*[\S]+',  # Token fields
            r'\bAKIA[0-9A-Z]{16}\b',  # AWS Access Key
            r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',  # UUID
            r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}\b',  # GitHub tokens
        ]
    
    def validate_request(self, request_data: Dict[str, Any], session_id: str, 
                        ip_address: str, user_agent: str, endpoint: str) -> Tuple[bool, str, ThreatLevel]:
        """Comprehensive request validation."""
        # Check if IP is blocked
        if self.threat_engine.is_ip_blocked(ip_address):
            return False, "IP address is blocked due to security threats", ThreatLevel.CRITICAL
        
        # Validate session
        session = self.rbac_manager.validate_session(session_id)
        if not session:
            return False, "Invalid or expired session", ThreatLevel.MEDIUM
        
        # Analyze for threats
        request_str = json.dumps(request_data, default=str)
        threat_level = self.threat_engine.analyze_request(
            ip_address, user_agent, request_str, endpoint
        )
        
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            return False, f"Request blocked due to security threat: {threat_level.value}", threat_level
        
        # Check permissions for endpoint
        required_permission = self._get_endpoint_permission(endpoint)
        if required_permission and not self.rbac_manager.check_permission(session_id, required_permission):
            return False, f"Insufficient permissions for {endpoint}", ThreatLevel.MEDIUM
        
        return True, "Request validated successfully", threat_level
    
    def _get_endpoint_permission(self, endpoint: str) -> Optional[str]:
        """Get required permission for endpoint."""
        endpoint_permissions = {
            "/export": "export_data",
            "/admin": "manage_users",
            "/settings": "modify_settings",
            "/audit": "view_audit_logs"
        }
        return endpoint_permissions.get(endpoint)
    
    def validate_export_path(self, path: str, session_id: str = None) -> Tuple[bool, str]:
        """Validate export path with enhanced security checks."""
        # Session validation for export operations
        if session_id and not self.rbac_manager.check_permission(session_id, "export_data"):
            return False, "Insufficient permissions for data export"
        
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
        
        # Enhanced path traversal detection
        if any(dangerous in str(path) for dangerous in ["..", "~", "$"]):
            return False, "Potential path traversal or variable expansion detected"
        
        # Check for absolute paths (security risk)
        if os.path.isabs(path):
            return False, "Absolute paths not allowed for security reasons"
        
        # Check file extension with enhanced validation
        allowed_extensions = {".json", ".csv", ".txt", ".html", ".md", ".pdf"}
        if path_obj.suffix.lower() not in allowed_extensions:
            return False, f"File extension not allowed: {path_obj.suffix}"
        
        # Additional security checks for file integrity
        if self.config.enable_file_integrity_checks:
            if self._is_suspicious_filename(path_obj.name):
                return False, "Suspicious filename pattern detected"
        
        return True, "Path is valid"
    
    def _is_suspicious_filename(self, filename: str) -> bool:
        """Check for suspicious filename patterns."""
        suspicious_patterns = [
            r"\.(exe|bat|cmd|sh|ps1|vbs)$",  # Executable files
            r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])\.",  # Windows reserved names
            r"[<>:\"|?*]",  # Invalid characters
            r"^\.",  # Hidden files (starting with dot)
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return True
        
        return False
    
    def calculate_file_integrity_hash(self, file_path: Path) -> str:
        """Calculate file integrity hash."""
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {e}")
            return ""
    
    def validate_file_integrity(self, file_path: Path, expected_hash: str) -> bool:
        """Validate file integrity against expected hash."""
        actual_hash = self.calculate_file_integrity_hash(file_path)
        return actual_hash == expected_hash
    
    def scan_for_pii(self, data: str, anonymize: bool = False) -> List[Dict[str, Any]]:
        """Scan data for potential PII with enhanced detection."""
        if not self.config.enable_pii_detection:
            return []
        
        findings = []
        
        for i, pattern in enumerate(self.sensitive_patterns):
            matches = re.finditer(pattern, data, re.IGNORECASE)
            for match in matches:
                pattern_type = self._get_pattern_type(i)
                
                finding = {
                    "pattern_id": i,
                    "pattern_type": pattern_type,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "context": data[max(0, match.start()-20):match.end()+20],
                    "confidence": self._calculate_confidence(match.group(), pattern_type),
                    "anonymized_value": self._anonymize_pii(match.group(), pattern_type) if anonymize else None
                }
                findings.append(finding)
        
        return findings
    
    def _get_pattern_type(self, pattern_id: int) -> str:
        """Get human-readable pattern type."""
        pattern_types = [
            "email", "credit_card", "ssn", "openai_key", "generic_key", 
            "password", "token", "aws_key", "uuid", "github_token"
        ]
        return pattern_types[pattern_id] if pattern_id < len(pattern_types) else "unknown"
    
    def _calculate_confidence(self, value: str, pattern_type: str) -> str:
        """Calculate confidence level for PII detection."""
        # Enhanced confidence calculation based on pattern specificity
        if pattern_type in ["email", "credit_card", "ssn"]:
            return "high"
        elif pattern_type in ["openai_key", "aws_key", "github_token"]:
            return "high"
        elif pattern_type in ["uuid", "generic_key"]:
            return "medium"
        else:
            return "low"
    
    def _anonymize_pii(self, value: str, pattern_type: str) -> str:
        """Anonymize PII value based on type."""
        if pattern_type == "email":
            # Keep domain, anonymize local part
            if "@" in value:
                local, domain = value.split("@", 1)
                return f"***@{domain}"
        elif pattern_type == "credit_card":
            # Show only last 4 digits
            digits_only = re.sub(r'\D', '', value)
            if len(digits_only) >= 4:
                return f"****-****-****-{digits_only[-4:]}"
        elif pattern_type in ["api_key", "token", "password"]:
            # Complete redaction for sensitive credentials
            return "[REDACTED]"
        elif pattern_type == "ssn":
            # Show only last 4 digits
            return f"***-**-{value[-4:]}"
        
        # Default anonymization
        return f"[{pattern_type.upper()}_REDACTED]"
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        return {
            "security_status": {
                "encryption_enabled": self.config.enable_data_encryption,
                "rbac_enabled": self.config.enable_rbac,
                "threat_detection_enabled": self.config.enable_threat_detection,
                "audit_logging_enabled": self.config.enable_audit_logging,
                "gdpr_compliance_enabled": self.config.enable_gdpr_compliance
            },
            "threat_summary": self.threat_engine.get_threat_summary(),
            "session_stats": {
                "active_sessions": len(self.rbac_manager.sessions),
                "failed_attempts": sum(len(attempts) for attempts in self.rbac_manager.failed_attempts.values())
            },
            "blocked_ips": list(self.threat_engine.blocked_ips),
            "timestamp": datetime.now().isoformat()
        }


class EnterpriseSecurityManager:
    """Enterprise-grade security manager with advanced threat protection."""
    
    def __init__(self, config: Optional[EnterpriseSecurityConfig] = None):
        """Initialize enterprise security manager."""
        self.config = config or EnterpriseSecurityConfig()
        self.validator = EnhancedSecurityValidator(self.config)
        self.audit_secret_key = os.environ.get('HF_ECO2AI_AUDIT_KEY', 'default-audit-key')
        self._lock = threading.Lock()
        
        # Security monitoring
        self._security_events: List[Dict[str, Any]] = []
        self._start_security_monitoring()
        
        logger.info("Initialized EnterpriseSecurityManager with advanced threat protection")
    
    def _start_security_monitoring(self):
        """Start continuous security monitoring."""
        if self.config.enable_threat_detection:
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._security_monitoring_loop,
                daemon=True,
                name="security-monitor"
            )
            monitor_thread.start()
    
    def _security_monitoring_loop(self):
        """Continuous security monitoring loop."""
        while True:
            try:
                # Monitor for security anomalies
                self._check_security_health()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                time.sleep(60)
    
    def _check_security_health(self):
        """Check overall security system health."""
        health_issues = []
        
        # Check for too many failed login attempts
        failed_attempts = sum(
            len(attempts) for attempts in self.validator.rbac_manager.failed_attempts.values()
        )
        if failed_attempts > 50:  # Threshold for concern
            health_issues.append("high_failed_login_attempts")
        
        # Check for blocked IPs
        blocked_count = len(self.validator.threat_engine.blocked_ips)
        if blocked_count > 10:
            health_issues.append("high_blocked_ip_count")
        
        # Check recent threat events
        recent_threats = [
            event for event in self.validator.threat_engine.threat_events
            if datetime.now() - event.timestamp < timedelta(hours=1)
        ]
        if len(recent_threats) > 20:
            health_issues.append("high_threat_activity")
        
        if health_issues:
            self._log_audit_event("security_health_alert", {
                "issues": health_issues,
                "failed_attempts": failed_attempts,
                "blocked_ips": blocked_count,
                "recent_threats": len(recent_threats)
            }, "warning")
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any], 
                        severity: str, user_id: str = None, session_id: str = None,
                        source_ip: str = "127.0.0.1", user_agent: str = "system"):
        """Log audit event with digital signature."""
        audit_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            resource="carbon_tracking_system",
            action=event_type,
            outcome="success" if severity in ["info", "warning"] else "failure",
            source_ip=source_ip,
            user_agent=user_agent,
            details=details
        )
        
        # Calculate digital signature
        audit_event.digital_signature = audit_event.calculate_signature(self.audit_secret_key)
        
        # Store audit event
        with self._lock:
            self._security_events.append(asdict(audit_event))
        
        # Log to standard logger
        log_level = getattr(logging, severity.upper(), logging.INFO)
        logger.log(log_level, f"AUDIT [{event_type}]: {details}")
    
    def validate_request(self, request_data: Dict[str, Any], session_id: str,
                        ip_address: str, user_agent: str, endpoint: str) -> Dict[str, Any]:
        """Comprehensive request validation with enterprise security."""
        with self._lock:
            try:
                # Validate request through security validator
                is_valid, reason, threat_level = self.validator.validate_request(
                    request_data, session_id, ip_address, user_agent, endpoint
                )
                
                validation_result = {
                    "valid": is_valid,
                    "reason": reason,
                    "threat_level": threat_level.value,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Log validation event
                self._log_audit_event("request_validation", {
                    "endpoint": endpoint,
                    "valid": is_valid,
                    "threat_level": threat_level.value,
                    "session_id": session_id,
                    "source_ip": ip_address
                }, "info" if is_valid else "warning", source_ip=ip_address, 
                   user_agent=user_agent)
                
                # PII scanning if request is valid
                if is_valid and isinstance(request_data, dict):
                    pii_findings = self.validator.scan_for_pii(
                        json.dumps(request_data),
                        anonymize=self.config.enable_data_anonymization
                    )
                    if pii_findings:
                        validation_result["pii_detected"] = True
                        validation_result["pii_count"] = len(pii_findings)
                        
                        self._log_audit_event("pii_detected", {
                            "endpoint": endpoint,
                            "findings_count": len(pii_findings),
                            "pattern_types": [f["pattern_type"] for f in pii_findings]
                        }, "high", source_ip=ip_address, user_agent=user_agent)
                
                return validation_result
                
            except Exception as e:
                self._log_audit_event("security_validation_error", {
                    "error": str(e),
                    "endpoint": endpoint
                }, "error", source_ip=ip_address, user_agent=user_agent)
                return {
                    "valid": False,
                    "reason": f"Security validation failed: {str(e)}",
                    "threat_level": "high"
                }
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard."""
        # Get threat summary
        threat_summary = self.validator.threat_engine.get_threat_summary()
        
        # Get session statistics
        active_sessions = len(self.validator.rbac_manager.sessions)
        failed_attempts = sum(
            len(attempts) for attempts in self.validator.rbac_manager.failed_attempts.values()
        )
        
        # Get recent security events
        recent_events = [event for event in self._security_events[-10:]]
        
        return {
            "security_overview": {
                "status": "active",
                "encryption_enabled": self.config.enable_data_encryption,
                "rbac_enabled": self.config.enable_rbac,
                "threat_detection_enabled": self.config.enable_threat_detection,
                "compliance_monitoring": self.config.enable_gdpr_compliance
            },
            "threat_intelligence": threat_summary,
            "access_control": {
                "active_sessions": active_sessions,
                "failed_login_attempts_24h": failed_attempts,
                "blocked_ips": len(threat_summary.get("blocked_ips", []))
            },
            "compliance_status": {
                "gdpr_compliant": True,  # Would be calculated from actual compliance check
                "audit_trail_enabled": self.config.enable_audit_logging,
                "data_retention_enforced": self.config.enable_data_retention_policies
            },
            "recent_security_events": recent_events,
            "last_updated": datetime.now().isoformat()
        }
    
    def perform_security_audit(self) -> Dict[str, Any]:
        """Perform comprehensive security audit."""
        audit_results = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "security_configuration": asdict(self.config),
            "threat_analysis": self.validator.threat_engine.get_threat_summary(),
            "audit_trail_integrity": self._verify_audit_trail_integrity(),
            "recommendations": []
        }
        
        # Add general security recommendations
        if not self.config.enable_data_encryption:
            audit_results["recommendations"].append("Enable data encryption for enhanced security")
        
        if self.config.session_timeout_minutes > 120:
            audit_results["recommendations"].append("Consider reducing session timeout for better security")
        
        if not self.config.enable_tamper_detection:
            audit_results["recommendations"].append("Enable tamper detection for audit logs")
        
        # Log audit event
        self._log_audit_event("security_audit_completed", {
            "audit_id": audit_results["audit_id"],
            "issues_found": len(audit_results["recommendations"])
        }, "info")
        
        return audit_results
    
    def _verify_audit_trail_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit trail using digital signatures."""
        integrity_results = {
            "total_events": len(self._security_events),
            "verified_events": 0,
            "tampered_events": 0,
            "unsigned_events": 0
        }
        
        for event_data in self._security_events:
            if "digital_signature" not in event_data or not event_data["digital_signature"]:
                integrity_results["unsigned_events"] += 1
                continue
            
            # Recreate audit event object to verify signature
            try:
                event = AuditEvent(**event_data)
                if event.verify_signature(self.audit_secret_key):
                    integrity_results["verified_events"] += 1
                else:
                    integrity_results["tampered_events"] += 1
            except Exception as e:
                logger.error(f"Error verifying audit event signature: {e}")
                integrity_results["tampered_events"] += 1
        
        return integrity_results


# Global instances
_enterprise_security_config = EnterpriseSecurityConfig()
_enterprise_security_manager = EnterpriseSecurityManager(_enterprise_security_config)


def get_enterprise_security_manager() -> EnterpriseSecurityManager:
    """Get global enterprise security manager instance."""
    return _enterprise_security_manager


def get_enterprise_security_config() -> EnterpriseSecurityConfig:
    """Get global enterprise security configuration."""
    return _enterprise_security_config


def secure_carbon_operation(required_permission: str = "view_reports"):
    """Decorator for securing carbon tracking operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session info from kwargs
            session_id = kwargs.get('session_id')
            ip_address = kwargs.get('ip_address', '127.0.0.1')
            user_agent = kwargs.get('user_agent', 'unknown')
            
            if session_id:
                security_manager = get_enterprise_security_manager()
                
                # Validate session and permissions
                session = security_manager.validator.rbac_manager.validate_session(session_id)
                if not session:
                    raise PermissionError("Invalid or expired session")
                
                if not security_manager.validator.rbac_manager.check_permission(session_id, required_permission):
                    raise PermissionError(f"Insufficient permissions: {required_permission} required")
                
                # Log the operation
                security_manager._log_audit_event("carbon_operation", {
                    "function": func.__name__,
                    "permission": required_permission,
                    "user_id": session.user_id
                }, "info", session.user_id, session_id, ip_address, user_agent)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator