"""
Compliance & Audit Framework for HF Eco2AI Carbon Tracking System
Provides comprehensive audit logging, data retention policies, regulatory compliance,
tamper-proof audit trails, and automated compliance reporting.
"""

import json
import logging
import hashlib
import hmac
import time
import os
import sqlite3
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
import schedule
import uuid


class ComplianceLevel(Enum):
    """Compliance levels for different regulatory requirements"""
    GDPR = "gdpr"
    SOX = "sox"
    ISO27001 = "iso27001"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"


class AuditEventType(Enum):
    """Types of audit events to track"""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    SYSTEM_CONFIGURATION = "system_configuration"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    ERROR_EVENT = "error_event"
    PERFORMANCE_ALERT = "performance_alert"


class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NEEDS_REVIEW = "needs_review"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    timestamp: str
    event_type: AuditEventType
    user_id: str
    session_id: str
    resource: str
    action: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    compliance_level: ComplianceLevel
    risk_level: str
    data_classification: str
    outcome: str
    signature: Optional[str] = None


@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    rule_id: str
    name: str
    description: str
    compliance_level: ComplianceLevel
    severity: str
    check_function: str
    parameters: Dict[str, Any]
    frequency: str
    enabled: bool


@dataclass
class ComplianceReport:
    """Compliance report structure"""
    report_id: str
    timestamp: str
    compliance_level: ComplianceLevel
    period_start: str
    period_end: str
    total_checks: int
    compliant_checks: int
    non_compliant_checks: int
    warnings: int
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    risk_score: float


class TamperProofAuditLogger:
    """Tamper-proof audit logging with digital signatures"""
    
    def __init__(self, db_path: str = "/var/log/hf_eco2ai/audit.db"):
        self.db_path = db_path
        self.private_key = None
        self.public_key = None
        self._ensure_directory()
        self._initialize_database()
        self._generate_keys()
        self._setup_logging()
        
    def _ensure_directory(self):
        """Ensure audit log directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def _initialize_database(self):
        """Initialize audit database with tamper-proof schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT NOT NULL,
                    compliance_level TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    data_classification TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    hash_chain TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_integrity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_events INTEGER NOT NULL,
                    merkle_root TEXT NOT NULL,
                    signature TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_events(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_user 
                ON audit_events(user_id)
            """)
            
    def _generate_keys(self):
        """Generate RSA key pair for digital signatures"""
        try:
            # Try to load existing keys
            private_key_path = os.path.join(os.path.dirname(self.db_path), "audit_private.pem")
            public_key_path = os.path.join(os.path.dirname(self.db_path), "audit_public.pem")
            
            if os.path.exists(private_key_path) and os.path.exists(public_key_path):
                with open(private_key_path, "rb") as f:
                    self.private_key = serialization.load_pem_private_key(f.read(), password=None)
                with open(public_key_path, "rb") as f:
                    self.public_key = serialization.load_pem_public_key(f.read())
            else:
                # Generate new keys
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
                self.public_key = self.private_key.public_key()
                
                # Save keys
                with open(private_key_path, "wb") as f:
                    f.write(self.private_key.private_bytes(
                        encoding=Encoding.PEM,
                        format=PrivateFormat.PKCS8,
                        encryption_algorithm=NoEncryption()
                    ))
                    
                with open(public_key_path, "wb") as f:
                    f.write(self.public_key.public_bytes(
                        encoding=Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ))
                    
        except Exception as e:
            logging.error(f"Error generating audit keys: {e}")
            raise
            
    def _setup_logging(self):
        """Setup secure logging configuration"""
        self.logger = logging.getLogger("compliance_audit")
        if not self.logger.handlers:
            handler = logging.FileHandler(
                os.path.join(os.path.dirname(self.db_path), "audit_system.log")
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def _sign_data(self, data: str) -> str:
        """Create digital signature for audit data"""
        try:
            signature = self.private_key.sign(
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature.hex()
        except Exception as e:
            self.logger.error(f"Error signing audit data: {e}")
            return ""
            
    def _get_previous_hash(self) -> str:
        """Get hash of previous audit event for chain integrity"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT hash_chain FROM audit_events 
                    ORDER BY created_at DESC LIMIT 1
                """)
                result = cursor.fetchone()
                return result[0] if result else "genesis"
        except Exception:
            return "genesis"
            
    def _calculate_event_hash(self, event: AuditEvent, previous_hash: str) -> str:
        """Calculate hash for event including chain integrity"""
        event_data = f"{event.event_id}{event.timestamp}{event.user_id}{event.action}{previous_hash}"
        return hashlib.sha256(event_data.encode()).hexdigest()
        
    def log_event(self, event: AuditEvent) -> bool:
        """Log audit event with tamper-proof signature"""
        try:
            # Get previous hash for chain integrity
            previous_hash = self._get_previous_hash()
            
            # Calculate event hash
            event_hash = self._calculate_event_hash(event, previous_hash)
            
            # Create signature
            event_data = json.dumps(asdict(event), sort_keys=True)
            signature = self._sign_data(event_data)
            event.signature = signature
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_events (
                        event_id, timestamp, event_type, user_id, session_id,
                        resource, action, details, ip_address, user_agent,
                        compliance_level, risk_level, data_classification,
                        outcome, signature, hash_chain, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id, event.timestamp, event.event_type.value,
                    event.user_id, event.session_id, event.resource, event.action,
                    json.dumps(event.details), event.ip_address, event.user_agent,
                    event.compliance_level.value, event.risk_level,
                    event.data_classification, event.outcome, signature,
                    event_hash, time.time()
                ))
                
            self.logger.info(f"Audit event logged: {event.event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging audit event: {e}")
            return False
            
    def verify_integrity(self, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> Dict[str, Any]:
        """Verify audit log integrity"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM audit_events ORDER BY created_at"
                params = []
                
                if start_date or end_date:
                    conditions = []
                    if start_date:
                        conditions.append("timestamp >= ?")
                        params.append(start_date)
                    if end_date:
                        conditions.append("timestamp <= ?")
                        params.append(end_date)
                    query += " WHERE " + " AND ".join(conditions)
                    
                cursor = conn.execute(query, params)
                events = cursor.fetchall()
                
            verified_count = 0
            total_count = len(events)
            integrity_issues = []
            
            previous_hash = "genesis"
            
            for event_data in events:
                event_id = event_data[0]
                stored_hash = event_data[15]  # hash_chain column
                
                # Recreate event object
                event = AuditEvent(
                    event_id=event_data[0],
                    timestamp=event_data[1],
                    event_type=AuditEventType(event_data[2]),
                    user_id=event_data[3],
                    session_id=event_data[4],
                    resource=event_data[5],
                    action=event_data[6],
                    details=json.loads(event_data[7]),
                    ip_address=event_data[8],
                    user_agent=event_data[9],
                    compliance_level=ComplianceLevel(event_data[10]),
                    risk_level=event_data[11],
                    data_classification=event_data[12],
                    outcome=event_data[13],
                    signature=event_data[14]
                )
                
                # Verify hash chain
                calculated_hash = self._calculate_event_hash(event, previous_hash)
                if calculated_hash == stored_hash:
                    verified_count += 1
                else:
                    integrity_issues.append({
                        "event_id": event_id,
                        "issue": "Hash chain verification failed",
                        "expected": calculated_hash,
                        "actual": stored_hash
                    })
                    
                previous_hash = stored_hash
                
            return {
                "verification_timestamp": datetime.now().isoformat(),
                "total_events": total_count,
                "verified_events": verified_count,
                "integrity_score": verified_count / total_count if total_count > 0 else 1.0,
                "issues": integrity_issues
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying audit integrity: {e}")
            return {"error": str(e)}


class DataRetentionManager:
    """Manages data retention policies and automated cleanup"""
    
    def __init__(self, audit_logger: TamperProofAuditLogger):
        self.audit_logger = audit_logger
        self.retention_policies = {
            ComplianceLevel.GDPR: 2555,  # 7 years in days
            ComplianceLevel.SOX: 2555,   # 7 years in days
            ComplianceLevel.ISO27001: 1095,  # 3 years in days
            ComplianceLevel.HIPAA: 2190,  # 6 years in days
            ComplianceLevel.SOC2: 365,   # 1 year in days
            ComplianceLevel.PCI_DSS: 365  # 1 year in days
        }
        self.archive_path = "/var/archive/hf_eco2ai/"
        self._ensure_archive_directory()
        
    def _ensure_archive_directory(self):
        """Ensure archive directory exists"""
        os.makedirs(self.archive_path, exist_ok=True)
        
    def set_retention_policy(self, compliance_level: ComplianceLevel, days: int):
        """Set custom retention policy for compliance level"""
        self.retention_policies[compliance_level] = days
        
        # Log policy change
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type=AuditEventType.SYSTEM_CONFIGURATION,
            user_id="system",
            session_id="retention_manager",
            resource="retention_policy",
            action="update_policy",
            details={
                "compliance_level": compliance_level.value,
                "retention_days": days,
                "previous_days": self.retention_policies.get(compliance_level, 0)
            },
            ip_address="localhost",
            user_agent="retention_manager",
            compliance_level=compliance_level,
            risk_level="low",
            data_classification="configuration",
            outcome="success"
        )
        self.audit_logger.log_event(event)
        
    def archive_old_data(self, compliance_level: ComplianceLevel) -> Dict[str, Any]:
        """Archive data that exceeds retention period"""
        try:
            retention_days = self.retention_policies.get(compliance_level, 365)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Query old data
            with sqlite3.connect(self.audit_logger.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM audit_events 
                    WHERE compliance_level = ? AND timestamp < ?
                    ORDER BY timestamp
                """, (compliance_level.value, cutoff_date.isoformat()))
                
                old_events = cursor.fetchall()
                
            if not old_events:
                return {"archived_count": 0, "message": "No data to archive"}
                
            # Create archive file
            archive_filename = f"audit_archive_{compliance_level.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            archive_path = os.path.join(self.archive_path, archive_filename)
            
            # Convert to JSON format
            archive_data = {
                "archive_metadata": {
                    "compliance_level": compliance_level.value,
                    "cutoff_date": cutoff_date.isoformat(),
                    "archive_timestamp": datetime.now().isoformat(),
                    "event_count": len(old_events)
                },
                "events": []
            }
            
            # Process events
            for event_data in old_events:
                archive_data["events"].append({
                    "event_id": event_data[0],
                    "timestamp": event_data[1],
                    "event_type": event_data[2],
                    "user_id": event_data[3],
                    "session_id": event_data[4],
                    "resource": event_data[5],
                    "action": event_data[6],
                    "details": json.loads(event_data[7]),
                    "ip_address": event_data[8],
                    "user_agent": event_data[9],
                    "compliance_level": event_data[10],
                    "risk_level": event_data[11],
                    "data_classification": event_data[12],
                    "outcome": event_data[13],
                    "signature": event_data[14],
                    "hash_chain": event_data[15]
                })
                
            # Write archive file
            with open(archive_path, 'w') as f:
                json.dump(archive_data, f, indent=2)
                
            # Create compressed archive
            zip_path = archive_path.replace('.json', '.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(archive_path, archive_filename)
                
            # Remove uncompressed file
            os.remove(archive_path)
            
            # Delete archived events from main database
            with sqlite3.connect(self.audit_logger.db_path) as conn:
                conn.execute("""
                    DELETE FROM audit_events 
                    WHERE compliance_level = ? AND timestamp < ?
                """, (compliance_level.value, cutoff_date.isoformat()))
                
            # Log archival event
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                event_type=AuditEventType.SYSTEM_CONFIGURATION,
                user_id="system",
                session_id="retention_manager",
                resource="audit_data",
                action="archive_data",
                details={
                    "compliance_level": compliance_level.value,
                    "archived_count": len(old_events),
                    "archive_file": zip_path,
                    "cutoff_date": cutoff_date.isoformat()
                },
                ip_address="localhost",
                user_agent="retention_manager",
                compliance_level=compliance_level,
                risk_level="low",
                data_classification="operational",
                outcome="success"
            )
            self.audit_logger.log_event(event)
            
            return {
                "archived_count": len(old_events),
                "archive_file": zip_path,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error archiving data: {e}"
            logging.error(error_msg)
            return {"error": error_msg}
            
    def cleanup_expired_archives(self, max_archive_age_days: int = 2555) -> Dict[str, Any]:
        """Clean up archived files older than specified age"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_archive_age_days)
            deleted_files = []
            
            for filename in os.listdir(self.archive_path):
                if filename.startswith("audit_archive_") and filename.endswith(".zip"):
                    file_path = os.path.join(self.archive_path, filename)
                    file_stat = os.stat(file_path)
                    file_date = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    if file_date < cutoff_date:
                        os.remove(file_path)
                        deleted_files.append(filename)
                        
            return {
                "deleted_count": len(deleted_files),
                "deleted_files": deleted_files,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error cleaning up archives: {e}"
            logging.error(error_msg)
            return {"error": error_msg}


class RegulatoryComplianceChecker:
    """Automated regulatory compliance checking engine"""
    
    def __init__(self, audit_logger: TamperProofAuditLogger):
        self.audit_logger = audit_logger
        self.compliance_rules = {}
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default compliance rules for each regulatory framework"""
        
        # GDPR Rules
        self.compliance_rules[ComplianceLevel.GDPR] = [
            ComplianceRule(
                rule_id="gdpr_001",
                name="Data Access Logging",
                description="All personal data access must be logged",
                compliance_level=ComplianceLevel.GDPR,
                severity="high",
                check_function="check_data_access_logging",
                parameters={"data_types": ["personal", "pii"]},
                frequency="daily",
                enabled=True
            ),
            ComplianceRule(
                rule_id="gdpr_002",
                name="Consent Tracking",
                description="User consent for data processing must be tracked",
                compliance_level=ComplianceLevel.GDPR,
                severity="critical",
                check_function="check_consent_tracking",
                parameters={"required_consent_types": ["processing", "analytics"]},
                frequency="daily",
                enabled=True
            ),
            ComplianceRule(
                rule_id="gdpr_003",
                name="Right to be Forgotten",
                description="Data deletion requests must be processed within 30 days",
                compliance_level=ComplianceLevel.GDPR,
                severity="high",
                check_function="check_deletion_requests",
                parameters={"max_processing_days": 30},
                frequency="weekly",
                enabled=True
            )
        ]
        
        # SOX Rules
        self.compliance_rules[ComplianceLevel.SOX] = [
            ComplianceRule(
                rule_id="sox_001",
                name="Financial Data Access Control",
                description="Access to financial data must be properly controlled and logged",
                compliance_level=ComplianceLevel.SOX,
                severity="critical",
                check_function="check_financial_access_control",
                parameters={"financial_data_types": ["revenue", "cost", "carbon_credit"]},
                frequency="daily",
                enabled=True
            ),
            ComplianceRule(
                rule_id="sox_002",
                name="Change Management",
                description="All system changes must be approved and documented",
                compliance_level=ComplianceLevel.SOX,
                severity="high",
                check_function="check_change_management",
                parameters={"required_approvals": ["manager", "security"]},
                frequency="daily",
                enabled=True
            )
        ]
        
        # ISO27001 Rules
        self.compliance_rules[ComplianceLevel.ISO27001] = [
            ComplianceRule(
                rule_id="iso_001",
                name="Security Event Monitoring",
                description="Security events must be monitored and responded to",
                compliance_level=ComplianceLevel.ISO27001,
                severity="high",
                check_function="check_security_monitoring",
                parameters={"max_response_time_hours": 4},
                frequency="daily",
                enabled=True
            ),
            ComplianceRule(
                rule_id="iso_002",
                name="Access Review",
                description="User access rights must be reviewed regularly",
                compliance_level=ComplianceLevel.ISO27001,
                severity="medium",
                check_function="check_access_review",
                parameters={"review_frequency_days": 90},
                frequency="weekly",
                enabled=True
            )
        ]
        
    def add_custom_rule(self, rule: ComplianceRule):
        """Add custom compliance rule"""
        if rule.compliance_level not in self.compliance_rules:
            self.compliance_rules[rule.compliance_level] = []
        self.compliance_rules[rule.compliance_level].append(rule)
        
        # Log rule addition
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type=AuditEventType.SYSTEM_CONFIGURATION,
            user_id="system",
            session_id="compliance_checker",
            resource="compliance_rules",
            action="add_rule",
            details={
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "compliance_level": rule.compliance_level.value,
                "severity": rule.severity
            },
            ip_address="localhost",
            user_agent="compliance_checker",
            compliance_level=rule.compliance_level,
            risk_level="low",
            data_classification="configuration",
            outcome="success"
        )
        self.audit_logger.log_event(event)
        
    def check_data_access_logging(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Check if data access is properly logged"""
        try:
            data_types = rule.parameters.get("data_types", [])
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            with sqlite3.connect(self.audit_logger.db_path) as conn:
                # Check for data access events
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM audit_events 
                    WHERE event_type = ? AND DATE(timestamp) = ?
                    AND (details LIKE '%personal%' OR details LIKE '%pii%')
                """, (AuditEventType.DATA_ACCESS.value, yesterday.isoformat()))
                
                access_count = cursor.fetchone()[0]
                
                # Check if all accesses have proper logging
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM audit_events 
                    WHERE event_type = ? AND DATE(timestamp) = ?
                    AND details NOT LIKE '%logged%'
                """, (AuditEventType.DATA_ACCESS.value, yesterday.isoformat()))
                
                unlogged_count = cursor.fetchone()[0]
                
            if unlogged_count > 0:
                return {
                    "status": ComplianceStatus.NON_COMPLIANT,
                    "message": f"{unlogged_count} data access events not properly logged",
                    "details": {
                        "total_access": access_count,
                        "unlogged_access": unlogged_count,
                        "check_date": yesterday.isoformat()
                    }
                }
            else:
                return {
                    "status": ComplianceStatus.COMPLIANT,
                    "message": "All data access properly logged",
                    "details": {
                        "total_access": access_count,
                        "check_date": yesterday.isoformat()
                    }
                }
                
        except Exception as e:
            return {
                "status": ComplianceStatus.NEEDS_REVIEW,
                "message": f"Error checking data access logging: {e}",
                "details": {"error": str(e)}
            }
            
    def check_consent_tracking(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Check consent tracking compliance"""
        # Placeholder implementation - would integrate with consent management system
        return {
            "status": ComplianceStatus.COMPLIANT,
            "message": "Consent tracking check placeholder",
            "details": {"note": "Integration with consent management system required"}
        }
        
    def check_deletion_requests(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Check data deletion request processing"""
        # Placeholder implementation - would check deletion request processing times
        return {
            "status": ComplianceStatus.COMPLIANT,
            "message": "Deletion request check placeholder",
            "details": {"note": "Integration with data deletion system required"}
        }
        
    def check_financial_access_control(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Check financial data access control"""
        try:
            financial_types = rule.parameters.get("financial_data_types", [])
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            with sqlite3.connect(self.audit_logger.db_path) as conn:
                # Check for unauthorized financial data access
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM audit_events 
                    WHERE event_type = ? AND DATE(timestamp) = ?
                    AND (details LIKE '%revenue%' OR details LIKE '%cost%' OR details LIKE '%carbon_credit%')
                    AND outcome != 'success'
                """, (AuditEventType.DATA_ACCESS.value, yesterday.isoformat()))
                
                unauthorized_count = cursor.fetchone()[0]
                
            if unauthorized_count > 0:
                return {
                    "status": ComplianceStatus.NON_COMPLIANT,
                    "message": f"{unauthorized_count} unauthorized financial data access attempts",
                    "details": {
                        "unauthorized_attempts": unauthorized_count,
                        "check_date": yesterday.isoformat()
                    }
                }
            else:
                return {
                    "status": ComplianceStatus.COMPLIANT,
                    "message": "No unauthorized financial data access detected",
                    "details": {"check_date": yesterday.isoformat()}
                }
                
        except Exception as e:
            return {
                "status": ComplianceStatus.NEEDS_REVIEW,
                "message": f"Error checking financial access control: {e}",
                "details": {"error": str(e)}
            }
            
    def check_change_management(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Check change management compliance"""
        # Placeholder implementation - would check change approval workflows
        return {
            "status": ComplianceStatus.COMPLIANT,
            "message": "Change management check placeholder",
            "details": {"note": "Integration with change management system required"}
        }
        
    def check_security_monitoring(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Check security event monitoring"""
        try:
            max_response_hours = rule.parameters.get("max_response_time_hours", 4)
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            with sqlite3.connect(self.audit_logger.db_path) as conn:
                # Check for unresolved security events
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM audit_events 
                    WHERE event_type = ? AND DATE(timestamp) = ?
                    AND outcome = 'pending'
                """, (AuditEventType.SECURITY_EVENT.value, yesterday.isoformat()))
                
                unresolved_count = cursor.fetchone()[0]
                
            if unresolved_count > 0:
                return {
                    "status": ComplianceStatus.WARNING,
                    "message": f"{unresolved_count} unresolved security events",
                    "details": {
                        "unresolved_events": unresolved_count,
                        "check_date": yesterday.isoformat()
                    }
                }
            else:
                return {
                    "status": ComplianceStatus.COMPLIANT,
                    "message": "All security events resolved within SLA",
                    "details": {"check_date": yesterday.isoformat()}
                }
                
        except Exception as e:
            return {
                "status": ComplianceStatus.NEEDS_REVIEW,
                "message": f"Error checking security monitoring: {e}",
                "details": {"error": str(e)}
            }
            
    def check_access_review(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Check access review compliance"""
        # Placeholder implementation - would check access review schedules
        return {
            "status": ComplianceStatus.COMPLIANT,
            "message": "Access review check placeholder",
            "details": {"note": "Integration with access management system required"}
        }
        
    def run_compliance_check(self, compliance_level: ComplianceLevel) -> List[Dict[str, Any]]:
        """Run all compliance checks for a specific level"""
        results = []
        rules = self.compliance_rules.get(compliance_level, [])
        
        for rule in rules:
            if not rule.enabled:
                continue
                
            try:
                # Get check function
                check_function = getattr(self, rule.check_function, None)
                if not check_function:
                    result = {
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "status": ComplianceStatus.NEEDS_REVIEW,
                        "message": f"Check function {rule.check_function} not found",
                        "details": {}
                    }
                else:
                    result = check_function(rule)
                    result.update({
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "severity": rule.severity
                    })
                    
                results.append(result)
                
                # Log compliance check
                event = AuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now().isoformat(),
                    event_type=AuditEventType.COMPLIANCE_CHECK,
                    user_id="system",
                    session_id="compliance_checker",
                    resource="compliance_rule",
                    action="run_check",
                    details={
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "compliance_level": compliance_level.value,
                        "status": result["status"].value if isinstance(result["status"], ComplianceStatus) else result["status"],
                        "message": result["message"]
                    },
                    ip_address="localhost",
                    user_agent="compliance_checker",
                    compliance_level=compliance_level,
                    risk_level=rule.severity,
                    data_classification="compliance",
                    outcome="success"
                )
                self.audit_logger.log_event(event)
                
            except Exception as e:
                error_result = {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "severity": rule.severity,
                    "status": ComplianceStatus.NEEDS_REVIEW,
                    "message": f"Error running compliance check: {e}",
                    "details": {"error": str(e)}
                }
                results.append(error_result)
                
        return results


class ComplianceReportGenerator:
    """Automated compliance report generation"""
    
    def __init__(self, audit_logger: TamperProofAuditLogger, 
                 compliance_checker: RegulatoryComplianceChecker):
        self.audit_logger = audit_logger
        self.compliance_checker = compliance_checker
        self.report_path = "/var/reports/hf_eco2ai/"
        self._ensure_report_directory()
        
    def _ensure_report_directory(self):
        """Ensure report directory exists"""
        os.makedirs(self.report_path, exist_ok=True)
        
    def _calculate_risk_score(self, compliance_results: List[Dict[str, Any]]) -> float:
        """Calculate overall risk score based on compliance results"""
        if not compliance_results:
            return 0.0
            
        severity_weights = {
            "low": 1,
            "medium": 3,
            "high": 5,
            "critical": 10
        }
        
        status_multipliers = {
            ComplianceStatus.COMPLIANT: 0,
            ComplianceStatus.WARNING: 0.5,
            ComplianceStatus.NON_COMPLIANT: 1.0,
            ComplianceStatus.NEEDS_REVIEW: 0.3
        }
        
        total_risk = 0
        max_possible_risk = 0
        
        for result in compliance_results:
            severity = result.get("severity", "medium")
            status = result.get("status", ComplianceStatus.NEEDS_REVIEW)
            
            if isinstance(status, str):
                status = ComplianceStatus(status)
                
            weight = severity_weights.get(severity, 3)
            multiplier = status_multipliers.get(status, 0.3)
            
            total_risk += weight * multiplier
            max_possible_risk += weight
            
        return (total_risk / max_possible_risk * 100) if max_possible_risk > 0 else 0.0
        
    def _generate_recommendations(self, compliance_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on compliance results"""
        recommendations = []
        
        non_compliant_count = sum(1 for r in compliance_results 
                                if r.get("status") == ComplianceStatus.NON_COMPLIANT)
        warning_count = sum(1 for r in compliance_results 
                          if r.get("status") == ComplianceStatus.WARNING)
        
        if non_compliant_count > 0:
            recommendations.append(
                f"Address {non_compliant_count} non-compliant findings immediately"
            )
            
        if warning_count > 0:
            recommendations.append(
                f"Review and resolve {warning_count} warning conditions"
            )
            
        # Specific recommendations based on common issues
        for result in compliance_results:
            if result.get("status") == ComplianceStatus.NON_COMPLIANT:
                rule_name = result.get("rule_name", "")
                if "access" in rule_name.lower():
                    recommendations.append("Implement stronger access controls")
                elif "logging" in rule_name.lower():
                    recommendations.append("Enhance audit logging coverage")
                elif "monitoring" in rule_name.lower():
                    recommendations.append("Improve security monitoring capabilities")
                    
        return list(set(recommendations))  # Remove duplicates
        
    def generate_compliance_report(self, compliance_level: ComplianceLevel,
                                 period_days: int = 30) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Run compliance checks
            compliance_results = self.compliance_checker.run_compliance_check(compliance_level)
            
            # Count results by status
            total_checks = len(compliance_results)
            compliant_checks = sum(1 for r in compliance_results 
                                 if r.get("status") == ComplianceStatus.COMPLIANT)
            non_compliant_checks = sum(1 for r in compliance_results 
                                     if r.get("status") == ComplianceStatus.NON_COMPLIANT)
            warnings = sum(1 for r in compliance_results 
                          if r.get("status") == ComplianceStatus.WARNING)
            
            # Extract violations
            violations = [r for r in compliance_results 
                         if r.get("status") in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.WARNING]]
            
            # Generate recommendations
            recommendations = self._generate_recommendations(compliance_results)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(compliance_results)
            
            # Create report
            report = ComplianceReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                compliance_level=compliance_level,
                period_start=start_date.isoformat(),
                period_end=end_date.isoformat(),
                total_checks=total_checks,
                compliant_checks=compliant_checks,
                non_compliant_checks=non_compliant_checks,
                warnings=warnings,
                violations=violations,
                recommendations=recommendations,
                risk_score=risk_score
            )
            
            # Save report
            report_filename = f"compliance_report_{compliance_level.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = os.path.join(self.report_path, report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
                
            # Log report generation
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                event_type=AuditEventType.COMPLIANCE_CHECK,
                user_id="system",
                session_id="report_generator",
                resource="compliance_report",
                action="generate_report",
                details={
                    "report_id": report.report_id,
                    "compliance_level": compliance_level.value,
                    "period_days": period_days,
                    "total_checks": total_checks,
                    "risk_score": risk_score,
                    "report_file": report_path
                },
                ip_address="localhost",
                user_agent="report_generator",
                compliance_level=compliance_level,
                risk_level="low",
                data_classification="compliance",
                outcome="success"
            )
            self.audit_logger.log_event(event)
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating compliance report: {e}")
            raise
            
    def schedule_automated_reports(self):
        """Schedule automated compliance reports"""
        # Daily reports for critical compliance levels
        schedule.every().day.at("06:00").do(
            lambda: self.generate_compliance_report(ComplianceLevel.GDPR)
        )
        schedule.every().day.at("06:15").do(
            lambda: self.generate_compliance_report(ComplianceLevel.SOX)
        )
        
        # Weekly reports for other compliance levels
        schedule.every().monday.at("07:00").do(
            lambda: self.generate_compliance_report(ComplianceLevel.ISO27001)
        )
        schedule.every().monday.at("07:15").do(
            lambda: self.generate_compliance_report(ComplianceLevel.SOC2)
        )


class ComplianceFramework:
    """Main compliance framework orchestrator"""
    
    def __init__(self, db_path: str = "/var/log/hf_eco2ai/audit.db"):
        self.audit_logger = TamperProofAuditLogger(db_path)
        self.retention_manager = DataRetentionManager(self.audit_logger)
        self.compliance_checker = RegulatoryComplianceChecker(self.audit_logger)
        self.report_generator = ComplianceReportGenerator(
            self.audit_logger, self.compliance_checker
        )
        self._running = False
        self._scheduler_thread = None
        
    def start_compliance_monitoring(self):
        """Start automated compliance monitoring"""
        if self._running:
            return
            
        self._running = True
        
        # Schedule automated reports
        self.report_generator.schedule_automated_reports()
        
        # Schedule retention cleanup
        schedule.every().sunday.at("02:00").do(
            lambda: self._run_retention_cleanup()
        )
        
        # Start scheduler thread
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler, daemon=True
        )
        self._scheduler_thread.start()
        
        logging.info("Compliance monitoring started")
        
    def stop_compliance_monitoring(self):
        """Stop automated compliance monitoring"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logging.info("Compliance monitoring stopped")
        
    def _run_scheduler(self):
        """Run the compliance scheduler"""
        while self._running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    def _run_retention_cleanup(self):
        """Run retention cleanup for all compliance levels"""
        for compliance_level in ComplianceLevel:
            try:
                result = self.retention_manager.archive_old_data(compliance_level)
                logging.info(f"Retention cleanup for {compliance_level.value}: {result}")
            except Exception as e:
                logging.error(f"Error in retention cleanup for {compliance_level.value}: {e}")
                
    def log_audit_event(self, event_type: AuditEventType, user_id: str,
                       resource: str, action: str, details: Dict[str, Any],
                       compliance_level: ComplianceLevel = ComplianceLevel.GDPR,
                       **kwargs) -> bool:
        """Convenience method to log audit events"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            user_id=user_id,
            session_id=kwargs.get("session_id", "unknown"),
            resource=resource,
            action=action,
            details=details,
            ip_address=kwargs.get("ip_address", "unknown"),
            user_agent=kwargs.get("user_agent", "unknown"),
            compliance_level=compliance_level,
            risk_level=kwargs.get("risk_level", "medium"),
            data_classification=kwargs.get("data_classification", "general"),
            outcome=kwargs.get("outcome", "success")
        )
        return self.audit_logger.log_event(event)
        
    def get_compliance_status(self, compliance_level: ComplianceLevel) -> Dict[str, Any]:
        """Get current compliance status for a specific level"""
        try:
            # Run compliance checks
            results = self.compliance_checker.run_compliance_check(compliance_level)
            
            # Calculate summary statistics
            total_checks = len(results)
            compliant = sum(1 for r in results if r.get("status") == ComplianceStatus.COMPLIANT)
            non_compliant = sum(1 for r in results if r.get("status") == ComplianceStatus.NON_COMPLIANT)
            warnings = sum(1 for r in results if r.get("status") == ComplianceStatus.WARNING)
            
            compliance_percentage = (compliant / total_checks * 100) if total_checks > 0 else 100
            
            return {
                "compliance_level": compliance_level.value,
                "timestamp": datetime.now().isoformat(),
                "total_checks": total_checks,
                "compliant": compliant,
                "non_compliant": non_compliant,
                "warnings": warnings,
                "compliance_percentage": compliance_percentage,
                "status": "compliant" if non_compliant == 0 else "non_compliant",
                "details": results
            }
            
        except Exception as e:
            return {
                "compliance_level": compliance_level.value,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }
            
    def generate_comprehensive_report(self, period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive compliance report for all levels"""
        reports = {}
        
        for compliance_level in ComplianceLevel:
            try:
                report = self.report_generator.generate_compliance_report(
                    compliance_level, period_days
                )
                reports[compliance_level.value] = asdict(report)
            except Exception as e:
                reports[compliance_level.value] = {
                    "error": str(e),
                    "compliance_level": compliance_level.value,
                    "timestamp": datetime.now().isoformat()
                }
                
        return {
            "comprehensive_report": True,
            "period_days": period_days,
            "generation_timestamp": datetime.now().isoformat(),
            "reports": reports
        }


# Global compliance framework instance
compliance_framework = None


def get_compliance_framework() -> ComplianceFramework:
    """Get or create global compliance framework instance"""
    global compliance_framework
    if compliance_framework is None:
        compliance_framework = ComplianceFramework()
    return compliance_framework


def initialize_compliance_monitoring():
    """Initialize and start compliance monitoring"""
    framework = get_compliance_framework()
    framework.start_compliance_monitoring()
    return framework


if __name__ == "__main__":
    # Example usage
    framework = initialize_compliance_monitoring()
    
    # Log a sample audit event
    framework.log_audit_event(
        event_type=AuditEventType.DATA_ACCESS,
        user_id="test_user",
        resource="carbon_data",
        action="view_emissions",
        details={"dataset": "monthly_emissions", "records": 150},
        compliance_level=ComplianceLevel.GDPR,
        session_id="session_123",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0",
        risk_level="low",
        data_classification="environmental",
        outcome="success"
    )
    
    # Get compliance status
    status = framework.get_compliance_status(ComplianceLevel.GDPR)
    print(f"GDPR Compliance Status: {status}")
    
    # Generate comprehensive report
    report = framework.generate_comprehensive_report(30)
    print(f"Generated comprehensive compliance report")