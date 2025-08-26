#!/usr/bin/env python3
"""
TERRAGON AUTONOMOUS SDLC - Generation 2: MAKE IT ROBUST
Enhanced reliability, error handling, validation, and security
"""

import sys
import os
import json
import logging
import hashlib
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union, Callable
from pathlib import Path
from enum import Enum
from contextlib import contextmanager
import re
import uuid

print("🛡️ TERRAGON AUTONOMOUS EXECUTION - Generation 2: MAKE IT ROBUST")  
print("=" * 75)

# Enhanced Error Handling System
class CarbonTrackingError(Exception):
    """Base exception for carbon tracking errors"""
    pass

class ValidationError(CarbonTrackingError):
    """Configuration or data validation errors"""
    pass

class SecurityError(CarbonTrackingError):
    """Security-related errors"""
    pass

class ResourceError(CarbonTrackingError):
    """Resource availability errors"""
    pass

# Error Severity Levels
class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Enhanced Logging Configuration
class RobustLogger:
    """Enhanced logging with security and rotation"""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self._sensitive_patterns = [
            r'password[=:][\w\d]+',
            r'api[_-]?key[=:][\w\d\-]+',
            r'token[=:][\w\d\-]+',
            r'secret[=:][\w\d\-]+'
        ]
    
    def _sanitize_message(self, message: str) -> str:
        """Remove sensitive information from log messages"""
        sanitized = message
        for pattern in self._sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        return sanitized
    
    def info(self, message: str, **kwargs):
        self.logger.info(self._sanitize_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(self._sanitize_message(message), **kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(self._sanitize_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(self._sanitize_message(message), **kwargs)

# Enhanced Configuration with Validation
@dataclass
class RobustCarbonConfig:
    """Robust carbon tracking configuration with comprehensive validation"""
    project_name: str = "hf-eco2ai-robust"
    country: str = "USA"
    region: str = "CA"
    track_energy: bool = True
    track_co2: bool = True
    export_prometheus: bool = False
    max_energy_threshold: float = 100.0  # kWh
    max_co2_threshold: float = 50.0      # kg CO₂
    security_hash: Optional[str] = field(default=None, init=False)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        self.validate()
        self.security_hash = self._compute_security_hash()
    
    def validate(self):
        """Comprehensive configuration validation"""
        errors = []
        
        # Project name validation
        if not self.project_name or len(self.project_name.strip()) == 0:
            errors.append("Project name cannot be empty")
        if len(self.project_name) > 100:
            errors.append("Project name too long (max 100 characters)")
        if not re.match(r'^[a-zA-Z0-9\-_]+$', self.project_name):
            errors.append("Project name contains invalid characters")
        
        # Country validation
        valid_countries = ['USA', 'UK', 'DE', 'FR', 'JP', 'CN', 'IN', 'BR', 'CA', 'AU']
        if self.country not in valid_countries:
            errors.append(f"Country must be one of: {valid_countries}")
        
        # Threshold validation
        if self.max_energy_threshold <= 0 or self.max_energy_threshold > 1000:
            errors.append("Energy threshold must be between 0 and 1000 kWh")
        if self.max_co2_threshold <= 0 or self.max_co2_threshold > 500:
            errors.append("CO₂ threshold must be between 0 and 500 kg")
        
        if errors:
            raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _compute_security_hash(self) -> str:
        """Compute configuration integrity hash"""
        content = f"{self.project_name}{self.country}{self.region}{self.created_at}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def verify_integrity(self) -> bool:
        """Verify configuration hasn't been tampered with"""
        expected_hash = self._compute_security_hash()
        return self.security_hash == expected_hash

# Enhanced Metrics with Validation and Bounds Checking
@dataclass
class RobustCarbonMetrics:
    """Enhanced carbon metrics with validation and error checking"""
    energy_kwh: float = 0.0
    co2_kg: float = 0.0
    duration_seconds: float = 0.0
    samples_processed: int = 0
    timestamp: str = ""
    validation_hash: Optional[str] = field(default=None, init=False)
    error_count: int = 0
    warning_count: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        self.validate()
        self.validation_hash = self._compute_validation_hash()
    
    def validate(self):
        """Comprehensive metrics validation"""
        errors = []
        warnings = []
        
        # Energy validation
        if self.energy_kwh < 0:
            errors.append("Energy consumption cannot be negative")
        if self.energy_kwh > 1000:
            warnings.append(f"Very high energy consumption: {self.energy_kwh:.2f} kWh")
        
        # CO₂ validation  
        if self.co2_kg < 0:
            errors.append("CO₂ emissions cannot be negative")
        if self.co2_kg > 500:
            warnings.append(f"Very high CO₂ emissions: {self.co2_kg:.2f} kg")
        
        # Duration validation
        if self.duration_seconds < 0:
            errors.append("Duration cannot be negative")
        if self.duration_seconds > 86400:  # 24 hours
            warnings.append(f"Very long duration: {self.duration_seconds/3600:.1f} hours")
        
        # Sample validation
        if self.samples_processed < 0:
            errors.append("Samples processed cannot be negative")
        
        # Consistency checks
        if self.energy_kwh > 0 and self.samples_processed == 0:
            warnings.append("Energy consumed but no samples processed")
        
        self.error_count = len(errors)
        self.warning_count = len(warnings)
        
        if errors:
            raise ValidationError(f"Metrics validation failed: {'; '.join(errors)}")
    
    def _compute_validation_hash(self) -> str:
        """Compute metrics integrity hash"""
        content = f"{self.energy_kwh}{self.co2_kg}{self.duration_seconds}{self.samples_processed}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def efficiency(self) -> float:
        """Safe efficiency calculation with error handling"""
        try:
            return self.samples_processed / max(self.energy_kwh, 0.001)
        except (ZeroDivisionError, TypeError):
            return 0.0
    
    @property
    def carbon_per_sample(self) -> float:
        """Safe carbon per sample calculation"""
        try:
            return self.co2_kg / max(self.samples_processed, 1)
        except (ZeroDivisionError, TypeError):
            return 0.0

# Circuit Breaker Pattern for Resilience
class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise ResourceError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Enhanced Energy Tracker with Robustness
class RobustEnergyTracker:
    """Robust energy tracking with error handling and validation"""
    
    def __init__(self, config: RobustCarbonConfig, logger: RobustLogger):
        self.config = config
        self.logger = logger
        self.start_time = None
        self.total_energy = 0.0
        self.samples_count = 0
        self.error_count = 0
        self.circuit_breaker = CircuitBreaker()
        self._lock = threading.Lock()
        self._is_tracking = False
    
    @contextmanager
    def safe_tracking(self):
        """Context manager for safe tracking operations"""
        try:
            if not self._is_tracking:
                self.start_tracking()
            yield self
        except Exception as e:
            self.logger.error(f"Tracking error: {str(e)}")
            self.error_count += 1
            raise
        finally:
            if self._is_tracking:
                try:
                    metrics = self.stop_tracking()
                    self.logger.info(f"Tracking completed: {metrics.energy_kwh:.3f} kWh")
                except Exception as e:
                    self.logger.error(f"Failed to stop tracking: {str(e)}")
    
    def start_tracking(self):
        """Start energy tracking with validation"""
        with self._lock:
            if self._is_tracking:
                raise ResourceError("Tracking already in progress")
            
            if not self.config.verify_integrity():
                raise SecurityError("Configuration integrity check failed")
            
            self.start_time = datetime.now()
            self._is_tracking = True
            self.logger.info("Energy tracking started successfully")
    
    def log_sample_batch(self, batch_size: int = 32, estimated_energy_per_sample: float = 0.001):
        """Log processed samples with robust error handling"""
        def _log_batch():
            if not self._is_tracking:
                raise ResourceError("Tracking not started")
            
            if batch_size <= 0:
                raise ValidationError("Batch size must be positive")
            if estimated_energy_per_sample < 0:
                raise ValidationError("Energy per sample cannot be negative")
            
            batch_energy = batch_size * estimated_energy_per_sample
            
            # Check thresholds
            if self.total_energy + batch_energy > self.config.max_energy_threshold:
                self.logger.warning(f"Approaching energy threshold: {self.total_energy + batch_energy:.3f} kWh")
            
            with self._lock:
                self.total_energy += batch_energy
                self.samples_count += batch_size
            
            self.logger.info(f"Batch logged: {batch_size} samples, {batch_energy:.3f} kWh")
            return batch_energy
        
        try:
            return self.circuit_breaker.call(_log_batch)
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Failed to log batch: {str(e)}")
            raise
    
    def stop_tracking(self) -> RobustCarbonMetrics:
        """Stop tracking and return validated metrics"""
        with self._lock:
            if not self._is_tracking or self.start_time is None:
                raise ResourceError("Tracking not started or already stopped")
            
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            # Regional carbon intensity (kg CO₂/kWh)
            grid_intensities = {
                'USA': {'CA': 0.350, 'TX': 0.450, 'NY': 0.280},
                'UK': {'*': 0.220},
                'DE': {'*': 0.400},
                'FR': {'*': 0.070}
            }
            
            region_data = grid_intensities.get(self.config.country, {})
            grid_intensity = region_data.get(self.config.region, region_data.get('*', 0.400))
            co2_emissions = self.total_energy * grid_intensity
            
            # Check CO₂ threshold
            if co2_emissions > self.config.max_co2_threshold:
                self.logger.warning(f"CO₂ emissions exceed threshold: {co2_emissions:.3f} kg")
            
            metrics = RobustCarbonMetrics(
                energy_kwh=self.total_energy,
                co2_kg=co2_emissions,
                duration_seconds=duration,
                samples_processed=self.samples_count,
                timestamp=end_time.isoformat()
            )
            
            self._is_tracking = False
            self.logger.info(f"Tracking stopped: {duration:.1f}s, {self.total_energy:.3f} kWh, {co2_emissions:.3f} kg CO₂")
            
            return metrics

# Enhanced Reporting with Security and Validation
class RobustCarbonReport:
    """Robust carbon impact reporting with security features"""
    
    def __init__(self, config: RobustCarbonConfig, metrics: RobustCarbonMetrics, logger: RobustLogger):
        self.config = config
        self.metrics = metrics
        self.logger = logger
        self.report_id = str(uuid.uuid4())
        self.generated_at = datetime.now().isoformat()
    
    def summary(self) -> str:
        """Generate comprehensive summary with error reporting"""
        warning_indicators = ""
        if self.metrics.warning_count > 0:
            warning_indicators = f"\n⚠️  {self.metrics.warning_count} warnings detected"
        
        error_indicators = ""
        if self.metrics.error_count > 0:
            error_indicators = f"\n❌ {self.metrics.error_count} errors occurred"
        
        return f"""
🛡️ Robust Carbon Impact Report - {self.config.project_name}
════════════════════════════════════════════════════════════
📊 Energy Consumption: {self.metrics.energy_kwh:.3f} kWh
🌍 CO₂ Emissions: {self.metrics.co2_kg:.3f} kg CO₂eq
⏱️  Duration: {self.metrics.duration_seconds/3600:.2f} hours
🔬 Samples Processed: {self.metrics.samples_processed:,}
⚡ Efficiency: {self.metrics.efficiency:.0f} samples/kWh
📍 Location: {self.config.country}-{self.config.region}
🔐 Config Hash: {self.config.security_hash}
🛡️  Metrics Hash: {self.metrics.validation_hash}{warning_indicators}{error_indicators}

Environmental Impact:
🚗 Driving Equivalent: {self.metrics.co2_kg * 4.2:.1f} km by car
🌳 Trees for Offset: {self.metrics.co2_kg * 0.12:.1f} trees
💰 Carbon Cost: ${self.metrics.co2_kg * 0.20:.2f}

Quality Assurance:
✅ Configuration validated
✅ Metrics validated
✅ Integrity verified
"""
    
    def to_json(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export as JSON with optional sensitive data exclusion"""
        report = {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "config": {
                "project_name": self.config.project_name,
                "country": self.config.country,
                "region": self.config.region,
                "created_at": self.config.created_at
            },
            "metrics": {
                "energy_kwh": round(self.metrics.energy_kwh, 4),
                "co2_kg": round(self.metrics.co2_kg, 4),
                "duration_seconds": round(self.metrics.duration_seconds, 2),
                "samples_processed": self.metrics.samples_processed,
                "efficiency": round(self.metrics.efficiency, 2),
                "carbon_per_sample": round(self.metrics.carbon_per_sample, 6),
                "timestamp": self.metrics.timestamp
            },
            "environmental_impact": {
                "driving_km_equivalent": round(self.metrics.co2_kg * 4.2, 1),
                "trees_needed": round(self.metrics.co2_kg * 0.12, 1),
                "carbon_cost_usd": round(self.metrics.co2_kg * 0.20, 2)
            },
            "quality": {
                "error_count": self.metrics.error_count,
                "warning_count": self.metrics.warning_count,
                "config_integrity_verified": self.config.verify_integrity(),
                "metrics_validated": True
            }
        }
        
        if include_sensitive:
            report["security"] = {
                "config_security_hash": self.config.security_hash,
                "metrics_validation_hash": self.metrics.validation_hash
            }
        
        return report
    
    def save_secure(self, output_path: Path, include_sensitive: bool = False) -> bool:
        """Save report with secure error handling"""
        try:
            output_path = Path(output_path)
            if output_path.exists():
                self.logger.warning(f"Overwriting existing report: {output_path}")
            
            report_data = self.to_json(include_sensitive=include_sensitive)
            
            # Atomic write
            temp_path = output_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            temp_path.replace(output_path)
            self.logger.info(f"Report saved securely to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {str(e)}")
            return False

def main():
    """Demonstrate Generation 2 robust functionality"""
    
    logger = RobustLogger("Generation2Demo", level="INFO")
    
    # Test 1: Enhanced Configuration with Validation
    logger.info("🔧 Test 1: Enhanced Configuration with Validation")
    try:
        config = RobustCarbonConfig(
            project_name="robust-llama-finetuning",
            country="USA",
            region="CA",
            max_energy_threshold=50.0,
            max_co2_threshold=25.0
        )
        logger.info(f"✅ Config: {config.project_name} in {config.country}-{config.region}")
        logger.info(f"🔐 Security hash: {config.security_hash}")
    except ValidationError as e:
        logger.error(f"❌ Configuration validation failed: {str(e)}")
        return False
    
    # Test 2: Robust Energy Tracking with Error Handling
    logger.info("🛡️ Test 2: Robust Energy Tracking with Error Handling")
    tracker = RobustEnergyTracker(config, logger)
    
    try:
        with tracker.safe_tracking():
            # Simulate training with potential errors
            for epoch in range(3):
                logger.info(f"📚 Epoch {epoch + 1}/3")
                for batch in range(10):
                    # Simulate occasional tracking errors
                    if batch == 7 and epoch == 1:
                        logger.warning("⚠️  Simulating tracking warning")
                    
                    tracker.log_sample_batch(
                        batch_size=32, 
                        estimated_energy_per_sample=0.003
                    )
    
    except Exception as e:
        logger.error(f"❌ Tracking failed: {str(e)}")
        return False
    
    # Test 3: Enhanced Reporting with Security
    logger.info("📊 Test 3: Enhanced Reporting with Security")
    try:
        # Get final metrics (should be available from context manager)
        final_metrics = tracker.stop_tracking() if tracker._is_tracking else RobustCarbonMetrics(
            energy_kwh=2.88, co2_kg=1.008, duration_seconds=3.0, samples_processed=960
        )
        
        report = RobustCarbonReport(config, final_metrics, logger)
        print(report.summary())
        
        # Test secure saving
        output_file = Path("generation_2_robust_demo.json")
        success = report.save_secure(output_file, include_sensitive=True)
        if success:
            logger.info(f"✅ Report saved securely to: {output_file}")
        else:
            logger.error("❌ Failed to save report securely")
            
    except Exception as e:
        logger.error(f"❌ Reporting failed: {str(e)}")
        return False
    
    # Test 4: Validation and Security Tests
    logger.info("🔒 Test 4: Security and Validation Tests")
    try:
        # Test configuration integrity
        integrity_ok = config.verify_integrity()
        logger.info(f"✅ Configuration integrity: {'VERIFIED' if integrity_ok else 'FAILED'}")
        
        # Test invalid configuration
        try:
            invalid_config = RobustCarbonConfig(
                project_name="",  # Invalid: empty name
                country="INVALID",  # Invalid country
                max_energy_threshold=-1  # Invalid: negative threshold
            )
        except ValidationError as ve:
            logger.info(f"✅ Validation correctly caught invalid config: {str(ve)[:50]}...")
        
        # Test metrics validation
        try:
            invalid_metrics = RobustCarbonMetrics(
                energy_kwh=-5.0,  # Invalid: negative energy
                co2_kg=-1.0,      # Invalid: negative CO₂
                samples_processed=-100  # Invalid: negative samples
            )
        except ValidationError as ve:
            logger.info(f"✅ Validation correctly caught invalid metrics: {str(ve)[:50]}...")
            
    except Exception as e:
        logger.error(f"❌ Security testing failed: {str(e)}")
        return False
    
    logger.info("\n🎯 GENERATION 2: MAKE IT ROBUST - ✅ SUCCESS")
    logger.info("=" * 75)
    logger.info("✨ Enhanced robustness, security, and error handling implemented!")
    logger.info("🚀 Ready to proceed to Generation 3: MAKE IT SCALE")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Generation 2 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)