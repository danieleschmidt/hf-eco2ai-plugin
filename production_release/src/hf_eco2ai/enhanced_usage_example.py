"""
Enhanced HF Eco2AI Usage Examples
Demonstrates how to use the enhanced carbon tracking system with robust security,
monitoring, fault tolerance, error handling, and compliance features.
"""

import os
import time
from datetime import datetime
from hf_eco2ai import (
    # Enhanced Generation 2 components
    initialize_enhanced_system,
    create_enhanced_callback,
    get_integration_manager,
    ComplianceLevel,
    AuditEventType,
    
    # Core components for comparison
    CarbonConfig,
    Eco2AICallback
)


def basic_usage_example():
    """Example of basic enhanced system usage"""
    print("=== Basic Enhanced System Usage ===")
    
    # Initialize the enhanced system with all security, monitoring, and compliance features
    try:
        manager = initialize_enhanced_system()
        print("✓ Enhanced system initialized successfully")
        
        # Get system status
        status = manager.get_system_status()
        print(f"✓ System mode: {status.mode.value}")
        print(f"✓ Active components: {len(status.components)}")
        print(f"✓ System health: {status.risk_level}")
        
        # Create enhanced callback for ML training
        callback = create_enhanced_callback(
            project_name="carbon_aware_training",
            experiment_description="Enhanced carbon tracking with full monitoring"
        )
        print("✓ Enhanced callback created")
        
        # Simulate some training activity
        print("Simulating training activity...")
        callback.on_train_begin({"epochs": 5, "batch_size": 32})
        
        for epoch in range(3):
            time.sleep(1)  # Simulate epoch processing
            callback.on_epoch_end(epoch, {"loss": 0.5 - epoch * 0.1})
            
        callback.on_train_end({"final_loss": 0.2})
        print("✓ Training simulation completed")
        
        # Check compliance status
        compliance_status = manager.compliance_framework.get_compliance_status(ComplianceLevel.GDPR)
        print(f"✓ GDPR compliance: {compliance_status.get('compliance_percentage', 0):.1f}%")
        
        # Get final system status
        final_status = manager.get_system_status()
        print(f"✓ Final system status: {len(final_status.active_alerts)} alerts")
        
        # Graceful shutdown
        manager.shutdown()
        print("✓ System shutdown completed")
        
    except Exception as e:
        print(f"✗ Error in basic usage: {e}")


def advanced_security_example():
    """Example demonstrating advanced security features"""
    print("\n=== Advanced Security Features ===")
    
    try:
        manager = get_integration_manager()
        if not manager._initialization_complete:
            manager.initialize_components()
            
        # Access security manager
        security = manager.security_manager
        
        # Demonstrate RBAC
        print("Testing Role-Based Access Control...")
        
        # Create a user session (in real usage, this would be from authentication)
        user_session = security.rbac_manager.create_session("test_user", "USER")
        if user_session:
            print(f"✓ User session created: {user_session['session_id'][:8]}...")
            
            # Test permission check
            can_access = security.rbac_manager.check_permission(
                user_session['session_id'], 
                "carbon_data", 
                "read"
            )
            print(f"✓ Permission check - can read carbon data: {can_access}")
            
            # Test data export with security validation
            export_data = {
                "carbon_emissions": [1.2, 1.5, 0.8],
                "energy_consumption": [100, 120, 85],
                "model_name": "test_model"
            }
            
            is_safe = security.data_validator.validate_export_data(export_data)
            print(f"✓ Export data validation: {'Safe' if is_safe else 'Blocked'}")
            
            # Clean up session
            security.rbac_manager.invalidate_session(user_session['session_id'])
            print("✓ Session cleanup completed")
        else:
            print("✗ Failed to create user session")
            
    except Exception as e:
        print(f"✗ Error in security example: {e}")


def health_monitoring_example():
    """Example demonstrating health monitoring capabilities"""
    print("\n=== Health Monitoring Features ===")
    
    try:
        manager = get_integration_manager()
        health_monitor = manager.health_monitor
        
        if health_monitor:
            # Get current system health
            health_data = health_monitor.get_system_health()
            print(f"✓ CPU Usage: {health_data.get('cpu_usage', 0):.1f}%")
            print(f"✓ Memory Usage: {health_data.get('memory_usage', 0):.1f}%")
            print(f"✓ Disk Usage: {health_data.get('disk_usage', 0):.1f}%")
            print(f"✓ Overall Health Score: {health_data.get('overall_health', 0):.2f}")
            
            # Test memory leak detection
            leak_status = health_monitor.memory_leak_detector.check_memory_trend()
            print(f"✓ Memory Leak Status: {leak_status.get('status', 'unknown')}")
            
            # Test network health
            network_health = health_monitor.network_checker.check_connectivity([
                "google.com", "github.com"
            ])
            print(f"✓ Network Health: {len(network_health.get('reachable', []))} endpoints reachable")
            
            # Test predictive analytics
            prediction = health_monitor.predictive_analyzer.predict_failure_probability()
            print(f"✓ Failure Prediction: {prediction.get('probability', 0):.2%} risk")
            
        else:
            print("✗ Health monitor not available")
            
    except Exception as e:
        print(f"✗ Error in health monitoring example: {e}")


def fault_tolerance_example():
    """Example demonstrating fault tolerance features"""
    print("\n=== Fault Tolerance Features ===")
    
    try:
        manager = get_integration_manager()
        fault_tolerance = manager.fault_tolerance
        
        if fault_tolerance:
            # Test backup functionality
            print("Testing backup functionality...")
            backup_result = fault_tolerance.backup_manager.create_backup({
                "test_data": "sample_carbon_metrics",
                "timestamp": datetime.now().isoformat()
            })
            if backup_result.get("success"):
                print(f"✓ Backup created: {backup_result.get('backup_id', 'unknown')[:8]}...")
            else:
                print("✗ Backup creation failed")
                
            # Test circuit breaker
            print("Testing circuit breaker...")
            breaker = fault_tolerance.get_circuit_breaker("test_service")
            
            # Simulate some successful calls
            for i in range(3):
                with breaker:
                    # Simulate successful operation
                    time.sleep(0.1)
                    print(f"✓ Circuit breaker call {i+1} succeeded")
                    
            print(f"✓ Circuit breaker state: {breaker.current_state}")
            
            # Test graceful degradation
            print("Testing graceful degradation...")
            fault_tolerance.enable_graceful_degradation()
            print("✓ Graceful degradation mode enabled")
            
        else:
            print("✗ Fault tolerance manager not available")
            
    except Exception as e:
        print(f"✗ Error in fault tolerance example: {e}")


def compliance_audit_example():
    """Example demonstrating compliance and audit features"""
    print("\n=== Compliance & Audit Features ===")
    
    try:
        manager = get_integration_manager()
        compliance = manager.compliance_framework
        
        if compliance:
            # Log various audit events
            events = [
                {
                    "event_type": AuditEventType.DATA_ACCESS,
                    "action": "view_carbon_report",
                    "resource": "monthly_emissions",
                    "details": {"report_type": "summary", "date_range": "2024-01"}
                },
                {
                    "event_type": AuditEventType.DATA_EXPORT,
                    "action": "export_metrics", 
                    "resource": "carbon_metrics",
                    "details": {"format": "json", "records": 150}
                },
                {
                    "event_type": AuditEventType.SYSTEM_CONFIGURATION,
                    "action": "update_thresholds",
                    "resource": "alert_config", 
                    "details": {"cpu_threshold": 80, "memory_threshold": 85}
                }
            ]
            
            for event_data in events:
                success = compliance.log_audit_event(
                    event_type=event_data["event_type"],
                    user_id="demo_user",
                    resource=event_data["resource"],
                    action=event_data["action"],
                    details=event_data["details"],
                    compliance_level=ComplianceLevel.GDPR,
                    session_id="demo_session",
                    ip_address="127.0.0.1",
                    user_agent="demo_script",
                    risk_level="low",
                    data_classification="operational"
                )
                if success:
                    print(f"✓ Audit event logged: {event_data['action']}")
                else:
                    print(f"✗ Failed to log: {event_data['action']}")
                    
            # Generate compliance report
            print("Generating compliance report...")
            report = compliance.report_generator.generate_compliance_report(
                ComplianceLevel.GDPR, period_days=7
            )
            print(f"✓ Compliance report generated: {report.report_id[:8]}...")
            print(f"  - Total checks: {report.total_checks}")
            print(f"  - Compliant: {report.compliant_checks}")
            print(f"  - Risk score: {report.risk_score:.1f}%")
            
            # Verify audit integrity
            print("Verifying audit integrity...")
            integrity = compliance.audit_logger.verify_integrity()
            print(f"✓ Audit integrity: {integrity.get('integrity_score', 0):.2%}")
            
        else:
            print("✗ Compliance framework not available")
            
    except Exception as e:
        print(f"✗ Error in compliance example: {e}")


def integration_workflow_example():
    """Example demonstrating complete integrated workflow"""
    print("\n=== Complete Integration Workflow ===")
    
    try:
        # Custom configuration
        config = CarbonConfig(
            project_name="enhanced_ml_project",
            experiment_description="Full-featured carbon tracking with compliance",
            country_iso_code="USA",
            region="california"
        )
        
        # Initialize with custom config
        manager = initialize_enhanced_system(config)
        print("✓ Enhanced system initialized with custom config")
        
        # Register custom event handlers
        def custom_security_handler(threat_info):
            print(f"🚨 Security Alert: {threat_info.get('message', 'Unknown threat')}")
            
        def custom_health_handler(alert_info):
            print(f"⚕️ Health Alert: {alert_info.get('message', 'Unknown health issue')}")
            
        manager.register_event_handler('security_threat', custom_security_handler)
        manager.register_event_handler('health_alert', custom_health_handler)
        print("✓ Custom event handlers registered")
        
        # Create callback with comprehensive monitoring
        callback = manager.create_enhanced_callback(
            project_name=config.project_name,
            experiment_description=config.experiment_description
        )
        print("✓ Enhanced callback created with full monitoring")
        
        # Simulate a complete ML training workflow
        print("Starting ML training workflow simulation...")
        
        # Training start
        callback.on_train_begin({
            "model_type": "transformer",
            "dataset": "climate_data",
            "epochs": 10,
            "batch_size": 64
        })
        
        # Training epochs with health monitoring
        for epoch in range(5):
            # Simulate epoch processing
            time.sleep(0.5)
            
            # Epoch end with metrics
            callback.on_epoch_end(epoch, {
                "loss": 1.0 - epoch * 0.15,
                "accuracy": 0.6 + epoch * 0.08,
                "carbon_emissions": 0.5 + epoch * 0.1
            })
            
            # Check system health every few epochs
            if epoch % 2 == 0:
                status = manager.get_system_status()
                print(f"  Epoch {epoch}: System health check - {len(status.active_alerts)} alerts")
                
        # Training completion
        callback.on_train_end({
            "final_loss": 0.25,
            "final_accuracy": 0.92,
            "total_carbon": 2.5,
            "training_duration": 300
        })
        print("✓ Training workflow completed")
        
        # Generate comprehensive compliance report
        print("Generating final compliance report...")
        comprehensive_report = manager.compliance_framework.generate_comprehensive_report(7)
        compliant_frameworks = sum(1 for report in comprehensive_report["reports"].values() 
                                 if not isinstance(report, dict) or "error" not in report)
        print(f"✓ Comprehensive report: {compliant_frameworks} frameworks compliant")
        
        # Final system status
        final_status = manager.get_system_status()
        print(f"✓ Final status: {final_status.mode.value} mode, {final_status.risk_level} risk")
        
        # Graceful shutdown
        manager.shutdown()
        print("✓ Complete workflow finished successfully")
        
    except Exception as e:
        print(f"✗ Error in integration workflow: {e}")


def main():
    """Run all examples"""
    print("HF Eco2AI Enhanced System Examples")
    print("=" * 50)
    
    # Run examples in sequence
    basic_usage_example()
    advanced_security_example()
    health_monitoring_example()
    fault_tolerance_example()
    compliance_audit_example()
    integration_workflow_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()