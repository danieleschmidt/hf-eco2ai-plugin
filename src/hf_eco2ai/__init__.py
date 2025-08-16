"""HF Eco2AI Plugin: Carbon tracking for Hugging Face Transformers training."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .callback import Eco2AICallback, CarbonBudgetCallback
from .config import CarbonConfig
from .models import CarbonMetrics, CarbonReport, CarbonSummary
from .monitoring import EnergyTracker, GPUMonitor, CarbonIntensityProvider
from .exporters import PrometheusExporter, ReportExporter

# Enhanced Generation 2 components for robust production deployments
from .enhanced_integration import (
    EnhancedIntegrationManager, 
    EnhancedEco2AICallback,
    get_integration_manager,
    initialize_enhanced_system,
    create_enhanced_callback
)
from .security_enhanced import EnhancedSecurityManager
from .health_monitor_enhanced import EnhancedHealthMonitor
from .fault_tolerance_enhanced import EnhancedFaultToleranceManager
from .error_handling_enhanced import EnhancedErrorHandler
from .compliance import ComplianceFramework, AuditEventType, ComplianceLevel

# Generation 3 Enterprise-Scale Components for massive deployment
from .quantum_performance_engine import (
    QuantumPerformanceEngine,
    PerformanceTier,
    QuantumPerformanceMetrics,
    GPUTensorCoreConfig,
    get_quantum_engine,
    initialize_quantum_performance
)
from .enterprise_autoscaling import (
    EnterpriseAutoScaler,
    ScalingPolicy,
    LoadLevel,
    ScalingDirection,
    GeographicRegion,
    TrainingLoadPredictor,
    LoadBalancer,
    get_enterprise_autoscaler,
    initialize_enterprise_autoscaling
)
from .advanced_caching import (
    InMemoryCache,
    RedisDistributedCache,
    HierarchicalStorageManager,
    TimeSeriesOptimizer,
    CacheLevel,
    CompressionAlgorithm,
    EvictionPolicy,
    get_memory_cache,
    get_distributed_cache,
    get_hierarchical_storage,
    initialize_advanced_caching
)
from .distributed_processing_engine import (
    DistributedProcessingEngine,
    KafkaEventBus,
    DaskDistributedProcessor,
    RayDistributedProcessor,
    KubernetesOrchestrator,
    ProcessingEngine,
    MessagePattern,
    ServiceType,
    Microservice,
    get_distributed_engine,
    initialize_distributed_processing
)
from .ai_optimization_engine import (
    AIOptimizationEngine,
    CarbonPredictionModel,
    AnomalyDetectionSystem,
    ReinforcementLearningOptimizer,
    HyperparameterTuner,
    FederatedLearningCoordinator,
    OptimizationAlgorithm,
    AnomalyDetectionMethod,
    HyperparameterOptimizer,
    PredictionResult,
    AnomalyResult,
    OptimizationRecommendation,
    get_ai_optimization_engine,
    initialize_ai_optimization
)

__all__ = [
    # Core components
    "Eco2AICallback", 
    "CarbonBudgetCallback",
    "CarbonConfig",
    "CarbonMetrics",
    "CarbonReport", 
    "CarbonSummary",
    "EnergyTracker",
    "GPUMonitor",
    "CarbonIntensityProvider",
    "PrometheusExporter",
    "ReportExporter",
    
    # Enhanced Generation 2 components
    "EnhancedIntegrationManager",
    "EnhancedEco2AICallback", 
    "get_integration_manager",
    "initialize_enhanced_system",
    "create_enhanced_callback",
    "EnhancedSecurityManager",
    "EnhancedHealthMonitor", 
    "EnhancedFaultToleranceManager",
    "EnhancedErrorHandler",
    "ComplianceFramework",
    "AuditEventType",
    "ComplianceLevel",
    
    # Generation 3 Enterprise-Scale Components
    # Quantum Performance Engine
    "QuantumPerformanceEngine",
    "PerformanceTier",
    "QuantumPerformanceMetrics", 
    "GPUTensorCoreConfig",
    "get_quantum_engine",
    "initialize_quantum_performance",
    
    # Enterprise Auto-Scaling
    "EnterpriseAutoScaler",
    "ScalingPolicy",
    "LoadLevel", 
    "ScalingDirection",
    "GeographicRegion",
    "TrainingLoadPredictor",
    "LoadBalancer",
    "get_enterprise_autoscaler",
    "initialize_enterprise_autoscaling",
    
    # Advanced Caching & Storage
    "InMemoryCache",
    "RedisDistributedCache",
    "HierarchicalStorageManager",
    "TimeSeriesOptimizer",
    "CacheLevel",
    "CompressionAlgorithm",
    "EvictionPolicy",
    "get_memory_cache",
    "get_distributed_cache", 
    "get_hierarchical_storage",
    "initialize_advanced_caching",
    
    # Distributed Processing Engine
    "DistributedProcessingEngine",
    "KafkaEventBus",
    "DaskDistributedProcessor",
    "RayDistributedProcessor", 
    "KubernetesOrchestrator",
    "ProcessingEngine",
    "MessagePattern",
    "ServiceType",
    "Microservice",
    "get_distributed_engine",
    "initialize_distributed_processing",
    
    # AI Optimization Engine
    "AIOptimizationEngine",
    "CarbonPredictionModel",
    "AnomalyDetectionSystem",
    "ReinforcementLearningOptimizer",
    "HyperparameterTuner",
    "FederatedLearningCoordinator",
    "OptimizationAlgorithm",
    "AnomalyDetectionMethod", 
    "HyperparameterOptimizer",
    "PredictionResult",
    "AnomalyResult",
    "OptimizationRecommendation",
    "get_ai_optimization_engine",
    "initialize_ai_optimization"
]