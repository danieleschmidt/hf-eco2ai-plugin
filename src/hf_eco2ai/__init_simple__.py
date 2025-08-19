"""HF Eco2AI Plugin: Carbon tracking for Hugging Face Transformers training - Simple Version."""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Core components - Generation 1 Simple Implementation
from .callback import Eco2AICallback, CarbonBudgetCallback
from .config import CarbonConfig
from .models import CarbonMetrics, CarbonReport, CarbonSummary
from .monitoring import EnergyTracker, GPUMonitor, CarbonIntensityProvider
from .exporters import PrometheusExporter, ReportExporter

__all__ = [
    # Core Generation 1 components
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
]