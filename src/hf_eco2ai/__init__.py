"""HF Eco2AI Plugin: Carbon tracking for Hugging Face Transformers training."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .callback import Eco2AICallback, CarbonBudgetCallback
from .config import CarbonConfig
from .models import CarbonMetrics, CarbonReport, CarbonSummary
from .monitoring import EnergyTracker, GPUMonitor, CarbonIntensityProvider
from .exporters import PrometheusExporter, ReportExporter

__all__ = [
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
    "ReportExporter"
]