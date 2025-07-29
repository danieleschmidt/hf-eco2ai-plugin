"""HF Eco2AI Plugin: Carbon tracking for Hugging Face Transformers training."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .callback import Eco2AICallback
from .config import CarbonConfig

__all__ = ["Eco2AICallback", "CarbonConfig"]