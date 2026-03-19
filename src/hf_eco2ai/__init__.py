"""hf-eco2ai-plugin: Carbon emission tracking for HuggingFace Trainer runs.

Drop-in TrainerCallback that estimates GPU energy consumption and converts
it to CO2 emissions using regional grid carbon intensity data.

Quickstart::

    from hf_eco2ai import Eco2AICallback

    # With real HuggingFace Trainer:
    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[Eco2AICallback(region="eu-west-3", gpu_tdp_w=400)],
    )

    # Standalone / no-HF mode:
    cb = Eco2AICallback(region="us-west-2")
    cb.on_train_begin()
    for step in range(100):
        cb.on_step_end(step=step)
    report = cb.on_train_end()
    print(report)
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

from .carbon_tracker import CarbonTracker, EmissionsReport, StepSnapshot
from .callback import Eco2AICallback
from .regional_grid import RegionalGrid, GridRegion

__all__ = [
    "CarbonTracker",
    "EmissionsReport",
    "StepSnapshot",
    "Eco2AICallback",
    "RegionalGrid",
    "GridRegion",
]
