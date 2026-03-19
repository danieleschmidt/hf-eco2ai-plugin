"""Eco2AICallback: HuggingFace Trainer-compatible carbon tracking callback.

Stubs the TrainerCallback interface so no actual transformers install
is required. When transformers IS installed, this is a drop-in callback.

Usage with real HF Trainer::

    from hf_eco2ai import Eco2AICallback
    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[Eco2AICallback(region="eu-west-3", gpu_tdp_w=400)],
    )

Usage in standalone / test mode::

    from hf_eco2ai import Eco2AICallback
    cb = Eco2AICallback(region="us-west-2", gpu_tdp_w=250)
    cb.on_train_begin()
    for step in range(100):
        cb.on_step_end(step=step)
    report = cb.on_train_end()
    print(report)
"""

from __future__ import annotations

import logging
from typing import Optional

from .carbon_tracker import CarbonTracker, EmissionsReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal TrainerCallback stub — replaced by the real class when transformers
# is installed. The stub preserves the same public method signature so any
# code that isinstance-checks for TrainerCallback still works.
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback as _BaseCallback  # type: ignore
    _HF_AVAILABLE = True
except ImportError:
    class _BaseCallback:  # type: ignore
        """Minimal TrainerCallback stub (transformers not installed)."""

        def on_train_begin(self, args=None, state=None, control=None, **kwargs):
            ...

        def on_step_end(self, args=None, state=None, control=None, **kwargs):
            ...

        def on_train_end(self, args=None, state=None, control=None, **kwargs):
            ...

        def on_epoch_end(self, args=None, state=None, control=None, **kwargs):
            ...

    _HF_AVAILABLE = False


class Eco2AICallback(_BaseCallback):
    """HuggingFace TrainerCallback that tracks carbon emissions during training.

    Hooks into three lifecycle events:

    * ``on_train_begin`` — starts the CarbonTracker.
    * ``on_step_end``   — records per-step energy (pass ``step=<int>`` kwarg
                          when calling manually).
    * ``on_train_end``  — stops tracking and returns the EmissionsReport.

    Args:
        region:        Cloud/grid region for carbon intensity lookup.
                       Accepts AWS (``us-east-1``), GCP (``us-central1``),
                       Azure (``eastus``), or country codes (``DE``, ``FR``).
        gpu_tdp_w:     GPU thermal design power in watts.  Used as a
                       conservative estimate when live power readings are
                       unavailable.  Default: 300 W.
        utilization:   Assumed GPU utilization fraction (0–1). Default: 0.8.
        capture_steps: If True, record a StepSnapshot for every step.
                       Useful for plotting; uses more memory. Default: False.
        power_override_w: Override the TDP estimate with a live power reading
                       (e.g. from pynvml). Useful for production usage.
        log_every_n_steps: Print a brief progress line every N steps (0 = off).
    """

    def __init__(
        self,
        region: str = "us-east-1",
        gpu_tdp_w: float = 300.0,
        utilization: float = 0.8,
        capture_steps: bool = False,
        power_override_w: Optional[float] = None,
        log_every_n_steps: int = 0,
    ) -> None:
        self._tracker = CarbonTracker(
            gpu_tdp_w=gpu_tdp_w,
            region=region,
            utilization=utilization,
            capture_steps=capture_steps,
            power_override_w=power_override_w,
        )
        self._log_every_n_steps = log_every_n_steps
        self._report: Optional[EmissionsReport] = None
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # HuggingFace Trainer lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, args=None, state=None, control=None, **kwargs) -> None:
        """Called at the start of training — begins energy tracking."""
        self._tracker.start()
        self._step_counter = 0
        logger.info(
            "Eco2AICallback: tracking started "
            f"[region={self._tracker.region}, "
            f"TDP={self._tracker.gpu_tdp_w:.0f}W, "
            f"intensity={self._tracker._intensity_g_per_kwh:.0f} gCO₂/kWh]"
        )

    def on_step_end(self, args=None, state=None, control=None, step: int = -1, **kwargs) -> None:
        """Called at the end of each training step — records incremental energy.

        When invoked by the real HF Trainer, `state.global_step` is used
        automatically.  For standalone use, pass ``step=<int>``.
        """
        # Prefer state.global_step when available (real HF Trainer context)
        effective_step = step
        if state is not None and hasattr(state, "global_step"):
            effective_step = state.global_step

        self._tracker.step(effective_step)
        self._step_counter += 1

        if self._log_every_n_steps > 0 and self._step_counter % self._log_every_n_steps == 0:
            logger.info(
                f"  step {self._step_counter:>6} | "
                f"energy={self._tracker.current_energy_kwh:.4f} kWh | "
                f"CO₂={self._tracker.current_co2_kg * 1000:.2f} g"
            )

    def on_epoch_end(self, args=None, state=None, control=None, **kwargs) -> None:
        """Called at epoch end — logs a progress line."""
        epoch = getattr(state, "epoch", "?") if state is not None else "?"
        logger.info(
            f"Eco2AICallback epoch {epoch} | "
            f"energy={self._tracker.current_energy_kwh:.4f} kWh | "
            f"CO₂={self._tracker.current_co2_kg * 1000:.3f} g"
        )

    def on_train_end(self, args=None, state=None, control=None, **kwargs) -> EmissionsReport:
        """Called at the end of training — stops tracking and returns the report.

        Returns:
            EmissionsReport with full training emissions data.
        """
        self._report = self._tracker.stop()
        self._print_report(self._report)
        return self._report

    # ------------------------------------------------------------------
    # Convenience access
    # ------------------------------------------------------------------

    @property
    def report(self) -> Optional[EmissionsReport]:
        """The final EmissionsReport (available after on_train_end)."""
        return self._report

    @property
    def current_co2_kg(self) -> float:
        """Current cumulative CO2 in kg (usable mid-training)."""
        return self._tracker.current_co2_kg

    @property
    def current_energy_kwh(self) -> float:
        """Current cumulative energy in kWh (usable mid-training)."""
        return self._tracker.current_energy_kwh

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_report(self, report: EmissionsReport) -> None:
        lines = [
            "",
            "=" * 52,
            "  🌱  Eco2AI Emissions Report",
            "=" * 52,
        ]
        lines.extend(report.summary_lines())
        lines.append("=" * 52)
        print("\n".join(lines))
