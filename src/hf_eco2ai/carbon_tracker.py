"""CarbonTracker: energy and CO2 estimation during ML training.

Estimates GPU power draw from a configurable TDP, tracks wall time,
and converts energy consumption to kg CO2 using the regional carbon
intensity from RegionalGrid.

No NVML / actual GPU telemetry required — works in CPU-only and
restricted environments. Drop in real power readings where available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .regional_grid import RegionalGrid


# Car emission factor: average EU passenger car ≈ 0.21 kg CO2/km (ICCT 2021)
_CAR_KG_CO2_PER_KM = 0.21


@dataclass
class StepSnapshot:
    """Energy snapshot captured at a single training step."""
    step: int
    timestamp: float
    elapsed_s: float
    energy_kwh: float          # cumulative
    co2_kg: float              # cumulative
    power_w: float             # instantaneous estimate


@dataclass
class EmissionsReport:
    """Structured emissions report for a completed training run.

    Attributes:
        total_energy_kwh:   Total electrical energy consumed.
        total_co2_kg:       Total CO2-equivalent emissions.
        equivalent_car_km:  Driving distance with equivalent emissions.
        total_steps:        Number of training steps completed.
        emissions_per_step: Average kg CO2 per training step.
        duration_hours:     Wall-clock training duration in hours.
        region:             Cloud/grid region used for intensity lookup.
        carbon_intensity_g_per_kwh: Grid intensity used in calculation.
        gpu_tdp_w:          Configured GPU TDP (watts) used for estimation.
        snapshots:          Per-step snapshots (if capture_steps=True).
    """
    total_energy_kwh: float = 0.0
    total_co2_kg: float = 0.0
    equivalent_car_km: float = 0.0
    total_steps: int = 0
    emissions_per_step: float = 0.0
    duration_hours: float = 0.0
    region: str = "us-east-1"
    carbon_intensity_g_per_kwh: float = 0.0
    gpu_tdp_w: float = 300.0
    snapshots: list = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        """Return a list of human-readable summary lines."""
        return [
            f"  Region:            {self.region}",
            f"  Grid intensity:    {self.carbon_intensity_g_per_kwh:.0f} gCO₂/kWh",
            f"  GPU TDP assumed:   {self.gpu_tdp_w:.0f} W",
            f"  Duration:          {self.duration_hours:.3f} h",
            f"  Training steps:    {self.total_steps:,}",
            f"  Energy consumed:   {self.total_energy_kwh:.4f} kWh",
            f"  CO₂ emitted:       {self.total_co2_kg * 1000:.2f} gCO₂  ({self.total_co2_kg:.6f} kg)",
            f"  Per-step CO₂:      {self.emissions_per_step * 1e6:.2f} µgCO₂/step",
            f"  Equiv. car travel: {self.equivalent_car_km:.4f} km",
        ]

    def __str__(self) -> str:
        lines = ["EmissionsReport:"] + self.summary_lines()
        return "\n".join(lines)


class CarbonTracker:
    """Tracks energy and CO2 emissions during ML training.

    Usage::

        tracker = CarbonTracker(gpu_tdp_w=350, region="eu-west-3")
        tracker.start()
        for step in range(num_steps):
            # ... training logic ...
            tracker.step(step)
        report = tracker.stop()
        print(report)

    Args:
        gpu_tdp_w:       GPU thermal design power in watts. Used as a
                         conservative upper-bound estimate when live power
                         readings are unavailable. Default: 300 W (A100-class).
        region:          Cloud or grid region ID for carbon intensity lookup.
                         Accepts AWS (us-east-1), GCP (us-central1), Azure
                         (eastus), or ISO country codes (DE, FR …).
                         Default: "us-east-1".
        grid:            Optional custom RegionalGrid instance.
        utilization:     Assumed GPU utilization fraction (0-1). Scales TDP
                         estimate. Default: 0.8 (80% utilization during training).
        capture_steps:   If True, record a StepSnapshot at every .step() call.
                         Useful for plotting but uses more memory. Default: False.
        power_override_w: If provided, use this fixed power reading (watts)
                         instead of the TDP-based estimate. Useful when live
                         NVML/ROCM readings are available.
    """

    def __init__(
        self,
        gpu_tdp_w: float = 300.0,
        region: str = "us-east-1",
        grid: Optional[RegionalGrid] = None,
        utilization: float = 0.8,
        capture_steps: bool = False,
        power_override_w: Optional[float] = None,
    ) -> None:
        self.gpu_tdp_w = gpu_tdp_w
        self.region = region
        self.grid = grid or RegionalGrid()
        self.utilization = max(0.0, min(1.0, utilization))
        self.capture_steps = capture_steps
        self.power_override_w = power_override_w

        grid_region = self.grid.get_region(region)
        self._intensity_g_per_kwh: float = grid_region.carbon_intensity_g_per_kwh

        self._start_time: Optional[float] = None
        self._last_step_time: Optional[float] = None
        self._step_count: int = 0
        self._cumulative_energy_kwh: float = 0.0
        self._snapshots: list[StepSnapshot] = []
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin tracking. Call once before the training loop."""
        if self._running:
            raise RuntimeError("CarbonTracker.start() called while already running")
        self._start_time = time.time()
        self._last_step_time = self._start_time
        self._step_count = 0
        self._cumulative_energy_kwh = 0.0
        self._snapshots = []
        self._running = True

    def step(self, step_index: int = -1) -> float:
        """Record energy consumed since the last step call.

        Returns the incremental energy consumed in kWh for this step.
        """
        if not self._running:
            raise RuntimeError("CarbonTracker.step() called before start()")

        now = time.time()
        elapsed_since_last = now - self._last_step_time  # seconds
        self._last_step_time = now
        self._step_count += 1

        power_w = self._effective_power_w()
        step_energy_kwh = (power_w * elapsed_since_last) / 3_600_000  # W·s → kWh
        self._cumulative_energy_kwh += step_energy_kwh

        if self.capture_steps:
            self._snapshots.append(
                StepSnapshot(
                    step=step_index if step_index >= 0 else self._step_count,
                    timestamp=now,
                    elapsed_s=now - self._start_time,
                    energy_kwh=self._cumulative_energy_kwh,
                    co2_kg=self._energy_to_co2(self._cumulative_energy_kwh),
                    power_w=power_w,
                )
            )

        return step_energy_kwh

    def stop(self) -> EmissionsReport:
        """Stop tracking and return the final EmissionsReport."""
        if not self._running:
            raise RuntimeError("CarbonTracker.stop() called before start()")

        now = time.time()
        self._running = False
        duration_s = now - self._start_time
        total_co2_kg = self._energy_to_co2(self._cumulative_energy_kwh)

        report = EmissionsReport(
            total_energy_kwh=self._cumulative_energy_kwh,
            total_co2_kg=total_co2_kg,
            equivalent_car_km=total_co2_kg / _CAR_KG_CO2_PER_KM,
            total_steps=self._step_count,
            emissions_per_step=total_co2_kg / max(1, self._step_count),
            duration_hours=duration_s / 3600,
            region=self.region,
            carbon_intensity_g_per_kwh=self._intensity_g_per_kwh,
            gpu_tdp_w=self.gpu_tdp_w,
            snapshots=list(self._snapshots),
        )
        return report

    # ------------------------------------------------------------------
    # State access (usable while running)
    # ------------------------------------------------------------------

    @property
    def current_energy_kwh(self) -> float:
        """Cumulative energy consumed so far in kWh."""
        return self._cumulative_energy_kwh

    @property
    def current_co2_kg(self) -> float:
        """Cumulative CO2 emitted so far in kg."""
        return self._energy_to_co2(self._cumulative_energy_kwh)

    @property
    def current_power_w(self) -> float:
        """Effective power draw in watts used for estimation."""
        return self._effective_power_w()

    @property
    def step_count(self) -> int:
        """Number of steps recorded so far."""
        return self._step_count

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_power_w(self) -> float:
        """Return the power (W) used for incremental energy calculation."""
        if self.power_override_w is not None:
            return self.power_override_w
        return self.gpu_tdp_w * self.utilization

    def _energy_to_co2(self, energy_kwh: float) -> float:
        """Convert kWh to kg CO2 using the regional intensity."""
        # intensity is in gCO2/kWh → divide by 1000 for kg
        return energy_kwh * self._intensity_g_per_kwh / 1000.0
