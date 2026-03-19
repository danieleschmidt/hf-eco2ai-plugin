#!/usr/bin/env python3
"""Demo: simulate a 100-step training run and generate an emissions report.

Runs without any HuggingFace or GPU dependency.
"""

import time
import sys
import os

# Add src to path so we can import without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hf_eco2ai import Eco2AICallback, CarbonTracker, RegionalGrid

REGIONS_TO_COMPARE = [
    ("us-east-1",   "N. Virginia  (US)"),
    ("eu-west-3",   "Paris        (FR)"),
    ("eu-north-1",  "Stockholm    (SE)"),
    ("ap-south-1",  "Mumbai       (IN)"),
    ("us-west-2",   "Oregon       (US)"),
]


def run_demo_callback():
    """Show Eco2AICallback in standalone (no-HF) mode."""
    print("\n" + "=" * 60)
    print("  Demo 1: Eco2AICallback — 100 simulated training steps")
    print("=" * 60)

    # A100 80GB in us-east-1 at 80% utilisation
    cb = Eco2AICallback(
        region="us-east-1",
        gpu_tdp_w=400,
        utilization=0.8,
        capture_steps=False,
        log_every_n_steps=25,
    )

    cb.on_train_begin()

    for step in range(100):
        # Simulate ~20ms per training step
        time.sleep(0.02)
        cb.on_step_end(step=step)

    report = cb.on_train_end()

    print(f"\n  Returned EmissionsReport object:")
    print(f"    total_energy_kwh  = {report.total_energy_kwh:.6f}")
    print(f"    total_co2_kg      = {report.total_co2_kg:.8f}")
    print(f"    equivalent_car_km = {report.equivalent_car_km:.6f}")
    print(f"    emissions_per_step= {report.emissions_per_step:.2e} kg/step")


def run_region_comparison():
    """Compare CO2 cost of the same training run across grid regions."""
    print("\n" + "=" * 60)
    print("  Demo 2: Regional CO₂ comparison — same 100-step run")
    print("=" * 60)

    grid = RegionalGrid()
    STEPS = 100
    STEP_DURATION_S = 0.02  # 20ms per step
    GPU_TDP_W = 400
    UTIL = 0.8

    total_duration_h = (STEPS * STEP_DURATION_S) / 3600
    power_w = GPU_TDP_W * UTIL
    energy_kwh = power_w * total_duration_h / 1000  # W * h / 1000 = kWh

    print(f"\n  Workload: {STEPS} steps × {STEP_DURATION_S*1000:.0f}ms | {GPU_TDP_W}W GPU @ {UTIL:.0%} util")
    print(f"  Energy:  {energy_kwh*1000:.4f} Wh = {energy_kwh:.6f} kWh\n")
    print(f"  {'Region':<16} {'Location':<22} {'gCO₂/kWh':>9} {'gCO₂ total':>12} {'car metres':>12}")
    print("  " + "-" * 75)

    for region_id, label in REGIONS_TO_COMPARE:
        r = grid.get_region(region_id)
        co2_g = energy_kwh * r.carbon_intensity_g_per_kwh
        car_m = (co2_g / 1000) / 0.21 * 1000  # kg → km → m
        print(
            f"  {region_id:<16} {label:<22} "
            f"{r.carbon_intensity_g_per_kwh:>9.0f} "
            f"{co2_g:>12.4f} "
            f"{car_m:>12.2f}"
        )

    best = min(REGIONS_TO_COMPARE, key=lambda r: grid.get_intensity(r[0]))
    worst = max(REGIONS_TO_COMPARE, key=lambda r: grid.get_intensity(r[0]))
    ratio = grid.get_intensity(worst[0]) / grid.get_intensity(best[0])
    print(f"\n  💡 Switching from {worst[1].strip()} → {best[1].strip()} saves {ratio:.1f}x CO₂")


def run_tracker_direct():
    """Show the CarbonTracker API used directly (lower level)."""
    print("\n" + "=" * 60)
    print("  Demo 3: CarbonTracker direct API")
    print("=" * 60)

    tracker = CarbonTracker(
        gpu_tdp_w=300,
        region="eu-west-1",   # Ireland
        utilization=0.9,
        capture_steps=True,
    )

    tracker.start()
    print(f"\n  Simulating 50 steps on eu-west-1 (Ireland)…")

    for step in range(50):
        time.sleep(0.01)
        tracker.step(step)

        if step % 10 == 9:
            print(
                f"    step {step+1:>3} | "
                f"energy={tracker.current_energy_kwh*1000:.3f} Wh | "
                f"CO₂={tracker.current_co2_kg*1e6:.2f} µg"
            )

    report = tracker.stop()
    print(f"\n  Final report:")
    print(str(report))
    print(f"\n  Snapshots captured: {len(report.snapshots)}")


if __name__ == "__main__":
    run_demo_callback()
    run_region_comparison()
    run_tracker_direct()
    print("\n✅ Demo complete.\n")
