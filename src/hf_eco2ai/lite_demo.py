#!/usr/bin/env python3
"""Lightweight demo of HF Eco2AI Plugin core functionality without external dependencies."""

import os
import json
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from pathlib import Path


@dataclass
class CarbonMetricsLite:
    """Lightweight carbon metrics without external dependencies."""
    timestamp: float
    energy_kwh: float = 0.0
    co2_kg: float = 0.0
    power_watts: float = 0.0
    duration_seconds: float = 0.0
    samples_processed: int = 0
    gpu_utilization: float = 0.0


class EnergyTrackerLite:
    """Lightweight energy tracking for demo purposes."""
    
    def __init__(self):
        self.start_time = None
        self.total_energy = 0.0
        self.base_power = 100.0  # Mock base power consumption in watts
        self.running = False
        
    def start(self) -> None:
        """Start energy tracking."""
        self.start_time = time.time()
        self.running = True
        print("🔋 Energy tracking started")
        
    def stop(self) -> CarbonMetricsLite:
        """Stop tracking and return metrics."""
        if not self.running:
            return CarbonMetricsLite(timestamp=time.time())
            
        duration = time.time() - self.start_time
        energy_kwh = (self.base_power * duration) / 3600000  # Convert to kWh
        
        # Mock carbon intensity (California grid ~240g CO2/kWh)
        co2_kg = energy_kwh * 0.240
        
        self.running = False
        print(f"⚡ Energy tracking stopped - {duration:.2f}s, {energy_kwh:.4f} kWh, {co2_kg:.4f} kg CO₂")
        
        return CarbonMetricsLite(
            timestamp=time.time(),
            energy_kwh=energy_kwh,
            co2_kg=co2_kg,
            power_watts=self.base_power,
            duration_seconds=duration,
            samples_processed=1000,  # Mock processed samples
            gpu_utilization=85.0     # Mock GPU utilization
        )


class CarbonReportLite:
    """Lightweight carbon reporting."""
    
    def __init__(self, metrics: List[CarbonMetricsLite]):
        self.metrics = metrics
        self.total_energy = sum(m.energy_kwh for m in metrics)
        self.total_co2 = sum(m.co2_kg for m in metrics)
        self.total_duration = sum(m.duration_seconds for m in metrics)
        
    def generate_summary(self) -> str:
        """Generate a human-readable summary."""
        return f"""
🌱 CARBON IMPACT REPORT
=======================
Total Energy: {self.total_energy:.4f} kWh
Total CO₂: {self.total_co2:.4f} kg CO₂eq
Duration: {self.total_duration:.2f} seconds
Efficiency: {1000/self.total_energy:.0f} samples/kWh

Environmental Equivalent:
🚗 {self.total_co2 * 4.2:.1f} km of car driving
🌳 {self.total_co2 * 0.05:.2f} trees needed for offset
💰 ${self.total_co2 * 0.25:.2f} in carbon credits
"""

    def save_json(self, path: str) -> None:
        """Save report to JSON file."""
        report_data = {
            "total_energy_kwh": self.total_energy,
            "total_co2_kg": self.total_co2,
            "total_duration_seconds": self.total_duration,
            "metrics": [asdict(m) for m in self.metrics],
            "generated_at": time.time()
        }
        
        with open(path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"📊 Report saved to {path}")


class Eco2AILiteDemo:
    """Lightweight demo of Eco2AI functionality."""
    
    def __init__(self, project_name: str = "hf-eco2ai-demo"):
        self.project_name = project_name
        self.tracker = EnergyTrackerLite()
        self.metrics_history = []
        
    def simulate_training_epoch(self, epoch: int, duration: float = 2.0) -> None:
        """Simulate a training epoch with carbon tracking."""
        print(f"\n🚀 Starting epoch {epoch}")
        
        # Start energy tracking
        self.tracker.start()
        
        # Simulate training work
        time.sleep(duration)
        
        # Stop tracking and collect metrics
        metrics = self.tracker.stop()
        self.metrics_history.append(metrics)
        
        print(f"✅ Epoch {epoch} completed:")
        print(f"   Energy: {metrics.energy_kwh:.4f} kWh")
        print(f"   CO₂: {metrics.co2_kg:.4f} kg")
        print(f"   Power: {metrics.power_watts:.1f}W")
        
    def generate_final_report(self) -> CarbonReportLite:
        """Generate final carbon impact report."""
        return CarbonReportLite(self.metrics_history)


def main():
    """Run the lightweight demo."""
    print("🌱 HF Eco2AI Plugin - Lightweight Demo")
    print("=" * 40)
    
    # Initialize demo
    demo = Eco2AILiteDemo("llama-finetune-demo")
    
    # Simulate training for 3 epochs
    for epoch in range(1, 4):
        demo.simulate_training_epoch(epoch, duration=1.5)
    
    # Generate final report
    report = demo.generate_final_report()
    print(report.generate_summary())
    
    # Save report to file
    report_path = "/root/repo/carbon_impact_demo.json"
    report.save_json(report_path)
    
    print("\n✅ Demo completed successfully!")
    print(f"📁 Report saved to: {report_path}")


if __name__ == "__main__":
    main()