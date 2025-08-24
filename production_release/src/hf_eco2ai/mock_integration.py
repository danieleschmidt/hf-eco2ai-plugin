"""
Generation 1: MAKE IT WORK (Simple) - Mock Integration for Testing
TERRAGON AUTONOMOUS SDLC v4.0 - Fallback systems for missing dependencies
"""

import time
import logging
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Suppress import errors by ensuring we don't import transformers
if 'transformers' in sys.modules:
    del sys.modules['transformers']


class MockModalityType(Enum):
    """Mock modality types for testing without transformers."""
    TEXT = "text"
    VISION = "vision" 
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


@dataclass
class MockCarbonConfig:
    """Mock configuration for testing without external dependencies."""
    project_name: str = "terragon-test"
    experiment_name: str = "generation-1-test"
    country: str = "USA"
    region: str = "CA"
    gpu_ids: List[int] = None
    export_prometheus: bool = False
    log_level: str = "EPOCH"
    save_report: bool = True
    report_path: str = "carbon_report.json"
    
    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [0]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "country": self.country,
            "region": self.region,
            "gpu_ids": self.gpu_ids,
            "export_prometheus": self.export_prometheus,
            "log_level": self.log_level,
            "save_report": self.save_report,
            "report_path": self.report_path
        }


@dataclass
class MockCarbonMetrics:
    """Mock carbon metrics for testing."""
    timestamp: float = 0.0
    step: int = 0
    epoch: int = 0
    energy_kwh: float = 0.0
    cumulative_energy_kwh: float = 0.0
    power_watts: float = 0.0
    co2_kg: float = 0.0
    cumulative_co2_kg: float = 0.0
    grid_intensity: float = 412.0  # Global average g CO₂/kWh
    samples_processed: int = 0
    samples_per_kwh: float = 0.0
    duration_seconds: float = 0.0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class MockCarbonSummary:
    """Mock carbon summary for testing."""
    total_energy_kwh: float = 0.0
    total_co2_kg: float = 0.0
    avg_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    total_duration_hours: float = 0.0
    samples_processed: int = 0
    model_parameters: int = 0
    final_loss: Optional[float] = None
    
    # Environmental equivalents
    equivalent_km_driven: float = 0.0
    equivalent_trees_planted: float = 0.0
    estimated_cost_usd: float = 0.0
    
    def calculate_equivalents(self):
        """Calculate environmental equivalents."""
        # CO₂ equivalents
        self.equivalent_km_driven = self.total_co2_kg * 4.6  # kg CO₂ per km for average car
        self.equivalent_trees_planted = self.total_co2_kg * 0.0365  # trees needed per kg CO₂
        
        # Cost estimates (rough)
        self.estimated_cost_usd = self.total_energy_kwh * 0.12  # $0.12/kWh average


class MockCarbonReport:
    """Mock carbon report for testing."""
    
    def __init__(self, report_id: str = None, config: Dict[str, Any] = None):
        self.report_id = report_id or f"mock-{int(time.time())}"
        self.config = config or {}
        self.detailed_metrics: List[MockCarbonMetrics] = []
        self.summary = MockCarbonSummary()
        self.training_metadata: Dict[str, Any] = {}
        self.system_metadata: Dict[str, Any] = {}
        self.recommendations: List[str] = []
        self.created_at = time.time()
    
    def add_metric(self, metric: MockCarbonMetrics):
        """Add a metric to the detailed metrics."""
        self.detailed_metrics.append(metric)
        
        # Update summary
        if self.detailed_metrics:
            latest = self.detailed_metrics[-1]
            self.summary.total_energy_kwh = latest.cumulative_energy_kwh
            self.summary.total_co2_kg = latest.cumulative_co2_kg
            self.summary.samples_processed = latest.samples_processed
            
            # Calculate averages
            powers = [m.power_watts for m in self.detailed_metrics if m.power_watts > 0]
            if powers:
                self.summary.avg_power_watts = sum(powers) / len(powers)
                self.summary.peak_power_watts = max(powers)
    
    def generate_recommendations(self):
        """Generate optimization recommendations."""
        self.recommendations = [
            "Consider using mixed precision training to reduce energy consumption by 20-40%",
            "Enable gradient checkpointing to reduce memory usage",
            "Schedule training during low-carbon grid periods (typically 2-6 AM)",
            "Use efficient data preprocessing to minimize I/O overhead",
            "Consider model pruning or knowledge distillation for deployment"
        ]
    
    def summary_text(self) -> str:
        """Generate human-readable summary."""
        self.summary.calculate_equivalents()
        
        return f"""
Carbon Tracking Summary
======================
Energy Consumed: {self.summary.total_energy_kwh:.4f} kWh
CO₂ Emissions: {self.summary.total_co2_kg:.4f} kg
Average Power: {self.summary.avg_power_watts:.1f} W
Peak Power: {self.summary.peak_power_watts:.1f} W
Duration: {self.summary.total_duration_hours:.2f} hours
Samples Processed: {self.summary.samples_processed:,}

Environmental Impact:
• Equivalent to driving {self.summary.equivalent_km_driven:.1f} km
• {self.summary.equivalent_trees_planted:.2f} trees needed for offset
• Estimated cost: ${self.summary.estimated_cost_usd:.2f}

Efficiency:
• {self.summary.samples_processed / max(self.summary.total_energy_kwh, 0.001):.0f} samples/kWh
"""


class MockEnergyTracker:
    """Mock energy tracker for testing without GPU dependencies."""
    
    def __init__(self, gpu_ids=None, country="USA", region="CA"):
        self.gpu_ids = gpu_ids or [0]
        self.country = country
        self.region = region
        self.tracking_active = False
        self.start_time = None
        self.current_energy = 0.0
        self.current_co2 = 0.0
        self.power_readings = []
        
        # Mock power consumption (realistic values for training)
        self.base_power = 150  # Base system power
        self.gpu_power_range = (50, 200)  # Per GPU
        
    def is_available(self) -> bool:
        """Check if energy tracking is available."""
        return True  # Always available in mock mode
    
    def start_tracking(self):
        """Start energy tracking."""
        self.tracking_active = True
        self.start_time = time.time()
        logger.info("Mock energy tracking started")
    
    def stop_tracking(self):
        """Stop energy tracking."""
        self.tracking_active = False
        logger.info("Mock energy tracking stopped")
    
    def get_current_consumption(self):
        """Get current power, energy, and CO₂ consumption."""
        if not self.tracking_active or not self.start_time:
            return 0.0, 0.0, 0.0
        
        # Calculate elapsed time
        elapsed_hours = (time.time() - self.start_time) / 3600
        
        # Mock power consumption (varies slightly over time)
        import random
        current_power = self.base_power + len(self.gpu_ids) * random.uniform(*self.gpu_power_range)
        self.power_readings.append(current_power)
        
        # Calculate energy (simple integration)
        if len(self.power_readings) > 1:
            avg_power = sum(self.power_readings) / len(self.power_readings)
            self.current_energy = avg_power * elapsed_hours / 1000  # Convert to kWh
        
        # Calculate CO₂ (using global average grid intensity)
        grid_intensity = 412  # g CO₂/kWh global average
        self.current_co2 = self.current_energy * grid_intensity / 1000  # Convert to kg
        
        return current_power, self.current_energy, self.current_co2
    
    def get_efficiency_metrics(self, samples_processed: int) -> Dict[str, float]:
        """Calculate efficiency metrics."""
        if self.current_energy <= 0 or samples_processed <= 0:
            return {
                "samples_per_kwh": 0.0,
                "energy_per_sample": 0.0,
                "co2_per_sample": 0.0
            }
        
        return {
            "samples_per_kwh": samples_processed / self.current_energy,
            "energy_per_sample": self.current_energy / samples_processed,
            "co2_per_sample": self.current_co2 / samples_processed
        }


class MockEco2AICallback:
    """Mock Eco2AI callback for testing without transformers dependency."""
    
    def __init__(self, config: Optional[MockCarbonConfig] = None):
        self.config = config or MockCarbonConfig()
        self.energy_tracker = MockEnergyTracker(
            gpu_ids=self.config.gpu_ids,
            country=self.config.country,
            region=self.config.region
        )
        
        self.carbon_report = MockCarbonReport(
            config=self.config.to_dict()
        )
        
        self._training_start_time = None
        self._samples_processed = 0
        self._current_epoch = 0
        self._current_step = 0
        
        logger.info(f"Mock Eco2AI callback initialized for {self.config.project_name}")
    
    def on_train_begin(self, **kwargs):
        """Mock train begin handler."""
        self._training_start_time = time.time()
        self.energy_tracker.start_tracking()
        
        # Mock training metadata
        self.carbon_report.training_metadata.update({
            "project_name": self.config.project_name,
            "experiment_name": self.config.experiment_name,
            "model_name": "mock-transformer-model",
            "num_train_epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.001,
        })
        
        logger.info("Mock training begun - carbon tracking active")
    
    def on_step_end(self, step: int, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Mock step end handler."""
        self._current_step = step
        self._samples_processed += 8  # Mock batch size
        
        # Create mock metrics
        power, energy, co2 = self.energy_tracker.get_current_consumption()
        
        metric = MockCarbonMetrics(
            timestamp=time.time(),
            step=step,
            epoch=self._current_epoch,
            energy_kwh=energy / max(step, 1) * 0.01,  # Step energy
            cumulative_energy_kwh=energy,
            power_watts=power,
            co2_kg=co2 / max(step, 1) * 0.01,  # Step CO₂
            cumulative_co2_kg=co2,
            samples_processed=self._samples_processed
        )
        
        if logs:
            metric.loss = logs.get("loss", 0.5 - 0.01 * step)  # Mock decreasing loss
            metric.learning_rate = logs.get("lr", 0.001)
        
        self.carbon_report.add_metric(metric)
        
        # Log periodically
        if step % 10 == 0:
            logger.info(f"Step {step} - Energy: {energy:.4f} kWh, CO₂: {co2:.4f} kg")
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Mock epoch end handler."""
        self._current_epoch = epoch
        power, energy, co2 = self.energy_tracker.get_current_consumption()
        logger.info(f"Epoch {epoch} complete - Total energy: {energy:.4f} kWh, CO₂: {co2:.4f} kg")
    
    def on_train_end(self, **kwargs):
        """Mock train end handler."""
        self.energy_tracker.stop_tracking()
        
        # Finalize report
        if self._training_start_time:
            duration_hours = (time.time() - self._training_start_time) / 3600
            self.carbon_report.summary.total_duration_hours = duration_hours
        
        self.carbon_report.summary.calculate_equivalents()
        self.carbon_report.generate_recommendations()
        
        logger.info("Mock training completed - final report generated")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        power, energy, co2 = self.energy_tracker.get_current_consumption()
        efficiency = self.energy_tracker.get_efficiency_metrics(self._samples_processed)
        
        return {
            "power_watts": power,
            "energy_kwh": energy,
            "co2_kg": co2,
            "samples_processed": self._samples_processed,
            "samples_per_kwh": efficiency["samples_per_kwh"],
            "grid_intensity": 412.0  # Mock global average
        }
    
    def generate_report(self) -> MockCarbonReport:
        """Generate final report."""
        return self.carbon_report
    
    @property
    def carbon_summary(self) -> str:
        """Get carbon summary."""
        return self.carbon_report.summary_text()


def run_mock_training_demo():
    """Run a complete mock training demonstration."""
    print("🚀 TERRAGON Mock Training Demonstration")
    print("Generation 1: MAKE IT WORK (Simple)")
    print("="*50)
    
    # Initialize mock callback
    config = MockCarbonConfig(
        project_name="terragon-gen1-demo",
        experiment_name="mock-training",
        gpu_ids=[0, 1]  # Mock dual GPU
    )
    
    callback = MockEco2AICallback(config)
    
    # Simulate training
    callback.on_train_begin()
    
    epochs = 3
    steps_per_epoch = 50
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch + 1}/{epochs}")
        
        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step
            
            # Mock loss decreases over time
            loss = 2.0 * (1 - global_step / (epochs * steps_per_epoch)) + 0.1
            
            callback.on_step_end(
                step=global_step,
                logs={"loss": loss, "lr": 0.001}
            )
        
        callback.on_epoch_end(epoch=epoch)
    
    # End training
    callback.on_train_end()
    
    # Show results
    print("\n📊 Final Results:")
    print(callback.carbon_summary)
    
    # Save report
    import json
    report_data = {
        "report_id": callback.carbon_report.report_id,
        "summary": {
            "total_energy_kwh": callback.carbon_report.summary.total_energy_kwh,
            "total_co2_kg": callback.carbon_report.summary.total_co2_kg,
            "duration_hours": callback.carbon_report.summary.total_duration_hours,
            "samples_processed": callback.carbon_report.summary.samples_processed
        },
        "recommendations": callback.carbon_report.recommendations
    }
    
    with open("/tmp/mock_training_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print("✅ Mock training demonstration completed successfully")
    print(f"📄 Report saved to: /tmp/mock_training_report.json")
    
    return True


if __name__ == "__main__":
    run_mock_training_demo()