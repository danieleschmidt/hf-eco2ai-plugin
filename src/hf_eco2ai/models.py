"""Data models for carbon tracking metrics and reports."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path


@dataclass
class CarbonMetrics:
    """Real-time carbon tracking metrics for a single measurement point."""
    
    # Timestamp and identification
    timestamp: float = field(default_factory=time.time)
    step: Optional[int] = None
    epoch: Optional[int] = None
    
    # Energy consumption
    energy_kwh: float = 0.0
    cumulative_energy_kwh: float = 0.0
    power_watts: float = 0.0
    
    # Carbon emissions
    co2_kg: float = 0.0
    cumulative_co2_kg: float = 0.0
    grid_intensity: float = 0.0  # g CO₂/kWh
    
    # GPU-specific metrics
    gpu_energy_kwh: Dict[int, float] = field(default_factory=dict)
    gpu_power_watts: Dict[int, float] = field(default_factory=dict)
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    gpu_memory_used: Dict[int, float] = field(default_factory=dict)
    gpu_temperature: Dict[int, float] = field(default_factory=dict)
    
    # Training efficiency
    samples_processed: int = 0
    samples_per_kwh: float = 0.0
    tokens_processed: Optional[int] = None
    tokens_per_kwh: Optional[float] = None
    
    # Duration tracking
    duration_seconds: float = 0.0
    training_step_duration: Optional[float] = None
    
    # Model and training context
    model_parameters: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    loss: Optional[float] = None
    
    # Regional and environmental
    location: Optional[str] = None
    weather_impact: Optional[float] = None
    renewable_percentage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "step": self.step,
            "epoch": self.epoch,
            "energy_kwh": self.energy_kwh,
            "cumulative_energy_kwh": self.cumulative_energy_kwh,
            "power_watts": self.power_watts,
            "co2_kg": self.co2_kg,
            "cumulative_co2_kg": self.cumulative_co2_kg,
            "grid_intensity": self.grid_intensity,
            "gpu_metrics": {
                "energy_kwh": self.gpu_energy_kwh,
                "power_watts": self.gpu_power_watts,
                "utilization": self.gpu_utilization,
                "memory_used": self.gpu_memory_used,
                "temperature": self.gpu_temperature,
            },
            "efficiency": {
                "samples_processed": self.samples_processed,
                "samples_per_kwh": self.samples_per_kwh,
                "tokens_processed": self.tokens_processed,
                "tokens_per_kwh": self.tokens_per_kwh,
            },
            "duration_seconds": self.duration_seconds,
            "training_context": {
                "model_parameters": self.model_parameters,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "loss": self.loss,
            },
            "environment": {
                "location": self.location,
                "weather_impact": self.weather_impact,
                "renewable_percentage": self.renewable_percentage,
            }
        }


@dataclass
class CarbonSummary:
    """Summary statistics for a complete training run."""
    
    # Total consumption
    total_energy_kwh: float = 0.0
    total_co2_kg: float = 0.0
    average_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    
    # Training metrics
    total_duration_hours: float = 0.0
    total_steps: int = 0
    total_epochs: int = 0
    total_samples: int = 0
    
    # Efficiency metrics
    average_samples_per_kwh: float = 0.0
    energy_per_sample: float = 0.0
    co2_per_sample: float = 0.0
    
    # Cost estimates
    estimated_cost_usd: Optional[float] = None
    carbon_credit_cost_usd: Optional[float] = None
    
    # Environmental equivalents
    equivalent_km_driven: float = 0.0
    equivalent_trees_needed: float = 0.0
    equivalent_coal_burned_kg: float = 0.0
    
    # Model context
    model_name: Optional[str] = None
    model_parameters: Optional[int] = None
    dataset_size: Optional[int] = None
    final_loss: Optional[float] = None
    
    def calculate_equivalents(self):
        """Calculate environmental equivalents for CO₂ emissions."""
        # Average car emits 404g CO₂/mile = 251g CO₂/km
        self.equivalent_km_driven = (self.total_co2_kg * 1000) / 251
        
        # One tree absorbs ~22kg CO₂/year
        self.equivalent_trees_needed = self.total_co2_kg / 22
        
        # Coal emits ~2.4kg CO₂/kg when burned
        self.equivalent_coal_burned_kg = self.total_co2_kg / 2.4


@dataclass
class OptimizationRecommendation:
    """Recommendation for reducing carbon footprint."""
    
    category: str  # "training", "scheduling", "hardware", "model"
    title: str
    description: str
    potential_reduction_percent: float
    potential_co2_savings_kg: float
    implementation_difficulty: str  # "easy", "medium", "hard"
    estimated_time_hours: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return {
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "impact": {
                "reduction_percent": self.potential_reduction_percent,
                "co2_savings_kg": self.potential_co2_savings_kg,
            },
            "implementation": {
                "difficulty": self.implementation_difficulty,
                "estimated_time_hours": self.estimated_time_hours,
            }
        }


@dataclass
class EnvironmentalImpact:
    """Detailed environmental impact analysis."""
    
    # Grid composition
    renewable_percentage: float = 0.0
    fossil_fuel_percentage: float = 0.0
    nuclear_percentage: float = 0.0
    
    # Regional data
    region_name: str = ""
    country_code: str = ""
    timezone: str = ""
    
    # Time-based analysis
    peak_hours_co2: float = 0.0
    off_peak_hours_co2: float = 0.0
    optimal_training_window: Optional[str] = None
    
    # Comparison data
    regional_average_intensity: float = 0.0
    global_average_intensity: float = 475.0  # g CO₂/kWh
    efficiency_percentile: Optional[float] = None
    
    def calculate_optimization_potential(self, current_co2: float) -> float:
        """Calculate potential CO₂ reduction by moving to optimal region/time."""
        if self.regional_average_intensity > 0:
            # Find the cleanest grid available (Norway ~20g CO₂/kWh)
            cleanest_grid_intensity = 20.0
            potential_reduction = 1.0 - (cleanest_grid_intensity / self.regional_average_intensity)
            return current_co2 * potential_reduction
        return 0.0


@dataclass
class CarbonReport:
    """Comprehensive carbon tracking report for a training run."""
    
    # Report metadata
    report_id: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    
    # Configuration used
    config: Optional[Dict[str, Any]] = None
    
    # Summary statistics
    summary: CarbonSummary = field(default_factory=CarbonSummary)
    
    # Detailed metrics timeline
    detailed_metrics: List[CarbonMetrics] = field(default_factory=list)
    
    # Environmental context
    environmental_impact: EnvironmentalImpact = field(default_factory=EnvironmentalImpact)
    
    # Optimization recommendations
    recommendations: List[OptimizationRecommendation] = field(default_factory=list)
    
    # Additional metadata
    training_metadata: Dict[str, Any] = field(default_factory=dict)
    system_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric(self, metric: CarbonMetrics):
        """Add a new metric measurement to the report."""
        self.detailed_metrics.append(metric)
        
        # Update summary statistics
        self.summary.total_energy_kwh = metric.cumulative_energy_kwh
        self.summary.total_co2_kg = metric.cumulative_co2_kg
        self.summary.total_steps = len(self.detailed_metrics)
        
        if metric.epoch:
            self.summary.total_epochs = max(self.summary.total_epochs, metric.epoch)
        
        # Calculate averages and peaks
        powers = [m.power_watts for m in self.detailed_metrics if m.power_watts > 0]
        if powers:
            self.summary.average_power_watts = sum(powers) / len(powers)
            self.summary.peak_power_watts = max(powers)
        
        # Update efficiency metrics
        if self.summary.total_energy_kwh > 0 and metric.samples_processed > 0:
            self.summary.total_samples = metric.samples_processed
            self.summary.average_samples_per_kwh = metric.samples_processed / self.summary.total_energy_kwh
            self.summary.energy_per_sample = self.summary.total_energy_kwh / metric.samples_processed
            self.summary.co2_per_sample = self.summary.total_co2_kg / metric.samples_processed
    
    def generate_recommendations(self):
        """Generate optimization recommendations based on collected data."""
        recommendations = []
        
        # Analyze training efficiency
        if self.summary.average_samples_per_kwh < 1000:  # Example threshold
            recommendations.append(OptimizationRecommendation(
                category="training",
                title="Enable Mixed Precision Training",
                description="Use mixed precision (FP16) to reduce memory usage and energy consumption by up to 40%",
                potential_reduction_percent=40.0,
                potential_co2_savings_kg=self.summary.total_co2_kg * 0.4,
                implementation_difficulty="easy"
            ))
        
        # Analyze timing optimization
        if hasattr(self.environmental_impact, 'optimal_training_window'):
            recommendations.append(OptimizationRecommendation(
                category="scheduling",
                title="Schedule Training During Low-Carbon Hours",
                description="Run training during times when grid carbon intensity is lower",
                potential_reduction_percent=25.0,
                potential_co2_savings_kg=self.summary.total_co2_kg * 0.25,
                implementation_difficulty="medium"
            ))
        
        self.recommendations = recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "metadata": {
                "report_id": self.report_id,
                "generated_at": self.generated_at,
                "version": self.version,
            },
            "config": self.config,
            "summary": self.summary.__dict__,
            "environmental_impact": self.environmental_impact.__dict__,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "metrics_count": len(self.detailed_metrics),
            "training_metadata": self.training_metadata,
            "system_metadata": self.system_metadata,
        }
    
    def to_json(self, path: Optional[str] = None, include_detailed_metrics: bool = False) -> str:
        """Export report to JSON."""
        report_dict = self.to_dict()
        
        if include_detailed_metrics:
            report_dict["detailed_metrics"] = [m.to_dict() for m in self.detailed_metrics]
        
        json_str = json.dumps(report_dict, indent=2, default=str)
        
        if path:
            Path(path).write_text(json_str)
        
        return json_str
    
    def to_csv(self, path: str):
        """Export detailed metrics to CSV."""
        import csv
        
        if not self.detailed_metrics:
            return
        
        # Flatten metrics for CSV export
        fieldnames = [
            "timestamp", "datetime", "step", "epoch",
            "energy_kwh", "cumulative_energy_kwh", "power_watts",
            "co2_kg", "cumulative_co2_kg", "grid_intensity",
            "samples_processed", "samples_per_kwh",
            "duration_seconds", "loss"
        ]
        
        with open(path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in self.detailed_metrics:
                row = {
                    "timestamp": metric.timestamp,
                    "datetime": datetime.fromtimestamp(metric.timestamp).isoformat(),
                    "step": metric.step,
                    "epoch": metric.epoch,
                    "energy_kwh": metric.energy_kwh,
                    "cumulative_energy_kwh": metric.cumulative_energy_kwh,
                    "power_watts": metric.power_watts,
                    "co2_kg": metric.co2_kg,
                    "cumulative_co2_kg": metric.cumulative_co2_kg,
                    "grid_intensity": metric.grid_intensity,
                    "samples_processed": metric.samples_processed,
                    "samples_per_kwh": metric.samples_per_kwh,
                    "duration_seconds": metric.duration_seconds,
                    "loss": metric.loss,
                }
                writer.writerow(row)
    
    def summary_text(self) -> str:
        """Generate human-readable summary text."""
        return f"""
Training Carbon Impact Report
============================
Project: {self.training_metadata.get('project_name', 'Unknown')}
Duration: {self.summary.total_duration_hours:.1f} hours
Total Energy: {self.summary.total_energy_kwh:.2f} kWh
Total CO₂: {self.summary.total_co2_kg:.2f} kg CO₂eq
Average Power: {self.summary.average_power_watts:.0f} W
Peak Power: {self.summary.peak_power_watts:.0f} W

Training Efficiency:
- Samples processed: {self.summary.total_samples:,}
- Samples per kWh: {self.summary.average_samples_per_kwh:.0f}
- Energy per sample: {self.summary.energy_per_sample * 1000:.2f} Wh
- CO₂ per sample: {self.summary.co2_per_sample * 1000:.2f} g

Environmental Equivalents:
- Distance driven by car: {self.summary.equivalent_km_driven:.0f} km
- Trees needed to offset: {self.summary.equivalent_trees_needed:.1f}
- Coal burned equivalent: {self.summary.equivalent_coal_burned_kg:.1f} kg

Grid Information:
- Region: {self.environmental_impact.region_name}
- Average grid intensity: {self.environmental_impact.regional_average_intensity:.0f} g CO₂/kWh
- Renewable percentage: {self.environmental_impact.renewable_percentage:.1f}%

Optimization Recommendations: {len(self.recommendations)} available
"""