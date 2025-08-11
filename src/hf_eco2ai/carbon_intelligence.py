"""Next-generation carbon intelligence engine with advanced analytics."""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CarbonInsight:
    """Advanced carbon intelligence insight."""
    insight_id: str
    timestamp: datetime
    category: str  # efficiency, optimization, environmental, cost
    severity: str  # low, medium, high, critical
    title: str
    description: str
    impact_estimate: float  # potential CO2 savings in kg
    implementation_effort: str  # low, medium, high
    action_items: List[str]
    confidence_score: float  # 0-1


@dataclass
class CarbonPrediction:
    """ML-based carbon footprint prediction."""
    prediction_id: str
    timestamp: datetime
    training_duration_hours: float
    predicted_co2_kg: float
    predicted_energy_kwh: float
    confidence_interval: Tuple[float, float]
    assumptions: Dict[str, Any]
    optimization_recommendations: List[str]


@dataclass
class TrainingOptimization:
    """Advanced training optimization recommendations."""
    optimization_id: str
    timestamp: datetime
    optimization_type: str  # model, hardware, scheduling, environmental
    current_efficiency: float
    potential_efficiency: float
    co2_savings_kg: float
    cost_savings_usd: float
    implementation_complexity: str
    estimated_implementation_time_hours: float
    prerequisites: List[str]
    code_snippets: Dict[str, str]


class CarbonIntelligenceEngine:
    """Advanced carbon intelligence engine for ML training optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the carbon intelligence engine.
        
        Args:
            config: Configuration dictionary for the intelligence engine
        """
        self.config = config or {}
        self.insights: List[CarbonInsight] = []
        self.predictions: List[CarbonPrediction] = []
        self.optimizations: List[TrainingOptimization] = []
        self.learning_history: List[Dict[str, Any]] = []
        
        # ML models for prediction (placeholder for actual ML implementation)
        self.efficiency_model = None
        self.carbon_model = None
        self.optimization_model = None
        
    async def analyze_training_session(
        self, 
        metrics: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> List[CarbonInsight]:
        """Analyze a training session and generate carbon intelligence insights.
        
        Args:
            metrics: Training metrics including energy, CO2, performance data
            training_config: Training configuration parameters
            
        Returns:
            List of carbon insights and recommendations
        """
        insights = []
        
        # Energy efficiency analysis
        if 'energy_kwh' in metrics and 'samples_processed' in metrics:
            efficiency = metrics['samples_processed'] / metrics['energy_kwh']
            insight = await self._analyze_energy_efficiency(efficiency, metrics)
            if insight:
                insights.append(insight)
        
        # Carbon intensity analysis
        if 'co2_kg' in metrics and 'grid_intensity' in metrics:
            insight = await self._analyze_carbon_intensity(metrics)
            if insight:
                insights.append(insight)
        
        # Training optimization analysis
        insight = await self._analyze_training_optimization(training_config, metrics)
        if insight:
            insights.append(insight)
        
        # Environmental timing analysis
        insight = await self._analyze_environmental_timing(metrics)
        if insight:
            insights.append(insight)
        
        # Store insights
        self.insights.extend(insights)
        return insights
    
    async def _analyze_energy_efficiency(
        self, 
        current_efficiency: float, 
        metrics: Dict[str, Any]
    ) -> Optional[CarbonInsight]:
        """Analyze energy efficiency and generate insights."""
        # Benchmark against historical data
        historical_efficiency = np.mean([h.get('efficiency', 0) for h in self.learning_history])
        
        if historical_efficiency == 0:
            # First run, establish baseline
            return CarbonInsight(
                insight_id=f"baseline_{int(time.time())}",
                timestamp=datetime.now(),
                category="efficiency",
                severity="low",
                title="Energy Efficiency Baseline Established",
                description=f"Baseline energy efficiency: {current_efficiency:.0f} samples/kWh",
                impact_estimate=0.0,
                implementation_effort="low",
                action_items=["Continue monitoring to establish patterns"],
                confidence_score=0.9
            )
        
        efficiency_change = (current_efficiency - historical_efficiency) / historical_efficiency
        
        if efficiency_change < -0.15:  # 15% decrease
            return CarbonInsight(
                insight_id=f"efficiency_drop_{int(time.time())}",
                timestamp=datetime.now(),
                category="efficiency", 
                severity="high",
                title="Significant Energy Efficiency Drop Detected",
                description=f"Energy efficiency decreased by {abs(efficiency_change)*100:.1f}% from historical average",
                impact_estimate=metrics.get('energy_kwh', 0) * 0.15 * 0.4,  # potential savings
                implementation_effort="medium",
                action_items=[
                    "Review model architecture for efficiency improvements",
                    "Check for memory leaks or suboptimal GPU utilization", 
                    "Consider mixed precision training",
                    "Analyze batch size and gradient accumulation settings"
                ],
                confidence_score=0.85
            )
        
        elif efficiency_change > 0.20:  # 20% improvement
            return CarbonInsight(
                insight_id=f"efficiency_gain_{int(time.time())}",
                timestamp=datetime.now(),
                category="efficiency",
                severity="low", 
                title="Excellent Energy Efficiency Improvement",
                description=f"Energy efficiency improved by {efficiency_change*100:.1f}% - document this configuration!",
                impact_estimate=metrics.get('co2_kg', 0) * efficiency_change * 0.5,
                implementation_effort="low",
                action_items=[
                    "Document current configuration as best practice",
                    "Apply similar optimizations to other training runs",
                    "Share insights with team"
                ],
                confidence_score=0.95
            )
        
        return None
    
    async def _analyze_carbon_intensity(self, metrics: Dict[str, Any]) -> Optional[CarbonInsight]:
        """Analyze carbon intensity and suggest timing optimizations."""
        grid_intensity = metrics.get('grid_intensity', 0)
        
        if grid_intensity > 500:  # High carbon intensity (>500g CO2/kWh)
            return CarbonInsight(
                insight_id=f"high_carbon_{int(time.time())}",
                timestamp=datetime.now(),
                category="environmental",
                severity="medium",
                title="High Grid Carbon Intensity Detected",
                description=f"Current grid carbon intensity: {grid_intensity:.0f}g CO₂/kWh",
                impact_estimate=metrics.get('co2_kg', 0) * 0.3,  # potential 30% reduction
                implementation_effort="low",
                action_items=[
                    "Consider scheduling training during low-carbon hours (typically 2-6 AM)",
                    "Evaluate moving workloads to regions with cleaner energy",
                    "Implement carbon-aware scheduling",
                    "Consider purchasing carbon offsets for immediate training needs"
                ],
                confidence_score=0.9
            )
        
        elif grid_intensity < 200:  # Very clean energy
            return CarbonInsight(
                insight_id=f"clean_energy_{int(time.time())}",
                timestamp=datetime.now(),
                category="environmental",
                severity="low",
                title="Excellent Clean Energy Window",
                description=f"Training during optimal low-carbon period: {grid_intensity:.0f}g CO₂/kWh",
                impact_estimate=0.0,
                implementation_effort="low",
                action_items=[
                    "Continue training during these optimal hours",
                    "Document this timing for future training schedules",
                    "Consider increasing training intensity during clean energy windows"
                ],
                confidence_score=0.95
            )
        
        return None
    
    async def _analyze_training_optimization(
        self, 
        training_config: Dict[str, Any], 
        metrics: Dict[str, Any]
    ) -> Optional[CarbonInsight]:
        """Analyze training configuration for optimization opportunities."""
        optimizations = []
        
        # Batch size optimization
        batch_size = training_config.get('batch_size', 32)
        if batch_size < 64 and metrics.get('gpu_utilization', 100) < 80:
            optimizations.append("Increase batch size to improve GPU utilization")
        
        # Mixed precision check
        if not training_config.get('fp16', False) and not training_config.get('bf16', False):
            optimizations.append("Enable mixed precision training (fp16/bf16) for 30-50% speed improvement")
        
        # Gradient accumulation
        if training_config.get('gradient_accumulation_steps', 1) == 1 and batch_size < 128:
            optimizations.append("Use gradient accumulation to simulate larger batch sizes")
        
        if optimizations:
            return CarbonInsight(
                insight_id=f"training_opt_{int(time.time())}",
                timestamp=datetime.now(),
                category="optimization",
                severity="medium",
                title="Training Configuration Optimization Opportunities",
                description="Identified multiple opportunities to improve training efficiency",
                impact_estimate=metrics.get('co2_kg', 0) * 0.25,  # potential 25% reduction
                implementation_effort="medium",
                action_items=optimizations,
                confidence_score=0.8
            )
        
        return None
    
    async def _analyze_environmental_timing(self, metrics: Dict[str, Any]) -> Optional[CarbonInsight]:
        """Analyze environmental factors and suggest optimal timing."""
        current_hour = datetime.now().hour
        
        # Peak hours analysis (typically 6 PM - 10 PM have higher carbon intensity)
        if 18 <= current_hour <= 22:
            return CarbonInsight(
                insight_id=f"peak_hours_{int(time.time())}",
                timestamp=datetime.now(),
                category="environmental", 
                severity="medium",
                title="Training During Peak Carbon Hours",
                description="Training during peak electricity demand hours with higher carbon intensity",
                impact_estimate=metrics.get('co2_kg', 0) * 0.2,
                implementation_effort="low",
                action_items=[
                    "Consider rescheduling non-urgent training to off-peak hours (11 PM - 6 AM)",
                    "Implement automated scheduling to avoid peak carbon intensity periods",
                    "Use prediction models to find optimal training windows"
                ],
                confidence_score=0.75
            )
        
        return None
    
    async def predict_training_impact(
        self,
        training_config: Dict[str, Any],
        estimated_duration_hours: float
    ) -> CarbonPrediction:
        """Predict the carbon impact of a training run before it starts.
        
        Args:
            training_config: Training configuration parameters
            estimated_duration_hours: Estimated training duration
            
        Returns:
            Carbon prediction with confidence intervals
        """
        # Simple prediction model (in real implementation, use ML models)
        base_power_watts = 250 * training_config.get('num_gpus', 1)  # Estimate GPU power
        estimated_energy_kwh = (base_power_watts / 1000) * estimated_duration_hours
        
        # Factor in efficiency improvements
        efficiency_multiplier = 1.0
        if training_config.get('fp16') or training_config.get('bf16'):
            efficiency_multiplier *= 0.7  # 30% improvement with mixed precision
        
        if training_config.get('batch_size', 32) >= 64:
            efficiency_multiplier *= 0.9  # 10% improvement with larger batches
        
        estimated_energy_kwh *= efficiency_multiplier
        
        # Estimate CO2 based on average grid intensity
        avg_grid_intensity = 400  # g CO2/kWh (global average)
        estimated_co2_kg = estimated_energy_kwh * avg_grid_intensity / 1000
        
        # Create confidence interval (±20%)
        confidence_lower = estimated_co2_kg * 0.8
        confidence_upper = estimated_co2_kg * 1.2
        
        # Generate optimization recommendations
        recommendations = []
        if not (training_config.get('fp16') or training_config.get('bf16')):
            recommendations.append("Enable mixed precision training for 30% energy reduction")
        
        if training_config.get('batch_size', 32) < 64:
            recommendations.append("Increase batch size for better GPU utilization")
        
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 22:
            recommendations.append("Schedule training during off-peak hours (11 PM - 6 AM) for cleaner energy")
        
        prediction = CarbonPrediction(
            prediction_id=f"pred_{int(time.time())}",
            timestamp=datetime.now(),
            training_duration_hours=estimated_duration_hours,
            predicted_co2_kg=estimated_co2_kg,
            predicted_energy_kwh=estimated_energy_kwh,
            confidence_interval=(confidence_lower, confidence_upper),
            assumptions={
                "avg_gpu_power_watts": base_power_watts,
                "grid_carbon_intensity": avg_grid_intensity,
                "efficiency_improvements": efficiency_multiplier
            },
            optimization_recommendations=recommendations
        )
        
        self.predictions.append(prediction)
        return prediction
    
    def generate_intelligence_report(self) -> Dict[str, Any]:
        """Generate a comprehensive carbon intelligence report."""
        total_insights = len(self.insights)
        critical_insights = len([i for i in self.insights if i.severity == "critical"])
        high_insights = len([i for i in self.insights if i.severity == "high"])
        
        total_potential_savings = sum(i.impact_estimate for i in self.insights)
        
        # Categorize insights
        categories = {}
        for insight in self.insights:
            if insight.category not in categories:
                categories[insight.category] = []
            categories[insight.category].append(insight)
        
        # Recent predictions
        recent_predictions = self.predictions[-5:] if self.predictions else []
        
        report = {
            "report_id": f"intelligence_{int(time.time())}",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_insights": total_insights,
                "critical_insights": critical_insights,
                "high_priority_insights": high_insights,
                "total_potential_co2_savings_kg": round(total_potential_savings, 3),
                "carbon_credits_value_usd": round(total_potential_savings * 25, 2)  # $25/tCO2
            },
            "insights_by_category": {
                category: len(insights) for category, insights in categories.items()
            },
            "top_recommendations": [
                {
                    "title": insight.title,
                    "impact_kg_co2": insight.impact_estimate,
                    "effort": insight.implementation_effort,
                    "actions": insight.action_items
                }
                for insight in sorted(self.insights, key=lambda x: x.impact_estimate, reverse=True)[:5]
            ],
            "recent_predictions": [
                {
                    "prediction_id": p.prediction_id,
                    "predicted_co2_kg": p.predicted_co2_kg,
                    "confidence_interval": p.confidence_interval,
                    "recommendations": p.optimization_recommendations
                }
                for p in recent_predictions
            ]
        }
        
        return report
    
    def export_insights_json(self, filepath: Path) -> None:
        """Export insights to JSON file."""
        data = {
            "insights": [asdict(insight) for insight in self.insights],
            "predictions": [asdict(pred) for pred in self.predictions],
            "intelligence_report": self.generate_intelligence_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Carbon intelligence insights exported to {filepath}")


class CarbonIntelligenceCallback:
    """Integration callback for carbon intelligence with HF Trainer."""
    
    def __init__(self, intelligence_engine: Optional[CarbonIntelligenceEngine] = None):
        """Initialize the callback with an intelligence engine."""
        self.intelligence_engine = intelligence_engine or CarbonIntelligenceEngine()
        self.training_start_time: Optional[datetime] = None
        self.insights_generated = False
    
    async def on_training_start(self, training_config: Dict[str, Any]) -> None:
        """Called when training starts."""
        self.training_start_time = datetime.now()
        
        # Generate prediction
        estimated_duration = training_config.get('estimated_duration_hours', 2.0)
        prediction = await self.intelligence_engine.predict_training_impact(
            training_config, 
            estimated_duration
        )
        
        logger.info(f"Carbon Intelligence Prediction:")
        logger.info(f"  Estimated CO₂: {prediction.predicted_co2_kg:.2f} kg")
        logger.info(f"  Estimated Energy: {prediction.predicted_energy_kwh:.2f} kWh")
        logger.info(f"  Confidence Interval: {prediction.confidence_interval}")
        
        if prediction.optimization_recommendations:
            logger.info("  Optimization Recommendations:")
            for rec in prediction.optimization_recommendations:
                logger.info(f"    - {rec}")
    
    async def on_epoch_end(self, metrics: Dict[str, Any], training_config: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        # Generate insights every few epochs to avoid overhead
        epoch = metrics.get('epoch', 0)
        if epoch % 5 == 0 or epoch == 1:  # First epoch and every 5th epoch
            insights = await self.intelligence_engine.analyze_training_session(
                metrics, 
                training_config
            )
            
            if insights:
                logger.info(f"Generated {len(insights)} carbon intelligence insights:")
                for insight in insights:
                    logger.info(f"  [{insight.severity.upper()}] {insight.title}")
                    if insight.impact_estimate > 0:
                        logger.info(f"    Potential CO₂ savings: {insight.impact_estimate:.2f} kg")
    
    async def on_training_end(self, final_metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
        if not self.insights_generated:
            # Generate final comprehensive analysis
            training_duration = (datetime.now() - self.training_start_time).total_seconds() / 3600
            final_metrics['training_duration_hours'] = training_duration
            
            insights = await self.intelligence_engine.analyze_training_session(
                final_metrics, 
                {"training_duration_hours": training_duration}
            )
            
            # Generate and export intelligence report
            report = self.intelligence_engine.generate_intelligence_report()
            
            logger.info("\n" + "="*60)
            logger.info("CARBON INTELLIGENCE FINAL REPORT")
            logger.info("="*60)
            logger.info(f"Total Insights Generated: {report['summary']['total_insights']}")
            logger.info(f"Potential CO₂ Savings: {report['summary']['total_potential_co2_savings_kg']:.2f} kg")
            logger.info(f"Equivalent Carbon Credits Value: ${report['summary']['carbon_credits_value_usd']:.2f}")
            
            if report['top_recommendations']:
                logger.info("\nTop Recommendations:")
                for i, rec in enumerate(report['top_recommendations'], 1):
                    logger.info(f"{i}. {rec['title']} (Impact: {rec['impact_kg_co2']:.2f} kg CO₂)")
            
            logger.info("="*60)
            
            self.insights_generated = True
    
    def get_intelligence_engine(self) -> CarbonIntelligenceEngine:
        """Get the intelligence engine for direct access."""
        return self.intelligence_engine