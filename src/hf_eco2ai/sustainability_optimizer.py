"""Advanced sustainability optimization engine for ML training."""

import asyncio
import json
import time
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SustainabilityGoal:
    """Sustainability goal definition."""
    goal_id: str
    name: str
    goal_type: str  # carbon_budget, efficiency_target, renewable_energy, cost_limit
    target_value: float
    current_value: float
    unit: str
    deadline: Optional[datetime]
    priority: str  # low, medium, high, critical
    description: str
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress towards goal."""
        if self.target_value == 0:
            return 100.0
        return min(100.0, (self.current_value / self.target_value) * 100)
    
    @property
    def is_achieved(self) -> bool:
        """Check if goal is achieved."""
        return self.current_value >= self.target_value


@dataclass
class OptimizationStrategy:
    """Optimization strategy recommendation."""
    strategy_id: str
    name: str
    category: str  # model, hardware, scheduling, environmental, behavioral
    description: str
    co2_reduction_kg: float
    cost_reduction_usd: float
    implementation_effort: str  # low, medium, high
    estimated_hours: float
    prerequisites: List[str]
    implementation_steps: List[str]
    code_examples: Dict[str, str]
    success_metrics: List[str]
    risk_level: str  # low, medium, high
    confidence_score: float  # 0-1


@dataclass
class CarbonBudget:
    """Carbon budget management."""
    budget_id: str
    name: str
    total_budget_kg: float
    used_budget_kg: float
    remaining_budget_kg: float
    period_start: datetime
    period_end: datetime
    alert_threshold_percentage: float  # Alert when this % of budget is used
    
    @property
    def utilization_percentage(self) -> float:
        """Calculate budget utilization percentage."""
        return (self.used_budget_kg / self.total_budget_kg) * 100
    
    @property
    def days_remaining(self) -> int:
        """Days remaining in budget period."""
        return (self.period_end - datetime.now()).days
    
    @property
    def daily_burn_rate_kg(self) -> float:
        """Current daily carbon burn rate."""
        days_elapsed = (datetime.now() - self.period_start).days
        if days_elapsed <= 0:
            return 0.0
        return self.used_budget_kg / days_elapsed
    
    @property
    def projected_end_usage_kg(self) -> float:
        """Projected carbon usage at period end."""
        if self.days_remaining <= 0:
            return self.used_budget_kg
        return self.used_budget_kg + (self.daily_burn_rate_kg * self.days_remaining)


class SustainabilityOptimizer:
    """Advanced sustainability optimizer for ML training workflows."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the sustainability optimizer.
        
        Args:
            config: Configuration dictionary for the optimizer
        """
        self.config = config or {}
        self.goals: List[SustainabilityGoal] = []
        self.strategies: List[OptimizationStrategy] = []
        self.carbon_budgets: List[CarbonBudget] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Default sustainability goals
        self._initialize_default_goals()
    
    def _initialize_default_goals(self) -> None:
        """Initialize default sustainability goals."""
        default_goals = [
            SustainabilityGoal(
                goal_id="carbon_efficiency",
                name="Carbon Efficiency Target",
                goal_type="efficiency_target",
                target_value=5000.0,  # samples per kg CO2
                current_value=0.0,
                unit="samples/kg_CO2",
                deadline=None,
                priority="high",
                description="Achieve high carbon efficiency in training"
            ),
            SustainabilityGoal(
                goal_id="renewable_energy",
                name="Renewable Energy Usage",
                goal_type="renewable_energy",
                target_value=80.0,  # 80% renewable energy
                current_value=0.0,
                unit="percentage",
                deadline=datetime.now() + timedelta(days=365),
                priority="medium",
                description="Use primarily renewable energy sources"
            ),
            SustainabilityGoal(
                goal_id="monthly_carbon_budget",
                name="Monthly Carbon Budget",
                goal_type="carbon_budget",
                target_value=100.0,  # 100 kg CO2 per month
                current_value=0.0,
                unit="kg_CO2",
                deadline=None,
                priority="high",
                description="Stay within monthly carbon budget"
            )
        ]
        
        self.goals.extend(default_goals)
    
    async def analyze_sustainability_opportunities(
        self,
        training_metrics: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> List[OptimizationStrategy]:
        """Analyze training session and identify sustainability optimization opportunities.
        
        Args:
            training_metrics: Current training metrics
            training_config: Training configuration
            
        Returns:
            List of optimization strategies
        """
        strategies = []
        
        # Model optimization strategies
        model_strategies = await self._analyze_model_optimizations(
            training_metrics, training_config
        )
        strategies.extend(model_strategies)
        
        # Hardware optimization strategies
        hardware_strategies = await self._analyze_hardware_optimizations(
            training_metrics, training_config
        )
        strategies.extend(hardware_strategies)
        
        # Scheduling optimization strategies
        scheduling_strategies = await self._analyze_scheduling_optimizations(
            training_metrics, training_config
        )
        strategies.extend(scheduling_strategies)
        
        # Environmental optimization strategies
        environmental_strategies = await self._analyze_environmental_optimizations(
            training_metrics, training_config
        )
        strategies.extend(environmental_strategies)
        
        # Behavioral optimization strategies
        behavioral_strategies = await self._analyze_behavioral_optimizations(
            training_metrics, training_config
        )
        strategies.extend(behavioral_strategies)
        
        # Sort by potential impact
        strategies.sort(key=lambda s: s.co2_reduction_kg, reverse=True)
        
        self.strategies.extend(strategies)
        return strategies
    
    async def _analyze_model_optimizations(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[OptimizationStrategy]:
        """Analyze model-level optimization opportunities."""
        strategies = []
        
        # Mixed precision optimization
        if not config.get('fp16', False) and not config.get('bf16', False):
            strategies.append(OptimizationStrategy(
                strategy_id=f"mixed_precision_{int(time.time())}",
                name="Enable Mixed Precision Training",
                category="model",
                description="Use fp16 or bf16 to reduce memory usage and increase training speed",
                co2_reduction_kg=metrics.get('co2_kg', 0) * 0.3,  # 30% reduction
                cost_reduction_usd=metrics.get('cost_usd', 0) * 0.3,
                implementation_effort="low",
                estimated_hours=0.5,
                prerequisites=["NVIDIA Ampere GPU or newer", "PyTorch >= 1.9"],
                implementation_steps=[
                    "Add fp16=True to TrainingArguments",
                    "Test for numerical stability",
                    "Monitor for any accuracy degradation",
                    "Benchmark performance improvement"
                ],
                code_examples={
                    "training_args": """
training_args = TrainingArguments(
    fp16=True,  # Enable mixed precision
    dataloader_pin_memory=False,  # Reduces memory pressure
    # ... other args
)
""",
                    "alternative_bf16": """
# For newer GPUs, bf16 may be more stable
training_args = TrainingArguments(
    bf16=True,  # Better numerical stability than fp16
    # ... other args
)
"""
                },
                success_metrics=["Training speed increase", "Memory usage reduction", "CO2 reduction"],
                risk_level="low",
                confidence_score=0.9
            ))
        
        # Gradient checkpointing
        if not config.get('gradient_checkpointing', False):
            strategies.append(OptimizationStrategy(
                strategy_id=f"gradient_checkpoint_{int(time.time())}",
                name="Enable Gradient Checkpointing",
                category="model",
                description="Trade computation for memory by recomputing activations during backward pass",
                co2_reduction_kg=metrics.get('co2_kg', 0) * 0.15,  # 15% reduction through better memory efficiency
                cost_reduction_usd=metrics.get('cost_usd', 0) * 0.15,
                implementation_effort="low",
                estimated_hours=0.25,
                prerequisites=["Sufficient compute capacity"],
                implementation_steps=[
                    "Enable gradient_checkpointing in TrainingArguments",
                    "Monitor training speed (may slow down slightly)",
                    "Test with larger batch sizes if memory allows"
                ],
                code_examples={
                    "implementation": """
training_args = TrainingArguments(
    gradient_checkpointing=True,  # Save memory at cost of computation
    # May allow larger batch sizes
    per_device_train_batch_size=32,  # Increase if memory allows
)
"""
                },
                success_metrics=["Memory usage reduction", "Ability to use larger batch sizes"],
                risk_level="low",
                confidence_score=0.8
            ))
        
        # Model pruning suggestion
        if config.get('num_train_epochs', 1) > 3:
            strategies.append(OptimizationStrategy(
                strategy_id=f"model_pruning_{int(time.time())}",
                name="Implement Model Pruning",
                category="model",
                description="Remove less important model parameters to reduce computation",
                co2_reduction_kg=metrics.get('co2_kg', 0) * 0.25,  # 25% reduction
                cost_reduction_usd=metrics.get('cost_usd', 0) * 0.25,
                implementation_effort="high",
                estimated_hours=8.0,
                prerequisites=["PyTorch pruning utilities", "Model analysis tools"],
                implementation_steps=[
                    "Analyze model parameter importance",
                    "Implement structured or unstructured pruning",
                    "Fine-tune pruned model",
                    "Validate accuracy retention"
                ],
                code_examples={
                    "basic_pruning": """
import torch.nn.utils.prune as prune

# Magnitude-based pruning example
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)
"""
                },
                success_metrics=["Model size reduction", "Inference speed improvement", "Training efficiency"],
                risk_level="medium",
                confidence_score=0.7
            ))
        
        return strategies
    
    async def _analyze_hardware_optimizations(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[OptimizationStrategy]:
        """Analyze hardware-level optimization opportunities."""
        strategies = []
        
        # Batch size optimization
        batch_size = config.get('per_device_train_batch_size', 8)
        gpu_utilization = metrics.get('gpu_utilization', 50)
        
        if batch_size < 32 and gpu_utilization < 80:
            strategies.append(OptimizationStrategy(
                strategy_id=f"batch_size_opt_{int(time.time())}",
                name="Optimize Batch Size",
                category="hardware",
                description="Increase batch size to improve GPU utilization and training efficiency",
                co2_reduction_kg=metrics.get('co2_kg', 0) * 0.15,  # 15% reduction
                cost_reduction_usd=metrics.get('cost_usd', 0) * 0.15,
                implementation_effort="low",
                estimated_hours=1.0,
                prerequisites=["Sufficient GPU memory"],
                implementation_steps=[
                    "Gradually increase batch size",
                    "Monitor GPU memory usage",
                    "Adjust learning rate accordingly",
                    "Benchmark training speed"
                ],
                code_examples={
                    "batch_size_scaling": """
# Rule of thumb: scale learning rate with batch size
original_lr = 5e-5
original_batch = 8
new_batch = 32
new_lr = original_lr * (new_batch / original_batch)

training_args = TrainingArguments(
    per_device_train_batch_size=32,  # Increased from 8
    learning_rate=new_lr,  # Scale learning rate
)
"""
                },
                success_metrics=["GPU utilization improvement", "Training speed increase"],
                risk_level="low",
                confidence_score=0.85
            ))
        
        # Multi-GPU optimization
        num_gpus = config.get('num_gpus', 1)
        if num_gpus == 1 and metrics.get('model_size_gb', 0) < 10:
            strategies.append(OptimizationStrategy(
                strategy_id=f"multi_gpu_{int(time.time())}",
                name="Consider Multi-GPU Training",
                category="hardware",
                description="Distribute training across multiple GPUs for faster training",
                co2_reduction_kg=metrics.get('co2_kg', 0) * 0.4,  # 40% reduction through faster training
                cost_reduction_usd=metrics.get('cost_usd', 0) * 0.2,
                implementation_effort="medium",
                estimated_hours=4.0,
                prerequisites=["Multiple GPUs available", "Distributed training setup"],
                implementation_steps=[
                    "Set up distributed training configuration",
                    "Test with DataParallel or DistributedDataParallel",
                    "Optimize communication overhead",
                    "Benchmark scaling efficiency"
                ],
                code_examples={
                    "simple_data_parallel": """
# Simple DataParallel approach
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Or use HF Trainer's built-in distributed training
training_args = TrainingArguments(
    dataloader_num_workers=4,
    ddp_backend="nccl",  # For multi-GPU
)
"""
                },
                success_metrics=["Training time reduction", "GPU utilization across devices"],
                risk_level="medium",
                confidence_score=0.75
            ))
        
        return strategies
    
    async def _analyze_scheduling_optimizations(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[OptimizationStrategy]:
        """Analyze scheduling optimization opportunities."""
        strategies = []
        
        current_hour = datetime.now().hour
        
        # Off-peak training
        if 8 <= current_hour <= 18:  # During business hours
            strategies.append(OptimizationStrategy(
                strategy_id=f"off_peak_{int(time.time())}",
                name="Schedule Training During Off-Peak Hours",
                category="scheduling",
                description="Run training during off-peak hours for cleaner energy and lower carbon intensity",
                co2_reduction_kg=metrics.get('co2_kg', 0) * 0.25,  # 25% reduction
                cost_reduction_usd=metrics.get('cost_usd', 0) * 0.15,  # Lower electricity costs
                implementation_effort="low",
                estimated_hours=1.0,
                prerequisites=["Job scheduling system", "Flexible training schedule"],
                implementation_steps=[
                    "Set up automated job scheduling",
                    "Identify optimal low-carbon time windows",
                    "Configure delayed training starts",
                    "Monitor grid carbon intensity"
                ],
                code_examples={
                    "cron_scheduling": """
# Example cron job for 2 AM training start
# 0 2 * * * /path/to/training_script.py

# Or use Python scheduling
import schedule
import time

def start_training():
    # Your training code here
    pass

# Schedule training at 2 AM daily
schedule.every().day.at("02:00").do(start_training)
"""
                },
                success_metrics=["Carbon intensity reduction", "Cost savings", "Grid load balancing"],
                risk_level="low",
                confidence_score=0.9
            ))
        
        return strategies
    
    async def _analyze_environmental_optimizations(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[OptimizationStrategy]:
        """Analyze environmental optimization opportunities."""
        strategies = []
        
        # Carbon offset strategy
        co2_emissions = metrics.get('co2_kg', 0)
        if co2_emissions > 5.0:  # Significant emissions
            strategies.append(OptimizationStrategy(
                strategy_id=f"carbon_offset_{int(time.time())}",
                name="Carbon Offset Investment",
                category="environmental",
                description="Invest in verified carbon offset projects to neutralize training emissions",
                co2_reduction_kg=co2_emissions,  # Full offset
                cost_reduction_usd=0,  # This is an investment, not a cost reduction
                implementation_effort="low",
                estimated_hours=2.0,
                prerequisites=["Budget allocation", "Verified offset provider"],
                implementation_steps=[
                    "Calculate total carbon footprint",
                    "Select verified offset projects",
                    "Purchase carbon credits",
                    "Track and report offset investments"
                ],
                code_examples={
                    "offset_calculation": """
# Calculate offset cost
co2_emissions_kg = 25.5  # From training
offset_cost_per_ton = 25  # USD per ton CO2
offset_cost = (co2_emissions_kg / 1000) * offset_cost_per_ton
print(f"Offset cost: ${offset_cost:.2f}")
"""
                },
                success_metrics=["Carbon neutrality achievement", "Offset project support"],
                risk_level="low",
                confidence_score=0.95
            ))
        
        # Renewable energy migration
        strategies.append(OptimizationStrategy(
            strategy_id=f"renewable_energy_{int(time.time())}",
            name="Migrate to Renewable Energy Providers",
            category="environmental",
            description="Switch to cloud providers or regions with higher renewable energy usage",
            co2_reduction_kg=metrics.get('co2_kg', 0) * 0.6,  # Up to 60% reduction
            cost_reduction_usd=0,  # May have neutral cost
            implementation_effort="medium",
            estimated_hours=6.0,
            prerequisites=["Cloud provider research", "Data migration planning"],
            implementation_steps=[
                "Research renewable energy commitments of cloud providers",
                "Identify regions with cleanest energy mix",
                "Plan workload migration",
                "Implement gradual transition"
            ],
            code_examples={
                "cloud_regions": """
# Example: AWS regions with high renewable energy
CLEAN_REGIONS = {
    'us-west-2': 'Oregon - 50% renewable',
    'eu-north-1': 'Stockholm - 95% renewable', 
    'ca-central-1': 'Canada Central - 80% renewable'
}

# Configure training in clean regions
training_config = {
    'preferred_regions': ['eu-north-1', 'ca-central-1'],
    'fallback_regions': ['us-west-2']
}
"""
            },
            success_metrics=["Renewable energy percentage", "Carbon intensity reduction"],
            risk_level="medium",
            confidence_score=0.8
        ))
        
        return strategies
    
    async def _analyze_behavioral_optimizations(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[OptimizationStrategy]:
        """Analyze behavioral optimization opportunities."""
        strategies = []
        
        # Early stopping strategy
        if not config.get('early_stopping_enabled', False):
            strategies.append(OptimizationStrategy(
                strategy_id=f"early_stopping_{int(time.time())}",
                name="Implement Intelligent Early Stopping",
                category="behavioral",
                description="Stop training automatically when improvement plateaus to avoid wasted computation",
                co2_reduction_kg=metrics.get('co2_kg', 0) * 0.2,  # 20% reduction
                cost_reduction_usd=metrics.get('cost_usd', 0) * 0.2,
                implementation_effort="low",
                estimated_hours=1.5,
                prerequisites=["Validation dataset", "EarlyStoppingCallback"],
                implementation_steps=[
                    "Define early stopping criteria",
                    "Configure patience and minimum delta",
                    "Set up validation monitoring",
                    "Test stopping behavior"
                ],
                code_examples={
                    "early_stopping": """
from transformers import EarlyStoppingCallback

# Add early stopping to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,  # Stop after 3 epochs without improvement
            early_stopping_threshold=0.001  # Minimum improvement threshold
        )
    ]
)
"""
                },
                success_metrics=["Reduced unnecessary training", "Faster convergence detection"],
                risk_level="low",
                confidence_score=0.85
            ))
        
        # Hyperparameter optimization
        strategies.append(OptimizationStrategy(
            strategy_id=f"hyperopt_{int(time.time())}",
            name="Carbon-Aware Hyperparameter Optimization",
            category="behavioral",
            description="Use efficient hyperparameter search methods that minimize total carbon footprint",
            co2_reduction_kg=metrics.get('co2_kg', 0) * 0.3,  # 30% reduction
            cost_reduction_usd=metrics.get('cost_usd', 0) * 0.3,
            implementation_effort="high",
            estimated_hours=12.0,
            prerequisites=["Hyperparameter optimization library", "Carbon tracking integration"],
            implementation_steps=[
                "Define carbon-efficiency metric",
                "Implement Bayesian optimization",
                "Configure multi-objective optimization",
                "Monitor carbon cost per experiment"
            ],
            code_examples={
                "carbon_aware_optim": """
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Train model and get both accuracy and carbon metrics
    accuracy, carbon_kg = train_model(lr, batch_size)
    
    # Multi-objective: maximize accuracy, minimize carbon
    # Normalize and combine objectives
    carbon_score = 1.0 / (1.0 + carbon_kg)  # Lower carbon = higher score
    combined_score = 0.7 * accuracy + 0.3 * carbon_score
    
    return combined_score
"""
            },
            success_metrics=["Optimal accuracy/carbon trade-off", "Reduced hyperparameter search time"],
            risk_level="medium",
            confidence_score=0.75
        ))
        
        return strategies
    
    def create_carbon_budget(
        self,
        name: str,
        budget_kg: float,
        period_days: int,
        alert_threshold: float = 80.0
    ) -> CarbonBudget:
        """Create a new carbon budget.
        
        Args:
            name: Budget name
            budget_kg: Total carbon budget in kg CO2
            period_days: Budget period in days
            alert_threshold: Alert threshold percentage (default 80%)
            
        Returns:
            Created carbon budget
        """
        budget = CarbonBudget(
            budget_id=f"budget_{int(time.time())}",
            name=name,
            total_budget_kg=budget_kg,
            used_budget_kg=0.0,
            remaining_budget_kg=budget_kg,
            period_start=datetime.now(),
            period_end=datetime.now() + timedelta(days=period_days),
            alert_threshold_percentage=alert_threshold
        )
        
        self.carbon_budgets.append(budget)
        logger.info(f"Created carbon budget '{name}': {budget_kg} kg CO₂ for {period_days} days")
        return budget
    
    def update_carbon_budget(self, budget_id: str, co2_used: float) -> bool:
        """Update carbon budget with new usage.
        
        Args:
            budget_id: Budget ID to update
            co2_used: CO2 usage in kg to add to budget
            
        Returns:
            True if budget is still within limits, False if exceeded
        """
        budget = next((b for b in self.carbon_budgets if b.budget_id == budget_id), None)
        if not budget:
            logger.error(f"Budget {budget_id} not found")
            return False
        
        budget.used_budget_kg += co2_used
        budget.remaining_budget_kg = budget.total_budget_kg - budget.used_budget_kg
        
        utilization = budget.utilization_percentage
        
        if utilization >= 100:
            logger.error(f"Carbon budget '{budget.name}' EXCEEDED: {utilization:.1f}% used")
            return False
        elif utilization >= budget.alert_threshold_percentage:
            logger.warning(f"Carbon budget '{budget.name}' alert: {utilization:.1f}% used")
        
        logger.info(f"Carbon budget '{budget.name}' updated: {utilization:.1f}% used")
        return True
    
    def get_sustainability_dashboard(self) -> Dict[str, Any]:
        """Generate sustainability dashboard data."""
        # Calculate overall progress
        achieved_goals = [g for g in self.goals if g.is_achieved]
        avg_progress = np.mean([g.progress_percentage for g in self.goals]) if self.goals else 0
        
        # Calculate total impact of strategies
        total_co2_reduction = sum(s.co2_reduction_kg for s in self.strategies)
        total_cost_savings = sum(s.cost_reduction_usd for s in self.strategies)
        
        # Budget status
        budget_status = []
        for budget in self.carbon_budgets:
            budget_status.append({
                "name": budget.name,
                "utilization": budget.utilization_percentage,
                "days_remaining": budget.days_remaining,
                "daily_burn_rate": budget.daily_burn_rate_kg,
                "projected_end_usage": budget.projected_end_usage_kg
            })
        
        return {
            "dashboard_id": f"sustainability_{int(time.time())}",
            "generated_at": datetime.now().isoformat(),
            "overview": {
                "total_goals": len(self.goals),
                "achieved_goals": len(achieved_goals),
                "average_progress": round(avg_progress, 1),
                "total_strategies": len(self.strategies),
                "potential_co2_reduction_kg": round(total_co2_reduction, 2),
                "potential_cost_savings_usd": round(total_cost_savings, 2)
            },
            "goals_status": [
                {
                    "name": goal.name,
                    "progress": goal.progress_percentage,
                    "achieved": goal.is_achieved,
                    "priority": goal.priority
                }
                for goal in self.goals
            ],
            "top_strategies": [
                {
                    "name": strategy.name,
                    "category": strategy.category,
                    "co2_reduction_kg": strategy.co2_reduction_kg,
                    "effort": strategy.implementation_effort,
                    "confidence": strategy.confidence_score
                }
                for strategy in sorted(self.strategies, key=lambda s: s.co2_reduction_kg, reverse=True)[:10]
            ],
            "budget_status": budget_status,
            "sustainability_score": min(100, avg_progress + (total_co2_reduction * 0.1))  # Composite score
        }
    
    def export_sustainability_report(self, filepath: Path) -> None:
        """Export comprehensive sustainability report."""
        dashboard = self.get_sustainability_dashboard()
        
        report = {
            "sustainability_report": dashboard,
            "goals": [asdict(goal) for goal in self.goals],
            "strategies": [asdict(strategy) for strategy in self.strategies],
            "carbon_budgets": [asdict(budget) for budget in self.carbon_budgets],
            "optimization_history": self.optimization_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Sustainability report exported to {filepath}")


# Integration with existing callback system
class SustainabilityCallback:
    """Callback to integrate sustainability optimization with HF Trainer."""
    
    def __init__(self, optimizer: Optional[SustainabilityOptimizer] = None):
        """Initialize sustainability callback."""
        self.optimizer = optimizer or SustainabilityOptimizer()
        self.session_start_time = datetime.now()
        self.total_session_co2 = 0.0
        
        # Create default monthly budget
        self.monthly_budget = self.optimizer.create_carbon_budget(
            name="Monthly Training Budget",
            budget_kg=50.0,  # 50 kg CO2 per month
            period_days=30
        )
    
    async def on_training_start(self, training_config: Dict[str, Any]) -> None:
        """Called when training starts."""
        self.session_start_time = datetime.now()
        logger.info("Sustainability optimization activated")
        
        # Generate optimization strategies
        strategies = await self.optimizer.analyze_sustainability_opportunities(
            {"co2_kg": 0, "cost_usd": 0},  # Initial estimates
            training_config
        )
        
        if strategies:
            logger.info(f"Found {len(strategies)} sustainability optimization opportunities:")
            for strategy in strategies[:3]:  # Show top 3
                logger.info(f"  • {strategy.name} (Impact: {strategy.co2_reduction_kg:.2f} kg CO₂)")
    
    async def on_epoch_end(self, epoch_metrics: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        epoch_co2 = epoch_metrics.get('epoch_co2_kg', 0)
        if epoch_co2 > 0:
            self.total_session_co2 += epoch_co2
            
            # Update carbon budget
            self.optimizer.update_carbon_budget(self.monthly_budget.budget_id, epoch_co2)
    
    async def on_training_end(self, final_metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
        duration = (datetime.now() - self.session_start_time).total_seconds() / 3600
        
        # Final sustainability analysis
        final_metrics['training_duration_hours'] = duration
        final_metrics['total_co2_kg'] = self.total_session_co2
        
        # Generate final sustainability report
        dashboard = self.optimizer.get_sustainability_dashboard()
        
        logger.info("\n" + "="*60)
        logger.info("SUSTAINABILITY OPTIMIZATION REPORT")
        logger.info("="*60)
        logger.info(f"Sustainability Score: {dashboard['sustainability_score']:.1f}/100")
        logger.info(f"Total CO₂ Reduction Potential: {dashboard['overview']['potential_co2_reduction_kg']:.2f} kg")
        logger.info(f"Total Cost Savings Potential: ${dashboard['overview']['potential_cost_savings_usd']:.2f}")
        
        if dashboard['top_strategies']:
            logger.info("\nTop Optimization Opportunities:")
            for strategy in dashboard['top_strategies'][:5]:
                logger.info(f"  • {strategy['name']} ({strategy['co2_reduction_kg']:.2f} kg CO₂ reduction)")
        
        logger.info("="*60)
    
    def get_optimizer(self) -> SustainabilityOptimizer:
        """Get the sustainability optimizer for direct access."""
        return self.optimizer