"""Main Hugging Face Trainer callback for carbon tracking."""

import time
import logging
from typing import Dict, Optional, Any, List
import uuid
from pathlib import Path

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from .config import CarbonConfig
from .models import CarbonMetrics, CarbonReport, CarbonSummary, EnvironmentalImpact
from .monitoring import EnergyTracker
from .exporters import PrometheusExporter, ReportExporter
from .error_handling import get_error_handler, resilient_operation, handle_gracefully, ErrorSeverity
from .validation import CarbonTrackingValidator as DataValidator
from .health_monitor import get_health_monitor, start_health_monitoring
from .performance_optimizer import get_performance_optimizer, optimized, start_performance_optimization

logger = logging.getLogger(__name__)


class Eco2AICallback(TrainerCallback):
    """Hugging Face Trainer callback for carbon tracking and reporting.
    
    This callback integrates with the HF Trainer lifecycle to monitor energy
    consumption, calculate CO₂ emissions, and generate comprehensive reports.
    
    Usage:
        trainer = Trainer(
            model=model,
            args=training_args,
            callbacks=[Eco2AICallback()]
        )
        
        # Or with custom configuration
        config = CarbonConfig(gpu_ids=[0,1], export_prometheus=True)
        trainer = Trainer(
            model=model,
            args=training_args, 
            callbacks=[Eco2AICallback(config=config)]
        )
    """
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize the carbon tracking callback.
        
        Args:
            config: Configuration for carbon tracking. If None, uses defaults.
        """
        self.config = config or CarbonConfig()
        
        # Initialize error handling, validation, health monitoring, and performance optimization
        self.error_handler = get_error_handler()
        self.validator = DataValidator()
        self.health_monitor = get_health_monitor()
        self.performance_optimizer = get_performance_optimizer()
        
        # Start health monitoring if not already started
        if hasattr(self.config, 'enable_health_monitoring') and self.config.enable_health_monitoring:
            start_health_monitoring()
        
        # Start performance optimization
        if hasattr(self.config, 'enable_performance_optimization') and self.config.enable_performance_optimization:
            start_performance_optimization()
            # Optimize for expected load
            load = getattr(self.config, 'expected_load', 'medium')
            self.performance_optimizer.optimize_for_scale(load)
        
        # Validate environment and dependencies
        try:
            missing_deps = self.config.validate_environment()
            if missing_deps:
                self.error_handler.handle_error(
                    ImportError(f"Missing dependencies: {missing_deps}"),
                    {"dependencies": missing_deps},
                    ErrorSeverity.MEDIUM
                )
                logger.warning(f"Missing dependencies: {missing_deps}")
                logger.warning("Some features may not work correctly")
        except Exception as e:
            self.error_handler.handle_error(e, {"phase": "initialization"}, ErrorSeverity.HIGH)
        
        # Initialize tracking components
        self.energy_tracker = EnergyTracker(
            gpu_ids=self.config.gpu_ids,
            sampling_interval=self.config.gpu_sampling_interval,
            country=self.config.country,
            region=self.config.region
        )
        
        # Initialize exporters
        self.prometheus_exporter = None
        if self.config.export_prometheus:
            try:
                self.prometheus_exporter = PrometheusExporter(
                    port=self.config.prometheus_port,
                    host=self.config.prometheus_host
                )
            except Exception as e:
                logger.error(f"Failed to initialize Prometheus exporter: {e}")
        
        self.report_exporter = ReportExporter()
        
        # Initialize report
        self.carbon_report = CarbonReport(
            report_id=str(uuid.uuid4()),
            config=self.config.to_dict()
        )
        
        # Tracking state
        self._training_start_time = None
        self._last_step_time = None
        self._samples_processed = 0
        self._current_epoch = 0
        self._current_step = 0
        
        # Integration tracking
        self._mlflow_client = None
        self._wandb_run = None
        
        if self.config.mlflow_tracking:
            self._setup_mlflow()
        
        if self.config.wandb_tracking:
            self._setup_wandb()
        
        logger.info(f"Initialized Eco2AI callback for project: {self.config.project_name}")
    
    def _setup_mlflow(self):
        """Setup MLflow integration."""
        try:
            import mlflow
            self._mlflow_client = mlflow.tracking.MlflowClient()
            logger.info("MLflow integration enabled")
        except ImportError:
            logger.warning("MLflow not available despite mlflow_tracking=True")
    
    def _setup_wandb(self):
        """Setup Weights & Biases integration."""
        try:
            import wandb
            self._wandb_run = wandb.run
            if self._wandb_run is None:
                logger.warning("No active wandb run found")
            else:
                logger.info("Weights & Biases integration enabled")
        except ImportError:
            logger.warning("wandb not available despite wandb_tracking=True")
    
    @resilient_operation(max_attempts=3, circuit_breaker=True)
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                       control: TrainerControl, model=None, **kwargs):
        """Called at the beginning of training."""
        try:
            # Validate training arguments
            validation_result = self.validator.validate_training_args(args)
            if not validation_result.get("valid", True):
                logger.warning(f"Training args validation warning: {validation_result.get('message')}")
            
            self._training_start_time = time.time()
            self._last_step_time = self._training_start_time
        except Exception as e:
            self.error_handler.handle_error(e, {"phase": "train_begin"}, ErrorSeverity.HIGH)
            raise
        
        # Start energy tracking
        if self.energy_tracker.is_available():
            self.energy_tracker.start_tracking()
            logger.info("Started carbon tracking for training")
        else:
            logger.warning("Energy tracking not available - using estimation mode")
        
        # Initialize training metadata
        self.carbon_report.training_metadata.update({
            "project_name": self.config.project_name,
            "experiment_name": self.config.experiment_name,
            "run_id": self.config.run_id,
            "model_name": getattr(model, "__class__", {}).get("__name__", "unknown"),
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "training_args": args.to_dict(),
        })
        
        # Count model parameters
        if model is not None:
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.carbon_report.training_metadata.update({
                    "model_parameters": total_params,
                    "trainable_parameters": trainable_params,
                })
                self.carbon_report.summary.model_parameters = total_params
            except Exception as e:
                logger.warning(f"Could not count model parameters: {e}")
        
        # Set up system metadata
        import platform
        import sys
        self.carbon_report.system_metadata.update({
            "python_version": sys.version,
            "platform": platform.platform(),
            "gpu_available": self.energy_tracker.is_available(),
            "region": f"{self.config.country}/{self.config.region}",
            "carbon_intensity": self.energy_tracker.carbon_provider.get_carbon_intensity(),
            "renewable_percentage": self.energy_tracker.carbon_provider.get_renewable_percentage(),
        })
        
        if self.config.log_to_console:
            logger.info(f"Training started - tracking carbon for {self.config.country}/{self.config.region}")
            logger.info(f"Grid carbon intensity: {self.carbon_report.system_metadata['carbon_intensity']:.0f} g CO₂/kWh")
    
    @handle_gracefully(severity=ErrorSeverity.MEDIUM, fallback_value=None)
    def on_step_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Called at the end of each training step."""
        self._current_step = state.global_step
        current_time = time.time()
        
        # Calculate samples processed (estimate based on batch size)
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if hasattr(args, 'world_size'):
            batch_size *= args.world_size
        
        self._samples_processed += batch_size
        
        # Only log on specified intervals
        should_log = False
        if self.config.log_level == "STEP":
            should_log = True
        elif self.config.log_level == "BATCH" and state.global_step % args.logging_steps == 0:
            should_log = True
        
        if should_log:
            self._log_metrics(logs, current_time)
        
        # Check carbon budget if enabled
        if self.config.enable_carbon_budget and state.global_step % self.config.check_frequency == 0:
            self._check_carbon_budget(control)
        
        self._last_step_time = current_time
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Called at the end of each epoch."""
        self._current_epoch = state.epoch
        current_time = time.time()
        
        # Always log at epoch end
        self._log_metrics(logs, current_time, is_epoch_end=True)
        
        if self.config.log_to_console:
            power, energy, co2 = self.energy_tracker.get_current_consumption()
            logger.info(f"Epoch {state.epoch:.0f} - Energy: {energy:.3f} kWh, CO₂: {co2:.3f} kg, Power: {power:.0f} W")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Called at the end of training."""
        self.energy_tracker.stop_tracking()
        
        # Final metrics logging
        current_time = time.time()
        self._log_metrics(logs, current_time, is_train_end=True)
        
        # Calculate final summary
        self._finalize_report()
        
        # Export reports
        if self.config.save_report:
            self.report_exporter.export_json(self.carbon_report, self.config.report_path)
        
        if self.config.export_csv:
            self.report_exporter.export_csv(self.carbon_report, self.config.csv_path)
        
        # Log final summary
        if self.config.log_to_console:
            print("\n" + "="*50)
            print("CARBON TRACKING SUMMARY")
            print("="*50)
            print(self.carbon_report.summary_text())
            print("="*50)
        
        logger.info("Carbon tracking completed")
    
    @optimized(cache_ttl=10.0)  # Cache for 10 seconds for rapid step logging
    @handle_gracefully(severity=ErrorSeverity.LOW, fallback_value=None)
    def _log_metrics(self, logs: Optional[Dict[str, float]], timestamp: float, 
                     is_epoch_end: bool = False, is_train_end: bool = False):
        """Log current carbon metrics."""
        # Get current energy consumption
        power, energy, co2 = self.energy_tracker.get_current_consumption()
        
        # Calculate step-specific metrics
        step_duration = timestamp - self._last_step_time if self._last_step_time else 0
        step_energy = 0
        step_co2 = 0
        
        if len(self.carbon_report.detailed_metrics) > 0:
            last_metric = self.carbon_report.detailed_metrics[-1]
            step_energy = energy - last_metric.cumulative_energy_kwh
            step_co2 = co2 - last_metric.cumulative_co2_kg
        
        # Get efficiency metrics
        efficiency = self.energy_tracker.get_efficiency_metrics(self._samples_processed)
        
        # Create metric record
        metric = CarbonMetrics(
            timestamp=timestamp,
            step=self._current_step,
            epoch=self._current_epoch,
            energy_kwh=step_energy,
            cumulative_energy_kwh=energy,
            power_watts=power,
            co2_kg=step_co2,
            cumulative_co2_kg=co2,
            grid_intensity=self.energy_tracker.carbon_provider.get_carbon_intensity(),
            samples_processed=self._samples_processed,
            samples_per_kwh=efficiency["samples_per_kwh"],
            duration_seconds=step_duration,
            model_parameters=self.carbon_report.summary.model_parameters,
            location=f"{self.config.country}/{self.config.region}",
        )
        
        # Add training context from logs
        if logs:
            metric.loss = logs.get("train_loss") or logs.get("loss")
            metric.learning_rate = logs.get("learning_rate")
        
        # Add to report
        self.carbon_report.add_metric(metric)
        
        # Export to Prometheus
        if self.prometheus_exporter:
            self.prometheus_exporter.record_metrics(metric)
        
        # Export to MLflow
        if self._mlflow_client and logs:
            mlflow_metrics = {
                "carbon/energy_kwh": energy,
                "carbon/co2_kg": co2,
                "carbon/power_watts": power,
                "carbon/samples_per_kwh": efficiency["samples_per_kwh"],
                "carbon/grid_intensity": metric.grid_intensity,
            }
            
            try:
                import mlflow
                for key, value in mlflow_metrics.items():
                    mlflow.log_metric(key, value, step=self._current_step)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        # Export to wandb
        if self._wandb_run and logs:
            wandb_metrics = {
                "carbon/energy_kwh": energy,
                "carbon/co2_kg": co2,
                "carbon/power_watts": power,
                "carbon/samples_per_kwh": efficiency["samples_per_kwh"],
                "carbon/efficiency_percentile": 75,  # Would calculate based on benchmarks
            }
            
            try:
                self._wandb_run.log(wandb_metrics, step=self._current_step)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
    
    def _check_carbon_budget(self, control: TrainerControl):
        """Check if carbon budget has been exceeded."""
        if not self.config.max_co2_kg:
            return
        
        _, _, current_co2 = self.energy_tracker.get_current_consumption()
        
        if current_co2 > self.config.max_co2_kg:
            message = f"Carbon budget exceeded! Current: {current_co2:.2f} kg, Budget: {self.config.max_co2_kg:.2f} kg"
            
            if self.config.budget_action == "stop":
                logger.error(message + " - Stopping training")
                control.should_training_stop = True
            else:
                logger.warning(message)
    
    def _finalize_report(self):
        """Finalize the carbon report with summary statistics."""
        if not self.carbon_report.detailed_metrics:
            return
        
        # Calculate duration
        if self._training_start_time:
            total_duration = time.time() - self._training_start_time
            self.carbon_report.summary.total_duration_hours = total_duration / 3600
        
        # Set final values
        if self.carbon_report.detailed_metrics:
            final_metric = self.carbon_report.detailed_metrics[-1]
            self.carbon_report.summary.final_loss = final_metric.loss
        
        # Calculate environmental equivalents
        self.carbon_report.summary.calculate_equivalents()
        
        # Set environmental impact
        self.carbon_report.environmental_impact = EnvironmentalImpact(
            renewable_percentage=self.energy_tracker.carbon_provider.get_renewable_percentage(),
            fossil_fuel_percentage=100 - self.energy_tracker.carbon_provider.get_renewable_percentage(),
            region_name=f"{self.config.country}/{self.config.region}",
            country_code=self.config.country,
            regional_average_intensity=self.energy_tracker.carbon_provider.get_carbon_intensity(),
        )
        
        # Generate optimization recommendations
        self.carbon_report.generate_recommendations()
        
        # Calculate costs if enabled
        if self.config.estimate_costs:
            energy_cost = self.carbon_report.summary.total_energy_kwh * self.config.electricity_price_per_kwh
            carbon_credit_cost = self.carbon_report.summary.total_co2_kg * 15.0  # $15/tonne CO₂
            
            self.carbon_report.summary.estimated_cost_usd = energy_cost
            self.carbon_report.summary.carbon_credit_cost_usd = carbon_credit_cost
    
    @optimized(cache_ttl=5.0)  # Cache metrics for 5 seconds
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current carbon tracking metrics.
        
        Returns:
            Dictionary containing current energy and carbon metrics.
        """
        power, energy, co2 = self.energy_tracker.get_current_consumption()
        efficiency = self.energy_tracker.get_efficiency_metrics(self._samples_processed)
        
        return {
            "power_watts": power,
            "energy_kwh": energy,
            "co2_kg": co2,
            "samples_processed": self._samples_processed,
            "samples_per_kwh": efficiency["samples_per_kwh"],
            "energy_per_sample": efficiency["energy_per_sample"],
            "co2_per_sample": efficiency["co2_per_sample"],
            "grid_intensity": self.energy_tracker.carbon_provider.get_carbon_intensity(),
        }
    
    def generate_report(self) -> CarbonReport:
        """Generate and return the current carbon report.
        
        Returns:
            Complete carbon tracking report.
        """
        self._finalize_report()
        return self.carbon_report
    
    @property
    def carbon_summary(self) -> str:
        """Get a human-readable carbon summary."""
        return self.carbon_report.summary_text()


class CarbonBudgetCallback(TrainerCallback):
    """Specialized callback for enforcing carbon budgets during training."""
    
    def __init__(self, max_co2_kg: float, action: str = "warn", check_frequency: int = 100):
        """Initialize carbon budget enforcement.
        
        Args:
            max_co2_kg: Maximum CO₂ emissions allowed (kg)
            action: Action to take when budget exceeded ("warn" or "stop")
            check_frequency: Check budget every N steps
        """
        self.max_co2_kg = max_co2_kg
        self.action = action
        self.check_frequency = check_frequency
        self._eco2ai_callback = None
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                       control: TrainerControl, **kwargs):
        """Find the Eco2AI callback to monitor."""
        # Look for Eco2AI callback in the trainer's callbacks
        trainer = kwargs.get('trainer')
        if trainer:
            for callback in trainer.callback_handler.callbacks:
                if isinstance(callback, Eco2AICallback):
                    self._eco2ai_callback = callback
                    break
        
        if not self._eco2ai_callback:
            raise ValueError("CarbonBudgetCallback requires Eco2AICallback to be present")
        
        logger.info(f"Carbon budget enforcer active: {self.max_co2_kg:.2f} kg CO₂ limit")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Check carbon budget on specified intervals."""
        if state.global_step % self.check_frequency == 0:
            metrics = self._eco2ai_callback.get_current_metrics()
            current_co2 = metrics["co2_kg"]
            
            if current_co2 > self.max_co2_kg:
                message = f"Carbon budget exceeded! Current: {current_co2:.2f} kg, Budget: {self.max_co2_kg:.2f} kg"
                
                if self.action == "stop":
                    logger.error(message + " - Stopping training")
                    control.should_training_stop = True
                else:
                    logger.warning(message)