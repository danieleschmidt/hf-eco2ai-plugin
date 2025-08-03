"""Configuration management for HF Eco2AI Plugin."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import json
from pathlib import Path


@dataclass
class CarbonConfig:
    """Configuration for carbon tracking during training.
    
    This class handles all configuration options for the Eco2AI callback,
    including tracking parameters, export settings, and regional data.
    """
    
    # Project identification
    project_name: str = "hf-training"
    experiment_name: Optional[str] = None
    run_id: Optional[str] = None
    
    # Geographic settings for carbon intensity
    country: str = "USA"
    region: str = "California"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    grid_carbon_intensity: Optional[float] = None  # g CO₂/kWh override
    
    # GPU monitoring configuration
    gpu_ids: Union[List[int], str] = "auto"
    track_gpu_energy: bool = True
    aggregate_gpus: bool = True
    per_gpu_metrics: bool = False
    gpu_sampling_interval: float = 1.0  # seconds
    
    # Logging configuration
    log_level: str = "EPOCH"  # "STEP", "EPOCH", "BATCH"
    log_to_console: bool = True
    console_log_level: str = "INFO"
    
    # Export and storage
    export_prometheus: bool = False
    prometheus_port: int = 9091
    prometheus_host: str = "localhost"
    save_report: bool = True
    report_path: str = "carbon_report.json"
    export_csv: bool = False
    csv_path: str = "carbon_metrics.csv"
    
    # Real-time features
    use_real_time_carbon: bool = True
    auto_detect_location: bool = False
    update_interval: int = 3600  # seconds for carbon intensity updates
    
    # Training optimization
    enable_carbon_budget: bool = False
    max_co2_kg: Optional[float] = None
    budget_action: str = "warn"  # "warn", "stop"
    check_frequency: int = 100  # steps
    
    # Integration settings
    mlflow_tracking: bool = False
    wandb_tracking: bool = False
    custom_tags: Dict[str, str] = field(default_factory=dict)
    
    # Advanced features
    enable_recommendations: bool = True
    track_model_size: bool = True
    track_dataset_size: bool = True
    estimate_costs: bool = False
    electricity_price_per_kwh: float = 0.12  # USD
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_paths()
        self._detect_environment()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.log_level not in ["STEP", "EPOCH", "BATCH"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
        
        if self.budget_action not in ["warn", "stop"]:
            raise ValueError(f"Invalid budget_action: {self.budget_action}")
        
        if self.gpu_sampling_interval <= 0:
            raise ValueError("gpu_sampling_interval must be positive")
        
        if self.enable_carbon_budget and self.max_co2_kg is None:
            raise ValueError("max_co2_kg must be set when enable_carbon_budget is True")
        
        if isinstance(self.gpu_ids, str) and self.gpu_ids != "auto":
            raise ValueError("gpu_ids must be 'auto' or a list of integers")
    
    def _setup_paths(self):
        """Ensure output directories exist."""
        if self.save_report:
            Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
        
        if self.export_csv:
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _detect_environment(self):
        """Detect environment settings automatically."""
        # Auto-detect experiment name from environment
        if self.experiment_name is None:
            self.experiment_name = os.getenv("EXPERIMENT_NAME", self.project_name)
        
        # Auto-detect run ID from MLflow or other tracking systems
        if self.run_id is None:
            self.run_id = os.getenv("MLFLOW_RUN_ID") or os.getenv("WANDB_RUN_ID")
        
        # Enable tracking if environment variables are set
        if os.getenv("MLFLOW_TRACKING_URI"):
            self.mlflow_tracking = True
        
        if os.getenv("WANDB_PROJECT"):
            self.wandb_tracking = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Export configuration to JSON."""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, default=str)
        
        if path:
            Path(path).write_text(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CarbonConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> "CarbonConfig":
        """Load configuration from JSON file."""
        config_dict = json.loads(Path(path).read_text())
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> "CarbonConfig":
        """Create configuration from environment variables."""
        config_dict = {}
        
        # Map environment variables to config fields
        env_mapping = {
            "HF_ECO2AI_PROJECT_NAME": "project_name",
            "HF_ECO2AI_COUNTRY": "country",
            "HF_ECO2AI_REGION": "region",
            "HF_ECO2AI_GPU_IDS": "gpu_ids",
            "HF_ECO2AI_LOG_LEVEL": "log_level",
            "HF_ECO2AI_EXPORT_PROMETHEUS": "export_prometheus",
            "HF_ECO2AI_PROMETHEUS_PORT": "prometheus_port",
            "HF_ECO2AI_SAVE_REPORT": "save_report",
            "HF_ECO2AI_REPORT_PATH": "report_path",
            "HF_ECO2AI_MAX_CO2_KG": "max_co2_kg",
        }
        
        for env_var, field_name in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion based on field type
                field_type = cls.__dataclass_fields__[field_name].type
                if field_type == bool:
                    value = value.lower() in ("true", "1", "yes", "on")
                elif field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)
                elif field_type == List[int]:
                    value = [int(x.strip()) for x in value.split(",")]
                
                config_dict[field_name] = value
        
        return cls(**config_dict)
    
    def validate_environment(self) -> List[str]:
        """Validate that required dependencies are available."""
        missing_deps = []
        
        try:
            import eco2ai
        except ImportError:
            missing_deps.append("eco2ai>=2.0.0")
        
        try:
            import pynvml
        except ImportError:
            missing_deps.append("pynvml>=11.5.0")
        
        if self.export_prometheus:
            try:
                import prometheus_client
            except ImportError:
                missing_deps.append("prometheus-client>=0.20.0")
        
        if self.mlflow_tracking:
            try:
                import mlflow
            except ImportError:
                missing_deps.append("mlflow>=2.0.0")
        
        if self.wandb_tracking:
            try:
                import wandb
            except ImportError:
                missing_deps.append("wandb>=0.15.0")
        
        return missing_deps