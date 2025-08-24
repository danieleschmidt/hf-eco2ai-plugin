"""Utility functions for HF Eco2AI Plugin."""

import logging
import time
import hashlib
import platform
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys
import os


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration for carbon tracking.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("hf_eco2ai")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for carbon tracking context.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
    }
    
    # Try to get GPU information
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        
        gpus = []
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpus.append({
                "id": i,
                "name": name,
                "memory_total_mb": memory_info.total // (1024 * 1024),
                "memory_free_mb": memory_info.free // (1024 * 1024),
            })
        
        info["gpus"] = gpus
        info["gpu_count"] = gpu_count
        
    except (ImportError, Exception):
        info["gpus"] = []
        info["gpu_count"] = 0
    
    # Try to get memory information
    try:
        import psutil
        memory = psutil.virtual_memory()
        info["memory_total_gb"] = memory.total // (1024**3)
        info["memory_available_gb"] = memory.available // (1024**3)
    except ImportError:
        info["memory_total_gb"] = None
        info["memory_available_gb"] = None
    
    return info


def generate_run_id(project_name: str, timestamp: Optional[float] = None) -> str:
    """Generate a unique run ID for carbon tracking.
    
    Args:
        project_name: Name of the project
        timestamp: Optional timestamp (uses current time if not provided)
        
    Returns:
        Unique run ID string
    """
    if timestamp is None:
        timestamp = time.time()
    
    # Create hash from project name, timestamp, and system info
    hash_input = f"{project_name}_{timestamp}_{platform.node()}"
    run_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    # Format as readable ID
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
    return f"{project_name}_{time_str}_{run_hash}"


def validate_gpu_ids(gpu_ids: Any, available_gpus: int) -> List[int]:
    """Validate and normalize GPU IDs.
    
    Args:
        gpu_ids: GPU IDs specification (auto, list, or comma-separated string)
        available_gpus: Number of available GPUs
        
    Returns:
        List of valid GPU IDs
        
    Raises:
        ValueError: If GPU IDs are invalid
    """
    if gpu_ids == "auto" or gpu_ids is None:
        return list(range(available_gpus))
    
    if isinstance(gpu_ids, str):
        # Parse comma-separated string
        try:
            gpu_ids = [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid GPU IDs format: {gpu_ids}")
    
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    
    if not isinstance(gpu_ids, list):
        raise ValueError(f"GPU IDs must be a list, string, or 'auto', got: {type(gpu_ids)}")
    
    # Validate GPU IDs are within range
    for gpu_id in gpu_ids:
        if not isinstance(gpu_id, int) or gpu_id < 0 or gpu_id >= available_gpus:
            raise ValueError(f"GPU ID {gpu_id} is invalid. Available GPUs: 0-{available_gpus-1}")
    
    return gpu_ids


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(bytes_value: float) -> str:
    """Format bytes to human-readable string.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def estimate_model_memory_usage(num_parameters: int, precision: str = "fp32") -> Dict[str, float]:
    """Estimate memory usage for a model with given parameters.
    
    Args:
        num_parameters: Number of model parameters
        precision: Model precision (fp32, fp16, bf16, int8)
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Bytes per parameter based on precision
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    
    param_bytes = bytes_per_param.get(precision, 4)
    
    # Model weights
    model_memory_mb = (num_parameters * param_bytes) / (1024 * 1024)
    
    # Gradients (same size as model for training)
    gradient_memory_mb = model_memory_mb
    
    # Optimizer states (depends on optimizer, assume Adam = 2x model size)
    optimizer_memory_mb = model_memory_mb * 2
    
    # Activations (estimate 4x model size for large models)
    activation_memory_mb = model_memory_mb * 4
    
    return {
        "model_mb": model_memory_mb,
        "gradients_mb": gradient_memory_mb,
        "optimizer_mb": optimizer_memory_mb,
        "activations_mb": activation_memory_mb,
        "total_mb": model_memory_mb + gradient_memory_mb + optimizer_memory_mb + activation_memory_mb,
    }


def estimate_training_time(num_samples: int, batch_size: int, epochs: int, 
                          throughput_samples_per_second: float) -> float:
    """Estimate training time based on throughput.
    
    Args:
        num_samples: Number of training samples
        batch_size: Training batch size
        epochs: Number of training epochs
        throughput_samples_per_second: Training throughput
        
    Returns:
        Estimated training time in seconds
    """
    total_samples = num_samples * epochs
    estimated_seconds = total_samples / throughput_samples_per_second
    return estimated_seconds


def calculate_carbon_equivalents(co2_kg: float) -> Dict[str, float]:
    """Calculate environmental equivalents for CO₂ emissions.
    
    Args:
        co2_kg: CO₂ emissions in kilograms
        
    Returns:
        Dictionary with various equivalents
    """
    return {
        "km_driven_car": co2_kg * 1000 / 251,  # g CO₂/km for average car
        "trees_needed_offset": co2_kg / 22,     # kg CO₂/tree/year
        "coal_burned_kg": co2_kg / 2.4,        # kg CO₂/kg coal
        "gasoline_liters": co2_kg / 2.3,       # kg CO₂/liter gasoline
        "flights_short_haul": co2_kg / 200,    # kg CO₂ per short flight
        "days_of_household_electricity": co2_kg / 30,  # kg CO₂/day average household
    }


def get_carbon_intensity_by_time(region: str, hour: int) -> float:
    """Get estimated carbon intensity by time of day for a region.
    
    Args:
        region: Region/country code
        hour: Hour of day (0-23)
        
    Returns:
        Carbon intensity in g CO₂/kWh
    """
    # Simplified time-based carbon intensity patterns
    # In reality, this would use real-time grid data APIs
    
    base_intensities = {
        "USA": 400,
        "Germany": 475, 
        "France": 85,
        "Norway": 20,
        "China": 681,
        "India": 708,
    }
    
    base_intensity = base_intensities.get(region, 475)
    
    # Typical daily pattern: lower at night, higher during day
    # This is simplified - real patterns vary by region and season
    hourly_factors = {
        0: 0.8, 1: 0.75, 2: 0.7, 3: 0.7, 4: 0.75, 5: 0.8,   # Night (low demand)
        6: 0.9, 7: 1.0, 8: 1.1, 9: 1.2, 10: 1.15, 11: 1.1,  # Morning ramp
        12: 1.2, 13: 1.25, 14: 1.3, 15: 1.25, 16: 1.2, 17: 1.15,  # Peak day
        18: 1.3, 19: 1.35, 20: 1.2, 21: 1.1, 22: 1.0, 23: 0.9,    # Evening peak
    }
    
    factor = hourly_factors.get(hour, 1.0)
    return base_intensity * factor


def find_optimal_training_window(duration_hours: float, region: str = "USA") -> Tuple[int, float]:
    """Find optimal time window for training to minimize carbon footprint.
    
    Args:
        duration_hours: Training duration in hours
        region: Region for carbon intensity lookup
        
    Returns:
        Tuple of (optimal_start_hour, carbon_savings_percent)
    """
    # Calculate average carbon intensity for each possible start time
    best_hour = 0
    min_avg_intensity = float('inf')
    
    for start_hour in range(24):
        # Calculate average intensity over the training duration
        total_intensity = 0
        hours_in_window = int(duration_hours) + 1
        
        for i in range(hours_in_window):
            hour = (start_hour + i) % 24
            intensity = get_carbon_intensity_by_time(region, hour)
            total_intensity += intensity
        
        avg_intensity = total_intensity / hours_in_window
        
        if avg_intensity < min_avg_intensity:
            min_avg_intensity = avg_intensity
            best_hour = start_hour
    
    # Calculate savings compared to peak time training
    peak_intensity = max(get_carbon_intensity_by_time(region, h) for h in range(24))
    carbon_savings_percent = (1 - min_avg_intensity / peak_intensity) * 100
    
    return best_hour, carbon_savings_percent


def check_dependencies() -> Dict[str, bool]:
    """Check availability of optional dependencies.
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    dependencies = {}
    
    # Required dependencies
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        dependencies["transformers"] = False
    
    # Optional monitoring dependencies
    try:
        import pynvml
        dependencies["pynvml"] = True
    except ImportError:
        dependencies["pynvml"] = False
    
    try:
        import eco2ai
        dependencies["eco2ai"] = True
    except ImportError:
        dependencies["eco2ai"] = False
    
    try:
        import psutil
        dependencies["psutil"] = True
    except ImportError:
        dependencies["psutil"] = False
    
    # Optional export dependencies
    try:
        import prometheus_client
        dependencies["prometheus_client"] = True
    except ImportError:
        dependencies["prometheus_client"] = False
    
    try:
        import mlflow
        dependencies["mlflow"] = True
    except ImportError:
        dependencies["mlflow"] = False
    
    try:
        import wandb
        dependencies["wandb"] = True
    except ImportError:
        dependencies["wandb"] = False
    
    # Optional visualization dependencies
    try:
        import plotly
        dependencies["plotly"] = True
    except ImportError:
        dependencies["plotly"] = False
    
    try:
        import pandas
        dependencies["pandas"] = True
    except ImportError:
        dependencies["pandas"] = False
    
    return dependencies


def get_git_info() -> Dict[str, Optional[str]]:
    """Get git repository information if available.
    
    Returns:
        Dictionary with git information
    """
    git_info = {
        "commit_hash": None,
        "branch": None,
        "remote_url": None,
        "is_dirty": None,
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], 
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()
        
        # Get branch name
        result = subprocess.run(
            ["git", "branch", "--show-current"], 
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        
        # Get remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"], 
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["remote_url"] = result.stdout.strip()
        
        # Check if repo is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"], 
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["is_dirty"] = bool(result.stdout.strip())
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # Git not available or not in a git repo
        pass
    
    return git_info


def create_example_config() -> Dict[str, Any]:
    """Create an example configuration with explanations.
    
    Returns:
        Example configuration dictionary
    """
    return {
        "project_name": "my-ml-project",
        "country": "USA",
        "region": "California",
        "gpu_ids": "auto",
        "log_level": "EPOCH",
        "export_prometheus": False,
        "save_report": True,
        "report_path": "./carbon_reports/training_report.json",
        "enable_carbon_budget": False,
        "max_co2_kg": 5.0,
        "mlflow_tracking": False,
        "wandb_tracking": False,
        "_comments": {
            "project_name": "Unique identifier for your project",
            "country": "Country for carbon intensity data (USA, Germany, France, etc.)",
            "region": "State/region for more accurate carbon data",
            "gpu_ids": "List of GPU IDs to monitor, or 'auto' for all",
            "log_level": "When to log metrics: STEP, EPOCH, or BATCH",
            "export_prometheus": "Enable Prometheus metrics export for monitoring",
            "save_report": "Save detailed JSON report after training",
            "enable_carbon_budget": "Stop training if CO₂ budget exceeded",
            "max_co2_kg": "Maximum CO₂ emissions allowed (kg)",
            "mlflow_tracking": "Log metrics to MLflow (requires active run)",
            "wandb_tracking": "Log metrics to Weights & Biases",
        }
    }


def benchmark_system_performance() -> Dict[str, float]:
    """Benchmark system performance for energy estimation.
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    import numpy as np
    
    results = {}
    
    # CPU benchmark
    start_time = time.time()
    # Simple matrix multiplication benchmark
    size = 1000
    a = np.random.random((size, size))
    b = np.random.random((size, size))
    c = np.dot(a, b)
    cpu_time = time.time() - start_time
    results["cpu_matrix_mult_time"] = cpu_time
    
    # Memory benchmark
    start_time = time.time()
    large_array = np.random.random((10000, 1000))
    memory_sum = np.sum(large_array)
    memory_time = time.time() - start_time
    results["memory_ops_time"] = memory_time
    
    # GPU benchmark (if available)
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            start_time = time.time()
            
            # Simple GPU tensor operations
            a_gpu = torch.randn(1000, 1000, device=device)
            b_gpu = torch.randn(1000, 1000, device=device)
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            
            gpu_time = time.time() - start_time
            results["gpu_matrix_mult_time"] = gpu_time
        else:
            results["gpu_matrix_mult_time"] = None
    except ImportError:
        results["gpu_matrix_mult_time"] = None
    
    return results