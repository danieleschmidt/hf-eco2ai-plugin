"""Energy and GPU monitoring utilities for carbon tracking."""

import time
import threading
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os

# Optional imports with fallbacks
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

try:
    import eco2ai
    ECO2AI_AVAILABLE = True
except ImportError:
    ECO2AI_AVAILABLE = False
    eco2ai = None
    # Use mock implementation for testing
    from . import mock_eco2ai as eco2ai

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """GPU metrics for a single device."""
    device_id: int
    name: str
    power_watts: float = 0.0
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_c: float = 0.0
    energy_kwh: float = 0.0  # Cumulative energy consumption


@dataclass
class CarbonBudgetForecast:
    """Forecast for carbon budget usage."""
    estimated_completion_co2_kg: float
    budget_utilization_percent: float  
    forecasted_overage_kg: float
    confidence_level: float
    recommendation: str


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    power_watts: float = 0.0
    timestamp: float = 0.0


class GPUMonitor:
    """Monitor GPU energy consumption and performance metrics."""
    
    def __init__(self, gpu_ids: Optional[List[int]] = None, sampling_interval: float = 1.0):
        self.gpu_ids = gpu_ids
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.metrics_history: Dict[int, List[GPUMetrics]] = {}
        self._lock = threading.Lock()
        
        if not PYNVML_AVAILABLE:
            logger.warning("pynvml not available. GPU monitoring will be disabled.")
            return
        
        try:
            pynvml.nvmlInit()
            self._initialize_gpus()
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA ML: {e}")
            raise
    
    def _initialize_gpus(self):
        """Initialize GPU device handles."""
        if not PYNVML_AVAILABLE:
            return
        
        self.device_count = pynvml.nvmlDeviceGetCount()
        
        if self.gpu_ids is None or self.gpu_ids == "auto":
            self.gpu_ids = list(range(self.device_count))
        elif isinstance(self.gpu_ids, str):
            # Parse comma-separated string
            self.gpu_ids = [int(x.strip()) for x in self.gpu_ids.split(",")]
        
        # Validate GPU IDs
        for gpu_id in self.gpu_ids:
            if gpu_id >= self.device_count:
                raise ValueError(f"GPU ID {gpu_id} not available. Only {self.device_count} GPUs detected.")
        
        # Initialize device handles and metrics history
        self.handles = {}
        for gpu_id in self.gpu_ids:
            self.handles[gpu_id] = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.metrics_history[gpu_id] = []
        
        logger.info(f"Initialized GPU monitoring for devices: {self.gpu_ids}")
    
    def get_current_metrics(self) -> Dict[int, GPUMetrics]:
        """Get current metrics for all monitored GPUs."""
        if not PYNVML_AVAILABLE or not self.handles:
            return {}
        
        metrics = {}
        
        for gpu_id in self.gpu_ids:
            try:
                handle = self.handles[gpu_id]
                
                # Get device info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Power consumption
                try:
                    power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except pynvml.NVMLError:
                    power_watts = 0.0
                
                # GPU utilization
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization_percent = utilization.gpu
                except pynvml.NVMLError:
                    utilization_percent = 0.0
                
                # Memory usage
                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used_mb = memory_info.used / (1024 * 1024)
                    memory_total_mb = memory_info.total / (1024 * 1024)
                except pynvml.NVMLError:
                    memory_used_mb = memory_total_mb = 0.0
                
                # Temperature
                try:
                    temperature_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError:
                    temperature_c = 0.0
                
                # Calculate cumulative energy
                energy_kwh = 0.0
                if gpu_id in self.metrics_history and self.metrics_history[gpu_id]:
                    last_metric = self.metrics_history[gpu_id][-1]
                    time_diff_hours = (time.time() - last_metric.timestamp if hasattr(last_metric, 'timestamp') else self.sampling_interval) / 3600
                    energy_increment = (power_watts * time_diff_hours) / 1000
                    energy_kwh = last_metric.energy_kwh + energy_increment
                
                metric = GPUMetrics(
                    device_id=gpu_id,
                    name=name,
                    power_watts=power_watts,
                    utilization_percent=utilization_percent,
                    memory_used_mb=memory_used_mb,
                    memory_total_mb=memory_total_mb,
                    temperature_c=temperature_c,
                    energy_kwh=energy_kwh
                )
                
                # Add timestamp for energy calculation
                setattr(metric, 'timestamp', time.time())
                
                metrics[gpu_id] = metric
                
            except Exception as e:
                logger.warning(f"Failed to get metrics for GPU {gpu_id}: {e}")
                continue
        
        return metrics
    
    def start_monitoring(self, callback: Optional[Callable[[Dict[int, GPUMetrics]], None]] = None):
        """Start continuous GPU monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._callback = callback
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    metrics = self.get_current_metrics()
                    
                    # Store metrics history
                    with self._lock:
                        for gpu_id, metric in metrics.items():
                            self.metrics_history[gpu_id].append(metric)
                    
                    # Call callback if provided
                    if self._callback:
                        self._callback(metrics)
                    
                except Exception as e:
                    logger.error(f"Error in GPU monitoring loop: {e}")
                
                time.sleep(self.sampling_interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started GPU monitoring thread")
    
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.is_monitoring = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped GPU monitoring")
    
    def get_aggregated_metrics(self) -> Tuple[float, float, float]:
        """Get aggregated power, energy, and utilization across all GPUs."""
        current_metrics = self.get_current_metrics()
        
        if not current_metrics:
            return 0.0, 0.0, 0.0
        
        total_power = sum(metric.power_watts for metric in current_metrics.values())
        total_energy = sum(metric.energy_kwh for metric in current_metrics.values())
        avg_utilization = sum(metric.utilization_percent for metric in current_metrics.values()) / len(current_metrics)
        
        return total_power, total_energy, avg_utilization
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics for all monitored GPUs."""
        with self._lock:
            if not self.metrics_history:
                return {}
            
            all_metrics = []
            for gpu_metrics in self.metrics_history.values():
                all_metrics.extend(gpu_metrics)
            
            if not all_metrics:
                return {}
            
            powers = [m.power_watts for m in all_metrics if m.power_watts > 0]
            utilizations = [m.utilization_percent for m in all_metrics]
            
            return {
                "total_gpus": len(self.gpu_ids),
                "avg_power_watts": sum(powers) / len(powers) if powers else 0.0,
                "peak_power_watts": max(powers) if powers else 0.0,
                "avg_utilization": sum(utilizations) / len(utilizations) if utilizations else 0.0,
                "peak_utilization": max(utilizations) if utilizations else 0.0,
                "total_samples": len(all_metrics),
            }


class SystemMonitor:
    """Monitor system-wide metrics including CPU and memory."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.metrics_history: List[SystemMetrics] = []
        self._lock = threading.Lock()
        
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available. System monitoring will be limited.")
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
        else:
            cpu_percent = memory_percent = 0.0
        
        # Power estimation (simplified - would need actual power monitoring)
        power_watts = 0.0
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            power_watts=power_watts,
            timestamp=time.time()
        )
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    metrics = self.get_current_metrics()
                    with self._lock:
                        self.metrics_history.append(metrics)
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                
                time.sleep(self.sampling_interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started system monitoring")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.is_monitoring = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped system monitoring")


class CarbonIntensityProvider:
    """Provide regional carbon intensity data."""
    
    # Default carbon intensities by region (g CO₂/kWh)
    CARBON_INTENSITIES = {
        "USA": {
            "California": 250,
            "Texas": 450,
            "New York": 300,
            "Washington": 120,
            "default": 400
        },
        "Germany": {
            "Bavaria": 411,
            "default": 475
        },
        "France": {
            "default": 85
        },
        "Norway": {
            "default": 20
        },
        "China": {
            "default": 681
        },
        "India": {
            "default": 708
        },
        "default": 475  # Global average
    }
    
    def __init__(self, country: str = "USA", region: str = "California"):
        self.country = country
        self.region = region
        self._cache = {}
        self._cache_timestamp = 0
        self._cache_duration = 3600  # 1 hour
    
    def get_carbon_intensity(self) -> float:
        """Get current carbon intensity for the region."""
        cache_key = f"{self.country}_{self.region}"
        current_time = time.time()
        
        # Check cache
        if (cache_key in self._cache and 
            current_time - self._cache_timestamp < self._cache_duration):
            return self._cache[cache_key]
        
        # Get intensity from static data (in production, would call real APIs)
        intensity = self._get_static_intensity()
        
        # Cache result
        self._cache[cache_key] = intensity
        self._cache_timestamp = current_time
        
        return intensity
    
    def _get_static_intensity(self) -> float:
        """Get carbon intensity from static data."""
        country_data = self.CARBON_INTENSITIES.get(self.country, {})
        
        if isinstance(country_data, dict):
            return country_data.get(self.region, country_data.get("default", 475))
        else:
            return country_data
    
    def get_renewable_percentage(self) -> float:
        """Get renewable energy percentage for the region."""
        # Simplified mapping (in production, would use real data)
        renewable_map = {
            ("Norway", ""): 98.0,
            ("France", ""): 75.0,
            ("USA", "California"): 45.0,
            ("USA", "Washington"): 85.0,
            ("Germany", ""): 42.0,
        }
        
        key = (self.country, self.region)
        return renewable_map.get(key, renewable_map.get((self.country, ""), 25.0))


class EnergyTracker:
    """Main energy tracking coordinator."""
    
    def __init__(self, gpu_ids: Optional[List[int]] = None, 
                 sampling_interval: float = 1.0,
                 country: str = "USA", 
                 region: str = "California"):
        self.gpu_monitor = GPUMonitor(gpu_ids, sampling_interval)
        self.system_monitor = SystemMonitor(sampling_interval)
        self.carbon_provider = CarbonIntensityProvider(country, region)
        
        self._start_time = None
        self._total_energy_kwh = 0.0
        self._total_co2_kg = 0.0
        
        logger.info(f"Initialized energy tracker for {country}/{region}")
    
    def start_tracking(self):
        """Start energy tracking."""
        self._start_time = time.time()
        self.gpu_monitor.start_monitoring()
        self.system_monitor.start_monitoring()
        logger.info("Started energy tracking")
    
    def stop_tracking(self):
        """Stop energy tracking."""
        self.gpu_monitor.stop_monitoring()
        self.system_monitor.stop_monitoring()
        logger.info("Stopped energy tracking")
    
    def get_current_consumption(self) -> Tuple[float, float, float]:
        """Get current power consumption, total energy, and CO₂ emissions."""
        # Try GPU monitoring first
        if PYNVML_AVAILABLE and self.gpu_monitor.handles:
            gpu_power, gpu_energy, _ = self.gpu_monitor.get_aggregated_metrics()
            
            # Calculate total energy (GPU + estimated system overhead)
            system_overhead_factor = 1.2  # 20% overhead for CPU, cooling, etc.
            total_power_watts = gpu_power * system_overhead_factor
            
            # Update cumulative energy
            if self._start_time:
                elapsed_hours = (time.time() - self._start_time) / 3600
                self._total_energy_kwh = total_power_watts * elapsed_hours / 1000
            
        else:
            # Use mock eco2ai tracker for development/testing
            if not hasattr(self, '_mock_tracker'):
                self._mock_tracker = eco2ai.Tracker(
                    project_name="hf-training",
                    country=self.carbon_provider.country,
                    region=self.carbon_provider.region
                )
                if self._start_time:
                    self._mock_tracker.start()
            
            # Get mock consumption data
            mock_data = self._mock_tracker.get_current()
            total_power_watts = mock_data["power"]
            self._total_energy_kwh = mock_data["energy"]
        
        # Calculate CO₂ emissions
        carbon_intensity = self.carbon_provider.get_carbon_intensity()
        self._total_co2_kg = self._total_energy_kwh * carbon_intensity / 1000
        
        return total_power_watts, self._total_energy_kwh, self._total_co2_kg
    
    def get_efficiency_metrics(self, samples_processed: int) -> Dict[str, float]:
        """Calculate training efficiency metrics."""
        power, energy, co2 = self.get_current_consumption()
        
        if energy > 0 and samples_processed > 0:
            return {
                "samples_per_kwh": samples_processed / energy,
                "energy_per_sample": energy / samples_processed,
                "co2_per_sample": co2 / samples_processed,
                "watts_per_sample": power / samples_processed if samples_processed > 0 else 0,
            }
        
        return {
            "samples_per_kwh": 0.0,
            "energy_per_sample": 0.0,
            "co2_per_sample": 0.0,
            "watts_per_sample": 0.0,
        }
    
    def predict_carbon_budget_exhaustion(self, current_usage_kg: float, budget_limit_kg: float, 
                                       training_progress: float) -> CarbonBudgetForecast:
        """Predict when carbon budget will be exhausted."""
        # Get historical efficiency data
        historical_efficiency = []
        if hasattr(self.gpu_monitor, 'metrics_history'):
            with self.gpu_monitor._lock:
                for gpu_metrics in self.gpu_monitor.metrics_history.values():
                    if gpu_metrics:
                        efficiency_values = [m.power_efficiency for m in gpu_metrics[-20:] if hasattr(m, 'power_efficiency') and m.power_efficiency > 0]
                        historical_efficiency.extend(efficiency_values)
        
        return self.gpu_monitor.analytics_engine.predict_carbon_budget(
            current_usage_kg, budget_limit_kg, training_progress, historical_efficiency
        )
    
    def get_anomaly_report(self, hours: float = 1.0) -> Dict[str, Any]:
        """Get detailed anomaly report for specified time period."""
        if hasattr(self.gpu_monitor, 'get_anomaly_report'):
            return self.gpu_monitor.get_anomaly_report(hours)
        return {"error": "Advanced analytics not available"}
    
    def get_optimization_recommendations(self, hours: float = 1.0) -> List[Dict[str, Any]]:
        """Get recent optimization recommendations."""
        if hasattr(self.gpu_monitor, 'get_optimization_recommendations'):
            return self.gpu_monitor.get_optimization_recommendations(hours)
        return []
    
    def add_anomaly_callback(self, callback: Callable):
        """Add callback for anomaly notifications."""
        if hasattr(self.gpu_monitor, 'add_anomaly_callback'):
            self.gpu_monitor.add_anomaly_callback(callback)
    
    def add_recommendation_callback(self, callback: Callable):
        """Add callback for optimization recommendations."""
        if hasattr(self.gpu_monitor, 'add_recommendation_callback'):
            self.gpu_monitor.add_recommendation_callback(callback)
    
    def is_available(self) -> bool:
        """Check if energy tracking is available."""
        # Allow mock mode for development
        return (PYNVML_AVAILABLE and bool(self.gpu_monitor.handles)) or eco2ai is not None