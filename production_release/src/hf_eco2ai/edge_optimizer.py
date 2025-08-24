"""Edge Computing Carbon Optimization for HF Eco2AI.

This module provides specialized carbon tracking and optimization for edge computing environments:
- Mobile device carbon tracking
- IoT training environment support  
- Federated learning carbon aggregation
- Edge-cloud hybrid optimization
- Battery-aware training optimization
"""

import time
import logging
import json
import asyncio
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from enum import Enum
import platform
import subprocess
import socket
from collections import deque, defaultdict
import statistics

# Optional edge computing imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Mobile device monitoring (Android/iOS)
try:
    import py_mobilenet
    MOBILE_MONITORING_AVAILABLE = True
except ImportError:
    MOBILE_MONITORING_AVAILABLE = False

# Import existing components
try:
    from .performance_optimizer import optimized, get_performance_optimizer
    from .error_handling import handle_gracefully, ErrorSeverity, resilient_operation
    from .monitoring import GPUMetrics, SystemMetrics
except ImportError:
    # Fallback decorators
    def optimized(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def handle_gracefully(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def resilient_operation(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class ErrorSeverity:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"


logger = logging.getLogger(__name__)


class EdgeDeviceType(Enum):
    """Types of edge devices."""
    MOBILE_PHONE = "mobile_phone"
    TABLET = "tablet"
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    CORAL_DEV_BOARD = "coral_dev_board"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    IOT_SENSOR = "iot_sensor"
    EMBEDDED_SYSTEM = "embedded_system"
    UNKNOWN = "unknown"


class PowerSource(Enum):
    """Power source types for edge devices."""
    BATTERY = "battery"
    AC_POWER = "ac_power"
    SOLAR = "solar"
    WIND = "wind"
    HYBRID = "hybrid"
    USB_POWER = "usb_power"
    UNKNOWN = "unknown"


class NetworkType(Enum):
    """Network connection types."""
    WIFI = "wifi"
    CELLULAR_4G = "cellular_4g"
    CELLULAR_5G = "cellular_5g"
    ETHERNET = "ethernet"
    BLUETOOTH = "bluetooth"
    ZIGBEE = "zigbee"
    LORA = "lora"
    OFFLINE = "offline"


@dataclass
class EdgeDeviceProfile:
    """Profile of an edge device."""
    device_id: str
    device_type: EdgeDeviceType
    cpu_cores: int
    memory_mb: int
    storage_gb: int
    has_gpu: bool = False
    gpu_memory_mb: int = 0
    power_source: PowerSource = PowerSource.UNKNOWN
    network_type: NetworkType = NetworkType.UNKNOWN
    operating_system: str = ""
    battery_capacity_mah: int = 0
    thermal_design_power_watts: float = 0.0
    carbon_intensity_region: str = ""
    
    def __post_init__(self):
        """Auto-detect device characteristics if not provided."""
        if not self.operating_system:
            self.operating_system = platform.system()
        
        if not self.cpu_cores and PSUTIL_AVAILABLE:
            self.cpu_cores = psutil.cpu_count()
        
        if not self.memory_mb and PSUTIL_AVAILABLE:
            self.memory_mb = psutil.virtual_memory().total // (1024 * 1024)


@dataclass
class EdgeCarbonMetrics:
    """Carbon metrics specific to edge computing."""
    device_id: str
    timestamp: float
    power_consumption_watts: float = 0.0
    battery_level_percent: float = 100.0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    network_usage_mb: float = 0.0
    temperature_celsius: float = 0.0
    throttling_active: bool = False
    carbon_intensity_gco2_kwh: float = 400.0  # Grid carbon intensity
    renewable_percentage: float = 0.0
    local_energy_source: bool = False  # Solar, wind, etc.
    federated_round: Optional[int] = None
    model_size_mb: float = 0.0
    inference_count: int = 0
    training_samples: int = 0


@dataclass
class FederatedTrainingNode:
    """Node in federated learning network."""
    node_id: str
    device_profile: EdgeDeviceProfile
    last_seen: float
    total_samples_trained: int = 0
    total_energy_consumed_kwh: float = 0.0
    total_co2_emissions_kg: float = 0.0
    average_power_watts: float = 0.0
    network_efficiency_mbps_per_watt: float = 0.0
    participation_rate: float = 1.0  # How often this node participates
    carbon_efficiency_samples_per_kg_co2: float = 0.0


@dataclass
class EdgeCloudSplit:
    """Configuration for edge-cloud hybrid training."""
    edge_percentage: float  # Percentage of computation on edge
    cloud_percentage: float  # Percentage of computation on cloud
    edge_layers: List[str]  # Which layers run on edge
    cloud_layers: List[str]  # Which layers run on cloud
    communication_cost_kwh_per_mb: float = 0.001  # Energy cost of data transfer
    latency_ms: float = 0.0
    total_energy_kwh: float = 0.0
    total_co2_kg: float = 0.0


class MobileDeviceMonitor:
    """Monitor mobile device power consumption and performance."""
    
    def __init__(self, device_profile: EdgeDeviceProfile):
        self.device_profile = device_profile
        self.metrics_history: deque = deque(maxlen=1000)
        self.is_monitoring = False
        self._lock = threading.Lock()
        
        # Platform-specific monitoring setup
        self.platform = platform.system().lower()
        self.monitoring_interval = 5.0  # Longer interval for battery preservation
        
    @optimized(cache_ttl=30.0)
    def get_battery_info(self) -> Dict[str, float]:
        """Get battery information."""
        if not PSUTIL_AVAILABLE:
            return {"level": 100.0, "charging": False, "time_remaining": 0}
        
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    "level": battery.percent,
                    "charging": battery.power_plugged,
                    "time_remaining": battery.secsleft if battery.secsleft != -1 else 0
                }
        except Exception as e:
            logger.debug(f"Battery info not available: {e}")
        
        return {"level": 100.0, "charging": False, "time_remaining": 0}
    
    @handle_gracefully(severity=ErrorSeverity.LOW, fallback_value={})
    def get_mobile_power_consumption(self) -> Dict[str, float]:
        """Estimate power consumption on mobile device."""
        power_breakdown = {
            "cpu": 0.0,
            "gpu": 0.0,
            "display": 0.0,
            "network": 0.0,
            "total": 0.0
        }
        
        if not PSUTIL_AVAILABLE:
            return power_breakdown
        
        try:
            # CPU power estimation
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            
            if cpu_freq:
                # Rough estimation: P = C * V² * f where C is capacitance, V is voltage, f is frequency
                base_cpu_power = 1.0  # Base power consumption in watts
                freq_factor = cpu_freq.current / cpu_freq.max if cpu_freq.max > 0 else 1.0
                power_breakdown["cpu"] = base_cpu_power * (cpu_percent / 100) * freq_factor
            else:
                power_breakdown["cpu"] = 0.5 * (cpu_percent / 100)  # Fallback estimation
            
            # Memory power (rough estimation)
            memory_percent = psutil.virtual_memory().percent
            power_breakdown["memory"] = 0.3 * (memory_percent / 100)
            
            # Network power estimation
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                bytes_sent = net_io.bytes_sent - self._last_net_io.bytes_sent
                bytes_recv = net_io.bytes_recv - self._last_net_io.bytes_recv
                network_mb = (bytes_sent + bytes_recv) / (1024 * 1024)
                # Network power varies by type (WiFi ~0.1W, 4G ~0.5W, 5G ~1.0W)
                network_power_per_mb = {
                    NetworkType.WIFI: 0.01,
                    NetworkType.CELLULAR_4G: 0.05,
                    NetworkType.CELLULAR_5G: 0.1
                }.get(self.device_profile.network_type, 0.03)
                power_breakdown["network"] = network_mb * network_power_per_mb
            
            self._last_net_io = net_io
            
            # Display power (if device has display)
            if self.device_profile.device_type in [EdgeDeviceType.MOBILE_PHONE, EdgeDeviceType.TABLET, EdgeDeviceType.LAPTOP]:
                power_breakdown["display"] = 1.5  # Typical display power consumption
            
            # Total power
            power_breakdown["total"] = sum(power_breakdown[k] for k in ["cpu", "memory", "network", "display"])
            
            # Apply thermal throttling factor
            if self.is_thermal_throttling():
                power_breakdown["total"] *= 0.7  # Reduced power due to throttling
                
        except Exception as e:
            logger.warning(f"Error estimating mobile power consumption: {e}")
        
        return power_breakdown
    
    def is_thermal_throttling(self) -> bool:
        """Check if device is thermal throttling."""
        try:
            if PSUTIL_AVAILABLE and hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Check if any temperature sensor is above 80°C
                    for sensor_name, sensor_list in temps.items():
                        for sensor in sensor_list:
                            if sensor.current and sensor.current > 80.0:
                                return True
        except Exception:
            pass
        
        # Fallback: check CPU frequency scaling
        try:
            if PSUTIL_AVAILABLE:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq and cpu_freq.current < cpu_freq.max * 0.8:
                    return True
        except Exception:
            pass
        
        return False
    
    @resilient_operation(max_attempts=2)
    def start_monitoring(self, callback: Optional[Callable] = None):
        """Start mobile device monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    battery_info = self.get_battery_info()
                    power_info = self.get_mobile_power_consumption()
                    
                    # Create edge carbon metrics
                    metrics = EdgeCarbonMetrics(
                        device_id=self.device_profile.device_id,
                        timestamp=time.time(),
                        power_consumption_watts=power_info["total"],
                        battery_level_percent=battery_info["level"],
                        cpu_usage_percent=psutil.cpu_percent() if PSUTIL_AVAILABLE else 0,
                        memory_usage_percent=psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0,
                        temperature_celsius=self._get_average_temperature(),
                        throttling_active=self.is_thermal_throttling(),
                        carbon_intensity_gco2_kwh=self._get_local_carbon_intensity()
                    )
                    
                    with self._lock:
                        self.metrics_history.append(metrics)
                    
                    if callback:
                        callback(metrics)
                    
                    # Adaptive monitoring interval based on battery level
                    if battery_info["level"] < 20:
                        sleep_time = self.monitoring_interval * 2  # Reduce monitoring frequency
                    elif battery_info["level"] < 50:
                        sleep_time = self.monitoring_interval * 1.5
                    else:
                        sleep_time = self.monitoring_interval
                    
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in mobile monitoring loop: {e}")
                    time.sleep(self.monitoring_interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"Started mobile monitoring for device {self.device_profile.device_id}")
    
    def stop_monitoring(self):
        """Stop mobile device monitoring."""
        self.is_monitoring = False
        logger.info("Stopped mobile device monitoring")
    
    def _get_average_temperature(self) -> float:
        """Get average device temperature."""
        try:
            if PSUTIL_AVAILABLE and hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    all_temps = []
                    for sensor_list in temps.values():
                        for sensor in sensor_list:
                            if sensor.current:
                                all_temps.append(sensor.current)
                    
                    if all_temps:
                        return statistics.mean(all_temps)
        except Exception:
            pass
        
        return 25.0  # Default room temperature
    
    def _get_local_carbon_intensity(self) -> float:
        """Get local grid carbon intensity."""
        # This would ideally query local carbon intensity APIs
        # For now, return regional defaults
        regional_intensities = {
            "california": 250,
            "texas": 450,
            "germany": 475,
            "france": 85,
            "norway": 20,
            "china": 681
        }
        
        region = self.device_profile.carbon_intensity_region.lower()
        return regional_intensities.get(region, 400)  # Global average
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Get mobile device efficiency metrics."""
        with self._lock:
            if len(self.metrics_history) < 2:
                return {}
            
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 readings
            
            avg_power = statistics.mean([m.power_consumption_watts for m in recent_metrics])
            avg_cpu = statistics.mean([m.cpu_usage_percent for m in recent_metrics])
            avg_temp = statistics.mean([m.temperature_celsius for m in recent_metrics])
            
            # Calculate efficiency scores
            power_efficiency = (100 - avg_power) / 100  # Lower power = higher efficiency
            thermal_efficiency = max(0, (85 - avg_temp) / 85)  # Cooler = more efficient
            battery_efficiency = recent_metrics[-1].battery_level_percent / 100
            
            return {
                "average_power_watts": avg_power,
                "average_cpu_percent": avg_cpu,
                "average_temperature": avg_temp,
                "power_efficiency_score": power_efficiency,
                "thermal_efficiency_score": thermal_efficiency,
                "battery_efficiency_score": battery_efficiency,
                "overall_efficiency_score": (power_efficiency + thermal_efficiency + battery_efficiency) / 3,
                "throttling_incidents": sum(1 for m in recent_metrics if m.throttling_active)
            }


class FederatedLearningCarbonAggregator:
    """Aggregate and analyze carbon emissions across federated learning nodes."""
    
    def __init__(self):
        self.nodes: Dict[str, FederatedTrainingNode] = {}
        self.round_history: List[Dict[str, Any]] = []
        self.total_emissions_kg = 0.0
        self.total_energy_kwh = 0.0
        self.current_round = 0
        
    def add_node(self, node: FederatedTrainingNode):
        """Add a federated learning node."""
        self.nodes[node.node_id] = node
        logger.info(f"Added FL node: {node.node_id} ({node.device_profile.device_type.value})")
    
    def update_node_metrics(self, node_id: str, metrics: EdgeCarbonMetrics, 
                           samples_trained: int):
        """Update metrics for a federated learning node."""
        if node_id not in self.nodes:
            logger.warning(f"Unknown FL node: {node_id}")
            return
        
        node = self.nodes[node_id]
        
        # Update cumulative metrics
        energy_increment = metrics.power_consumption_watts * (5.0 / 3600)  # 5 seconds to hours
        co2_increment = energy_increment * (metrics.carbon_intensity_gco2_kwh / 1000)
        
        node.total_energy_consumed_kwh += energy_increment
        node.total_co2_emissions_kg += co2_increment
        node.total_samples_trained += samples_trained
        node.last_seen = metrics.timestamp
        
        # Update efficiency metrics
        if node.total_co2_emissions_kg > 0:
            node.carbon_efficiency_samples_per_kg_co2 = node.total_samples_trained / node.total_co2_emissions_kg
        
        # Update global totals
        self.total_energy_kwh += energy_increment
        self.total_emissions_kg += co2_increment
        
        logger.debug(f"Updated FL node {node_id}: {samples_trained} samples, {co2_increment:.4f} kg CO₂")
    
    def start_federated_round(self, selected_nodes: List[str], 
                            model_size_mb: float) -> Dict[str, Any]:
        """Start a new federated learning round."""
        self.current_round += 1
        
        round_info = {
            "round_number": self.current_round,
            "start_time": time.time(),
            "selected_nodes": selected_nodes,
            "model_size_mb": model_size_mb,
            "participating_nodes": len(selected_nodes),
            "total_nodes": len(self.nodes)
        }
        
        # Calculate communication energy cost
        communication_energy = 0.0
        for node_id in selected_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Energy for downloading model and uploading updates
                download_energy = model_size_mb * 0.001  # kWh per MB
                upload_energy = model_size_mb * 0.001  # Assuming same size for updates
                communication_energy += (download_energy + upload_energy)
        
        round_info["communication_energy_kwh"] = communication_energy
        
        logger.info(f"Started FL round {self.current_round} with {len(selected_nodes)} nodes")
        return round_info
    
    def finish_federated_round(self, round_info: Dict[str, Any]) -> Dict[str, Any]:
        """Finish a federated learning round and calculate metrics."""
        round_info["end_time"] = time.time()
        round_info["duration_seconds"] = round_info["end_time"] - round_info["start_time"]
        
        # Calculate round-specific metrics
        total_samples_this_round = 0
        total_energy_this_round = round_info["communication_energy_kwh"]
        total_co2_this_round = 0.0
        
        for node_id in round_info["selected_nodes"]:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Estimate samples trained in this round (simplified)
                samples_this_round = 100  # This would be provided by the FL framework
                total_samples_this_round += samples_this_round
                
                # Add node training energy (estimated)
                training_energy = node.average_power_watts * (round_info["duration_seconds"] / 3600)
                total_energy_this_round += training_energy
                
                # Calculate CO₂ for this round
                node_carbon_intensity = 400  # Would get from node's region
                co2_this_round = training_energy * (node_carbon_intensity / 1000)
                total_co2_this_round += co2_this_round
        
        round_info.update({
            "total_samples": total_samples_this_round,
            "total_energy_kwh": total_energy_this_round,
            "total_co2_kg": total_co2_this_round,
            "samples_per_kwh": total_samples_this_round / total_energy_this_round if total_energy_this_round > 0 else 0,
            "co2_per_sample": total_co2_this_round / total_samples_this_round if total_samples_this_round > 0 else 0
        })
        
        self.round_history.append(round_info)
        
        logger.info(f"Completed FL round {self.current_round}: {total_samples_this_round} samples, "
                   f"{total_energy_this_round:.3f} kWh, {total_co2_this_round:.3f} kg CO₂")
        
        return round_info
    
    def optimize_node_selection(self, target_nodes: int, 
                               optimization_criteria: str = "carbon_efficiency") -> List[str]:
        """Select optimal nodes for federated learning based on carbon efficiency."""
        available_nodes = [
            node_id for node_id, node in self.nodes.items()
            if time.time() - node.last_seen < 300  # Active in last 5 minutes
        ]
        
        if len(available_nodes) <= target_nodes:
            return available_nodes
        
        # Sort nodes by optimization criteria
        if optimization_criteria == "carbon_efficiency":
            sorted_nodes = sorted(
                available_nodes,
                key=lambda nid: self.nodes[nid].carbon_efficiency_samples_per_kg_co2,
                reverse=True
            )
        elif optimization_criteria == "low_power":
            sorted_nodes = sorted(
                available_nodes,
                key=lambda nid: self.nodes[nid].average_power_watts
            )
        elif optimization_criteria == "renewable_energy":
            # Prioritize nodes with renewable energy sources
            sorted_nodes = sorted(
                available_nodes,
                key=lambda nid: (
                    self.nodes[nid].device_profile.power_source == PowerSource.SOLAR or
                    self.nodes[nid].device_profile.power_source == PowerSource.WIND
                ),
                reverse=True
            )
        else:
            # Random selection
            import random
            sorted_nodes = available_nodes.copy()
            random.shuffle(sorted_nodes)
        
        selected = sorted_nodes[:target_nodes]
        logger.info(f"Selected {len(selected)} nodes using {optimization_criteria} criteria")
        
        return selected
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated carbon metrics across all nodes."""
        if not self.nodes:
            return {}
        
        active_nodes = [
            node for node in self.nodes.values()
            if time.time() - node.last_seen < 600  # Active in last 10 minutes
        ]
        
        if not active_nodes:
            return {"error": "No active nodes"}
        
        total_samples = sum(node.total_samples_trained for node in active_nodes)
        avg_efficiency = statistics.mean([
            node.carbon_efficiency_samples_per_kg_co2 for node in active_nodes
            if node.carbon_efficiency_samples_per_kg_co2 > 0
        ]) if active_nodes else 0
        
        device_type_breakdown = defaultdict(int)
        power_source_breakdown = defaultdict(int)
        
        for node in active_nodes:
            device_type_breakdown[node.device_profile.device_type.value] += 1
            power_source_breakdown[node.device_profile.power_source.value] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len(active_nodes),
            "total_samples_trained": total_samples,
            "total_energy_consumed_kwh": self.total_energy_kwh,
            "total_co2_emissions_kg": self.total_emissions_kg,
            "average_carbon_efficiency": avg_efficiency,
            "total_rounds_completed": len(self.round_history),
            "device_type_distribution": dict(device_type_breakdown),
            "power_source_distribution": dict(power_source_breakdown),
            "last_round_metrics": self.round_history[-1] if self.round_history else None
        }


class EdgeCloudHybridOptimizer:
    """Optimize carbon footprint for edge-cloud hybrid training."""
    
    def __init__(self):
        self.optimization_history: List[EdgeCloudSplit] = []
        self.cloud_carbon_intensity = 400.0  # g CO₂/kWh
        self.edge_carbon_intensity = 300.0   # Often better due to local renewable sources
        
    def optimize_edge_cloud_split(self, model_layers: List[str], 
                                 edge_device: EdgeDeviceProfile,
                                 cloud_specs: Dict[str, Any],
                                 target_latency_ms: float = 1000.0) -> EdgeCloudSplit:
        """Optimize the split between edge and cloud computation."""
        
        # Analyze layer computational requirements
        layer_compute_costs = self._estimate_layer_costs(model_layers, edge_device)
        
        # Generate candidate splits
        candidates = self._generate_split_candidates(model_layers, layer_compute_costs)
        
        # Evaluate each candidate
        best_split = None
        best_score = float('inf')
        
        for candidate in candidates:
            score = self._evaluate_split(candidate, edge_device, cloud_specs, target_latency_ms)
            if score < best_score:
                best_score = score
                best_split = candidate
        
        if best_split:
            self.optimization_history.append(best_split)
            logger.info(f"Optimized edge-cloud split: {best_split.edge_percentage:.1f}% edge, "
                       f"{best_split.cloud_percentage:.1f}% cloud")
        
        return best_split or self._create_default_split(model_layers)
    
    def _estimate_layer_costs(self, layers: List[str], device: EdgeDeviceProfile) -> Dict[str, Dict[str, float]]:
        """Estimate computational costs for each layer."""
        costs = {}
        
        for layer in layers:
            # Rough estimates based on layer type
            if "conv" in layer.lower() or "attention" in layer.lower():
                # Computation-heavy layers
                edge_cost = 2.0 if device.has_gpu else 5.0
                cloud_cost = 1.0
                memory_mb = 100
            elif "linear" in layer.lower() or "dense" in layer.lower():
                # Medium computation
                edge_cost = 1.5 if device.has_gpu else 3.0
                cloud_cost = 0.8
                memory_mb = 50
            elif "norm" in layer.lower() or "dropout" in layer.lower():
                # Light computation
                edge_cost = 0.5
                cloud_cost = 0.3
                memory_mb = 10
            else:
                # Default
                edge_cost = 1.0
                cloud_cost = 0.7
                memory_mb = 30
            
            costs[layer] = {
                "edge_compute_cost": edge_cost,
                "cloud_compute_cost": cloud_cost,
                "memory_mb": memory_mb,
                "data_transfer_mb": memory_mb * 0.5  # Approximate activation size
            }
        
        return costs
    
    def _generate_split_candidates(self, layers: List[str], 
                                  costs: Dict[str, Dict[str, float]]) -> List[EdgeCloudSplit]:
        """Generate candidate edge-cloud splits."""
        candidates = []
        
        # Strategy 1: Early layers on edge, later on cloud
        for split_point in range(1, len(layers)):
            edge_layers = layers[:split_point]
            cloud_layers = layers[split_point:]
            
            edge_cost = sum(costs[layer]["edge_compute_cost"] for layer in edge_layers)
            cloud_cost = sum(costs[layer]["cloud_compute_cost"] for layer in cloud_layers)
            total_cost = edge_cost + cloud_cost
            
            if total_cost > 0:
                candidates.append(EdgeCloudSplit(
                    edge_percentage=(edge_cost / total_cost) * 100,
                    cloud_percentage=(cloud_cost / total_cost) * 100,
                    edge_layers=edge_layers,
                    cloud_layers=cloud_layers
                ))
        
        # Strategy 2: Lightweight layers on edge, heavy on cloud
        light_layers = [layer for layer in layers if costs[layer]["edge_compute_cost"] < 1.0]
        heavy_layers = [layer for layer in layers if layer not in light_layers]
        
        if light_layers and heavy_layers:
            edge_cost = sum(costs[layer]["edge_compute_cost"] for layer in light_layers)
            cloud_cost = sum(costs[layer]["cloud_compute_cost"] for layer in heavy_layers)
            total_cost = edge_cost + cloud_cost
            
            if total_cost > 0:
                candidates.append(EdgeCloudSplit(
                    edge_percentage=(edge_cost / total_cost) * 100,
                    cloud_percentage=(cloud_cost / total_cost) * 100,
                    edge_layers=light_layers,
                    cloud_layers=heavy_layers
                ))
        
        return candidates
    
    def _evaluate_split(self, split: EdgeCloudSplit, edge_device: EdgeDeviceProfile,
                       cloud_specs: Dict[str, Any], target_latency: float) -> float:
        """Evaluate a split configuration (lower score is better)."""
        
        # Calculate energy consumption
        edge_energy = self._calculate_edge_energy(split.edge_layers, edge_device)
        cloud_energy = self._calculate_cloud_energy(split.cloud_layers, cloud_specs)
        
        # Calculate communication energy
        data_transfer_mb = sum(10.0 for _ in split.cloud_layers)  # Simplified
        comm_energy = data_transfer_mb * split.communication_cost_kwh_per_mb
        
        total_energy = edge_energy + cloud_energy + comm_energy
        
        # Calculate CO₂ emissions
        edge_co2 = edge_energy * (self.edge_carbon_intensity / 1000)
        cloud_co2 = cloud_energy * (self.cloud_carbon_intensity / 1000)
        total_co2 = edge_co2 + cloud_co2
        
        # Estimate latency
        edge_latency = len(split.edge_layers) * 50  # ms per layer
        cloud_latency = len(split.cloud_layers) * 30  # Cloud is faster
        network_latency = 100  # Round-trip network latency
        total_latency = edge_latency + cloud_latency + network_latency
        
        # Update split with calculated values
        split.total_energy_kwh = total_energy
        split.total_co2_kg = total_co2
        split.latency_ms = total_latency
        
        # Scoring function (prioritize carbon efficiency with latency constraint)
        if total_latency > target_latency:
            latency_penalty = (total_latency - target_latency) * 0.01
        else:
            latency_penalty = 0
        
        score = total_co2 + latency_penalty
        return score
    
    def _calculate_edge_energy(self, layers: List[str], device: EdgeDeviceProfile) -> float:
        """Calculate energy consumption for edge computation."""
        # Base power consumption
        base_power = device.thermal_design_power_watts or 5.0
        
        # Computation factor based on layers
        compute_factor = sum(1.0 for layer in layers if "conv" in layer.lower()) * 0.5
        compute_factor += sum(0.3 for layer in layers if "linear" in layer.lower())
        compute_factor += sum(0.1 for layer in layers)  # Base cost per layer
        
        # Time factor (simplified - would depend on actual inference time)
        time_hours = 0.001  # 1 second inference
        
        return base_power * compute_factor * time_hours / 1000  # Convert to kWh
    
    def _calculate_cloud_energy(self, layers: List[str], cloud_specs: Dict[str, Any]) -> float:
        """Calculate energy consumption for cloud computation."""
        # Cloud is more efficient but has baseline overhead
        baseline_power = cloud_specs.get("gpu_power_watts", 250)
        pue = cloud_specs.get("power_usage_effectiveness", 1.4)  # Data center efficiency
        
        compute_factor = sum(0.8 for layer in layers if "conv" in layer.lower())
        compute_factor += sum(0.2 for layer in layers if "linear" in layer.lower())
        compute_factor += sum(0.05 for layer in layers)
        
        time_hours = 0.0005  # Cloud is faster
        
        return baseline_power * pue * compute_factor * time_hours / 1000
    
    def _create_default_split(self, layers: List[str]) -> EdgeCloudSplit:
        """Create a default 50-50 split."""
        mid_point = len(layers) // 2
        return EdgeCloudSplit(
            edge_percentage=50.0,
            cloud_percentage=50.0,
            edge_layers=layers[:mid_point],
            cloud_layers=layers[mid_point:]
        )
    
    def get_optimization_recommendations(self, current_split: EdgeCloudSplit) -> List[Dict[str, Any]]:
        """Get recommendations for optimizing edge-cloud split."""
        recommendations = []
        
        # Analyze current split efficiency
        if current_split.edge_percentage > 70:
            recommendations.append({
                "type": "optimization",
                "description": "Consider moving some computation to cloud for better efficiency",
                "potential_savings_co2_kg": current_split.total_co2_kg * 0.15,
                "implementation": "Move heavy layers (conv, attention) to cloud"
            })
        
        if current_split.latency_ms > 1000:
            recommendations.append({
                "type": "performance",
                "description": "Latency is high, consider optimizing network or moving more to edge",
                "target_latency_ms": 800,
                "implementation": "Optimize model compression or improve network connection"
            })
        
        if current_split.total_energy_kwh > 0.01:
            recommendations.append({
                "type": "energy",
                "description": "Energy consumption is high, optimize model or hardware",
                "potential_savings_kwh": current_split.total_energy_kwh * 0.2,
                "implementation": "Use quantization, pruning, or more efficient hardware"
            })
        
        return recommendations


class EdgeCarbonOptimizer:
    """Main optimizer for edge computing carbon efficiency."""
    
    def __init__(self):
        self.mobile_monitors: Dict[str, MobileDeviceMonitor] = {}
        self.federated_aggregator = FederatedLearningCarbonAggregator()
        self.hybrid_optimizer = EdgeCloudHybridOptimizer()
        self.optimization_strategies: List[Callable] = []
        
        # Register default optimization strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default optimization strategies."""
        self.optimization_strategies.extend([
            self._battery_aware_optimization,
            self._thermal_optimization,
            self._network_aware_optimization,
            self._renewable_energy_optimization
        ])
    
    def add_mobile_device(self, device_profile: EdgeDeviceProfile) -> str:
        """Add a mobile device for monitoring."""
        monitor = MobileDeviceMonitor(device_profile)
        self.mobile_monitors[device_profile.device_id] = monitor
        
        # Also add to federated learning if applicable
        if device_profile.device_type in [EdgeDeviceType.MOBILE_PHONE, EdgeDeviceType.TABLET]:
            fl_node = FederatedTrainingNode(
                node_id=device_profile.device_id,
                device_profile=device_profile,
                last_seen=time.time()
            )
            self.federated_aggregator.add_node(fl_node)
        
        logger.info(f"Added mobile device: {device_profile.device_id}")
        return device_profile.device_id
    
    def start_monitoring(self, device_id: str) -> bool:
        """Start monitoring a specific device."""
        if device_id in self.mobile_monitors:
            self.mobile_monitors[device_id].start_monitoring(
                callback=lambda metrics: self._process_edge_metrics(device_id, metrics)
            )
            return True
        return False
    
    def _process_edge_metrics(self, device_id: str, metrics: EdgeCarbonMetrics):
        """Process edge metrics and apply optimizations."""
        # Update federated learning metrics
        if device_id in self.federated_aggregator.nodes:
            self.federated_aggregator.update_node_metrics(device_id, metrics, metrics.training_samples)
        
        # Apply optimization strategies
        for strategy in self.optimization_strategies:
            try:
                strategy(device_id, metrics)
            except Exception as e:
                logger.warning(f"Optimization strategy failed: {e}")
    
    def _battery_aware_optimization(self, device_id: str, metrics: EdgeCarbonMetrics):
        """Optimize based on battery level."""
        if metrics.battery_level_percent < 20:
            logger.info(f"Device {device_id} low battery ({metrics.battery_level_percent:.1f}%) - reducing monitoring")
            # Increase monitoring interval
            if device_id in self.mobile_monitors:
                self.mobile_monitors[device_id].monitoring_interval = 30.0  # 30 seconds
        elif metrics.battery_level_percent > 80:
            # Normal monitoring
            if device_id in self.mobile_monitors:
                self.mobile_monitors[device_id].monitoring_interval = 5.0  # 5 seconds
    
    def _thermal_optimization(self, device_id: str, metrics: EdgeCarbonMetrics):
        """Optimize based on thermal conditions."""
        if metrics.throttling_active or metrics.temperature_celsius > 80:
            logger.warning(f"Device {device_id} overheating ({metrics.temperature_celsius:.1f}°C) - reducing load")
            # Could trigger model compression, reduced batch size, etc.
    
    def _network_aware_optimization(self, device_id: str, metrics: EdgeCarbonMetrics):
        """Optimize based on network conditions."""
        if metrics.network_usage_mb > 100:  # High network usage
            logger.info(f"Device {device_id} high network usage - optimizing communication")
            # Could trigger model compression for federated learning
    
    def _renewable_energy_optimization(self, device_id: str, metrics: EdgeCarbonMetrics):
        """Optimize based on renewable energy availability."""
        if metrics.renewable_percentage > 70:
            logger.info(f"Device {device_id} high renewable energy - opportunistic training")
            # Could trigger additional training when renewable energy is high
    
    def get_edge_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive edge optimization report."""
        # Aggregate mobile device metrics
        mobile_summary = {}
        for device_id, monitor in self.mobile_monitors.items():
            efficiency = monitor.get_efficiency_metrics()
            mobile_summary[device_id] = efficiency
        
        # Get federated learning metrics
        fl_metrics = self.federated_aggregator.get_aggregated_metrics()
        
        # Get hybrid optimization metrics
        hybrid_metrics = {
            "total_optimizations": len(self.hybrid_optimizer.optimization_history),
            "latest_optimization": self.hybrid_optimizer.optimization_history[-1].__dict__ if self.hybrid_optimizer.optimization_history else None
        }
        
        return {
            "mobile_devices": mobile_summary,
            "federated_learning": fl_metrics,
            "hybrid_optimization": hybrid_metrics,
            "total_edge_devices": len(self.mobile_monitors),
            "optimization_strategies_active": len(self.optimization_strategies),
            "generated_timestamp": datetime.now()
        }


# Convenience functions for easy integration
def create_edge_device_profile(device_type: str = "auto") -> EdgeDeviceProfile:
    """Create an edge device profile with auto-detection."""
    device_id = f"edge_device_{uuid.uuid4().hex[:8]}"
    
    # Auto-detect device type
    if device_type == "auto":
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if "arm" in machine or "aarch64" in machine:
            if "android" in system:
                detected_type = EdgeDeviceType.MOBILE_PHONE
            else:
                detected_type = EdgeDeviceType.RASPBERRY_PI
        else:
            detected_type = EdgeDeviceType.LAPTOP
    else:
        detected_type = EdgeDeviceType(device_type)
    
    return EdgeDeviceProfile(
        device_id=device_id,
        device_type=detected_type,
        cpu_cores=0,  # Will be auto-detected
        memory_mb=0,  # Will be auto-detected
        storage_gb=32  # Default
    )


def optimize_for_federated_learning(nodes: List[EdgeDeviceProfile], 
                                   target_participants: int = 10) -> List[str]:
    """Optimize node selection for federated learning."""
    aggregator = FederatedLearningCarbonAggregator()
    
    # Add all nodes
    for device_profile in nodes:
        fl_node = FederatedTrainingNode(
            node_id=device_profile.device_id,
            device_profile=device_profile,
            last_seen=time.time()
        )
        aggregator.add_node(fl_node)
    
    # Select optimal nodes
    selected = aggregator.optimize_node_selection(target_participants, "carbon_efficiency")
    return selected