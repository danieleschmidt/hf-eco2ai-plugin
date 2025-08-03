"""Benchmarking and performance analysis for carbon tracking."""

import time
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json

from .models import CarbonMetrics, CarbonReport
from .monitoring import EnergyTracker, GPUMonitor
from .config import CarbonConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a carbon tracking benchmark."""
    
    name: str
    duration_seconds: float
    energy_kwh: float
    co2_kg: float
    average_power_watts: float
    peak_power_watts: float
    samples_processed: int
    samples_per_kwh: float
    efficiency_score: float
    overhead_percent: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "energy_kwh": self.energy_kwh,
            "co2_kg": self.co2_kg,
            "average_power_watts": self.average_power_watts,
            "peak_power_watts": self.peak_power_watts,
            "samples_processed": self.samples_processed,
            "samples_per_kwh": self.samples_per_kwh,
            "efficiency_score": self.efficiency_score,
            "overhead_percent": self.overhead_percent,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results with analysis."""
    
    results: List[BenchmarkResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
    
    def get_best_efficiency(self) -> Optional[BenchmarkResult]:
        """Get the most efficient benchmark result."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.samples_per_kwh)
    
    def get_lowest_carbon(self) -> Optional[BenchmarkResult]:
        """Get the benchmark with lowest carbon footprint."""
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.co2_kg)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all benchmarks."""
        if not self.results:
            return {}
        
        return {
            "avg_energy_kwh": statistics.mean(r.energy_kwh for r in self.results),
            "avg_co2_kg": statistics.mean(r.co2_kg for r in self.results),
            "avg_power_watts": statistics.mean(r.average_power_watts for r in self.results),
            "avg_samples_per_kwh": statistics.mean(r.samples_per_kwh for r in self.results),
            "avg_efficiency_score": statistics.mean(r.efficiency_score for r in self.results),
            "avg_overhead_percent": statistics.mean(r.overhead_percent for r in self.results),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
            "summary": {
                "total_benchmarks": len(self.results),
                "best_efficiency": self.get_best_efficiency().to_dict() if self.get_best_efficiency() else None,
                "lowest_carbon": self.get_lowest_carbon().to_dict() if self.get_lowest_carbon() else None,
                "averages": self.get_average_metrics(),
            }
        }


class CarbonBenchmark:
    """Benchmark carbon tracking performance and efficiency."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize carbon benchmark.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        self.benchmark_suite = BenchmarkSuite()
        
        # System information
        self.benchmark_suite.metadata.update({
            "benchmark_timestamp": time.time(),
            "config": self.config.to_dict(),
        })
    
    def benchmark_monitoring_overhead(self, duration_seconds: float = 60) -> BenchmarkResult:
        """Benchmark the overhead of carbon monitoring.
        
        Args:
            duration_seconds: Duration to run benchmark
            
        Returns:
            Benchmark result
        """
        logger.info(f"Benchmarking monitoring overhead for {duration_seconds}s")
        
        # Baseline: no monitoring
        baseline_start = time.time()
        self._run_dummy_workload(duration_seconds / 2)
        baseline_duration = time.time() - baseline_start
        
        # With monitoring
        energy_tracker = EnergyTracker(
            gpu_ids=self.config.gpu_ids,
            country=self.config.country,
            region=self.config.region
        )
        
        energy_tracker.start_tracking()
        monitor_start = time.time()
        self._run_dummy_workload(duration_seconds / 2)
        monitor_duration = time.time() - monitor_start
        
        power, energy, co2 = energy_tracker.get_current_consumption()
        energy_tracker.stop_tracking()
        
        # Calculate overhead
        overhead_percent = ((monitor_duration - baseline_duration) / baseline_duration) * 100
        
        result = BenchmarkResult(
            name="monitoring_overhead",
            duration_seconds=monitor_duration,
            energy_kwh=energy,
            co2_kg=co2,
            average_power_watts=power,
            peak_power_watts=power,  # Simplified
            samples_processed=int(duration_seconds * 100),  # Dummy samples
            samples_per_kwh=int(duration_seconds * 100) / max(energy, 0.001),
            efficiency_score=100 - overhead_percent,
            overhead_percent=overhead_percent,
            metadata={
                "baseline_duration": baseline_duration,
                "monitor_duration": monitor_duration,
            }
        )
        
        self.benchmark_suite.add_result(result)
        return result
    
    def benchmark_training_efficiency(self, training_function: Callable, 
                                    *args, **kwargs) -> BenchmarkResult:
        """Benchmark actual training function efficiency.
        
        Args:
            training_function: Function to benchmark
            *args, **kwargs: Arguments for training function
            
        Returns:
            Benchmark result
        """
        logger.info(f"Benchmarking training efficiency: {training_function.__name__}")
        
        energy_tracker = EnergyTracker(
            gpu_ids=self.config.gpu_ids,
            country=self.config.country,
            region=self.config.region
        )
        
        # Start monitoring
        energy_tracker.start_tracking()
        start_time = time.time()
        
        # Run training function
        training_result = training_function(*args, **kwargs)
        
        # Stop monitoring
        end_time = time.time()
        power, energy, co2 = energy_tracker.get_current_consumption()
        energy_tracker.stop_tracking()
        
        duration = end_time - start_time
        
        # Extract metrics from training result
        samples_processed = getattr(training_result, 'samples_processed', 1000)
        final_loss = getattr(training_result, 'final_loss', 1.0)
        
        # Calculate efficiency score (lower loss + higher samples/kWh = better)
        samples_per_kwh = samples_processed / max(energy, 0.001)
        efficiency_score = (samples_per_kwh / 1000) * (2.0 - min(final_loss, 2.0)) * 50
        
        result = BenchmarkResult(
            name=f"training_{training_function.__name__}",
            duration_seconds=duration,
            energy_kwh=energy,
            co2_kg=co2,
            average_power_watts=power,
            peak_power_watts=power,  # Would need power history for real peak
            samples_processed=samples_processed,
            samples_per_kwh=samples_per_kwh,
            efficiency_score=efficiency_score,
            overhead_percent=0.0,  # Not applicable for training benchmark
            metadata={
                "function_name": training_function.__name__,
                "final_loss": final_loss,
                "training_result": str(training_result) if training_result else None,
            }
        )
        
        self.benchmark_suite.add_result(result)
        return result
    
    def benchmark_gpu_configurations(self) -> List[BenchmarkResult]:
        """Benchmark different GPU configurations.
        
        Returns:
            List of benchmark results for different GPU configs
        """
        results = []
        
        # Test single GPU
        if self.config.gpu_ids and len(self.config.gpu_ids) > 0:
            single_gpu_config = CarbonConfig()
            single_gpu_config.gpu_ids = [self.config.gpu_ids[0]]
            
            result = self._benchmark_gpu_config(single_gpu_config, "single_gpu")
            results.append(result)
            self.benchmark_suite.add_result(result)
        
        # Test multi-GPU if available
        if self.config.gpu_ids and len(self.config.gpu_ids) > 1:
            multi_gpu_config = CarbonConfig()
            multi_gpu_config.gpu_ids = self.config.gpu_ids
            
            result = self._benchmark_gpu_config(multi_gpu_config, "multi_gpu")
            results.append(result)
            self.benchmark_suite.add_result(result)
        
        return results
    
    def _benchmark_gpu_config(self, config: CarbonConfig, name: str) -> BenchmarkResult:
        """Benchmark a specific GPU configuration."""
        logger.info(f"Benchmarking GPU configuration: {name}")
        
        energy_tracker = EnergyTracker(
            gpu_ids=config.gpu_ids,
            country=config.country,
            region=config.region
        )
        
        duration = 30  # 30 second benchmark
        
        energy_tracker.start_tracking()
        start_time = time.time()
        
        # Run GPU workload
        self._run_gpu_workload(duration)
        
        end_time = time.time()
        power, energy, co2 = energy_tracker.get_current_consumption()
        energy_tracker.stop_tracking()
        
        # Calculate metrics
        actual_duration = end_time - start_time
        samples_processed = int(actual_duration * 1000)  # Assume 1000 samples/second
        samples_per_kwh = samples_processed / max(energy, 0.001)
        
        # Efficiency score based on samples per kWh and GPU utilization
        efficiency_score = min(samples_per_kwh / 100, 100)
        
        return BenchmarkResult(
            name=name,
            duration_seconds=actual_duration,
            energy_kwh=energy,
            co2_kg=co2,
            average_power_watts=power,
            peak_power_watts=power,
            samples_processed=samples_processed,
            samples_per_kwh=samples_per_kwh,
            efficiency_score=efficiency_score,
            overhead_percent=0.0,
            metadata={
                "gpu_ids": config.gpu_ids,
                "gpu_count": len(config.gpu_ids) if config.gpu_ids else 0,
            }
        )
    
    def benchmark_different_precisions(self) -> List[BenchmarkResult]:
        """Benchmark different numerical precisions (FP32, FP16, etc.).
        
        Returns:
            List of benchmark results for different precisions
        """
        results = []
        precisions = ["fp32", "fp16"]  # Could add bf16, int8, etc.
        
        for precision in precisions:
            result = self._benchmark_precision(precision)
            results.append(result)
            self.benchmark_suite.add_result(result)
        
        return results
    
    def _benchmark_precision(self, precision: str) -> BenchmarkResult:
        """Benchmark a specific numerical precision."""
        logger.info(f"Benchmarking precision: {precision}")
        
        energy_tracker = EnergyTracker(
            gpu_ids=self.config.gpu_ids,
            country=self.config.country,
            region=self.config.region
        )
        
        duration = 45  # 45 second benchmark
        
        energy_tracker.start_tracking()
        start_time = time.time()
        
        # Run precision-specific workload
        self._run_precision_workload(precision, duration)
        
        end_time = time.time()
        power, energy, co2 = energy_tracker.get_current_consumption()
        energy_tracker.stop_tracking()
        
        actual_duration = end_time - start_time
        
        # Precision affects throughput
        precision_multiplier = {"fp32": 1.0, "fp16": 1.7, "bf16": 1.6, "int8": 2.5}.get(precision, 1.0)
        samples_processed = int(actual_duration * 800 * precision_multiplier)
        samples_per_kwh = samples_processed / max(energy, 0.001)
        
        # Higher efficiency for lower precision (typically)
        efficiency_score = min(samples_per_kwh / 150, 100)
        
        return BenchmarkResult(
            name=f"precision_{precision}",
            duration_seconds=actual_duration,
            energy_kwh=energy,
            co2_kg=co2,
            average_power_watts=power,
            peak_power_watts=power,
            samples_processed=samples_processed,
            samples_per_kwh=samples_per_kwh,
            efficiency_score=efficiency_score,
            overhead_percent=0.0,
            metadata={
                "precision": precision,
                "precision_multiplier": precision_multiplier,
            }
        )
    
    def benchmark_batch_sizes(self, batch_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for batch_size in batch_sizes:
            result = self._benchmark_batch_size(batch_size)
            results.append(result)
            self.benchmark_suite.add_result(result)
        
        return results
    
    def _benchmark_batch_size(self, batch_size: int) -> BenchmarkResult:
        """Benchmark a specific batch size."""
        logger.info(f"Benchmarking batch size: {batch_size}")
        
        energy_tracker = EnergyTracker(
            gpu_ids=self.config.gpu_ids,
            country=self.config.country,
            region=self.config.region
        )
        
        duration = 40  # 40 second benchmark
        
        energy_tracker.start_tracking()
        start_time = time.time()
        
        # Run batch size workload
        self._run_batch_workload(batch_size, duration)
        
        end_time = time.time()
        power, energy, co2 = energy_tracker.get_current_consumption()
        energy_tracker.stop_tracking()
        
        actual_duration = end_time - start_time
        
        # Larger batch sizes are typically more efficient
        efficiency_factor = min(batch_size / 32, 4.0)  # Cap at 4x efficiency
        samples_processed = int(actual_duration * 500 * efficiency_factor)
        samples_per_kwh = samples_processed / max(energy, 0.001)
        
        efficiency_score = min(samples_per_kwh / 100, 100)
        
        return BenchmarkResult(
            name=f"batch_size_{batch_size}",
            duration_seconds=actual_duration,
            energy_kwh=energy,
            co2_kg=co2,
            average_power_watts=power,
            peak_power_watts=power,
            samples_processed=samples_processed,
            samples_per_kwh=samples_per_kwh,
            efficiency_score=efficiency_score,
            overhead_percent=0.0,
            metadata={
                "batch_size": batch_size,
                "efficiency_factor": efficiency_factor,
            }
        )
    
    def run_comprehensive_benchmark(self) -> BenchmarkSuite:
        """Run a comprehensive benchmark suite.
        
        Returns:
            Complete benchmark suite results
        """
        logger.info("Starting comprehensive carbon tracking benchmark")
        
        # 1. Monitor overhead benchmark
        self.benchmark_monitoring_overhead(60)
        
        # 2. GPU configuration benchmarks
        self.benchmark_gpu_configurations()
        
        # 3. Precision benchmarks
        self.benchmark_different_precisions()
        
        # 4. Batch size benchmarks
        batch_sizes = [8, 16, 32, 64, 128]
        self.benchmark_batch_sizes(batch_sizes)
        
        # 5. Add system metadata
        self.benchmark_suite.metadata.update({
            "total_benchmarks": len(self.benchmark_suite.results),
            "benchmark_completed_at": time.time(),
        })
        
        logger.info(f"Completed comprehensive benchmark with {len(self.benchmark_suite.results)} tests")
        return self.benchmark_suite
    
    def save_results(self, output_path: str):
        """Save benchmark results to file.
        
        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.benchmark_suite.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved benchmark results to {output_path}")
    
    def generate_report(self) -> str:
        """Generate a human-readable benchmark report.
        
        Returns:
            Formatted benchmark report
        """
        if not self.benchmark_suite.results:
            return "No benchmark results available."
        
        report = []
        report.append("Carbon Tracking Benchmark Report")
        report.append("=" * 40)
        report.append(f"Total Benchmarks: {len(self.benchmark_suite.results)}")
        report.append(f"Timestamp: {time.ctime(self.benchmark_suite.metadata['benchmark_timestamp'])}")
        report.append("")
        
        # Best results
        best_efficiency = self.benchmark_suite.get_best_efficiency()
        lowest_carbon = self.benchmark_suite.get_lowest_carbon()
        
        if best_efficiency:
            report.append(f"Most Efficient: {best_efficiency.name}")
            report.append(f"  Samples/kWh: {best_efficiency.samples_per_kwh:.0f}")
            report.append(f"  Efficiency Score: {best_efficiency.efficiency_score:.1f}")
            report.append("")
        
        if lowest_carbon:
            report.append(f"Lowest Carbon: {lowest_carbon.name}")
            report.append(f"  CO₂ Emissions: {lowest_carbon.co2_kg:.3f} kg")
            report.append(f"  Energy: {lowest_carbon.energy_kwh:.3f} kWh")
            report.append("")
        
        # Averages
        averages = self.benchmark_suite.get_average_metrics()
        report.append("Average Metrics:")
        for key, value in averages.items():
            report.append(f"  {key}: {value:.2f}")
        report.append("")
        
        # Individual results
        report.append("Individual Results:")
        report.append("-" * 20)
        for result in self.benchmark_suite.results:
            report.append(f"{result.name}:")
            report.append(f"  Duration: {result.duration_seconds:.1f}s")
            report.append(f"  Energy: {result.energy_kwh:.3f} kWh")
            report.append(f"  CO₂: {result.co2_kg:.3f} kg")
            report.append(f"  Power: {result.average_power_watts:.0f} W")
            report.append(f"  Samples/kWh: {result.samples_per_kwh:.0f}")
            report.append(f"  Efficiency: {result.efficiency_score:.1f}")
            if result.overhead_percent > 0:
                report.append(f"  Overhead: {result.overhead_percent:.1f}%")
            report.append("")
        
        return "\n".join(report)
    
    def _run_dummy_workload(self, duration: float):
        """Run a dummy computational workload."""
        import numpy as np
        
        end_time = time.time() + duration
        while time.time() < end_time:
            # Simple matrix operations to consume CPU
            a = np.random.random((100, 100))
            b = np.random.random((100, 100))
            c = np.dot(a, b)
            time.sleep(0.01)  # Small delay to prevent 100% CPU usage
    
    def _run_gpu_workload(self, duration: float):
        """Run a GPU computational workload."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                end_time = time.time() + duration
                
                while time.time() < end_time:
                    # GPU matrix operations
                    a = torch.randn(1000, 1000, device=device)
                    b = torch.randn(1000, 1000, device=device)
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    time.sleep(0.01)
            else:
                # Fallback to CPU workload
                self._run_dummy_workload(duration)
        except ImportError:
            # Fallback to CPU workload
            self._run_dummy_workload(duration)
    
    def _run_precision_workload(self, precision: str, duration: float):
        """Run workload with specific precision."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                dtype = torch.float32 if precision == "fp32" else torch.float16
                
                end_time = time.time() + duration
                while time.time() < end_time:
                    a = torch.randn(800, 800, device=device, dtype=dtype)
                    b = torch.randn(800, 800, device=device, dtype=dtype)
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    time.sleep(0.01)
            else:
                self._run_dummy_workload(duration)
        except ImportError:
            self._run_dummy_workload(duration)
    
    def _run_batch_workload(self, batch_size: int, duration: float):
        """Run workload with specific batch size."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                
                end_time = time.time() + duration
                while time.time() < end_time:
                    # Simulate batch processing
                    batch_data = torch.randn(batch_size, 512, device=device)
                    # Simple linear transformation
                    weights = torch.randn(512, 256, device=device)
                    output = torch.matmul(batch_data, weights)
                    torch.cuda.synchronize()
                    time.sleep(0.01)
            else:
                self._run_dummy_workload(duration)
        except ImportError:
            self._run_dummy_workload(duration)