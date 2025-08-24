"""Export utilities for carbon tracking metrics and reports."""

import time
import threading
from typing import Optional, Dict, Any
import logging
from pathlib import Path

from .models import CarbonMetrics, CarbonReport

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from prometheus_client import CollectorRegistry, Gauge, Counter, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = Gauge = Counter = start_http_server = None


class PrometheusExporter:
    """Export carbon tracking metrics to Prometheus."""
    
    def __init__(self, port: int = 9091, host: str = "localhost", registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus exporter.
        
        Args:
            port: Port to serve Prometheus metrics
            host: Host to bind the metrics server
            registry: Custom Prometheus registry (optional)
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not available. Install with: pip install prometheus-client")
        
        self.port = port
        self.host = host
        self.registry = registry or CollectorRegistry()
        self._server_started = False
        self._lock = threading.Lock()
        
        # Initialize Prometheus metrics
        self._init_metrics()
        
        # Start HTTP server
        self._start_server()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Energy metrics
        self.energy_total = Counter(
            'carbon_tracking_energy_total_kwh',
            'Total energy consumption in kWh',
            ['project', 'experiment', 'gpu_id'],
            registry=self.registry
        )
        
        self.energy_current = Gauge(
            'carbon_tracking_energy_current_kwh',
            'Current cumulative energy consumption in kWh',
            ['project', 'experiment'],
            registry=self.registry
        )
        
        # Power metrics
        self.power_current = Gauge(
            'carbon_tracking_power_current_watts',
            'Current power consumption in watts',
            ['project', 'experiment', 'gpu_id'],
            registry=self.registry
        )
        
        # Carbon metrics
        self.co2_total = Counter(
            'carbon_tracking_co2_total_kg',
            'Total CO₂ emissions in kg',
            ['project', 'experiment', 'region'],
            registry=self.registry
        )
        
        self.co2_current = Gauge(
            'carbon_tracking_co2_current_kg',
            'Current cumulative CO₂ emissions in kg',
            ['project', 'experiment', 'region'],
            registry=self.registry
        )
        
        self.grid_intensity = Gauge(
            'carbon_tracking_grid_intensity_g_co2_per_kwh',
            'Current grid carbon intensity in g CO₂/kWh',
            ['region', 'country'],
            registry=self.registry
        )
        
        # Training efficiency metrics
        self.samples_per_kwh = Gauge(
            'carbon_tracking_samples_per_kwh',
            'Training samples processed per kWh',
            ['project', 'experiment'],
            registry=self.registry
        )
        
        self.training_steps = Counter(
            'carbon_tracking_training_steps_total',
            'Total training steps completed',
            ['project', 'experiment'],
            registry=self.registry
        )
        
        # GPU-specific metrics
        self.gpu_utilization = Gauge(
            'carbon_tracking_gpu_utilization_percent',
            'GPU utilization percentage',
            ['project', 'experiment', 'gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_memory_used = Gauge(
            'carbon_tracking_gpu_memory_used_mb',
            'GPU memory used in MB',
            ['project', 'experiment', 'gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_temperature = Gauge(
            'carbon_tracking_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['project', 'experiment', 'gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        # Efficiency and cost metrics
        self.energy_per_sample = Gauge(
            'carbon_tracking_energy_per_sample_wh',
            'Energy consumption per training sample in Wh',
            ['project', 'experiment'],
            registry=self.registry
        )
        
        self.co2_per_sample = Gauge(
            'carbon_tracking_co2_per_sample_g',
            'CO₂ emissions per training sample in grams',
            ['project', 'experiment'],
            registry=self.registry
        )
        
        logger.info("Initialized Prometheus metrics")
    
    def _start_server(self):
        """Start Prometheus HTTP server."""
        with self._lock:
            if not self._server_started:
                try:
                    start_http_server(self.port, addr=self.host, registry=self.registry)
                    self._server_started = True
                    logger.info(f"Started Prometheus server on {self.host}:{self.port}")
                except Exception as e:
                    logger.error(f"Failed to start Prometheus server: {e}")
                    raise
    
    def record_metrics(self, metric: CarbonMetrics, project: str = "hf-training", 
                       experiment: str = "default", region: str = "unknown"):
        """Record a carbon metric to Prometheus.
        
        Args:
            metric: Carbon metric to record
            project: Project name for labeling
            experiment: Experiment name for labeling
            region: Region name for labeling
        """
        try:
            # Basic labels
            base_labels = {'project': project, 'experiment': experiment}
            region_labels = {'region': region, 'country': region.split('/')[0] if '/' in region else region}
            
            # Energy metrics
            if metric.energy_kwh > 0:
                self.energy_total.labels(**base_labels, gpu_id='all').inc(metric.energy_kwh)
            
            self.energy_current.labels(**base_labels).set(metric.cumulative_energy_kwh)
            
            # Power metrics
            if metric.power_watts > 0:
                self.power_current.labels(**base_labels, gpu_id='all').set(metric.power_watts)
            
            # Carbon metrics
            if metric.co2_kg > 0:
                self.co2_total.labels(**base_labels, **region_labels).inc(metric.co2_kg)
            
            self.co2_current.labels(**base_labels, **region_labels).set(metric.cumulative_co2_kg)
            self.grid_intensity.labels(**region_labels).set(metric.grid_intensity)
            
            # Efficiency metrics
            if metric.samples_per_kwh > 0:
                self.samples_per_kwh.labels(**base_labels).set(metric.samples_per_kwh)
            
            if metric.step:
                self.training_steps.labels(**base_labels).inc()
            
            # Energy efficiency
            if metric.samples_processed > 0 and metric.cumulative_energy_kwh > 0:
                energy_per_sample_wh = (metric.cumulative_energy_kwh / metric.samples_processed) * 1000
                co2_per_sample_g = (metric.cumulative_co2_kg / metric.samples_processed) * 1000
                
                self.energy_per_sample.labels(**base_labels).set(energy_per_sample_wh)
                self.co2_per_sample.labels(**base_labels).set(co2_per_sample_g)
            
            # GPU-specific metrics
            for gpu_id, power in metric.gpu_power_watts.items():
                gpu_labels = {**base_labels, 'gpu_id': str(gpu_id), 'gpu_name': f'gpu_{gpu_id}'}
                
                self.power_current.labels(**gpu_labels).set(power)
                
                if gpu_id in metric.gpu_utilization:
                    self.gpu_utilization.labels(**gpu_labels).set(metric.gpu_utilization[gpu_id])
                
                if gpu_id in metric.gpu_memory_used:
                    self.gpu_memory_used.labels(**gpu_labels).set(metric.gpu_memory_used[gpu_id])
                
                if gpu_id in metric.gpu_temperature:
                    self.gpu_temperature.labels(**gpu_labels).set(metric.gpu_temperature[gpu_id])
            
        except Exception as e:
            logger.error(f"Failed to record Prometheus metrics: {e}")
    
    def get_metrics_url(self) -> str:
        """Get the URL for accessing Prometheus metrics."""
        return f"http://{self.host}:{self.port}/metrics"


class ReportExporter:
    """Export carbon reports to various formats."""
    
    def __init__(self):
        """Initialize report exporter."""
        pass
    
    def export_json(self, report: CarbonReport, path: str, include_detailed_metrics: bool = True):
        """Export report to JSON format.
        
        Args:
            report: Carbon report to export
            path: Output file path
            include_detailed_metrics: Whether to include detailed metrics timeline
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            report.to_json(path, include_detailed_metrics=include_detailed_metrics)
            logger.info(f"Exported carbon report to {path}")
        except Exception as e:
            logger.error(f"Failed to export JSON report to {path}: {e}")
    
    def export_csv(self, report: CarbonReport, path: str):
        """Export detailed metrics to CSV format.
        
        Args:
            report: Carbon report to export
            path: Output file path
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(path)
            logger.info(f"Exported carbon metrics CSV to {path}")
        except Exception as e:
            logger.error(f"Failed to export CSV report to {path}: {e}")
    
    def export_summary(self, report: CarbonReport, path: str):
        """Export human-readable summary to text file.
        
        Args:
            report: Carbon report to export
            path: Output file path
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(report.summary_text())
            logger.info(f"Exported carbon summary to {path}")
        except Exception as e:
            logger.error(f"Failed to export summary to {path}: {e}")
    
    def export_html_dashboard(self, report: CarbonReport, path: str):
        """Export interactive HTML dashboard.
        
        Args:
            report: Carbon report to export
            path: Output file path
        """
        try:
            html_content = self._generate_html_dashboard(report)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(html_content)
            logger.info(f"Exported HTML dashboard to {path}")
        except Exception as e:
            logger.error(f"Failed to export HTML dashboard to {path}: {e}")
    
    def _generate_html_dashboard(self, report: CarbonReport) -> str:
        """Generate HTML dashboard content."""
        # Generate basic HTML dashboard
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Carbon Tracking Report - {project_name}</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .recommendations {{ background: #e8f5e8; padding: 20px; border-radius: 6px; margin: 20px 0; }}
        .warning {{ background: #fff3cd; padding: 15px; border-radius: 6px; border-left: 4px solid #ffc107; margin: 20px 0; }}
        .chart-placeholder {{ background: #f8f9fa; padding: 40px; text-align: center; border-radius: 6px; margin: 20px 0; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌱 Carbon Tracking Report</h1>
            <p>Project: <strong>{project_name}</strong> | Generated: {timestamp}</p>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{total_energy:.2f} kWh</div>
                <div class="metric-label">Total Energy Consumed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_co2:.2f} kg</div>
                <div class="metric-label">Total CO₂ Emissions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_power:.0f} W</div>
                <div class="metric-label">Average Power</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{duration:.1f} hrs</div>
                <div class="metric-label">Training Duration</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{samples_per_kwh:.0f}</div>
                <div class="metric-label">Samples per kWh</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{equivalent_km:.0f} km</div>
                <div class="metric-label">Equivalent Car Distance</div>
            </div>
        </div>
        
        <div class="chart-placeholder">
            <h3>📊 Energy & CO₂ Timeline</h3>
            <p>Interactive charts would be displayed here with plotly.js or similar</p>
            <p>Metrics: {metrics_count} data points collected</p>
        </div>
        
        {recommendations_html}
        
        <div class="warning">
            <h4>⚠️ Environmental Context</h4>
            <p><strong>Region:</strong> {region} | <strong>Grid Intensity:</strong> {grid_intensity:.0f} g CO₂/kWh</p>
            <p><strong>Renewable Energy:</strong> {renewable_percent:.1f}% | <strong>Grid Type:</strong> {grid_type}</p>
        </div>
        
        <div style="margin-top: 40px; text-align: center; color: #666; font-size: 12px;">
            Generated by HF Eco2AI Plugin • <a href="https://github.com/danieleschmidt/hf-eco2ai-plugin">View on GitHub</a>
        </div>
    </div>
</body>
</html>
        """
        
        # Prepare recommendations HTML
        recommendations_html = ""
        if report.recommendations:
            recommendations_html = """
            <div class="recommendations">
                <h4>💡 Optimization Recommendations</h4>
                <ul>
            """
            for rec in report.recommendations[:5]:  # Top 5 recommendations
                recommendations_html += f"""
                    <li><strong>{rec.title}</strong> - {rec.description} 
                    (Potential reduction: {rec.potential_reduction_percent:.0f}%)</li>
                """
            recommendations_html += "</ul></div>"
        
        # Determine grid type
        renewable_percent = report.environmental_impact.renewable_percentage
        if renewable_percent > 80:
            grid_type = "Very Clean"
        elif renewable_percent > 50:
            grid_type = "Clean"
        elif renewable_percent > 25:
            grid_type = "Mixed"
        else:
            grid_type = "Carbon Intensive"
        
        # Format the HTML
        return html_template.format(
            project_name=report.training_metadata.get('project_name', 'Unknown'),
            timestamp=report.generated_at,
            total_energy=report.summary.total_energy_kwh,
            total_co2=report.summary.total_co2_kg,
            avg_power=report.summary.average_power_watts,
            duration=report.summary.total_duration_hours,
            samples_per_kwh=report.summary.average_samples_per_kwh,
            equivalent_km=report.summary.equivalent_km_driven,
            metrics_count=len(report.detailed_metrics),
            recommendations_html=recommendations_html,
            region=report.environmental_impact.region_name,
            grid_intensity=report.environmental_impact.regional_average_intensity,
            renewable_percent=renewable_percent,
            grid_type=grid_type,
        )


class MLflowExporter:
    """Export carbon metrics to MLflow."""
    
    def __init__(self):
        """Initialize MLflow exporter."""
        try:
            import mlflow
            self.mlflow = mlflow
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("MLflow not available for carbon metric export")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log carbon metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step number
        """
        if not self.available:
            return
        
        try:
            for name, value in metrics.items():
                self.mlflow.log_metric(f"carbon/{name}", value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")
    
    def log_report_artifact(self, report: CarbonReport, artifact_path: str = "carbon_reports"):
        """Log carbon report as MLflow artifact.
        
        Args:
            report: Carbon report to log
            artifact_path: Path within MLflow artifacts
        """
        if not self.available:
            return
        
        try:
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Export report to temporary file
                report_path = os.path.join(tmp_dir, "carbon_report.json")
                report.to_json(report_path, include_detailed_metrics=True)
                
                # Log as artifact
                self.mlflow.log_artifact(report_path, artifact_path)
                
        except Exception as e:
            logger.error(f"Failed to log report artifact to MLflow: {e}")


class WandbExporter:
    """Export carbon metrics to Weights & Biases."""
    
    def __init__(self):
        """Initialize wandb exporter."""
        try:
            import wandb
            self.wandb = wandb
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("wandb not available for carbon metric export")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log carbon metrics to wandb.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step number
        """
        if not self.available or not self.wandb.run:
            return
        
        try:
            # Prefix metrics with carbon/
            carbon_metrics = {f"carbon/{k}": v for k, v in metrics.items()}
            self.wandb.log(carbon_metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics to wandb: {e}")
    
    def log_report_table(self, report: CarbonReport):
        """Log carbon report as wandb table.
        
        Args:
            report: Carbon report to log
        """
        if not self.available or not self.wandb.run:
            return
        
        try:
            # Create summary table
            summary_data = [
                ["Total Energy (kWh)", f"{report.summary.total_energy_kwh:.3f}"],
                ["Total CO₂ (kg)", f"{report.summary.total_co2_kg:.3f}"],
                ["Average Power (W)", f"{report.summary.average_power_watts:.0f}"],
                ["Training Duration (hrs)", f"{report.summary.total_duration_hours:.1f}"],
                ["Samples per kWh", f"{report.summary.average_samples_per_kwh:.0f}"],
                ["Equivalent km driven", f"{report.summary.equivalent_km_driven:.0f}"],
            ]
            
            table = self.wandb.Table(columns=["Metric", "Value"], data=summary_data)
            self.wandb.log({"carbon/summary": table})
            
        except Exception as e:
            logger.error(f"Failed to log report table to wandb: {e}")