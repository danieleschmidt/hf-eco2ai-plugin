"""Data storage and persistence layer for carbon tracking metrics."""

import sqlite3
import json
import time
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
from contextlib import contextmanager
import logging
from dataclasses import asdict

from .models import CarbonMetrics, CarbonReport

logger = logging.getLogger(__name__)


class CarbonDataStorage:
    """Persistent storage for carbon tracking data using SQLite."""
    
    def __init__(self, db_path: str = "carbon_tracking.db"):
        """Initialize carbon data storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_database()
        
        logger.info(f"Initialized carbon data storage: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Create metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS carbon_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_name TEXT NOT NULL,
                    experiment_name TEXT,
                    run_id TEXT,
                    timestamp REAL NOT NULL,
                    step INTEGER,
                    epoch INTEGER,
                    energy_kwh REAL NOT NULL,
                    cumulative_energy_kwh REAL NOT NULL,
                    power_watts REAL NOT NULL,
                    co2_kg REAL NOT NULL,
                    cumulative_co2_kg REAL NOT NULL,
                    grid_intensity REAL NOT NULL,
                    samples_processed INTEGER,
                    samples_per_kwh REAL,
                    duration_seconds REAL,
                    model_parameters INTEGER,
                    batch_size INTEGER,
                    learning_rate REAL,
                    loss REAL,
                    location TEXT,
                    gpu_metrics TEXT,  -- JSON for GPU-specific data
                    raw_data TEXT,     -- Full JSON of the metric
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create reports table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS carbon_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE NOT NULL,
                    project_name TEXT NOT NULL,
                    experiment_name TEXT,
                    run_id TEXT,
                    total_energy_kwh REAL NOT NULL,
                    total_co2_kg REAL NOT NULL,
                    total_duration_hours REAL NOT NULL,
                    total_steps INTEGER,
                    total_epochs INTEGER,
                    total_samples INTEGER,
                    average_power_watts REAL,
                    peak_power_watts REAL,
                    model_parameters INTEGER,
                    final_loss REAL,
                    report_data TEXT,  -- Full JSON of the report
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_project_timestamp ON carbon_metrics(project_name, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON carbon_metrics(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_project ON carbon_reports(project_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_run_id ON carbon_reports(run_id)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def store_metric(self, metric: CarbonMetrics, project_name: str = "default", 
                     experiment_name: Optional[str] = None, run_id: Optional[str] = None):
        """Store a carbon metric.
        
        Args:
            metric: Carbon metric to store
            project_name: Project name
            experiment_name: Experiment name (optional)
            run_id: Run ID (optional)
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO carbon_metrics (
                    project_name, experiment_name, run_id, timestamp, step, epoch,
                    energy_kwh, cumulative_energy_kwh, power_watts, co2_kg, cumulative_co2_kg,
                    grid_intensity, samples_processed, samples_per_kwh, duration_seconds,
                    model_parameters, batch_size, learning_rate, loss, location,
                    gpu_metrics, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project_name, experiment_name, run_id, metric.timestamp,
                metric.step, metric.epoch, metric.energy_kwh, metric.cumulative_energy_kwh,
                metric.power_watts, metric.co2_kg, metric.cumulative_co2_kg,
                metric.grid_intensity, metric.samples_processed, metric.samples_per_kwh,
                metric.duration_seconds, metric.model_parameters, metric.batch_size,
                metric.learning_rate, metric.loss, metric.location,
                json.dumps({
                    "power_watts": metric.gpu_power_watts,
                    "utilization": metric.gpu_utilization,
                    "memory_used": metric.gpu_memory_used,
                    "temperature": metric.gpu_temperature,
                }),
                json.dumps(metric.to_dict())
            ))
            conn.commit()
    
    def store_report(self, report: CarbonReport, project_name: str = "default"):
        """Store a carbon report.
        
        Args:
            report: Carbon report to store
            project_name: Project name
        """
        with self._get_connection() as conn:
            experiment_name = report.training_metadata.get("experiment_name")
            run_id = report.training_metadata.get("run_id")
            
            conn.execute("""
                INSERT OR REPLACE INTO carbon_reports (
                    report_id, project_name, experiment_name, run_id,
                    total_energy_kwh, total_co2_kg, total_duration_hours,
                    total_steps, total_epochs, total_samples,
                    average_power_watts, peak_power_watts, model_parameters,
                    final_loss, report_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_id, project_name, experiment_name, run_id,
                report.summary.total_energy_kwh, report.summary.total_co2_kg,
                report.summary.total_duration_hours, report.summary.total_steps,
                report.summary.total_epochs, report.summary.total_samples,
                report.summary.average_power_watts, report.summary.peak_power_watts,
                report.summary.model_parameters, report.summary.final_loss,
                report.to_json(include_detailed_metrics=True)
            ))
            conn.commit()
    
    def get_metrics(self, project_name: Optional[str] = None, run_id: Optional[str] = None,
                   start_time: Optional[float] = None, end_time: Optional[float] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve carbon metrics with filtering.
        
        Args:
            project_name: Filter by project name
            run_id: Filter by run ID
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum number of results
            
        Returns:
            List of metric dictionaries
        """
        query = "SELECT * FROM carbon_metrics WHERE 1=1"
        params = []
        
        if project_name:
            query += " AND project_name = ?"
            params.append(project_name)
        
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_reports(self, project_name: Optional[str] = None, 
                   experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve carbon reports with filtering.
        
        Args:
            project_name: Filter by project name
            experiment_name: Filter by experiment name
            
        Returns:
            List of report dictionaries
        """
        query = "SELECT * FROM carbon_reports WHERE 1=1"
        params = []
        
        if project_name:
            query += " AND project_name = ?"
            params.append(project_name)
        
        if experiment_name:
            query += " AND experiment_name = ?"
            params.append(experiment_name)
        
        query += " ORDER BY created_at DESC"
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_project_summary(self, project_name: str) -> Dict[str, Any]:
        """Get summary statistics for a project.
        
        Args:
            project_name: Project name
            
        Returns:
            Project summary statistics
        """
        with self._get_connection() as conn:
            # Get metrics summary
            metrics_cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_metrics,
                    SUM(cumulative_energy_kwh) as total_energy,
                    SUM(cumulative_co2_kg) as total_co2,
                    AVG(power_watts) as avg_power,
                    MAX(power_watts) as peak_power,
                    MIN(timestamp) as first_measurement,
                    MAX(timestamp) as last_measurement
                FROM carbon_metrics 
                WHERE project_name = ?
            """, (project_name,))
            
            metrics_row = metrics_cursor.fetchone()
            
            # Get reports summary
            reports_cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_reports,
                    AVG(total_energy_kwh) as avg_energy_per_run,
                    AVG(total_co2_kg) as avg_co2_per_run,
                    AVG(total_duration_hours) as avg_duration_hours
                FROM carbon_reports 
                WHERE project_name = ?
            """, (project_name,))
            
            reports_row = reports_cursor.fetchone()
            
            return {
                "project_name": project_name,
                "metrics": dict(metrics_row) if metrics_row else {},
                "reports": dict(reports_row) if reports_row else {},
                "time_range": {
                    "start": metrics_row["first_measurement"] if metrics_row and metrics_row["first_measurement"] else None,
                    "end": metrics_row["last_measurement"] if metrics_row and metrics_row["last_measurement"] else None,
                }
            }
    
    def get_efficiency_trends(self, project_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get efficiency trends over time.
        
        Args:
            project_name: Project name
            days: Number of days to analyze
            
        Returns:
            List of daily efficiency metrics
        """
        cutoff_time = time.time() - (days * 24 * 3600)
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    DATE(datetime(timestamp, 'unixepoch')) as date,
                    AVG(samples_per_kwh) as avg_samples_per_kwh,
                    AVG(power_watts) as avg_power,
                    COUNT(*) as measurements
                FROM carbon_metrics 
                WHERE project_name = ? AND timestamp >= ?
                GROUP BY DATE(datetime(timestamp, 'unixepoch'))
                ORDER BY date ASC
            """, (project_name, cutoff_time))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to manage storage.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        with self._get_connection() as conn:
            # Delete old metrics
            metrics_cursor = conn.execute(
                "DELETE FROM carbon_metrics WHERE timestamp < ?", 
                (cutoff_time,)
            )
            
            # Delete old reports (keep reports longer - 1 year)
            report_cutoff = time.time() - (365 * 24 * 3600)
            reports_cursor = conn.execute(
                "DELETE FROM carbon_reports WHERE created_at < datetime(?, 'unixepoch')", 
                (report_cutoff,)
            )
            
            conn.commit()
            
            logger.info(f"Cleaned up {metrics_cursor.rowcount} old metrics and {reports_cursor.rowcount} old reports")
    
    def export_data(self, output_path: str, project_name: Optional[str] = None, 
                   format: str = "json"):
        """Export data to file.
        
        Args:
            output_path: Output file path
            project_name: Filter by project name (optional)
            format: Export format (json, csv)
        """
        metrics = self.get_metrics(project_name=project_name)
        reports = self.get_reports(project_name=project_name)
        
        export_data = {
            "metrics": metrics,
            "reports": reports,
            "exported_at": time.time(),
            "export_info": {
                "project_name": project_name,
                "total_metrics": len(metrics),
                "total_reports": len(reports),
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format == "csv":
            import csv
            
            # Export metrics to CSV
            metrics_path = output_path.with_suffix('.metrics.csv')
            with open(metrics_path, 'w', newline='') as f:
                if metrics:
                    writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
                    writer.writeheader()
                    writer.writerows(metrics)
            
            # Export reports to CSV
            reports_path = output_path.with_suffix('.reports.csv')
            with open(reports_path, 'w', newline='') as f:
                if reports:
                    # Flatten report data for CSV
                    flattened_reports = []
                    for report in reports:
                        flat_report = {k: v for k, v in report.items() if k != 'report_data'}
                        flattened_reports.append(flat_report)
                    
                    writer = csv.DictWriter(f, fieldnames=flattened_reports[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened_reports)
        
        logger.info(f"Exported data to {output_path} in {format} format")


class CarbonDataCache:
    """In-memory cache for carbon tracking data."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
        
        logger.info(f"Initialized carbon data cache (max_size={max_size}, ttl={ttl}s)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None
        
        # Check expiration
        if time.time() - self._timestamps[key] > self.ttl:
            self._remove(key)
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any):
        """Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove oldest items if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
            self._remove(oldest_key)
        
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def _remove(self, key: str):
        """Remove item from cache."""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
    
    def clear(self):
        """Clear all cached items."""
        self._cache.clear()
        self._timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def cleanup_expired(self):
        """Remove expired items from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            self._remove(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")


class CarbonDataAggregator:
    """Aggregate carbon data for analytics and reporting."""
    
    def __init__(self, storage: CarbonDataStorage):
        """Initialize aggregator.
        
        Args:
            storage: Data storage instance
        """
        self.storage = storage
    
    def aggregate_by_project(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """Aggregate data by project over time period.
        
        Args:
            days: Number of days to aggregate
            
        Returns:
            Project aggregation data
        """
        cutoff_time = time.time() - (days * 24 * 3600)
        
        with self.storage._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    project_name,
                    COUNT(*) as total_measurements,
                    SUM(energy_kwh) as total_energy,
                    SUM(co2_kg) as total_co2,
                    AVG(power_watts) as avg_power,
                    MAX(power_watts) as peak_power,
                    AVG(samples_per_kwh) as avg_efficiency,
                    COUNT(DISTINCT run_id) as total_runs
                FROM carbon_metrics 
                WHERE timestamp >= ?
                GROUP BY project_name
                ORDER BY total_energy DESC
            """, (cutoff_time,))
            
            return {
                row["project_name"]: dict(row)
                for row in cursor.fetchall()
            }
    
    def aggregate_by_time(self, project_name: str, 
                         interval: str = "day", days: int = 30) -> List[Dict[str, Any]]:
        """Aggregate data by time intervals.
        
        Args:
            project_name: Project to aggregate
            interval: Time interval (hour, day, week)
            days: Number of days to aggregate
            
        Returns:
            Time-series aggregation data
        """
        cutoff_time = time.time() - (days * 24 * 3600)
        
        # SQL time grouping based on interval
        if interval == "hour":
            time_group = "datetime(timestamp, 'unixepoch', 'start of hour')"
        elif interval == "day":
            time_group = "date(timestamp, 'unixepoch')"
        elif interval == "week":
            time_group = "date(timestamp, 'unixepoch', 'weekday 0', '-6 days')"
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        
        with self.storage._get_connection() as conn:
            cursor = conn.execute(f"""
                SELECT 
                    {time_group} as time_period,
                    SUM(energy_kwh) as total_energy,
                    SUM(co2_kg) as total_co2,
                    AVG(power_watts) as avg_power,
                    MAX(power_watts) as peak_power,
                    AVG(samples_per_kwh) as avg_efficiency,
                    COUNT(*) as measurements
                FROM carbon_metrics 
                WHERE project_name = ? AND timestamp >= ?
                GROUP BY {time_group}
                ORDER BY time_period ASC
            """, (project_name, cutoff_time))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_efficiency_percentiles(self, project_name: str) -> Dict[str, float]:
        """Calculate efficiency percentiles for a project.
        
        Args:
            project_name: Project name
            
        Returns:
            Efficiency percentiles
        """
        with self.storage._get_connection() as conn:
            cursor = conn.execute("""
                SELECT samples_per_kwh 
                FROM carbon_metrics 
                WHERE project_name = ? AND samples_per_kwh IS NOT NULL
                ORDER BY samples_per_kwh ASC
            """, (project_name,))
            
            values = [row[0] for row in cursor.fetchall()]
            
            if not values:
                return {}
            
            def percentile(data, p):
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = k - f
                if f == len(data) - 1:
                    return data[f]
                return data[f] * (1 - c) + data[f + 1] * c
            
            return {
                "p25": percentile(values, 25),
                "p50": percentile(values, 50),
                "p75": percentile(values, 75),
                "p90": percentile(values, 90),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }