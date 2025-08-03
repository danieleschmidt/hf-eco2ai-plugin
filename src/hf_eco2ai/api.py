"""REST API server for carbon tracking management and monitoring."""

import time
import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from .config import CarbonConfig
from .models import CarbonMetrics, CarbonReport
from .storage import CarbonDataStorage, CarbonDataAggregator
from .monitoring import EnergyTracker
from .exporters import ReportExporter

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class CarbonConfigRequest(BaseModel):
    """Request model for carbon configuration."""
    project_name: str = Field(..., description="Project name")
    country: str = Field(default="USA", description="Country for carbon intensity")
    region: str = Field(default="California", description="Region for carbon intensity")
    gpu_ids: List[int] = Field(default=[], description="GPU IDs to monitor")
    log_level: str = Field(default="EPOCH", description="Logging level")
    export_prometheus: bool = Field(default=False, description="Enable Prometheus export")
    save_report: bool = Field(default=True, description="Save reports to disk")
    max_co2_kg: Optional[float] = Field(default=None, description="Carbon budget limit")

class TrainingSessionRequest(BaseModel):
    """Request model for starting a training session."""
    session_id: str = Field(..., description="Unique session identifier")
    config: CarbonConfigRequest
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")

class MetricSubmission(BaseModel):
    """Model for submitting metrics via API."""
    session_id: str
    timestamp: float
    step: Optional[int] = None
    epoch: Optional[int] = None
    energy_kwh: float
    power_watts: float
    co2_kg: float
    samples_processed: int = 0
    loss: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = {}

class TrainingSessionResponse(BaseModel):
    """Response model for training session."""
    session_id: str
    status: str  # "active", "completed", "failed"
    config: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    current_metrics: Optional[Dict[str, Any]] = None
    total_energy_kwh: float = 0.0
    total_co2_kg: float = 0.0

class CarbonAPIServer:
    """REST API server for carbon tracking."""
    
    def __init__(self, storage_path: str = "carbon_tracking.db", 
                 host: str = "localhost", port: int = 8000):
        """Initialize API server.
        
        Args:
            storage_path: Path to SQLite database
            host: Host to bind server
            port: Port to bind server
        """
        self.storage = CarbonDataStorage(storage_path)
        self.aggregator = CarbonDataAggregator(self.storage)
        self.report_exporter = ReportExporter()
        
        self.host = host
        self.port = port
        
        # Active training sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.energy_trackers: Dict[str, EnergyTracker] = {}
        
        # Create FastAPI app
        self.app = FastAPI(
            title="HF Eco2AI Carbon Tracking API",
            description="REST API for carbon footprint tracking during ML training",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
        logger.info(f"Initialized Carbon API server on {host}:{port}")
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "service": "HF Eco2AI Carbon Tracking API",
                "version": "1.0.0",
                "status": "active",
                "timestamp": time.time(),
                "active_sessions": len(self.active_sessions),
                "endpoints": {
                    "health": "/health",
                    "sessions": "/sessions",
                    "metrics": "/metrics",
                    "reports": "/reports",
                    "projects": "/projects",
                    "docs": "/docs"
                }
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "database": "connected",
                "active_sessions": len(self.active_sessions)
            }
        
        @self.app.post("/sessions", response_model=TrainingSessionResponse)
        async def start_session(request: TrainingSessionRequest, background_tasks: BackgroundTasks):
            """Start a new carbon tracking session."""
            session_id = request.session_id
            
            if session_id in self.active_sessions:
                raise HTTPException(status_code=400, detail=f"Session {session_id} already exists")
            
            # Create carbon config
            config = CarbonConfig(
                project_name=request.config.project_name,
                country=request.config.country,
                region=request.config.region,
                gpu_ids=request.config.gpu_ids or "auto",
                log_level=request.config.log_level,
                export_prometheus=request.config.export_prometheus,
                save_report=request.config.save_report,
                max_co2_kg=request.config.max_co2_kg
            )
            
            # Create energy tracker
            energy_tracker = EnergyTracker(
                gpu_ids=config.gpu_ids,
                country=config.country,
                region=config.region
            )
            
            # Start tracking
            energy_tracker.start_tracking()
            
            # Store session
            session_data = {
                "session_id": session_id,
                "status": "active",
                "config": config.to_dict(),
                "start_time": time.time(),
                "end_time": None,
                "metadata": request.metadata,
                "metrics_count": 0
            }
            
            self.active_sessions[session_id] = session_data
            self.energy_trackers[session_id] = energy_tracker
            
            # Schedule cleanup after 24 hours
            background_tasks.add_task(self._cleanup_session_after_timeout, session_id, 24 * 3600)
            
            return TrainingSessionResponse(
                session_id=session_id,
                status="active",
                config=session_data["config"],
                start_time=session_data["start_time"]
            )
        
        @self.app.get("/sessions/{session_id}", response_model=TrainingSessionResponse)
        async def get_session(session_id: str):
            """Get information about a training session."""
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            current_metrics = None
            
            # Get current metrics if session is active
            if session["status"] == "active" and session_id in self.energy_trackers:
                tracker = self.energy_trackers[session_id]
                power, energy, co2 = tracker.get_current_consumption()
                current_metrics = {
                    "power_watts": power,
                    "energy_kwh": energy,
                    "co2_kg": co2,
                    "timestamp": time.time()
                }
            
            return TrainingSessionResponse(
                session_id=session_id,
                status=session["status"],
                config=session["config"],
                start_time=session["start_time"],
                end_time=session.get("end_time"),
                current_metrics=current_metrics,
                total_energy_kwh=session.get("total_energy_kwh", 0.0),
                total_co2_kg=session.get("total_co2_kg", 0.0)
            )
        
        @self.app.post("/sessions/{session_id}/stop")
        async def stop_session(session_id: str):
            """Stop a carbon tracking session."""
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            if session["status"] != "active":
                raise HTTPException(status_code=400, detail=f"Session {session_id} is not active")
            
            # Stop energy tracking
            if session_id in self.energy_trackers:
                tracker = self.energy_trackers[session_id]
                power, energy, co2 = tracker.get_current_consumption()
                tracker.stop_tracking()
                
                # Update session
                session["status"] = "completed"
                session["end_time"] = time.time()
                session["total_energy_kwh"] = energy
                session["total_co2_kg"] = co2
                
                # Clean up tracker
                del self.energy_trackers[session_id]
            
            return {"message": f"Session {session_id} stopped successfully"}
        
        @self.app.get("/sessions")
        async def list_sessions(status: Optional[str] = Query(None, description="Filter by status")):
            """List all training sessions."""
            sessions = list(self.active_sessions.values())
            
            if status:
                sessions = [s for s in sessions if s["status"] == status]
            
            return {
                "sessions": sessions,
                "total": len(sessions),
                "active": len([s for s in sessions if s["status"] == "active"])
            }
        
        @self.app.post("/metrics/{session_id}")
        async def submit_metrics(session_id: str, metric: MetricSubmission):
            """Submit carbon metrics for a session."""
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            if session["status"] != "active":
                raise HTTPException(status_code=400, detail=f"Session {session_id} is not active")
            
            # Create carbon metric
            carbon_metric = CarbonMetrics(
                timestamp=metric.timestamp,
                step=metric.step,
                epoch=metric.epoch,
                energy_kwh=metric.energy_kwh,
                power_watts=metric.power_watts,
                co2_kg=metric.co2_kg,
                samples_processed=metric.samples_processed,
                loss=metric.loss
            )
            
            # Store metric
            project_name = session["config"]["project_name"]
            self.storage.store_metric(carbon_metric, project_name, session_id, session_id)
            
            # Update session metrics count
            session["metrics_count"] += 1
            
            return {"message": "Metrics submitted successfully"}
        
        @self.app.get("/metrics/{session_id}")
        async def get_session_metrics(
            session_id: str,
            limit: Optional[int] = Query(None, description="Limit number of results"),
            start_time: Optional[float] = Query(None, description="Start timestamp"),
            end_time: Optional[float] = Query(None, description="End timestamp")
        ):
            """Get metrics for a specific session."""
            metrics = self.storage.get_metrics(
                run_id=session_id,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            return {
                "session_id": session_id,
                "metrics": metrics,
                "count": len(metrics)
            }
        
        @self.app.get("/projects")
        async def list_projects():
            """List all projects with summary statistics."""
            # Get all unique project names
            with self.storage._get_connection() as conn:
                cursor = conn.execute("SELECT DISTINCT project_name FROM carbon_metrics")
                project_names = [row[0] for row in cursor.fetchall()]
            
            projects = []
            for project_name in project_names:
                summary = self.storage.get_project_summary(project_name)
                projects.append(summary)
            
            return {
                "projects": projects,
                "total": len(projects)
            }
        
        @self.app.get("/projects/{project_name}")
        async def get_project_details(project_name: str):
            """Get detailed information about a project."""
            summary = self.storage.get_project_summary(project_name)
            
            if not summary["metrics"]["total_metrics"]:
                raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
            
            # Get efficiency trends
            trends = self.storage.get_efficiency_trends(project_name, days=30)
            
            # Get efficiency percentiles
            percentiles = self.aggregator.get_efficiency_percentiles(project_name)
            
            return {
                "project_name": project_name,
                "summary": summary,
                "efficiency_trends": trends,
                "efficiency_percentiles": percentiles
            }
        
        @self.app.get("/projects/{project_name}/export")
        async def export_project_data(
            project_name: str,
            format: str = Query("json", description="Export format: json, csv"),
            background_tasks: BackgroundTasks = None
        ):
            """Export project data."""
            if format not in ["json", "csv"]:
                raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
            
            # Generate export file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{project_name}_export_{timestamp}.{format}"
            output_path = Path(f"/tmp/{filename}")
            
            try:
                self.storage.export_data(str(output_path), project_name, format)
                
                # Schedule file cleanup after 1 hour
                if background_tasks:
                    background_tasks.add_task(self._cleanup_file, str(output_path), 3600)
                
                return FileResponse(
                    path=str(output_path),
                    filename=filename,
                    media_type="application/json" if format == "json" else "text/csv"
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
        
        @self.app.get("/reports")
        async def list_reports(
            project_name: Optional[str] = Query(None, description="Filter by project"),
            limit: Optional[int] = Query(20, description="Limit number of results")
        ):
            """List carbon reports."""
            reports = self.storage.get_reports(project_name=project_name)
            
            if limit:
                reports = reports[:limit]
            
            return {
                "reports": reports,
                "total": len(reports)
            }
        
        @self.app.get("/reports/{report_id}")
        async def get_report(report_id: str):
            """Get a specific carbon report."""
            with self.storage._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM carbon_reports WHERE report_id = ?",
                    (report_id,)
                )
                report_row = cursor.fetchone()
            
            if not report_row:
                raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
            
            # Parse the full report data
            report_data = json.loads(report_row["report_data"])
            
            return report_data
        
        @self.app.post("/analytics/aggregate")
        async def aggregate_data(
            project_name: Optional[str] = None,
            days: int = Query(30, description="Number of days to aggregate"),
            interval: str = Query("day", description="Time interval: hour, day, week")
        ):
            """Generate aggregated analytics data."""
            if project_name:
                # Project-specific aggregation
                data = self.aggregator.aggregate_by_time(project_name, interval, days)
                return {
                    "project_name": project_name,
                    "interval": interval,
                    "days": days,
                    "data": data
                }
            else:
                # Cross-project aggregation
                data = self.aggregator.aggregate_by_project(days)
                return {
                    "interval": interval,
                    "days": days,
                    "projects": data
                }
        
        @self.app.post("/optimize/{project_name}")
        async def get_optimization_recommendations(project_name: str):
            """Get optimization recommendations for a project."""
            summary = self.storage.get_project_summary(project_name)
            
            if not summary["metrics"]["total_metrics"]:
                raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
            
            # Generate recommendations based on data
            recommendations = []
            
            avg_samples_per_kwh = summary["metrics"].get("total_energy", 0)
            if avg_samples_per_kwh > 0:
                total_samples = summary["metrics"].get("total_metrics", 0) * 1000  # Estimate
                efficiency = total_samples / avg_samples_per_kwh
                
                if efficiency < 1000:  # Low efficiency
                    recommendations.append({
                        "title": "Enable Mixed Precision Training",
                        "description": "Use FP16 to reduce energy consumption by 30-40%",
                        "potential_reduction": 35,
                        "category": "training_optimization"
                    })
                
                if avg_samples_per_kwh > 5:  # High energy usage
                    recommendations.append({
                        "title": "Optimize Batch Size",
                        "description": "Increase batch size to improve GPU utilization",
                        "potential_reduction": 20,
                        "category": "hardware_optimization"
                    })
            
            return {
                "project_name": project_name,
                "recommendations": recommendations,
                "analysis_date": time.time()
            }
    
    async def _cleanup_session_after_timeout(self, session_id: str, timeout_seconds: int):
        """Clean up session after timeout."""
        await asyncio.sleep(timeout_seconds)
        
        if session_id in self.active_sessions:
            if self.active_sessions[session_id]["status"] == "active":
                # Auto-stop the session
                if session_id in self.energy_trackers:
                    self.energy_trackers[session_id].stop_tracking()
                    del self.energy_trackers[session_id]
                
                self.active_sessions[session_id]["status"] = "timeout"
                self.active_sessions[session_id]["end_time"] = time.time()
                
                logger.warning(f"Session {session_id} timed out after {timeout_seconds} seconds")
    
    async def _cleanup_file(self, file_path: str, delay_seconds: int):
        """Clean up temporary file after delay."""
        await asyncio.sleep(delay_seconds)
        
        try:
            Path(file_path).unlink(missing_ok=True)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up file {file_path}: {e}")
    
    def run(self, debug: bool = False):
        """Run the API server.
        
        Args:
            debug: Enable debug mode
        """
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="debug" if debug else "info",
            access_log=True
        )
    
    def get_app(self):
        """Get the FastAPI app instance for testing."""
        return self.app


# CLI command to start the API server
def start_api_server(host: str = "localhost", port: int = 8000, 
                    storage_path: str = "carbon_tracking.db", debug: bool = False):
    """Start the Carbon Tracking API server.
    
    Args:
        host: Host to bind server
        port: Port to bind server  
        storage_path: Path to SQLite database
        debug: Enable debug mode
    """
    server = CarbonAPIServer(storage_path=storage_path, host=host, port=port)
    
    logger.info(f"Starting Carbon Tracking API server on {host}:{port}")
    logger.info(f"Database: {storage_path}")
    logger.info(f"API Documentation: http://{host}:{port}/docs")
    
    server.run(debug=debug)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Carbon Tracking API Server")
    parser.add_argument("--host", default="localhost", help="Host to bind server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server")
    parser.add_argument("--storage", default="carbon_tracking.db", help="SQLite database path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    start_api_server(
        host=args.host,
        port=args.port,
        storage_path=args.storage,
        debug=args.debug
    )