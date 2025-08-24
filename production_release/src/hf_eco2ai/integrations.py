"""Integration modules for external services and frameworks."""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json

from .models import CarbonMetrics, CarbonReport
from .config import CarbonConfig

logger = logging.getLogger(__name__)


class PyTorchLightningIntegration:
    """Integration with PyTorch Lightning for carbon tracking."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize Lightning integration.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        
        try:
            import pytorch_lightning as pl
            self.pl = pl
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("PyTorch Lightning not available")
    
    def create_callback(self):
        """Create Lightning callback for carbon tracking."""
        if not self.available:
            raise ImportError("PyTorch Lightning not available")
        
        from .callback import Eco2AICallback
        
        class Eco2AILightningCallback(self.pl.Callback):
            """Lightning callback for carbon tracking."""
            
            def __init__(self, config: CarbonConfig):
                super().__init__()
                self.eco2ai_callback = Eco2AICallback(config)
                self._step_count = 0
            
            def on_train_start(self, trainer, pl_module):
                """Called when training starts."""
                # Mock trainer args for compatibility
                class MockArgs:
                    def __init__(self):
                        self.num_train_epochs = trainer.max_epochs
                        self.per_device_train_batch_size = trainer.datamodule.batch_size if trainer.datamodule else 32
                        self.learning_rate = 0.001  # Default
                        self.warmup_steps = 0
                        self.weight_decay = 0.0
                    
                    def to_dict(self):
                        return {
                            "max_epochs": self.num_train_epochs,
                            "batch_size": self.per_device_train_batch_size,
                            "learning_rate": self.learning_rate,
                        }
                
                class MockState:
                    def __init__(self):
                        self.global_step = 0
                        self.epoch = 0
                
                class MockControl:
                    def __init__(self):
                        self.should_training_stop = False
                
                self.eco2ai_callback.on_train_begin(
                    MockArgs(), MockState(), MockControl(), pl_module
                )
            
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                """Called after each training batch."""
                self._step_count += 1
                
                class MockState:
                    def __init__(self):
                        self.global_step = self._step_count
                        self.epoch = trainer.current_epoch
                
                logs = {}
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    logs["train_loss"] = float(outputs.loss.item())
                
                self.eco2ai_callback.on_step_end(
                    None, MockState(), None, logs
                )
            
            def on_train_epoch_end(self, trainer, pl_module):
                """Called at the end of each epoch."""
                class MockState:
                    def __init__(self):
                        self.global_step = self._step_count
                        self.epoch = trainer.current_epoch
                
                self.eco2ai_callback.on_epoch_end(
                    None, MockState(), None, {}
                )
            
            def on_train_end(self, trainer, pl_module):
                """Called when training ends."""
                self.eco2ai_callback.on_train_end(None, None, None, {})
        
        return Eco2AILightningCallback(self.config)


class AccelerateIntegration:
    """Integration with HuggingFace Accelerate for distributed training."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize Accelerate integration.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        
        try:
            from accelerate import Accelerator
            self.accelerator_cls = Accelerator
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("Accelerate not available")
    
    def wrap_accelerator(self, accelerator):
        """Wrap accelerator with carbon tracking."""
        if not self.available:
            raise ImportError("Accelerate not available")
        
        from .monitoring import EnergyTracker
        
        # Add carbon tracking to accelerator
        accelerator.carbon_tracker = EnergyTracker(
            gpu_ids=self.config.gpu_ids,
            country=self.config.country,
            region=self.config.region
        )
        
        # Store original methods
        original_prepare = accelerator.prepare
        original_free_memory = getattr(accelerator, 'free_memory', None)
        
        def prepare_with_carbon(*args, **kwargs):
            """Prepare models with carbon tracking."""
            result = original_prepare(*args, **kwargs)
            
            # Start carbon tracking if not already started
            if hasattr(accelerator.carbon_tracker, 'is_tracking') and not accelerator.carbon_tracker.is_tracking:
                accelerator.carbon_tracker.start_tracking()
            
            return result
        
        def free_memory_with_carbon():
            """Free memory and stop carbon tracking."""
            if original_free_memory:
                original_free_memory()
            
            if hasattr(accelerator.carbon_tracker, 'stop_tracking'):
                accelerator.carbon_tracker.stop_tracking()
        
        # Replace methods
        accelerator.prepare = prepare_with_carbon
        accelerator.free_memory = free_memory_with_carbon
        
        return accelerator


class DockerIntegration:
    """Integration with Docker for containerized training carbon tracking."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize Docker integration.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        
        try:
            import docker
            self.docker = docker
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("Docker SDK not available")
    
    def monitor_container(self, container_name: str) -> Dict[str, Any]:
        """Monitor carbon footprint of a Docker container.
        
        Args:
            container_name: Name or ID of container to monitor
            
        Returns:
            Container carbon metrics
        """
        if not self.available:
            raise ImportError("Docker SDK not available")
        
        client = self.docker.from_env()
        
        try:
            container = client.containers.get(container_name)
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Extract relevant metrics
            cpu_percent = self._calculate_cpu_percent(stats)
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100
            
            # Estimate power consumption (simplified)
            # In production, would integrate with actual power monitoring
            estimated_power = cpu_percent * 2.0 + memory_percent * 0.5  # Rough estimate
            
            return {
                "container_name": container_name,
                "cpu_percent": cpu_percent,
                "memory_usage_mb": memory_usage / (1024 * 1024),
                "memory_percent": memory_percent,
                "estimated_power_watts": estimated_power,
                "timestamp": time.time(),
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor container {container_name}: {e}")
            return {}
    
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU percentage from Docker stats."""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * \
                             len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                return cpu_percent
        except (KeyError, ZeroDivisionError):
            pass
        
        return 0.0


class KubernetesIntegration:
    """Integration with Kubernetes for distributed training monitoring."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize Kubernetes integration.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
        
        try:
            from kubernetes import client, config as k8s_config
            self.k8s_client = client
            self.k8s_config = k8s_config
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("Kubernetes client not available")
    
    def monitor_training_job(self, job_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Monitor carbon footprint of a Kubernetes training job.
        
        Args:
            job_name: Name of the training job
            namespace: Kubernetes namespace
            
        Returns:
            Job carbon metrics
        """
        if not self.available:
            raise ImportError("Kubernetes client not available")
        
        try:
            # Load Kubernetes config
            self.k8s_config.load_incluster_config()
        except:
            try:
                self.k8s_config.load_kube_config()
            except Exception as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                return {}
        
        v1 = self.k8s_client.CoreV1Api()
        batch_v1 = self.k8s_client.BatchV1Api()
        
        try:
            # Get job details
            job = batch_v1.read_namespaced_job(job_name, namespace)
            
            # Get pods for this job
            pods = v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"job-name={job_name}"
            )
            
            total_cpu_usage = 0
            total_memory_usage = 0
            pod_count = len(pods.items)
            
            # Get resource usage for each pod
            for pod in pods.items:
                # In production, would use metrics server or Prometheus
                # This is a simplified example
                if pod.spec.containers:
                    for container in pod.spec.containers:
                        if container.resources and container.resources.requests:
                            cpu_req = container.resources.requests.get('cpu', '0')
                            memory_req = container.resources.requests.get('memory', '0')
                            
                            # Parse CPU (simplified)
                            if 'm' in cpu_req:
                                cpu_millicores = int(cpu_req.replace('m', ''))
                                total_cpu_usage += cpu_millicores / 1000
                            else:
                                total_cpu_usage += float(cpu_req or 0)
                            
                            # Parse memory (simplified)
                            if 'Gi' in memory_req:
                                memory_gb = int(memory_req.replace('Gi', ''))
                                total_memory_usage += memory_gb * 1024
                            elif 'Mi' in memory_req:
                                memory_mb = int(memory_req.replace('Mi', ''))
                                total_memory_usage += memory_mb
            
            # Estimate power consumption
            estimated_power = total_cpu_usage * 100 + (total_memory_usage / 1024) * 10  # Rough estimate
            
            return {
                "job_name": job_name,
                "namespace": namespace,
                "pod_count": pod_count,
                "total_cpu_cores": total_cpu_usage,
                "total_memory_mb": total_memory_usage,
                "estimated_power_watts": estimated_power,
                "job_status": job.status.conditions[-1].type if job.status.conditions else "Unknown",
                "timestamp": time.time(),
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor Kubernetes job {job_name}: {e}")
            return {}


class SLURMIntegration:
    """Integration with SLURM for HPC cluster carbon tracking."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize SLURM integration.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
    
    def monitor_slurm_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor carbon footprint of a SLURM job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Job carbon metrics
        """
        import subprocess
        
        try:
            # Get job information
            result = subprocess.run([
                'scontrol', 'show', 'job', job_id
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.error(f"Failed to get SLURM job info: {result.stderr}")
                return {}
            
            # Parse job info (simplified)
            job_info = {}
            for line in result.stdout.split('\n'):
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        job_info[key] = value
            
            # Get resource allocation
            num_nodes = int(job_info.get('NumNodes', '0'))
            num_cpus = int(job_info.get('NumCPUs', '0'))
            
            # Estimate power consumption
            # Typical HPC node: ~200-500W per node
            estimated_power_per_node = 350  # Watts
            total_estimated_power = num_nodes * estimated_power_per_node
            
            # Get job duration
            start_time = job_info.get('StartTime', '')
            end_time = job_info.get('EndTime', '')
            
            return {
                "job_id": job_id,
                "num_nodes": num_nodes,
                "num_cpus": num_cpus,
                "estimated_power_watts": total_estimated_power,
                "start_time": start_time,
                "end_time": end_time,
                "job_state": job_info.get('JobState', 'Unknown'),
                "partition": job_info.get('Partition', 'Unknown'),
                "timestamp": time.time(),
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor SLURM job {job_id}: {e}")
            return {}


class CloudProviderIntegration:
    """Integration with cloud providers for carbon tracking."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize cloud provider integration.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
    
    def get_aws_carbon_data(self, instance_type: str, region: str) -> Dict[str, Any]:
        """Get carbon data for AWS instances.
        
        Args:
            instance_type: EC2 instance type
            region: AWS region
            
        Returns:
            Carbon intensity and instance data
        """
        # AWS carbon intensity by region (g CO₂/kWh)
        # These are approximate values - in production would use AWS Carbon Footprint Tool
        aws_carbon_intensity = {
            "us-east-1": 415,      # Virginia
            "us-east-2": 500,      # Ohio
            "us-west-1": 250,      # N. California
            "us-west-2": 120,      # Oregon (high renewable)
            "eu-west-1": 350,      # Ireland
            "eu-central-1": 475,   # Frankfurt
            "ap-southeast-1": 600, # Singapore
            "ap-northeast-1": 550, # Tokyo
        }
        
        # Typical power consumption by instance type (watts)
        instance_power = {
            "t3.micro": 10,
            "t3.small": 20,
            "t3.medium": 30,
            "t3.large": 50,
            "t3.xlarge": 80,
            "m5.large": 100,
            "m5.xlarge": 150,
            "m5.2xlarge": 200,
            "m5.4xlarge": 300,
            "c5.large": 120,
            "c5.xlarge": 180,
            "c5.2xlarge": 250,
            "r5.large": 130,
            "r5.xlarge": 200,
            "p3.2xlarge": 1000,    # GPU instance
            "p3.8xlarge": 2500,    # GPU instance
            "p4d.24xlarge": 4000,  # GPU instance
        }
        
        carbon_intensity = aws_carbon_intensity.get(region, 400)
        power_watts = instance_power.get(instance_type, 100)
        
        return {
            "provider": "aws",
            "instance_type": instance_type,
            "region": region,
            "carbon_intensity_g_co2_kwh": carbon_intensity,
            "estimated_power_watts": power_watts,
            "renewable_percentage": self._get_aws_renewable_percentage(region),
        }
    
    def get_gcp_carbon_data(self, machine_type: str, zone: str) -> Dict[str, Any]:
        """Get carbon data for Google Cloud instances.
        
        Args:
            machine_type: GCP machine type
            zone: GCP zone
            
        Returns:
            Carbon intensity and instance data
        """
        # GCP carbon intensity by region
        gcp_carbon_intensity = {
            "us-central1": 450,    # Iowa
            "us-east1": 415,       # South Carolina
            "us-west1": 85,        # Oregon (very clean)
            "europe-west1": 350,   # Belgium
            "europe-west4": 475,   # Netherlands
            "asia-southeast1": 600, # Singapore
            "asia-northeast1": 550, # Tokyo
        }
        
        # Extract region from zone
        region = '-'.join(zone.split('-')[:-1])
        carbon_intensity = gcp_carbon_intensity.get(region, 400)
        
        # Estimate power consumption (simplified)
        power_watts = 100
        if "n1-standard" in machine_type:
            cpu_count = int(machine_type.split('-')[-1])
            power_watts = cpu_count * 15
        
        return {
            "provider": "gcp",
            "machine_type": machine_type,
            "zone": zone,
            "region": region,
            "carbon_intensity_g_co2_kwh": carbon_intensity,
            "estimated_power_watts": power_watts,
            "renewable_percentage": self._get_gcp_renewable_percentage(region),
        }
    
    def _get_aws_renewable_percentage(self, region: str) -> float:
        """Get renewable energy percentage for AWS region."""
        aws_renewable = {
            "us-west-2": 85,  # Oregon
            "us-west-1": 45,  # California
            "eu-west-1": 65,  # Ireland
            "us-east-1": 25,  # Virginia
        }
        return aws_renewable.get(region, 30)
    
    def _get_gcp_renewable_percentage(self, region: str) -> float:
        """Get renewable energy percentage for GCP region."""
        gcp_renewable = {
            "us-west1": 95,      # Oregon
            "europe-west1": 75,  # Belgium
            "us-central1": 55,   # Iowa
        }
        return gcp_renewable.get(region, 50)


class NotificationIntegration:
    """Integration for sending carbon tracking notifications."""
    
    def __init__(self, config: Optional[CarbonConfig] = None):
        """Initialize notification integration.
        
        Args:
            config: Carbon tracking configuration
        """
        self.config = config or CarbonConfig()
    
    def send_slack_notification(self, webhook_url: str, message: Dict[str, Any]):
        """Send notification to Slack.
        
        Args:
            webhook_url: Slack webhook URL
            message: Message content
        """
        try:
            import requests
            
            payload = {
                "text": f"🌱 Carbon Tracking Alert: {message['title']}",
                "attachments": [{
                    "color": "warning" if message.get('alert_type') == 'warning' else "good",
                    "fields": [
                        {
                            "title": "Energy Consumption",
                            "value": f"{message.get('energy_kwh', 0):.2f} kWh",
                            "short": True
                        },
                        {
                            "title": "CO₂ Emissions",
                            "value": f"{message.get('co2_kg', 0):.2f} kg",
                            "short": True
                        },
                        {
                            "title": "Project",
                            "value": message.get('project_name', 'Unknown'),
                            "short": True
                        },
                        {
                            "title": "Duration",
                            "value": f"{message.get('duration_hours', 0):.1f} hours",
                            "short": True
                        }
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Sent Slack notification successfully")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def send_email_notification(self, smtp_config: Dict[str, str], message: Dict[str, Any]):
        """Send email notification.
        
        Args:
            smtp_config: SMTP configuration
            message: Message content
        """
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = smtp_config['username']
            msg['To'] = smtp_config['to']
            msg['Subject'] = f"Carbon Tracking Alert: {message['title']}"
            
            body = f"""
            Carbon Tracking Alert
            
            Project: {message.get('project_name', 'Unknown')}
            Energy Consumption: {message.get('energy_kwh', 0):.2f} kWh
            CO₂ Emissions: {message.get('co2_kg', 0):.2f} kg
            Duration: {message.get('duration_hours', 0):.1f} hours
            
            {message.get('description', '')}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_config['host'], smtp_config['port'])
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("Sent email notification successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")