"""Distributed Processing Engine for Enterprise-Scale Carbon Analytics.

This module implements a comprehensive distributed processing engine with Apache Kafka
integration, Dask/Ray distributed computing, microservices architecture, Kubernetes
native support, and event-driven messaging for carbon metrics processing.
"""

import asyncio
import logging
import time
import json
import uuid
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import concurrent.futures
import multiprocessing as mp
from pathlib import Path
import socket
import subprocess

try:
    import kafka
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.admin import KafkaAdminClient, ConfigResource, ConfigResourceType
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import dask
    import dask.distributed
    from dask.distributed import Client, LocalCluster, as_completed
    from dask import delayed, compute
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray
    from ray import remote, get, put, wait
    import ray.util
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import kubernetes
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

try:
    import celery
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

try:
    import grpc
    import grpc_tools
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessingEngine(Enum):
    """Available distributed processing engines."""
    DASK = "dask"
    RAY = "ray"
    CELERY = "celery"
    MULTIPROCESSING = "multiprocessing"
    KUBERNETES = "kubernetes"


class MessagePattern(Enum):
    """Message patterns for distributed communication."""
    PUBLISH_SUBSCRIBE = "pub_sub"
    REQUEST_RESPONSE = "req_resp"
    MESSAGE_QUEUE = "message_queue"
    EVENT_STREAMING = "event_streaming"
    COMMAND_QUERY = "command_query"


class ServiceType(Enum):
    """Microservice types in the carbon tracking system."""
    METRIC_COLLECTOR = "metric_collector"
    CARBON_CALCULATOR = "carbon_calculator"
    AGGREGATOR = "aggregator"
    STORAGE_SERVICE = "storage_service"
    NOTIFICATION_SERVICE = "notification_service"
    OPTIMIZATION_SERVICE = "optimization_service"
    GATEWAY_SERVICE = "gateway_service"


@dataclass
class ProcessingTask:
    """Task for distributed processing."""
    
    task_id: str
    task_type: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling
    priority: int = 0
    estimated_duration: float = 60.0  # seconds
    required_resources: Dict[str, float] = field(default_factory=dict)
    
    # Lifecycle
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Distributed execution
    worker_id: Optional[str] = None
    execution_node: Optional[str] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate task execution time."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def age(self) -> float:
        """Calculate task age since creation."""
        return time.time() - self.created_at


@dataclass
class Microservice:
    """Microservice definition for carbon tracking."""
    
    service_id: str
    service_type: ServiceType
    service_name: str
    version: str
    
    # Deployment
    image: str
    replicas: int = 1
    resources: Dict[str, str] = field(default_factory=dict)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Networking
    ports: List[Dict[str, Any]] = field(default_factory=list)
    service_dependencies: List[str] = field(default_factory=list)
    
    # Health and monitoring
    health_check_path: str = "/health"
    metrics_path: str = "/metrics"
    
    # Runtime status
    status: str = "stopped"
    last_health_check: Optional[float] = None
    
    def to_k8s_deployment(self) -> Dict[str, Any]:
        """Convert to Kubernetes deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.service_name,
                "labels": {
                    "app": self.service_name,
                    "version": self.version,
                    "service-type": self.service_type.value
                }
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.service_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.service_name,
                            "version": self.version
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": self.service_name,
                            "image": self.image,
                            "ports": self.ports,
                            "env": [
                                {"name": k, "value": v} 
                                for k, v in self.environment.items()
                            ],
                            "resources": self.resources,
                            "livenessProbe": {
                                "httpGet": {
                                    "path": self.health_check_path,
                                    "port": self.ports[0]["containerPort"] if self.ports else 8080
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            }
                        }]
                    }
                }
            }
        }


class KafkaEventBus:
    """Apache Kafka event bus for carbon metrics streaming."""
    
    def __init__(self,
                 bootstrap_servers: str = "localhost:9092",
                 topic_prefix: str = "carbon-tracking"):
        """Initialize Kafka event bus.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic_prefix: Prefix for topic names
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        
        # Kafka clients
        self._producer: Optional[KafkaProducer] = None
        self._admin_client: Optional[KafkaAdminClient] = None
        self._consumers: Dict[str, KafkaConsumer] = {}
        
        # Topic configuration
        self.topics = {
            "metrics": f"{topic_prefix}.metrics",
            "events": f"{topic_prefix}.events",
            "commands": f"{topic_prefix}.commands",
            "results": f"{topic_prefix}.results",
            "alerts": f"{topic_prefix}.alerts"
        }
        
        # Message handlers
        self._message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "messages_produced": 0,
            "messages_consumed": 0,
            "topics_created": 0,
            "consumer_groups": 0
        }
        
        if KAFKA_AVAILABLE:
            self._initialize_kafka()
        else:
            logger.warning("Kafka not available, using mock event bus")
    
    def _initialize_kafka(self):
        """Initialize Kafka connections and topics."""
        try:
            # Create producer
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                compression_type='snappy',
                batch_size=16384,
                linger_ms=10,
                max_in_flight_requests_per_connection=5
            )
            
            # Create admin client
            self._admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers
            )
            
            # Create topics
            self._create_topics()
            
            logger.info("Kafka event bus initialized successfully")
            
        except Exception as e:
            logger.error(f"Kafka initialization failed: {e}")
            self._producer = None
            self._admin_client = None
    
    def _create_topics(self):
        """Create Kafka topics with appropriate configuration."""
        if not self._admin_client:
            return
        
        try:
            from kafka.admin import NewTopic
            
            topic_configs = [
                NewTopic(
                    name=self.topics["metrics"],
                    num_partitions=12,  # For high throughput
                    replication_factor=3,
                    topic_configs={
                        'retention.ms': str(7 * 24 * 3600 * 1000),  # 7 days
                        'compression.type': 'snappy',
                        'cleanup.policy': 'delete'
                    }
                ),
                NewTopic(
                    name=self.topics["events"],
                    num_partitions=6,
                    replication_factor=3,
                    topic_configs={
                        'retention.ms': str(30 * 24 * 3600 * 1000),  # 30 days
                        'compression.type': 'snappy'
                    }
                ),
                NewTopic(
                    name=self.topics["commands"],
                    num_partitions=3,
                    replication_factor=3,
                    topic_configs={
                        'retention.ms': str(24 * 3600 * 1000),  # 1 day
                        'compression.type': 'snappy'
                    }
                ),
                NewTopic(
                    name=self.topics["results"],
                    num_partitions=6,
                    replication_factor=3,
                    topic_configs={
                        'retention.ms': str(7 * 24 * 3600 * 1000),  # 7 days
                        'compression.type': 'snappy'
                    }
                ),
                NewTopic(
                    name=self.topics["alerts"],
                    num_partitions=1,
                    replication_factor=3,
                    topic_configs={
                        'retention.ms': str(90 * 24 * 3600 * 1000),  # 90 days
                        'compression.type': 'snappy'
                    }
                )
            ]
            
            # Create topics
            result = self._admin_client.create_topics(topic_configs, validate_only=False)
            
            for topic, future in result.topic_futures.items():
                try:
                    future.result()  # Block until topic is created
                    self.stats["topics_created"] += 1
                    logger.info(f"Created Kafka topic: {topic}")
                except Exception as e:
                    if "already exists" not in str(e):
                        logger.error(f"Failed to create topic {topic}: {e}")
            
        except Exception as e:
            logger.error(f"Topic creation failed: {e}")
    
    async def publish_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """Publish carbon metrics to Kafka.
        
        Args:
            metrics: List of carbon metrics
            
        Returns:
            True if successfully published
        """
        if not self._producer:
            return False
        
        try:
            futures = []
            
            for metric in metrics:
                # Determine partition key based on cluster_id for even distribution
                partition_key = metric.get("cluster_id", str(uuid.uuid4()))
                
                # Add timestamp if not present
                if "timestamp" not in metric:
                    metric["timestamp"] = time.time()
                
                # Publish message
                future = self._producer.send(
                    topic=self.topics["metrics"],
                    key=partition_key,
                    value=metric,
                    timestamp_ms=int(metric["timestamp"] * 1000)
                )
                
                futures.append(future)
            
            # Wait for all messages to be sent
            for future in futures:
                future.get(timeout=10)
            
            self.stats["messages_produced"] += len(metrics)
            logger.debug(f"Published {len(metrics)} metrics to Kafka")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish metrics: {e}")
            return False
    
    async def publish_event(self, 
                          event_type: str, 
                          event_data: Dict[str, Any],
                          key: str = None) -> bool:
        """Publish event to Kafka.
        
        Args:
            event_type: Type of event
            event_data: Event payload
            key: Optional partition key
            
        Returns:
            True if successfully published
        """
        if not self._producer:
            return False
        
        try:
            event = {
                "event_type": event_type,
                "event_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "data": event_data
            }
            
            future = self._producer.send(
                topic=self.topics["events"],
                key=key or event_type,
                value=event
            )
            
            future.get(timeout=10)
            self.stats["messages_produced"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    def subscribe_to_metrics(self, 
                           consumer_group: str,
                           handler: Callable[[Dict[str, Any]], None]) -> str:
        """Subscribe to carbon metrics stream.
        
        Args:
            consumer_group: Kafka consumer group
            handler: Message handler function
            
        Returns:
            Subscription ID
        """
        subscription_id = f"metrics_{consumer_group}_{uuid.uuid4().hex[:8]}"
        
        if KAFKA_AVAILABLE:
            self._start_kafka_consumer(
                subscription_id=subscription_id,
                topics=[self.topics["metrics"]],
                consumer_group=consumer_group,
                handler=handler
            )
        else:
            # Mock subscription
            self._message_handlers[subscription_id].append(handler)
        
        return subscription_id
    
    def subscribe_to_events(self,
                          consumer_group: str,
                          event_types: List[str],
                          handler: Callable[[Dict[str, Any]], None]) -> str:
        """Subscribe to specific event types.
        
        Args:
            consumer_group: Kafka consumer group
            event_types: List of event types to subscribe to
            handler: Message handler function
            
        Returns:
            Subscription ID
        """
        subscription_id = f"events_{consumer_group}_{uuid.uuid4().hex[:8]}"
        
        if KAFKA_AVAILABLE:
            # Create filtered handler
            def filtered_handler(message):
                if message.get("event_type") in event_types:
                    handler(message)
            
            self._start_kafka_consumer(
                subscription_id=subscription_id,
                topics=[self.topics["events"]],
                consumer_group=consumer_group,
                handler=filtered_handler
            )
        else:
            # Mock subscription
            self._message_handlers[subscription_id].append(handler)
        
        return subscription_id
    
    def _start_kafka_consumer(self,
                            subscription_id: str,
                            topics: List[str],
                            consumer_group: str,
                            handler: Callable):
        """Start Kafka consumer in background thread."""
        def consumer_loop():
            try:
                consumer = KafkaConsumer(
                    *topics,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=consumer_group,
                    auto_offset_reset='latest',
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    key_deserializer=lambda m: m.decode('utf-8') if m else None,
                    enable_auto_commit=True,
                    session_timeout_ms=30000,
                    heartbeat_interval_ms=10000
                )
                
                self._consumers[subscription_id] = consumer
                self.stats["consumer_groups"] += 1
                
                logger.info(f"Started Kafka consumer {subscription_id} for topics: {topics}")
                
                # Consume messages
                for message in consumer:
                    try:
                        handler(message.value)
                        self.stats["messages_consumed"] += 1
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")
                
            except Exception as e:
                logger.error(f"Kafka consumer error: {e}")
            finally:
                if subscription_id in self._consumers:
                    del self._consumers[subscription_id]
        
        # Start consumer in background thread
        thread = threading.Thread(target=consumer_loop, daemon=True)
        thread.start()
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from topic.
        
        Args:
            subscription_id: Subscription ID to cancel
            
        Returns:
            True if successfully unsubscribed
        """
        try:
            if subscription_id in self._consumers:
                consumer = self._consumers[subscription_id]
                consumer.close()
                del self._consumers[subscription_id]
                self.stats["consumer_groups"] -= 1
                return True
            
            if subscription_id in self._message_handlers:
                del self._message_handlers[subscription_id]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Kafka event bus statistics."""
        return {
            **self.stats,
            "kafka_available": KAFKA_AVAILABLE,
            "active_consumers": len(self._consumers),
            "topics": list(self.topics.values()),
            "message_handlers": len(self._message_handlers)
        }


class DaskDistributedProcessor:
    """Dask-based distributed processor for carbon analytics."""
    
    def __init__(self,
                 cluster_address: str = None,
                 n_workers: int = None,
                 threads_per_worker: int = 2,
                 memory_limit: str = "4GB"):
        """Initialize Dask distributed processor.
        
        Args:
            cluster_address: Address of existing Dask cluster
            n_workers: Number of workers for local cluster
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
        """
        self.cluster_address = cluster_address
        self.n_workers = n_workers or mp.cpu_count()
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        
        # Dask client and cluster
        self._client: Optional[dask.distributed.Client] = None
        self._cluster: Optional[dask.distributed.LocalCluster] = None
        
        # Task tracking
        self._active_tasks: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "workers_count": 0,
            "total_cores": 0,
            "total_memory_gb": 0
        }
        
        if DASK_AVAILABLE:
            self._initialize_dask()
        else:
            logger.warning("Dask not available, using fallback processor")
    
    def _initialize_dask(self):
        """Initialize Dask cluster and client."""
        try:
            if self.cluster_address:
                # Connect to existing cluster
                self._client = dask.distributed.Client(self.cluster_address)
                logger.info(f"Connected to Dask cluster: {self.cluster_address}")
            else:
                # Create local cluster
                self._cluster = dask.distributed.LocalCluster(
                    n_workers=self.n_workers,
                    threads_per_worker=self.threads_per_worker,
                    memory_limit=self.memory_limit,
                    dashboard_address=":8787"
                )
                
                self._client = dask.distributed.Client(self._cluster)
                logger.info(f"Created Dask local cluster with {self.n_workers} workers")
            
            # Update statistics
            cluster_info = self._client.scheduler_info()
            self.stats["workers_count"] = len(cluster_info["workers"])
            self.stats["total_cores"] = sum(
                worker["nthreads"] for worker in cluster_info["workers"].values()
            )
            
        except Exception as e:
            logger.error(f"Dask initialization failed: {e}")
            self._client = None
            self._cluster = None
    
    async def submit_task(self, 
                         task_func: Callable,
                         *args,
                         task_id: str = None,
                         **kwargs) -> str:
        """Submit task for distributed execution.
        
        Args:
            task_func: Function to execute
            *args: Function arguments
            task_id: Optional task ID
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if not self._client:
            raise RuntimeError("Dask client not available")
        
        task_id = task_id or str(uuid.uuid4())
        
        try:
            # Submit task to Dask
            future = self._client.submit(task_func, *args, **kwargs)
            
            self._active_tasks[task_id] = {
                "future": future,
                "submitted_at": time.time(),
                "task_func": task_func.__name__,
                "status": "submitted"
            }
            
            self.stats["tasks_submitted"] += 1
            
            logger.debug(f"Submitted Dask task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit Dask task: {e}")
            raise
    
    async def get_result(self, task_id: str, timeout: float = None) -> Any:
        """Get task result.
        
        Args:
            task_id: Task identifier
            timeout: Timeout in seconds
            
        Returns:
            Task result
        """
        if task_id not in self._active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_info = self._active_tasks[task_id]
        future = task_info["future"]
        
        try:
            # Get result with timeout
            if timeout:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future.result()), 
                    timeout=timeout
                )
            else:
                result = future.result()
            
            # Update task status
            task_info["status"] = "completed"
            task_info["completed_at"] = time.time()
            
            self.stats["tasks_completed"] += 1
            
            return result
            
        except Exception as e:
            task_info["status"] = "failed"
            task_info["error"] = str(e)
            self.stats["tasks_failed"] += 1
            
            logger.error(f"Dask task {task_id} failed: {e}")
            raise
    
    async def map_reduce_carbon_metrics(self,
                                      metrics: List[Dict[str, Any]],
                                      chunk_size: int = 1000) -> Dict[str, Any]:
        """Perform MapReduce operation on carbon metrics.
        
        Args:
            metrics: List of carbon metrics
            chunk_size: Size of processing chunks
            
        Returns:
            Aggregated results
        """
        if not self._client:
            raise RuntimeError("Dask client not available")
        
        if not metrics:
            return {"total_metrics": 0, "aggregated_results": {}}
        
        try:
            # Split metrics into chunks
            chunks = [
                metrics[i:i + chunk_size] 
                for i in range(0, len(metrics), chunk_size)
            ]
            
            # Map phase: process chunks in parallel
            map_futures = []
            for chunk in chunks:
                future = self._client.submit(self._process_carbon_chunk, chunk)
                map_futures.append(future)
            
            # Wait for map phase completion
            map_results = [future.result() for future in map_futures]
            
            # Reduce phase: aggregate results
            reduce_future = self._client.submit(self._reduce_carbon_results, map_results)
            final_result = reduce_future.result()
            
            logger.info(f"MapReduce completed: processed {len(metrics)} metrics in {len(chunks)} chunks")
            
            return final_result
            
        except Exception as e:
            logger.error(f"MapReduce operation failed: {e}")
            raise
    
    @staticmethod
    def _process_carbon_chunk(chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a chunk of carbon metrics (map function).
        
        Args:
            chunk: Chunk of metrics to process
            
        Returns:
            Processed chunk results
        """
        import statistics
        
        total_power = 0.0
        total_carbon = 0.0
        gpu_count = 0
        cluster_metrics = defaultdict(list)
        
        for metric in chunk:
            power = metric.get("power_consumption_kw", 0)
            carbon = metric.get("carbon_emission_kg_hr", 0)
            cluster_id = metric.get("cluster_id", "unknown")
            
            total_power += power
            total_carbon += carbon
            gpu_count += 1
            
            cluster_metrics[cluster_id].append({
                "power": power,
                "carbon": carbon,
                "timestamp": metric.get("timestamp", time.time())
            })
        
        # Calculate chunk statistics
        chunk_stats = {
            "total_power_kw": total_power,
            "total_carbon_kg_hr": total_carbon,
            "gpu_count": gpu_count,
            "avg_power_per_gpu": total_power / max(gpu_count, 1),
            "avg_carbon_per_gpu": total_carbon / max(gpu_count, 1),
            "cluster_count": len(cluster_metrics)
        }
        
        return {
            "chunk_stats": chunk_stats,
            "cluster_metrics": dict(cluster_metrics)
        }
    
    @staticmethod
    def _reduce_carbon_results(map_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reduce carbon metrics results (reduce function).
        
        Args:
            map_results: Results from map phase
            
        Returns:
            Final aggregated results
        """
        global_stats = {
            "total_power_kw": 0.0,
            "total_carbon_kg_hr": 0.0,
            "total_gpu_count": 0,
            "total_clusters": 0,
            "chunks_processed": len(map_results)
        }
        
        cluster_aggregates = defaultdict(lambda: {
            "total_power": 0.0,
            "total_carbon": 0.0,
            "gpu_count": 0,
            "timestamps": []
        })
        
        # Aggregate results from all chunks
        for result in map_results:
            chunk_stats = result["chunk_stats"]
            
            # Global aggregation
            global_stats["total_power_kw"] += chunk_stats["total_power_kw"]
            global_stats["total_carbon_kg_hr"] += chunk_stats["total_carbon_kg_hr"]
            global_stats["total_gpu_count"] += chunk_stats["gpu_count"]
            
            # Cluster-level aggregation
            for cluster_id, metrics in result["cluster_metrics"].items():
                cluster_agg = cluster_aggregates[cluster_id]
                
                for metric in metrics:
                    cluster_agg["total_power"] += metric["power"]
                    cluster_agg["total_carbon"] += metric["carbon"]
                    cluster_agg["gpu_count"] += 1
                    cluster_agg["timestamps"].append(metric["timestamp"])
        
        global_stats["total_clusters"] = len(cluster_aggregates)
        
        # Calculate cluster summaries
        cluster_summaries = {}
        for cluster_id, agg in cluster_aggregates.items():
            cluster_summaries[cluster_id] = {
                "total_power_kw": agg["total_power"],
                "total_carbon_kg_hr": agg["total_carbon"],
                "gpu_count": agg["gpu_count"],
                "avg_power_per_gpu": agg["total_power"] / max(agg["gpu_count"], 1),
                "avg_carbon_per_gpu": agg["total_carbon"] / max(agg["gpu_count"], 1),
                "time_span_hours": (max(agg["timestamps"]) - min(agg["timestamps"])) / 3600 if agg["timestamps"] else 0
            }
        
        return {
            "global_stats": global_stats,
            "cluster_summaries": cluster_summaries,
            "processing_timestamp": time.time()
        }
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get Dask cluster status."""
        if not self._client:
            return {"status": "not_available"}
        
        try:
            scheduler_info = self._client.scheduler_info()
            
            workers_info = []
            for worker_addr, worker_info in scheduler_info["workers"].items():
                workers_info.append({
                    "address": worker_addr,
                    "nthreads": worker_info["nthreads"],
                    "memory_limit": worker_info.get("memory_limit", 0),
                    "status": worker_info.get("status", "unknown")
                })
            
            return {
                "status": "connected",
                "cluster_type": "local" if self._cluster else "remote",
                "scheduler_address": self._client.scheduler.address,
                "workers": workers_info,
                "total_workers": len(workers_info),
                "total_cores": sum(w["nthreads"] for w in workers_info),
                "dashboard_link": getattr(self._cluster, "dashboard_link", None),
                "active_tasks": len(self._active_tasks),
                "statistics": self.stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {"status": "error", "error": str(e)}
    
    def shutdown(self):
        """Shutdown Dask cluster and client."""
        try:
            if self._client:
                self._client.close()
            
            if self._cluster:
                self._cluster.close()
            
            logger.info("Dask distributed processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Dask shutdown error: {e}")


class RayDistributedProcessor:
    """Ray-based distributed processor for carbon analytics."""
    
    def __init__(self,
                 ray_address: str = None,
                 num_cpus: int = None,
                 num_gpus: int = 0,
                 object_store_memory: int = None):
        """Initialize Ray distributed processor.
        
        Args:
            ray_address: Address of Ray cluster
            num_cpus: Number of CPUs to use
            num_gpus: Number of GPUs to use
            object_store_memory: Object store memory in bytes
        """
        self.ray_address = ray_address
        self.num_cpus = num_cpus or mp.cpu_count()
        self.num_gpus = num_gpus
        self.object_store_memory = object_store_memory
        
        # Task tracking
        self._active_tasks: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "objects_stored": 0
        }
        
        if RAY_AVAILABLE:
            self._initialize_ray()
        else:
            logger.warning("Ray not available, using fallback processor")
    
    def _initialize_ray(self):
        """Initialize Ray cluster."""
        try:
            if not ray.is_initialized():
                init_args = {
                    "ignore_reinit_error": True,
                    "logging_level": logging.WARNING
                }
                
                if self.ray_address:
                    init_args["address"] = self.ray_address
                else:
                    init_args["num_cpus"] = self.num_cpus
                    init_args["num_gpus"] = self.num_gpus
                    
                    if self.object_store_memory:
                        init_args["object_store_memory"] = self.object_store_memory
                
                ray.init(**init_args)
            
            cluster_resources = ray.cluster_resources()
            logger.info(f"Ray cluster initialized: {cluster_resources}")
            
        except Exception as e:
            logger.error(f"Ray initialization failed: {e}")
    
    async def submit_task(self,
                         task_func: Callable,
                         *args,
                         task_id: str = None,
                         num_cpus: float = 1,
                         num_gpus: float = 0,
                         **kwargs) -> str:
        """Submit task for Ray execution.
        
        Args:
            task_func: Function to execute
            *args: Function arguments
            task_id: Optional task ID
            num_cpus: CPUs required for task
            num_gpus: GPUs required for task
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if not RAY_AVAILABLE or not ray.is_initialized():
            raise RuntimeError("Ray not available")
        
        task_id = task_id or str(uuid.uuid4())
        
        try:
            # Create Ray remote function
            remote_func = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(task_func)
            
            # Submit task
            future = remote_func.remote(*args, **kwargs)
            
            self._active_tasks[task_id] = {
                "future": future,
                "submitted_at": time.time(),
                "task_func": task_func.__name__,
                "status": "submitted",
                "resources": {"num_cpus": num_cpus, "num_gpus": num_gpus}
            }
            
            self.stats["tasks_submitted"] += 1
            
            logger.debug(f"Submitted Ray task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit Ray task: {e}")
            raise
    
    async def get_result(self, task_id: str, timeout: float = None) -> Any:
        """Get task result.
        
        Args:
            task_id: Task identifier
            timeout: Timeout in seconds
            
        Returns:
            Task result
        """
        if task_id not in self._active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_info = self._active_tasks[task_id]
        future = task_info["future"]
        
        try:
            # Get result with timeout
            if timeout:
                ready, not_ready = ray.wait([future], timeout=timeout)
                if not ready:
                    raise asyncio.TimeoutError(f"Task {task_id} timed out")
                result = ray.get(ready[0])
            else:
                result = ray.get(future)
            
            # Update task status
            task_info["status"] = "completed"
            task_info["completed_at"] = time.time()
            
            self.stats["tasks_completed"] += 1
            
            return result
            
        except Exception as e:
            task_info["status"] = "failed"
            task_info["error"] = str(e)
            self.stats["tasks_failed"] += 1
            
            logger.error(f"Ray task {task_id} failed: {e}")
            raise
    
    async def parallel_carbon_processing(self,
                                       metrics_batches: List[List[Dict[str, Any]]],
                                       processing_func: Callable) -> List[Any]:
        """Process multiple batches of metrics in parallel.
        
        Args:
            metrics_batches: List of metric batches
            processing_func: Function to process each batch
            
        Returns:
            List of processing results
        """
        if not RAY_AVAILABLE or not ray.is_initialized():
            raise RuntimeError("Ray not available")
        
        try:
            # Create remote function
            remote_func = ray.remote(processing_func)
            
            # Submit all batches for parallel processing
            futures = []
            for batch in metrics_batches:
                future = remote_func.remote(batch)
                futures.append(future)
            
            # Wait for all results
            results = ray.get(futures)
            
            logger.info(f"Processed {len(metrics_batches)} batches in parallel")
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            raise
    
    def store_object(self, obj: Any, object_id: str = None) -> str:
        """Store object in Ray object store.
        
        Args:
            obj: Object to store
            object_id: Optional object ID
            
        Returns:
            Object reference ID
        """
        if not RAY_AVAILABLE or not ray.is_initialized():
            raise RuntimeError("Ray not available")
        
        try:
            object_ref = ray.put(obj)
            self.stats["objects_stored"] += 1
            
            ref_id = object_id or str(object_ref)
            logger.debug(f"Stored object in Ray object store: {ref_id}")
            
            return ref_id
            
        except Exception as e:
            logger.error(f"Failed to store object: {e}")
            raise
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get Ray cluster status."""
        if not RAY_AVAILABLE or not ray.is_initialized():
            return {"status": "not_available"}
        
        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            return {
                "status": "connected",
                "cluster_resources": cluster_resources,
                "available_resources": available_resources,
                "nodes": len(ray.nodes()),
                "active_tasks": len(self._active_tasks),
                "statistics": self.stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get Ray cluster status: {e}")
            return {"status": "error", "error": str(e)}
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        try:
            if RAY_AVAILABLE and ray.is_initialized():
                ray.shutdown()
            
            logger.info("Ray distributed processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Ray shutdown error: {e}")


class KubernetesOrchestrator:
    """Kubernetes orchestrator for carbon tracking microservices."""
    
    def __init__(self,
                 namespace: str = "carbon-tracking",
                 kubeconfig_path: str = None):
        """Initialize Kubernetes orchestrator.
        
        Args:
            namespace: Kubernetes namespace
            kubeconfig_path: Path to kubeconfig file
        """
        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        
        # Kubernetes clients
        self._v1_client: Optional[client.CoreV1Api] = None
        self._apps_v1_client: Optional[client.AppsV1Api] = None
        self._autoscaling_client: Optional[client.AutoscalingV1Api] = None
        
        # Service registry
        self._services: Dict[str, Microservice] = {}
        
        # Statistics
        self.stats = {
            "deployments_created": 0,
            "services_created": 0,
            "pods_running": 0,
            "hpa_created": 0
        }
        
        if K8S_AVAILABLE:
            self._initialize_kubernetes()
        else:
            logger.warning("Kubernetes not available, using mock orchestrator")
    
    def _initialize_kubernetes(self):
        """Initialize Kubernetes clients."""
        try:
            # Load kubeconfig
            if self.kubeconfig_path:
                config.load_kube_config(config_file=self.kubeconfig_path)
            else:
                try:
                    config.load_incluster_config()  # Try in-cluster config first
                except config.ConfigException:
                    config.load_kube_config()  # Fall back to local config
            
            # Create clients
            self._v1_client = client.CoreV1Api()
            self._apps_v1_client = client.AppsV1Api()
            self._autoscaling_client = client.AutoscalingV1Api()
            
            # Ensure namespace exists
            self._ensure_namespace()
            
            logger.info(f"Kubernetes orchestrator initialized for namespace: {self.namespace}")
            
        except Exception as e:
            logger.error(f"Kubernetes initialization failed: {e}")
            self._v1_client = None
            self._apps_v1_client = None
    
    def _ensure_namespace(self):
        """Ensure the namespace exists."""
        if not self._v1_client:
            return
        
        try:
            # Check if namespace exists
            self._v1_client.read_namespace(name=self.namespace)
            logger.info(f"Namespace {self.namespace} already exists")
            
        except ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_manifest = client.V1Namespace(
                    metadata=client.V1ObjectMeta(
                        name=self.namespace,
                        labels={
                            "app": "carbon-tracking",
                            "managed-by": "hf-eco2ai"
                        }
                    )
                )
                
                self._v1_client.create_namespace(namespace_manifest)
                logger.info(f"Created namespace: {self.namespace}")
            else:
                logger.error(f"Failed to check namespace: {e}")
    
    async def deploy_service(self, service: Microservice) -> bool:
        """Deploy microservice to Kubernetes.
        
        Args:
            service: Microservice to deploy
            
        Returns:
            True if deployment successful
        """
        if not self._apps_v1_client:
            logger.warning("Kubernetes not available, cannot deploy service")
            return False
        
        try:
            # Create deployment
            deployment_manifest = self._create_deployment_manifest(service)
            
            try:
                # Try to get existing deployment
                self._apps_v1_client.read_namespaced_deployment(
                    name=service.service_name,
                    namespace=self.namespace
                )
                
                # Update existing deployment
                self._apps_v1_client.patch_namespaced_deployment(
                    name=service.service_name,
                    namespace=self.namespace,
                    body=deployment_manifest
                )
                
                logger.info(f"Updated deployment: {service.service_name}")
                
            except ApiException as e:
                if e.status == 404:
                    # Create new deployment
                    self._apps_v1_client.create_namespaced_deployment(
                        namespace=self.namespace,
                        body=deployment_manifest
                    )
                    
                    self.stats["deployments_created"] += 1
                    logger.info(f"Created deployment: {service.service_name}")
                else:
                    raise
            
            # Create service
            service_manifest = self._create_service_manifest(service)
            
            try:
                # Try to get existing service
                self._v1_client.read_namespaced_service(
                    name=service.service_name,
                    namespace=self.namespace
                )
                
                # Update existing service
                self._v1_client.patch_namespaced_service(
                    name=service.service_name,
                    namespace=self.namespace,
                    body=service_manifest
                )
                
            except ApiException as e:
                if e.status == 404:
                    # Create new service
                    self._v1_client.create_namespaced_service(
                        namespace=self.namespace,
                        body=service_manifest
                    )
                    
                    self.stats["services_created"] += 1
                    logger.info(f"Created service: {service.service_name}")
                else:
                    raise
            
            # Create HPA if needed
            if service.replicas > 1:
                await self._create_hpa(service)
            
            # Register service
            self._services[service.service_id] = service
            service.status = "deployed"
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy service {service.service_name}: {e}")
            return False
    
    def _create_deployment_manifest(self, service: Microservice) -> client.V1Deployment:
        """Create Kubernetes deployment manifest."""
        return client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=service.service_name,
                namespace=self.namespace,
                labels={
                    "app": service.service_name,
                    "version": service.version,
                    "service-type": service.service_type.value,
                    "managed-by": "hf-eco2ai"
                }
            ),
            spec=client.V1DeploymentSpec(
                replicas=service.replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": service.service_name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": service.service_name,
                            "version": service.version
                        }
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name=service.service_name,
                                image=service.image,
                                ports=[
                                    client.V1ContainerPort(
                                        container_port=port["containerPort"],
                                        name=port.get("name", "http"),
                                        protocol=port.get("protocol", "TCP")
                                    )
                                    for port in service.ports
                                ],
                                env=[
                                    client.V1EnvVar(name=k, value=v)
                                    for k, v in service.environment.items()
                                ],
                                resources=client.V1ResourceRequirements(
                                    requests=service.resources.get("requests", {}),
                                    limits=service.resources.get("limits", {})
                                ),
                                liveness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path=service.health_check_path,
                                        port=service.ports[0]["containerPort"] if service.ports else 8080
                                    ),
                                    initial_delay_seconds=30,
                                    period_seconds=10
                                ),
                                readiness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path=service.health_check_path,
                                        port=service.ports[0]["containerPort"] if service.ports else 8080
                                    ),
                                    initial_delay_seconds=5,
                                    period_seconds=5
                                )
                            )
                        ]
                    )
                )
            )
        )
    
    def _create_service_manifest(self, service: Microservice) -> client.V1Service:
        """Create Kubernetes service manifest."""
        return client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=service.service_name,
                namespace=self.namespace,
                labels={
                    "app": service.service_name,
                    "service-type": service.service_type.value
                }
            ),
            spec=client.V1ServiceSpec(
                selector={"app": service.service_name},
                ports=[
                    client.V1ServicePort(
                        port=port.get("port", port["containerPort"]),
                        target_port=port["containerPort"],
                        name=port.get("name", "http"),
                        protocol=port.get("protocol", "TCP")
                    )
                    for port in service.ports
                ],
                type="ClusterIP"
            )
        )
    
    async def _create_hpa(self, service: Microservice):
        """Create Horizontal Pod Autoscaler."""
        if not self._autoscaling_client:
            return
        
        try:
            hpa_manifest = client.V1HorizontalPodAutoscaler(
                api_version="autoscaling/v1",
                kind="HorizontalPodAutoscaler",
                metadata=client.V1ObjectMeta(
                    name=f"{service.service_name}-hpa",
                    namespace=self.namespace
                ),
                spec=client.V1HorizontalPodAutoscalerSpec(
                    scale_target_ref=client.V1CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=service.service_name
                    ),
                    min_replicas=1,
                    max_replicas=service.replicas * 3,  # Allow scaling up to 3x
                    target_cpu_utilization_percentage=70
                )
            )
            
            self._autoscaling_client.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa_manifest
            )
            
            self.stats["hpa_created"] += 1
            logger.info(f"Created HPA for service: {service.service_name}")
            
        except Exception as e:
            logger.warning(f"Failed to create HPA for {service.service_name}: {e}")
    
    async def scale_service(self, service_id: str, replicas: int) -> bool:
        """Scale service replicas.
        
        Args:
            service_id: Service identifier
            replicas: Target replica count
            
        Returns:
            True if scaling successful
        """
        if service_id not in self._services:
            logger.error(f"Service {service_id} not found")
            return False
        
        service = self._services[service_id]
        
        if not self._apps_v1_client:
            logger.warning("Kubernetes not available, cannot scale service")
            return False
        
        try:
            # Update deployment replicas
            deployment = self._apps_v1_client.read_namespaced_deployment(
                name=service.service_name,
                namespace=self.namespace
            )
            
            deployment.spec.replicas = replicas
            
            self._apps_v1_client.patch_namespaced_deployment(
                name=service.service_name,
                namespace=self.namespace,
                body=deployment
            )
            
            # Update service configuration
            service.replicas = replicas
            
            logger.info(f"Scaled service {service.service_name} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale service {service.service_name}: {e}")
            return False
    
    async def get_service_status(self, service_id: str) -> Dict[str, Any]:
        """Get service status.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Service status information
        """
        if service_id not in self._services:
            return {"status": "not_found"}
        
        service = self._services[service_id]
        
        if not self._v1_client or not self._apps_v1_client:
            return {"status": "kubernetes_unavailable"}
        
        try:
            # Get deployment status
            deployment = self._apps_v1_client.read_namespaced_deployment(
                name=service.service_name,
                namespace=self.namespace
            )
            
            # Get pods
            pods = self._v1_client.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={service.service_name}"
            )
            
            pod_statuses = []
            for pod in pods.items:
                pod_status = {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "ready": False,
                    "node": pod.spec.node_name
                }
                
                if pod.status.conditions:
                    for condition in pod.status.conditions:
                        if condition.type == "Ready":
                            pod_status["ready"] = condition.status == "True"
                            break
                
                pod_statuses.append(pod_status)
            
            return {
                "status": "deployed",
                "service_name": service.service_name,
                "replicas": {
                    "desired": deployment.spec.replicas,
                    "ready": deployment.status.ready_replicas or 0,
                    "available": deployment.status.available_replicas or 0
                },
                "pods": pod_statuses,
                "deployment_conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason
                    }
                    for condition in (deployment.status.conditions or [])
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        if not self._v1_client:
            return {"status": "kubernetes_unavailable"}
        
        try:
            # Get nodes
            nodes = self._v1_client.list_node()
            
            # Get pods in our namespace
            pods = self._v1_client.list_namespaced_pod(namespace=self.namespace)
            
            # Count running pods
            running_pods = sum(1 for pod in pods.items if pod.status.phase == "Running")
            self.stats["pods_running"] = running_pods
            
            node_info = []
            for node in nodes.items:
                node_info.append({
                    "name": node.metadata.name,
                    "ready": any(
                        condition.status == "True" for condition in node.status.conditions 
                        if condition.type == "Ready"
                    ),
                    "cpu_capacity": node.status.capacity.get("cpu"),
                    "memory_capacity": node.status.capacity.get("memory")
                })
            
            return {
                "status": "connected",
                "namespace": self.namespace,
                "nodes": node_info,
                "total_nodes": len(node_info),
                "total_pods": len(pods.items),
                "running_pods": running_pods,
                "deployed_services": len(self._services),
                "statistics": self.stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {"status": "error", "error": str(e)}


class DistributedProcessingEngine:
    """Main distributed processing engine coordinating all components."""
    
    def __init__(self,
                 preferred_engine: ProcessingEngine = ProcessingEngine.RAY,
                 kafka_config: Dict[str, Any] = None,
                 k8s_config: Dict[str, Any] = None):
        """Initialize distributed processing engine.
        
        Args:
            preferred_engine: Preferred processing engine
            kafka_config: Kafka configuration
            k8s_config: Kubernetes configuration
        """
        self.preferred_engine = preferred_engine
        
        # Components
        self.event_bus = KafkaEventBus(**(kafka_config or {}))
        self.k8s_orchestrator = KubernetesOrchestrator(**(k8s_config or {}))
        
        # Processing engines
        self.dask_processor: Optional[DaskDistributedProcessor] = None
        self.ray_processor: Optional[RayDistributedProcessor] = None
        
        # Initialize preferred engine
        self._initialize_processing_engine()
        
        # Task coordination
        self._task_queue = asyncio.Queue()
        self._result_cache: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "tasks_processed": 0,
            "messages_routed": 0,
            "services_deployed": 0,
            "active_workers": 0
        }
        
        logger.info(f"Distributed processing engine initialized with {preferred_engine.value}")
    
    def _initialize_processing_engine(self):
        """Initialize the preferred processing engine."""
        if self.preferred_engine == ProcessingEngine.RAY and RAY_AVAILABLE:
            self.ray_processor = RayDistributedProcessor()
        elif self.preferred_engine == ProcessingEngine.DASK and DASK_AVAILABLE:
            self.dask_processor = DaskDistributedProcessor()
        else:
            # Fallback to available engines
            if RAY_AVAILABLE:
                self.ray_processor = RayDistributedProcessor()
                self.preferred_engine = ProcessingEngine.RAY
            elif DASK_AVAILABLE:
                self.dask_processor = DaskDistributedProcessor()
                self.preferred_engine = ProcessingEngine.DASK
            else:
                logger.warning("No distributed processing engines available")
    
    async def deploy_carbon_tracking_stack(self) -> Dict[str, bool]:
        """Deploy complete carbon tracking microservices stack.
        
        Returns:
            Deployment results for each service
        """
        # Define carbon tracking microservices
        services = [
            Microservice(
                service_id="metric-collector",
                service_type=ServiceType.METRIC_COLLECTOR,
                service_name="carbon-metric-collector",
                version="1.0.0",
                image="hf-eco2ai/metric-collector:latest",
                replicas=3,
                ports=[{"containerPort": 8080, "name": "http"}],
                resources={
                    "requests": {"cpu": "200m", "memory": "256Mi"},
                    "limits": {"cpu": "500m", "memory": "512Mi"}
                },
                environment={
                    "KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
                    "LOG_LEVEL": "INFO"
                }
            ),
            Microservice(
                service_id="carbon-calculator",
                service_type=ServiceType.CARBON_CALCULATOR,
                service_name="carbon-calculator",
                version="1.0.0",
                image="hf-eco2ai/carbon-calculator:latest",
                replicas=2,
                ports=[{"containerPort": 8081, "name": "http"}],
                resources={
                    "requests": {"cpu": "300m", "memory": "512Mi"},
                    "limits": {"cpu": "1000m", "memory": "1Gi"}
                }
            ),
            Microservice(
                service_id="aggregator",
                service_type=ServiceType.AGGREGATOR,
                service_name="carbon-aggregator",
                version="1.0.0",
                image="hf-eco2ai/aggregator:latest",
                replicas=2,
                ports=[{"containerPort": 8082, "name": "http"}],
                resources={
                    "requests": {"cpu": "500m", "memory": "1Gi"},
                    "limits": {"cpu": "2000m", "memory": "2Gi"}
                }
            ),
            Microservice(
                service_id="gateway",
                service_type=ServiceType.GATEWAY_SERVICE,
                service_name="carbon-gateway",
                version="1.0.0",
                image="hf-eco2ai/gateway:latest",
                replicas=2,
                ports=[{"containerPort": 8080, "name": "http"}],
                resources={
                    "requests": {"cpu": "100m", "memory": "128Mi"},
                    "limits": {"cpu": "200m", "memory": "256Mi"}
                }
            )
        ]
        
        # Deploy services
        deployment_results = {}
        
        for service in services:
            try:
                success = await self.k8s_orchestrator.deploy_service(service)
                deployment_results[service.service_name] = success
                
                if success:
                    self.stats["services_deployed"] += 1
                
            except Exception as e:
                logger.error(f"Failed to deploy {service.service_name}: {e}")
                deployment_results[service.service_name] = False
        
        logger.info(f"Deployed {sum(deployment_results.values())} out of {len(services)} services")
        
        return deployment_results
    
    async def process_carbon_metrics_pipeline(self,
                                            metrics: List[Dict[str, Any]],
                                            processing_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process carbon metrics through distributed pipeline.
        
        Args:
            metrics: Carbon metrics to process
            processing_config: Processing configuration
            
        Returns:
            Processing results
        """
        config = processing_config or {}
        batch_size = config.get("batch_size", 1000)
        
        start_time = time.time()
        
        try:
            # Step 1: Publish metrics to event bus
            await self.event_bus.publish_metrics(metrics)
            
            # Step 2: Batch metrics for processing
            batches = [
                metrics[i:i + batch_size] 
                for i in range(0, len(metrics), batch_size)
            ]
            
            # Step 3: Distributed processing
            if self.ray_processor:
                processing_results = await self.ray_processor.parallel_carbon_processing(
                    batches, self._process_carbon_batch
                )
            elif self.dask_processor:
                processing_results = await self.dask_processor.map_reduce_carbon_metrics(
                    metrics, batch_size
                )
            else:
                # Fallback to sequential processing
                processing_results = [self._process_carbon_batch(batch) for batch in batches]
            
            # Step 4: Aggregate results
            final_result = self._aggregate_processing_results(processing_results)
            
            # Step 5: Publish results
            await self.event_bus.publish_event(
                "processing_completed",
                {
                    "metrics_count": len(metrics),
                    "processing_time": time.time() - start_time,
                    "result_summary": final_result
                }
            )
            
            self.stats["tasks_processed"] += 1
            
            return {
                "status": "success",
                "metrics_processed": len(metrics),
                "processing_time": time.time() - start_time,
                "batches_processed": len(batches),
                "results": final_result
            }
            
        except Exception as e:
            logger.error(f"Carbon metrics pipeline failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "metrics_processed": 0,
                "processing_time": time.time() - start_time
            }
    
    @staticmethod
    def _process_carbon_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of carbon metrics."""
        total_power = sum(metric.get("power_consumption_kw", 0) for metric in batch)
        total_carbon = sum(metric.get("carbon_emission_kg_hr", 0) for metric in batch)
        gpu_count = len(batch)
        
        return {
            "batch_size": len(batch),
            "total_power_kw": total_power,
            "total_carbon_kg_hr": total_carbon,
            "gpu_count": gpu_count,
            "avg_power_per_gpu": total_power / max(gpu_count, 1),
            "avg_carbon_per_gpu": total_carbon / max(gpu_count, 1),
            "processing_timestamp": time.time()
        }
    
    def _aggregate_processing_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from distributed processing."""
        if not results:
            return {}
        
        total_power = sum(r.get("total_power_kw", 0) for r in results)
        total_carbon = sum(r.get("total_carbon_kg_hr", 0) for r in results)
        total_gpus = sum(r.get("gpu_count", 0) for r in results)
        
        return {
            "total_batches": len(results),
            "total_power_kw": total_power,
            "total_carbon_kg_hr": total_carbon,
            "total_gpus": total_gpus,
            "avg_power_per_gpu": total_power / max(total_gpus, 1),
            "avg_carbon_per_gpu": total_carbon / max(total_gpus, 1),
            "carbon_efficiency_score": self._calculate_efficiency_score(total_power, total_carbon),
            "aggregation_timestamp": time.time()
        }
    
    def _calculate_efficiency_score(self, power_kw: float, carbon_kg_hr: float) -> float:
        """Calculate carbon efficiency score."""
        if power_kw == 0:
            return 100.0
        
        # Carbon intensity (g CO2/kWh)
        carbon_intensity = (carbon_kg_hr * 1000) / power_kw if power_kw > 0 else 0
        
        # Baseline is 500 g CO2/kWh (global average)
        baseline_intensity = 500.0
        
        # Efficiency score (0-100, higher is better)
        efficiency = max(0, (baseline_intensity - carbon_intensity) / baseline_intensity * 100)
        
        return min(100.0, efficiency)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.
        
        Returns:
            System status information
        """
        status = {
            "timestamp": time.time(),
            "preferred_engine": self.preferred_engine.value,
            "statistics": self.stats
        }
        
        # Event bus status
        status["event_bus"] = self.event_bus.get_stats()
        
        # Processing engine status
        if self.ray_processor:
            status["ray_processor"] = self.ray_processor.get_cluster_status()
        
        if self.dask_processor:
            status["dask_processor"] = self.dask_processor.get_cluster_status()
        
        # Kubernetes status
        status["kubernetes"] = self.k8s_orchestrator.get_cluster_status()
        
        return status
    
    async def shutdown(self):
        """Shutdown distributed processing engine."""
        logger.info("Shutting down distributed processing engine")
        
        # Shutdown processing engines
        if self.ray_processor:
            self.ray_processor.shutdown()
        
        if self.dask_processor:
            self.dask_processor.shutdown()
        
        logger.info("Distributed processing engine shutdown complete")


# Global engine instance
_distributed_engine: Optional[DistributedProcessingEngine] = None


def get_distributed_engine(**kwargs) -> DistributedProcessingEngine:
    """Get global distributed processing engine instance."""
    global _distributed_engine
    
    if _distributed_engine is None:
        _distributed_engine = DistributedProcessingEngine(**kwargs)
    
    return _distributed_engine


async def initialize_distributed_processing():
    """Initialize distributed processing with optimal configuration."""
    engine = get_distributed_engine(
        preferred_engine=ProcessingEngine.RAY,
        kafka_config={
            "bootstrap_servers": "localhost:9092",
            "topic_prefix": "carbon-tracking"
        },
        k8s_config={
            "namespace": "carbon-tracking"
        }
    )
    
    logger.info("Distributed processing engine initialized for enterprise scale")
    return engine