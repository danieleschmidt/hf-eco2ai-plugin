"""Distributed computing and async processing for scalable carbon tracking."""

import asyncio
import logging
import time
import json
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from concurrent.futures import Future
from enum import Enum
import threading
import queue
import socket
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerStatus(Enum):
    """Status of worker nodes."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    
    task_id: str
    function_name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
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
class WorkerNode:
    """Represents a worker node in the distributed system."""
    
    worker_id: str
    host: str
    port: int
    status: WorkerStatus = WorkerStatus.AVAILABLE
    capabilities: Set[str] = field(default_factory=set)
    current_task: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    total_tasks_completed: int = 0
    total_errors: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy based on heartbeat."""
        return time.time() - self.last_heartbeat < 30.0  # 30 second timeout
    
    @property
    def error_rate(self) -> float:
        """Calculate worker error rate."""
        total = self.total_tasks_completed + self.total_errors
        return self.total_errors / total if total > 0 else 0.0


class DistributedTaskScheduler:
    """Scheduler for managing distributed carbon tracking tasks."""
    
    def __init__(self, 
                 max_queue_size: int = 1000,
                 worker_timeout: float = 30.0):
        """Initialize distributed task scheduler.
        
        Args:
            max_queue_size: Maximum task queue size
            worker_timeout: Worker heartbeat timeout
        """
        self.max_queue_size = max_queue_size
        self.worker_timeout = worker_timeout
        
        # Task management
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_assignments: Dict[str, str] = {}  # worker_id -> task_id
        
        # Event loops and locks
        self._scheduler_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0
        }
        
        logger.info("Distributed task scheduler initialized")
    
    async def start(self):
        """Start the distributed scheduler."""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        
        # Start scheduler loop
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # Start heartbeat monitor
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        logger.info("Distributed scheduler started")
    
    async def stop(self):
        """Stop the distributed scheduler."""
        self._scheduler_running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Distributed scheduler stopped")
    
    async def submit_task(self,
                         function_name: str,
                         args: Tuple[Any, ...] = (),
                         kwargs: Dict[str, Any] = None,
                         priority: int = 0,
                         timeout: Optional[float] = None,
                         max_retries: int = 3) -> str:
        """Submit a task for distributed execution.
        
        Args:
            function_name: Name of function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority (higher = more important)
            timeout: Task timeout in seconds
            max_retries: Maximum retry attempts
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = DistributedTask(
            task_id=task_id,
            function_name=function_name,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        async with self._lock:
            self.tasks[task_id] = task
            await self.task_queue.put((-priority, time.time(), task_id))  # Negative for max-heap
            self.stats["tasks_submitted"] += 1
        
        logger.debug(f"Task submitted: {task_id}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information
        """
        async with self._lock:
            task = self.tasks.get(task_id) or self.completed_tasks.get(task_id)
            
            if not task:
                return None
            
            return {
                "task_id": task_id,
                "status": task.status.value,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "execution_time": task.execution_time,
                "worker_id": task.worker_id,
                "error": task.error,
                "retry_count": task.retry_count
            }
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for task completion and return result.
        
        Args:
            task_id: Task identifier
            timeout: Wait timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            TimeoutError: If task doesn't complete within timeout
            RuntimeError: If task failed
        """
        start_time = time.time()
        
        while True:
            async with self._lock:
                task = self.tasks.get(task_id) or self.completed_tasks.get(task_id)
                
                if not task:
                    raise ValueError(f"Task {task_id} not found")
                
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise RuntimeError(f"Task failed: {task.error}")
                elif task.status == TaskStatus.CANCELLED:
                    raise RuntimeError("Task was cancelled")
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            await asyncio.sleep(0.1)
    
    async def register_worker(self,
                            worker_id: str,
                            host: str,
                            port: int,
                            capabilities: Set[str] = None) -> bool:
        """Register a new worker node.
        
        Args:
            worker_id: Unique worker identifier
            host: Worker host address
            port: Worker port
            capabilities: Set of worker capabilities
            
        Returns:
            True if registration successful
        """
        async with self._lock:
            if worker_id in self.workers:
                logger.warning(f"Worker {worker_id} already registered")
                return False
            
            worker = WorkerNode(
                worker_id=worker_id,
                host=host,
                port=port,
                capabilities=capabilities or set()
            )
            
            self.workers[worker_id] = worker
            logger.info(f"Worker registered: {worker_id} at {host}:{port}")
            return True
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker node.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            True if unregistration successful
        """
        async with self._lock:
            if worker_id not in self.workers:
                return False
            
            # Handle any currently assigned task
            if worker_id in self.worker_assignments:
                task_id = self.worker_assignments[worker_id]
                task = self.tasks.get(task_id)
                
                if task and task.status == TaskStatus.RUNNING:
                    # Requeue the task
                    task.status = TaskStatus.PENDING
                    task.worker_id = None
                    task.started_at = None
                    await self.task_queue.put((-task.priority, time.time(), task_id))
                
                del self.worker_assignments[worker_id]
            
            del self.workers[worker_id]
            logger.info(f"Worker unregistered: {worker_id}")
            return True
    
    async def update_worker_heartbeat(self,
                                    worker_id: str,
                                    cpu_usage: float = 0.0,
                                    memory_usage: float = 0.0) -> bool:
        """Update worker heartbeat and status.
        
        Args:
            worker_id: Worker identifier
            cpu_usage: Current CPU usage percentage
            memory_usage: Current memory usage percentage
            
        Returns:
            True if update successful
        """
        async with self._lock:
            if worker_id not in self.workers:
                return False
            
            worker = self.workers[worker_id]
            worker.last_heartbeat = time.time()
            worker.cpu_usage = cpu_usage
            worker.memory_usage = memory_usage
            
            # Update status based on health
            if worker.is_healthy:
                if worker.status == WorkerStatus.OFFLINE:
                    worker.status = WorkerStatus.AVAILABLE
            else:
                worker.status = WorkerStatus.OFFLINE
            
            return True
    
    async def complete_task(self,
                          task_id: str,
                          worker_id: str,
                          result: Any = None,
                          error: str = None) -> bool:
        """Mark task as completed by worker.
        
        Args:
            task_id: Task identifier
            worker_id: Worker that completed the task
            result: Task result (if successful)
            error: Error message (if failed)
            
        Returns:
            True if completion successful
        """
        async with self._lock:
            task = self.tasks.get(task_id)
            
            if not task or task.worker_id != worker_id:
                return False
            
            task.completed_at = time.time()
            
            if error:
                task.status = TaskStatus.FAILED
                task.error = error
                self.stats["tasks_failed"] += 1
                
                # Retry if possible
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.worker_id = None
                    task.started_at = None
                    task.completed_at = None
                    task.error = None
                    await self.task_queue.put((-task.priority, time.time(), task_id))
                    logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
                else:
                    # Move to completed tasks
                    self.completed_tasks[task_id] = task
                    del self.tasks[task_id]
                    logger.error(f"Task {task_id} failed permanently: {error}")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                self.stats["tasks_completed"] += 1
                
                # Update statistics
                if task.execution_time:
                    self.stats["total_execution_time"] += task.execution_time
                    self.stats["average_execution_time"] = (
                        self.stats["total_execution_time"] / self.stats["tasks_completed"]
                    )
                
                # Move to completed tasks
                self.completed_tasks[task_id] = task
                del self.tasks[task_id]
                logger.debug(f"Task {task_id} completed successfully")
            
            # Update worker status
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.status = WorkerStatus.AVAILABLE
                worker.current_task = None
                
                if error:
                    worker.total_errors += 1
                else:
                    worker.total_tasks_completed += 1
            
            # Remove worker assignment
            if worker_id in self.worker_assignments:
                del self.worker_assignments[worker_id]
            
            return True
    
    async def _scheduler_loop(self):
        """Main scheduler loop for assigning tasks to workers."""
        while self._scheduler_running:
            try:
                # Get next task from queue (with timeout to allow shutdown)
                try:
                    priority, timestamp, task_id = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Find available worker
                worker = await self._find_available_worker(task_id)
                
                if worker:
                    await self._assign_task_to_worker(task_id, worker)
                else:
                    # No workers available, requeue task
                    await self.task_queue.put((priority, timestamp, task_id))
                    await asyncio.sleep(0.1)  # Short delay before retry
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _find_available_worker(self, task_id: str) -> Optional[WorkerNode]:
        """Find the best available worker for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Best available worker or None
        """
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            available_workers = [
                worker for worker in self.workers.values()
                if (worker.status == WorkerStatus.AVAILABLE and 
                    worker.is_healthy)
            ]
            
            if not available_workers:
                return None
            
            # Score workers based on load and capabilities
            def score_worker(worker):
                base_score = 100
                
                # Prefer workers with lower CPU usage
                base_score -= worker.cpu_usage
                
                # Prefer workers with lower memory usage
                base_score -= worker.memory_usage
                
                # Prefer workers with lower error rate
                base_score -= worker.error_rate * 50
                
                return base_score
            
            # Return best worker
            return max(available_workers, key=score_worker)
    
    async def _assign_task_to_worker(self, task_id: str, worker: WorkerNode):
        """Assign task to worker.
        
        Args:
            task_id: Task identifier
            worker: Worker to assign task to
        """
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.worker_id = worker.worker_id
            
            worker.status = WorkerStatus.BUSY
            worker.current_task = task_id
            
            self.worker_assignments[worker.worker_id] = task_id
        
        # Send task to worker (this would be implemented based on communication protocol)
        await self._send_task_to_worker(task, worker)
        
        logger.debug(f"Task {task_id} assigned to worker {worker.worker_id}")
    
    async def _send_task_to_worker(self, task: DistributedTask, worker: WorkerNode):
        """Send task to worker node.
        
        This is a placeholder for the actual communication implementation.
        In a real system, this would use HTTP, gRPC, or message queues.
        """
        # Placeholder for task communication
        logger.debug(f"Sending task {task.task_id} to worker {worker.worker_id}")
    
    async def _heartbeat_monitor(self):
        """Monitor worker heartbeats and handle failures."""
        while self._scheduler_running:
            try:
                async with self._lock:
                    offline_workers = []
                    
                    for worker_id, worker in self.workers.items():
                        if not worker.is_healthy and worker.status != WorkerStatus.OFFLINE:
                            worker.status = WorkerStatus.OFFLINE
                            offline_workers.append(worker_id)
                            
                            # Handle assigned task
                            if worker_id in self.worker_assignments:
                                task_id = self.worker_assignments[worker_id]
                                task = self.tasks.get(task_id)
                                
                                if task and task.status == TaskStatus.RUNNING:
                                    # Requeue the task
                                    task.status = TaskStatus.PENDING
                                    task.worker_id = None
                                    task.started_at = None
                                    await self.task_queue.put((-task.priority, time.time(), task_id))
                                    logger.warning(f"Requeuing task {task_id} due to worker {worker_id} failure")
                                
                                del self.worker_assignments[worker_id]
                
                if offline_workers:
                    logger.warning(f"Workers offline: {offline_workers}")
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5.0)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.
        
        Returns:
            System status information
        """
        async with self._lock:
            worker_status = {}
            for status in WorkerStatus:
                worker_status[status.value] = sum(
                    1 for w in self.workers.values() if w.status == status
                )
            
            task_status = {}
            for status in TaskStatus:
                task_status[status.value] = sum(
                    1 for t in self.tasks.values() if t.status == status
                )
            
            return {
                "scheduler_running": self._scheduler_running,
                "workers": {
                    "total": len(self.workers),
                    "by_status": worker_status,
                    "average_cpu": sum(w.cpu_usage for w in self.workers.values()) / len(self.workers) if self.workers else 0,
                    "average_memory": sum(w.memory_usage for w in self.workers.values()) / len(self.workers) if self.workers else 0
                },
                "tasks": {
                    "total_active": len(self.tasks),
                    "total_completed": len(self.completed_tasks),
                    "queue_size": self.task_queue.qsize(),
                    "by_status": task_status
                },
                "statistics": self.stats.copy(),
                "timestamp": time.time()
            }


class AsyncCarbonProcessor:
    """Async processor for carbon tracking computations."""
    
    def __init__(self, max_concurrent: int = 10):
        """Initialize async carbon processor.
        
        Args:
            max_concurrent: Maximum concurrent operations
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks: Set[asyncio.Task] = set()
        
        logger.info(f"Async carbon processor initialized with {max_concurrent} max concurrent operations")
    
    async def process_metrics_batch(self,
                                  metrics_list: List[Dict[str, Any]],
                                  processing_func: Callable) -> List[Any]:
        """Process batch of metrics asynchronously.
        
        Args:
            metrics_list: List of metrics to process
            processing_func: Function to apply to each metric
            
        Returns:
            List of processed results
        """
        async def process_single(metrics):
            async with self.semaphore:
                # Convert sync function to async if needed
                if asyncio.iscoroutinefunction(processing_func):
                    return await processing_func(metrics)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, processing_func, metrics)
        
        # Create tasks for all metrics
        tasks = [process_single(metrics) for metrics in metrics_list]
        
        # Track active tasks
        for task in tasks:
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def stream_process(self,
                           data_stream: asyncio.Queue,
                           processing_func: Callable,
                           output_queue: asyncio.Queue) -> None:
        """Process streaming data asynchronously.
        
        Args:
            data_stream: Input data queue
            processing_func: Processing function
            output_queue: Output queue for results
        """
        async def process_item():
            while True:
                try:
                    # Get item from stream with timeout
                    item = await asyncio.wait_for(data_stream.get(), timeout=1.0)
                    
                    async with self.semaphore:
                        # Process item
                        if asyncio.iscoroutinefunction(processing_func):
                            result = await processing_func(item)
                        else:
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(None, processing_func, item)
                        
                        # Put result in output queue
                        await output_queue.put(result)
                        
                        # Mark task as done
                        data_stream.task_done()
                        
                except asyncio.TimeoutError:
                    # No new items, continue loop
                    continue
                except Exception as e:
                    logger.error(f"Error processing stream item: {e}")
                    await output_queue.put({"error": str(e)})
        
        # Start processing tasks
        workers = [asyncio.create_task(process_item()) for _ in range(self.max_concurrent)]
        
        try:
            # Wait for all items to be processed
            await data_stream.join()
        finally:
            # Cancel workers
            for worker in workers:
                worker.cancel()
            
            await asyncio.gather(*workers, return_exceptions=True)
    
    async def shutdown(self):
        """Shutdown async processor and wait for completion."""
        # Wait for all active tasks to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        
        logger.info("Async carbon processor shutdown")


# Global instances
_distributed_scheduler = None
_async_processor = None


def get_distributed_scheduler() -> DistributedTaskScheduler:
    """Get global distributed scheduler instance."""
    global _distributed_scheduler
    if _distributed_scheduler is None:
        _distributed_scheduler = DistributedTaskScheduler()
    return _distributed_scheduler


def get_async_processor() -> AsyncCarbonProcessor:
    """Get global async processor instance."""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncCarbonProcessor()
    return _async_processor


async def initialize_distributed_computing():
    """Initialize distributed computing components."""
    # Initialize scheduler
    scheduler = get_distributed_scheduler()
    await scheduler.start()
    
    # Initialize async processor
    get_async_processor()
    
    logger.info("Distributed computing initialized")