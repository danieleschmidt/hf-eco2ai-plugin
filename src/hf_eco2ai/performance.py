"""Performance optimization and scaling features for carbon tracking."""

import time
import logging
import asyncio
import threading
import multiprocessing
import queue
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import weakref
import gc
import psutil
from collections import defaultdict, deque
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking system performance."""
    
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    gpu_memory_usage: Dict[int, float] = field(default_factory=dict)
    disk_io_read: float = 0.0
    disk_io_write: float = 0.0
    network_bytes_sent: float = 0.0
    network_bytes_recv: float = 0.0
    cache_hit_rate: float = 0.0
    active_threads: int = 0
    active_processes: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class AdvancedCache:
    """High-performance cache with multiple eviction strategies."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl: Optional[float] = None,
                 eviction_strategy: str = "lru"):
        """Initialize advanced cache.
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time-to-live for cache entries in seconds
            eviction_strategy: Eviction strategy (lru, lfu, random)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.eviction_strategy = eviction_strategy
        
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._access_order: deque = deque()
        self._lock = threading.RLock()
        
        self.stats = CacheStats(max_size=max_size)
        
        logger.debug(f"Initialized cache with max_size={max_size}, strategy={eviction_strategy}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            # Check if key exists and is not expired
            if key in self._cache:
                if not self._is_expired(key):
                    self._update_access(key)
                    self.stats.hits += 1
                    return self._cache[key]
                else:
                    # Remove expired key
                    self._remove_key(key)
            
            self.stats.misses += 1
            return default
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Remove existing key if present
            if key in self._cache:
                self._remove_key(key)
            
            # Check if we need to evict items
            if len(self._cache) >= self.max_size:
                self._evict_items(1)
            
            # Add new item
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._update_access(key)
            
            self.stats.size = len(self._cache)
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if key was removed
        """
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._access_counts.clear()
            self._access_order.clear()
            self.stats.size = 0
    
    def cleanup_expired(self) -> int:
        """Remove expired entries.
        
        Returns:
            Number of expired entries removed
        """
        if not self.ttl:
            return 0
        
        with self._lock:
            expired_keys = [
                key for key in self._cache.keys()
                if self._is_expired(key)
            ]
            
            for key in expired_keys:
                self._remove_key(key)
            
            return len(expired_keys)
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if not self.ttl:
            return False
        
        return time.time() - self._timestamps.get(key, 0) > self.ttl
    
    def _update_access(self, key: str) -> None:
        """Update access tracking for key."""
        self._access_counts[key] += 1
        
        # Update LRU order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all tracking structures."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._access_counts.pop(key, None)
        
        if key in self._access_order:
            self._access_order.remove(key)
        
        self.stats.size = len(self._cache)
    
    def _evict_items(self, count: int) -> None:
        """Evict items based on eviction strategy."""
        if self.eviction_strategy == "lru":
            self._evict_lru(count)
        elif self.eviction_strategy == "lfu":
            self._evict_lfu(count)
        elif self.eviction_strategy == "random":
            self._evict_random(count)
        else:
            self._evict_lru(count)  # Default to LRU
        
        self.stats.evictions += count
    
    def _evict_lru(self, count: int) -> None:
        """Evict least recently used items."""
        for _ in range(min(count, len(self._access_order))):
            if self._access_order:
                key = self._access_order.popleft()
                self._remove_key(key)
    
    def _evict_lfu(self, count: int) -> None:
        """Evict least frequently used items."""
        # Sort by access count and remove least used
        sorted_keys = sorted(
            self._access_counts.keys(),
            key=lambda k: self._access_counts[k]
        )
        
        for i in range(min(count, len(sorted_keys))):
            self._remove_key(sorted_keys[i])
    
    def _evict_random(self, count: int) -> None:
        """Evict random items."""
        import random
        keys = list(self._cache.keys())
        random.shuffle(keys)
        
        for i in range(min(count, len(keys))):
            self._remove_key(keys[i])


class ResourcePool:
    """Pool for managing expensive resources like connections or objects."""
    
    def __init__(self,
                 factory: Callable[[], Any],
                 max_size: int = 10,
                 idle_timeout: float = 300.0):
        """Initialize resource pool.
        
        Args:
            factory: Function to create new resources
            max_size: Maximum pool size
            idle_timeout: Timeout for idle resources
        """
        self.factory = factory
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        
        self._pool: queue.Queue = queue.Queue(maxsize=max_size)
        self._created_count = 0
        self._active_count = 0
        self._lock = threading.Lock()
        
        # Track resource timestamps for cleanup
        self._resource_timestamps: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    @contextmanager
    def get_resource(self):
        """Get resource from pool with automatic return."""
        resource = self._acquire_resource()
        try:
            yield resource
        finally:
            self._release_resource(resource)
    
    def _acquire_resource(self) -> Any:
        """Acquire resource from pool."""
        with self._lock:
            try:
                # Try to get existing resource
                resource = self._pool.get_nowait()
                self._active_count += 1
                return resource
            except queue.Empty:
                # Create new resource if under limit
                if self._created_count < self.max_size:
                    resource = self.factory()
                    self._created_count += 1
                    self._active_count += 1
                    self._resource_timestamps[resource] = time.time()
                    return resource
                else:
                    # Wait for available resource
                    resource = self._pool.get()
                    self._active_count += 1
                    return resource
    
    def _release_resource(self, resource: Any) -> None:
        """Release resource back to pool."""
        with self._lock:
            self._active_count -= 1
            self._resource_timestamps[resource] = time.time()
            
            try:
                self._pool.put_nowait(resource)
            except queue.Full:
                # Pool is full, resource will be garbage collected
                self._created_count -= 1
    
    def _cleanup_worker(self) -> None:
        """Background worker to cleanup idle resources."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_idle_resources()
            except Exception as e:
                logger.error(f"Error in resource pool cleanup: {e}")
    
    def _cleanup_idle_resources(self) -> None:
        """Remove idle resources from pool."""
        current_time = time.time()
        resources_to_remove = []
        
        # Check pool for idle resources
        temp_resources = []
        
        try:
            while True:
                resource = self._pool.get_nowait()
                resource_time = self._resource_timestamps.get(resource, current_time)
                
                if current_time - resource_time > self.idle_timeout:
                    resources_to_remove.append(resource)
                else:
                    temp_resources.append(resource)
        except queue.Empty:
            pass
        
        # Return non-idle resources to pool
        for resource in temp_resources:
            try:
                self._pool.put_nowait(resource)
            except queue.Full:
                break
        
        # Update counts
        with self._lock:
            self._created_count -= len(resources_to_remove)
        
        if resources_to_remove:
            logger.debug(f"Cleaned up {len(resources_to_remove)} idle resources")


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        """Initialize performance monitor.
        
        Args:
            monitoring_interval: Interval between monitoring samples
        """
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_io_total": 100.0,  # MB/s
            "cache_hit_rate": 0.8
        }
        
        logger.info("Performance monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    
                    # Keep only recent metrics (last hour)
                    cutoff_time = time.time() - 3600
                    self.metrics_history = [
                        m for m in self.metrics_history
                        if m.timestamp > cutoff_time
                    ]
                
                # Check for performance issues
                self._check_performance_thresholds(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # CPU and memory
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            # Thread and process counts
            active_threads = threading.active_count()
            active_processes = len(psutil.pids())
            
            # GPU memory (if available)
            gpu_memory = self._get_gpu_memory_usage()
            
            return PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                memory_available=memory.available / (1024**3),  # GB
                gpu_memory_usage=gpu_memory,
                disk_io_read=disk_io.read_bytes / (1024**2) if disk_io else 0,  # MB
                disk_io_write=disk_io.write_bytes / (1024**2) if disk_io else 0,  # MB
                network_bytes_sent=network_io.bytes_sent / (1024**2) if network_io else 0,  # MB
                network_bytes_recv=network_io.bytes_recv / (1024**2) if network_io else 0,  # MB
                active_threads=active_threads,
                active_processes=active_processes
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return PerformanceMetrics()
    
    def _get_gpu_memory_usage(self) -> Dict[int, float]:
        """Get GPU memory usage."""
        gpu_memory = {}
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_memory[i] = (memory_info.used / memory_info.total) * 100
                
        except Exception:
            # GPU monitoring not available
            pass
        
        return gpu_memory
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check if performance metrics exceed thresholds."""
        if metrics.cpu_usage > self.thresholds["cpu_usage"]:
            logger.warning(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.thresholds["memory_usage"]:
            logger.warning(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        total_disk_io = metrics.disk_io_read + metrics.disk_io_write
        if total_disk_io > self.thresholds["disk_io_total"]:
            logger.warning(f"High disk I/O: {total_disk_io:.1f} MB/s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics.
        
        Returns:
            Performance summary
        """
        with self._lock:
            if not self.metrics_history:
                return {"status": "no_data"}
            
            recent_metrics = self.metrics_history[-10:]  # Last 10 samples
            
            # Calculate averages
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_threads = sum(m.active_threads for m in recent_metrics) / len(recent_metrics)
            
            # Peak values
            peak_cpu = max(m.cpu_usage for m in recent_metrics)
            peak_memory = max(m.memory_usage for m in recent_metrics)
            
            return {
                "status": "healthy" if avg_cpu < 50 and avg_memory < 70 else "stressed",
                "averages": {
                    "cpu_usage": avg_cpu,
                    "memory_usage": avg_memory,
                    "active_threads": avg_threads
                },
                "peaks": {
                    "cpu_usage": peak_cpu,
                    "memory_usage": peak_memory
                },
                "samples_count": len(recent_metrics),
                "monitoring_active": self._monitoring
            }


class ConcurrencyManager:
    """Manager for handling concurrent operations efficiently."""
    
    def __init__(self,
                 max_threads: int = None,
                 max_processes: int = None):
        """Initialize concurrency manager.
        
        Args:
            max_threads: Maximum thread pool size
            max_processes: Maximum process pool size
        """
        self.max_threads = max_threads or min(32, multiprocessing.cpu_count() * 4)
        self.max_processes = max_processes or multiprocessing.cpu_count()
        
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
        
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_results: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        logger.info(f"Concurrency manager initialized: {self.max_threads} threads, {self.max_processes} processes")
    
    async def run_async_batch(self,
                             tasks: List[Callable],
                             max_concurrent: int = 10) -> List[Any]:
        """Run batch of async tasks with concurrency limit.
        
        Args:
            tasks: List of async callables
            max_concurrent: Maximum concurrent tasks
            
        Returns:
            List of task results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(task):
            async with semaphore:
                return await task()
        
        # Create tasks with semaphore
        limited_tasks = [run_with_semaphore(task) for task in tasks]
        
        # Run all tasks
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        return results
    
    def run_parallel_cpu_bound(self,
                              func: Callable,
                              args_list: List[Tuple],
                              timeout: Optional[float] = None) -> List[Any]:
        """Run CPU-bound tasks in parallel using process pool.
        
        Args:
            func: Function to execute
            args_list: List of argument tuples for each call
            timeout: Timeout for all tasks
            
        Returns:
            List of results
        """
        futures = []
        
        for args in args_list:
            future = self.process_pool.submit(func, *args)
            futures.append(future)
        
        results = []
        
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel task failed: {e}")
                results.append(e)
        
        return results
    
    def run_parallel_io_bound(self,
                             func: Callable,
                             args_list: List[Tuple],
                             timeout: Optional[float] = None) -> List[Any]:
        """Run I/O-bound tasks in parallel using thread pool.
        
        Args:
            func: Function to execute
            args_list: List of argument tuples for each call
            timeout: Timeout for all tasks
            
        Returns:
            List of results
        """
        futures = []
        
        for args in args_list:
            future = self.thread_pool.submit(func, *args)
            futures.append(future)
        
        results = []
        
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel task failed: {e}")
                results.append(e)
        
        return results
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown thread and process pools.
        
        Args:
            wait: Whether to wait for shutdown
        """
        self.thread_pool.shutdown(wait=wait)
        self.process_pool.shutdown(wait=wait)
        
        logger.info("Concurrency manager shutdown")


def memoize(max_size: int = 128, ttl: Optional[float] = None):
    """Advanced memoization decorator with TTL and size limits.
    
    Args:
        max_size: Maximum cache size
        ttl: Time-to-live for cache entries
    """
    def decorator(func: Callable) -> Callable:
        cache = AdvancedCache(max_size=max_size, ttl=ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            
            return result
        
        # Expose cache methods
        wrapper.cache_info = lambda: cache.stats
        wrapper.cache_clear = cache.clear
        
        return wrapper
    return decorator


def async_cache(max_size: int = 128, ttl: Optional[float] = None):
    """Async version of memoization decorator.
    
    Args:
        max_size: Maximum cache size
        ttl: Time-to-live for cache entries
    """
    def decorator(func: Callable) -> Callable:
        cache = AdvancedCache(max_size=max_size, ttl=ttl)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = await func(*args, **kwargs)
            cache.put(key, result)
            
            return result
        
        # Expose cache methods
        wrapper.cache_info = lambda: cache.stats
        wrapper.cache_clear = cache.clear
        
        return wrapper
    return decorator


def optimize_numpy_operations():
    """Optimize NumPy operations for better performance."""
    try:
        import numpy as np
        
        # Set optimal BLAS threads
        cpu_count = multiprocessing.cpu_count()
        
        # For BLAS operations
        import os
        os.environ['OMP_NUM_THREADS'] = str(cpu_count)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
        
        logger.info(f"Optimized NumPy for {cpu_count} threads")
        
    except ImportError:
        logger.warning("NumPy not available for optimization")


def memory_efficient_generator(iterable, chunk_size: int = 1000):
    """Generator that processes data in chunks to reduce memory usage.
    
    Args:
        iterable: Input iterable
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of data
    """
    chunk = []
    
    for item in iterable:
        chunk.append(item)
        
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
            
            # Force garbage collection for memory management
            if len(chunk) % (chunk_size * 10) == 0:
                gc.collect()
    
    # Yield remaining items
    if chunk:
        yield chunk


# Global instances
_performance_monitor = None
_concurrency_manager = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_concurrency_manager() -> ConcurrencyManager:
    """Get global concurrency manager instance."""
    global _concurrency_manager
    if _concurrency_manager is None:
        _concurrency_manager = ConcurrencyManager()
    return _concurrency_manager


def initialize_performance_optimizations():
    """Initialize all performance optimizations."""
    # Optimize NumPy operations
    optimize_numpy_operations()
    
    # Initialize monitoring
    monitor = get_performance_monitor()
    monitor.start_monitoring()
    
    # Initialize concurrency manager
    get_concurrency_manager()
    
    logger.info("Performance optimizations initialized")