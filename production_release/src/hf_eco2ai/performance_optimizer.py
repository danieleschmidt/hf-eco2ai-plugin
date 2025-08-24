"""Performance optimization and scaling features for carbon tracking."""

import time
import threading
import asyncio
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import queue
import weakref
import gc
from functools import lru_cache, wraps
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    timestamp: float
    thread_id: str
    process_id: int
    cache_hit: bool = False
    batch_size: Optional[int] = None


class PerformanceCache:
    """High-performance caching system with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        self._ttls = {}
        self._lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str, default=None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return default
            
            # Check TTL
            if time.time() > self._ttls.get(key, float('inf')):
                self._remove_key(key)
                self.misses += 1
                return default
            
            # Update access time for LRU
            self._access_times[key] = time.time()
            self.hits += 1
            return self._cache[key]
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._ttls[key] = time.time() + (ttl or self.default_ttl)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all structures."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._ttls.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        self._remove_key(lru_key)
        self.evictions += 1
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._ttls.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class BatchProcessor:
    """Efficient batch processing for metrics and data."""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 5.0,
                 max_workers: int = None):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        
        # Batch storage
        self._batches = {}
        self._batch_locks = {}
        self._last_flush = {}
        
        # Processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._running = False
        self._flush_thread = None
        
        logger.info(f"Initialized batch processor: batch_size={batch_size}, workers={self.max_workers}")
    
    def start(self) -> None:
        """Start batch processing."""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        logger.info("Batch processor started")
    
    def stop(self) -> None:
        """Stop batch processing and flush remaining."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        
        # Flush all remaining batches
        for batch_type in list(self._batches.keys()):
            self._flush_batch(batch_type)
        
        self.executor.shutdown(wait=True)
        logger.info("Batch processor stopped")
    
    def add_item(self, batch_type: str, item: Any, processor: Callable[[List], Any]) -> None:
        """Add item to batch for processing."""
        if batch_type not in self._batches:
            self._batches[batch_type] = []
            self._batch_locks[batch_type] = threading.Lock()
            self._last_flush[batch_type] = time.time()
        
        with self._batch_locks[batch_type]:
            self._batches[batch_type].append((item, processor))
            
            # Flush if batch is full
            if len(self._batches[batch_type]) >= self.batch_size:
                self._schedule_flush(batch_type)
    
    def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            try:
                current_time = time.time()
                for batch_type, last_flush in self._last_flush.items():
                    if current_time - last_flush >= self.flush_interval:
                        self._schedule_flush(batch_type)
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in batch flush loop: {e}")
    
    def _schedule_flush(self, batch_type: str) -> None:
        """Schedule batch for flushing."""
        self.executor.submit(self._flush_batch, batch_type)
    
    def _flush_batch(self, batch_type: str) -> None:
        """Flush a batch of items."""
        with self._batch_locks.get(batch_type, threading.Lock()):
            if not self._batches.get(batch_type):
                return
            
            batch = self._batches[batch_type].copy()
            self._batches[batch_type].clear()
            self._last_flush[batch_type] = time.time()
        
        if not batch:
            return
        
        try:
            # Group by processor
            processor_groups = {}
            for item, processor in batch:
                if processor not in processor_groups:
                    processor_groups[processor] = []
                processor_groups[processor].append(item)
            
            # Process each group
            for processor, items in processor_groups.items():
                try:
                    processor(items)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
            
            logger.debug(f"Flushed batch {batch_type} with {len(batch)} items")
            
        except Exception as e:
            logger.error(f"Error flushing batch {batch_type}: {e}")


class AsyncMetricsCollector:
    """Asynchronous metrics collection system."""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.metrics_queue = asyncio.Queue(maxsize=max_queue_size)
        self.collectors = {}
        self.running = False
        
    async def start(self) -> None:
        """Start async metrics collection."""
        self.running = True
        asyncio.create_task(self._process_metrics())
        logger.info("Async metrics collector started")
    
    async def stop(self) -> None:
        """Stop metrics collection."""
        self.running = False
        logger.info("Async metrics collector stopped")
    
    async def collect_metric(self, metric_type: str, data: Any) -> None:
        """Collect a metric asynchronously."""
        try:
            await self.metrics_queue.put((metric_type, data, time.time()), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning("Metrics queue full, dropping metric")
    
    def register_collector(self, metric_type: str, collector: Callable[[Any], None]) -> None:
        """Register a collector for a metric type."""
        self.collectors[metric_type] = collector
    
    async def _process_metrics(self) -> None:
        """Process metrics from queue."""
        while self.running:
            try:
                metric_type, data, timestamp = await asyncio.wait_for(
                    self.metrics_queue.get(), timeout=1.0
                )
                
                collector = self.collectors.get(metric_type)
                if collector:
                    # Run collector in thread pool for CPU-bound work
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, collector, data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing metric: {e}")


def cached_property(ttl: float = 300.0):
    """Property decorator with TTL caching."""
    def decorator(func):
        cache_attr = f"_cached_{func.__name__}"
        timestamp_attr = f"_cached_{func.__name__}_timestamp"
        
        @property
        @wraps(func)
        def wrapper(self):
            current_time = time.time()
            
            # Check if cached value exists and is still valid
            if (hasattr(self, cache_attr) and 
                hasattr(self, timestamp_attr) and
                current_time - getattr(self, timestamp_attr) < ttl):
                return getattr(self, cache_attr)
            
            # Compute new value
            value = func(self)
            setattr(self, cache_attr, value)
            setattr(self, timestamp_attr, current_time)
            
            return value
        
        return wrapper
    
    return decorator


def memory_efficient_hash(data: Any) -> str:
    """Memory-efficient hashing for cache keys."""
    if isinstance(data, (str, int, float, bool)):
        return hashlib.md5(str(data).encode()).hexdigest()[:16]
    elif isinstance(data, dict):
        # Sort keys for consistent hashing
        sorted_items = sorted(data.items())
        return hashlib.md5(json.dumps(sorted_items, sort_keys=True).encode()).hexdigest()[:16]
    else:
        # Use pickle for complex objects
        try:
            return hashlib.md5(pickle.dumps(data)).hexdigest()[:16]
        except (TypeError, pickle.PickleError):
            return hashlib.md5(str(data).encode()).hexdigest()[:16]


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.cache = PerformanceCache(max_size=5000)
        self.batch_processor = BatchProcessor(batch_size=50)
        self.metrics = []
        self._metrics_lock = threading.Lock()
        
        # Resource pools
        self.thread_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="perf-opt")
        
        # Optimization features
        self.enable_caching = True
        self.enable_batching = True
        self.enable_compression = False
        
        logger.info("Performance optimizer initialized")
    
    def start(self) -> None:
        """Start performance optimization."""
        if self.enable_batching:
            self.batch_processor.start()
        logger.info("Performance optimizer started")
    
    def stop(self) -> None:
        """Stop performance optimization."""
        self.batch_processor.stop()
        self.thread_pool.shutdown(wait=True)
        logger.info("Performance optimizer stopped")
    
    def optimize_function(self, func: Callable, cache_ttl: float = 300.0) -> Callable:
        """Apply performance optimizations to a function."""
        if self.enable_caching:
            func = self._add_caching(func, cache_ttl)
        
        func = self._add_performance_tracking(func)
        return func
    
    def _add_caching(self, func: Callable, ttl: float) -> Callable:
        """Add caching to function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{memory_efficient_hash((args, kwargs))}"
            
            # Try cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache
            result = func(*args, **kwargs)
            self.cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    def _add_performance_tracking(self, func: Callable) -> Callable:
        """Add performance tracking to function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                # Record metrics
                duration = time.time() - start_time
                memory_delta = self._get_memory_usage() - start_memory
                
                metric = PerformanceMetrics(
                    operation=func.__name__,
                    duration=duration,
                    memory_usage=memory_delta,
                    cpu_usage=0.0,  # Could be enhanced with psutil
                    timestamp=time.time(),
                    thread_id=str(threading.current_thread().ident),
                    process_id=multiprocessing.current_process().pid
                )
                
                with self._metrics_lock:
                    self.metrics.append(metric)
                    # Keep only recent metrics
                    if len(self.metrics) > 1000:
                        self.metrics = self.metrics[-500:]
                
                return result
                
            except Exception as e:
                logger.error(f"Performance tracking error in {func.__name__}: {e}")
                raise
        
        return wrapper
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance optimization summary."""
        with self._metrics_lock:
            if not self.metrics:
                return {"status": "no_data"}
            
            # Aggregate metrics
            total_operations = len(self.metrics)
            avg_duration = sum(m.duration for m in self.metrics) / total_operations
            total_memory = sum(m.memory_usage for m in self.metrics)
            
            # Operations by type
            operations = {}
            for metric in self.metrics:
                op = metric.operation
                if op not in operations:
                    operations[op] = {"count": 0, "total_duration": 0.0}
                operations[op]["count"] += 1
                operations[op]["total_duration"] += metric.duration
            
            # Calculate averages
            for op_data in operations.values():
                op_data["avg_duration"] = op_data["total_duration"] / op_data["count"]
        
        return {
            "total_operations": total_operations,
            "avg_duration": avg_duration,
            "total_memory_delta": total_memory,
            "cache_stats": self.cache.stats(),
            "operations": operations,
            "optimizations": {
                "caching_enabled": self.enable_caching,
                "batching_enabled": self.enable_batching,
                "compression_enabled": self.enable_compression
            }
        }
    
    def optimize_for_scale(self, expected_load: str = "medium") -> None:
        """Optimize configuration for expected load."""
        if expected_load == "low":
            self.cache.max_size = 1000
            self.batch_processor.batch_size = 25
        elif expected_load == "medium":
            self.cache.max_size = 5000
            self.batch_processor.batch_size = 50
        elif expected_load == "high":
            self.cache.max_size = 10000
            self.batch_processor.batch_size = 100
            self.enable_compression = True
        
        logger.info(f"Optimized for {expected_load} load")
    
    def cleanup_resources(self) -> None:
        """Cleanup optimization resources."""
        self.cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Performance optimizer resources cleaned up")


# Global performance optimizer
_performance_optimizer = PerformanceOptimizer()


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    return _performance_optimizer


def optimized(cache_ttl: float = 300.0):
    """Decorator to optimize function performance."""
    def decorator(func):
        return _performance_optimizer.optimize_function(func, cache_ttl)
    return decorator


def start_performance_optimization():
    """Start global performance optimization."""
    _performance_optimizer.start()


def stop_performance_optimization():
    """Stop global performance optimization."""
    _performance_optimizer.stop()