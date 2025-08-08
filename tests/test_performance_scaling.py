"""Performance and scaling tests for carbon tracking system."""

import pytest
import asyncio
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
import psutil

from src.hf_eco2ai.performance import (
    PerformanceMonitor,
    AdvancedCache,
    ResourcePool,
    ConcurrencyManager,
    memoize,
    async_cache,
    memory_efficient_generator
)
from src.hf_eco2ai.distributed import (
    DistributedTaskScheduler,
    AsyncCarbonProcessor,
    DistributedTask,
    TaskStatus,
    WorkerNode,
    WorkerStatus
)


class TestAdvancedCache:
    """Test advanced caching functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        return AdvancedCache(max_size=10, ttl=1.0, eviction_strategy="lru")
    
    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_size == 10
        assert cache.ttl == 1.0
        assert cache.eviction_strategy == "lru"
        assert cache.stats.max_size == 10
    
    def test_cache_put_and_get(self, cache):
        """Test basic cache put and get operations."""
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        
        assert cache.stats.size == 2
    
    def test_cache_hit_miss_tracking(self, cache):
        """Test cache hit and miss tracking."""
        cache.put("key1", "value1")
        
        # Hit
        result = cache.get("key1")
        assert result == "value1"
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0
        
        # Miss
        result = cache.get("key2")
        assert result is None
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        
        # Hit rate
        assert cache.stats.hit_rate == 0.5
    
    def test_cache_ttl_expiration(self, cache):
        """Test TTL-based cache expiration."""
        cache.put("key1", "value1")
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("key1") is None
    
    def test_cache_size_eviction(self, cache):
        """Test size-based cache eviction."""
        # Fill cache to capacity
        for i in range(10):
            cache.put(f"key{i}", f"value{i}")
        
        assert cache.stats.size == 10
        
        # Add one more item to trigger eviction
        cache.put("key10", "value10")
        
        assert cache.stats.size == 10
        assert cache.stats.evictions == 1
        
        # Oldest item (key0) should be evicted in LRU
        assert cache.get("key0") is None
        assert cache.get("key10") == "value10"
    
    def test_cache_lfu_eviction(self):
        """Test LFU eviction strategy."""
        cache = AdvancedCache(max_size=3, eviction_strategy="lfu")
        
        # Add items
        cache.put("a", "value_a")
        cache.put("b", "value_b")
        cache.put("c", "value_c")
        
        # Access items with different frequencies
        cache.get("a")  # accessed 1 time
        cache.get("b")  # accessed 1 time
        cache.get("b")  # accessed 2 times total
        cache.get("c")  # accessed 1 time
        cache.get("c")  # accessed 2 times total
        cache.get("c")  # accessed 3 times total
        
        # Add new item to trigger eviction
        cache.put("d", "value_d")
        
        # Item "a" should be evicted (least frequently used)
        assert cache.get("a") is None
        assert cache.get("b") == "value_b"
        assert cache.get("c") == "value_c"
        assert cache.get("d") == "value_d"
    
    def test_cache_cleanup_expired(self, cache):
        """Test expired entries cleanup."""
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Cleanup expired entries
        expired_count = cache.cleanup_expired()
        
        assert expired_count == 2
        assert cache.stats.size == 0
    
    def test_cache_invalidation(self, cache):
        """Test cache invalidation."""
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Invalidate specific key
        result = cache.invalidate("key1")
        assert result == True
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        
        # Try to invalidate non-existent key
        result = cache.invalidate("nonexistent")
        assert result == False


class TestResourcePool:
    """Test resource pool functionality."""
    
    def test_resource_pool_creation(self):
        """Test resource pool creation."""
        def create_resource():
            return {"connection": "mock_connection"}
        
        pool = ResourcePool(factory=create_resource, max_size=5)
        
        assert pool.max_size == 5
        assert pool._created_count == 0
        assert pool._active_count == 0
    
    def test_resource_acquisition_and_release(self):
        """Test resource acquisition and release."""
        def create_resource():
            return {"id": time.time()}
        
        pool = ResourcePool(factory=create_resource, max_size=2)
        
        # Acquire resource
        with pool.get_resource() as resource:
            assert resource is not None
            assert "id" in resource
            assert pool._active_count == 1
        
        # Resource should be released
        assert pool._active_count == 0
    
    def test_resource_pool_limit(self):
        """Test resource pool size limit."""
        creation_count = 0
        
        def create_resource():
            nonlocal creation_count
            creation_count += 1
            return {"id": creation_count}
        
        pool = ResourcePool(factory=create_resource, max_size=2)
        
        # Acquire maximum resources
        with pool.get_resource() as resource1:
            with pool.get_resource() as resource2:
                assert pool._created_count == 2
                assert pool._active_count == 2
                
                # Both resources should be different
                assert resource1["id"] != resource2["id"]
    
    def test_resource_reuse(self):
        """Test resource reuse from pool."""
        def create_resource():
            return {"created_at": time.time()}
        
        pool = ResourcePool(factory=create_resource, max_size=2)
        
        # Use resource and release it
        first_resource_id = None
        with pool.get_resource() as resource:
            first_resource_id = resource["created_at"]
        
        # Acquire another resource - should reuse the first one
        with pool.get_resource() as resource:
            assert resource["created_at"] == first_resource_id


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    @pytest.fixture
    def monitor(self):
        """Create performance monitor instance."""
        return PerformanceMonitor(monitoring_interval=0.1)
    
    def test_monitor_initialization(self, monitor):
        """Test performance monitor initialization."""
        assert monitor.monitoring_interval == 0.1
        assert len(monitor.metrics_history) == 0
        assert monitor._monitoring == False
    
    def test_metrics_collection(self, monitor):
        """Test performance metrics collection."""
        metrics = monitor._collect_metrics()
        
        assert hasattr(metrics, 'cpu_usage')
        assert hasattr(metrics, 'memory_usage')
        assert hasattr(metrics, 'timestamp')
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0
        assert metrics.timestamp > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self, monitor):
        """Test starting and stopping monitoring."""
        monitor.start_monitoring()
        assert monitor._monitoring == True
        
        # Let it collect some metrics
        await asyncio.sleep(0.3)
        
        monitor.stop_monitoring()
        assert monitor._monitoring == False
        
        # Should have collected some metrics
        assert len(monitor.metrics_history) > 0
    
    def test_performance_summary(self, monitor):
        """Test performance summary generation."""
        # Add some mock metrics
        from src.hf_eco2ai.performance import PerformanceMetrics
        
        for i in range(5):
            metrics = PerformanceMetrics(
                cpu_usage=50.0 + i * 5,
                memory_usage=60.0 + i * 2,
                active_threads=10 + i
            )
            monitor.metrics_history.append(metrics)
        
        summary = monitor.get_performance_summary()
        
        assert "status" in summary
        assert "averages" in summary
        assert "peaks" in summary
        assert summary["samples_count"] == 5


class TestConcurrencyManager:
    """Test concurrency management functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create concurrency manager instance."""
        return ConcurrencyManager(max_threads=4, max_processes=2)
    
    def test_manager_initialization(self, manager):
        """Test concurrency manager initialization."""
        assert manager.max_threads == 4
        assert manager.max_processes == 2
        assert manager.thread_pool is not None
        assert manager.process_pool is not None
    
    def test_parallel_io_bound_execution(self, manager):
        """Test parallel execution of I/O-bound tasks."""
        def io_task(x):
            time.sleep(0.1)  # Simulate I/O
            return x * 2
        
        args_list = [(1,), (2,), (3,), (4,)]
        
        start_time = time.time()
        results = manager.run_parallel_io_bound(io_task, args_list, timeout=5.0)
        end_time = time.time()
        
        # Results should be correct
        assert sorted(results) == [2, 4, 6, 8]
        
        # Should be faster than sequential execution
        assert end_time - start_time < 0.4  # Less than 4 * 0.1
    
    def test_parallel_cpu_bound_execution(self, manager):
        """Test parallel execution of CPU-bound tasks."""
        def cpu_task(n):
            # Simulate CPU-intensive work
            total = 0
            for i in range(n * 1000):
                total += i
            return total
        
        args_list = [(100,), (200,), (300,), (400,)]
        
        results = manager.run_parallel_cpu_bound(cpu_task, args_list, timeout=10.0)
        
        # All tasks should complete
        assert len(results) == 4
        assert all(isinstance(result, int) for result in results)
    
    @pytest.mark.asyncio
    async def test_async_batch_execution(self, manager):
        """Test async batch execution."""
        async def async_task():
            await asyncio.sleep(0.1)
            return time.time()
        
        tasks = [async_task for _ in range(5)]
        
        start_time = time.time()
        results = await manager.run_async_batch(tasks, max_concurrent=3)
        end_time = time.time()
        
        # All tasks should complete
        assert len(results) == 5
        assert all(isinstance(result, float) for result in results)
        
        # Should respect concurrency limit (roughly)
        assert end_time - start_time >= 0.2  # At least 2 batches
        assert end_time - start_time < 0.5   # But not sequential


class TestMemoization:
    """Test memoization functionality."""
    
    def test_memoize_decorator(self):
        """Test memoization decorator."""
        call_count = 0
        
        @memoize(max_size=10, ttl=1.0)
        def expensive_function(x, y=1):
            nonlocal call_count
            call_count += 1
            return x * y
        
        # First call
        result1 = expensive_function(5, y=2)
        assert result1 == 10
        assert call_count == 1
        
        # Second call with same args - should use cache
        result2 = expensive_function(5, y=2)
        assert result2 == 10
        assert call_count == 1  # No additional call
        
        # Different args - should call function
        result3 = expensive_function(3, y=2)
        assert result3 == 6
        assert call_count == 2
        
        # Check cache info
        cache_info = expensive_function.cache_info()
        assert cache_info.hits == 1
        assert cache_info.misses == 2
    
    @pytest.mark.asyncio
    async def test_async_cache_decorator(self):
        """Test async cache decorator."""
        call_count = 0
        
        @async_cache(max_size=10, ttl=1.0)
        async def async_expensive_function(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x * 2
        
        # First call
        result1 = await async_expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call with same args - should use cache
        result2 = await async_expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # No additional call
        
        # Check cache info
        cache_info = async_expensive_function.cache_info()
        assert cache_info.hits == 1
        assert cache_info.misses == 1


class TestDistributedTaskScheduler:
    """Test distributed task scheduling."""
    
    @pytest.fixture
    async def scheduler(self):
        """Create and start scheduler instance."""
        scheduler = DistributedTaskScheduler(max_queue_size=100)
        await scheduler.start()
        yield scheduler
        await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.max_queue_size == 100
        assert scheduler._scheduler_running == True
        assert len(scheduler.tasks) == 0
        assert len(scheduler.workers) == 0
    
    @pytest.mark.asyncio
    async def test_task_submission(self, scheduler):
        """Test task submission."""
        task_id = await scheduler.submit_task(
            function_name="test_function",
            args=(1, 2),
            kwargs={"key": "value"},
            priority=5
        )
        
        assert task_id is not None
        assert task_id in scheduler.tasks
        
        task = scheduler.tasks[task_id]
        assert task.function_name == "test_function"
        assert task.args == (1, 2)
        assert task.kwargs == {"key": "value"}
        assert task.priority == 5
        assert task.status == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_worker_registration(self, scheduler):
        """Test worker registration."""
        success = await scheduler.register_worker(
            worker_id="worker1",
            host="localhost",
            port=8000,
            capabilities={"cpu", "gpu"}
        )
        
        assert success == True
        assert "worker1" in scheduler.workers
        
        worker = scheduler.workers["worker1"]
        assert worker.host == "localhost"
        assert worker.port == 8000
        assert worker.capabilities == {"cpu", "gpu"}
        assert worker.status == WorkerStatus.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_task_status_tracking(self, scheduler):
        """Test task status tracking."""
        task_id = await scheduler.submit_task("test_function")
        
        # Check initial status
        status = await scheduler.get_task_status(task_id)
        assert status["status"] == "pending"
        assert status["task_id"] == task_id
        
        # Register worker and let scheduler assign task
        await scheduler.register_worker("worker1", "localhost", 8000)
        await asyncio.sleep(0.1)  # Let scheduler process
        
        # Complete task
        await scheduler.complete_task(task_id, "worker1", result="test_result")
        
        # Check final status
        status = await scheduler.get_task_status(task_id)
        assert status["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_worker_heartbeat(self, scheduler):
        """Test worker heartbeat functionality."""
        await scheduler.register_worker("worker1", "localhost", 8000)
        
        # Update heartbeat
        success = await scheduler.update_worker_heartbeat(
            "worker1",
            cpu_usage=25.0,
            memory_usage=40.0
        )
        
        assert success == True
        
        worker = scheduler.workers["worker1"]
        assert worker.cpu_usage == 25.0
        assert worker.memory_usage == 40.0
        assert worker.is_healthy == True
    
    @pytest.mark.asyncio
    async def test_system_status(self, scheduler):
        """Test system status reporting."""
        # Add some workers and tasks
        await scheduler.register_worker("worker1", "localhost", 8000)
        await scheduler.register_worker("worker2", "localhost", 8001)
        
        await scheduler.submit_task("function1")
        await scheduler.submit_task("function2")
        
        status = await scheduler.get_system_status()
        
        assert "scheduler_running" in status
        assert "workers" in status
        assert "tasks" in status
        assert "statistics" in status
        
        assert status["scheduler_running"] == True
        assert status["workers"]["total"] == 2
        assert status["tasks"]["total_active"] == 2


class TestAsyncCarbonProcessor:
    """Test async carbon processing."""
    
    @pytest.fixture
    def processor(self):
        """Create async processor instance."""
        return AsyncCarbonProcessor(max_concurrent=5)
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.max_concurrent == 5
        assert processor.semaphore._value == 5
        assert len(processor._active_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processor):
        """Test batch processing functionality."""
        def processing_func(metrics):
            return metrics["value"] * 2
        
        metrics_list = [
            {"value": 1},
            {"value": 2},
            {"value": 3},
            {"value": 4},
            {"value": 5}
        ]
        
        results = await processor.process_metrics_batch(metrics_list, processing_func)
        
        assert len(results) == 5
        assert results == [2, 4, 6, 8, 10]
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self, processor):
        """Test async batch processing functionality."""
        async def async_processing_func(metrics):
            await asyncio.sleep(0.01)
            return metrics["value"] * 3
        
        metrics_list = [{"value": i} for i in range(10)]
        
        start_time = time.time()
        results = await processor.process_metrics_batch(metrics_list, async_processing_func)
        end_time = time.time()
        
        assert len(results) == 10
        assert results == [i * 3 for i in range(10)]
        
        # Should be faster than sequential execution
        assert end_time - start_time < 0.1  # 10 * 0.01 would be sequential
    
    @pytest.mark.asyncio
    async def test_stream_processing(self, processor):
        """Test stream processing functionality."""
        input_queue = asyncio.Queue()
        output_queue = asyncio.Queue()
        
        def processing_func(item):
            return item * 2
        
        # Start stream processing
        processing_task = asyncio.create_task(
            processor.stream_process(input_queue, processing_func, output_queue)
        )
        
        # Add items to stream
        for i in range(5):
            await input_queue.put(i)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Get results
        results = []
        while not output_queue.empty():
            results.append(await output_queue.get())
        
        # Cleanup
        processing_task.cancel()
        
        assert len(results) == 5
        assert sorted(results) == [0, 2, 4, 6, 8]


class TestMemoryEfficiency:
    """Test memory efficiency features."""
    
    def test_memory_efficient_generator(self):
        """Test memory efficient generator."""
        large_dataset = range(1000)
        
        chunks = list(memory_efficient_generator(large_dataset, chunk_size=100))
        
        assert len(chunks) == 10  # 1000 / 100
        assert len(chunks[0]) == 100
        assert len(chunks[-1]) == 100
        
        # Verify data integrity
        flattened = [item for chunk in chunks for item in chunk]
        assert flattened == list(range(1000))
    
    def test_generator_partial_chunk(self):
        """Test generator with partial last chunk."""
        dataset = range(250)
        
        chunks = list(memory_efficient_generator(dataset, chunk_size=100))
        
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 50  # Partial chunk


@pytest.mark.performance
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for various components."""
    
    def test_cache_performance(self, benchmark):
        """Benchmark cache performance."""
        cache = AdvancedCache(max_size=1000)
        
        def cache_operations():
            # Fill cache
            for i in range(500):
                cache.put(f"key{i}", f"value{i}")
            
            # Random access
            import random
            for _ in range(1000):
                key = f"key{random.randint(0, 499)}"
                cache.get(key)
        
        benchmark(cache_operations)
    
    def test_memoization_performance(self, benchmark):
        """Benchmark memoization performance."""
        @memoize(max_size=100)
        def fibonacci(n):
            if n < 2:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        def fibonacci_calculations():
            results = []
            for i in range(30):
                results.append(fibonacci(i))
            return results
        
        results = benchmark(fibonacci_calculations)
        assert len(results) == 30
    
    @pytest.mark.asyncio
    async def test_async_processing_performance(self, benchmark):
        """Benchmark async processing performance."""
        processor = AsyncCarbonProcessor(max_concurrent=10)
        
        async def process_large_batch():
            def simple_processing(metrics):
                return metrics["value"] ** 2
            
            metrics_list = [{"value": i} for i in range(1000)]
            return await processor.process_metrics_batch(metrics_list, simple_processing)
        
        results = await benchmark.pedantic(process_large_batch, rounds=5, iterations=1)
        assert len(results) == 1000


@pytest.mark.stress
class TestStressTests:
    """Stress tests for system limits."""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_task_submission(self):
        """Test system under high task submission load."""
        scheduler = DistributedTaskScheduler(max_queue_size=10000)
        await scheduler.start()
        
        try:
            # Submit many tasks quickly
            task_ids = []
            for i in range(1000):
                task_id = await scheduler.submit_task(f"function_{i}")
                task_ids.append(task_id)
            
            assert len(task_ids) == 1000
            assert len(scheduler.tasks) == 1000
            
        finally:
            await scheduler.stop()
    
    def test_cache_memory_pressure(self):
        """Test cache behavior under memory pressure."""
        # Create large cache
        cache = AdvancedCache(max_size=10000)
        
        # Fill with large objects
        large_data = "x" * 1000  # 1KB strings
        
        for i in range(5000):
            cache.put(f"key{i}", large_data)
        
        # Cache should handle this gracefully
        assert cache.stats.size <= cache.max_size
        assert cache.stats.evictions > 0
    
    def test_resource_pool_exhaustion(self):
        """Test resource pool under heavy load."""
        def create_resource():
            return {"data": "resource_data"}
        
        pool = ResourcePool(factory=create_resource, max_size=10)
        
        # Try to acquire more resources than pool size
        acquired_resources = []
        
        for i in range(15):
            try:
                with pool.get_resource() as resource:
                    acquired_resources.append(resource)
            except Exception:
                # Pool exhaustion is expected
                pass
        
        # Should have created resources up to limit
        assert pool._created_count <= pool.max_size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-skip"])