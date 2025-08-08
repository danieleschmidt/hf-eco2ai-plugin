#!/usr/bin/env python3
"""Test scaling and performance optimization features of HF Eco2AI Plugin."""

import sys
import time
import threading
import concurrent.futures
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hf_eco2ai import Eco2AICallback, CarbonConfig
from hf_eco2ai.performance_optimizer import (
    get_performance_optimizer, PerformanceCache, BatchProcessor, optimized
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_performance_cache():
    """Test high-performance caching system."""
    print("🧪 Testing performance caching...")
    
    cache = PerformanceCache(max_size=100, default_ttl=1.0)
    
    # Test basic operations
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    
    # Test cache miss
    assert cache.get("nonexistent") is None
    
    # Test TTL expiration
    cache.put("ttl_test", "value", ttl=0.1)
    time.sleep(0.2)
    assert cache.get("ttl_test") is None
    
    # Test cache stats
    stats = cache.stats()
    assert "hit_rate" in stats
    assert stats["total_requests"] > 0
    
    print(f"✓ Cache working - Hit rate: {stats['hit_rate']:.2f}, Size: {stats['size']}")
    return True


def test_batch_processing():
    """Test efficient batch processing."""
    print("🧪 Testing batch processing...")
    
    processed_batches = []
    
    def batch_processor(items):
        """Mock batch processor."""
        processed_batches.append(len(items))
    
    batch_proc = BatchProcessor(batch_size=5, flush_interval=0.5)
    batch_proc.start()
    
    # Add items to batch
    for i in range(12):  # Should create 2 full batches + 1 partial
        batch_proc.add_item("test_batch", f"item_{i}", batch_processor)
    
    # Wait for processing
    time.sleep(1.0)
    batch_proc.stop()
    
    # Should have processed items (might be in fewer batches due to timing)
    assert len(processed_batches) >= 1
    assert sum(processed_batches) == 12  # All items should be processed
    print(f"✓ Batch processing working - {len(processed_batches)} batches processed")
    return True


def test_function_optimization():
    """Test function optimization decorators."""
    print("🧪 Testing function optimization...")
    
    call_count = 0
    
    @optimized(cache_ttl=0.5)
    def expensive_function(x, y):
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)  # Simulate expensive operation
        return x + y
    
    # First call should execute
    result1 = expensive_function(1, 2)
    assert result1 == 3
    assert call_count == 1
    
    # Second call should be cached
    result2 = expensive_function(1, 2)
    assert result2 == 3
    assert call_count == 1  # Should not increment
    
    # Different parameters should execute
    result3 = expensive_function(2, 3)
    assert result3 == 5
    assert call_count == 2
    
    print(f"✓ Function optimization working - {call_count} actual calls for 3 invocations")
    return True


def test_concurrent_callback_operations():
    """Test callback under concurrent load."""
    print("🧪 Testing concurrent callback operations...")
    
    config = CarbonConfig(
        project_name="concurrent-test",
        country="USA",
        region="CA",
        enable_performance_optimization=True,
        expected_load="high"
    )
    
    callback = Eco2AICallback(config)
    results = []
    errors = []
    
    def worker_function(worker_id):
        """Simulate concurrent callback operations."""
        try:
            for i in range(10):
                # Simulate getting metrics
                metrics = callback.get_current_metrics()
                assert isinstance(metrics, dict)
                
                # Simulate processing
                time.sleep(0.001)  # 1ms work
                
                results.append(f"worker_{worker_id}_op_{i}")
                
        except Exception as e:
            errors.append(f"Worker {worker_id}: {e}")
    
    # Run concurrent workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_function, i) for i in range(10)]
        concurrent.futures.wait(futures, timeout=10.0)
    
    # Check results
    assert len(results) == 100  # 10 workers * 10 operations each
    assert len(errors) == 0, f"Errors occurred: {errors}"
    
    print(f"✓ Concurrent operations working - {len(results)} operations completed")
    return True


def test_memory_efficiency():
    """Test memory efficiency under load."""
    print("🧪 Testing memory efficiency...")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create multiple callbacks to simulate load
    callbacks = []
    for i in range(50):  # Create 50 callbacks
        config = CarbonConfig(
            project_name=f"memory-test-{i}",
            enable_performance_optimization=True
        )
        callback = Eco2AICallback(config)
        callbacks.append(callback)
        
        # Generate some activity
        for j in range(5):
            callback.get_current_metrics()
    
    # Check memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Should not increase memory by more than 100MB for 50 callbacks
    assert memory_increase < 100, f"Memory increased by {memory_increase:.1f} MB"
    
    # Cleanup
    optimizer = get_performance_optimizer()
    optimizer.cleanup_resources()
    
    print(f"✓ Memory efficiency good - Increased by {memory_increase:.1f} MB for 50 callbacks")
    return True


def test_performance_optimizer():
    """Test the performance optimizer system."""
    print("🧪 Testing performance optimizer...")
    
    optimizer = get_performance_optimizer()
    optimizer.start()
    
    # Test optimization for different loads
    for load in ["low", "medium", "high"]:
        optimizer.optimize_for_scale(load)
        
        # Verify cache size changes
        if load == "low":
            assert optimizer.cache.max_size == 1000
        elif load == "medium":
            assert optimizer.cache.max_size == 5000
        elif load == "high":
            assert optimizer.cache.max_size == 10000
    
    # Test performance summary
    summary = optimizer.get_performance_summary()
    assert "total_operations" in summary
    assert "cache_stats" in summary
    assert "optimizations" in summary
    
    optimizer.stop()
    
    print("✓ Performance optimizer working correctly")
    return True


def test_cache_performance():
    """Test cache performance under load."""
    print("🧪 Testing cache performance under load...")
    
    cache = PerformanceCache(max_size=1000)
    
    # Warm up cache
    for i in range(500):
        cache.put(f"key_{i}", f"value_{i}")
    
    # Test read performance
    start_time = time.time()
    for i in range(1000):
        key = f"key_{i % 500}"  # 50% cache hits
        cache.get(key)
    
    duration = time.time() - start_time
    operations_per_second = 1000 / duration
    
    # Should handle at least 10k ops/sec
    assert operations_per_second > 10000, f"Cache too slow: {operations_per_second:.0f} ops/sec"
    
    stats = cache.stats()
    hit_rate = stats["hit_rate"]
    
    # Should have reasonable hit rate
    assert hit_rate > 0.4, f"Hit rate too low: {hit_rate:.2f}"
    
    print(f"✓ Cache performance good - {operations_per_second:.0f} ops/sec, {hit_rate:.2f} hit rate")
    return True


def test_scaling_configuration():
    """Test auto-scaling configuration features."""
    print("🧪 Testing scaling configuration...")
    
    # Test different scale configurations
    configs = {
        "low_scale": CarbonConfig(
            project_name="low-scale-test",
            enable_performance_optimization=True,
            expected_load="low"
        ),
        "high_scale": CarbonConfig(
            project_name="high-scale-test", 
            enable_performance_optimization=True,
            expected_load="high"
        )
    }
    
    for scale_name, config in configs.items():
        callback = Eco2AICallback(config)
        
        # Should initialize without errors
        assert callback.performance_optimizer is not None
        
        # Test that it can handle operations
        metrics = callback.get_current_metrics()
        assert isinstance(metrics, dict)
        
        print(f"✓ {scale_name} configuration working")
    
    return True


def main():
    """Run all scaling tests."""
    print("🚀 Starting HF Eco2AI Plugin Scaling Tests\n")
    
    tests = [
        test_performance_cache,
        test_batch_processing,
        test_function_optimization,
        test_concurrent_callback_operations,
        test_memory_efficiency,
        test_performance_optimizer,
        test_cache_performance,
        test_scaling_configuration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print("✅ PASSED\n")
        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"🎯 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All scaling tests passed! Generation 3 implementation scales!")
        return 0
    else:
        print("⚠️  Some scaling tests failed. Check implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())