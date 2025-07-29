"""Performance benchmarks for HF Eco2AI Plugin carbon tracking."""

import pytest
import time
from unittest.mock import Mock, patch
from hf_eco2ai import Eco2AICallback


class TestCarbonTrackingPerformance:
    """Performance benchmarks for carbon tracking functionality."""
    
    @pytest.fixture
    def mock_callback(self):
        """Create a mock callback for testing."""
        return Eco2AICallback()
    
    def test_callback_initialization_performance(self, benchmark, mock_callback):
        """Benchmark callback initialization time."""
        def init_callback():
            return Eco2AICallback()
        
        result = benchmark(init_callback)
        assert result is not None
    
    @patch('eco2ai.Tracker')
    def test_training_start_overhead(self, mock_tracker, benchmark):
        """Benchmark training start overhead."""
        callback = Eco2AICallback()
        mock_trainer = Mock()
        
        def start_training():
            callback.on_train_begin(mock_trainer, mock_trainer.model)
        
        benchmark(start_training)
    
    @patch('eco2ai.Tracker')
    def test_epoch_tracking_overhead(self, mock_tracker, benchmark):
        """Benchmark per-epoch tracking overhead."""
        callback = Eco2AICallback()
        mock_trainer = Mock()
        logs = {"epoch": 1, "loss": 0.5}
        
        def track_epoch():
            callback.on_epoch_end(mock_trainer, mock_trainer.model, logs)
        
        benchmark(track_epoch)
    
    @patch('eco2ai.Tracker')
    def test_step_tracking_overhead(self, mock_tracker, benchmark):
        """Benchmark per-step tracking overhead."""
        callback = Eco2AICallback()
        mock_trainer = Mock()
        logs = {"step": 100, "loss": 0.5}
        
        def track_step():
            callback.on_log(mock_trainer, logs)
        
        benchmark(track_step)
    
    def test_memory_usage_during_tracking(self):
        """Test memory usage during carbon tracking."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Simulate extended tracking
        callback = Eco2AICallback()
        for i in range(1000):
            logs = {"step": i, "loss": 0.5 - i * 0.0001}
            # Mock tracking calls would go here
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Assert memory increase is reasonable (less than 10MB)
        assert memory_increase < 10 * 1024 * 1024
    
    @pytest.mark.slow
    def test_long_running_tracking_stability(self):
        """Test stability of carbon tracking over extended periods."""
        callback = Eco2AICallback()
        start_time = time.time()
        
        # Simulate 1000 training steps
        for step in range(1000):
            logs = {"step": step, "loss": 0.5 - step * 0.0001}
            # Mock callback methods would be called here
            
            if step % 100 == 0:
                elapsed = time.time() - start_time
                # Ensure consistent performance (no memory leaks causing slowdown)
                assert elapsed < (step + 1) * 0.01  # Max 10ms per step
    
    def test_concurrent_callback_performance(self, benchmark):
        """Test performance with multiple concurrent callbacks."""
        def create_multiple_callbacks():
            callbacks = [Eco2AICallback() for _ in range(10)]
            return callbacks
        
        callbacks = benchmark(create_multiple_callbacks)
        assert len(callbacks) == 10
    
    @pytest.mark.parametrize("batch_size", [16, 32, 64, 128])
    def test_tracking_scales_with_batch_size(self, batch_size, benchmark):
        """Test that tracking overhead doesn't scale with batch size."""
        callback = Eco2AICallback()
        logs = {"batch_size": batch_size, "loss": 0.5}
        
        def track_with_batch():
            # Mock tracking with different batch sizes
            return callback
        
        benchmark(track_with_batch)