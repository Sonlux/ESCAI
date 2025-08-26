"""
Unit tests for error handling and resilience components.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from escai_framework.utils.exceptions import (
    ESCAIBaseException, InstrumentationError, ProcessingError,
    StorageError, APIError, NetworkError, ConfigurationError,
    MonitoringOverheadError, AuthenticationError, ValidationError,
    ErrorSeverity, ErrorCategory
)
from escai_framework.utils.retry import (
    RetryManager, RetryConfig, BackoffStrategy, RetryExhaustedException,
    retry_async, retry_sync
)
from escai_framework.utils.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState,
    MonitoringCircuitBreaker, get_circuit_breaker
)
from escai_framework.utils.fallback import (
    FallbackManager, FallbackResult, FallbackStrategy,
    RuleBasedEpistemicExtractor, execute_with_fallback
)
from escai_framework.utils.load_shedding import (
    LoadShedder, LoadLevel, Priority, SystemMonitor,
    GracefulDegradationManager, LoadSheddingError
)
from escai_framework.utils.error_tracking import (
    ErrorTracker, ErrorEvent, StructuredLogger, MonitoringDecorator,
    LogLevel, get_error_tracker, monitor_errors
)


class TestExceptions:
    """Test custom exception hierarchy."""
    
    def test_base_exception_creation(self):
        """Test ESCAIBaseException creation and serialization."""
        exception = ESCAIBaseException(
            "Test error",
            error_code="TEST_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PROCESSING,
            context={"key": "value"},
            recovery_hint="Try again"
        )
        
        assert exception.message == "Test error"
        assert exception.error_code == "TEST_001"
        assert exception.severity == ErrorSeverity.HIGH
        assert exception.category == ErrorCategory.PROCESSING
        assert exception.context == {"key": "value"}
        assert exception.recovery_hint == "Try again"
        
        # Test serialization
        data = exception.to_dict()
        assert data["error_type"] == "ESCAIBaseException"
        assert data["message"] == "Test error"
        assert data["severity"] == "high"
    
    def test_instrumentation_errors(self):
        """Test instrumentation-specific errors."""
        error = MonitoringOverheadError(0.15, 0.10)
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.INSTRUMENTATION
        assert "15.00%" in error.message
        assert "10.00%" in error.message
    
    def test_processing_errors(self):
        """Test processing-specific errors."""
        error = ProcessingError("Analysis failed")
        assert error.category == ErrorCategory.PROCESSING
        
        # Test with context
        data = error.to_dict()
        assert data["category"] == "processing"
    
    def test_api_errors(self):
        """Test API-specific errors."""
        auth_error = AuthenticationError("Invalid token")
        assert auth_error.status_code == 401
        assert auth_error.category == ErrorCategory.AUTHENTICATION
        
        validation_error = ValidationError({"field1": "required", "field2": "invalid"})
        assert validation_error.status_code == 400
        assert "field1: required" in validation_error.message


class TestRetryMechanism:
    """Test retry mechanisms with exponential backoff."""
    
    def test_retry_config(self):
        """Test retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        
        retry_manager = RetryManager(config)
        
        # Test delay calculation
        assert retry_manager.calculate_delay(1) == 2.0
        assert retry_manager.calculate_delay(2) == 4.0
        assert retry_manager.calculate_delay(3) == 8.0
    
    def test_exponential_backoff_with_jitter(self):
        """Test exponential backoff with jitter."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            jitter_range=0.1
        )
        
        retry_manager = RetryManager(config)
        
        # Test multiple calculations to ensure jitter varies
        delays = [retry_manager.calculate_delay(2) for _ in range(10)]
        
        # All delays should be around 2.0 but with variation
        assert all(1.8 <= delay <= 2.2 for delay in delays)
        assert len(set(delays)) > 1  # Should have variation due to jitter
    
    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test async retry with eventual success."""
        call_count = 0
        
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        config = RetryConfig(max_attempts=5, base_delay=0.01)
        retry_manager = RetryManager(config)
        
        result = await retry_manager.execute_async(flaky_function)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_exhausted(self):
        """Test async retry exhaustion."""
        async def always_fails():
            raise ConnectionError("Always fails")
        
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        retry_manager = RetryManager(config)
        
        with pytest.raises(RetryExhaustedException) as exc_info:
            await retry_manager.execute_async(always_fails)
        
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, ConnectionError)
    
    def test_sync_retry_success(self):
        """Test synchronous retry with eventual success."""
        call_count = 0
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"
        
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        retry_manager = RetryManager(config)
        
        result = retry_manager.execute_sync(flaky_function)
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_decorator(self):
        """Test retry decorator."""
        call_count = 0
        
        @retry_async(max_attempts=3, base_delay=0.01)
        async def decorated_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "decorated_success"
        
        result = await decorated_function()
        
        assert result == "decorated_success"
        assert call_count == 2


class TestCircuitBreaker:
    """Test circuit breaker patterns."""
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0
        )
        
        breaker = CircuitBreaker("test", config)
        
        assert breaker.name == "test"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.config.failure_threshold == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        breaker = CircuitBreaker("test", config)
        
        async def failing_function():
            raise ConnectionError("Service unavailable")
        
        # First failure
        with pytest.raises(ConnectionError):
            await breaker.call_async(failing_function)
        assert breaker.state == CircuitState.CLOSED
        
        # Second failure - should open circuit
        with pytest.raises(ConnectionError):
            await breaker.call_async(failing_function)
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=1
        )
        breaker = CircuitBreaker("test", config)
        
        # Cause failure to open circuit
        async def failing_function():
            raise ConnectionError("Service unavailable")
        
        with pytest.raises(ConnectionError):
            await breaker.call_async(failing_function)
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        
        # Successful call should close circuit
        async def success_function():
            return "success"
        
        result = await breaker.call_async(success_function)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
    
    def test_monitoring_circuit_breaker(self):
        """Test monitoring-specific circuit breaker."""
        breaker = MonitoringCircuitBreaker()
        
        # Set baseline performance
        breaker.set_baseline_performance(1.0, 1000, 0.5)
        
        # Test overhead calculation
        breaker.metrics.response_times.extend([1.2, 1.3, 1.1])
        breaker.metrics.memory_usage.extend([1100, 1150, 1050])
        
        overhead = breaker.calculate_overhead()
        
        assert overhead["response_time"] > 0  # Should show increased response time
        assert overhead["memory_usage"] > 0   # Should show increased memory usage


class TestFallbackMechanisms:
    """Test fallback mechanisms for ML/NLP failures."""
    
    @pytest.mark.asyncio
    async def test_rule_based_epistemic_extractor(self):
        """Test rule-based fallback for epistemic extraction."""
        extractor = RuleBasedEpistemicExtractor()
        
        # Test with text containing beliefs and goals
        text = "I believe that the system is working correctly. I want to complete this task successfully."
        
        result = await extractor.execute_fallback(text, Exception("Model failed"))
        
        assert result.success
        assert result.strategy_used == FallbackStrategy.RULE_BASED
        assert result.confidence == 0.6
        
        epistemic_state = result.result
        assert len(epistemic_state.belief_states) > 0
        assert len(epistemic_state.goal_state.primary_goals) > 0
    
    @pytest.mark.asyncio
    async def test_fallback_manager(self):
        """Test fallback manager coordination."""
        manager = FallbackManager()
        
        async def failing_primary():
            raise ProcessingError("Primary method failed")
        
        # Test with input that should trigger rule-based fallback
        result = await manager.execute_with_fallback(
            failing_primary,
            "I think this will work. My goal is to succeed."
        )
        
        assert result.success
        assert result.strategy_used in [FallbackStrategy.RULE_BASED, FallbackStrategy.STATISTICAL_BASELINE]
    
    @pytest.mark.asyncio
    async def test_fallback_caching(self):
        """Test fallback result caching."""
        manager = FallbackManager()
        
        call_count = 0
        
        async def counting_primary():
            nonlocal call_count
            call_count += 1
            raise ProcessingError("Always fails")
        
        # First call
        result1 = await manager.execute_with_fallback(
            counting_primary,
            "test input",
            cache_key="test_cache"
        )
        
        # Second call with same cache key
        result2 = await manager.execute_with_fallback(
            counting_primary,
            "test input",
            cache_key="test_cache"
        )
        
        # Primary should only be called once due to caching
        assert call_count == 1
        assert result1.success == result2.success


class TestLoadShedding:
    """Test load shedding and graceful degradation."""
    
    @pytest.mark.asyncio
    async def test_system_monitor(self):
        """Test system performance monitoring."""
        monitor = SystemMonitor(monitoring_interval=0.1)
        
        await monitor.start_monitoring()
        await asyncio.sleep(0.2)  # Let it collect some metrics
        await monitor.stop_monitoring()
        
        metrics = monitor.get_current_metrics()
        assert metrics is not None
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0
    
    def test_load_level_determination(self):
        """Test load level determination logic."""
        from escai_framework.utils.load_shedding import SystemMetrics, LoadThresholds
        
        shedder = LoadShedder()
        
        # Test low load
        low_metrics = SystemMetrics(cpu_usage=50, memory_usage=50, response_time=0.5)
        assert shedder.determine_load_level(low_metrics) == LoadLevel.LOW
        
        # Test high load
        high_metrics = SystemMetrics(cpu_usage=90, memory_usage=90, response_time=5.0)
        assert shedder.determine_load_level(high_metrics) == LoadLevel.HIGH
        
        # Test critical load
        critical_metrics = SystemMetrics(cpu_usage=98, memory_usage=98, response_time=15.0)
        assert shedder.determine_load_level(critical_metrics) == LoadLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_load_shedding_behavior(self):
        """Test load shedding decisions."""
        shedder = LoadShedder()
        
        # Mock high system load
        with patch.object(shedder.system_monitor, 'get_current_metrics') as mock_metrics:
            from escai_framework.utils.load_shedding import SystemMetrics
            mock_metrics.return_value = SystemMetrics(
                cpu_usage=90, memory_usage=90, response_time=5.0
            )
            
            # Optional priority requests should be shed
            should_shed_optional = any(
                shedder.should_shed_request(Priority.OPTIONAL) for _ in range(10)
            )
            assert should_shed_optional
            
            # Critical priority requests should not be shed
            should_shed_critical = any(
                shedder.should_shed_request(Priority.CRITICAL) for _ in range(10)
            )
            assert not should_shed_critical
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_manager(self):
        """Test graceful degradation manager."""
        manager = GracefulDegradationManager()
        
        # Register a feature
        manager.register_feature("test_feature", Priority.MEDIUM)
        
        # Feature should be available initially
        assert manager.is_feature_available("test_feature")
        
        # Degrade the feature
        manager.degrade_feature("test_feature", "Testing degradation")
        assert not manager.is_feature_available("test_feature")
        
        # Restore the feature
        manager.restore_feature("test_feature")
        assert manager.is_feature_available("test_feature")


class TestErrorTracking:
    """Test error tracking and monitoring."""
    
    def test_error_event_creation(self):
        """Test error event creation and serialization."""
        exception = ValueError("Test error")
        event = ErrorEvent.from_exception(
            exception,
            context={"key": "value"},
            component="test_component",
            function_name="test_function"
        )
        
        assert event.exception_type == "ValueError"
        assert event.exception_message == "Test error"
        assert event.component == "test_component"
        assert event.function_name == "test_function"
        assert event.context == {"key": "value"}
        
        # Test serialization
        data = event.to_dict()
        assert data["exception_type"] == "ValueError"
        assert data["component"] == "test_component"
    
    def test_error_tracker(self):
        """Test error tracking functionality."""
        tracker = ErrorTracker()
        
        # Track some errors
        exception1 = ValueError("Error 1")
        exception2 = ConnectionError("Error 2")
        
        event_id1 = tracker.track_exception(exception1, component="comp1")
        event_id2 = tracker.track_exception(exception2, component="comp2")
        
        assert event_id1 != event_id2
        
        # Check metrics
        metrics = tracker.get_metrics()
        assert metrics["total_errors"] == 2
        assert "ValueError" in str(metrics["errors_by_component"])
    
    def test_error_pattern_detection(self):
        """Test error pattern detection."""
        tracker = ErrorTracker()
        
        # Create multiple similar errors
        for i in range(5):
            exception = ValueError(f"Recurring error {i}")
            tracker.track_exception(exception, component="test_component")
        
        patterns = tracker.get_error_patterns(min_occurrences=3)
        assert len(patterns) > 0
        
        pattern_key = "ValueError:test_component"
        assert pattern_key in patterns
        assert patterns[pattern_key]["count"] == 5
    
    def test_structured_logger(self):
        """Test structured logging with error tracking."""
        tracker = ErrorTracker()
        logger = StructuredLogger("test_logger", tracker)
        
        # Log an error
        exception = RuntimeError("Test runtime error")
        logger.error("Test error message", exception=exception, function_name="test_func")
        
        # Check that error was tracked
        metrics = tracker.get_metrics()
        assert metrics["total_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_monitoring_decorator(self):
        """Test monitoring decorator functionality."""
        tracker = ErrorTracker()
        decorator = MonitoringDecorator(tracker, "test_component")
        
        @decorator
        async def test_function():
            raise ValueError("Decorated function error")
        
        with pytest.raises(ValueError):
            await test_function()
        
        # Check that error was tracked
        metrics = tracker.get_metrics()
        assert metrics["total_errors"] == 1
        
        events = tracker.get_events(limit=1)
        assert len(events) == 1
        assert events[0].function_name == "test_function"
        assert events[0].component == "test_component"


class TestIntegration:
    """Integration tests for error handling components."""
    
    @pytest.mark.asyncio
    async def test_full_error_handling_pipeline(self):
        """Test complete error handling pipeline."""
        # Setup components
        error_tracker = ErrorTracker()
        fallback_manager = FallbackManager()
        
        # Create a function that fails and needs fallback
        async def unreliable_function(data):
            if "fail" in data:
                raise ProcessingError("Simulated processing failure")
            return f"processed: {data}"
        
        # Test successful execution
        result = await fallback_manager.execute_with_fallback(
            unreliable_function,
            "success_data"
        )
        assert result.success
        assert "processed: success_data" in str(result.result)
        
        # Test fallback execution
        result = await fallback_manager.execute_with_fallback(
            unreliable_function,
            "fail_data with beliefs and goals"
        )
        assert result.success  # Should succeed via fallback
        assert result.strategy_used != FallbackStrategy.RULE_BASED  # Used fallback
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retry(self):
        """Test circuit breaker combined with retry mechanism."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        breaker = CircuitBreaker("integration_test", config)
        
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        retry_manager = RetryManager(retry_config)
        
        call_count = 0
        
        async def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count <= 4:  # Fail first 4 calls
                raise ConnectionError("Service down")
            return "service_response"
        
        # This should fail and open the circuit
        with pytest.raises(ConnectionError):
            await retry_manager.execute_async(
                lambda: breaker.call_async(flaky_service)
            )
        
        # Circuit should be open now
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery
        await asyncio.sleep(0.15)
        
        # Should work after recovery
        result = await breaker.call_async(flaky_service)
        assert result == "service_response"
        assert breaker.state == CircuitState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__])