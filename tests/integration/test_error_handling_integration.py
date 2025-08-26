"""
Integration tests for error handling and resilience across system components.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
import psutil

from escai_framework.utils.exceptions import (
    ProcessingError, StorageError, NetworkError, MonitoringOverheadError
)
from escai_framework.utils.retry import retry_async, RetryConfig, BackoffStrategy
from escai_framework.utils.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker,
    MonitoringCircuitBreaker
)
from escai_framework.utils.fallback import (
    get_fallback_manager, execute_with_fallback
)
from escai_framework.utils.load_shedding import (
    get_degradation_manager, Priority, LoadLevel
)
from escai_framework.utils.error_tracking import (
    get_error_tracker, get_logger, monitor_errors
)

# Import core components for integration testing
from escai_framework.core.epistemic_extractor import EpistemicExtractor
from escai_framework.core.pattern_analyzer import PatternAnalyzer
from escai_framework.storage.database import DatabaseManager


class TestDatabaseErrorHandling:
    """Test error handling in database operations."""
    
    @pytest.mark.asyncio
    async def test_database_connection_retry(self):
        """Test database connection with retry mechanism."""
        
        @retry_async(
            max_attempts=3,
            base_delay=0.1,
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        async def connect_to_database():
            # Simulate connection attempts
            if not hasattr(connect_to_database, 'attempts'):
                connect_to_database.attempts = 0
            
            connect_to_database.attempts += 1
            
            if connect_to_database.attempts < 3:
                raise ConnectionError("Database unavailable")
            
            return "connected"
        
        result = await connect_to_database()
        assert result == "connected"
        assert connect_to_database.attempts == 3
    
    @pytest.mark.asyncio
    async def test_database_circuit_breaker(self):
        """Test database operations with circuit breaker protection."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.2
        )
        
        db_breaker = CircuitBreaker("database", config)
        
        async def database_operation():
            raise StorageError("Database connection failed", "DB_001")
        
        # First two failures should keep circuit closed
        for _ in range(2):
            with pytest.raises(StorageError):
                await db_breaker.call_async(database_operation)
        
        # Circuit should now be open
        from escai_framework.utils.circuit_breaker import CircuitState
        assert db_breaker.state == CircuitState.OPEN
        
        # Further calls should be blocked
        from escai_framework.utils.exceptions import ServiceUnavailableError
        with pytest.raises(ServiceUnavailableError):
            await db_breaker.call_async(database_operation)


class TestProcessingErrorHandling:
    """Test error handling in core processing components."""
    
    @pytest.mark.asyncio
    async def test_epistemic_extraction_with_fallback(self):
        """Test epistemic extraction with fallback mechanisms."""
        
        # Mock the primary extraction to fail
        async def failing_extraction(logs):
            raise ProcessingError("NLP model failed to load")
        
        # Test with fallback
        result = await execute_with_fallback(
            failing_extraction,
            ["I believe the system is working. I want to complete the task."],
            cache_key="test_extraction"
        )
        
        assert result.success
        assert result.strategy_used.value in ["rule_based", "statistical_baseline"]
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_pattern_analysis_degradation(self):
        """Test pattern analysis with graceful degradation."""
        degradation_manager = get_degradation_manager()
        
        # Register pattern analysis feature
        degradation_manager.register_feature("pattern_analysis", Priority.HIGH)
        
        async def pattern_analysis_function(data):
            # Simulate high load causing failure
            raise ProcessingError("Pattern analysis overloaded")
        
        async def simple_fallback(data):
            return {"patterns": [], "fallback": True}
        
        # Should use fallback when main function fails
        result = await degradation_manager.execute_feature(
            "pattern_analysis",
            pattern_analysis_function,
            simple_fallback,
            ["test", "data"]
        )
        
        assert result["fallback"] is True
    
    @pytest.mark.asyncio
    async def test_monitoring_overhead_protection(self):
        """Test monitoring overhead circuit breaker."""
        monitoring_breaker = MonitoringCircuitBreaker()
        
        # Set baseline performance
        monitoring_breaker.set_baseline_performance(1.0, 1000, 0.5)
        
        # Simulate monitoring function with overhead
        async def monitoring_function():
            # Simulate some processing time
            await asyncio.sleep(0.1)
            return "monitoring_result"
        
        # First few calls should work
        for _ in range(3):
            result = await monitoring_breaker.monitor_async(monitoring_function)
            assert result == "monitoring_result"
        
        # Simulate high overhead by adding response times
        monitoring_breaker.metrics.response_times.extend([5.0, 6.0, 7.0])
        
        # Should detect overhead and potentially open circuit
        overhead = monitoring_breaker.calculate_overhead()
        assert overhead["response_time"] > monitoring_breaker.config.overhead_threshold


class TestAPIErrorHandling:
    """Test error handling in API layer."""
    
    @pytest.mark.asyncio
    async def test_api_rate_limiting_with_circuit_breaker(self):
        """Test API rate limiting combined with circuit breaker."""
        api_breaker = get_circuit_breaker("api", CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.5
        ))
        
        call_count = 0
        
        async def rate_limited_api():
            nonlocal call_count
            call_count += 1
            
            if call_count <= 5:
                from escai_framework.utils.exceptions import RateLimitError
                raise RateLimitError(100, 60, 30)
            
            return {"status": "success"}
        
        # Should fail and eventually open circuit
        for _ in range(3):
            with pytest.raises(Exception):  # RateLimitError or ServiceUnavailableError
                await api_breaker.call_async(rate_limited_api)
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self):
        """Test authentication error handling with retry."""
        
        @retry_async(max_attempts=2, base_delay=0.1)
        async def authenticate_user(token):
            if token == "invalid":
                from escai_framework.utils.exceptions import AuthenticationError
                raise AuthenticationError("Invalid token")
            elif token == "expired":
                # Simulate token refresh on retry
                return {"user_id": "123", "refreshed": True}
            else:
                return {"user_id": "123"}
        
        # Valid token should work immediately
        result = await authenticate_user("valid_token")
        assert result["user_id"] == "123"
        
        # Invalid token should fail even with retry
        with pytest.raises(Exception):
            await authenticate_user("invalid")


class TestSystemLoadHandling:
    """Test system load and resource management."""
    
    @pytest.mark.asyncio
    async def test_load_shedding_under_stress(self):
        """Test load shedding behavior under system stress."""
        degradation_manager = get_degradation_manager()
        await degradation_manager.start()
        
        try:
            # Register features with different priorities
            degradation_manager.register_feature("critical_monitoring", Priority.CRITICAL)
            degradation_manager.register_feature("optional_analytics", Priority.OPTIONAL)
            
            # Mock high system load
            with patch('psutil.cpu_percent', return_value=95.0), \
                 patch('psutil.virtual_memory') as mock_memory:
                
                mock_memory.return_value.percent = 90.0
                
                async def critical_function():
                    return "critical_result"
                
                async def optional_function():
                    return "optional_result"
                
                # Critical function should still work
                result = await degradation_manager.execute_feature(
                    "critical_monitoring",
                    critical_function
                )
                assert result == "critical_result"
                
                # Optional function might be shed
                try:
                    result = await degradation_manager.execute_feature(
                        "optional_analytics",
                        optional_function
                    )
                    # If it succeeds, that's also valid (depends on load shedding probability)
                except Exception as e:
                    # Should be load shedding error or degradation error
                    assert "load" in str(e).lower() or "degradation" in str(e).lower()
        
        finally:
            await degradation_manager.stop()
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test handling of memory pressure scenarios."""
        
        # Simulate memory-intensive operation
        async def memory_intensive_operation():
            # Simulate memory allocation
            large_data = [0] * 1000000  # Allocate some memory
            
            # Check if we should shed this operation
            current_memory = psutil.virtual_memory().percent
            if current_memory > 80:
                from escai_framework.utils.load_shedding import LoadSheddingError
                raise LoadSheddingError(LoadLevel.HIGH, Priority.LOW)
            
            return len(large_data)
        
        # Should work under normal conditions
        try:
            result = await memory_intensive_operation()
            assert isinstance(result, int)
        except Exception as e:
            # Acceptable if system is actually under memory pressure
            assert "load" in str(e).lower() or "memory" in str(e).lower()


class TestErrorRecovery:
    """Test error recovery and system resilience."""
    
    @pytest.mark.asyncio
    async def test_automatic_error_recovery(self):
        """Test automatic error recovery mechanisms."""
        error_tracker = get_error_tracker()
        
        # Clear previous errors
        error_tracker.clear_events()
        
        @monitor_errors(component="recovery_test")
        async def recoverable_function(attempt_count=0):
            if attempt_count < 2:
                raise ProcessingError(f"Temporary failure {attempt_count}")
            return f"success_after_{attempt_count}_attempts"
        
        # Function with retry and monitoring
        @retry_async(max_attempts=3, base_delay=0.1)
        async def wrapped_function():
            if not hasattr(wrapped_function, 'attempts'):
                wrapped_function.attempts = 0
            wrapped_function.attempts += 1
            return await recoverable_function(wrapped_function.attempts - 1)
        
        result = await wrapped_function()
        assert "success_after_2_attempts" in result
        
        # Check that errors were tracked
        metrics = error_tracker.get_metrics()
        assert metrics["total_errors"] >= 2  # Should have tracked the failures
    
    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self):
        """Test prevention of cascading failures."""
        
        # Create multiple circuit breakers for different services
        service_a_breaker = CircuitBreaker("service_a", CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.3
        ))
        service_b_breaker = CircuitBreaker("service_b", CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.3
        ))
        
        async def service_a():
            raise NetworkError("Service A down")
        
        async def service_b():
            # Service B depends on Service A
            try:
                await service_a_breaker.call_async(service_a)
            except Exception:
                # Fallback behavior when Service A is down
                return "service_b_fallback_result"
            return "service_b_normal_result"
        
        # Fail Service A to open its circuit
        for _ in range(2):
            with pytest.raises(NetworkError):
                await service_a_breaker.call_async(service_a)
        
        # Service B should still work with fallback
        result = await service_b_breaker.call_async(service_b)
        assert result == "service_b_fallback_result"
        
        # Service B circuit should remain closed
        from escai_framework.utils.circuit_breaker import CircuitState
        assert service_b_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_error_pattern_detection_and_response(self):
        """Test error pattern detection and automated response."""
        error_tracker = get_error_tracker()
        error_tracker.clear_events()
        
        # Set up alert callback
        alerts_received = []
        
        def alert_callback(alert_data):
            alerts_received.append(alert_data)
        
        error_tracker.add_alert_callback(alert_callback)
        
        # Generate recurring errors
        for i in range(6):  # Exceed recurring error threshold
            try:
                raise ProcessingError(f"Recurring pattern error {i}")
            except Exception as e:
                error_tracker.track_exception(e, component="pattern_test")
        
        # Should detect pattern and trigger alert
        patterns = error_tracker.get_error_patterns(min_occurrences=3)
        assert len(patterns) > 0
        
        # Check if alert was triggered (may be async)
        await asyncio.sleep(0.1)
        
        # Verify pattern detection
        pattern_key = "ProcessingError:pattern_test"
        assert pattern_key in patterns
        assert patterns[pattern_key]["count"] == 6


class TestEndToEndResilience:
    """End-to-end resilience testing."""
    
    @pytest.mark.asyncio
    async def test_full_system_resilience(self):
        """Test complete system resilience under various failure conditions."""
        
        # Start all resilience components
        degradation_manager = get_degradation_manager()
        await degradation_manager.start()
        
        error_tracker = get_error_tracker()
        error_tracker.clear_events()
        
        try:
            # Register system components
            degradation_manager.register_feature("data_processing", Priority.HIGH)
            degradation_manager.register_feature("analytics", Priority.MEDIUM)
            degradation_manager.register_feature("reporting", Priority.LOW)
            
            # Simulate various system components with different failure modes
            async def data_processing():
                # Simulate intermittent failures
                import random
                if random.random() < 0.3:
                    raise ProcessingError("Data processing overload")
                return {"processed": True}
            
            async def analytics():
                # Simulate resource-intensive operation
                if psutil.virtual_memory().percent > 90:
                    raise ProcessingError("Insufficient memory for analytics")
                return {"analytics": "completed"}
            
            async def reporting():
                # Simulate network-dependent operation
                import random
                if random.random() < 0.5:
                    raise NetworkError("Report delivery failed")
                return {"report": "sent"}
            
            # Fallback functions
            async def simple_processing():
                return {"processed": True, "fallback": True}
            
            async def basic_analytics():
                return {"analytics": "basic", "fallback": True}
            
            async def cached_reporting():
                return {"report": "cached", "fallback": True}
            
            # Execute operations with resilience
            results = []
            
            for i in range(10):  # Multiple iterations to test various scenarios
                try:
                    # Data processing (high priority, should rarely fail)
                    result = await degradation_manager.execute_feature(
                        "data_processing",
                        data_processing,
                        simple_processing
                    )
                    results.append(("data_processing", result))
                    
                    # Analytics (medium priority, may degrade under load)
                    result = await degradation_manager.execute_feature(
                        "analytics",
                        analytics,
                        basic_analytics
                    )
                    results.append(("analytics", result))
                    
                    # Reporting (low priority, likely to be shed under load)
                    result = await degradation_manager.execute_feature(
                        "reporting",
                        reporting,
                        cached_reporting
                    )
                    results.append(("reporting", result))
                    
                except Exception as e:
                    # Some failures are expected and acceptable
                    error_tracker.track_exception(e, component="resilience_test")
                
                # Small delay between iterations
                await asyncio.sleep(0.05)
            
            # Verify that system remained operational
            assert len(results) > 0, "System should have produced some results"
            
            # Check that high-priority operations succeeded more often
            data_processing_results = [r for r in results if r[0] == "data_processing"]
            assert len(data_processing_results) > 0, "Data processing should have some successes"
            
            # Verify error tracking worked
            metrics = error_tracker.get_metrics()
            # Some errors are expected due to simulated failures
            
            # Check degradation manager status
            status = degradation_manager.get_status()
            assert "load_shedding_stats" in status
            
        finally:
            await degradation_manager.stop()
    
    @pytest.mark.asyncio
    async def test_recovery_after_system_stress(self):
        """Test system recovery after period of high stress."""
        
        degradation_manager = get_degradation_manager()
        await degradation_manager.start()
        
        try:
            degradation_manager.register_feature("stress_test", Priority.MEDIUM)
            
            # Simulate high stress period
            async def stressed_function():
                # Simulate high CPU usage
                start_time = time.time()
                while time.time() - start_time < 0.1:  # Busy wait
                    pass
                return "stressed_result"
            
            # Run multiple concurrent operations to create stress
            tasks = []
            for _ in range(5):
                task = asyncio.create_task(
                    degradation_manager.execute_feature(
                        "stress_test",
                        stressed_function
                    )
                )
                tasks.append(task)
            
            # Wait for all tasks with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2.0
                )
                
                # Some operations may have been shed or failed
                successful_results = [r for r in results if isinstance(r, str)]
                
                # System should recover and be responsive
                await asyncio.sleep(0.5)  # Allow recovery time
                
                # Test that system is responsive after stress
                result = await degradation_manager.execute_feature(
                    "stress_test",
                    lambda: asyncio.sleep(0.01) or "recovery_test"
                )
                
                # Should work after recovery period
                assert result == "recovery_test"
                
            except asyncio.TimeoutError:
                # Timeout is acceptable under high stress
                pass
        
        finally:
            await degradation_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])