"""
Integration tests for framework robustness and error handling.

This module tests the enhanced framework integration robustness features
including version compatibility, error recovery, graceful degradation,
and adaptive sampling.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from escai_framework.instrumentation.framework_compatibility import (
    FrameworkCompatibilityManager, FrameworkDetector, FrameworkInfo, FrameworkStatus,
    CompatibilityRequirement, FrameworkVersion, get_compatibility_manager
)
from escai_framework.instrumentation.adaptive_sampling import (
    AdaptiveSamplingManager, SamplingConfig, SamplingStrategy, SamplingDecision,
    get_sampling_manager
)
from escai_framework.instrumentation.robust_instrumentor import (
    RobustInstrumentor, RobustConfig, InstrumentorState, RecoveryStrategy
)
from escai_framework.instrumentation.events import AgentEvent, EventType, EventSeverity
from escai_framework.utils.exceptions import (
    FrameworkNotSupportedError, MonitoringOverheadError, InstrumentationError
)


class MockFrameworkDetector(FrameworkDetector):
    """Mock framework detector for testing."""
    
    def __init__(self, name: str, status: FrameworkStatus = FrameworkStatus.AVAILABLE):
        self.name = name
        self.status = status
        self.version_str = "1.0.0"
        self.should_fail = False
    
    @property
    def framework_name(self) -> str:
        return self.name
    
    async def detect_framework(self) -> FrameworkInfo:
        if self.should_fail:
            raise Exception("Mock detection failure")
        
        version = FrameworkVersion.from_string(self.name, self.version_str)
        
        return FrameworkInfo(
            name=self.name,
            status=self.status,
            version=version,
            compatibility=self.get_compatibility_requirements(),
            available_features=["feature1", "feature2"],
            missing_features=[]
        )
    
    def get_compatibility_requirements(self) -> CompatibilityRequirement:
        return CompatibilityRequirement(
            min_version="0.5.0",
            max_version="2.0.0",
            required_features=["feature1", "feature2"]
        )
    
    async def validate_features(self, framework_module: Any) -> tuple:
        return ["feature1", "feature2"], []


class MockRobustInstrumentor(RobustInstrumentor):
    """Mock robust instrumentor for testing."""
    
    def __init__(self, framework_name: str = "test_framework", **kwargs):
        super().__init__(framework_name, **kwargs)
        self.start_monitoring_calls = []
        self.stop_monitoring_calls = []
        self.capture_event_calls = []
        self.should_fail_start = False
        self.should_fail_capture = False
        self.initialization_success = True
    
    async def _initialize_framework_specific(self):
        if not self.initialization_success:
            raise Exception("Mock initialization failure")
    
    async def _start_monitoring_protected(self, agent_id: str, config: Dict[str, Any]) -> str:
        if self.should_fail_start:
            raise Exception("Mock start monitoring failure")
        
        self.start_monitoring_calls.append((agent_id, config))
        return f"session_{len(self.start_monitoring_calls)}"
    
    async def _stop_monitoring_protected(self, session_id: str):
        self.stop_monitoring_calls.append(session_id)
        return Mock(session_id=session_id, total_events=100)
    
    async def _capture_event_protected(self, event: AgentEvent) -> None:
        if self.should_fail_capture:
            raise Exception("Mock capture event failure")
        
        self.capture_event_calls.append(event)
    
    async def _test_basic_functionality(self):
        if self.should_fail_start:
            raise Exception("Basic functionality test failed")
    
    def get_supported_events(self) -> List[EventType]:
        return [EventType.AGENT_START, EventType.AGENT_STOP, EventType.CUSTOM]
    
    def get_framework_name(self) -> str:
        return self.framework_name


@pytest.fixture
def compatibility_manager():
    """Create a fresh compatibility manager for testing."""
    manager = FrameworkCompatibilityManager()
    manager.clear_cache()
    return manager


@pytest.fixture
def sampling_manager():
    """Create a fresh sampling manager for testing."""
    config = SamplingConfig(
        strategy=SamplingStrategy.ADAPTIVE_RATE,
        base_sampling_rate=1.0,
        min_sampling_rate=0.1
    )
    manager = AdaptiveSamplingManager(config)
    manager.reset_metrics()
    return manager


@pytest.fixture
def mock_instrumentor():
    """Create a mock robust instrumentor for testing."""
    config = RobustConfig(
        max_consecutive_errors=3,
        error_recovery_timeout=1.0,
        adaptive_sampling=True
    )
    return MockRobustInstrumentor(config=config)


class TestFrameworkCompatibility:
    """Test framework compatibility detection and validation."""
    
    @pytest.mark.asyncio
    async def test_framework_detection_success(self, compatibility_manager):
        """Test successful framework detection."""
        # Register mock detector
        mock_detector = MockFrameworkDetector("test_framework")
        compatibility_manager.register_detector(mock_detector)
        
        # Test detection
        framework_info = await compatibility_manager.detect_framework("test_framework")
        
        assert framework_info.name == "test_framework"
        assert framework_info.status == FrameworkStatus.AVAILABLE
        assert framework_info.version.version == "1.0.0"
        assert "feature1" in framework_info.available_features
    
    @pytest.mark.asyncio
    async def test_framework_detection_failure(self, compatibility_manager):
        """Test framework detection failure handling."""
        # Register failing mock detector
        mock_detector = MockFrameworkDetector("failing_framework")
        mock_detector.should_fail = True
        compatibility_manager.register_detector(mock_detector)
        
        # Test detection
        framework_info = await compatibility_manager.detect_framework("failing_framework")
        
        assert framework_info.status == FrameworkStatus.INCOMPATIBLE
        assert framework_info.error_message is not None
    
    @pytest.mark.asyncio
    async def test_unsupported_framework(self, compatibility_manager):
        """Test handling of unsupported framework."""
        with pytest.raises(FrameworkNotSupportedError):
            await compatibility_manager.detect_framework("nonexistent_framework")
    
    @pytest.mark.asyncio
    async def test_version_compatibility_check(self):
        """Test version compatibility checking."""
        version = FrameworkVersion.from_string("test", "1.5.0")
        
        # Test compatible version
        assert version.is_compatible_with("1.0.0", "2.0.0")
        
        # Test incompatible version (too old)
        assert not version.is_compatible_with("2.0.0", "3.0.0")
        
        # Test incompatible version (too new)
        assert not version.is_compatible_with("0.5.0", "1.0.0")
    
    @pytest.mark.asyncio
    async def test_detect_all_frameworks(self, compatibility_manager):
        """Test detecting all frameworks concurrently."""
        # Register multiple mock detectors
        for i in range(3):
            detector = MockFrameworkDetector(f"framework_{i}")
            compatibility_manager.register_detector(detector)
        
        # Test detection
        results = await compatibility_manager.detect_all_frameworks()
        
        assert len(results) == 3
        for i in range(3):
            assert f"framework_{i}" in results
            assert results[f"framework_{i}"].status == FrameworkStatus.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, compatibility_manager):
        """Test framework configuration validation."""
        # Register mock detector
        mock_detector = MockFrameworkDetector("test_framework")
        compatibility_manager.register_detector(mock_detector)
        
        # Test valid configuration
        config = {"valid_param": "value"}
        is_valid, errors = await compatibility_manager.validate_framework_configuration(
            "test_framework", config
        )
        
        assert is_valid
        assert len(errors) == 0
    
    def test_compatibility_matrix(self, compatibility_manager):
        """Test getting compatibility matrix."""
        # Register mock detector
        mock_detector = MockFrameworkDetector("test_framework")
        compatibility_manager.register_detector(mock_detector)
        
        matrix = compatibility_manager.get_framework_matrix()
        
        assert "test_framework" in matrix
        assert "min_version" in matrix["test_framework"]
        assert "required_features" in matrix["test_framework"]


class TestAdaptiveSampling:
    """Test adaptive sampling functionality."""
    
    @pytest.mark.asyncio
    async def test_fixed_rate_sampling(self, sampling_manager):
        """Test fixed rate sampling strategy."""
        # Configure fixed rate sampling
        config = SamplingConfig(
            strategy=SamplingStrategy.FIXED_RATE,
            base_sampling_rate=0.5
        )
        sampling_manager.update_configuration(config)
        
        # Test sampling decisions
        sample_count = 0
        total_events = 100
        
        for i in range(total_events):
            event = AgentEvent(
                event_type=EventType.CUSTOM,
                agent_id="test_agent",
                session_id="test_session",
                message=f"Test event {i}"
            )
            
            decision = await sampling_manager.should_sample_event(event)
            if decision in [SamplingDecision.SAMPLE, SamplingDecision.PRIORITY_SAMPLE]:
                sample_count += 1
        
        # Should be approximately 50% sampling rate
        sampling_rate = sample_count / total_events
        assert 0.4 <= sampling_rate <= 0.6
    
    @pytest.mark.asyncio
    async def test_importance_based_sampling(self, sampling_manager):
        """Test importance-based sampling strategy."""
        # Configure importance-based sampling
        config = SamplingConfig(
            strategy=SamplingStrategy.IMPORTANCE_BASED,
            base_sampling_rate=0.5,
            high_priority_events=[EventType.AGENT_ERROR]
        )
        sampling_manager.update_configuration(config)
        
        # Test high priority event
        high_priority_event = AgentEvent(
            event_type=EventType.AGENT_ERROR,
            agent_id="test_agent",
            session_id="test_session",
            message="Error event"
        )
        
        decision = await sampling_manager.should_sample_event(high_priority_event)
        assert decision == SamplingDecision.PRIORITY_SAMPLE
        
        # Test low priority event
        low_priority_event = AgentEvent(
            event_type=EventType.CUSTOM,
            agent_id="test_agent",
            session_id="test_session",
            message="Custom event"
        )
        
        # Should have lower sampling probability
        sample_count = 0
        for _ in range(100):
            decision = await sampling_manager.should_sample_event(low_priority_event)
            if decision in [SamplingDecision.SAMPLE, SamplingDecision.PRIORITY_SAMPLE]:
                sample_count += 1
        
        # Should be less than 100% sampling
        assert sample_count < 100
    
    @pytest.mark.asyncio
    async def test_adaptive_rate_adjustment(self, sampling_manager):
        """Test adaptive rate adjustment based on performance."""
        # Configure adaptive sampling
        config = SamplingConfig(
            strategy=SamplingStrategy.ADAPTIVE_RATE,
            base_sampling_rate=1.0,
            max_overhead_ms=5.0,
            target_overhead_ms=2.0
        )
        sampling_manager.update_configuration(config)
        
        # Simulate high overhead
        sampling_manager.metrics.total_events = 100
        sampling_manager.metrics.sampled_events = 100
        sampling_manager.metrics.overhead_ms = 1000.0  # 10ms per event
        
        # Update metrics to trigger adaptation
        sampling_manager._sampler.update_metrics(sampling_manager.metrics)
        
        # Sampling rate should be reduced
        new_rate = sampling_manager._sampler.get_current_rate()
        assert new_rate < 1.0
    
    def test_sampling_statistics(self, sampling_manager):
        """Test sampling statistics collection."""
        # Simulate some events
        sampling_manager.metrics.total_events = 100
        sampling_manager.metrics.sampled_events = 50
        sampling_manager.metrics.skipped_events = 40
        sampling_manager.metrics.priority_events = 10
        
        stats = sampling_manager.get_sampling_statistics()
        
        assert stats["total_events"] == 100
        assert stats["sampled_events"] == 50
        assert stats["sampling_rate"] == 0.5
        assert "strategy" in stats


class TestRobustInstrumentor:
    """Test robust instrumentor functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_initialization(self, mock_instrumentor):
        """Test successful instrumentor initialization."""
        # Mock framework compatibility
        with patch.object(mock_instrumentor.compatibility_manager, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo(
                name="test_framework",
                status=FrameworkStatus.AVAILABLE
            )
            
            success = await mock_instrumentor.initialize()
            
            assert success
            assert mock_instrumentor.health.state == InstrumentorState.HEALTHY
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, mock_instrumentor):
        """Test instrumentor initialization failure handling."""
        # Make initialization fail
        mock_instrumentor.initialization_success = False
        
        # Mock framework compatibility
        with patch.object(mock_instrumentor.compatibility_manager, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo(
                name="test_framework",
                status=FrameworkStatus.AVAILABLE
            )
            
            success = await mock_instrumentor.initialize()
            
            assert not success
            assert mock_instrumentor.health.state == InstrumentorState.FAILED
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_instrumentor):
        """Test error handling and automatic recovery."""
        # Initialize instrumentor
        with patch.object(mock_instrumentor.compatibility_manager, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo(
                name="test_framework",
                status=FrameworkStatus.AVAILABLE
            )
            await mock_instrumentor.initialize()
        
        # Simulate errors
        mock_instrumentor.should_fail_capture = True
        
        # Create test event
        event = AgentEvent(
            event_type=EventType.CUSTOM,
            agent_id="test_agent",
            session_id="test_session",
            message="Test event"
        )
        
        # Capture events to trigger errors
        for _ in range(5):  # Exceed error threshold
            await mock_instrumentor.capture_event(event)
        
        # Should be in degraded or failed state
        assert mock_instrumentor.health.state in [InstrumentorState.DEGRADED, InstrumentorState.FAILED]
        assert mock_instrumentor.health.error_count >= 3
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_instrumentor):
        """Test graceful degradation functionality."""
        # Initialize instrumentor
        with patch.object(mock_instrumentor.compatibility_manager, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo(
                name="test_framework",
                status=FrameworkStatus.AVAILABLE
            )
            await mock_instrumentor.initialize()
        
        # Trigger graceful degradation
        success = await mock_instrumentor._graceful_degradation_recovery()
        
        assert success
        assert mock_instrumentor.health.degradation_level > 0.0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mock_instrumentor):
        """Test circuit breaker integration."""
        # Initialize instrumentor
        with patch.object(mock_instrumentor.compatibility_manager, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo(
                name="test_framework",
                status=FrameworkStatus.AVAILABLE
            )
            await mock_instrumentor.initialize()
        
        # Check circuit breaker is available
        assert mock_instrumentor.main_circuit_breaker is not None
        
        # Test circuit breaker status
        status = mock_instrumentor.main_circuit_breaker.get_status()
        assert "state" in status
        assert "metrics" in status
    
    @pytest.mark.asyncio
    async def test_adaptive_sampling_integration(self, mock_instrumentor):
        """Test adaptive sampling integration."""
        # Configure with adaptive sampling
        config = RobustConfig(adaptive_sampling=True)
        mock_instrumentor.robust_config = config
        
        # Initialize instrumentor
        with patch.object(mock_instrumentor.compatibility_manager, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo(
                name="test_framework",
                status=FrameworkStatus.AVAILABLE
            )
            await mock_instrumentor.initialize()
        
        # Create test event
        event = AgentEvent(
            event_type=EventType.CUSTOM,
            agent_id="test_agent",
            session_id="test_session",
            message="Test event"
        )
        
        # Capture event (should use sampling)
        await mock_instrumentor.capture_event(event)
        
        # Check sampling manager was used
        stats = mock_instrumentor.sampling_manager.get_sampling_statistics()
        assert stats["total_events"] > 0
    
    def test_health_status_reporting(self, mock_instrumentor):
        """Test health status reporting."""
        health_status = mock_instrumentor.get_health_status()
        
        assert "framework" in health_status
        assert "state" in health_status
        assert "is_healthy" in health_status
        assert "error_count" in health_status
        assert "circuit_breaker_state" in health_status
        assert "framework_compatibility" in health_status
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, mock_instrumentor):
        """Test complete monitoring lifecycle with robustness."""
        # Initialize instrumentor
        with patch.object(mock_instrumentor.compatibility_manager, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo(
                name="test_framework",
                status=FrameworkStatus.AVAILABLE
            )
            await mock_instrumentor.initialize()
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring("test_agent", {})
        assert session_id is not None
        assert len(mock_instrumentor.start_monitoring_calls) == 1
        
        # Capture events
        event = AgentEvent(
            event_type=EventType.CUSTOM,
            agent_id="test_agent",
            session_id=session_id,
            message="Test event"
        )
        
        await mock_instrumentor.capture_event(event)
        
        # Stop monitoring
        summary = await mock_instrumentor.stop_monitoring(session_id)
        assert summary is not None
        assert len(mock_instrumentor.stop_monitoring_calls) == 1
    
    @pytest.mark.asyncio
    async def test_recovery_strategies(self, mock_instrumentor):
        """Test different recovery strategies."""
        # Test immediate retry
        success = await mock_instrumentor._immediate_retry_recovery()
        # Should fail without proper setup, but shouldn't crash
        
        # Test exponential backoff
        start_time = time.time()
        success = await mock_instrumentor._exponential_backoff_recovery()
        elapsed = time.time() - start_time
        # Should have some delay
        assert elapsed >= 0.0
        
        # Test circuit breaker recovery
        success = await mock_instrumentor._circuit_breaker_recovery()
        # Should not crash
        
        # Test fallback mode recovery
        success = await mock_instrumentor._fallback_mode_recovery()
        assert success  # Fallback mode should always succeed


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_framework_unavailable_scenario(self):
        """Test scenario where framework is unavailable."""
        manager = FrameworkCompatibilityManager()
        
        # Register unavailable framework detector
        mock_detector = MockFrameworkDetector("unavailable_framework", FrameworkStatus.UNAVAILABLE)
        manager.register_detector(mock_detector)
        
        # Test detection
        framework_info = await manager.detect_framework("unavailable_framework")
        
        assert framework_info.status == FrameworkStatus.UNAVAILABLE
        
        # Test that instrumentor handles this gracefully
        config = RobustConfig()
        instrumentor = MockRobustInstrumentor("unavailable_framework", config=config)
        
        # Should fail to initialize
        success = await instrumentor.initialize()
        assert not success
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self):
        """Test behavior under high load conditions."""
        config = SamplingConfig(
            strategy=SamplingStrategy.PERFORMANCE_AWARE,
            base_sampling_rate=1.0
        )
        sampling_manager = AdaptiveSamplingManager(config)
        
        # Simulate high system load
        high_load_context = {"system_load": 0.9}
        
        # Create many events
        sample_count = 0
        total_events = 100
        
        for i in range(total_events):
            event = AgentEvent(
                event_type=EventType.CUSTOM,
                agent_id="test_agent",
                session_id="test_session",
                message=f"Event {i}"
            )
            
            decision = await sampling_manager.should_sample_event(event, high_load_context)
            if decision in [SamplingDecision.SAMPLE, SamplingDecision.PRIORITY_SAMPLE]:
                sample_count += 1
        
        # Should have reduced sampling under high load
        sampling_rate = sample_count / total_events
        assert sampling_rate < 1.0  # Should be less than 100%
    
    @pytest.mark.asyncio
    async def test_error_cascade_prevention(self, mock_instrumentor):
        """Test prevention of error cascades."""
        # Initialize instrumentor
        with patch.object(mock_instrumentor.compatibility_manager, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo(
                name="test_framework",
                status=FrameworkStatus.AVAILABLE
            )
            await mock_instrumentor.initialize()
        
        # Configure to fail operations
        mock_instrumentor.should_fail_capture = True
        
        # Generate many events to trigger circuit breaker
        events = []
        for i in range(10):
            event = AgentEvent(
                event_type=EventType.CUSTOM,
                agent_id="test_agent",
                session_id="test_session",
                message=f"Event {i}"
            )
            events.append(event)
        
        # Process events - should trigger circuit breaker
        for event in events:
            await mock_instrumentor.capture_event(event)
        
        # Circuit breaker should be open or instrumentor should be degraded
        cb_status = mock_instrumentor.main_circuit_breaker.get_status()
        assert (cb_status["state"] == "open" or 
                mock_instrumentor.health.state in [InstrumentorState.DEGRADED, InstrumentorState.FAILED])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])