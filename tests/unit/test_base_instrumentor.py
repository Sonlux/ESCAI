"""
Unit tests for the base instrumentor functionality.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
import threading
import time

from escai_framework.instrumentation.base_instrumentor import (
    BaseInstrumentor, InstrumentationError, MonitoringOverheadError, EventProcessingError
)
from escai_framework.instrumentation.events import (
    AgentEvent, EventType, EventSeverity, MonitoringSession, MonitoringSummary
)


class TestInstrumentor(BaseInstrumentor):
    """Test implementation of BaseInstrumentor for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_monitoring_calls = []
        self.stop_monitoring_calls = []
        self.capture_event_calls = []
    
    async def start_monitoring(self, agent_id: str, config: Dict[str, Any]) -> str:
        self.start_monitoring_calls.append((agent_id, config))
        session = self._create_session(agent_id, config)
        return session.session_id
    
    async def stop_monitoring(self, session_id: str) -> MonitoringSummary:
        self.stop_monitoring_calls.append(session_id)
        session = self._end_session(session_id)
        if not session:
            raise InstrumentationError(f"Session {session_id} not found")
        
        return MonitoringSummary(
            session_id=session.session_id,
            agent_id=session.agent_id,
            framework=session.framework,
            start_time=session.start_time,
            end_time=session.end_time or datetime.utcnow(),
            total_duration_ms=1000,
            total_events=10
        )
    
    async def capture_event(self, event: AgentEvent) -> None:
        self.capture_event_calls.append(event)
        await self._queue_event(event)
    
    def get_supported_events(self) -> List[EventType]:
        return [EventType.AGENT_START, EventType.AGENT_STOP, EventType.TASK_START]
    
    def get_framework_name(self) -> str:
        return "test_framework"


class TestBaseInstrumentor:
    """Test cases for BaseInstrumentor."""
    
    @pytest.fixture
    def instrumentor(self):
        """Create a test instrumentor instance."""
        return TestInstrumentor()
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample agent event."""
        return AgentEvent(
            event_type=EventType.AGENT_START,
            agent_id="test_agent",
            session_id="test_session",
            message="Test event",
            framework="test_framework"
        )
    
    def test_initialization(self, instrumentor):
        """Test instrumentor initialization."""
        assert instrumentor.max_overhead_percent == 10.0
        assert instrumentor.max_events_per_second == 1000
        assert instrumentor.event_buffer_size == 10000
        assert instrumentor.get_framework_name() == "test_framework"
        assert len(instrumentor.get_supported_events()) == 3
    
    def test_initialization_with_custom_params(self):
        """Test instrumentor initialization with custom parameters."""
        instrumentor = TestInstrumentor(
            max_overhead_percent=5.0,
            max_events_per_second=500,
            event_buffer_size=5000
        )
        
        assert instrumentor.max_overhead_percent == 5.0
        assert instrumentor.max_events_per_second == 500
        assert instrumentor.event_buffer_size == 5000
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, instrumentor):
        """Test instrumentor start/stop lifecycle."""
        # Start instrumentor
        await instrumentor.start()
        assert instrumentor._processing_task is not None
        assert not instrumentor._processing_task.done()
        
        # Stop instrumentor
        await instrumentor.stop()
        assert instrumentor._processing_task.done()
    
    @pytest.mark.asyncio
    async def test_session_management(self, instrumentor):
        """Test monitoring session management."""
        await instrumentor.start()
        
        try:
            # Start monitoring
            session_id = await instrumentor.start_monitoring("test_agent", {"param": "value"})
            assert session_id is not None
            assert len(instrumentor.start_monitoring_calls) == 1
            
            # Check active sessions
            active_sessions = instrumentor.get_active_sessions()
            assert len(active_sessions) == 1
            assert active_sessions[0].agent_id == "test_agent"
            assert active_sessions[0].configuration == {"param": "value"}
            
            # Stop monitoring
            summary = await instrumentor.stop_monitoring(session_id)
            assert summary.session_id == session_id
            assert summary.agent_id == "test_agent"
            assert len(instrumentor.stop_monitoring_calls) == 1
            
            # Check no active sessions
            active_sessions = instrumentor.get_active_sessions()
            assert len(active_sessions) == 0
            
        finally:
            await instrumentor.stop()
    
    @pytest.mark.asyncio
    async def test_event_handling(self, instrumentor, sample_event):
        """Test event handling and processing."""
        await instrumentor.start()
        
        try:
            # Add event handler
            events_received = []
            
            def event_handler(event: AgentEvent):
                events_received.append(event)
            
            instrumentor.add_event_handler(event_handler)
            
            # Capture event
            await instrumentor.capture_event(sample_event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check event was processed
            assert len(events_received) == 1
            assert events_received[0].event_id == sample_event.event_id
            
            # Remove handler
            instrumentor.remove_event_handler(event_handler)
            
            # Capture another event
            sample_event.event_id = "new_event_id"
            await instrumentor.capture_event(sample_event)
            await asyncio.sleep(0.1)
            
            # Should still be only 1 event (handler was removed)
            assert len(events_received) == 1
            
        finally:
            await instrumentor.stop()
    
    @pytest.mark.asyncio
    async def test_async_event_handler(self, instrumentor, sample_event):
        """Test async event handler."""
        await instrumentor.start()
        
        try:
            events_received = []
            
            async def async_event_handler(event: AgentEvent):
                await asyncio.sleep(0.01)  # Simulate async work
                events_received.append(event)
            
            instrumentor.add_event_handler(async_event_handler)
            
            # Capture event
            await instrumentor.capture_event(sample_event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check event was processed
            assert len(events_received) == 1
            assert events_received[0].event_id == sample_event.event_id
            
        finally:
            await instrumentor.stop()
    
    @pytest.mark.asyncio
    async def test_event_validation(self, instrumentor):
        """Test event validation."""
        await instrumentor.start()
        
        try:
            # Create invalid event
            invalid_event = AgentEvent(
                event_type=EventType.AGENT_START,
                agent_id="",  # Invalid empty agent_id
                session_id="test_session",
                message="Test event"
            )
            
            # Should raise EventProcessingError
            with pytest.raises(EventProcessingError):
                await instrumentor.capture_event(invalid_event)
                
        finally:
            await instrumentor.stop()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, instrumentor, sample_event):
        """Test circuit breaker functionality."""
        await instrumentor.start()
        
        try:
            # Add failing event handler
            def failing_handler(event: AgentEvent):
                raise Exception("Handler failure")
            
            instrumentor.add_event_handler(failing_handler)
            
            # Trigger multiple failures
            for i in range(6):  # Exceed threshold of 5
                await instrumentor.capture_event(sample_event)
                await asyncio.sleep(0.01)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Circuit breaker should be open
            assert instrumentor._circuit_breaker_open
            
            # New events should be dropped
            initial_dropped = instrumentor.get_performance_metrics()["events_dropped"]
            await instrumentor.capture_event(sample_event)
            await asyncio.sleep(0.01)
            
            final_dropped = instrumentor.get_performance_metrics()["events_dropped"]
            assert final_dropped > initial_dropped
            
        finally:
            await instrumentor.stop()
    
    def test_performance_metrics(self, instrumentor):
        """Test performance metrics tracking."""
        metrics = instrumentor.get_performance_metrics()
        
        expected_keys = [
            "events_processed",
            "events_dropped", 
            "processing_errors",
            "average_processing_time_ms",
            "overhead_percentage"
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
    
    def test_overhead_checking(self, instrumentor):
        """Test monitoring overhead checking."""
        # Normal overhead should not raise exception
        instrumentor._check_overhead(1.0, 0.05)  # 5% overhead
        
        # Excessive overhead should raise exception
        with pytest.raises(MonitoringOverheadError):
            instrumentor._check_overhead(1.0, 0.15)  # 15% overhead
    
    def test_create_event_utility(self, instrumentor):
        """Test event creation utility method."""
        event = instrumentor.create_event(
            event_type=EventType.TASK_START,
            agent_id="test_agent",
            session_id="test_session",
            message="Task started",
            component="test_component",
            operation="execute"
        )
        
        assert event.event_type == EventType.TASK_START
        assert event.agent_id == "test_agent"
        assert event.session_id == "test_session"
        assert event.message == "Task started"
        assert event.framework == "test_framework"
        assert event.component == "test_component"
        assert event.operation == "execute"
        assert event.validate()
    
    @pytest.mark.asyncio
    async def test_multiple_sessions(self, instrumentor):
        """Test handling multiple concurrent sessions."""
        await instrumentor.start()
        
        try:
            # Start multiple sessions
            session_ids = []
            for i in range(3):
                session_id = await instrumentor.start_monitoring(
                    f"agent_{i}", 
                    {"config": f"value_{i}"}
                )
                session_ids.append(session_id)
            
            # Check all sessions are active
            active_sessions = instrumentor.get_active_sessions()
            assert len(active_sessions) == 3
            
            # Stop sessions one by one
            for session_id in session_ids:
                summary = await instrumentor.stop_monitoring(session_id)
                assert summary.session_id == session_id
            
            # Check no active sessions
            active_sessions = instrumentor.get_active_sessions()
            assert len(active_sessions) == 0
            
        finally:
            await instrumentor.stop()
    
    @pytest.mark.asyncio
    async def test_event_queue_overflow(self, instrumentor):
        """Test event queue overflow handling."""
        # Create instrumentor with small buffer
        small_instrumentor = TestInstrumentor(event_buffer_size=2)
        await small_instrumentor.start()
        
        try:
            # Fill the queue beyond capacity
            events = []
            for i in range(5):
                event = AgentEvent(
                    event_type=EventType.AGENT_START,
                    agent_id=f"agent_{i}",
                    session_id="test_session",
                    message=f"Event {i}"
                )
                events.append(event)
                await small_instrumentor.capture_event(event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Some events should have been dropped
            metrics = small_instrumentor.get_performance_metrics()
            assert metrics["events_dropped"] > 0
            
        finally:
            await small_instrumentor.stop()
    
    def test_context_manager(self, instrumentor):
        """Test context manager interface."""
        with instrumentor as inst:
            assert inst is instrumentor
        # Context manager exit should not raise exceptions
    
    @pytest.mark.asyncio
    async def test_stop_nonexistent_session(self, instrumentor):
        """Test stopping a non-existent session."""
        with pytest.raises(InstrumentationError):
            await instrumentor.stop_monitoring("nonexistent_session")
    
    @pytest.mark.asyncio
    async def test_thread_safety(self, instrumentor):
        """Test thread safety of instrumentor operations."""
        await instrumentor.start()
        
        try:
            events_received = []
            
            def event_handler(event: AgentEvent):
                events_received.append(event)
            
            instrumentor.add_event_handler(event_handler)
            
            # Create events from multiple threads
            def create_events(thread_id: int):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def async_create():
                    for i in range(10):
                        event = AgentEvent(
                            event_type=EventType.AGENT_START,
                            agent_id=f"agent_{thread_id}_{i}",
                            session_id="test_session",
                            message=f"Event from thread {thread_id}"
                        )
                        await instrumentor.capture_event(event)
                
                loop.run_until_complete(async_create())
                loop.close()
            
            # Start multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=create_events, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Should have received events from all threads
            assert len(events_received) == 30  # 3 threads * 10 events each
            
        finally:
            await instrumentor.stop()


if __name__ == "__main__":
    pytest.main([__file__])