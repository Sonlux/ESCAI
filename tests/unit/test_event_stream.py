"""
Unit tests for the event streaming functionality.
"""

import asyncio
import pytest
import threading
import time
from datetime import datetime, timedelta
from typing import List

from escai_framework.instrumentation.event_stream import (
    EventStream, StreamFilter, Subscriber, EventBuffer,
    StreamingError, SubscriberError, BufferOverflowError,
    create_agent_filter, create_event_type_filter, create_severity_filter,
    create_time_window_filter, create_framework_filter
)
from escai_framework.instrumentation.events import AgentEvent, EventType, EventSeverity


class TestStreamFilter:
    """Test cases for StreamFilter."""
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample agent event."""
        return AgentEvent(
            event_type=EventType.TASK_START,
            agent_id="test_agent",
            session_id="test_session",
            severity=EventSeverity.INFO,
            framework="test_framework",
            component="test_component",
            message="Test event"
        )
    
    def test_filter_no_criteria(self, sample_event):
        """Test filter with no criteria (should match everything)."""
        filter = StreamFilter()
        assert filter.matches(sample_event) is True
    
    def test_filter_event_types(self, sample_event):
        """Test filtering by event types."""
        filter = StreamFilter(event_types={EventType.TASK_START, EventType.TASK_COMPLETE})
        assert filter.matches(sample_event) is True
        
        filter = StreamFilter(event_types={EventType.AGENT_START, EventType.AGENT_STOP})
        assert filter.matches(sample_event) is False
    
    def test_filter_agent_ids(self, sample_event):
        """Test filtering by agent IDs."""
        filter = StreamFilter(agent_ids={"test_agent", "other_agent"})
        assert filter.matches(sample_event) is True
        
        filter = StreamFilter(agent_ids={"different_agent"})
        assert filter.matches(sample_event) is False
    
    def test_filter_session_ids(self, sample_event):
        """Test filtering by session IDs."""
        filter = StreamFilter(session_ids={"test_session"})
        assert filter.matches(sample_event) is True
        
        filter = StreamFilter(session_ids={"different_session"})
        assert filter.matches(sample_event) is False
    
    def test_filter_severity_levels(self, sample_event):
        """Test filtering by severity levels."""
        filter = StreamFilter(severity_levels={EventSeverity.INFO, EventSeverity.WARNING})
        assert filter.matches(sample_event) is True
        
        filter = StreamFilter(severity_levels={EventSeverity.ERROR, EventSeverity.CRITICAL})
        assert filter.matches(sample_event) is False
    
    def test_filter_frameworks(self, sample_event):
        """Test filtering by frameworks."""
        filter = StreamFilter(frameworks={"test_framework"})
        assert filter.matches(sample_event) is True
        
        filter = StreamFilter(frameworks={"other_framework"})
        assert filter.matches(sample_event) is False
    
    def test_filter_components(self, sample_event):
        """Test filtering by components."""
        filter = StreamFilter(components={"test_component"})
        assert filter.matches(sample_event) is True
        
        filter = StreamFilter(components={"other_component"})
        assert filter.matches(sample_event) is False
    
    def test_filter_time_window(self, sample_event):
        """Test filtering by time window."""
        # Event should be recent, so should match
        filter = StreamFilter(time_window=timedelta(minutes=1))
        assert filter.matches(sample_event) is True
        
        # Make event old
        sample_event.timestamp = datetime.utcnow() - timedelta(hours=1)
        filter = StreamFilter(time_window=timedelta(minutes=30))
        assert filter.matches(sample_event) is False
    
    def test_filter_custom_function(self, sample_event):
        """Test filtering with custom function."""
        def custom_filter(event: AgentEvent) -> bool:
            return "test" in event.message.lower()
        
        filter = StreamFilter(custom_filter=custom_filter)
        assert filter.matches(sample_event) is True
        
        sample_event.message = "No match here"
        assert filter.matches(sample_event) is False
    
    def test_filter_multiple_criteria(self, sample_event):
        """Test filtering with multiple criteria (AND logic)."""
        filter = StreamFilter(
            event_types={EventType.TASK_START},
            agent_ids={"test_agent"},
            severity_levels={EventSeverity.INFO}
        )
        assert filter.matches(sample_event) is True
        
        # Change one criterion to not match
        sample_event.severity = EventSeverity.ERROR
        assert filter.matches(sample_event) is False


class TestEventBuffer:
    """Test cases for EventBuffer."""
    
    @pytest.fixture
    def buffer(self):
        """Create an event buffer."""
        return EventBuffer(max_size=5)
    
    @pytest.fixture
    def sample_events(self):
        """Create sample events."""
        events = []
        for i in range(10):
            event = AgentEvent(
                event_type=EventType.TASK_START,
                agent_id=f"agent_{i}",
                session_id="test_session",
                message=f"Event {i}"
            )
            events.append(event)
        return events
    
    def test_buffer_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.max_size == 5
        assert buffer.size() == 0
        assert buffer.overflow_count() == 0
    
    def test_buffer_add_events(self, buffer, sample_events):
        """Test adding events to buffer."""
        # Add events within capacity
        for i in range(3):
            buffer.add(sample_events[i])
        
        assert buffer.size() == 3
        assert buffer.overflow_count() == 0
        
        # Add events beyond capacity
        for i in range(3, 8):
            buffer.add(sample_events[i])
        
        assert buffer.size() == 5  # Should be capped at max_size
        assert buffer.overflow_count() == 3  # 3 events caused overflow
    
    def test_buffer_get_events(self, buffer, sample_events):
        """Test getting events from buffer."""
        # Add some events
        for i in range(3):
            buffer.add(sample_events[i])
        
        # Get all events
        events = buffer.get_events()
        assert len(events) == 3
        
        # Get limited count
        events = buffer.get_events(count=2)
        assert len(events) == 2
        
        # Get with filter
        filter = StreamFilter(agent_ids={"agent_1"})
        events = buffer.get_events(filter=filter)
        assert len(events) == 1
        assert events[0].agent_id == "agent_1"
    
    def test_buffer_clear(self, buffer, sample_events):
        """Test clearing buffer."""
        # Add events
        for i in range(3):
            buffer.add(sample_events[i])
        
        assert buffer.size() == 3
        
        # Clear buffer
        buffer.clear()
        assert buffer.size() == 0
        assert buffer.overflow_count() == 0
    
    def test_buffer_thread_safety(self, buffer, sample_events):
        """Test buffer thread safety."""
        def add_events(start_idx: int, count: int):
            for i in range(start_idx, start_idx + count):
                if i < len(sample_events):
                    buffer.add(sample_events[i])
        
        # Start multiple threads adding events
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_events, args=(i * 2, 2))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Buffer should have events (exact count depends on timing)
        assert buffer.size() > 0


class TestSubscriber:
    """Test cases for Subscriber."""
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample event."""
        return AgentEvent(
            event_type=EventType.TASK_START,
            agent_id="test_agent",
            session_id="test_session",
            message="Test event"
        )
    
    @pytest.mark.asyncio
    async def test_subscriber_creation(self):
        """Test subscriber creation."""
        events_received = []
        
        def callback(event: AgentEvent):
            events_received.append(event)
        
        subscriber = Subscriber(
            subscriber_id="test_subscriber",
            callback=callback
        )
        
        assert subscriber.subscriber_id == "test_subscriber"
        assert subscriber.callback == callback
        assert subscriber.active is True
        assert subscriber.events_received == 0
        assert subscriber.last_event_time is None
    
    @pytest.mark.asyncio
    async def test_subscriber_notify_sync_callback(self, sample_event):
        """Test subscriber notification with sync callback."""
        events_received = []
        
        def sync_callback(event: AgentEvent):
            events_received.append(event)
        
        subscriber = Subscriber(
            subscriber_id="test_subscriber",
            callback=sync_callback
        )
        
        result = await subscriber.notify(sample_event)
        
        assert result is True
        assert len(events_received) == 1
        assert events_received[0] == sample_event
        assert subscriber.events_received == 1
        assert subscriber.last_event_time is not None
    
    @pytest.mark.asyncio
    async def test_subscriber_notify_async_callback(self, sample_event):
        """Test subscriber notification with async callback."""
        events_received = []
        
        async def async_callback(event: AgentEvent):
            await asyncio.sleep(0.01)  # Simulate async work
            events_received.append(event)
        
        subscriber = Subscriber(
            subscriber_id="test_subscriber",
            callback=async_callback
        )
        
        result = await subscriber.notify(sample_event)
        
        assert result is True
        assert len(events_received) == 1
        assert events_received[0] == sample_event
    
    @pytest.mark.asyncio
    async def test_subscriber_notify_with_filter(self, sample_event):
        """Test subscriber notification with filter."""
        events_received = []
        
        def callback(event: AgentEvent):
            events_received.append(event)
        
        # Filter that should match
        matching_filter = StreamFilter(event_types={EventType.TASK_START})
        subscriber = Subscriber(
            subscriber_id="test_subscriber",
            callback=callback,
            filter=matching_filter
        )
        
        result = await subscriber.notify(sample_event)
        assert result is True
        assert len(events_received) == 1
        
        # Filter that should not match
        events_received.clear()
        non_matching_filter = StreamFilter(event_types={EventType.AGENT_START})
        subscriber.filter = non_matching_filter
        
        result = await subscriber.notify(sample_event)
        assert result is True  # Not an error, just filtered
        assert len(events_received) == 0
    
    @pytest.mark.asyncio
    async def test_subscriber_notify_inactive(self, sample_event):
        """Test notification to inactive subscriber."""
        events_received = []
        
        def callback(event: AgentEvent):
            events_received.append(event)
        
        subscriber = Subscriber(
            subscriber_id="test_subscriber",
            callback=callback,
            active=False
        )
        
        result = await subscriber.notify(sample_event)
        
        assert result is False
        assert len(events_received) == 0
    
    @pytest.mark.asyncio
    async def test_subscriber_notify_callback_error(self, sample_event):
        """Test notification with callback that raises error."""
        def failing_callback(event: AgentEvent):
            raise Exception("Callback error")
        
        subscriber = Subscriber(
            subscriber_id="test_subscriber",
            callback=failing_callback
        )
        
        result = await subscriber.notify(sample_event)
        
        assert result is False  # Should return False on error


class TestEventStream:
    """Test cases for EventStream."""
    
    @pytest.fixture
    def stream(self):
        """Create an event stream."""
        return EventStream(buffer_size=100, max_workers=2)
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample event."""
        return AgentEvent(
            event_type=EventType.TASK_START,
            agent_id="test_agent",
            session_id="test_session",
            message="Test event"
        )
    
    def test_stream_initialization(self, stream):
        """Test stream initialization."""
        assert stream.buffer_size == 100
        assert stream.max_workers == 2
        assert len(stream.list_subscribers()) == 0
    
    @pytest.mark.asyncio
    async def test_stream_lifecycle(self, stream):
        """Test stream start/stop lifecycle."""
        await stream.start()
        assert stream._processing_task is not None
        assert not stream._processing_task.done()
        
        await stream.stop()
        assert stream._processing_task.done()
    
    @pytest.mark.asyncio
    async def test_stream_publish_and_subscribe(self, stream, sample_event):
        """Test publishing events and subscribing to them."""
        await stream.start()
        
        try:
            events_received = []
            
            def callback(event: AgentEvent):
                events_received.append(event)
            
            # Subscribe
            subscriber_id = stream.subscribe(callback)
            assert subscriber_id is not None
            assert len(stream.list_subscribers()) == 1
            
            # Publish event
            await stream.publish(sample_event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check event was received
            assert len(events_received) == 1
            assert events_received[0].event_id == sample_event.event_id
            
        finally:
            await stream.stop()
    
    @pytest.mark.asyncio
    async def test_stream_multiple_subscribers(self, stream, sample_event):
        """Test multiple subscribers receiving events."""
        await stream.start()
        
        try:
            events_received_1 = []
            events_received_2 = []
            
            def callback_1(event: AgentEvent):
                events_received_1.append(event)
            
            def callback_2(event: AgentEvent):
                events_received_2.append(event)
            
            # Subscribe multiple callbacks
            subscriber_id_1 = stream.subscribe(callback_1)
            subscriber_id_2 = stream.subscribe(callback_2)
            
            assert len(stream.list_subscribers()) == 2
            
            # Publish event
            await stream.publish(sample_event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Both subscribers should receive the event
            assert len(events_received_1) == 1
            assert len(events_received_2) == 1
            
        finally:
            await stream.stop()
    
    @pytest.mark.asyncio
    async def test_stream_subscribe_with_filter(self, stream):
        """Test subscribing with event filter."""
        await stream.start()
        
        try:
            events_received = []
            
            def callback(event: AgentEvent):
                events_received.append(event)
            
            # Subscribe with filter for specific event type
            filter = StreamFilter(event_types={EventType.TASK_START})
            subscriber_id = stream.subscribe(callback, filter=filter)
            
            # Publish matching event
            matching_event = AgentEvent(
                event_type=EventType.TASK_START,
                agent_id="test_agent",
                session_id="test_session",
                message="Matching event"
            )
            await stream.publish(matching_event)
            
            # Publish non-matching event
            non_matching_event = AgentEvent(
                event_type=EventType.AGENT_START,
                agent_id="test_agent",
                session_id="test_session",
                message="Non-matching event"
            )
            await stream.publish(non_matching_event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Should only receive matching event
            assert len(events_received) == 1
            assert events_received[0].event_type == EventType.TASK_START
            
        finally:
            await stream.stop()
    
    @pytest.mark.asyncio
    async def test_stream_unsubscribe(self, stream, sample_event):
        """Test unsubscribing from stream."""
        await stream.start()
        
        try:
            events_received = []
            
            def callback(event: AgentEvent):
                events_received.append(event)
            
            # Subscribe
            subscriber_id = stream.subscribe(callback)
            assert len(stream.list_subscribers()) == 1
            
            # Publish event
            await stream.publish(sample_event)
            await asyncio.sleep(0.1)
            
            assert len(events_received) == 1
            
            # Unsubscribe
            result = stream.unsubscribe(subscriber_id)
            assert result is True
            assert len(stream.list_subscribers()) == 0
            
            # Publish another event
            await stream.publish(sample_event)
            await asyncio.sleep(0.1)
            
            # Should still be only 1 event (subscriber was removed)
            assert len(events_received) == 1
            
        finally:
            await stream.stop()
    
    @pytest.mark.asyncio
    async def test_stream_invalid_event(self, stream):
        """Test publishing invalid event."""
        await stream.start()
        
        try:
            invalid_event = AgentEvent(
                event_type=EventType.TASK_START,
                agent_id="",  # Invalid empty agent_id
                session_id="test_session"
            )
            
            with pytest.raises(StreamingError):
                await stream.publish(invalid_event)
                
        finally:
            await stream.stop()
    
    def test_stream_get_buffered_events(self, stream):
        """Test getting buffered events."""
        # Add events to buffer directly (without starting stream)
        events = []
        for i in range(5):
            event = AgentEvent(
                event_type=EventType.TASK_START,
                agent_id=f"agent_{i}",
                session_id="test_session",
                message=f"Event {i}"
            )
            events.append(event)
            stream._buffer.add(event)
        
        # Get all events
        buffered_events = stream.get_buffered_events()
        assert len(buffered_events) == 5
        
        # Get limited count
        buffered_events = stream.get_buffered_events(count=3)
        assert len(buffered_events) == 3
        
        # Get with filter
        filter = StreamFilter(agent_ids={"agent_1"})
        buffered_events = stream.get_buffered_events(filter=filter)
        assert len(buffered_events) == 1
        assert buffered_events[0].agent_id == "agent_1"
    
    def test_stream_metrics(self, stream):
        """Test stream metrics."""
        metrics = stream.get_metrics()
        
        expected_keys = [
            "events_streamed",
            "events_dropped",
            "active_subscribers",
            "notification_errors",
            "average_processing_time_ms",
            "buffer_size",
            "buffer_overflow_count",
            "queue_size"
        ]
        
        for key in expected_keys:
            assert key in metrics
    
    def test_stream_clear_buffer(self, stream):
        """Test clearing stream buffer."""
        # Add event to buffer
        event = AgentEvent(
            event_type=EventType.TASK_START,
            agent_id="test_agent",
            session_id="test_session"
        )
        stream._buffer.add(event)
        
        assert stream._buffer.size() == 1
        
        stream.clear_buffer()
        assert stream._buffer.size() == 0
    
    def test_stream_duplicate_subscriber_id(self, stream):
        """Test subscribing with duplicate subscriber ID."""
        def callback(event: AgentEvent):
            pass
        
        # Subscribe with custom ID
        subscriber_id = "custom_id"
        stream.subscribe(callback, subscriber_id=subscriber_id)
        
        # Try to subscribe again with same ID
        with pytest.raises(SubscriberError):
            stream.subscribe(callback, subscriber_id=subscriber_id)


class TestFilterHelpers:
    """Test cases for filter helper functions."""
    
    def test_create_agent_filter(self):
        """Test creating agent filter."""
        # Single agent
        filter = create_agent_filter("test_agent")
        assert filter.agent_ids == {"test_agent"}
        
        # Multiple agents
        filter = create_agent_filter(["agent1", "agent2"])
        assert filter.agent_ids == {"agent1", "agent2"}
    
    def test_create_event_type_filter(self):
        """Test creating event type filter."""
        # Single event type
        filter = create_event_type_filter(EventType.TASK_START)
        assert filter.event_types == {EventType.TASK_START}
        
        # Multiple event types
        filter = create_event_type_filter([EventType.TASK_START, EventType.TASK_COMPLETE])
        assert filter.event_types == {EventType.TASK_START, EventType.TASK_COMPLETE}
    
    def test_create_severity_filter(self):
        """Test creating severity filter."""
        # Single severity
        filter = create_severity_filter(EventSeverity.ERROR)
        assert filter.severity_levels == {EventSeverity.ERROR}
        
        # Multiple severities
        filter = create_severity_filter([EventSeverity.ERROR, EventSeverity.CRITICAL])
        assert filter.severity_levels == {EventSeverity.ERROR, EventSeverity.CRITICAL}
    
    def test_create_time_window_filter(self):
        """Test creating time window filter."""
        window = timedelta(minutes=30)
        filter = create_time_window_filter(window)
        assert filter.time_window == window
    
    def test_create_framework_filter(self):
        """Test creating framework filter."""
        # Single framework
        filter = create_framework_filter("langchain")
        assert filter.frameworks == {"langchain"}
        
        # Multiple frameworks
        filter = create_framework_filter(["langchain", "autogen"])
        assert filter.frameworks == {"langchain", "autogen"}


if __name__ == "__main__":
    pytest.main([__file__])