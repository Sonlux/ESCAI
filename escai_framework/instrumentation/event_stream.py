"""
Thread-safe event streaming capabilities for the ESCAI framework.

This module provides real-time event streaming with buffering, filtering,
and multiple subscriber support for monitoring agent execution.
"""

import asyncio
import threading
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import weakref
from asyncio import Queue

from .events import AgentEvent, EventType, EventSeverity


class StreamingError(Exception):
    """Base exception for streaming errors."""
    pass


class SubscriberError(StreamingError):
    """Raised when subscriber operations fail."""
    pass


class BufferOverflowError(StreamingError):
    """Raised when event buffer overflows."""
    pass


@dataclass
class StreamFilter:
    """Filter for event streams."""
    event_types: Optional[Set[EventType]] = None
    agent_ids: Optional[Set[str]] = None
    session_ids: Optional[Set[str]] = None
    severity_levels: Optional[Set[EventSeverity]] = None
    frameworks: Optional[Set[str]] = None
    components: Optional[Set[str]] = None
    time_window: Optional[timedelta] = None
    custom_filter: Optional[Callable[[AgentEvent], bool]] = None
    
    def matches(self, event: AgentEvent) -> bool:
        """Check if an event matches this filter."""
        # Check event types
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Check agent IDs
        if self.agent_ids and event.agent_id not in self.agent_ids:
            return False
        
        # Check session IDs
        if self.session_ids and event.session_id not in self.session_ids:
            return False
        
        # Check severity levels
        if self.severity_levels and event.severity not in self.severity_levels:
            return False
        
        # Check frameworks
        if self.frameworks and event.framework not in self.frameworks:
            return False
        
        # Check components
        if self.components and event.component not in self.components:
            return False
        
        # Check time window
        if self.time_window:
            cutoff_time = datetime.utcnow() - self.time_window
            if event.timestamp < cutoff_time:
                return False
        
        # Check custom filter
        if self.custom_filter and not self.custom_filter(event):
            return False
        
        return True


@dataclass
class Subscriber:
    """Represents a subscriber to an event stream."""
    subscriber_id: str
    callback: Callable[[AgentEvent], None]
    filter: Optional[StreamFilter] = None
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_event_time: Optional[datetime] = None
    events_received: int = 0
    
    async def notify(self, event: AgentEvent) -> bool:
        """
        Notify subscriber of an event.
        
        Args:
            event: Event to send to subscriber
            
        Returns:
            True if notification successful, False otherwise
        """
        if not self.active:
            return False
        
        # Apply filter if configured
        if self.filter and not self.filter.matches(event):
            return True  # Filtered out, but not an error
        
        try:
            # Call the callback
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(event)
            else:
                # Run synchronous callback in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.callback, event)
            
            # Update statistics
            self.last_event_time = datetime.utcnow()
            self.events_received += 1
            
            return True
            
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Error notifying subscriber {self.subscriber_id}: {str(e)}"
            )
            return False
        return True


class EventBuffer:
    """Thread-safe circular buffer for events."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize event buffer.
        
        Args:
            max_size: Maximum number of events to buffer
        """
        self.max_size = max_size
        self._buffer: Deque[AgentEvent] = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._overflow_count = 0
    
    def add(self, event: AgentEvent) -> None:
        """
        Add an event to the buffer.
        
        Args:
            event: Event to add
        """
        with self._lock:
            if len(self._buffer) >= self.max_size:
                self._overflow_count += 1
            self._buffer.append(event)
    
    def get_events(self, count: Optional[int] = None, 
                  filter: Optional[StreamFilter] = None) -> List[AgentEvent]:
        """
        Get events from the buffer.
        
        Args:
            count: Maximum number of events to return
            filter: Optional filter to apply
            
        Returns:
            List of events
        """
        with self._lock:
            events = list(self._buffer)
        
        # Apply filter if provided
        if filter:
            events = [event for event in events if filter.matches(event)]
        
        # Limit count if specified
        if count is not None:
            events = events[-count:]
        
        return events
    
    def clear(self) -> None:
        """Clear all events from the buffer."""
        with self._lock:
            self._buffer.clear()
            self._overflow_count = 0
    
    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)
    
    def overflow_count(self) -> int:
        """Get number of events that were dropped due to overflow."""
        with self._lock:
            return self._overflow_count


class EventStream:
    """
    Thread-safe event streaming system with multiple subscribers,
    filtering, and buffering capabilities.
    """
    
    def __init__(self, buffer_size: int = 10000, max_workers: int = 4):
        """
        Initialize event stream.
        
        Args:
            buffer_size: Size of the event buffer
            max_workers: Maximum number of worker threads
        """
        self.buffer_size = buffer_size
        self.max_workers = max_workers
        
        # Event buffer
        self._buffer: EventBuffer = EventBuffer(buffer_size)
        
        # Subscribers
        self._subscribers: Dict[str, Subscriber] = {}
        self._subscribers_lock = threading.RLock()
        
        # Event processing
        self._event_queue: Queue[AgentEvent] = asyncio.Queue(maxsize=buffer_size)
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Thread pool for synchronous callbacks
        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="escai-event-stream"
        )
        
        # Metrics
        self._metrics = {
            "events_streamed": 0,
            "events_dropped": 0,
            "active_subscribers": 0,
            "notification_errors": 0,
            "average_processing_time_ms": 0.0
        }
        self._metrics_lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    async def start(self) -> None:
        """Start the event stream processing."""
        if self._processing_task is None or self._processing_task.done():
            self._shutdown_event.clear()
            self._processing_task = asyncio.create_task(self._process_events())
            self.logger.info("Event stream started")
    
    async def stop(self) -> None:
        """Stop the event stream processing."""
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for processing task to complete
        if self._processing_task and not self._processing_task.done():
            try:
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Processing task did not complete within timeout")
                self._processing_task.cancel()
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        self.logger.info("Event stream stopped")
    
    async def publish(self, event: AgentEvent) -> None:
        """
        Publish an event to the stream.
        
        Args:
            event: Event to publish
            
        Raises:
            StreamingError: If event cannot be published
        """
        if not event.validate():
            raise StreamingError(f"Invalid event: {event.event_id}")
        
        try:
            # Add to buffer
            self._buffer.add(event)
            
            # Queue for processing
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest event and add new one
                try:
                    self._event_queue.get_nowait()
                    self._event_queue.put_nowait(event)
                    with self._metrics_lock:
                        self._metrics["events_dropped"] += 1
                except asyncio.QueueEmpty:
                    pass
                    
        except Exception as e:
            raise StreamingError(f"Failed to publish event: {str(e)}")
    
    async def _process_events(self) -> None:
        """Process events from the queue and notify subscribers."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for event or timeout
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the event
                start_time = time.time()
                
                try:
                    await self._notify_subscribers(event)
                    
                    # Update metrics
                    processing_time = (time.time() - start_time) * 1000  # ms
                    with self._metrics_lock:
                        self._metrics["events_streamed"] += 1
                        
                        # Update rolling average
                        current_avg = self._metrics["average_processing_time_ms"]
                        count = self._metrics["events_streamed"]
                        self._metrics["average_processing_time_ms"] = (
                            (current_avg * (count - 1) + processing_time) / count
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error processing event: {str(e)}")
                    with self._metrics_lock:
                        self._metrics["notification_errors"] += 1
                
            except Exception as e:
                self.logger.error(f"Unexpected error in event processing: {str(e)}")
                await asyncio.sleep(0.1)  # Brief pause before retrying
    
    async def _notify_subscribers(self, event: AgentEvent) -> None:
        """Notify all active subscribers of an event."""
        # Get active subscribers
        with self._subscribers_lock:
            active_subscribers = [
                sub for sub in self._subscribers.values() 
                if sub.active
            ]
        
        if not active_subscribers:
            return
        
        # Notify subscribers concurrently
        tasks = []
        for subscriber in active_subscribers:
            task = asyncio.create_task(subscriber.notify(event))
            tasks.append(task)
        
        # Wait for all notifications to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count failures
        failures = sum(1 for result in results if isinstance(result, Exception))
        if failures > 0:
            with self._metrics_lock:
                self._metrics["notification_errors"] += failures
    
    def subscribe(self, callback: Callable[[AgentEvent], None], 
                 filter: Optional[StreamFilter] = None,
                 subscriber_id: Optional[str] = None) -> str:
        """
        Subscribe to the event stream.
        
        Args:
            callback: Function to call when events are received
            filter: Optional filter for events
            subscriber_id: Optional custom subscriber ID
            
        Returns:
            Subscriber ID
            
        Raises:
            SubscriberError: If subscription fails
        """
        if subscriber_id is None:
            import uuid
            subscriber_id = f"subscriber_{str(uuid.uuid4())[:8]}"
        
        subscriber = Subscriber(
            subscriber_id=subscriber_id,
            callback=callback,
            filter=filter
        )
        
        with self._subscribers_lock:
            if subscriber_id in self._subscribers:
                raise SubscriberError(f"Subscriber {subscriber_id} already exists")
            
            self._subscribers[subscriber_id] = subscriber
            
            # Update metrics
            with self._metrics_lock:
                self._metrics["active_subscribers"] = len(self._subscribers)
        
        self.logger.info(f"Subscriber {subscriber_id} added")
        return subscriber_id
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from the event stream.
        
        Args:
            subscriber_id: ID of subscriber to remove
            
        Returns:
            True if subscriber was removed, False if not found
        """
        with self._subscribers_lock:
            subscriber = self._subscribers.pop(subscriber_id, None)
            
            if subscriber:
                subscriber.active = False
                
                # Update metrics
                with self._metrics_lock:
                    self._metrics["active_subscribers"] = len(self._subscribers)
                
                self.logger.info(f"Subscriber {subscriber_id} removed")
                return True
        
        return False
    
    def get_subscriber(self, subscriber_id: str) -> Optional[Subscriber]:
        """
        Get subscriber information.
        
        Args:
            subscriber_id: ID of subscriber
            
        Returns:
            Subscriber object or None if not found
        """
        with self._subscribers_lock:
            return self._subscribers.get(subscriber_id)
    
    def list_subscribers(self) -> List[str]:
        """
        Get list of active subscriber IDs.
        
        Returns:
            List of subscriber IDs
        """
        with self._subscribers_lock:
            return list(self._subscribers.keys())
    
    def get_buffered_events(self, count: Optional[int] = None,
                           filter: Optional[StreamFilter] = None) -> List[AgentEvent]:
        """
        Get events from the buffer.
        
        Args:
            count: Maximum number of events to return
            filter: Optional filter to apply
            
        Returns:
            List of buffered events
        """
        return self._buffer.get_events(count, filter)
    
    def clear_buffer(self) -> None:
        """Clear the event buffer."""
        self._buffer.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get streaming metrics.
        
        Returns:
            Dictionary of metrics
        """
        with self._metrics_lock:
            metrics = self._metrics.copy()
        
        # Add buffer metrics
        metrics.update({
            "buffer_size": self._buffer.size(),
            "buffer_overflow_count": self._buffer.overflow_count(),
            "queue_size": self._event_queue.qsize()
        })
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset streaming metrics."""
        with self._metrics_lock:
            self._metrics = {
                "events_streamed": 0,
                "events_dropped": 0,
                "active_subscribers": len(self._subscribers),
                "notification_errors": 0,
                "average_processing_time_ms": 0.0
            }
        
        self._buffer.clear()


# Convenience functions for creating common filters

def create_agent_filter(agent_ids: Union[str, List[str]]) -> StreamFilter:
    """Create a filter for specific agent IDs."""
    if isinstance(agent_ids, str):
        agent_ids = [agent_ids]
    return StreamFilter(agent_ids=set(agent_ids))


def create_event_type_filter(event_types: Union[EventType, List[EventType]]) -> StreamFilter:
    """Create a filter for specific event types."""
    if isinstance(event_types, EventType):
        event_types = [event_types]
    return StreamFilter(event_types=set(event_types))


def create_severity_filter(severity_levels: Union[EventSeverity, List[EventSeverity]]) -> StreamFilter:
    """Create a filter for specific severity levels."""
    if isinstance(severity_levels, EventSeverity):
        severity_levels = [severity_levels]
    return StreamFilter(severity_levels=set(severity_levels))


def create_time_window_filter(window: timedelta) -> StreamFilter:
    """Create a filter for events within a time window."""
    return StreamFilter(time_window=window)


def create_framework_filter(frameworks: Union[str, List[str]]) -> StreamFilter:
    """Create a filter for specific frameworks."""
    if isinstance(frameworks, str):
        frameworks = [frameworks]
    return StreamFilter(frameworks=set(frameworks))