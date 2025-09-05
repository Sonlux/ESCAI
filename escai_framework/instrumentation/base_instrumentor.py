"""
Base instrumentor abstract class for the ESCAI framework.

This module defines the abstract interface that all framework-specific
instrumentors must implement to ensure consistent monitoring capabilities.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import traceback
from asyncio import Queue

from .events import (
    AgentEvent, EventType, EventSeverity, MonitoringSession
)


class InstrumentationError(Exception):
    """Base exception for instrumentation errors."""
    pass


class MonitoringOverheadError(InstrumentationError):
    """Raised when monitoring overhead exceeds acceptable limits."""
    pass


class EventProcessingError(InstrumentationError):
    """Raised when event processing fails."""
    pass


@dataclass
class MonitoringConfig:
    """Configuration for monitoring sessions."""
    agent_id: str
    framework: str
    capture_epistemic_states: bool = True
    capture_behavioral_patterns: bool = True
    capture_performance_metrics: bool = True
    max_events_per_second: int = 100
    buffer_size: int = 1000
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "framework": self.framework,
            "capture_epistemic_states": self.capture_epistemic_states,
            "capture_behavioral_patterns": self.capture_behavioral_patterns,
            "capture_performance_metrics": self.capture_performance_metrics,
            "max_events_per_second": self.max_events_per_second,
            "buffer_size": self.buffer_size
        }


@dataclass
class MonitoringSummary:
    """Summary of monitoring session."""
    session_id: str
    agent_id: str
    framework: str
    start_time: datetime
    end_time: datetime
    total_duration_ms: int
    total_events: int
    event_types_count: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "framework": self.framework,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_duration_ms": self.total_duration_ms,
            "total_events": self.total_events,
            "event_types_count": self.event_types_count,
            "error_count": self.error_count,
            "performance_metrics": self.performance_metrics
        }


class BaseInstrumentor(ABC):
    """
    Abstract base class for framework-specific instrumentors.
    
    This class defines the core interface that all instrumentors must implement
    to provide consistent monitoring capabilities across different agent frameworks.
    """
    
    def __init__(self, max_overhead_percent: float = 10.0, 
                 max_events_per_second: int = 1000,
                 event_buffer_size: int = 10000):
        """
        Initialize the base instrumentor.
        
        Args:
            max_overhead_percent: Maximum acceptable monitoring overhead (default 10%)
            max_events_per_second: Maximum events to process per second
            event_buffer_size: Size of the event buffer for batching
        """
        self.max_overhead_percent = max_overhead_percent
        self.max_events_per_second = max_events_per_second
        self.event_buffer_size = event_buffer_size
        
        # Thread-safe event handling
        self._event_queue: Queue[AgentEvent] = asyncio.Queue(maxsize=event_buffer_size)
        self._event_handlers: List[Callable[[AgentEvent], Awaitable[None]] | Callable[[AgentEvent], None]] = []
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Session management
        self._active_sessions: Dict[str, MonitoringSession] = {}
        self._session_lock = threading.RLock()
        
        # Performance monitoring
        self._performance_metrics = {
            "events_processed": 0,
            "events_dropped": 0,
            "processing_errors": 0,
            "average_processing_time_ms": 0.0,
            "overhead_percentage": 0.0
        }
        self._metrics_lock = threading.RLock()
        
        # Circuit breaker for overhead protection
        self._circuit_breaker_open = False
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure: Optional[float] = None
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60  # seconds
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Thread pool for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="escai-instrumentor")
    
    @abstractmethod
    async def start_monitoring(self, agent_id: str, config: Dict[str, Any]) -> str:
        """
        Start monitoring an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration parameters for monitoring
            
        Returns:
            session_id: Unique identifier for the monitoring session
            
        Raises:
            InstrumentationError: If monitoring cannot be started
        """
        pass
    
    @abstractmethod
    async def stop_monitoring(self, session_id: str) -> MonitoringSummary:
        """
        Stop monitoring a session.
        
        Args:
            session_id: Identifier of the session to stop
            
        Returns:
            MonitoringSummary: Summary of the monitoring session
            
        Raises:
            InstrumentationError: If session cannot be stopped
        """
        pass
    
    @abstractmethod
    async def capture_event(self, event: AgentEvent) -> None:
        """
        Capture an agent event.
        
        Args:
            event: The agent event to capture
            
        Raises:
            EventProcessingError: If event cannot be processed
        """
        pass
    
    @abstractmethod
    def get_supported_events(self) -> List[EventType]:
        """
        Get the list of event types supported by this instrumentor.
        
        Returns:
            List of supported EventType values
        """
        pass
    
    @abstractmethod
    def get_framework_name(self) -> str:
        """
        Get the name of the framework this instrumentor supports.
        
        Returns:
            Framework name (e.g., "langchain", "autogen")
        """
        pass
    
    async def get_monitoring_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get monitoring statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary containing monitoring statistics
        """
        # Default implementation returns basic stats
        return {
            "events_captured": 0,
            "performance_overhead": 0.0,
            "error_count": 0,
            "last_activity": datetime.utcnow()
        }
    
    # Event handling methods
    
    def add_event_handler(self, handler: Callable[[AgentEvent], None]) -> None:
        """
        Add an event handler for processing captured events.
        
        Args:
            handler: Function to call when events are captured
        """
        if handler not in self._event_handlers:
            self._event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[AgentEvent], None]) -> None:
        """
        Remove an event handler.
        
        Args:
            handler: Handler function to remove
        """
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
    
    async def _queue_event(self, event: AgentEvent) -> None:
        """
        Queue an event for processing.
        
        Args:
            event: Event to queue
        """
        if self._circuit_breaker_open:
            if self._should_reset_circuit_breaker():
                self._reset_circuit_breaker()
            else:
                with self._metrics_lock:
                    self._performance_metrics["events_dropped"] += 1
                return
        
        try:
            # Validate event before queuing
            if not event.validate():
                raise EventProcessingError(f"Invalid event: {event.event_id}")
            
            # Try to queue the event (non-blocking)
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest event and add new one
                try:
                    self._event_queue.get_nowait()
                    self._event_queue.put_nowait(event)
                    with self._metrics_lock:
                        self._performance_metrics["events_dropped"] += 1
                except asyncio.QueueEmpty:
                    pass
                    
        except Exception as e:
            self._handle_circuit_breaker_failure()
            raise EventProcessingError(f"Failed to queue event: {str(e)}")
    
    async def _process_events(self) -> None:
        """
        Process events from the queue in a separate task.
        """
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
                    # Call all registered handlers
                    for handler in self._event_handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(event)
                            else:
                                # Run synchronous handlers in thread pool
                                await asyncio.get_event_loop().run_in_executor(
                                    self._thread_pool, handler, event
                                )
                        except Exception as e:
                            self.logger.error(f"Event handler failed: {str(e)}")
                            with self._metrics_lock:
                                self._performance_metrics["processing_errors"] += 1
                            self._handle_circuit_breaker_failure()
                    
                    # Update metrics
                    processing_time = (time.time() - start_time) * 1000  # ms
                    with self._metrics_lock:
                        self._performance_metrics["events_processed"] += 1
                        # Update rolling average
                        current_avg = self._performance_metrics["average_processing_time_ms"]
                        count = self._performance_metrics["events_processed"]
                        self._performance_metrics["average_processing_time_ms"] = (
                            (current_avg * (count - 1) + processing_time) / count
                        )
                    
                except Exception as e:
                    self.logger.error(f"Event processing failed: {str(e)}")
                    with self._metrics_lock:
                        self._performance_metrics["processing_errors"] += 1
                    self._handle_circuit_breaker_failure()
                
            except Exception as e:
                self.logger.error(f"Unexpected error in event processing: {str(e)}")
                await asyncio.sleep(0.1)  # Brief pause before retrying
    
    def _handle_circuit_breaker_failure(self) -> None:
        """Handle a circuit breaker failure."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = time.time()
        
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            self._circuit_breaker_open = True
            self.logger.warning("Circuit breaker opened due to repeated failures")
    
    def _should_reset_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be reset."""
        if not self._circuit_breaker_open:
            return False
        
        if self._circuit_breaker_last_failure is None:
            return True
        
        return (time.time() - self._circuit_breaker_last_failure) > self._circuit_breaker_timeout
    
    def _reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker."""
        self._circuit_breaker_open = False
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None
        self.logger.info("Circuit breaker reset")
    
    # Session management methods
    
    def _create_session(self, agent_id: str, config: Dict[str, Any]) -> MonitoringSession:
        """
        Create a new monitoring session.
        
        Args:
            agent_id: Agent identifier
            config: Configuration parameters
            
        Returns:
            MonitoringSession: New session object
        """
        session = MonitoringSession(
            agent_id=agent_id,
            framework=self.get_framework_name(),
            configuration=config.copy()
        )
        
        with self._session_lock:
            self._active_sessions[session.session_id] = session
        
        return session
    
    def _get_session(self, session_id: str) -> Optional[MonitoringSession]:
        """
        Get a monitoring session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            MonitoringSession or None if not found
        """
        with self._session_lock:
            return self._active_sessions.get(session_id)
    
    def _end_session(self, session_id: str) -> Optional[MonitoringSession]:
        """
        End a monitoring session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            MonitoringSession or None if not found
        """
        with self._session_lock:
            session = self._active_sessions.pop(session_id, None)
            if session:
                session.end_time = datetime.utcnow()
                session.status = "stopped"
        
        return session
    
    def get_active_sessions(self) -> List[MonitoringSession]:
        """
        Get all active monitoring sessions.
        
        Returns:
            List of active MonitoringSession objects
        """
        with self._session_lock:
            return list(self._active_sessions.values())
    
    # Performance monitoring methods
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        with self._metrics_lock:
            return self._performance_metrics.copy()
    
    def _check_overhead(self, agent_execution_time: float, monitoring_time: float) -> None:
        """
        Check if monitoring overhead is within acceptable limits.
        
        Args:
            agent_execution_time: Time spent on agent execution (seconds)
            monitoring_time: Time spent on monitoring (seconds)
            
        Raises:
            MonitoringOverheadError: If overhead exceeds limits
        """
        if agent_execution_time <= 0:
            return
        
        overhead_percent = (monitoring_time / agent_execution_time) * 100
        
        with self._metrics_lock:
            self._performance_metrics["overhead_percentage"] = overhead_percent
        
        if overhead_percent > self.max_overhead_percent:
            raise MonitoringOverheadError(
                f"Monitoring overhead {overhead_percent:.2f}% exceeds limit {self.max_overhead_percent}%"
            )
    
    # Lifecycle methods
    
    async def start(self) -> None:
        """Start the instrumentor's background processing."""
        if self._processing_task is None or self._processing_task.done():
            self._shutdown_event.clear()
            self._processing_task = asyncio.create_task(self._process_events())
            self.logger.info(f"{self.get_framework_name()} instrumentor started")
    
    async def stop(self) -> None:
        """Stop the instrumentor and clean up resources."""
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for processing task to complete
        if self._processing_task and not self._processing_task.done():
            try:
                if self._processing_task is None:
                    raise RuntimeError("Processing task is missing")
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Processing task did not complete within timeout")
                self._processing_task.cancel()
        
        # Stop all active sessions
        active_sessions = self.get_active_sessions()
        for session in active_sessions:
            try:
                await self.stop_monitoring(session.session_id)
            except Exception as e:
                self.logger.error(f"Error stopping session {session.session_id}: {str(e)}")
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        self.logger.info(f"{self.get_framework_name()} instrumentor stopped")
    
    # Utility methods
    
    def create_event(self, event_type: EventType, agent_id: str, session_id: str,
                    message: str = "", **kwargs) -> AgentEvent:
        """
        Create a standardized agent event.
        
        Args:
            event_type: Type of event
            agent_id: Agent identifier
            session_id: Session identifier
            message: Event message
            **kwargs: Additional event properties
            
        Returns:
            AgentEvent: Configured event object
        """
        event = AgentEvent(
            event_type=event_type,
            agent_id=agent_id,
            session_id=session_id,
            message=message,
            framework=self.get_framework_name()
        )
        
        # Set additional properties from kwargs
        for key, value in kwargs.items():
            if hasattr(event, key):
                setattr(event, key, value)
        
        return event
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Note: This is synchronous, so we can't call async stop() here
        # Users should call stop() explicitly in async contexts
        pass