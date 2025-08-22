"""Agent framework integrations for the ESCAI framework."""

from .base_instrumentor import BaseInstrumentor, InstrumentationError, MonitoringOverheadError, EventProcessingError
from .events import (
    AgentEvent, EventType, EventSeverity, MonitoringSession, MonitoringSummary
)
from .log_processor import LogProcessor, LogEntry, ProcessingRule, LogProcessingError, LogParsingError
from .event_stream import (
    EventStream, StreamFilter, Subscriber, EventBuffer,
    StreamingError, SubscriberError, BufferOverflowError,
    create_agent_filter, create_event_type_filter, create_severity_filter,
    create_time_window_filter, create_framework_filter
)

# Framework-specific instrumentors
from .langchain_instrumentor import LangChainInstrumentor
from .autogen_instrumentor import AutoGenInstrumentor
from .crewai_instrumentor import CrewAIInstrumentor
from .openai_instrumentor import OpenAIInstrumentor

__all__ = [
    # Base classes
    "BaseInstrumentor",
    
    # Framework-specific instrumentors
    "LangChainInstrumentor",
    "AutoGenInstrumentor", 
    "CrewAIInstrumentor",
    "OpenAIInstrumentor",
    
    # Events
    "AgentEvent",
    "EventType", 
    "EventSeverity",
    "MonitoringSession",
    "MonitoringSummary",
    
    # Log processing
    "LogProcessor",
    "LogEntry",
    "ProcessingRule",
    
    # Event streaming
    "EventStream",
    "StreamFilter",
    "Subscriber",
    "EventBuffer",
    
    # Filter helpers
    "create_agent_filter",
    "create_event_type_filter", 
    "create_severity_filter",
    "create_time_window_filter",
    "create_framework_filter",
    
    # Exceptions
    "InstrumentationError",
    "MonitoringOverheadError",
    "EventProcessingError",
    "LogProcessingError",
    "LogParsingError",
    "StreamingError",
    "SubscriberError",
    "BufferOverflowError",
]