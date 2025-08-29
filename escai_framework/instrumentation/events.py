"""
Event system for the ESCAI framework instrumentation layer.

This module defines standardized event types and structures for capturing
agent execution data across different frameworks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json
import uuid


class EventType(Enum):
    """Standardized event types for agent monitoring."""
    
    # Agent lifecycle events
    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    AGENT_ERROR = "agent_error"
    
    # Task execution events
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAIL = "task_fail"
    
    # Decision making events
    DECISION_START = "decision_start"
    DECISION_COMPLETE = "decision_complete"
    BELIEF_UPDATE = "belief_update"
    GOAL_UPDATE = "goal_update"
    KNOWLEDGE_UPDATE = "knowledge_update"
    
    # Tool and action events
    TOOL_START = "tool_start"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    ACTION_START = "action_start"
    ACTION_COMPLETE = "action_complete"
    
    # Communication events (for multi-agent systems)
    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    MESSAGE_SEND = "message_send"
    MESSAGE_RECEIVE = "message_receive"
    
    # Memory and context events
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    CONTEXT_UPDATE = "context_update"
    
    # Performance and monitoring events
    PERFORMANCE_METRIC = "performance_metric"
    RESOURCE_USAGE = "resource_usage"
    ERROR = "error"
    
    # Custom framework-specific events
    CUSTOM = "custom"


class EventSeverity(Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AgentEvent:
    """
    Standardized event structure for agent monitoring.
    
    This is the core event structure that all framework instrumentors
    should use to ensure consistent data capture and processing.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CUSTOM
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_id: str = ""
    session_id: str = ""
    severity: EventSeverity = EventSeverity.INFO
    
    # Core event data
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Context information
    framework: str = ""  # e.g., "langchain", "autogen", "crewai"
    component: str = ""  # e.g., "chain", "agent", "tool"
    operation: str = ""  # e.g., "execute", "plan", "reflect"
    
    # Performance metrics
    duration_ms: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Relationships
    parent_event_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate the event data."""
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            return False
        if not isinstance(self.event_type, EventType):
            return False
        if not isinstance(self.timestamp, datetime):
            return False
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            return False
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            return False
        if not isinstance(self.severity, EventSeverity):
            return False
        if not isinstance(self.message, str):
            return False
        if not isinstance(self.data, dict):
            return False
        if not isinstance(self.framework, str):
            return False
        if not isinstance(self.component, str):
            return False
        if not isinstance(self.operation, str):
            return False
        if self.duration_ms is not None and (not isinstance(self.duration_ms, int) or self.duration_ms < 0):
            return False
        if self.memory_usage_mb is not None and (not isinstance(self.memory_usage_mb, (int, float)) or self.memory_usage_mb < 0):
            return False
        if self.cpu_usage_percent is not None and (not isinstance(self.cpu_usage_percent, (int, float)) or not 0 <= self.cpu_usage_percent <= 100):
            return False
        if not isinstance(self.tags, list):
            return False
        if not isinstance(self.metadata, dict):
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "severity": self.severity.value,
            "message": self.message,
            "data": self.data,
            "framework": self.framework,
            "component": self.component,
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "parent_event_id": self.parent_event_id,
            "correlation_id": self.correlation_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentEvent':
        """Create from dictionary representation."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=EventType(data.get("event_type", EventType.CUSTOM.value)),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            agent_id=data.get("agent_id", ""),
            session_id=data.get("session_id", ""),
            severity=EventSeverity(data.get("severity", EventSeverity.INFO.value)),
            message=data.get("message", ""),
            data=data.get("data", {}),
            framework=data.get("framework", ""),
            component=data.get("component", ""),
            operation=data.get("operation", ""),
            duration_ms=data.get("duration_ms"),
            memory_usage_mb=data.get("memory_usage_mb"),
            cpu_usage_percent=data.get("cpu_usage_percent"),
            parent_event_id=data.get("parent_event_id"),
            correlation_id=data.get("correlation_id"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            stack_trace=data.get("stack_trace"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentEvent':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the event."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_error(self, error_type: str, error_message: str, stack_trace: Optional[str] = None) -> None:
        """Set error information for the event."""
        self.error_type = error_type
        self.error_message = error_message
        self.stack_trace = stack_trace
        self.severity = EventSeverity.ERROR
    
    def set_performance_metrics(self, duration_ms: Optional[int] = None, 
                              memory_usage_mb: Optional[float] = None,
                              cpu_usage_percent: Optional[float] = None) -> None:
        """Set performance metrics for the event."""
        if duration_ms is not None:
            self.duration_ms = duration_ms
        if memory_usage_mb is not None:
            self.memory_usage_mb = memory_usage_mb
        if cpu_usage_percent is not None:
            self.cpu_usage_percent = cpu_usage_percent


@dataclass
class MonitoringSession:
    """Represents a monitoring session for an agent."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    framework: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active, stopped, error
    event_count: int = 0
    
    def validate(self) -> bool:
        """Validate the monitoring session data."""
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            return False
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            return False
        if not isinstance(self.framework, str) or not self.framework.strip():
            return False
        if not isinstance(self.start_time, datetime):
            return False
        if self.end_time is not None and not isinstance(self.end_time, datetime):
            return False
        if not isinstance(self.configuration, dict):
            return False
        if not isinstance(self.status, str):
            return False
        if not isinstance(self.event_count, int) or self.event_count < 0:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "framework": self.framework,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "configuration": self.configuration,
            "status": self.status,
            "event_count": self.event_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringSession':
        """Create from dictionary representation."""
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            agent_id=data.get("agent_id", ""),
            framework=data.get("framework", ""),
            start_time=datetime.fromisoformat(data["start_time"]) if "start_time" in data else datetime.utcnow(),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            configuration=data.get("configuration", {}),
            status=data.get("status", "active"),
            event_count=data.get("event_count", 0)
        )


@dataclass
class MonitoringSummary:
    """Summary of a completed monitoring session."""
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
    
    def validate(self) -> bool:
        """Validate the monitoring summary data."""
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            return False
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            return False
        if not isinstance(self.framework, str) or not self.framework.strip():
            return False
        if not isinstance(self.start_time, datetime):
            return False
        if not isinstance(self.end_time, datetime):
            return False
        if not isinstance(self.total_duration_ms, int) or self.total_duration_ms < 0:
            return False
        if not isinstance(self.total_events, int) or self.total_events < 0:
            return False
        if not isinstance(self.event_types_count, dict):
            return False
        if not isinstance(self.error_count, int) or self.error_count < 0:
            return False
        if not isinstance(self.performance_metrics, dict):
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringSummary':
        """Create from dictionary representation."""
        return cls(
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            framework=data["framework"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            total_duration_ms=data["total_duration_ms"],
            total_events=data["total_events"],
            event_types_count=data.get("event_types_count", {}),
            error_count=data.get("error_count", 0),
            performance_metrics=data.get("performance_metrics", {})
        )