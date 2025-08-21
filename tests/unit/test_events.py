"""
Unit tests for the events module.
"""

import pytest
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

from escai_framework.instrumentation.events import (
    AgentEvent, EventType, EventSeverity, MonitoringSession, MonitoringSummary
)


class TestAgentEvent:
    """Test cases for AgentEvent."""
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample agent event."""
        return AgentEvent(
            event_type=EventType.AGENT_START,
            agent_id="test_agent",
            session_id="test_session",
            message="Test event message",
            framework="test_framework",
            component="test_component",
            operation="test_operation"
        )
    
    def test_event_creation(self, sample_event):
        """Test basic event creation."""
        assert sample_event.event_type == EventType.AGENT_START
        assert sample_event.agent_id == "test_agent"
        assert sample_event.session_id == "test_session"
        assert sample_event.message == "Test event message"
        assert sample_event.framework == "test_framework"
        assert sample_event.component == "test_component"
        assert sample_event.operation == "test_operation"
        assert sample_event.severity == EventSeverity.INFO
        assert isinstance(sample_event.timestamp, datetime)
        assert isinstance(sample_event.event_id, str)
        assert len(sample_event.event_id) > 0
    
    def test_event_validation_valid(self, sample_event):
        """Test validation of valid event."""
        assert sample_event.validate() is True
    
    def test_event_validation_invalid_event_id(self, sample_event):
        """Test validation with invalid event ID."""
        sample_event.event_id = ""
        assert sample_event.validate() is False
        
        sample_event.event_id = None
        assert sample_event.validate() is False
    
    def test_event_validation_invalid_confidence(self, sample_event):
        """Test validation with invalid confidence values."""
        sample_event.duration_ms = -1
        assert sample_event.validate() is False
        
        sample_event.duration_ms = 100
        sample_event.memory_usage_mb = -1.0
        assert sample_event.validate() is False
        
        sample_event.memory_usage_mb = 50.0
        sample_event.cpu_usage_percent = 150.0  # > 100%
        assert sample_event.validate() is False
    
    def test_event_to_dict(self, sample_event):
        """Test event serialization to dictionary."""
        event_dict = sample_event.to_dict()
        
        assert isinstance(event_dict, dict)
        assert event_dict["event_type"] == EventType.AGENT_START.value
        assert event_dict["agent_id"] == "test_agent"
        assert event_dict["session_id"] == "test_session"
        assert event_dict["message"] == "Test event message"
        assert event_dict["framework"] == "test_framework"
        assert event_dict["severity"] == EventSeverity.INFO.value
        assert "timestamp" in event_dict
        assert isinstance(event_dict["timestamp"], str)
    
    def test_event_from_dict(self, sample_event):
        """Test event deserialization from dictionary."""
        event_dict = sample_event.to_dict()
        restored_event = AgentEvent.from_dict(event_dict)
        
        assert restored_event.event_type == sample_event.event_type
        assert restored_event.agent_id == sample_event.agent_id
        assert restored_event.session_id == sample_event.session_id
        assert restored_event.message == sample_event.message
        assert restored_event.framework == sample_event.framework
        assert restored_event.severity == sample_event.severity
        assert restored_event.validate()
    
    def test_event_json_serialization(self, sample_event):
        """Test JSON serialization and deserialization."""
        json_str = sample_event.to_json()
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        json_data = json.loads(json_str)
        assert isinstance(json_data, dict)
        
        # Restore from JSON
        restored_event = AgentEvent.from_json(json_str)
        assert restored_event.event_type == sample_event.event_type
        assert restored_event.agent_id == sample_event.agent_id
        assert restored_event.validate()
    
    def test_event_add_tag(self, sample_event):
        """Test adding tags to event."""
        assert len(sample_event.tags) == 0
        
        sample_event.add_tag("important")
        assert "important" in sample_event.tags
        assert len(sample_event.tags) == 1
        
        # Adding same tag again should not duplicate
        sample_event.add_tag("important")
        assert len(sample_event.tags) == 1
        
        # Add different tag
        sample_event.add_tag("urgent")
        assert "urgent" in sample_event.tags
        assert len(sample_event.tags) == 2
    
    def test_event_set_error(self, sample_event):
        """Test setting error information."""
        sample_event.set_error("ValueError", "Invalid input", "Stack trace here")
        
        assert sample_event.error_type == "ValueError"
        assert sample_event.error_message == "Invalid input"
        assert sample_event.stack_trace == "Stack trace here"
        assert sample_event.severity == EventSeverity.ERROR
    
    def test_event_set_performance_metrics(self, sample_event):
        """Test setting performance metrics."""
        sample_event.set_performance_metrics(
            duration_ms=1500,
            memory_usage_mb=128.5,
            cpu_usage_percent=75.2
        )
        
        assert sample_event.duration_ms == 1500
        assert sample_event.memory_usage_mb == 128.5
        assert sample_event.cpu_usage_percent == 75.2
    
    def test_event_with_all_fields(self):
        """Test event creation with all fields populated."""
        event = AgentEvent(
            event_type=EventType.TASK_COMPLETE,
            agent_id="complex_agent",
            session_id="complex_session",
            severity=EventSeverity.WARNING,
            message="Complex event",
            framework="complex_framework",
            component="complex_component",
            operation="complex_operation",
            duration_ms=2000,
            memory_usage_mb=256.0,
            cpu_usage_percent=85.5,
            parent_event_id="parent_123",
            correlation_id="corr_456",
            error_type="CustomError",
            error_message="Something went wrong",
            stack_trace="Line 1\nLine 2",
            tags=["tag1", "tag2"],
            metadata={"key": "value"}
        )
        
        assert event.validate()
        assert event.event_type == EventType.TASK_COMPLETE
        assert event.severity == EventSeverity.WARNING
        assert event.duration_ms == 2000
        assert event.memory_usage_mb == 256.0
        assert event.cpu_usage_percent == 85.5
        assert event.parent_event_id == "parent_123"
        assert event.correlation_id == "corr_456"
        assert event.error_type == "CustomError"
        assert len(event.tags) == 2
        assert event.metadata["key"] == "value"


class TestMonitoringSession:
    """Test cases for MonitoringSession."""
    
    @pytest.fixture
    def sample_session(self):
        """Create a sample monitoring session."""
        return MonitoringSession(
            agent_id="test_agent",
            framework="test_framework",
            configuration={"param1": "value1", "param2": 42}
        )
    
    def test_session_creation(self, sample_session):
        """Test basic session creation."""
        assert sample_session.agent_id == "test_agent"
        assert sample_session.framework == "test_framework"
        assert sample_session.configuration == {"param1": "value1", "param2": 42}
        assert sample_session.status == "active"
        assert sample_session.event_count == 0
        assert isinstance(sample_session.session_id, str)
        assert len(sample_session.session_id) > 0
        assert isinstance(sample_session.start_time, datetime)
        assert sample_session.end_time is None
    
    def test_session_validation_valid(self, sample_session):
        """Test validation of valid session."""
        assert sample_session.validate() is True
    
    def test_session_validation_invalid(self, sample_session):
        """Test validation with invalid data."""
        sample_session.agent_id = ""
        assert sample_session.validate() is False
        
        sample_session.agent_id = "test_agent"
        sample_session.framework = ""
        assert sample_session.validate() is False
        
        sample_session.framework = "test_framework"
        sample_session.event_count = -1
        assert sample_session.validate() is False
    
    def test_session_serialization(self, sample_session):
        """Test session serialization and deserialization."""
        session_dict = sample_session.to_dict()
        
        assert isinstance(session_dict, dict)
        assert session_dict["agent_id"] == "test_agent"
        assert session_dict["framework"] == "test_framework"
        assert session_dict["status"] == "active"
        assert "start_time" in session_dict
        
        # Restore from dict
        restored_session = MonitoringSession.from_dict(session_dict)
        assert restored_session.agent_id == sample_session.agent_id
        assert restored_session.framework == sample_session.framework
        assert restored_session.configuration == sample_session.configuration
        assert restored_session.validate()


class TestMonitoringSummary:
    """Test cases for MonitoringSummary."""
    
    @pytest.fixture
    def sample_summary(self):
        """Create a sample monitoring summary."""
        start_time = datetime.utcnow() - timedelta(minutes=5)
        end_time = datetime.utcnow()
        
        return MonitoringSummary(
            session_id="test_session",
            agent_id="test_agent",
            framework="test_framework",
            start_time=start_time,
            end_time=end_time,
            total_duration_ms=300000,  # 5 minutes
            total_events=150,
            event_types_count={
                "agent_start": 1,
                "task_start": 10,
                "task_complete": 8,
                "agent_stop": 1
            },
            error_count=2,
            performance_metrics={
                "avg_response_time": 250.5,
                "peak_memory_mb": 512.0
            }
        )
    
    def test_summary_creation(self, sample_summary):
        """Test basic summary creation."""
        assert sample_summary.session_id == "test_session"
        assert sample_summary.agent_id == "test_agent"
        assert sample_summary.framework == "test_framework"
        assert sample_summary.total_duration_ms == 300000
        assert sample_summary.total_events == 150
        assert sample_summary.error_count == 2
        assert isinstance(sample_summary.start_time, datetime)
        assert isinstance(sample_summary.end_time, datetime)
        assert sample_summary.end_time > sample_summary.start_time
    
    def test_summary_validation_valid(self, sample_summary):
        """Test validation of valid summary."""
        assert sample_summary.validate() is True
    
    def test_summary_validation_invalid(self, sample_summary):
        """Test validation with invalid data."""
        sample_summary.session_id = ""
        assert sample_summary.validate() is False
        
        sample_summary.session_id = "test_session"
        sample_summary.total_duration_ms = -1
        assert sample_summary.validate() is False
        
        sample_summary.total_duration_ms = 300000
        sample_summary.total_events = -1
        assert sample_summary.validate() is False
        
        sample_summary.total_events = 150
        sample_summary.error_count = -1
        assert sample_summary.validate() is False
    
    def test_summary_serialization(self, sample_summary):
        """Test summary serialization and deserialization."""
        summary_dict = sample_summary.to_dict()
        
        assert isinstance(summary_dict, dict)
        assert summary_dict["session_id"] == "test_session"
        assert summary_dict["agent_id"] == "test_agent"
        assert summary_dict["total_events"] == 150
        assert summary_dict["error_count"] == 2
        assert "start_time" in summary_dict
        assert "end_time" in summary_dict
        
        # Restore from dict
        restored_summary = MonitoringSummary.from_dict(summary_dict)
        assert restored_summary.session_id == sample_summary.session_id
        assert restored_summary.agent_id == sample_summary.agent_id
        assert restored_summary.total_events == sample_summary.total_events
        assert restored_summary.error_count == sample_summary.error_count
        assert restored_summary.validate()
    
    def test_summary_event_types_count(self, sample_summary):
        """Test event types count tracking."""
        event_types = sample_summary.event_types_count
        
        assert event_types["agent_start"] == 1
        assert event_types["task_start"] == 10
        assert event_types["task_complete"] == 8
        assert event_types["agent_stop"] == 1
        
        # Total should match
        total_counted = sum(event_types.values())
        assert total_counted == 20  # Not necessarily equal to total_events
    
    def test_summary_performance_metrics(self, sample_summary):
        """Test performance metrics tracking."""
        metrics = sample_summary.performance_metrics
        
        assert metrics["avg_response_time"] == 250.5
        assert metrics["peak_memory_mb"] == 512.0


class TestEventTypes:
    """Test cases for event type enums."""
    
    def test_event_type_values(self):
        """Test EventType enum values."""
        assert EventType.AGENT_START.value == "agent_start"
        assert EventType.AGENT_STOP.value == "agent_stop"
        assert EventType.TASK_START.value == "task_start"
        assert EventType.TASK_COMPLETE.value == "task_complete"
        assert EventType.DECISION_START.value == "decision_start"
        assert EventType.TOOL_CALL.value == "tool_call"
        assert EventType.CUSTOM.value == "custom"
    
    def test_event_severity_values(self):
        """Test EventSeverity enum values."""
        assert EventSeverity.DEBUG.value == "debug"
        assert EventSeverity.INFO.value == "info"
        assert EventSeverity.WARNING.value == "warning"
        assert EventSeverity.ERROR.value == "error"
        assert EventSeverity.CRITICAL.value == "critical"
    
    def test_enum_serialization(self):
        """Test enum serialization in events."""
        event = AgentEvent(
            event_type=EventType.TOOL_CALL,
            severity=EventSeverity.WARNING,
            agent_id="test",
            session_id="test"
        )
        
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "tool_call"
        assert event_dict["severity"] == "warning"
        
        # Restore and verify
        restored = AgentEvent.from_dict(event_dict)
        assert restored.event_type == EventType.TOOL_CALL
        assert restored.severity == EventSeverity.WARNING


if __name__ == "__main__":
    pytest.main([__file__])