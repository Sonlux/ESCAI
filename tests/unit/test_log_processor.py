"""
Unit tests for the log processor functionality.
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

from escai_framework.instrumentation.log_processor import (
    LogProcessor, LogEntry, ProcessingRule, LogProcessingError, LogParsingError
)
from escai_framework.instrumentation.events import AgentEvent, EventType, EventSeverity


class TestLogEntry:
    """Test cases for LogEntry."""
    
    @pytest.fixture
    def sample_log_entry(self):
        """Create a sample log entry."""
        return LogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message="Test log message",
            source="test_source",
            framework="test_framework",
            component="test_component",
            raw_data={"key": "value"}
        )
    
    def test_log_entry_creation(self, sample_log_entry):
        """Test basic log entry creation."""
        assert isinstance(sample_log_entry.timestamp, datetime)
        assert sample_log_entry.level == "INFO"
        assert sample_log_entry.message == "Test log message"
        assert sample_log_entry.source == "test_source"
        assert sample_log_entry.framework == "test_framework"
        assert sample_log_entry.component == "test_component"
        assert sample_log_entry.raw_data == {"key": "value"}
    
    def test_log_entry_validation_valid(self, sample_log_entry):
        """Test validation of valid log entry."""
        assert sample_log_entry.validate() is True
    
    def test_log_entry_validation_invalid(self, sample_log_entry):
        """Test validation with invalid data."""
        sample_log_entry.timestamp = "not_a_datetime"
        assert sample_log_entry.validate() is False
        
        sample_log_entry.timestamp = datetime.utcnow()
        sample_log_entry.level = 123
        assert sample_log_entry.validate() is False
        
        sample_log_entry.level = "INFO"
        sample_log_entry.raw_data = "not_a_dict"
        assert sample_log_entry.validate() is False


class TestProcessingRule:
    """Test cases for ProcessingRule."""
    
    @pytest.fixture
    def sample_rule(self):
        """Create a sample processing rule."""
        return ProcessingRule(
            name="test_rule",
            pattern=r"task\s+(start|begin)",
            event_type=EventType.TASK_START,
            severity_mapping={"INFO": EventSeverity.INFO, "DEBUG": EventSeverity.DEBUG},
            field_extractors={"action": "1"},
            priority=5
        )
    
    @pytest.fixture
    def sample_log_entry(self):
        """Create a sample log entry."""
        return LogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message="Task start execution",
            source="test_source",
            framework="test_framework"
        )
    
    def test_rule_creation(self, sample_rule):
        """Test basic rule creation."""
        assert sample_rule.name == "test_rule"
        assert sample_rule.pattern == r"task\s+(start|begin)"
        assert sample_rule.event_type == EventType.TASK_START
        assert sample_rule.priority == 5
        assert "INFO" in sample_rule.severity_mapping
        assert "action" in sample_rule.field_extractors
    
    def test_rule_matches_positive(self, sample_rule, sample_log_entry):
        """Test rule matching with positive case."""
        assert sample_rule.matches(sample_log_entry) is True
        
        # Test with different message
        sample_log_entry.message = "Task begin processing"
        assert sample_rule.matches(sample_log_entry) is True
    
    def test_rule_matches_negative(self, sample_rule, sample_log_entry):
        """Test rule matching with negative case."""
        sample_log_entry.message = "Task complete execution"
        assert sample_rule.matches(sample_log_entry) is False
        
        sample_log_entry.message = "No match here"
        assert sample_rule.matches(sample_log_entry) is False
    
    def test_rule_extract_fields(self, sample_rule, sample_log_entry):
        """Test field extraction from log entry."""
        fields = sample_rule.extract_fields(sample_log_entry)
        
        assert "action" in fields
        assert fields["action"] == "start"
        
        # Test with different message
        sample_log_entry.message = "Task begin processing"
        fields = sample_rule.extract_fields(sample_log_entry)
        assert fields["action"] == "begin"
    
    def test_rule_with_condition(self):
        """Test rule with custom condition."""
        def custom_condition(log_entry: LogEntry) -> bool:
            return log_entry.framework == "specific_framework"
        
        rule = ProcessingRule(
            name="conditional_rule",
            pattern=r"test",
            event_type=EventType.CUSTOM,
            condition=custom_condition
        )
        
        # Should match pattern but fail condition
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message="test message",
            framework="other_framework"
        )
        assert rule.matches(log_entry) is False
        
        # Should match both pattern and condition
        log_entry.framework = "specific_framework"
        assert rule.matches(log_entry) is True


class TestLogProcessor:
    """Test cases for LogProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a log processor instance."""
        return LogProcessor(max_workers=2)
    
    @pytest.fixture
    def sample_log_entry(self):
        """Create a sample log entry."""
        return LogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message="Agent start execution",
            source="test_source",
            framework="test_framework",
            component="test_component"
        )
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.max_workers == 2
        assert len(processor.get_rules()) > 0  # Should have default rules
        
        # Check some default rules exist
        rule_names = [rule.name for rule in processor.get_rules()]
        assert "agent_start" in rule_names
        assert "agent_stop" in rule_names
        assert "task_start" in rule_names
        assert "error" in rule_names
    
    def test_add_remove_rules(self, processor):
        """Test adding and removing processing rules."""
        initial_count = len(processor.get_rules())
        
        # Add new rule
        new_rule = ProcessingRule(
            name="custom_rule",
            pattern=r"custom pattern",
            event_type=EventType.CUSTOM,
            priority=20
        )
        processor.add_rule(new_rule)
        
        assert len(processor.get_rules()) == initial_count + 1
        
        # Rules should be sorted by priority
        rules = processor.get_rules()
        assert rules[0].priority >= rules[-1].priority
        
        # Remove rule
        assert processor.remove_rule("custom_rule") is True
        assert len(processor.get_rules()) == initial_count
        
        # Try to remove non-existent rule
        assert processor.remove_rule("nonexistent") is False
    
    @pytest.mark.asyncio
    async def test_process_log_entry_success(self, processor, sample_log_entry):
        """Test successful log entry processing."""
        event = await processor.process_log_entry(
            sample_log_entry, 
            agent_id="test_agent",
            session_id="test_session"
        )
        
        assert event is not None
        assert isinstance(event, AgentEvent)
        assert event.event_type == EventType.AGENT_START
        assert event.agent_id == "test_agent"
        assert event.session_id == "test_session"
        assert event.message == sample_log_entry.message
        assert event.framework == sample_log_entry.framework
        assert event.validate()
    
    @pytest.mark.asyncio
    async def test_process_log_entry_no_match(self, processor):
        """Test log entry processing with no matching rule."""
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message="This message matches no rules",
            framework="test_framework"
        )
        
        event = await processor.process_log_entry(
            log_entry,
            agent_id="test_agent",
            session_id="test_session"
        )
        
        assert event is None
        
        # Check metrics
        metrics = processor.get_metrics()
        assert metrics["unmatched_logs"] > 0
    
    @pytest.mark.asyncio
    async def test_process_log_entry_invalid(self, processor):
        """Test processing invalid log entry."""
        invalid_log_entry = LogEntry(
            timestamp="not_a_datetime",  # Invalid
            level="INFO",
            message="Test message",
            framework="test_framework"
        )
        
        with pytest.raises(LogProcessingError):
            await processor.process_log_entry(invalid_log_entry)
    
    @pytest.mark.asyncio
    async def test_process_log_batch(self, processor):
        """Test batch processing of log entries."""
        log_entries = []
        for i in range(5):
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level="INFO",
                message=f"Agent start execution {i}",
                framework="test_framework"
            )
            log_entries.append(log_entry)
        
        events = await processor.process_log_batch(
            log_entries,
            agent_id="test_agent",
            session_id="test_session"
        )
        
        assert len(events) == 5
        for event in events:
            assert isinstance(event, AgentEvent)
            assert event.event_type == EventType.AGENT_START
            assert event.validate()
    
    @pytest.mark.asyncio
    async def test_process_empty_batch(self, processor):
        """Test processing empty batch."""
        events = await processor.process_log_batch([])
        assert events == []
    
    def test_parse_json_log_line(self, processor):
        """Test parsing JSON log lines."""
        json_log = {
            "timestamp": "2023-01-01T12:00:00Z",
            "level": "INFO",
            "message": "Test message",
            "component": "test_component"
        }
        json_line = json.dumps(json_log)
        
        log_entry = processor.parse_log_line(
            json_line,
            framework="test_framework",
            source="test_source"
        )
        
        assert log_entry is not None
        assert log_entry.level == "INFO"
        assert log_entry.message == "Test message"
        assert log_entry.framework == "test_framework"
        assert log_entry.component == "test_component"
        assert log_entry.validate()
    
    def test_parse_text_log_line(self, processor):
        """Test parsing text log lines."""
        test_cases = [
            # ISO timestamp with level
            "2023-01-01T12:00:00Z [INFO] Test message",
            # Standard timestamp with level
            "2023-01-01 12:00:00 INFO Test message",
            # Level first
            "INFO 2023-01-01 12:00:00 Test message",
            # Simple level and message
            "[INFO] Test message",
            # Just message
            "Test message"
        ]
        
        for log_line in test_cases:
            log_entry = processor.parse_log_line(
                log_line,
                framework="test_framework",
                source="test_source"
            )
            
            assert log_entry is not None
            assert log_entry.framework == "test_framework"
            assert log_entry.source == "test_source"
            assert "Test message" in log_entry.message
            assert log_entry.validate()
    
    def test_parse_invalid_log_line(self, processor):
        """Test parsing invalid log lines."""
        # Invalid JSON
        invalid_json = '{"invalid": json}'
        log_entry = processor.parse_log_line(invalid_json)
        assert log_entry is None
        
        # Empty line
        log_entry = processor.parse_log_line("")
        assert log_entry is None
    
    def test_metrics_tracking(self, processor):
        """Test metrics tracking."""
        initial_metrics = processor.get_metrics()
        
        # Check initial state
        assert "logs_processed" in initial_metrics
        assert "events_generated" in initial_metrics
        assert "processing_errors" in initial_metrics
        assert "unmatched_logs" in initial_metrics
        assert "average_processing_time_ms" in initial_metrics
        
        # Reset metrics
        processor.reset_metrics()
        reset_metrics = processor.get_metrics()
        
        assert reset_metrics["logs_processed"] == 0
        assert reset_metrics["events_generated"] == 0
        assert reset_metrics["processing_errors"] == 0
        assert reset_metrics["unmatched_logs"] == 0
        assert reset_metrics["average_processing_time_ms"] == 0.0
    
    def test_severity_mapping(self, processor):
        """Test severity level mapping."""
        # Add custom rule with severity mapping
        rule = ProcessingRule(
            name="severity_test",
            pattern=r"severity test",
            event_type=EventType.CUSTOM,
            severity_mapping={
                "ERROR": EventSeverity.ERROR,
                "WARNING": EventSeverity.WARNING,
                "INFO": EventSeverity.INFO
            }
        )
        processor.add_rule(rule)
        
        # Test different severity levels
        test_cases = [
            ("ERROR", EventSeverity.ERROR),
            ("WARNING", EventSeverity.WARNING),
            ("INFO", EventSeverity.INFO),
            ("DEBUG", EventSeverity.INFO)  # Default fallback
        ]
        
        for level, expected_severity in test_cases:
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level=level,
                message="severity test message",
                framework="test_framework"
            )
            
            # Process synchronously for testing
            event = processor._process_log_entry_sync(
                log_entry,
                agent_id="test_agent",
                session_id="test_session"
            )
            
            assert event is not None
            assert event.severity == expected_severity
    
    def test_field_extraction(self, processor):
        """Test field extraction from log messages."""
        # Add rule with field extractors
        rule = ProcessingRule(
            name="extraction_test",
            pattern=r"User (\w+) performed action (\w+)",
            event_type=EventType.CUSTOM,
            field_extractors={"user": "1", "action": "2"}
        )
        processor.add_rule(rule)
        
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message="User alice performed action login",
            framework="test_framework"
        )
        
        event = processor._process_log_entry_sync(
            log_entry,
            agent_id="test_agent",
            session_id="test_session"
        )
        
        assert event is not None
        assert event.data["user"] == "alice"
        assert event.data["action"] == "login"
    
    def test_processor_shutdown(self, processor):
        """Test processor shutdown."""
        # Should not raise exceptions
        processor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])