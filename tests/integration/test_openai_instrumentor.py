"""
Integration tests for OpenAI Assistants instrumentor.

These tests verify the integration with OpenAI Assistants API
and proper event capture during assistant interactions.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime

from escai_framework.instrumentation.openai_instrumentor import (
    OpenAIInstrumentor, 
    OpenAIAssistantMonitor,
    OPENAI_AVAILABLE
)
from escai_framework.instrumentation.events import EventType, EventSeverity, AgentEvent


# Mock OpenAI classes for testing when OpenAI is not available
class MockRun:
    def __init__(self, run_id="run_123", status="queued", usage=None, last_error=None, required_action=None):
        self.id = run_id
        self.status = status
        self.usage = usage or {}
        self.last_error = last_error
        self.required_action = required_action


class MockMessage:
    def __init__(self, message_id="msg_123", role="user", content="Hello"):
        self.id = message_id
        self.role = role
        self.content = content


class MockFunction:
    def __init__(self, name="test_function", arguments='{"param": "value"}'):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, call_id="call_123", function=None):
        self.id = call_id
        self.function = function or MockFunction()


class MockSubmitToolOutputs:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or [MockToolCall()]


class MockRequiredAction:
    def __init__(self, submit_tool_outputs=None):
        self.submit_tool_outputs = submit_tool_outputs or MockSubmitToolOutputs()


class MockRuns:
    def __init__(self):
        self._create_calls = []
        self._retrieve_calls = []
        self._submit_calls = []
    
    def create(self, thread_id, assistant_id, **kwargs):
        self._create_calls.append({
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "kwargs": kwargs
        })
        return MockRun()
    
    def retrieve(self, thread_id, run_id):
        self._retrieve_calls.append({
            "thread_id": thread_id,
            "run_id": run_id
        })
        return MockRun(run_id=run_id, status="completed")
    
    def submit_tool_outputs(self, thread_id, run_id, tool_outputs):
        self._submit_calls.append({
            "thread_id": thread_id,
            "run_id": run_id,
            "tool_outputs": tool_outputs
        })
        return MockRun(run_id=run_id, status="queued")


class MockMessages:
    def __init__(self):
        self._create_calls = []
    
    def create(self, thread_id, role, content, **kwargs):
        self._create_calls.append({
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "kwargs": kwargs
        })
        return MockMessage(role=role, content=content)


class MockThreads:
    def __init__(self):
        self.runs = MockRuns()
        self.messages = MockMessages()


class MockBeta:
    def __init__(self):
        self.threads = MockThreads()


class MockOpenAIClient:
    def __init__(self):
        self.beta = MockBeta()


@pytest.fixture
def instrumentor():
    """Create an OpenAI instrumentor for testing."""
    if not OPENAI_AVAILABLE:
        pytest.skip("OpenAI not available")
    
    return OpenAIInstrumentor()


@pytest.fixture
def mock_instrumentor():
    """Create a mock OpenAI instrumentor for testing without OpenAI."""
    with patch('escai_framework.instrumentation.openai_instrumentor.OPENAI_AVAILABLE', True):
        return OpenAIInstrumentor()


@pytest.fixture
def assistant_monitor(mock_instrumentor):
    """Create an assistant monitor for testing."""
    return OpenAIAssistantMonitor(
        instrumentor=mock_instrumentor,
        session_id="test_session",
        agent_id="test_agent"
    )


@pytest.fixture
def mock_client():
    """Create a mock OpenAI client for testing."""
    return MockOpenAIClient()


class TestOpenAIInstrumentor:
    """Test cases for OpenAI instrumentor."""
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, mock_instrumentor, mock_client):
        """Test starting monitoring for OpenAI Assistants."""
        agent_id = "test_assistant"
        config = {
            "clients": [mock_client],
            "assistant_ids": ["asst_123"],
            "monitor_functions": True,
            "monitor_reasoning": True
        }
        
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        assert session_id is not None
        assert len(session_id) > 0
        
        # Verify session was created
        session = mock_instrumentor._get_session(session_id)
        assert session is not None
        assert session.agent_id == agent_id
        assert session.framework == "openai"
        
        # Verify assistant monitor was created
        monitor = mock_instrumentor.get_assistant_monitor(session_id)
        assert monitor is not None
        assert monitor.agent_id == agent_id
        assert monitor.session_id == session_id
        
        # Verify client is being tracked
        assert session_id in mock_instrumentor._monitored_clients
        assert len(mock_instrumentor._monitored_clients[session_id]) == 1
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, mock_instrumentor, mock_client):
        """Test stopping monitoring for OpenAI Assistants."""
        agent_id = "test_assistant"
        config = {"clients": [mock_client]}
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        # Add some test events
        test_event = AgentEvent(
            event_type=EventType.TOOL_CALL,
            agent_id=agent_id,
            session_id=session_id,
            message="Test function call",
            framework="openai"
        )
        await mock_instrumentor.capture_event(test_event)
        
        # Stop monitoring
        summary = await mock_instrumentor.stop_monitoring(session_id)
        
        assert summary is not None
        assert summary.session_id == session_id
        assert summary.agent_id == agent_id
        assert summary.framework == "openai"
        assert summary.total_events >= 1  # At least the test event
        
        # Verify cleanup
        monitor = mock_instrumentor.get_assistant_monitor(session_id)
        assert monitor is None
        assert session_id not in mock_instrumentor._monitored_clients
    
    @pytest.mark.asyncio
    async def test_capture_event(self, mock_instrumentor, mock_client):
        """Test capturing events."""
        agent_id = "test_assistant"
        config = {"clients": [mock_client]}
        
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        # Create test event
        event = AgentEvent(
            event_type=EventType.TOOL_CALL,
            agent_id=agent_id,
            session_id=session_id,
            message="Test function call",
            framework="openai",
            component="function",
            operation="call",
            data={
                "function_name": "get_weather",
                "call_id": "call_123",
                "thread_id": "thread_456"
            }
        )
        
        # Capture event
        await mock_instrumentor.capture_event(event)
        
        # Verify session event count was updated
        session = mock_instrumentor._get_session(session_id)
        assert session.event_count >= 1
    
    def test_get_supported_events(self, mock_instrumentor):
        """Test getting supported event types."""
        supported_events = mock_instrumentor.get_supported_events()
        
        assert EventType.AGENT_START in supported_events
        assert EventType.AGENT_STOP in supported_events
        assert EventType.TOOL_CALL in supported_events
        assert EventType.TOOL_RESPONSE in supported_events
        assert EventType.MESSAGE_SEND in supported_events
        assert EventType.MESSAGE_RECEIVE in supported_events
        assert EventType.ACTION_START in supported_events
        assert EventType.ACTION_COMPLETE in supported_events
    
    def test_get_framework_name(self, mock_instrumentor):
        """Test getting framework name."""
        assert mock_instrumentor.get_framework_name() == "openai"


class TestOpenAIAssistantMonitor:
    """Test cases for OpenAI assistant monitor."""
    
    def test_monitor_run_creation(self, assistant_monitor, mock_client):
        """Test monitoring run creation."""
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(assistant_monitor, '_queue_event_safe', side_effect=capture_event):
            # Call monitor method
            result = assistant_monitor.monitor_run_creation(
                mock_client.beta.threads.runs.create,
                mock_client,
                thread_id="thread_123",
                assistant_id="asst_456",
                instructions="Test instructions"
            )
            
            # Verify original method was called
            assert len(mock_client.beta.threads.runs._create_calls) == 1
            call = mock_client.beta.threads.runs._create_calls[0]
            assert call["thread_id"] == "thread_123"
            assert call["assistant_id"] == "asst_456"
            
            # Verify events were captured
            assert len(events_captured) >= 2  # Start and created events
            
            start_event = events_captured[0]
            assert start_event.event_type == EventType.AGENT_START
            assert start_event.component == "assistant"
            assert start_event.operation == "run_create"
            assert start_event.data["thread_id"] == "thread_123"
            assert start_event.data["assistant_id"] == "asst_456"
            
            created_event = events_captured[1]
            assert created_event.event_type == EventType.TASK_START
            assert created_event.component == "assistant"
            assert created_event.operation == "run_created"
    
    def test_monitor_run_polling(self, assistant_monitor, mock_client):
        """Test monitoring run polling."""
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(assistant_monitor, '_queue_event_safe', side_effect=capture_event):
            # Call monitor method
            result = assistant_monitor.monitor_run_polling(
                mock_client.beta.threads.runs.retrieve,
                mock_client,
                thread_id="thread_123",
                run_id="run_456"
            )
            
            # Verify original method was called
            assert len(mock_client.beta.threads.runs._retrieve_calls) == 1
            call = mock_client.beta.threads.runs._retrieve_calls[0]
            assert call["thread_id"] == "thread_123"
            assert call["run_id"] == "run_456"
            
            # Verify events were captured
            assert len(events_captured) >= 1  # Status update event
            
            status_event = events_captured[0]
            assert status_event.event_type == EventType.CUSTOM
            assert status_event.component == "assistant"
            assert status_event.operation == "status_update"
            assert status_event.data["thread_id"] == "thread_123"
            assert status_event.data["run_id"] == "run_456"
            assert "status_update" in status_event.tags
    
    def test_monitor_message_creation(self, assistant_monitor, mock_client):
        """Test monitoring message creation."""
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(assistant_monitor, '_queue_event_safe', side_effect=capture_event):
            # Call monitor method for user message
            result = assistant_monitor.monitor_message_creation(
                mock_client.beta.threads.messages.create,
                mock_client,
                thread_id="thread_123",
                role="user",
                content="Hello, assistant!"
            )
            
            # Verify original method was called
            assert len(mock_client.beta.threads.messages._create_calls) == 1
            call = mock_client.beta.threads.messages._create_calls[0]
            assert call["thread_id"] == "thread_123"
            assert call["role"] == "user"
            assert call["content"] == "Hello, assistant!"
            
            # Verify events were captured
            assert len(events_captured) >= 1
            
            message_event = events_captured[0]
            assert message_event.event_type == EventType.MESSAGE_SEND  # User message
            assert message_event.component == "thread"
            assert message_event.operation == "message_create"
            assert message_event.data["thread_id"] == "thread_123"
            assert message_event.data["role"] == "user"
            assert message_event.data["content_length"] == len("Hello, assistant!")
    
    def test_monitor_function_submission(self, assistant_monitor, mock_client):
        """Test monitoring function call result submission."""
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        tool_outputs = [
            {"tool_call_id": "call_123", "output": "Weather is sunny"},
            {"tool_call_id": "call_456", "output": "Temperature is 25°C"}
        ]
        
        # Mock the event queuing
        with patch.object(assistant_monitor, '_queue_event_safe', side_effect=capture_event):
            # Call monitor method
            result = assistant_monitor.monitor_function_submission(
                mock_client.beta.threads.runs.submit_tool_outputs,
                mock_client,
                thread_id="thread_123",
                run_id="run_456",
                tool_outputs=tool_outputs
            )
            
            # Verify original method was called
            assert len(mock_client.beta.threads.runs._submit_calls) == 1
            call = mock_client.beta.threads.runs._submit_calls[0]
            assert call["thread_id"] == "thread_123"
            assert call["run_id"] == "run_456"
            assert call["tool_outputs"] == tool_outputs
            
            # Verify events were captured (2 function results + 1 completion)
            assert len(events_captured) >= 3
            
            # Check function result events
            result_events = [e for e in events_captured if e.event_type == EventType.TOOL_RESPONSE]
            assert len(result_events) == 2
            
            for i, event in enumerate(result_events):
                assert event.component == "function"
                assert event.operation == "result_submit"
                assert event.data["call_id"] == tool_outputs[i]["tool_call_id"]
                assert event.data["result"] == tool_outputs[i]["output"]
            
            # Check completion event
            complete_events = [e for e in events_captured if e.event_type == EventType.ACTION_COMPLETE]
            assert len(complete_events) >= 1
            
            complete_event = complete_events[0]
            assert complete_event.component == "function"
            assert complete_event.operation == "submission_complete"
            assert complete_event.data["outputs_count"] == 2
    
    def test_function_call_handling(self, assistant_monitor):
        """Test handling function calls that require action."""
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Create mock run with required action
        tool_calls = [
            MockToolCall("call_123", MockFunction("get_weather", '{"location": "NYC"}')),
            MockToolCall("call_456", MockFunction("get_time", '{"timezone": "UTC"}'))
        ]
        required_action = MockRequiredAction(MockSubmitToolOutputs(tool_calls))
        run = MockRun(status="requires_action", required_action=required_action)
        
        # Mock the event queuing
        with patch.object(assistant_monitor, '_queue_event_safe', side_effect=capture_event):
            # Call the function call handler
            assistant_monitor._handle_function_calls(run, "thread_123")
            
            # Verify function call events were captured
            call_events = [e for e in events_captured if e.event_type == EventType.TOOL_CALL]
            assert len(call_events) == 2
            
            for i, event in enumerate(call_events):
                assert event.component == "function"
                assert event.operation == "call"
                assert event.data["call_id"] == tool_calls[i].id
                assert event.data["function_name"] == tool_calls[i].function.name
                assert "tool_usage" in event.metadata
    
    def test_assistant_summary_generation(self, assistant_monitor, mock_client):
        """Test assistant summary generation."""
        # Mock event queuing
        with patch.object(assistant_monitor, '_queue_event_safe'):
            # Simulate some interactions
            assistant_monitor.monitor_message_creation(
                mock_client.beta.threads.messages.create,
                mock_client,
                thread_id="thread_123",
                role="user",
                content="Hello"
            )
            
            assistant_monitor.monitor_message_creation(
                mock_client.beta.threads.messages.create,
                mock_client,
                thread_id="thread_456",
                role="assistant",
                content="Hi there!"
            )
            
            # Simulate function calls
            tool_outputs = [{"tool_call_id": "call_123", "output": "Result"}]
            assistant_monitor.monitor_function_submission(
                mock_client.beta.threads.runs.submit_tool_outputs,
                mock_client,
                thread_id="thread_123",
                run_id="run_789",
                tool_outputs=tool_outputs
            )
        
        # Get summary
        summary = assistant_monitor.get_assistant_summary()
        
        assert summary["total_threads"] >= 2
        assert summary["total_messages"] >= 2
        assert summary["total_function_calls"] >= 1
        assert "thread_123" in summary["active_threads"]
        assert "thread_456" in summary["active_threads"]
    
    def test_error_handling_in_run_creation(self, assistant_monitor, mock_client):
        """Test error handling during run creation."""
        # Create a run creation method that raises an exception
        def failing_create(**kwargs):
            raise ValueError("Invalid assistant ID")
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(assistant_monitor, '_queue_event_safe', side_effect=capture_event):
            # Call monitor method and expect exception
            with pytest.raises(ValueError):
                assistant_monitor.monitor_run_creation(
                    failing_create,
                    mock_client,
                    thread_id="thread_123",
                    assistant_id="invalid_asst"
                )
            
            # Verify error event was captured
            error_events = [e for e in events_captured if e.severity == EventSeverity.ERROR]
            assert len(error_events) >= 1
            
            error_event = error_events[0]
            assert error_event.event_type == EventType.AGENT_ERROR
            assert error_event.component == "assistant"
            assert error_event.operation == "run_error"
            assert error_event.error_type == "ValueError"
            assert error_event.error_message == "Invalid assistant ID"


class TestOpenAIIntegration:
    """Integration tests with mock OpenAI workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_assistant_workflow(self, mock_instrumentor, mock_client):
        """Test a complete assistant interaction workflow."""
        agent_id = "integration_test"
        config = {
            "clients": [mock_client],
            "assistant_ids": ["asst_123"],
            "monitor_functions": True,
            "monitor_reasoning": True
        }
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        monitor = mock_instrumentor.get_assistant_monitor(session_id)
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock event queuing to capture events
        with patch.object(monitor, '_queue_event_safe', side_effect=capture_event):
            # Simulate complete assistant workflow
            
            # 1. Create user message
            monitor.monitor_message_creation(
                mock_client.beta.threads.messages.create,
                mock_client,
                thread_id="thread_123",
                role="user",
                content="What's the weather like in New York?"
            )
            
            # 2. Create assistant run
            monitor.monitor_run_creation(
                mock_client.beta.threads.runs.create,
                mock_client,
                thread_id="thread_123",
                assistant_id="asst_123",
                instructions="You are a helpful weather assistant"
            )
            
            # 3. Poll run status (requires action for function call)
            run_with_action = MockRun(
                status="requires_action",
                required_action=MockRequiredAction(
                    MockSubmitToolOutputs([
                        MockToolCall("call_123", MockFunction("get_weather", '{"location": "New York"}'))
                    ])
                )
            )
            
            def mock_retrieve(thread_id, run_id):
                return run_with_action
            
            monitor.monitor_run_polling(mock_retrieve, mock_client, "thread_123", "run_123")
            
            # 4. Submit function results
            tool_outputs = [{"tool_call_id": "call_123", "output": "Sunny, 25°C"}]
            monitor.monitor_function_submission(
                mock_client.beta.threads.runs.submit_tool_outputs,
                mock_client,
                thread_id="thread_123",
                run_id="run_123",
                tool_outputs=tool_outputs
            )
            
            # 5. Poll run completion
            completed_run = MockRun(status="completed", usage={"total_tokens": 150})
            
            def mock_retrieve_completed(thread_id, run_id):
                return completed_run
            
            monitor.monitor_run_polling(mock_retrieve_completed, mock_client, "thread_123", "run_123")
            
            # 6. Create assistant response message
            monitor.monitor_message_creation(
                mock_client.beta.threads.messages.create,
                mock_client,
                thread_id="thread_123",
                role="assistant",
                content="The weather in New York is sunny with a temperature of 25°C."
            )
        
        # Verify workflow events were captured
        assert len(events_captured) >= 10  # Multiple events for each step
        
        # Check for different event types
        event_types = [event.event_type for event in events_captured]
        assert EventType.MESSAGE_SEND in event_types      # User message
        assert EventType.MESSAGE_RECEIVE in event_types   # Assistant message
        assert EventType.AGENT_START in event_types       # Run creation
        assert EventType.TASK_START in event_types        # Run created
        assert EventType.TOOL_CALL in event_types         # Function call
        assert EventType.TOOL_RESPONSE in event_types     # Function result
        assert EventType.ACTION_COMPLETE in event_types   # Function submission
        assert EventType.AGENT_STOP in event_types        # Run completion
        
        # Verify function call workflow
        function_events = [e for e in events_captured if e.component == "function"]
        assert len(function_events) >= 3  # Call, result, submission complete
        
        # Verify thread conversation tracking
        message_events = [e for e in events_captured if e.component == "thread"]
        assert len(message_events) >= 2  # User and assistant messages
        
        # Get assistant summary
        summary = monitor.get_assistant_summary()
        assert summary["total_threads"] >= 1
        assert summary["total_messages"] >= 2
        assert summary["total_function_calls"] >= 1
        
        # Stop monitoring
        monitoring_summary = await mock_instrumentor.stop_monitoring(session_id)
        assert "assistant_summary" in monitoring_summary.performance_metrics
    
    @pytest.mark.asyncio
    async def test_reasoning_trace_extraction(self, mock_instrumentor, mock_client):
        """Test reasoning trace extraction from assistant messages."""
        agent_id = "reasoning_test"
        config = {"clients": [mock_client], "monitor_reasoning": True}
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        monitor = mock_instrumentor.get_assistant_monitor(session_id)
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock event queuing to capture events
        with patch.object(monitor, '_queue_event_safe', side_effect=capture_event):
            # Create messages with reasoning patterns
            reasoning_messages = [
                "Let me think about this step by step. First, I need to understand the problem.",
                "I need to analyze the data because it contains important patterns.",
                "Therefore, considering all the factors, the best approach would be...",
                "However, we should also consider the alternative reasoning that...",
                "My analysis shows that the conclusion is supported by the evidence."
            ]
            
            for i, content in enumerate(reasoning_messages):
                monitor.monitor_message_creation(
                    mock_client.beta.threads.messages.create,
                    mock_client,
                    thread_id="thread_reasoning",
                    role="assistant",
                    content=content
                )
            
            # Simulate run completion to trigger reasoning extraction
            completed_run = MockRun(status="completed")
            
            def mock_retrieve_completed(thread_id, run_id):
                return completed_run
            
            monitor.monitor_run_polling(mock_retrieve_completed, mock_client, "thread_reasoning", "run_reasoning")
        
        # Verify reasoning trace events were captured
        reasoning_events = [e for e in events_captured 
                           if "reasoning" in e.tags or e.component == "reasoning"]
        
        assert len(reasoning_events) >= 1
        
        # Check reasoning trace data
        for event in reasoning_events:
            if event.component == "reasoning":
                assert "reasoning_patterns" in event.data
                assert "patterns_count" in event.data
                assert event.data["patterns_count"] > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, mock_instrumentor, mock_client):
        """Test error handling and recovery in assistant workflows."""
        agent_id = "error_recovery_test"
        config = {"clients": [mock_client]}
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        monitor = mock_instrumentor.get_assistant_monitor(session_id)
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock event queuing to capture events
        with patch.object(monitor, '_queue_event_safe', side_effect=capture_event):
            # Simulate successful message creation
            monitor.monitor_message_creation(
                mock_client.beta.threads.messages.create,
                mock_client,
                thread_id="thread_123",
                role="user",
                content="Test message"
            )
            
            # Simulate failed run creation
            def failing_create(**kwargs):
                raise ConnectionError("API connection failed")
            
            with pytest.raises(ConnectionError):
                monitor.monitor_run_creation(
                    failing_create,
                    mock_client,
                    thread_id="thread_123",
                    assistant_id="asst_123"
                )
            
            # Simulate successful recovery with new run
            monitor.monitor_run_creation(
                mock_client.beta.threads.runs.create,
                mock_client,
                thread_id="thread_123",
                assistant_id="asst_123",
                instructions="Retry with recovery"
            )
            
            # Simulate failed run (expired)
            failed_run = MockRun(status="expired", last_error={"code": "timeout", "message": "Run expired"})
            
            def mock_retrieve_failed(thread_id, run_id):
                return failed_run
            
            monitor.monitor_run_polling(mock_retrieve_failed, mock_client, "thread_123", "run_failed")
        
        # Verify error and recovery events
        error_events = [e for e in events_captured if e.severity == EventSeverity.ERROR]
        success_events = [e for e in events_captured 
                         if e.event_type in [EventType.MESSAGE_SEND, EventType.TASK_START]]
        
        assert len(error_events) >= 1  # At least one failure
        assert len(success_events) >= 2  # At least two successes
        
        # Verify assistant summary reflects mixed results
        summary = monitor.get_assistant_summary()
        assert summary["total_messages"] >= 1
        assert summary["total_threads"] >= 1


if __name__ == "__main__":
    pytest.main([__file__])