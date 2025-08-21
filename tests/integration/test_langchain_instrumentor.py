"""
Integration tests for LangChain instrumentor.

These tests verify the integration with LangChain's callback system
and proper event capture during chain execution.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from escai_framework.instrumentation.langchain_instrumentor import (
    LangChainInstrumentor, 
    LangChainCallbackHandler,
    LANGCHAIN_AVAILABLE
)
from escai_framework.instrumentation.events import EventType, EventSeverity, AgentEvent


# Mock LangChain classes for testing when LangChain is not available
class MockLLMResult:
    def __init__(self, generations=None, llm_output=None):
        self.generations = generations or []
        self.llm_output = llm_output or {}


class MockAgentAction:
    def __init__(self, tool="test_tool", tool_input=None, log=""):
        self.tool = tool
        self.tool_input = tool_input or {}
        self.log = log


class MockAgentFinish:
    def __init__(self, return_values=None, log=""):
        self.return_values = return_values or {}
        self.log = log


@pytest.fixture
def instrumentor():
    """Create a LangChain instrumentor for testing."""
    if not LANGCHAIN_AVAILABLE:
        pytest.skip("LangChain not available")
    
    return LangChainInstrumentor()


@pytest.fixture
def mock_instrumentor():
    """Create a mock LangChain instrumentor for testing without LangChain."""
    with patch('escai_framework.instrumentation.langchain_instrumentor.LANGCHAIN_AVAILABLE', True):
        return LangChainInstrumentor()


@pytest.fixture
def callback_handler(mock_instrumentor):
    """Create a callback handler for testing."""
    return LangChainCallbackHandler(
        instrumentor=mock_instrumentor,
        session_id="test_session",
        agent_id="test_agent"
    )


class TestLangChainInstrumentor:
    """Test cases for LangChain instrumentor."""
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, mock_instrumentor):
        """Test starting monitoring for a LangChain agent."""
        agent_id = "test_agent"
        config = {
            "monitor_memory": True,
            "monitor_context": True
        }
        
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        assert session_id is not None
        assert len(session_id) > 0
        
        # Verify session was created
        session = mock_instrumentor._get_session(session_id)
        assert session is not None
        assert session.agent_id == agent_id
        assert session.framework == "langchain"
        
        # Verify callback handler was created
        callback_handler = mock_instrumentor.get_callback_handler(session_id)
        assert callback_handler is not None
        assert callback_handler.agent_id == agent_id
        assert callback_handler.session_id == session_id
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, mock_instrumentor):
        """Test stopping monitoring for a LangChain agent."""
        agent_id = "test_agent"
        config = {"monitor_memory": True}
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        # Add some test events
        test_event = AgentEvent(
            event_type=EventType.TASK_START,
            agent_id=agent_id,
            session_id=session_id,
            message="Test event",
            framework="langchain"
        )
        await mock_instrumentor.capture_event(test_event)
        
        # Stop monitoring
        summary = await mock_instrumentor.stop_monitoring(session_id)
        
        assert summary is not None
        assert summary.session_id == session_id
        assert summary.agent_id == agent_id
        assert summary.framework == "langchain"
        assert summary.total_events >= 1  # At least the test event
        
        # Verify callback handler was removed
        callback_handler = mock_instrumentor.get_callback_handler(session_id)
        assert callback_handler is None
    
    @pytest.mark.asyncio
    async def test_capture_event(self, mock_instrumentor):
        """Test capturing events."""
        agent_id = "test_agent"
        config = {}
        
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        # Create test event
        event = AgentEvent(
            event_type=EventType.DECISION_START,
            agent_id=agent_id,
            session_id=session_id,
            message="Test decision",
            framework="langchain",
            component="llm",
            operation="generate"
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
        assert EventType.TASK_START in supported_events
        assert EventType.TASK_COMPLETE in supported_events
        assert EventType.DECISION_START in supported_events
        assert EventType.DECISION_COMPLETE in supported_events
        assert EventType.TOOL_CALL in supported_events
        assert EventType.TOOL_RESPONSE in supported_events
        assert EventType.ACTION_START in supported_events
        assert EventType.ACTION_COMPLETE in supported_events
    
    def test_get_framework_name(self, mock_instrumentor):
        """Test getting framework name."""
        assert mock_instrumentor.get_framework_name() == "langchain"


class TestLangChainCallbackHandler:
    """Test cases for LangChain callback handler."""
    
    def test_on_chain_start(self, callback_handler):
        """Test chain start callback."""
        serialized = {"name": "test_chain"}
        inputs = {"input": "test input"}
        run_id = "test_run_123"
        
        # Mock the event queuing
        with patch.object(callback_handler, '_queue_event_safe') as mock_queue:
            callback_handler.on_chain_start(serialized, inputs, run_id=run_id)
            
            # Verify event was queued
            mock_queue.assert_called_once()
            event = mock_queue.call_args[0][0]
            
            assert event.event_type == EventType.TASK_START
            assert event.component == "chain"
            assert event.operation == "start"
            assert event.data["chain_type"] == "test_chain"
            assert event.data["inputs"] == inputs
            assert event.correlation_id == run_id
    
    def test_on_chain_end(self, callback_handler):
        """Test chain end callback."""
        # First start a chain to set up timing
        serialized = {"name": "test_chain"}
        inputs = {"input": "test input"}
        run_id = "test_run_123"
        
        with patch.object(callback_handler, '_queue_event_safe'):
            callback_handler.on_chain_start(serialized, inputs, run_id=run_id)
        
        # Now end the chain
        outputs = {"output": "test output"}
        
        with patch.object(callback_handler, '_queue_event_safe') as mock_queue:
            callback_handler.on_chain_end(outputs, run_id=run_id)
            
            # Verify event was queued
            mock_queue.assert_called_once()
            event = mock_queue.call_args[0][0]
            
            assert event.event_type == EventType.TASK_COMPLETE
            assert event.component == "chain"
            assert event.operation == "complete"
            assert event.data["outputs"] == outputs
            assert event.correlation_id == run_id
            assert event.duration_ms is not None
    
    def test_on_chain_error(self, callback_handler):
        """Test chain error callback."""
        # First start a chain
        serialized = {"name": "test_chain"}
        inputs = {"input": "test input"}
        run_id = "test_run_123"
        
        with patch.object(callback_handler, '_queue_event_safe'):
            callback_handler.on_chain_start(serialized, inputs, run_id=run_id)
        
        # Now trigger an error
        error = ValueError("Test error")
        
        with patch.object(callback_handler, '_queue_event_safe') as mock_queue:
            callback_handler.on_chain_error(error, run_id=run_id)
            
            # Verify error event was queued
            mock_queue.assert_called_once()
            event = mock_queue.call_args[0][0]
            
            assert event.event_type == EventType.TASK_FAIL
            assert event.component == "chain"
            assert event.operation == "error"
            assert event.severity == EventSeverity.ERROR
            assert event.error_type == "ValueError"
            assert event.error_message == "Test error"
            assert event.correlation_id == run_id
    
    def test_on_llm_start(self, callback_handler):
        """Test LLM start callback."""
        serialized = {"name": "test_llm"}
        prompts = ["What is the capital of France?"]
        run_id = "llm_run_123"
        
        with patch.object(callback_handler, '_queue_event_safe') as mock_queue:
            callback_handler.on_llm_start(serialized, prompts, run_id=run_id)
            
            # Verify event was queued
            mock_queue.assert_called_once()
            event = mock_queue.call_args[0][0]
            
            assert event.event_type == EventType.DECISION_START
            assert event.component == "llm"
            assert event.operation == "generate"
            assert event.data["llm_type"] == "test_llm"
            assert event.data["prompts"] == prompts
            assert event.data["prompt_count"] == 1
            assert event.correlation_id == run_id
    
    def test_on_llm_end(self, callback_handler):
        """Test LLM end callback."""
        # First start LLM
        serialized = {"name": "test_llm"}
        prompts = ["Test prompt"]
        run_id = "llm_run_123"
        
        with patch.object(callback_handler, '_queue_event_safe'):
            callback_handler.on_llm_start(serialized, prompts, run_id=run_id)
        
        # Now end LLM
        response = MockLLMResult(
            generations=[{"text": "Paris"}],
            llm_output={"token_usage": {"total_tokens": 10}}
        )
        
        with patch.object(callback_handler, '_queue_event_safe') as mock_queue:
            callback_handler.on_llm_end(response, run_id=run_id)
            
            # Verify event was queued
            mock_queue.assert_called_once()
            event = mock_queue.call_args[0][0]
            
            assert event.event_type == EventType.DECISION_COMPLETE
            assert event.component == "llm"
            assert event.operation == "complete"
            assert event.data["generations_count"] == 1
            assert "token_usage" in event.data
            assert event.correlation_id == run_id
            assert event.duration_ms is not None
    
    def test_on_tool_start(self, callback_handler):
        """Test tool start callback."""
        serialized = {"name": "calculator"}
        input_str = "2 + 2"
        run_id = "tool_run_123"
        
        with patch.object(callback_handler, '_queue_event_safe') as mock_queue:
            callback_handler.on_tool_start(serialized, input_str, run_id=run_id)
            
            # Verify event was queued
            mock_queue.assert_called_once()
            event = mock_queue.call_args[0][0]
            
            assert event.event_type == EventType.TOOL_CALL
            assert event.component == "tool"
            assert event.operation == "start"
            assert event.data["tool_name"] == "calculator"
            assert event.data["input"] == input_str
            assert event.correlation_id == run_id
    
    def test_on_tool_end(self, callback_handler):
        """Test tool end callback."""
        # First start tool
        serialized = {"name": "calculator"}
        input_str = "2 + 2"
        run_id = "tool_run_123"
        
        with patch.object(callback_handler, '_queue_event_safe'):
            callback_handler.on_tool_start(serialized, input_str, run_id=run_id)
        
        # Now end tool
        output = "4"
        
        with patch.object(callback_handler, '_queue_event_safe') as mock_queue:
            callback_handler.on_tool_end(output, run_id=run_id)
            
            # Verify event was queued
            mock_queue.assert_called_once()
            event = mock_queue.call_args[0][0]
            
            assert event.event_type == EventType.TOOL_RESPONSE
            assert event.component == "tool"
            assert event.operation == "complete"
            assert event.data["output"] == output
            assert event.data["output_length"] == 1
            assert event.correlation_id == run_id
            assert event.duration_ms is not None
    
    def test_on_agent_action(self, callback_handler):
        """Test agent action callback."""
        action = MockAgentAction(
            tool="search",
            tool_input={"query": "test"},
            log="Thought: I need to search for information"
        )
        run_id = "agent_run_123"
        
        with patch.object(callback_handler, '_queue_event_safe') as mock_queue:
            callback_handler.on_agent_action(action, run_id=run_id)
            
            # Verify event was queued
            mock_queue.assert_called_once()
            event = mock_queue.call_args[0][0]
            
            assert event.event_type == EventType.ACTION_START
            assert event.component == "agent"
            assert event.operation == "action"
            assert event.data["tool"] == "search"
            assert event.data["tool_input"] == {"query": "test"}
            assert event.correlation_id == run_id
            assert "reasoning_trace" in event.metadata
    
    def test_on_agent_finish(self, callback_handler):
        """Test agent finish callback."""
        finish = MockAgentFinish(
            return_values={"answer": "The answer is 42"},
            log="Final Answer: The answer is 42"
        )
        run_id = "agent_run_123"
        
        with patch.object(callback_handler, '_queue_event_safe') as mock_queue:
            callback_handler.on_agent_finish(finish, run_id=run_id)
            
            # Verify event was queued
            mock_queue.assert_called_once()
            event = mock_queue.call_args[0][0]
            
            assert event.event_type == EventType.ACTION_COMPLETE
            assert event.component == "agent"
            assert event.operation == "finish"
            assert event.data["return_values"] == {"answer": "The answer is 42"}
            assert event.correlation_id == run_id
            assert "reasoning_trace" in event.metadata
    
    def test_on_text_with_reasoning(self, callback_handler):
        """Test text callback with reasoning patterns."""
        text = "Thought: I need to calculate 2+2\nAction: calculator\nAction Input: 2+2"
        run_id = "text_run_123"
        
        with patch.object(callback_handler, '_queue_event_safe') as mock_queue:
            callback_handler.on_text(text, run_id=run_id)
            
            # Verify event was queued
            mock_queue.assert_called_once()
            event = mock_queue.call_args[0][0]
            
            assert event.event_type == EventType.CUSTOM
            assert event.component == "agent"
            assert event.operation == "text"
            assert event.data["text"] == text
            assert "reasoning" in event.tags
            assert "reasoning_trace" in event.metadata
            assert event.correlation_id == run_id


class TestLangChainIntegration:
    """Integration tests with mock LangChain workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_chain_workflow(self, mock_instrumentor):
        """Test a complete chain execution workflow."""
        agent_id = "integration_test_agent"
        config = {"monitor_memory": True, "monitor_context": True}
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        callback_handler = mock_instrumentor.get_callback_handler(session_id)
        
        # Simulate chain execution
        chain_run_id = "chain_123"
        llm_run_id = "llm_456"
        tool_run_id = "tool_789"
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock event queuing to capture events
        with patch.object(callback_handler, '_queue_event_safe', side_effect=capture_event):
            # Chain starts
            callback_handler.on_chain_start(
                {"name": "ReActChain"}, 
                {"input": "What is 2+2?"}, 
                run_id=chain_run_id
            )
            
            # LLM starts
            callback_handler.on_llm_start(
                {"name": "OpenAI"}, 
                ["Human: What is 2+2?\nAI:"], 
                run_id=llm_run_id, 
                parent_run_id=chain_run_id
            )
            
            # LLM ends
            callback_handler.on_llm_end(
                MockLLMResult(
                    generations=[{"text": "I need to calculate this"}],
                    llm_output={"token_usage": {"total_tokens": 15}}
                ),
                run_id=llm_run_id,
                parent_run_id=chain_run_id
            )
            
            # Agent action
            callback_handler.on_agent_action(
                MockAgentAction(
                    tool="calculator",
                    tool_input={"expression": "2+2"},
                    log="Thought: I need to use calculator"
                ),
                run_id=chain_run_id
            )
            
            # Tool starts
            callback_handler.on_tool_start(
                {"name": "calculator"},
                "2+2",
                run_id=tool_run_id,
                parent_run_id=chain_run_id
            )
            
            # Tool ends
            callback_handler.on_tool_end(
                "4",
                run_id=tool_run_id,
                parent_run_id=chain_run_id
            )
            
            # Agent finishes
            callback_handler.on_agent_finish(
                MockAgentFinish(
                    return_values={"output": "The answer is 4"},
                    log="Final Answer: The answer is 4"
                ),
                run_id=chain_run_id
            )
            
            # Chain ends
            callback_handler.on_chain_end(
                {"output": "The answer is 4"},
                run_id=chain_run_id
            )
        
        # Verify events were captured in correct order
        assert len(events_captured) >= 7  # At least one event for each callback
        
        # Check event types and relationships
        event_types = [event.event_type for event in events_captured]
        assert EventType.TASK_START in event_types  # Chain start
        assert EventType.DECISION_START in event_types  # LLM start
        assert EventType.DECISION_COMPLETE in event_types  # LLM end
        assert EventType.ACTION_START in event_types  # Agent action
        assert EventType.TOOL_CALL in event_types  # Tool start
        assert EventType.TOOL_RESPONSE in event_types  # Tool end
        assert EventType.ACTION_COMPLETE in event_types  # Agent finish
        assert EventType.TASK_COMPLETE in event_types  # Chain end
        
        # Verify parent-child relationships
        chain_events = [e for e in events_captured if e.correlation_id == chain_run_id]
        llm_events = [e for e in events_captured if e.correlation_id == llm_run_id]
        tool_events = [e for e in events_captured if e.correlation_id == tool_run_id]
        
        assert len(chain_events) >= 3  # Chain start, agent action, agent finish, chain end
        assert len(llm_events) >= 2  # LLM start, LLM end
        assert len(tool_events) >= 2  # Tool start, Tool end
        
        # Stop monitoring
        summary = await mock_instrumentor.stop_monitoring(session_id)
        # Note: In this test, events are captured by mock but not processed through capture_event
        # so we verify the events were captured correctly instead
        assert len(events_captured) >= 7
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, mock_instrumentor):
        """Test error handling in chain execution."""
        agent_id = "error_test_agent"
        config = {}
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        callback_handler = mock_instrumentor.get_callback_handler(session_id)
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock event queuing to capture events
        with patch.object(callback_handler, '_queue_event_safe', side_effect=capture_event):
            # Chain starts
            callback_handler.on_chain_start(
                {"name": "TestChain"}, 
                {"input": "test"}, 
                run_id="chain_error"
            )
            
            # LLM error
            callback_handler.on_llm_error(
                ValueError("API rate limit exceeded"),
                run_id="llm_error",
                parent_run_id="chain_error"
            )
            
            # Tool error
            callback_handler.on_tool_error(
                ConnectionError("Network timeout"),
                run_id="tool_error",
                parent_run_id="chain_error"
            )
            
            # Chain error
            callback_handler.on_chain_error(
                RuntimeError("Chain execution failed"),
                run_id="chain_error"
            )
        
        # Verify error events were captured
        error_events = [e for e in events_captured if e.severity == EventSeverity.ERROR]
        assert len(error_events) == 3  # LLM error, tool error, chain error
        
        # Verify error information is captured
        for event in error_events:
            assert event.error_type is not None
            assert event.error_message is not None
            assert event.stack_trace is not None
        
        # Stop monitoring
        summary = await mock_instrumentor.stop_monitoring(session_id)
        # Note: In this test, events are captured by mock but not processed through capture_event
        # so we verify the error events were captured correctly instead
        assert len(error_events) == 3


if __name__ == "__main__":
    pytest.main([__file__])