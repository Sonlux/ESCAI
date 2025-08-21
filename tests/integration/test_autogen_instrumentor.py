"""
Integration tests for AutoGen instrumentor.

These tests verify the integration with AutoGen's multi-agent system
and proper event capture during agent conversations.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from escai_framework.instrumentation.autogen_instrumentor import (
    AutoGenInstrumentor, 
    AutoGenMessageInterceptor,
    AUTOGEN_AVAILABLE
)
from escai_framework.instrumentation.events import EventType, EventSeverity, AgentEvent


# Mock AutoGen classes for testing when AutoGen is not available
class MockConversableAgent:
    def __init__(self, name="test_agent", system_message=""):
        self.name = name
        self.system_message = system_message
        self._send_calls = []
        self._receive_calls = []
    
    def send(self, recipient, message, request_reply=True, silent=False):
        self._send_calls.append({
            "recipient": recipient,
            "message": message,
            "request_reply": request_reply,
            "silent": silent
        })
        return f"Response from {recipient.name if hasattr(recipient, 'name') else 'unknown'}"
    
    def receive(self, message, sender):
        self._receive_calls.append({
            "message": message,
            "sender": sender
        })
        return f"Processed message from {sender.name if hasattr(sender, 'name') else 'unknown'}"


class MockGroupChat:
    def __init__(self, agents=None):
        self.agents = agents or []
        self._select_calls = []
    
    def select_speaker_msg(self, messages):
        self._select_calls.append(messages)
        # Return first agent as default
        return self.agents[0] if self.agents else None


@pytest.fixture
def instrumentor():
    """Create an AutoGen instrumentor for testing."""
    if not AUTOGEN_AVAILABLE:
        pytest.skip("AutoGen not available")
    
    return AutoGenInstrumentor()


@pytest.fixture
def mock_instrumentor():
    """Create a mock AutoGen instrumentor for testing without AutoGen."""
    with patch('escai_framework.instrumentation.autogen_instrumentor.AUTOGEN_AVAILABLE', True):
        return AutoGenInstrumentor()


@pytest.fixture
def message_interceptor(mock_instrumentor):
    """Create a message interceptor for testing."""
    return AutoGenMessageInterceptor(
        instrumentor=mock_instrumentor,
        session_id="test_session",
        agent_id="test_agent"
    )


@pytest.fixture
def mock_agents():
    """Create mock AutoGen agents for testing."""
    agent1 = MockConversableAgent("agent1", "You are a helpful assistant")
    agent2 = MockConversableAgent("agent2", "You are a code reviewer")
    agent3 = MockConversableAgent("agent3", "You are a project manager")
    return [agent1, agent2, agent3]


@pytest.fixture
def mock_group_chat(mock_agents):
    """Create a mock group chat for testing."""
    return MockGroupChat(mock_agents)


class TestAutoGenInstrumentor:
    """Test cases for AutoGen instrumentor."""
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, mock_instrumentor, mock_agents):
        """Test starting monitoring for AutoGen agents."""
        agent_id = "test_multi_agent_system"
        config = {
            "agents": mock_agents,
            "monitor_conversations": True,
            "monitor_decisions": True
        }
        
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        assert session_id is not None
        assert len(session_id) > 0
        
        # Verify session was created
        session = mock_instrumentor._get_session(session_id)
        assert session is not None
        assert session.agent_id == agent_id
        assert session.framework == "autogen"
        
        # Verify message interceptor was created
        interceptor = mock_instrumentor.get_message_interceptor(session_id)
        assert interceptor is not None
        assert interceptor.agent_id == agent_id
        assert interceptor.session_id == session_id
        
        # Verify agents are being tracked
        assert session_id in mock_instrumentor._monitored_agents
        assert len(mock_instrumentor._monitored_agents[session_id]) == len(mock_agents)
    
    @pytest.mark.asyncio
    async def test_start_monitoring_with_group_chat(self, mock_instrumentor, mock_group_chat):
        """Test starting monitoring with group chat."""
        agent_id = "test_group_chat_system"
        config = {
            "group_chats": [mock_group_chat],
            "monitor_conversations": True,
            "monitor_decisions": True
        }
        
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        # Verify group chat is being tracked
        assert session_id in mock_instrumentor._group_chats
        assert len(mock_instrumentor._group_chats[session_id]) == 1
        
        # Verify agents from group chat are being tracked
        assert session_id in mock_instrumentor._monitored_agents
        assert len(mock_instrumentor._monitored_agents[session_id]) == len(mock_group_chat.agents)
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, mock_instrumentor, mock_agents):
        """Test stopping monitoring for AutoGen agents."""
        agent_id = "test_agent"
        config = {"agents": mock_agents}
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        # Add some test events
        test_event = AgentEvent(
            event_type=EventType.MESSAGE_SEND,
            agent_id=agent_id,
            session_id=session_id,
            message="Test message",
            framework="autogen"
        )
        await mock_instrumentor.capture_event(test_event)
        
        # Stop monitoring
        summary = await mock_instrumentor.stop_monitoring(session_id)
        
        assert summary is not None
        assert summary.session_id == session_id
        assert summary.agent_id == agent_id
        assert summary.framework == "autogen"
        assert summary.total_events >= 1  # At least the test event
        
        # Verify cleanup
        interceptor = mock_instrumentor.get_message_interceptor(session_id)
        assert interceptor is None
        assert session_id not in mock_instrumentor._monitored_agents
        assert session_id not in mock_instrumentor._group_chats
    
    @pytest.mark.asyncio
    async def test_capture_event(self, mock_instrumentor, mock_agents):
        """Test capturing events."""
        agent_id = "test_agent"
        config = {"agents": mock_agents}
        
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        # Create test event
        event = AgentEvent(
            event_type=EventType.MESSAGE_SEND,
            agent_id=agent_id,
            session_id=session_id,
            message="Test conversation",
            framework="autogen",
            component="agent",
            operation="send_message",
            data={
                "sender": "agent1",
                "recipient": "agent2",
                "message_content": "Hello, agent2!"
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
        assert EventType.DECISION_START in supported_events
        assert EventType.DECISION_COMPLETE in supported_events
        assert EventType.MESSAGE_SEND in supported_events
        assert EventType.MESSAGE_RECEIVE in supported_events
        assert EventType.ACTION_START in supported_events
        assert EventType.ACTION_COMPLETE in supported_events
    
    def test_get_framework_name(self, mock_instrumentor):
        """Test getting framework name."""
        assert mock_instrumentor.get_framework_name() == "autogen"


class TestAutoGenMessageInterceptor:
    """Test cases for AutoGen message interceptor."""
    
    def test_intercept_send_message(self, message_interceptor, mock_agents):
        """Test intercepting agent message sending."""
        sender = mock_agents[0]
        recipient = mock_agents[1]
        message = "Hello, how are you?"
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(message_interceptor, '_queue_event_safe', side_effect=capture_event):
            # Call intercept method
            result = message_interceptor.intercept_send_message(
                sender.send, sender, recipient, message, request_reply=True, silent=False
            )
            
            # Verify original method was called
            assert len(sender._send_calls) == 1
            assert sender._send_calls[0]["recipient"] == recipient
            assert sender._send_calls[0]["message"] == message
            
            # Verify events were captured
            assert len(events_captured) >= 2  # Send event and delivery event
            
            send_event = events_captured[0]
            assert send_event.event_type == EventType.MESSAGE_SEND
            assert send_event.component == "agent"
            assert send_event.operation == "send_message"
            assert send_event.data["sender"] == "agent1"
            assert send_event.data["recipient"] == "agent2"
            assert send_event.data["message_content"] == message
            
            delivery_event = events_captured[1]
            assert delivery_event.event_type == EventType.MESSAGE_RECEIVE
            assert delivery_event.operation == "message_delivered"
    
    def test_intercept_receive_message(self, message_interceptor, mock_agents):
        """Test intercepting agent message receiving."""
        agent = mock_agents[0]
        sender = mock_agents[1]
        message = "Here's the information you requested"
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(message_interceptor, '_queue_event_safe', side_effect=capture_event):
            # Call intercept method
            result = message_interceptor.intercept_receive_message(
                agent.receive, agent, message, sender
            )
            
            # Verify original method was called
            assert len(agent._receive_calls) == 1
            assert agent._receive_calls[0]["message"] == message
            assert agent._receive_calls[0]["sender"] == sender
            
            # Verify events were captured
            assert len(events_captured) >= 2  # Receive event and process event
            
            receive_event = events_captured[0]
            assert receive_event.event_type == EventType.MESSAGE_RECEIVE
            assert receive_event.component == "agent"
            assert receive_event.operation == "receive_message"
            assert receive_event.data["agent"] == "agent1"
            assert receive_event.data["sender"] == "agent2"
            assert receive_event.data["message_content"] == message
            
            process_event = events_captured[1]
            assert process_event.event_type == EventType.DECISION_COMPLETE
            assert process_event.operation == "process_message"
    
    def test_intercept_group_chat_select_speaker(self, message_interceptor, mock_group_chat):
        """Test intercepting group chat speaker selection."""
        messages = [
            {"role": "user", "content": "What should we do next?"},
            {"role": "assistant", "content": "Let me think about this..."}
        ]
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(message_interceptor, '_queue_event_safe', side_effect=capture_event):
            # Call intercept method
            selected_speaker = message_interceptor.intercept_group_chat_select_speaker(
                mock_group_chat.select_speaker_msg, mock_group_chat, messages
            )
            
            # Verify original method was called
            assert len(mock_group_chat._select_calls) == 1
            assert mock_group_chat._select_calls[0] == messages
            
            # Verify speaker was selected
            assert selected_speaker is not None
            assert selected_speaker.name == "agent1"  # First agent in mock
            
            # Verify events were captured
            assert len(events_captured) >= 2  # Decision start and complete
            
            start_event = events_captured[0]
            assert start_event.event_type == EventType.DECISION_START
            assert start_event.component == "group_chat"
            assert start_event.operation == "select_speaker"
            assert start_event.data["message_count"] == len(messages)
            
            complete_event = events_captured[1]
            assert complete_event.event_type == EventType.DECISION_COMPLETE
            assert complete_event.component == "group_chat"
            assert complete_event.operation == "speaker_selected"
            assert complete_event.data["selected_speaker"] == "agent1"
    
    def test_conversation_tracking(self, message_interceptor, mock_agents):
        """Test conversation history tracking."""
        sender = mock_agents[0]
        recipient = mock_agents[1]
        
        # Mock event queuing
        with patch.object(message_interceptor, '_queue_event_safe'):
            # Send a message
            message_interceptor.intercept_send_message(
                sender.send, sender, recipient, "Hello!", request_reply=True, silent=False
            )
            
            # Receive a message
            message_interceptor.intercept_receive_message(
                recipient.receive, recipient, "Hi there!", sender
            )
        
        # Check conversation summary
        summary = message_interceptor.get_conversation_summary()
        
        assert summary["total_messages"] >= 2
        assert summary["unique_agents"] >= 2
        assert "agent1" in summary["agent_roles"]
        assert "agent2" in summary["agent_roles"]
        assert summary["conversation_duration"] >= 0
    
    def test_error_handling_in_send(self, message_interceptor, mock_agents):
        """Test error handling during message sending."""
        sender = mock_agents[0]
        recipient = mock_agents[1]
        
        # Create a send method that raises an exception
        def failing_send(recipient, message, request_reply=True, silent=False):
            raise ConnectionError("Network timeout")
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(message_interceptor, '_queue_event_safe', side_effect=capture_event):
            # Call intercept method and expect exception
            with pytest.raises(ConnectionError):
                message_interceptor.intercept_send_message(
                    failing_send, sender, recipient, "Test message"
                )
            
            # Verify error event was captured
            error_events = [e for e in events_captured if e.severity == EventSeverity.ERROR]
            assert len(error_events) >= 1
            
            error_event = error_events[0]
            assert error_event.event_type == EventType.AGENT_ERROR
            assert error_event.operation == "send_error"
            assert error_event.error_type == "ConnectionError"
            assert error_event.error_message == "Network timeout"


class TestAutoGenIntegration:
    """Integration tests with mock AutoGen workflows."""
    
    @pytest.mark.asyncio
    async def test_multi_agent_conversation_workflow(self, mock_instrumentor, mock_agents):
        """Test a complete multi-agent conversation workflow."""
        agent_id = "conversation_test"
        config = {
            "agents": mock_agents,
            "monitor_conversations": True,
            "monitor_decisions": True
        }
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        interceptor = mock_instrumentor.get_message_interceptor(session_id)
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock event queuing to capture events
        with patch.object(interceptor, '_queue_event_safe', side_effect=capture_event):
            # Simulate multi-agent conversation
            agent1, agent2, agent3 = mock_agents
            
            # Agent 1 sends message to Agent 2
            interceptor.intercept_send_message(
                agent1.send, agent1, agent2, 
                "Can you review this code for me?", 
                request_reply=True, silent=False
            )
            
            # Agent 2 receives and processes the message
            interceptor.intercept_receive_message(
                agent2.receive, agent2, 
                "Can you review this code for me?", 
                agent1
            )
            
            # Agent 2 sends response to Agent 1
            interceptor.intercept_send_message(
                agent2.send, agent2, agent1,
                "Sure, I'll review it. The code looks good overall.",
                request_reply=False, silent=False
            )
            
            # Agent 1 receives the response
            interceptor.intercept_receive_message(
                agent1.receive, agent1,
                "Sure, I'll review it. The code looks good overall.",
                agent2
            )
            
            # Agent 1 notifies Agent 3 (project manager)
            interceptor.intercept_send_message(
                agent1.send, agent1, agent3,
                "Code review is complete. Ready for deployment.",
                request_reply=False, silent=False
            )
        
        # Verify conversation flow was captured
        assert len(events_captured) >= 10  # Multiple events per interaction
        
        # Check for different event types
        event_types = [event.event_type for event in events_captured]
        assert EventType.MESSAGE_SEND in event_types
        assert EventType.MESSAGE_RECEIVE in event_types
        assert EventType.DECISION_COMPLETE in event_types
        
        # Verify conversation participants
        send_events = [e for e in events_captured if e.event_type == EventType.MESSAGE_SEND]
        participants = set()
        for event in send_events:
            participants.add(event.data.get("sender", ""))
            participants.add(event.data.get("recipient", ""))
        
        assert "agent1" in participants
        assert "agent2" in participants
        assert "agent3" in participants
        
        # Get conversation summary
        summary = interceptor.get_conversation_summary()
        assert summary["total_messages"] >= 3  # At least 3 send operations
        assert summary["unique_agents"] >= 3
        
        # Stop monitoring
        monitoring_summary = await mock_instrumentor.stop_monitoring(session_id)
        # Note: In this test, events are captured by mock but not processed through capture_event
        # so we verify the events were captured correctly instead
        assert len(events_captured) >= 10
    
    @pytest.mark.asyncio
    async def test_group_chat_workflow(self, mock_instrumentor, mock_group_chat):
        """Test group chat decision-making workflow."""
        agent_id = "group_chat_test"
        config = {
            "group_chats": [mock_group_chat],
            "monitor_conversations": True,
            "monitor_decisions": True
        }
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        interceptor = mock_instrumentor.get_message_interceptor(session_id)
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock event queuing to capture events
        with patch.object(interceptor, '_queue_event_safe', side_effect=capture_event):
            # Simulate group chat speaker selection
            messages = [
                {"role": "user", "content": "We need to decide on the project timeline"},
                {"role": "assistant", "content": "I think we should start with requirements gathering"},
                {"role": "assistant", "content": "Let me check the available resources"}
            ]
            
            # Multiple speaker selections
            for i in range(3):
                selected_speaker = interceptor.intercept_group_chat_select_speaker(
                    mock_group_chat.select_speaker_msg, mock_group_chat, messages
                )
                
                # Simulate the selected speaker sending a message
                if selected_speaker:
                    interceptor.intercept_send_message(
                        selected_speaker.send, selected_speaker, mock_group_chat.agents[1],
                        f"This is my contribution to the discussion (turn {i+1})",
                        request_reply=False, silent=False
                    )
        
        # Verify group decision events were captured
        decision_events = [e for e in events_captured 
                          if e.event_type in [EventType.DECISION_START, EventType.DECISION_COMPLETE]
                          and e.component == "group_chat"]
        
        assert len(decision_events) >= 6  # 3 selections × 2 events each (start + complete)
        
        # Verify speaker selection data
        selection_events = [e for e in events_captured 
                           if e.operation == "speaker_selected"]
        
        assert len(selection_events) >= 3
        for event in selection_events:
            assert "selected_speaker" in event.data
            assert "available_agents" in event.data
            assert event.data["selected_speaker"] in ["agent1", "agent2", "agent3"]
        
        # Stop monitoring
        summary = await mock_instrumentor.stop_monitoring(session_id)
        assert "conversation_summary" in summary.performance_metrics


if __name__ == "__main__":
    pytest.main([__file__])