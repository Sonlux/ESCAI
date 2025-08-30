"""
AutoGen instrumentor for the ESCAI framework.

This module provides monitoring capabilities for AutoGen multi-agent systems,
capturing conversation flows, message passing, and group decision-making processes.
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Union, Callable
from datetime import datetime
import threading
import weakref
import json

from .base_instrumentor import BaseInstrumentor, InstrumentationError, EventProcessingError
from .events import AgentEvent, EventType, EventSeverity, MonitoringSummary

# AutoGen imports (with fallback for optional dependency)
try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    from autogen.agentchat.conversable_agent import ConversableAgent as BaseConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    # Create mock classes for type hints when AutoGen is not available
    ConversableAgent = object
    GroupChat = object
    GroupChatManager = object
    BaseConversableAgent = object
    AUTOGEN_AVAILABLE = False


class AutoGenMessageInterceptor:
    """
    Message interceptor for capturing AutoGen agent communications.
    
    This class intercepts messages between agents to capture conversation flows,
    role assignments, and decision-making processes.
    """
    
    def __init__(self, instrumentor: 'AutoGenInstrumentor', session_id: str, agent_id: str):
        """
        Initialize the message interceptor.
        
        Args:
            instrumentor: The AutoGen instrumentor instance
            session_id: Monitoring session identifier
            agent_id: Agent identifier
        """
        self.instrumentor = weakref.ref(instrumentor)  # Avoid circular references
        self.session_id = session_id
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.AutoGenMessageInterceptor")
        
        # Track conversation state
        self._conversation_history: List[Dict[str, Any]] = []
        self._agent_roles: Dict[str, str] = {}
        self._message_sequence: int = 0
        self._group_decisions: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.RLock()
    
    def intercept_send_message(self, original_send: Callable, sender: Any, recipient: Any, 
                             message: Union[str, Dict], request_reply: bool = True, 
                             silent: bool = False) -> Any:
        """
        Intercept agent message sending.
        
        Args:
            original_send: Original send method
            sender: Sending agent
            recipient: Receiving agent
            message: Message content
            request_reply: Whether to request a reply
            silent: Whether to send silently
            
        Returns:
            Result of original send method
        """
        try:
            with self._lock:
                self._message_sequence += 1
                
                # Extract agent information
                sender_name = getattr(sender, 'name', 'unknown_sender')
                recipient_name = getattr(recipient, 'name', 'unknown_recipient')
                
                # Store agent roles if available
                if hasattr(sender, 'system_message'):
                    self._agent_roles[sender_name] = getattr(sender, 'system_message', '')
                if hasattr(recipient, 'system_message'):
                    self._agent_roles[recipient_name] = getattr(recipient, 'system_message', '')
                
                # Process message content
                message_content = message
                if isinstance(message, dict):
                    message_content = message.get('content', str(message))
                elif not isinstance(message, str):
                    message_content = str(message)
                
                # Create message send event
                send_event = self._create_event(
                    event_type=EventType.MESSAGE_SEND,
                    message=f"Message sent from {sender_name} to {recipient_name}",
                    component="agent",
                    operation="send_message",
                    data={
                        "sender": sender_name,
                        "recipient": recipient_name,
                        "message_content": message_content,
                        "message_sequence": self._message_sequence,
                        "request_reply": request_reply,
                        "silent": silent,
                        "message_type": type(message).__name__
                    }
                )
                
                # Add conversation context
                send_event.metadata["conversation_turn"] = self._message_sequence
                send_event.metadata["sender_role"] = self._agent_roles.get(sender_name, "unknown")
                send_event.metadata["recipient_role"] = self._agent_roles.get(recipient_name, "unknown")
                
                self._queue_event_safe(send_event)
                
                # Store in conversation history
                conversation_entry = {
                    "timestamp": datetime.utcnow(),
                    "sequence": self._message_sequence,
                    "sender": sender_name,
                    "recipient": recipient_name,
                    "message": message_content,
                    "type": "send"
                }
                self._conversation_history.append(conversation_entry)
            
            # Call original method
            start_time = time.time()
            result = original_send(recipient, message, request_reply, silent)
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Create completion event
            with self._lock:
                complete_event = self._create_event(
                    event_type=EventType.MESSAGE_RECEIVE,
                    message=f"Message delivery completed: {sender_name} -> {recipient_name}",
                    component="agent",
                    operation="message_delivered",
                    data={
                        "sender": sender_name,
                        "recipient": recipient_name,
                        "delivery_success": True,
                        "message_sequence": self._message_sequence
                    },
                    duration_ms=duration_ms
                )
                
                self._queue_event_safe(complete_event)
            
            return result
            
        except Exception as e:
            # Create error event
            error_event = self._create_event(
                event_type=EventType.AGENT_ERROR,
                message=f"Message send failed: {sender_name} -> {recipient_name}",
                component="agent",
                operation="send_error",
                severity=EventSeverity.ERROR,
                data={
                    "sender": sender_name,
                    "recipient": recipient_name,
                    "message_sequence": self._message_sequence
                }
            )
            
            error_event.set_error(
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            
            self._queue_event_safe(error_event)
            raise
    
    def intercept_receive_message(self, original_receive: Callable, agent: Any, 
                                message: Union[str, Dict], sender: Any) -> Any:
        """
        Intercept agent message receiving.
        
        Args:
            original_receive: Original receive method
            agent: Receiving agent
            message: Message content
            sender: Sending agent
            
        Returns:
            Result of original receive method
        """
        try:
            # Extract agent information
            agent_name = getattr(agent, 'name', 'unknown_agent')
            sender_name = getattr(sender, 'name', 'unknown_sender')
            
            # Process message content
            message_content = message
            if isinstance(message, dict):
                message_content = message.get('content', str(message))
            elif not isinstance(message, str):
                message_content = str(message)
            
            # Create receive event
            receive_event = self._create_event(
                event_type=EventType.MESSAGE_RECEIVE,
                message=f"Message received by {agent_name} from {sender_name}",
                component="agent",
                operation="receive_message",
                data={
                    "agent": agent_name,
                    "sender": sender_name,
                    "message_content": message_content,
                    "message_type": type(message).__name__
                }
            )
            
            # Add agent context
            receive_event.metadata["agent_role"] = self._agent_roles.get(agent_name, "unknown")
            receive_event.metadata["sender_role"] = self._agent_roles.get(sender_name, "unknown")
            
            self._queue_event_safe(receive_event)
            
            # Store in conversation history
            with self._lock:
                conversation_entry = {
                    "timestamp": datetime.utcnow(),
                    "sequence": self._message_sequence,
                    "agent": agent_name,
                    "sender": sender_name,
                    "message": message_content,
                    "type": "receive"
                }
                self._conversation_history.append(conversation_entry)
            
            # Call original method
            start_time = time.time()
            result = original_receive(message, sender)
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Create processing complete event
            process_event = self._create_event(
                event_type=EventType.DECISION_COMPLETE,
                message=f"Message processed by {agent_name}",
                component="agent",
                operation="process_message",
                data={
                    "agent": agent_name,
                    "sender": sender_name,
                    "processing_result": str(result) if result else None
                },
                duration_ms=duration_ms
            )
            
            self._queue_event_safe(process_event)
            
            return result
            
        except Exception as e:
            # Create error event
            error_event = self._create_event(
                event_type=EventType.AGENT_ERROR,
                message=f"Message receive failed: {agent_name}",
                component="agent",
                operation="receive_error",
                severity=EventSeverity.ERROR,
                data={
                    "agent": agent_name,
                    "sender": sender_name
                }
            )
            
            error_event.set_error(
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            
            self._queue_event_safe(error_event)
            raise
    
    def intercept_group_chat_select_speaker(self, original_select: Callable, 
                                          group_chat: Any, messages: List) -> Any:
        """
        Intercept group chat speaker selection.
        
        Args:
            original_select: Original select_speaker method
            group_chat: GroupChat instance
            messages: Conversation messages
            
        Returns:
            Selected speaker
        """
        try:
            # Create decision start event
            decision_event = self._create_event(
                event_type=EventType.DECISION_START,
                message="Group chat speaker selection started",
                component="group_chat",
                operation="select_speaker",
                data={
                    "message_count": len(messages),
                    "available_agents": [getattr(agent, 'name', 'unknown') 
                                       for agent in getattr(group_chat, 'agents', [])]
                }
            )
            
            self._queue_event_safe(decision_event)
            
            # Call original method
            start_time = time.time()
            selected_speaker = original_select(messages)
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Create decision complete event
            speaker_name = getattr(selected_speaker, 'name', 'unknown')
            available_agents = [getattr(agent, 'name', 'unknown') 
                              for agent in getattr(group_chat, 'agents', [])]
            complete_event = self._create_event(
                event_type=EventType.DECISION_COMPLETE,
                message=f"Speaker selected: {speaker_name}",
                component="group_chat",
                operation="speaker_selected",
                data={
                    "selected_speaker": speaker_name,
                    "available_agents": available_agents,
                    "selection_criteria": "group_chat_logic"
                },
                duration_ms=duration_ms
            )
            
            self._queue_event_safe(complete_event)
            
            # Track group decision
            with self._lock:
                decision_entry = {
                    "timestamp": datetime.utcnow(),
                    "type": "speaker_selection",
                    "selected_speaker": speaker_name,
                    "available_agents": [getattr(agent, 'name', 'unknown') 
                                       for agent in getattr(group_chat, 'agents', [])],
                    "message_count": len(messages)
                }
                self._group_decisions.append(decision_entry)
            
            return selected_speaker
            
        except Exception as e:
            # Create error event
            error_event = self._create_event(
                event_type=EventType.AGENT_ERROR,
                message="Group chat speaker selection failed",
                component="group_chat",
                operation="selection_error",
                severity=EventSeverity.ERROR
            )
            
            error_event.set_error(
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            
            self._queue_event_safe(error_event)
            raise
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        with self._lock:
            # Collect all unique agents from conversation history
            unique_agents = set()
            for entry in self._conversation_history:
                if "sender" in entry and entry["sender"]:
                    unique_agents.add(entry["sender"])
                if "agent" in entry and entry["agent"]:
                    unique_agents.add(entry["agent"])
                if "recipient" in entry and entry["recipient"]:
                    unique_agents.add(entry["recipient"])
            
            return {
                "total_messages": len(self._conversation_history),
                "unique_agents": len(unique_agents),
                "agent_roles": self._agent_roles.copy(),
                "group_decisions": len(self._group_decisions),
                "conversation_duration": (
                    (self._conversation_history[-1]["timestamp"] - self._conversation_history[0]["timestamp"]).total_seconds()
                    if len(self._conversation_history) >= 2 else 0
                )
            }
    
    def _create_event(self, event_type: EventType, message: str, **kwargs) -> AgentEvent:
        """Create a standardized agent event."""
        return AgentEvent(
            event_type=event_type,
            agent_id=self.agent_id,
            session_id=self.session_id,
            message=message,
            framework="autogen",
            **kwargs
        )
    
    def _queue_event_safe(self, event: AgentEvent) -> None:
        """Safely queue an event to the instrumentor."""
        try:
            instrumentor = self.instrumentor()
            if instrumentor:
                asyncio.create_task(instrumentor._queue_event(event))
        except Exception as e:
            self.logger.error(f"Failed to queue event: {str(e)}")


class AutoGenInstrumentor(BaseInstrumentor):
    """
    AutoGen-specific instrumentor for monitoring multi-agent conversations.
    
    This instrumentor monitors AutoGen multi-agent systems by intercepting
    message passing, tracking role assignments, and analyzing group decision-making.
    """
    
    def __init__(self, **kwargs):
        """Initialize the AutoGen instrumentor."""
        if not AUTOGEN_AVAILABLE:
            raise InstrumentationError(
                "AutoGen is not available. Please install it with: pip install pyautogen"
            )
        
        super().__init__(**kwargs)
        
        # AutoGen-specific configuration
        self._message_interceptors: Dict[str, AutoGenMessageInterceptor] = {}
        self._interceptor_lock = threading.RLock()
        
        # Agent tracking
        self._monitored_agents: Dict[str, Set[Any]] = {}  # session_id -> set of agents
        self._original_methods: Dict[str, Dict[str, Callable]] = {}  # session_id -> method_name -> original_method
        
        # Group chat tracking
        self._group_chats: Dict[str, List[Any]] = {}  # session_id -> list of group chats
        
        self.logger.info("AutoGen instrumentor initialized")
    
    async def start_monitoring(self, agent_id: str, config: Dict[str, Any]) -> str:
        """
        Start monitoring AutoGen agents.
        
        Args:
            agent_id: Unique identifier for the agent/system
            config: Configuration parameters including:
                - agents: List of AutoGen agents to monitor
                - group_chats: List of GroupChat instances to monitor
                - monitor_conversations: Whether to track conversations (default: True)
                - monitor_decisions: Whether to track group decisions (default: True)
        
        Returns:
            session_id: Unique identifier for the monitoring session
        """
        try:
            # Create monitoring session
            session = self._create_session(agent_id, config)
            
            # Create message interceptor
            interceptor = AutoGenMessageInterceptor(
                instrumentor=self,
                session_id=session.session_id,
                agent_id=agent_id
            )
            
            with self._interceptor_lock:
                self._message_interceptors[session.session_id] = interceptor
            
            # Initialize agent and group chat tracking
            self._monitored_agents[session.session_id] = set()
            self._original_methods[session.session_id] = {}
            self._group_chats[session.session_id] = []
            
            # Set up monitoring for provided agents
            agents = config.get("agents", [])
            for agent in agents:
                await self._instrument_agent(session.session_id, agent, interceptor)
            
            # Set up monitoring for group chats
            group_chats = config.get("group_chats", [])
            for group_chat in group_chats:
                await self._instrument_group_chat(session.session_id, group_chat, interceptor)
            
            # Create start event
            start_event = self.create_event(
                event_type=EventType.AGENT_START,
                agent_id=agent_id,
                session_id=session.session_id,
                message=f"Started monitoring AutoGen system: {agent_id}",
                component="instrumentor",
                operation="start",
                data={
                    "configuration": config,
                    "agents_count": len(agents),
                    "group_chats_count": len(group_chats)
                }
            )
            
            await self._queue_event(start_event)
            
            self.logger.info(f"Started monitoring AutoGen system {agent_id} (session: {session.session_id})")
            return session.session_id
            
        except Exception as e:
            raise InstrumentationError(f"Failed to start monitoring: {str(e)}")
    
    async def stop_monitoring(self, session_id: str) -> MonitoringSummary:  # type: ignore[override]
        """
        Stop monitoring AutoGen agents.
        
        Args:
            session_id: Identifier of the session to stop
        
        Returns:
            MonitoringSummary: Summary of the monitoring session
        """
        try:
            # Get session
            session = self._get_session(session_id)
            if not session:
                raise InstrumentationError(f"Session not found: {session_id}")
            
            # Restore original methods for monitored agents
            await self._restore_agent_methods(session_id)
            
            # Get conversation summary
            conversation_summary: Dict[str, Any] = {}
            with self._interceptor_lock:
                interceptor = self._message_interceptors.pop(session_id, None)
                if interceptor:
                    conversation_summary = interceptor.get_conversation_summary()
            
            # Clean up tracking data
            self._monitored_agents.pop(session_id, set())
            self._original_methods.pop(session_id, {})
            self._group_chats.pop(session_id, [])
            
            # End session
            ended_session = self._end_session(session_id)
            if not ended_session:
                raise InstrumentationError(f"Failed to end session: {session_id}")
            
            # Calculate summary metrics
            duration_ms = int((ended_session.end_time - ended_session.start_time).total_seconds() * 1000)
            
            # Create stop event
            stop_event = self.create_event(
                event_type=EventType.AGENT_STOP,
                agent_id=ended_session.agent_id,
                session_id=session_id,
                message=f"Stopped monitoring AutoGen system: {ended_session.agent_id}",
                component="instrumentor",
                operation="stop",
                duration_ms=duration_ms,
                data=conversation_summary
            )
            
            await self._queue_event(stop_event)
            
            # Create monitoring summary
            summary = MonitoringSummary(
                session_id=session_id,
                agent_id=ended_session.agent_id,
                framework=self.get_framework_name(),
                start_time=ended_session.start_time,
                end_time=ended_session.end_time,
                total_duration_ms=duration_ms,
                total_events=ended_session.event_count,
                performance_metrics={
                    "conversation_summary": conversation_summary,
                    **self.get_performance_metrics()
                }
            )
            
            self.logger.info(f"Stopped monitoring session {session_id}")
            return summary
            
        except Exception as e:
            raise InstrumentationError(f"Failed to stop monitoring: {str(e)}")
    
    async def capture_event(self, event: AgentEvent) -> None:
        """
        Capture an agent event from AutoGen.
        
        Args:
            event: The agent event to capture
        """
        try:
            # Validate event
            if not event.validate():
                raise EventProcessingError(f"Invalid event: {event.event_id}")
            
            # Update session event count
            session = self._get_session(event.session_id)
            if session:
                session.event_count += 1
            
            # Process conversation and decision tracking
            await self._process_conversation_context(event)
            
            # Queue event for processing
            await self._queue_event(event)
            
        except Exception as e:
            raise EventProcessingError(f"Failed to capture event: {str(e)}")
    
    def get_supported_events(self) -> List[EventType]:
        """Get the list of event types supported by AutoGen instrumentor."""
        return [
            EventType.AGENT_START,
            EventType.AGENT_STOP,
            EventType.AGENT_ERROR,
            EventType.DECISION_START,
            EventType.DECISION_COMPLETE,
            EventType.MESSAGE_SEND,
            EventType.MESSAGE_RECEIVE,
            EventType.ACTION_START,
            EventType.ACTION_COMPLETE,
            EventType.CUSTOM
        ]
    
    def get_framework_name(self) -> str:
        """Get the framework name."""
        return "autogen"
    
    def get_message_interceptor(self, session_id: str) -> Optional[AutoGenMessageInterceptor]:
        """
        Get the message interceptor for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            AutoGenMessageInterceptor or None if not found
        """
        with self._interceptor_lock:
            return self._message_interceptors.get(session_id)
    
    async def _instrument_agent(self, session_id: str, agent: Any, 
                              interceptor: AutoGenMessageInterceptor) -> None:
        """Instrument an individual AutoGen agent."""
        try:
            if not hasattr(agent, 'send') or not hasattr(agent, 'receive'):
                self.logger.warning(f"Agent {getattr(agent, 'name', 'unknown')} does not have send/receive methods")
                return
            
            # Store original methods
            agent_name = getattr(agent, 'name', f'agent_{id(agent)}')
            original_send = agent.send
            original_receive = agent.receive
            
            self._original_methods[session_id][f"{agent_name}_send"] = original_send
            self._original_methods[session_id][f"{agent_name}_receive"] = original_receive
            
            # Create wrapped methods
            def wrapped_send(recipient, message, request_reply=True, silent=False):
                return interceptor.intercept_send_message(
                    original_send, agent, recipient, message, request_reply, silent
                )
            
            def wrapped_receive(message, sender):
                return interceptor.intercept_receive_message(
                    original_receive, agent, message, sender
                )
            
            # Replace methods
            agent.send = wrapped_send
            agent.receive = wrapped_receive
            
            # Track agent
            self._monitored_agents[session_id].add(agent)
            
            self.logger.debug(f"Instrumented agent: {agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to instrument agent: {str(e)}")
    
    async def _instrument_group_chat(self, session_id: str, group_chat: Any,
                                   interceptor: AutoGenMessageInterceptor) -> None:
        """Instrument a GroupChat instance."""
        try:
            if not hasattr(group_chat, 'select_speaker_msg'):
                self.logger.warning("GroupChat does not have select_speaker_msg method")
                return
            
            # Store original method
            original_select = group_chat.select_speaker_msg
            self._original_methods[session_id][f"group_chat_{id(group_chat)}_select"] = original_select
            
            # Create wrapped method
            def wrapped_select_speaker(messages):
                return interceptor.intercept_group_chat_select_speaker(
                    original_select, group_chat, messages
                )
            
            # Replace method
            group_chat.select_speaker_msg = wrapped_select_speaker
            
            # Track group chat
            self._group_chats[session_id].append(group_chat)
            
            # Instrument all agents in the group chat
            if hasattr(group_chat, 'agents'):
                for agent in group_chat.agents:
                    await self._instrument_agent(session_id, agent, interceptor)
            
            self.logger.debug(f"Instrumented group chat with {len(getattr(group_chat, 'agents', []))} agents")
            
        except Exception as e:
            self.logger.error(f"Failed to instrument group chat: {str(e)}")
    
    async def _restore_agent_methods(self, session_id: str) -> None:
        """Restore original methods for all monitored agents."""
        try:
            # Restore agent methods
            for agent in self._monitored_agents.get(session_id, set()):
                agent_name = getattr(agent, 'name', f'agent_{id(agent)}')
                
                # Restore send method
                send_key = f"{agent_name}_send"
                if send_key in self._original_methods.get(session_id, {}):
                    agent.send = self._original_methods[session_id][send_key]
                
                # Restore receive method
                receive_key = f"{agent_name}_receive"
                if receive_key in self._original_methods.get(session_id, {}):
                    agent.receive = self._original_methods[session_id][receive_key]
            
            # Restore group chat methods
            for group_chat in self._group_chats.get(session_id, []):
                select_key = f"group_chat_{id(group_chat)}_select"
                if select_key in self._original_methods.get(session_id, {}):
                    group_chat.select_speaker_msg = self._original_methods[session_id][select_key]
            
            self.logger.debug(f"Restored original methods for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to restore agent methods: {str(e)}")
    
    def analyze_conversation_patterns(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze conversation patterns for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation pattern analysis
        """
        try:
            interceptor = self.get_message_interceptor(session_id)
            if not interceptor:
                return {}
            
            conversation_summary = interceptor.get_conversation_summary()
            
            # Analyze message flow patterns
            conversation_history = interceptor._conversation_history
            if not conversation_history:
                return conversation_summary
            
            # Calculate conversation metrics
            message_intervals = []
            for i in range(1, len(conversation_history)):
                prev_msg = conversation_history[i-1]
                curr_msg = conversation_history[i]
                interval = (curr_msg["timestamp"] - prev_msg["timestamp"]).total_seconds()
                message_intervals.append(interval)
            
            # Identify dominant speakers
            speaker_counts: Dict[str, int] = {}
            for msg in conversation_history:
                speaker = msg.get("sender", msg.get("agent", "unknown"))
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            # Calculate conversation balance
            total_messages = len(conversation_history)
            speaker_balance = {
                speaker: count / total_messages 
                for speaker, count in speaker_counts.items()
            } if total_messages > 0 else {}
            
            conversation_summary.update({
                "message_intervals": {
                    "average_seconds": sum(message_intervals) / len(message_intervals) if message_intervals else 0,
                    "min_seconds": min(message_intervals) if message_intervals else 0,
                    "max_seconds": max(message_intervals) if message_intervals else 0
                },
                "speaker_distribution": {k: float(v) for k, v in speaker_counts.items()},
                "speaker_balance": speaker_balance,
                "conversation_flow": "balanced" if all(0.1 <= balance <= 0.6 for balance in speaker_balance.values()) else "unbalanced"
            })
            
            return conversation_summary
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation patterns: {str(e)}")
            return {}
    
    def analyze_group_decision_patterns(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze group decision-making patterns for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Group decision pattern analysis
        """
        try:
            interceptor = self.get_message_interceptor(session_id)
            if not interceptor:
                return {}
            
            group_decisions = interceptor._group_decisions
            if not group_decisions:
                return {"total_decisions": 0}
            
            # Analyze decision patterns
            decision_types: Dict[str, int] = {}
            selected_speakers: Dict[str, int] = {}
            
            for decision in group_decisions:
                decision_type = decision.get("type", "unknown")
                decision_types[decision_type] = decision_types.get(decision_type, 0) + 1
                
                if decision_type == "speaker_selection":
                    speaker = decision.get("selected_speaker", "unknown")
                    selected_speakers[speaker] = selected_speakers.get(speaker, 0) + 1
            
            # Calculate selection fairness
            total_selections = sum(selected_speakers.values())
            selection_distribution = {
                speaker: count / total_selections 
                for speaker, count in selected_speakers.items()
            } if total_selections > 0 else {}
            
            return {
                "total_decisions": len(group_decisions),
                "decision_types": decision_types,
                "speaker_selections": selected_speakers,
                "selection_distribution": selection_distribution,
                "selection_fairness": "fair" if all(0.1 <= dist <= 0.5 for dist in selection_distribution.values()) else "biased"
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing group decision patterns: {str(e)}")
            return {}
    
    def get_agent_coordination_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get agent coordination metrics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Agent coordination metrics
        """
        try:
            conversation_analysis = self.analyze_conversation_patterns(session_id)
            decision_analysis = self.analyze_group_decision_patterns(session_id)
            
            # Calculate coordination score
            coordination_factors = []
            
            # Factor 1: Conversation balance (0-1)
            if conversation_analysis.get("conversation_flow") == "balanced":
                coordination_factors.append(0.8)
            else:
                coordination_factors.append(0.4)
            
            # Factor 2: Decision fairness (0-1)
            if decision_analysis.get("selection_fairness") == "fair":
                coordination_factors.append(0.9)
            else:
                coordination_factors.append(0.3)
            
            # Factor 3: Response time consistency (0-1)
            intervals = conversation_analysis.get("message_intervals", {})
            avg_interval = intervals.get("average_seconds", 0)
            max_interval = intervals.get("max_seconds", 0)
            if max_interval > 0 and avg_interval > 0:
                consistency = 1 - min(1, (max_interval - avg_interval) / max_interval)
                coordination_factors.append(consistency)
            else:
                coordination_factors.append(0.5)
            
            coordination_score = sum(coordination_factors) / len(coordination_factors) if coordination_factors else 0
            
            return {
                "coordination_score": coordination_score,
                "coordination_level": "high" if coordination_score > 0.7 else "medium" if coordination_score > 0.4 else "low",
                "conversation_analysis": conversation_analysis,
                "decision_analysis": decision_analysis,
                "coordination_factors": {
                    "conversation_balance": coordination_factors[0] if len(coordination_factors) > 0 else 0,
                    "decision_fairness": coordination_factors[1] if len(coordination_factors) > 1 else 0,
                    "response_consistency": coordination_factors[2] if len(coordination_factors) > 2 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting coordination metrics: {str(e)}")
            return {}
    
    async def _process_conversation_context(self, event: AgentEvent) -> None:
        """Process conversation and decision context from events."""
        try:
            # Extract conversation patterns
            if event.event_type in [EventType.MESSAGE_SEND, EventType.MESSAGE_RECEIVE]:
                # Create conversation analysis event
                conversation_event = self.create_event(
                    event_type=EventType.CUSTOM,
                    agent_id=event.agent_id,
                    session_id=event.session_id,
                    message="Conversation pattern detected",
                    component="conversation",
                    operation="pattern_analysis",
                    data={
                        "conversation_turn": event.metadata.get("conversation_turn", 0),
                        "participants": [
                            event.data.get("sender", ""),
                            event.data.get("recipient", "") or event.data.get("agent", "")
                        ],
                        "source_event_id": event.event_id
                    },
                    parent_event_id=event.event_id
                )
                conversation_event.add_tag("conversation")
                await self._queue_event(conversation_event)
            
            # Extract group decision patterns
            if event.event_type == EventType.DECISION_COMPLETE and event.component == "group_chat":
                decision_event = self.create_event(
                    event_type=EventType.CUSTOM,
                    agent_id=event.agent_id,
                    session_id=event.session_id,
                    message="Group decision pattern detected",
                    component="group_decision",
                    operation="decision_analysis",
                    data={
                        "decision_type": "speaker_selection",
                        "selected_option": event.data.get("selected_speaker", ""),
                        "available_options": event.data.get("available_agents", []),
                        "source_event_id": event.event_id
                    },
                    parent_event_id=event.event_id
                )
                decision_event.add_tag("group_decision")
                await self._queue_event(decision_event)
            
        except Exception as e:
            self.logger.error(f"Error processing conversation context: {str(e)}")