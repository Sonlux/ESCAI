"""
OpenAI Assistants instrumentor for the ESCAI framework.

This module provides monitoring capabilities for OpenAI Assistants,
capturing function calls, thread conversations, tool usage patterns, and reasoning processes.
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

# OpenAI imports (with fallback for optional dependency)
try:
    import openai
    from openai import OpenAI
    from openai.types.beta import Assistant, Thread
    from openai.types.beta.threads import Run, Message
    OPENAI_AVAILABLE = True
except ImportError:
    # Create mock classes for type hints when OpenAI is not available
    openai = object
    OpenAI = object
    Assistant = object
    Thread = object
    Run = object
    Message = object
    OPENAI_AVAILABLE = False


class OpenAIAssistantMonitor:
    """
    Monitor for capturing OpenAI Assistant execution patterns.
    
    This class monitors assistant interactions, function calls,
    thread conversations, and tool usage patterns.
    """
    
    def __init__(self, instrumentor: 'OpenAIInstrumentor', session_id: str, agent_id: str):
        """
        Initialize the assistant monitor.
        
        Args:
            instrumentor: The OpenAI instrumentor instance
            session_id: Monitoring session identifier
            agent_id: Agent identifier
        """
        self.instrumentor = weakref.ref(instrumentor)  # Avoid circular references
        self.session_id = session_id
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.OpenAIAssistantMonitor")
        
        # Track assistant state
        self._thread_conversations: Dict[str, List[Dict[str, Any]]] = {}
        self._function_calls: Dict[str, List[Dict[str, Any]]] = {}
        self._tool_usage: Dict[str, Dict[str, Any]] = {}
        self._reasoning_traces: Dict[str, List[str]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
    
    def monitor_run_creation(self, original_create: Callable, client: Any, 
                           thread_id: str, assistant_id: str, **kwargs) -> Any:
        """
        Monitor assistant run creation.
        
        Args:
            original_create: Original run creation method
            client: OpenAI client
            thread_id: Thread identifier
            assistant_id: Assistant identifier
            **kwargs: Additional run parameters
            
        Returns:
            Created run object
        """
        try:
            with self._lock:
                # Initialize thread tracking if needed
                if thread_id not in self._thread_conversations:
                    self._thread_conversations[thread_id] = []
                if thread_id not in self._function_calls:
                    self._function_calls[thread_id] = []
                if thread_id not in self._reasoning_traces:
                    self._reasoning_traces[thread_id] = []
                
                # Create run start event
                start_event = self._create_event(
                    event_type=EventType.AGENT_START,
                    message=f"Assistant run started: {assistant_id}",
                    component="assistant",
                    operation="run_create",
                    data={
                        "thread_id": thread_id,
                        "assistant_id": assistant_id,
                        "run_parameters": kwargs,
                        "instructions": kwargs.get("instructions", ""),
                        "model": kwargs.get("model", "")
                    }
                )
                
                start_event.metadata["thread_context"] = {
                    "thread_id": thread_id,
                    "conversation_length": len(self._thread_conversations[thread_id])
                }
                
                self._queue_event_safe(start_event)
            
            # Create the run
            start_time = time.time()
            run = original_create(thread_id=thread_id, assistant_id=assistant_id, **kwargs)
            creation_time = time.time() - start_time
            
            with self._lock:
                # Record run creation
                run_info = {
                    "run_id": getattr(run, 'id', 'unknown'),
                    "thread_id": thread_id,
                    "assistant_id": assistant_id,
                    "status": getattr(run, 'status', 'unknown'),
                    "created_at": datetime.utcnow(),
                    "creation_time": creation_time
                }
                
                # Create run created event
                created_event = self._create_event(
                    event_type=EventType.TASK_START,
                    message=f"Assistant run created: {run_info['run_id']}",
                    component="assistant",
                    operation="run_created",
                    data=run_info,
                    duration_ms=int(creation_time * 1000)
                )
                
                self._queue_event_safe(created_event)
            
            return run
            
        except Exception as e:
            # Handle run creation failure
            error_event = self._create_event(
                event_type=EventType.AGENT_ERROR,
                message=f"Assistant run creation failed: {assistant_id}",
                component="assistant",
                operation="run_error",
                severity=EventSeverity.ERROR,
                data={
                    "thread_id": thread_id,
                    "assistant_id": assistant_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            
            error_event.set_error(
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            
            self._queue_event_safe(error_event)
            raise
    
    def monitor_run_polling(self, original_retrieve: Callable, client: Any, 
                          thread_id: str, run_id: str) -> Any:
        """
        Monitor run status polling and completion.
        
        Args:
            original_retrieve: Original run retrieval method
            client: OpenAI client
            thread_id: Thread identifier
            run_id: Run identifier
            
        Returns:
            Run object with current status
        """
        try:
            # Retrieve run status
            run = original_retrieve(thread_id=thread_id, run_id=run_id)
            run_status = getattr(run, 'status', 'unknown')
            
            # Create status update event
            status_event = self._create_event(
                event_type=EventType.CUSTOM,
                message=f"Run status update: {run_status}",
                component="assistant",
                operation="status_update",
                data={
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "status": run_status,
                    "usage": getattr(run, 'usage', {}),
                    "last_error": getattr(run, 'last_error', None)
                }
            )
            
            status_event.add_tag("status_update")
            self._queue_event_safe(status_event)
            
            # Handle completion
            if run_status in ['completed', 'failed', 'cancelled', 'expired']:
                self._handle_run_completion(run, thread_id)
            
            # Handle function calls
            if run_status == 'requires_action':
                self._handle_function_calls(run, thread_id)
            
            return run
            
        except Exception as e:
            # Handle polling error
            error_event = self._create_event(
                event_type=EventType.AGENT_ERROR,
                message=f"Run polling failed: {run_id}",
                component="assistant",
                operation="polling_error",
                severity=EventSeverity.ERROR,
                data={
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            
            error_event.set_error(
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            
            self._queue_event_safe(error_event)
            raise
    
    def monitor_message_creation(self, original_create: Callable, client: Any,
                               thread_id: str, role: str, content: str, **kwargs) -> Any:
        """
        Monitor message creation in threads.
        
        Args:
            original_create: Original message creation method
            client: OpenAI client
            thread_id: Thread identifier
            role: Message role (user/assistant)
            content: Message content
            **kwargs: Additional message parameters
            
        Returns:
            Created message object
        """
        try:
            # Create the message
            start_time = time.time()
            message = original_create(thread_id=thread_id, role=role, content=content, **kwargs)
            creation_time = time.time() - start_time
            
            with self._lock:
                # Record message in conversation
                message_info = {
                    "message_id": getattr(message, 'id', 'unknown'),
                    "thread_id": thread_id,
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow(),
                    "creation_time": creation_time
                }
                
                if thread_id not in self._thread_conversations:
                    self._thread_conversations[thread_id] = []
                
                self._thread_conversations[thread_id].append(message_info)
                
                # Create message event
                message_event = self._create_event(
                    event_type=EventType.MESSAGE_SEND if role == "user" else EventType.MESSAGE_RECEIVE,
                    message=f"Message created: {role}",
                    component="thread",
                    operation="message_create",
                    data={
                        "message_id": message_info["message_id"],
                        "thread_id": thread_id,
                        "role": role,
                        "content_length": len(content),
                        "content_preview": content[:200] + "..." if len(content) > 200 else content
                    },
                    duration_ms=int(creation_time * 1000)
                )
                
                message_event.metadata["conversation_context"] = {
                    "thread_id": thread_id,
                    "message_count": len(self._thread_conversations[thread_id]),
                    "conversation_length": sum(len(msg["content"]) for msg in self._thread_conversations[thread_id])
                }
                
                self._queue_event_safe(message_event)
            
            return message
            
        except Exception as e:
            # Handle message creation failure
            error_event = self._create_event(
                event_type=EventType.AGENT_ERROR,
                message=f"Message creation failed: {thread_id}",
                component="thread",
                operation="message_error",
                severity=EventSeverity.ERROR,
                data={
                    "thread_id": thread_id,
                    "role": role,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            
            error_event.set_error(
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            
            self._queue_event_safe(error_event)
            raise
    
    def monitor_function_submission(self, original_submit: Callable, client: Any,
                                  thread_id: str, run_id: str, tool_outputs: List[Dict]) -> Any:
        """
        Monitor function call result submission.
        
        Args:
            original_submit: Original tool output submission method
            client: OpenAI client
            thread_id: Thread identifier
            run_id: Run identifier
            tool_outputs: Function call results
            
        Returns:
            Updated run object
        """
        try:
            with self._lock:
                # Record function call results
                for output in tool_outputs:
                    call_id = output.get("tool_call_id", "unknown")
                    result = output.get("output", "")
                    
                    function_result = {
                        "call_id": call_id,
                        "result": result,
                        "result_length": len(str(result)),
                        "submitted_at": datetime.utcnow(),
                        "thread_id": thread_id,
                        "run_id": run_id
                    }
                    
                    if thread_id not in self._function_calls:
                        self._function_calls[thread_id] = []
                    
                    self._function_calls[thread_id].append(function_result)
                    
                    # Create function result event
                    result_event = self._create_event(
                        event_type=EventType.TOOL_RESPONSE,
                        message=f"Function result submitted: {call_id}",
                        component="function",
                        operation="result_submit",
                        data=function_result
                    )
                    
                    self._queue_event_safe(result_event)
            
            # Submit the results
            start_time = time.time()
            run = original_submit(thread_id=thread_id, run_id=run_id, tool_outputs=tool_outputs)
            submission_time = time.time() - start_time
            
            # Create submission complete event
            complete_event = self._create_event(
                event_type=EventType.ACTION_COMPLETE,
                message=f"Function results submitted: {len(tool_outputs)} outputs",
                component="function",
                operation="submission_complete",
                data={
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "outputs_count": len(tool_outputs),
                    "submission_time_ms": int(submission_time * 1000)
                },
                duration_ms=int(submission_time * 1000)
            )
            
            self._queue_event_safe(complete_event)
            
            return run
            
        except Exception as e:
            # Handle submission failure
            error_event = self._create_event(
                event_type=EventType.AGENT_ERROR,
                message=f"Function submission failed: {run_id}",
                component="function",
                operation="submission_error",
                severity=EventSeverity.ERROR,
                data={
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "outputs_count": len(tool_outputs),
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            
            error_event.set_error(
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            
            self._queue_event_safe(error_event)
            raise
    
    def _handle_run_completion(self, run: Any, thread_id: str) -> None:
        """Handle run completion and extract reasoning traces."""
        try:
            run_id = getattr(run, 'id', 'unknown')
            status = getattr(run, 'status', 'unknown')
            usage = getattr(run, 'usage', {})
            
            with self._lock:
                # Create completion event
                if status == 'completed':
                    event_type = EventType.AGENT_STOP
                    message = f"Assistant run completed: {run_id}"
                else:
                    event_type = EventType.AGENT_ERROR
                    message = f"Assistant run {status}: {run_id}"
                
                completion_event = self._create_event(
                    event_type=event_type,
                    message=message,
                    component="assistant",
                    operation="run_complete",
                    severity=EventSeverity.ERROR if status != 'completed' else EventSeverity.INFO,
                    data={
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "status": status,
                        "usage": usage,
                        "last_error": getattr(run, 'last_error', None)
                    }
                )
                
                # Add reasoning traces if available
                if thread_id in self._reasoning_traces and self._reasoning_traces[thread_id]:
                    completion_event.metadata["reasoning_traces"] = self._reasoning_traces[thread_id]
                
                self._queue_event_safe(completion_event)
                
                # Extract reasoning from recent messages
                self._extract_reasoning_traces(thread_id, run_id)
            
        except Exception as e:
            self.logger.error(f"Error handling run completion: {str(e)}")
    
    def _handle_function_calls(self, run: Any, thread_id: str) -> None:
        """Handle function calls that require action."""
        try:
            required_action = getattr(run, 'required_action', None)
            if not required_action:
                return
            
            submit_tool_outputs = getattr(required_action, 'submit_tool_outputs', None)
            if not submit_tool_outputs:
                return
            
            tool_calls = getattr(submit_tool_outputs, 'tool_calls', [])
            
            with self._lock:
                for tool_call in tool_calls:
                    call_id = getattr(tool_call, 'id', 'unknown')
                    function = getattr(tool_call, 'function', None)
                    
                    if function:
                        function_name = getattr(function, 'name', 'unknown')
                        arguments = getattr(function, 'arguments', '{}')
                        
                        # Parse arguments safely
                        try:
                            parsed_args = json.loads(arguments) if arguments else {}
                        except json.JSONDecodeError:
                            parsed_args = {"raw_arguments": arguments}
                        
                        # Record function call
                        function_call = {
                            "call_id": call_id,
                            "function_name": function_name,
                            "arguments": parsed_args,
                            "called_at": datetime.utcnow(),
                            "thread_id": thread_id,
                            "run_id": getattr(run, 'id', 'unknown')
                        }
                        
                        if thread_id not in self._function_calls:
                            self._function_calls[thread_id] = []
                        
                        self._function_calls[thread_id].append(function_call)
                        
                        # Update tool usage statistics
                        if function_name not in self._tool_usage:
                            self._tool_usage[function_name] = {
                                "call_count": 0,
                                "total_execution_time": 0.0,
                                "success_count": 0,
                                "error_count": 0
                            }
                        
                        self._tool_usage[function_name]["call_count"] += 1
                        
                        # Create function call event
                        call_event = self._create_event(
                            event_type=EventType.TOOL_CALL,
                            message=f"Function called: {function_name}",
                            component="function",
                            operation="call",
                            data=function_call
                        )
                        
                        call_event.metadata["tool_usage"] = self._tool_usage[function_name].copy()
                        self._queue_event_safe(call_event)
            
        except Exception as e:
            self.logger.error(f"Error handling function calls: {str(e)}")
    
    def _extract_reasoning_traces(self, thread_id: str, run_id: str) -> None:
        """Extract reasoning traces from conversation."""
        try:
            if thread_id not in self._thread_conversations:
                return
            
            # Look for reasoning patterns in recent messages
            recent_messages = self._thread_conversations[thread_id][-5:]  # Last 5 messages
            reasoning_patterns = []
            
            for msg in recent_messages:
                content = msg.get("content", "")
                
                # Look for common reasoning patterns
                if any(pattern in content.lower() for pattern in [
                    "let me think", "i need to", "first", "then", "because", 
                    "therefore", "however", "considering", "analysis", "reasoning"
                ]):
                    reasoning_patterns.append({
                        "message_id": msg.get("message_id", ""),
                        "content": content,
                        "timestamp": msg.get("timestamp", datetime.utcnow()),
                        "reasoning_type": "step_by_step"
                    })
            
            if reasoning_patterns:
                with self._lock:
                    if thread_id not in self._reasoning_traces:
                        self._reasoning_traces[thread_id] = []
                    
                    self._reasoning_traces[thread_id].extend(reasoning_patterns)
                    
                    # Create reasoning trace event
                    reasoning_event = self._create_event(
                        event_type=EventType.CUSTOM,
                        message="Reasoning trace extracted",
                        component="reasoning",
                        operation="trace_extraction",
                        data={
                            "thread_id": thread_id,
                            "run_id": run_id,
                            "patterns_count": len(reasoning_patterns),
                            "reasoning_patterns": reasoning_patterns
                        }
                    )
                    
                    reasoning_event.add_tag("reasoning")
                    self._queue_event_safe(reasoning_event)
            
        except Exception as e:
            self.logger.error(f"Error extracting reasoning traces: {str(e)}")
    
    def get_assistant_summary(self) -> Dict[str, Any]:
        """Get a summary of assistant interactions."""
        with self._lock:
            return {
                "total_threads": len(self._thread_conversations),
                "total_messages": sum(len(msgs) for msgs in self._thread_conversations.values()),
                "total_function_calls": sum(len(calls) for calls in self._function_calls.values()),
                "tool_usage_summary": self._tool_usage.copy(),
                "reasoning_traces_count": sum(len(traces) for traces in self._reasoning_traces.values()),
                "active_threads": list(self._thread_conversations.keys())
            }
    
    def _create_event(self, event_type: EventType, message: str, **kwargs) -> AgentEvent:
        """Create a standardized agent event."""
        return AgentEvent(
            event_type=event_type,
            agent_id=self.agent_id,
            session_id=self.session_id,
            message=message,
            framework="openai",
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


class OpenAIInstrumentor(BaseInstrumentor):
    """
    OpenAI Assistants-specific instrumentor for monitoring assistant interactions.
    
    This instrumentor monitors OpenAI Assistants by tracking function calls,
    thread conversations, tool usage patterns, and reasoning processes.
    """
    
    def __init__(self, **kwargs):
        """Initialize the OpenAI instrumentor."""
        if not OPENAI_AVAILABLE:
            raise InstrumentationError(
                "OpenAI library is not available. Please install it with: pip install openai"
            )
        
        super().__init__(**kwargs)
        
        # OpenAI-specific configuration
        self._assistant_monitors: Dict[str, OpenAIAssistantMonitor] = {}
        self._monitor_lock = threading.RLock()
        
        # Client and object tracking
        self._monitored_clients: Dict[str, Set[Any]] = {}  # session_id -> set of clients
        self._original_methods: Dict[str, Dict[str, Callable]] = {}  # session_id -> method_name -> original_method
        
        self.logger.info("OpenAI instrumentor initialized")
    
    async def start_monitoring(self, agent_id: str, config: Dict[str, Any]) -> str:
        """
        Start monitoring OpenAI Assistants.
        
        Args:
            agent_id: Unique identifier for the assistant/system
            config: Configuration parameters including:
                - clients: List of OpenAI client instances to monitor
                - assistant_ids: List of assistant IDs to track
                - monitor_functions: Whether to track function calls (default: True)
                - monitor_reasoning: Whether to extract reasoning traces (default: True)
        
        Returns:
            session_id: Unique identifier for the monitoring session
        """
        try:
            # Create monitoring session
            session = self._create_session(agent_id, config)
            
            # Create assistant monitor
            monitor = OpenAIAssistantMonitor(
                instrumentor=self,
                session_id=session.session_id,
                agent_id=agent_id
            )
            
            with self._monitor_lock:
                self._assistant_monitors[session.session_id] = monitor
            
            # Initialize tracking sets
            self._monitored_clients[session.session_id] = set()
            self._original_methods[session.session_id] = {}
            
            # Set up monitoring for provided clients
            clients = config.get("clients", [])
            for client in clients:
                await self._instrument_client(session.session_id, client, monitor)
            
            # Create start event
            start_event = self.create_event(
                event_type=EventType.AGENT_START,
                agent_id=agent_id,
                session_id=session.session_id,
                message=f"Started monitoring OpenAI Assistants: {agent_id}",
                component="instrumentor",
                operation="start",
                data={
                    "configuration": config,
                    "clients_count": len(clients),
                    "assistant_ids": config.get("assistant_ids", [])
                }
            )
            
            await self._queue_event(start_event)
            
            self.logger.info(f"Started monitoring OpenAI Assistants {agent_id} (session: {session.session_id})")
            return session.session_id
            
        except Exception as e:
            raise InstrumentationError(f"Failed to start monitoring: {str(e)}")
    
    async def stop_monitoring(self, session_id: str) -> MonitoringSummary:
        """
        Stop monitoring OpenAI Assistants.
        
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
            
            # Restore original methods
            await self._restore_original_methods(session_id)
            
            # Get assistant summary
            assistant_summary = {}
            with self._monitor_lock:
                monitor = self._assistant_monitors.pop(session_id, None)
                if monitor:
                    assistant_summary = monitor.get_assistant_summary()
            
            # Clean up tracking data
            self._monitored_clients.pop(session_id, set())
            self._original_methods.pop(session_id, {})
            
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
                message=f"Stopped monitoring OpenAI Assistants: {ended_session.agent_id}",
                component="instrumentor",
                operation="stop",
                duration_ms=duration_ms,
                data=assistant_summary
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
                    "assistant_summary": assistant_summary,
                    **self.get_performance_metrics()
                }
            )
            
            self.logger.info(f"Stopped monitoring session {session_id}")
            return summary
            
        except Exception as e:
            raise InstrumentationError(f"Failed to stop monitoring: {str(e)}")
    
    async def capture_event(self, event: AgentEvent) -> None:
        """
        Capture an agent event from OpenAI Assistants.
        
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
            
            # Process assistant and reasoning context
            await self._process_assistant_context(event)
            
            # Queue event for processing
            await self._queue_event(event)
            
        except Exception as e:
            raise EventProcessingError(f"Failed to capture event: {str(e)}")
    
    def get_supported_events(self) -> List[EventType]:
        """Get the list of event types supported by OpenAI instrumentor."""
        return [
            EventType.AGENT_START,
            EventType.AGENT_STOP,
            EventType.AGENT_ERROR,
            EventType.TASK_START,
            EventType.TASK_COMPLETE,
            EventType.TASK_FAIL,
            EventType.TOOL_CALL,
            EventType.TOOL_RESPONSE,
            EventType.MESSAGE_SEND,
            EventType.MESSAGE_RECEIVE,
            EventType.ACTION_START,
            EventType.ACTION_COMPLETE,
            EventType.CUSTOM
        ]
    
    def get_framework_name(self) -> str:
        """Get the framework name."""
        return "openai"
    
    def get_assistant_monitor(self, session_id: str) -> Optional[OpenAIAssistantMonitor]:
        """
        Get the assistant monitor for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            OpenAIAssistantMonitor or None if not found
        """
        with self._monitor_lock:
            return self._assistant_monitors.get(session_id)
    
    async def _instrument_client(self, session_id: str, client: Any,
                               monitor: OpenAIAssistantMonitor) -> None:
        """Instrument an OpenAI client instance."""
        try:
            # Check if client has the expected structure
            if not hasattr(client, 'beta') or not hasattr(client.beta, 'threads'):
                self.logger.warning("Client does not have expected OpenAI Assistants API structure")
                return
            
            threads = client.beta.threads
            
            # Instrument runs.create
            if hasattr(threads, 'runs') and hasattr(threads.runs, 'create'):
                original_create = threads.runs.create
                create_key = f"client_{id(client)}_runs_create"
                self._original_methods[session_id][create_key] = original_create
                
                def wrapped_create(**kwargs):
                    return monitor.monitor_run_creation(original_create, client, **kwargs)
                
                threads.runs.create = wrapped_create
            
            # Instrument runs.retrieve
            if hasattr(threads, 'runs') and hasattr(threads.runs, 'retrieve'):
                original_retrieve = threads.runs.retrieve
                retrieve_key = f"client_{id(client)}_runs_retrieve"
                self._original_methods[session_id][retrieve_key] = original_retrieve
                
                def wrapped_retrieve(thread_id, run_id):
                    return monitor.monitor_run_polling(original_retrieve, client, thread_id, run_id)
                
                threads.runs.retrieve = wrapped_retrieve
            
            # Instrument messages.create
            if hasattr(threads, 'messages') and hasattr(threads.messages, 'create'):
                original_msg_create = threads.messages.create
                msg_create_key = f"client_{id(client)}_messages_create"
                self._original_methods[session_id][msg_create_key] = original_msg_create
                
                def wrapped_msg_create(**kwargs):
                    return monitor.monitor_message_creation(original_msg_create, client, **kwargs)
                
                threads.messages.create = wrapped_msg_create
            
            # Instrument runs.submit_tool_outputs
            if hasattr(threads, 'runs') and hasattr(threads.runs, 'submit_tool_outputs'):
                original_submit = threads.runs.submit_tool_outputs
                submit_key = f"client_{id(client)}_runs_submit_tool_outputs"
                self._original_methods[session_id][submit_key] = original_submit
                
                def wrapped_submit(thread_id, run_id, **kwargs):
                    return monitor.monitor_function_submission(original_submit, client, thread_id, run_id, **kwargs)
                
                threads.runs.submit_tool_outputs = wrapped_submit
            
            # Track client
            self._monitored_clients[session_id].add(client)
            
            self.logger.debug(f"Instrumented OpenAI client: {id(client)}")
            
        except Exception as e:
            self.logger.error(f"Failed to instrument client: {str(e)}")
    
    async def _restore_original_methods(self, session_id: str) -> None:
        """Restore original methods for all monitored clients."""
        try:
            for client in self._monitored_clients.get(session_id, set()):
                client_id = id(client)
                
                # Restore runs.create
                create_key = f"client_{client_id}_runs_create"
                if create_key in self._original_methods.get(session_id, {}):
                    if hasattr(client, 'beta') and hasattr(client.beta.threads, 'runs'):
                        client.beta.threads.runs.create = self._original_methods[session_id][create_key]
                
                # Restore runs.retrieve
                retrieve_key = f"client_{client_id}_runs_retrieve"
                if retrieve_key in self._original_methods.get(session_id, {}):
                    if hasattr(client, 'beta') and hasattr(client.beta.threads, 'runs'):
                        client.beta.threads.runs.retrieve = self._original_methods[session_id][retrieve_key]
                
                # Restore messages.create
                msg_create_key = f"client_{client_id}_messages_create"
                if msg_create_key in self._original_methods.get(session_id, {}):
                    if hasattr(client, 'beta') and hasattr(client.beta.threads, 'messages'):
                        client.beta.threads.messages.create = self._original_methods[session_id][msg_create_key]
                
                # Restore runs.submit_tool_outputs
                submit_key = f"client_{client_id}_runs_submit_tool_outputs"
                if submit_key in self._original_methods.get(session_id, {}):
                    if hasattr(client, 'beta') and hasattr(client.beta.threads, 'runs'):
                        client.beta.threads.runs.submit_tool_outputs = self._original_methods[session_id][submit_key]
            
            self.logger.debug(f"Restored original methods for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to restore original methods: {str(e)}")
    
    async def _process_assistant_context(self, event: AgentEvent) -> None:
        """Process assistant and reasoning context from events."""
        try:
            # Extract function call patterns
            if event.event_type in [EventType.TOOL_CALL, EventType.TOOL_RESPONSE]:
                # Create function analysis event
                function_event = self.create_event(
                    event_type=EventType.CUSTOM,
                    agent_id=event.agent_id,
                    session_id=event.session_id,
                    message="Function usage pattern detected",
                    component="function_analysis",
                    operation="pattern_analysis",
                    data={
                        "function_name": event.data.get("function_name", ""),
                        "call_id": event.data.get("call_id", ""),
                        "thread_id": event.data.get("thread_id", ""),
                        "source_event_id": event.event_id
                    },
                    parent_event_id=event.event_id
                )
                function_event.add_tag("function_usage")
                await self._queue_event(function_event)
            
            # Extract reasoning patterns
            if "reasoning" in event.tags:
                reasoning_event = self.create_event(
                    event_type=EventType.CUSTOM,
                    agent_id=event.agent_id,
                    session_id=event.session_id,
                    message="Reasoning pattern detected",
                    component="reasoning_analysis",
                    operation="pattern_analysis",
                    data={
                        "reasoning_type": event.data.get("reasoning_patterns", [{}])[0].get("reasoning_type", ""),
                        "patterns_count": event.data.get("patterns_count", 0),
                        "thread_id": event.data.get("thread_id", ""),
                        "source_event_id": event.event_id
                    },
                    parent_event_id=event.event_id
                )
                reasoning_event.add_tag("reasoning")
                await self._queue_event(reasoning_event)
            
        except Exception as e:
            self.logger.error(f"Error processing assistant context: {str(e)}")