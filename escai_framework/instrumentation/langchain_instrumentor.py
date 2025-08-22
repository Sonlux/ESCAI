"""
LangChain instrumentor for the ESCAI framework.

This module provides monitoring capabilities for LangChain agents and chains,
capturing execution events, reasoning traces, and performance metrics.
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
import threading
import weakref

from .base_instrumentor import BaseInstrumentor, InstrumentationError, EventProcessingError
from .events import AgentEvent, EventType, EventSeverity, MonitoringSummary

# LangChain imports (with fallback for optional dependency)
try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import BaseMessage, AgentAction, AgentFinish
    from langchain.schema.output import LLMResult
    from langchain.schema.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Create mock classes for type hints when LangChain is not available
    BaseCallbackHandler = object
    BaseMessage = object
    AgentAction = object
    AgentFinish = object
    LLMResult = object
    Document = object
    LANGCHAIN_AVAILABLE = False


class LangChainCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for capturing LangChain execution events.
    
    This handler integrates with LangChain's callback system to capture
    chain execution steps, LLM calls, tool usage, and reasoning traces.
    """
    
    def __init__(self, instrumentor: 'LangChainInstrumentor', session_id: str, agent_id: str):
        """
        Initialize the callback handler.
        
        Args:
            instrumentor: The LangChain instrumentor instance
            session_id: Monitoring session identifier
            agent_id: Agent identifier
        """
        super().__init__()
        self.instrumentor = weakref.ref(instrumentor)  # Avoid circular references
        self.session_id = session_id
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.LangChainCallbackHandler")
        
        # Track execution context
        self._execution_stack: List[Dict[str, Any]] = []
        self._chain_start_times: Dict[str, float] = {}
        self._llm_start_times: Dict[str, float] = {}
        self._tool_start_times: Dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], 
                      run_id: Optional[str] = None, parent_run_id: Optional[str] = None,
                      **kwargs: Any) -> None:
        """Called when a chain starts running."""
        try:
            with self._lock:
                start_time = time.time()
                self._chain_start_times[run_id or "unknown"] = start_time
                
                # Create execution context
                context = {
                    "type": "chain",
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "start_time": start_time,
                    "chain_type": serialized.get("name", "unknown"),
                    "inputs": inputs
                }
                self._execution_stack.append(context)
                
                # Create event
                event = self._create_event(
                    event_type=EventType.TASK_START,
                    message=f"Chain started: {serialized.get('name', 'unknown')}",
                    component="chain",
                    operation="start",
                    data={
                        "chain_type": serialized.get("name", "unknown"),
                        "inputs": inputs,
                        "serialized": serialized,
                        "run_id": run_id,
                        "parent_run_id": parent_run_id
                    },
                    correlation_id=run_id,
                    parent_event_id=parent_run_id
                )
                
                self._queue_event_safe(event)
                
        except Exception as e:
            self.logger.error(f"Error in on_chain_start: {str(e)}")
    
    def on_chain_end(self, outputs: Dict[str, Any], run_id: Optional[str] = None,
                    parent_run_id: Optional[str] = None, **kwargs: Any) -> None:
        """Called when a chain finishes running."""
        try:
            with self._lock:
                end_time = time.time()
                start_time = self._chain_start_times.pop(run_id or "unknown", end_time)
                duration_ms = int((end_time - start_time) * 1000)
                
                # Pop execution context
                if self._execution_stack:
                    context = self._execution_stack.pop()
                else:
                    context = {"type": "chain", "chain_type": "unknown"}
                
                # Create event
                event = self._create_event(
                    event_type=EventType.TASK_COMPLETE,
                    message=f"Chain completed: {context.get('chain_type', 'unknown')}",
                    component="chain",
                    operation="complete",
                    data={
                        "chain_type": context.get("chain_type", "unknown"),
                        "outputs": outputs,
                        "run_id": run_id,
                        "parent_run_id": parent_run_id
                    },
                    duration_ms=duration_ms,
                    correlation_id=run_id,
                    parent_event_id=parent_run_id
                )
                
                self._queue_event_safe(event)
                
        except Exception as e:
            self.logger.error(f"Error in on_chain_end: {str(e)}")
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], 
                      run_id: Optional[str] = None, parent_run_id: Optional[str] = None,
                      **kwargs: Any) -> None:
        """Called when a chain encounters an error."""
        try:
            with self._lock:
                end_time = time.time()
                start_time = self._chain_start_times.pop(run_id or "unknown", end_time)
                duration_ms = int((end_time - start_time) * 1000)
                
                # Pop execution context
                if self._execution_stack:
                    context = self._execution_stack.pop()
                else:
                    context = {"type": "chain", "chain_type": "unknown"}
                
                # Create error event
                event = self._create_event(
                    event_type=EventType.TASK_FAIL,
                    message=f"Chain failed: {context.get('chain_type', 'unknown')}",
                    component="chain",
                    operation="error",
                    severity=EventSeverity.ERROR,
                    data={
                        "chain_type": context.get("chain_type", "unknown"),
                        "run_id": run_id,
                        "parent_run_id": parent_run_id
                    },
                    duration_ms=duration_ms,
                    correlation_id=run_id,
                    parent_event_id=parent_run_id
                )
                
                # Set error information
                event.set_error(
                    error_type=type(error).__name__,
                    error_message=str(error),
                    stack_trace=traceback.format_exc()
                )
                
                self._queue_event_safe(event)
                
        except Exception as e:
            self.logger.error(f"Error in on_chain_error: {str(e)}")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                    run_id: Optional[str] = None, parent_run_id: Optional[str] = None,
                    **kwargs: Any) -> None:
        """Called when an LLM starts generating."""
        try:
            with self._lock:
                start_time = time.time()
                self._llm_start_times[run_id or "unknown"] = start_time
                
                # Create event
                event = self._create_event(
                    event_type=EventType.DECISION_START,
                    message=f"LLM generation started: {serialized.get('name', 'unknown')}",
                    component="llm",
                    operation="generate",
                    data={
                        "llm_type": serialized.get("name", "unknown"),
                        "prompts": prompts,
                        "prompt_count": len(prompts),
                        "serialized": serialized,
                        "run_id": run_id,
                        "parent_run_id": parent_run_id
                    },
                    correlation_id=run_id,
                    parent_event_id=parent_run_id
                )
                
                self._queue_event_safe(event)
                
        except Exception as e:
            self.logger.error(f"Error in on_llm_start: {str(e)}")
    
    def on_llm_end(self, response: LLMResult, run_id: Optional[str] = None,
                  parent_run_id: Optional[str] = None, **kwargs: Any) -> None:
        """Called when an LLM finishes generating."""
        try:
            with self._lock:
                end_time = time.time()
                start_time = self._llm_start_times.pop(run_id or "unknown", end_time)
                duration_ms = int((end_time - start_time) * 1000)
                
                # Extract response information
                generations = getattr(response, 'generations', [])
                llm_output = getattr(response, 'llm_output', {})
                
                # Create event
                event = self._create_event(
                    event_type=EventType.DECISION_COMPLETE,
                    message="LLM generation completed",
                    component="llm",
                    operation="complete",
                    data={
                        "generations_count": len(generations),
                        "llm_output": llm_output,
                        "token_usage": llm_output.get("token_usage", {}),
                        "run_id": run_id,
                        "parent_run_id": parent_run_id
                    },
                    duration_ms=duration_ms,
                    correlation_id=run_id,
                    parent_event_id=parent_run_id
                )
                
                # Add token usage as performance metrics
                token_usage = llm_output.get("token_usage", {})
                if token_usage:
                    event.metadata["token_usage"] = token_usage
                
                self._queue_event_safe(event)
                
        except Exception as e:
            self.logger.error(f"Error in on_llm_end: {str(e)}")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt],
                    run_id: Optional[str] = None, parent_run_id: Optional[str] = None,
                    **kwargs: Any) -> None:
        """Called when an LLM encounters an error."""
        try:
            with self._lock:
                end_time = time.time()
                start_time = self._llm_start_times.pop(run_id or "unknown", end_time)
                duration_ms = int((end_time - start_time) * 1000)
                
                # Create error event
                event = self._create_event(
                    event_type=EventType.AGENT_ERROR,
                    message="LLM generation failed",
                    component="llm",
                    operation="error",
                    severity=EventSeverity.ERROR,
                    data={
                        "run_id": run_id,
                        "parent_run_id": parent_run_id
                    },
                    duration_ms=duration_ms,
                    correlation_id=run_id,
                    parent_event_id=parent_run_id
                )
                
                # Set error information
                event.set_error(
                    error_type=type(error).__name__,
                    error_message=str(error),
                    stack_trace=traceback.format_exc()
                )
                
                self._queue_event_safe(event)
                
        except Exception as e:
            self.logger.error(f"Error in on_llm_error: {str(e)}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str,
                     run_id: Optional[str] = None, parent_run_id: Optional[str] = None,
                     **kwargs: Any) -> None:
        """Called when a tool starts running."""
        try:
            with self._lock:
                start_time = time.time()
                self._tool_start_times[run_id or "unknown"] = start_time
                
                # Create event
                event = self._create_event(
                    event_type=EventType.TOOL_CALL,
                    message=f"Tool started: {serialized.get('name', 'unknown')}",
                    component="tool",
                    operation="start",
                    data={
                        "tool_name": serialized.get("name", "unknown"),
                        "input": input_str,
                        "serialized": serialized,
                        "run_id": run_id,
                        "parent_run_id": parent_run_id
                    },
                    correlation_id=run_id,
                    parent_event_id=parent_run_id
                )
                
                self._queue_event_safe(event)
                
        except Exception as e:
            self.logger.error(f"Error in on_tool_start: {str(e)}")
    
    def on_tool_end(self, output: str, run_id: Optional[str] = None,
                   parent_run_id: Optional[str] = None, **kwargs: Any) -> None:
        """Called when a tool finishes running."""
        try:
            with self._lock:
                end_time = time.time()
                start_time = self._tool_start_times.pop(run_id or "unknown", end_time)
                duration_ms = int((end_time - start_time) * 1000)
                
                # Create event
                event = self._create_event(
                    event_type=EventType.TOOL_RESPONSE,
                    message="Tool completed",
                    component="tool",
                    operation="complete",
                    data={
                        "output": output,
                        "output_length": len(output) if output else 0,
                        "run_id": run_id,
                        "parent_run_id": parent_run_id
                    },
                    duration_ms=duration_ms,
                    correlation_id=run_id,
                    parent_event_id=parent_run_id
                )
                
                self._queue_event_safe(event)
                
        except Exception as e:
            self.logger.error(f"Error in on_tool_end: {str(e)}")
    
    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt],
                     run_id: Optional[str] = None, parent_run_id: Optional[str] = None,
                     **kwargs: Any) -> None:
        """Called when a tool encounters an error."""
        try:
            with self._lock:
                end_time = time.time()
                start_time = self._tool_start_times.pop(run_id or "unknown", end_time)
                duration_ms = int((end_time - start_time) * 1000)
                
                # Create error event
                event = self._create_event(
                    event_type=EventType.AGENT_ERROR,
                    message="Tool execution failed",
                    component="tool",
                    operation="error",
                    severity=EventSeverity.ERROR,
                    data={
                        "run_id": run_id,
                        "parent_run_id": parent_run_id
                    },
                    duration_ms=duration_ms,
                    correlation_id=run_id,
                    parent_event_id=parent_run_id
                )
                
                # Set error information
                event.set_error(
                    error_type=type(error).__name__,
                    error_message=str(error),
                    stack_trace=traceback.format_exc()
                )
                
                self._queue_event_safe(event)
                
        except Exception as e:
            self.logger.error(f"Error in on_tool_error: {str(e)}")
    
    def on_agent_action(self, action: AgentAction, run_id: Optional[str] = None,
                       parent_run_id: Optional[str] = None, **kwargs: Any) -> None:
        """Called when an agent takes an action."""
        try:
            # Create event
            event = self._create_event(
                event_type=EventType.ACTION_START,
                message=f"Agent action: {getattr(action, 'tool', 'unknown')}",
                component="agent",
                operation="action",
                data={
                    "tool": getattr(action, 'tool', 'unknown'),
                    "tool_input": getattr(action, 'tool_input', {}),
                    "log": getattr(action, 'log', ''),
                    "run_id": run_id,
                    "parent_run_id": parent_run_id
                },
                correlation_id=run_id,
                parent_event_id=parent_run_id
            )
            
            # Extract reasoning from log if available
            log = getattr(action, 'log', '')
            if log:
                event.metadata["reasoning_trace"] = log
            
            self._queue_event_safe(event)
            
        except Exception as e:
            self.logger.error(f"Error in on_agent_action: {str(e)}")
    
    def on_agent_finish(self, finish: AgentFinish, run_id: Optional[str] = None,
                       parent_run_id: Optional[str] = None, **kwargs: Any) -> None:
        """Called when an agent finishes."""
        try:
            # Create event
            event = self._create_event(
                event_type=EventType.ACTION_COMPLETE,
                message="Agent finished",
                component="agent",
                operation="finish",
                data={
                    "return_values": getattr(finish, 'return_values', {}),
                    "log": getattr(finish, 'log', ''),
                    "run_id": run_id,
                    "parent_run_id": parent_run_id
                },
                correlation_id=run_id,
                parent_event_id=parent_run_id
            )
            
            # Extract reasoning from log if available
            log = getattr(finish, 'log', '')
            if log:
                event.metadata["reasoning_trace"] = log
            
            self._queue_event_safe(event)
            
        except Exception as e:
            self.logger.error(f"Error in on_agent_finish: {str(e)}")
    
    def on_text(self, text: str, run_id: Optional[str] = None,
               parent_run_id: Optional[str] = None, **kwargs: Any) -> None:
        """Called when arbitrary text is logged."""
        try:
            # Create event for text output (often contains reasoning)
            event = self._create_event(
                event_type=EventType.CUSTOM,
                message="Text output",
                component="agent",
                operation="text",
                data={
                    "text": text,
                    "text_length": len(text),
                    "run_id": run_id,
                    "parent_run_id": parent_run_id
                },
                correlation_id=run_id,
                parent_event_id=parent_run_id
            )
            
            # Check if text contains reasoning patterns
            if any(keyword in text.lower() for keyword in ["thought:", "action:", "observation:", "final answer:"]):
                event.add_tag("reasoning")
                event.metadata["reasoning_trace"] = text
            
            self._queue_event_safe(event)
            
        except Exception as e:
            self.logger.error(f"Error in on_text: {str(e)}")
    
    def _create_event(self, event_type: EventType, message: str, **kwargs) -> AgentEvent:
        """Create a standardized agent event."""
        return AgentEvent(
            event_type=event_type,
            agent_id=self.agent_id,
            session_id=self.session_id,
            message=message,
            framework="langchain",
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


class LangChainInstrumentor(BaseInstrumentor):
    """
    LangChain-specific instrumentor for monitoring agent execution.
    
    This instrumentor integrates with LangChain's callback system to capture
    chain execution events, reasoning traces, and performance metrics.
    """
    
    def __init__(self, **kwargs):
        """Initialize the LangChain instrumentor."""
        if not LANGCHAIN_AVAILABLE:
            raise InstrumentationError(
                "LangChain is not available. Please install it with: pip install langchain"
            )
        
        super().__init__(**kwargs)
        
        # LangChain-specific configuration
        self._callback_handlers: Dict[str, LangChainCallbackHandler] = {}
        self._callback_lock = threading.RLock()
        
        # Memory and context tracking
        self._memory_usage: Dict[str, Dict[str, Any]] = {}
        self._context_windows: Dict[str, List[str]] = {}
        
        self.logger.info("LangChain instrumentor initialized")
    
    async def start_monitoring(self, agent_id: str, config: Dict[str, Any]) -> str:
        """
        Start monitoring a LangChain agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration parameters including:
                - monitor_memory: Whether to track memory usage (default: True)
                - monitor_context: Whether to track context windows (default: True)
                - callback_config: Additional callback configuration
        
        Returns:
            session_id: Unique identifier for the monitoring session
        """
        try:
            # Create monitoring session
            session = self._create_session(agent_id, config)
            
            # Create callback handler
            callback_handler = LangChainCallbackHandler(
                instrumentor=self,
                session_id=session.session_id,
                agent_id=agent_id
            )
            
            with self._callback_lock:
                self._callback_handlers[session.session_id] = callback_handler
            
            # Initialize memory and context tracking
            if config.get("monitor_memory", True):
                self._memory_usage[session.session_id] = {
                    "start_time": datetime.utcnow(),
                    "memory_snapshots": []
                }
            
            if config.get("monitor_context", True):
                self._context_windows[session.session_id] = []
            
            # Create start event
            start_event = self.create_event(
                event_type=EventType.AGENT_START,
                agent_id=agent_id,
                session_id=session.session_id,
                message=f"Started monitoring LangChain agent: {agent_id}",
                component="instrumentor",
                operation="start",
                data={"configuration": config}
            )
            
            await self._queue_event(start_event)
            
            self.logger.info(f"Started monitoring LangChain agent {agent_id} (session: {session.session_id})")
            return session.session_id
            
        except Exception as e:
            raise InstrumentationError(f"Failed to start monitoring: {str(e)}")
    
    async def stop_monitoring(self, session_id: str) -> MonitoringSummary:
        """
        Stop monitoring a LangChain agent session.
        
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
            
            # Remove callback handler
            with self._callback_lock:
                callback_handler = self._callback_handlers.pop(session_id, None)
            
            # Clean up memory and context tracking
            memory_data = self._memory_usage.pop(session_id, {})
            context_data = self._context_windows.pop(session_id, [])
            
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
                message=f"Stopped monitoring LangChain agent: {ended_session.agent_id}",
                component="instrumentor",
                operation="stop",
                duration_ms=duration_ms,
                data={
                    "memory_snapshots_count": len(memory_data.get("memory_snapshots", [])),
                    "context_updates_count": len(context_data)
                }
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
                    "memory_usage": memory_data,
                    "context_updates": len(context_data),
                    **self.get_performance_metrics()
                }
            )
            
            self.logger.info(f"Stopped monitoring session {session_id}")
            return summary
            
        except Exception as e:
            raise InstrumentationError(f"Failed to stop monitoring: {str(e)}")
    
    async def capture_event(self, event: AgentEvent) -> None:
        """
        Capture an agent event from LangChain.
        
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
            
            # Process memory and context updates
            await self._process_memory_context(event)
            
            # Queue event for processing
            await self._queue_event(event)
            
        except Exception as e:
            raise EventProcessingError(f"Failed to capture event: {str(e)}")
    
    def get_supported_events(self) -> List[EventType]:
        """Get the list of event types supported by LangChain instrumentor."""
        return [
            EventType.AGENT_START,
            EventType.AGENT_STOP,
            EventType.AGENT_ERROR,
            EventType.TASK_START,
            EventType.TASK_COMPLETE,
            EventType.TASK_FAIL,
            EventType.DECISION_START,
            EventType.DECISION_COMPLETE,
            EventType.TOOL_CALL,
            EventType.TOOL_RESPONSE,
            EventType.ACTION_START,
            EventType.ACTION_COMPLETE,
            EventType.MEMORY_READ,
            EventType.MEMORY_WRITE,
            EventType.CONTEXT_UPDATE,
            EventType.CUSTOM
        ]
    
    def get_framework_name(self) -> str:
        """Get the framework name."""
        return "langchain"
    
    def get_callback_handler(self, session_id: str) -> Optional[LangChainCallbackHandler]:
        """
        Get the callback handler for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            LangChainCallbackHandler or None if not found
        """
        with self._callback_lock:
            return self._callback_handlers.get(session_id)
    
    async def _process_memory_context(self, event: AgentEvent) -> None:
        """Process memory and context information from events."""
        try:
            session_id = event.session_id
            
            # Track memory usage if enabled
            if session_id in self._memory_usage:
                memory_info = {
                    "timestamp": event.timestamp,
                    "event_type": event.event_type.value,
                    "memory_usage_mb": event.memory_usage_mb
                }
                self._memory_usage[session_id]["memory_snapshots"].append(memory_info)
            
            # Track context updates
            if session_id in self._context_windows:
                if event.event_type in [EventType.CONTEXT_UPDATE, EventType.MEMORY_READ, EventType.MEMORY_WRITE]:
                    context_info = {
                        "timestamp": event.timestamp,
                        "event_type": event.event_type.value,
                        "data": event.data
                    }
                    self._context_windows[session_id].append(context_info)
            
            # Extract reasoning traces
            if "reasoning_trace" in event.metadata:
                reasoning_event = self.create_event(
                    event_type=EventType.CUSTOM,
                    agent_id=event.agent_id,
                    session_id=event.session_id,
                    message="Reasoning trace captured",
                    component="reasoning",
                    operation="trace",
                    data={
                        "reasoning_trace": event.metadata["reasoning_trace"],
                        "source_event_id": event.event_id
                    },
                    parent_event_id=event.event_id
                )
                reasoning_event.add_tag("reasoning")
                await self._queue_event(reasoning_event)
            
        except Exception as e:
            self.logger.error(f"Error processing memory/context: {str(e)}")
    
    async def monitor_memory_usage(self, session_id: str, memory_data: Dict[str, Any]) -> None:
        """
        Monitor memory usage for a LangChain session.
        
        Args:
            session_id: Session identifier
            memory_data: Memory usage information
        """
        try:
            if session_id not in self._memory_usage:
                return
            
            # Create memory usage event
            memory_event = self.create_event(
                event_type=EventType.MEMORY_READ,
                agent_id=self._get_session(session_id).agent_id if self._get_session(session_id) else "unknown",
                session_id=session_id,
                message="Memory usage monitored",
                component="memory",
                operation="usage_check",
                data=memory_data
            )
            
            await self._queue_event(memory_event)
            
        except Exception as e:
            self.logger.error(f"Error monitoring memory usage: {str(e)}")
    
    async def monitor_context_window(self, session_id: str, context_data: Dict[str, Any]) -> None:
        """
        Monitor context window changes for a LangChain session.
        
        Args:
            session_id: Session identifier
            context_data: Context window information
        """
        try:
            if session_id not in self._context_windows:
                return
            
            # Create context update event
            context_event = self.create_event(
                event_type=EventType.CONTEXT_UPDATE,
                agent_id=self._get_session(session_id).agent_id if self._get_session(session_id) else "unknown",
                session_id=session_id,
                message="Context window updated",
                component="context",
                operation="window_update",
                data=context_data
            )
            
            await self._queue_event(context_event)
            
        except Exception as e:
            self.logger.error(f"Error monitoring context window: {str(e)}")
    
    def extract_reasoning_from_chain_output(self, chain_output: str) -> Optional[str]:
        """
        Extract reasoning patterns from chain output.
        
        Args:
            chain_output: Output text from chain execution
            
        Returns:
            Extracted reasoning trace or None
        """
        try:
            # Common reasoning patterns in LangChain outputs
            reasoning_patterns = [
                r"Thought:\s*(.+?)(?=\n|Action:|$)",
                r"Reasoning:\s*(.+?)(?=\n|Action:|$)",
                r"Analysis:\s*(.+?)(?=\n|Action:|$)",
                r"I need to\s*(.+?)(?=\n|Action:|$)",
                r"Let me\s*(.+?)(?=\n|Action:|$)"
            ]
            
            import re
            extracted_reasoning = []
            
            for pattern in reasoning_patterns:
                matches = re.findall(pattern, chain_output, re.IGNORECASE | re.MULTILINE)
                extracted_reasoning.extend(matches)
            
            if extracted_reasoning:
                return " | ".join(extracted_reasoning)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting reasoning: {str(e)}")
            return None
    
    def get_memory_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get memory usage summary for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Memory usage summary
        """
        try:
            if session_id not in self._memory_usage:
                return {}
            
            memory_data = self._memory_usage[session_id]
            snapshots = memory_data.get("memory_snapshots", [])
            
            if not snapshots:
                return {"total_snapshots": 0}
            
            memory_values = [s.get("memory_usage_mb", 0) for s in snapshots if s.get("memory_usage_mb")]
            
            return {
                "total_snapshots": len(snapshots),
                "peak_memory_mb": max(memory_values) if memory_values else 0,
                "average_memory_mb": sum(memory_values) / len(memory_values) if memory_values else 0,
                "memory_trend": "increasing" if len(memory_values) > 1 and memory_values[-1] > memory_values[0] else "stable"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory summary: {str(e)}")
            return {}
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get context window summary for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context window summary
        """
        try:
            if session_id not in self._context_windows:
                return {}
            
            context_updates = self._context_windows[session_id]
            
            return {
                "total_updates": len(context_updates),
                "update_types": list(set(u.get("event_type", "unknown") for u in context_updates)),
                "last_update": context_updates[-1].get("timestamp") if context_updates else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting context summary: {str(e)}")
            return {}