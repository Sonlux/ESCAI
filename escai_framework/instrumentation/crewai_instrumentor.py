"""
CrewAI instrumentor for the ESCAI framework.

This module provides monitoring capabilities for CrewAI workflows,
capturing task delegation, crew collaboration patterns, and role-based performance metrics.
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

# CrewAI imports (with fallback for optional dependency)
try:
    from crewai import Agent, Task, Crew
    from crewai.agent import Agent as BaseAgent
    from crewai.task import Task as BaseTask
    from crewai.crew import Crew as BaseCrew
    CREWAI_AVAILABLE = True
except ImportError:
    # Create mock classes for type hints when CrewAI is not available
    Agent = object
    Task = object
    Crew = object
    BaseAgent = object
    BaseTask = object
    BaseCrew = object
    CREWAI_AVAILABLE = False


class CrewAIWorkflowMonitor:
    """
    Workflow monitor for capturing CrewAI execution patterns.
    
    This class monitors task delegation, crew collaboration,
    and role-based performance in CrewAI workflows.
    """
    
    def __init__(self, instrumentor: 'CrewAIInstrumentor', session_id: str, agent_id: str):
        """
        Initialize the workflow monitor.
        
        Args:
            instrumentor: The CrewAI instrumentor instance
            session_id: Monitoring session identifier
            agent_id: Agent identifier
        """
        self.instrumentor = weakref.ref(instrumentor)  # Avoid circular references
        self.session_id = session_id
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.CrewAIWorkflowMonitor")
        
        # Track workflow state
        self._task_assignments: Dict[str, Dict[str, Any]] = {}
        self._agent_performance: Dict[str, Dict[str, Any]] = {}
        self._collaboration_patterns: List[Dict[str, Any]] = []
        self._workflow_hierarchy: Dict[str, List[str]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
    
    def monitor_task_execution(self, original_execute: Callable, task: Any, agent: Any) -> Any:
        """
        Monitor task execution by an agent.
        
        Args:
            original_execute: Original task execution method
            task: Task being executed
            agent: Agent executing the task
            
        Returns:
            Result of task execution
        """
        try:
            with self._lock:
                # Extract task and agent information
                task_id = getattr(task, 'id', f'task_{id(task)}')
                task_description = getattr(task, 'description', 'Unknown task')
                agent_name = getattr(agent, 'role', getattr(agent, 'name', 'unknown_agent'))
                agent_role = getattr(agent, 'role', 'unknown_role')
                
                # Record task assignment
                assignment_info = {
                    "task_id": task_id,
                    "task_description": task_description,
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                    "assigned_at": datetime.utcnow(),
                    "status": "started"
                }
                self._task_assignments[task_id] = assignment_info
                
                # Initialize agent performance tracking
                if agent_name not in self._agent_performance:
                    self._agent_performance[agent_name] = {
                        "role": agent_role,
                        "tasks_completed": 0,
                        "tasks_failed": 0,
                        "total_execution_time": 0.0,
                        "average_execution_time": 0.0,
                        "skills_used": set()
                    }
                
                # Create task start event
                start_event = self._create_event(
                    event_type=EventType.TASK_START,
                    message=f"Task started: {task_description[:100]}...",
                    component="task",
                    operation="execute",
                    data={
                        "task_id": task_id,
                        "task_description": task_description,
                        "assigned_agent": agent_name,
                        "agent_role": agent_role,
                        "task_type": type(task).__name__
                    }
                )
                
                start_event.metadata["task_assignment"] = assignment_info
                self._queue_event_safe(start_event)
            
            # Execute the task
            start_time = time.time()
            result = original_execute(agent)
            execution_time = time.time() - start_time
            
            with self._lock:
                # Update task assignment status
                self._task_assignments[task_id]["status"] = "completed"
                self._task_assignments[task_id]["completed_at"] = datetime.utcnow()
                self._task_assignments[task_id]["execution_time"] = execution_time
                self._task_assignments[task_id]["result"] = str(result)[:500] if result else None
                
                # Update agent performance metrics
                agent_perf = self._agent_performance[agent_name]
                agent_perf["tasks_completed"] += 1
                agent_perf["total_execution_time"] += execution_time
                agent_perf["average_execution_time"] = (
                    agent_perf["total_execution_time"] / 
                    (agent_perf["tasks_completed"] + agent_perf["tasks_failed"])
                )
                
                # Extract skills used (if available)
                if hasattr(task, 'tools') and task.tools:
                    for tool in task.tools:
                        tool_name = getattr(tool, 'name', str(tool))
                        agent_perf["skills_used"].add(tool_name)
                
                # Create task completion event
                complete_event = self._create_event(
                    event_type=EventType.TASK_COMPLETE,
                    message=f"Task completed: {task_description[:100]}...",
                    component="task",
                    operation="complete",
                    data={
                        "task_id": task_id,
                        "assigned_agent": agent_name,
                        "execution_time_ms": int(execution_time * 1000),
                        "result_length": len(str(result)) if result else 0,
                        "success": True
                    },
                    duration_ms=int(execution_time * 1000)
                )
                
                complete_event.metadata["performance_metrics"] = {
                    "execution_time": execution_time,
                    "agent_performance": agent_perf.copy()
                }
                self._queue_event_safe(complete_event)
            
            return result
            
        except Exception as e:
            # Handle task execution failure
            with self._lock:
                if task_id in self._task_assignments:
                    self._task_assignments[task_id]["status"] = "failed"
                    self._task_assignments[task_id]["error"] = str(e)
                
                if agent_name in self._agent_performance:
                    self._agent_performance[agent_name]["tasks_failed"] += 1
                
                # Create task failure event
                error_event = self._create_event(
                    event_type=EventType.TASK_FAIL,
                    message=f"Task failed: {task_description[:100]}...",
                    component="task",
                    operation="error",
                    severity=EventSeverity.ERROR,
                    data={
                        "task_id": task_id,
                        "assigned_agent": agent_name,
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
    
    def monitor_crew_kickoff(self, original_kickoff: Callable, crew: Any) -> Any:
        """
        Monitor crew workflow kickoff.
        
        Args:
            original_kickoff: Original crew kickoff method
            crew: Crew instance
            
        Returns:
            Result of crew execution
        """
        try:
            # Extract crew information
            crew_name = getattr(crew, 'name', f'crew_{id(crew)}')
            agents = getattr(crew, 'agents', [])
            tasks = getattr(crew, 'tasks', [])
            
            with self._lock:
                # Build workflow hierarchy
                self._workflow_hierarchy[crew_name] = {
                    "agents": [getattr(agent, 'role', f'agent_{id(agent)}') for agent in agents],
                    "tasks": [getattr(task, 'id', f'task_{id(task)}') for task in tasks],
                    "started_at": datetime.utcnow()
                }
            
            # Create crew start event
            crew_start_event = self._create_event(
                event_type=EventType.AGENT_START,
                message=f"Crew workflow started: {crew_name}",
                component="crew",
                operation="kickoff",
                data={
                    "crew_name": crew_name,
                    "agents_count": len(agents),
                    "tasks_count": len(tasks),
                    "agent_roles": [getattr(agent, 'role', 'unknown') for agent in agents],
                    "workflow_type": "crew_execution"
                }
            )
            
            self._queue_event_safe(crew_start_event)
            
            # Execute crew workflow
            start_time = time.time()
            result = original_kickoff()
            execution_time = time.time() - start_time
            
            with self._lock:
                # Update workflow hierarchy
                if crew_name in self._workflow_hierarchy:
                    self._workflow_hierarchy[crew_name]["completed_at"] = datetime.utcnow()
                    self._workflow_hierarchy[crew_name]["total_execution_time"] = execution_time
                
                # Analyze collaboration patterns
                self._analyze_collaboration_patterns(crew_name, agents, tasks)
            
            # Create crew completion event
            crew_complete_event = self._create_event(
                event_type=EventType.AGENT_STOP,
                message=f"Crew workflow completed: {crew_name}",
                component="crew",
                operation="complete",
                data={
                    "crew_name": crew_name,
                    "total_execution_time_ms": int(execution_time * 1000),
                    "result_summary": str(result)[:500] if result else None,
                    "success": True
                },
                duration_ms=int(execution_time * 1000)
            )
            
            crew_complete_event.metadata["workflow_summary"] = self.get_workflow_summary()
            self._queue_event_safe(crew_complete_event)
            
            return result
            
        except Exception as e:
            # Handle crew execution failure
            error_event = self._create_event(
                event_type=EventType.AGENT_ERROR,
                message=f"Crew workflow failed: {crew_name}",
                component="crew",
                operation="error",
                severity=EventSeverity.ERROR,
                data={
                    "crew_name": crew_name,
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
    
    def monitor_agent_action(self, original_action: Callable, agent: Any, task: Any) -> Any:
        """
        Monitor individual agent actions within tasks.
        
        Args:
            original_action: Original agent action method
            agent: Agent performing the action
            task: Task context
            
        Returns:
            Result of agent action
        """
        try:
            agent_name = getattr(agent, 'role', getattr(agent, 'name', 'unknown_agent'))
            task_id = getattr(task, 'id', f'task_{id(task)}')
            
            # Create action start event
            action_event = self._create_event(
                event_type=EventType.ACTION_START,
                message=f"Agent action started: {agent_name}",
                component="agent",
                operation="action",
                data={
                    "agent_name": agent_name,
                    "task_id": task_id,
                    "action_type": "task_execution"
                }
            )
            
            self._queue_event_safe(action_event)
            
            # Execute action
            start_time = time.time()
            result = original_action(task)
            execution_time = time.time() - start_time
            
            # Create action completion event
            complete_event = self._create_event(
                event_type=EventType.ACTION_COMPLETE,
                message=f"Agent action completed: {agent_name}",
                component="agent",
                operation="complete",
                data={
                    "agent_name": agent_name,
                    "task_id": task_id,
                    "execution_time_ms": int(execution_time * 1000),
                    "success": True
                },
                duration_ms=int(execution_time * 1000)
            )
            
            self._queue_event_safe(complete_event)
            
            return result
            
        except Exception as e:
            # Handle action failure
            error_event = self._create_event(
                event_type=EventType.AGENT_ERROR,
                message=f"Agent action failed: {agent_name}",
                component="agent",
                operation="error",
                severity=EventSeverity.ERROR,
                data={
                    "agent_name": agent_name,
                    "task_id": task_id,
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
    
    def _analyze_collaboration_patterns(self, crew_name: str, agents: List, tasks: List) -> None:
        """Analyze collaboration patterns within the crew."""
        try:
            with self._lock:
                # Analyze task distribution
                task_distribution: Dict[str, int] = {}
                for task_id, assignment in self._task_assignments.items():
                    agent_name = assignment["agent_name"]
                    if agent_name not in task_distribution:
                        task_distribution[agent_name] = 0
                    task_distribution[agent_name] += 1
                
                # Analyze agent utilization
                agent_utilization: Dict[str, float] = {}
                for agent_name, perf in self._agent_performance.items():
                    total_tasks = perf["tasks_completed"] + perf["tasks_failed"]
                    success_rate = perf["tasks_completed"] / total_tasks if total_tasks > 0 else 0
                    agent_utilization[agent_name] = {
                        "total_tasks": total_tasks,
                        "success_rate": success_rate,
                        "average_execution_time": perf["average_execution_time"],
                        "skills_count": len(perf["skills_used"])
                    }
                
                # Store collaboration pattern
                collaboration_pattern = {
                    "crew_name": crew_name,
                    "timestamp": datetime.utcnow(),
                    "task_distribution": task_distribution,
                    "agent_utilization": agent_utilization,
                    "total_agents": len(agents),
                    "total_tasks": len(tasks)
                }
                
                self._collaboration_patterns.append(collaboration_pattern)
                
                # Create collaboration analysis event
                collab_event = self._create_event(
                    event_type=EventType.CUSTOM,
                    message="Collaboration pattern analyzed",
                    component="collaboration",
                    operation="analysis",
                    data=collaboration_pattern
                )
                
                collab_event.add_tag("collaboration")
                self._queue_event_safe(collab_event)
                
        except Exception as e:
            self.logger.error(f"Error analyzing collaboration patterns: {str(e)}")
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow execution."""
        with self._lock:
            return {
                "total_tasks": len(self._task_assignments),
                "completed_tasks": len([t for t in self._task_assignments.values() if t["status"] == "completed"]),
                "failed_tasks": len([t for t in self._task_assignments.values() if t["status"] == "failed"]),
                "total_agents": len(self._agent_performance),
                "collaboration_patterns": len(self._collaboration_patterns),
                "workflow_hierarchy": self._workflow_hierarchy.copy(),
                "agent_performance_summary": {
                    name: {
                        "tasks_completed": perf["tasks_completed"],
                        "tasks_failed": perf["tasks_failed"],
                        "average_execution_time": perf["average_execution_time"],
                        "skills_count": len(perf["skills_used"])
                    }
                    for name, perf in self._agent_performance.items()
                }
            }
    
    def _create_event(self, event_type: EventType, message: str, **kwargs) -> AgentEvent:
        """Create a standardized agent event."""
        return AgentEvent(
            event_type=event_type,
            agent_id=self.agent_id,
            session_id=self.session_id,
            message=message,
            framework="crewai",
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


class CrewAIInstrumentor(BaseInstrumentor):
    """
    CrewAI-specific instrumentor for monitoring workflow execution.
    
    This instrumentor monitors CrewAI workflows by tracking task delegation,
    crew collaboration patterns, and role-based performance metrics.
    """
    
    def __init__(self, **kwargs):
        """Initialize the CrewAI instrumentor."""
        if not CREWAI_AVAILABLE:
            raise InstrumentationError(
                "CrewAI is not available. Please install it with: pip install crewai"
            )
        
        super().__init__(**kwargs)
        
        # CrewAI-specific configuration
        self._workflow_monitors: Dict[str, CrewAIWorkflowMonitor] = {}
        self._monitor_lock = threading.RLock()
        
        # Crew and task tracking
        self._monitored_crews: Dict[str, Set[Any]] = {}  # session_id -> set of crews
        self._monitored_tasks: Dict[str, Set[Any]] = {}  # session_id -> set of tasks
        self._monitored_agents: Dict[str, Set[Any]] = {}  # session_id -> set of agents
        self._original_methods: Dict[str, Dict[str, Callable]] = {}  # session_id -> method_name -> original_method
        
        self.logger.info("CrewAI instrumentor initialized")
    
    async def start_monitoring(self, agent_id: str, config: Dict[str, Any]) -> str:
        """
        Start monitoring CrewAI workflows.
        
        Args:
            agent_id: Unique identifier for the workflow/system
            config: Configuration parameters including:
                - crews: List of Crew instances to monitor
                - agents: List of Agent instances to monitor
                - tasks: List of Task instances to monitor
                - monitor_collaboration: Whether to track collaboration patterns (default: True)
                - monitor_performance: Whether to track performance metrics (default: True)
        
        Returns:
            session_id: Unique identifier for the monitoring session
        """
        try:
            # Create monitoring session
            session = self._create_session(agent_id, config)
            
            # Create workflow monitor
            monitor = CrewAIWorkflowMonitor(
                instrumentor=self,
                session_id=session.session_id,
                agent_id=agent_id
            )
            
            with self._monitor_lock:
                self._workflow_monitors[session.session_id] = monitor
            
            # Initialize tracking sets
            self._monitored_crews[session.session_id] = set()
            self._monitored_tasks[session.session_id] = set()
            self._monitored_agents[session.session_id] = set()
            self._original_methods[session.session_id] = {}
            
            # Set up monitoring for provided crews
            crews = config.get("crews", [])
            for crew in crews:
                await self._instrument_crew(session.session_id, crew, monitor)
            
            # Set up monitoring for individual agents
            agents = config.get("agents", [])
            for agent in agents:
                await self._instrument_agent(session.session_id, agent, monitor)
            
            # Set up monitoring for individual tasks
            tasks = config.get("tasks", [])
            for task in tasks:
                await self._instrument_task(session.session_id, task, monitor)
            
            # Create start event
            start_event = self.create_event(
                event_type=EventType.AGENT_START,
                agent_id=agent_id,
                session_id=session.session_id,
                message=f"Started monitoring CrewAI workflow: {agent_id}",
                component="instrumentor",
                operation="start",
                data={
                    "configuration": config,
                    "crews_count": len(crews),
                    "agents_count": len(agents),
                    "tasks_count": len(tasks)
                }
            )
            
            await self._queue_event(start_event)
            
            self.logger.info(f"Started monitoring CrewAI workflow {agent_id} (session: {session.session_id})")
            return session.session_id
            
        except Exception as e:
            raise InstrumentationError(f"Failed to start monitoring: {str(e)}")
    
    async def stop_monitoring(self, session_id: str) -> MonitoringSummary:
        """
        Stop monitoring CrewAI workflows.
        
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
            
            # Get workflow summary
            workflow_summary: Dict[str, Any] = {}
            with self._monitor_lock:
                monitor = self._workflow_monitors.pop(session_id, None)
                if monitor:
                    workflow_summary = monitor.get_workflow_summary()
            
            # Clean up tracking data
            self._monitored_crews.pop(session_id, set())
            self._monitored_tasks.pop(session_id, set())
            self._monitored_agents.pop(session_id, set())
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
                message=f"Stopped monitoring CrewAI workflow: {ended_session.agent_id}",
                component="instrumentor",
                operation="stop",
                duration_ms=duration_ms,
                data=workflow_summary
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
                    "workflow_summary": workflow_summary,
                    **self.get_performance_metrics()
                }
            )
            
            self.logger.info(f"Stopped monitoring session {session_id}")
            return summary
            
        except Exception as e:
            raise InstrumentationError(f"Failed to stop monitoring: {str(e)}")
    
    async def capture_event(self, event: AgentEvent) -> None:
        """
        Capture an agent event from CrewAI.
        
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
            
            # Process workflow and collaboration context
            await self._process_workflow_context(event)
            
            # Queue event for processing
            await self._queue_event(event)
            
        except Exception as e:
            raise EventProcessingError(f"Failed to capture event: {str(e)}")
    
    def get_supported_events(self) -> List[EventType]:
        """Get the list of event types supported by CrewAI instrumentor."""
        return [
            EventType.AGENT_START,
            EventType.AGENT_STOP,
            EventType.AGENT_ERROR,
            EventType.TASK_START,
            EventType.TASK_COMPLETE,
            EventType.TASK_FAIL,
            EventType.ACTION_START,
            EventType.ACTION_COMPLETE,
            EventType.DECISION_START,
            EventType.DECISION_COMPLETE,
            EventType.CUSTOM
        ]
    
    def get_framework_name(self) -> str:
        """Get the framework name."""
        return "crewai"
    
    def get_workflow_monitor(self, session_id: str) -> Optional[CrewAIWorkflowMonitor]:
        """
        Get the workflow monitor for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            CrewAIWorkflowMonitor or None if not found
        """
        with self._monitor_lock:
            return self._workflow_monitors.get(session_id)
    
    async def _instrument_crew(self, session_id: str, crew: Any, 
                             monitor: CrewAIWorkflowMonitor) -> None:
        """Instrument a Crew instance."""
        try:
            if not hasattr(crew, 'kickoff'):
                self.logger.warning(f"Crew {id(crew)} does not have kickoff method")
                return
            
            # Store original method
            original_kickoff = crew.kickoff
            crew_key = f"crew_{id(crew)}_kickoff"
            self._original_methods[session_id][crew_key] = original_kickoff
            
            # Create wrapped method
            def wrapped_kickoff(*args, **kwargs):
                return monitor.monitor_crew_kickoff(original_kickoff, crew)
            
            # Replace method
            crew.kickoff = wrapped_kickoff
            
            # Track crew
            self._monitored_crews[session_id].add(crew)
            
            # Also instrument agents and tasks within the crew
            if hasattr(crew, 'agents'):
                for agent in crew.agents:
                    await self._instrument_agent(session_id, agent, monitor)
            
            if hasattr(crew, 'tasks'):
                for task in crew.tasks:
                    await self._instrument_task(session_id, task, monitor)
            
            self.logger.debug(f"Instrumented crew with {len(getattr(crew, 'agents', []))} agents and {len(getattr(crew, 'tasks', []))} tasks")
            
        except Exception as e:
            self.logger.error(f"Failed to instrument crew: {str(e)}")
    
    async def _instrument_crew(self, session_id: str, crew: Any,
                             monitor: CrewAIWorkflowMonitor) -> None:
        """Instrument a Crew instance."""
        try:
            if not hasattr(crew, 'kickoff'):
                self.logger.warning(f"Crew {id(crew)} does not have kickoff method")
                return
            
            # Store original method
            original_kickoff = crew.kickoff
            crew_key = f"crew_{id(crew)}_kickoff"
            self._original_methods[session_id][crew_key] = original_kickoff
            
            # Create wrapped method
            def wrapped_kickoff():
                return monitor.monitor_crew_kickoff(original_kickoff, crew)
            
            # Replace method
            crew.kickoff = wrapped_kickoff
            
            # Track crew
            self._monitored_crews[session_id].add(crew)
            
            # Also instrument agents and tasks within the crew
            if hasattr(crew, 'agents'):
                for agent in crew.agents:
                    await self._instrument_agent(session_id, agent, monitor)
            
            if hasattr(crew, 'tasks'):
                for task in crew.tasks:
                    await self._instrument_task(session_id, task, monitor)
            
            self.logger.debug(f"Instrumented crew with {len(getattr(crew, 'agents', []))} agents and {len(getattr(crew, 'tasks', []))} tasks")
            
        except Exception as e:
            self.logger.error(f"Failed to instrument crew: {str(e)}")
    
    async def _instrument_agent(self, session_id: str, agent: Any,
                              monitor: CrewAIWorkflowMonitor) -> None:
        """Instrument an Agent instance."""
        try:
            if not hasattr(agent, 'execute_task'):
                # Try alternative method names
                if hasattr(agent, 'perform_task'):
                    method_name = 'perform_task'
                elif hasattr(agent, 'run'):
                    method_name = 'run'
                else:
                    self.logger.warning(f"Agent {getattr(agent, 'role', id(agent))} does not have recognizable execution method")
                    return
            else:
                method_name = 'execute_task'
            
            # Store original method
            original_method = getattr(agent, method_name)
            agent_key = f"agent_{id(agent)}_{method_name}"
            self._original_methods[session_id][agent_key] = original_method
            
            # Create wrapped method
            def wrapped_method(task):
                return monitor.monitor_agent_action(original_method, agent, task)
            
            # Replace method
            setattr(agent, method_name, wrapped_method)
            
            # Track agent
            self._monitored_agents[session_id].add(agent)
            
            self.logger.debug(f"Instrumented agent: {getattr(agent, 'role', id(agent))}")
            
        except Exception as e:
            self.logger.error(f"Failed to instrument agent: {str(e)}")
    
    async def _instrument_task(self, session_id: str, task: Any,
                             monitor: CrewAIWorkflowMonitor) -> None:
        """Instrument a Task instance."""
        try:
            if not hasattr(task, 'execute'):
                self.logger.warning(f"Task {getattr(task, 'id', id(task))} does not have execute method")
                return
            
            # Store original method
            original_execute = task.execute
            task_key = f"task_{id(task)}_execute"
            self._original_methods[session_id][task_key] = original_execute
            
            # Create wrapped method
            def wrapped_execute(agent):
                return monitor.monitor_task_execution(original_execute, task, agent)
            
            # Replace method
            task.execute = wrapped_execute
            
            # Track task
            self._monitored_tasks[session_id].add(task)
            
            self.logger.debug(f"Instrumented task: {getattr(task, 'id', id(task))}")
            
        except Exception as e:
            self.logger.error(f"Failed to instrument task: {str(e)}")
    
    async def _restore_original_methods(self, session_id: str) -> None:
        """Restore original methods for all monitored objects."""
        try:
            # Restore crew methods
            for crew in self._monitored_crews.get(session_id, set()):
                crew_key = f"crew_{id(crew)}_kickoff"
                if crew_key in self._original_methods.get(session_id, {}):
                    crew.kickoff = self._original_methods[session_id][crew_key]
            
            # Restore agent methods
            for agent in self._monitored_agents.get(session_id, set()):
                for method_name in ['execute_task', 'perform_task', 'run']:
                    agent_key = f"agent_{id(agent)}_{method_name}"
                    if agent_key in self._original_methods.get(session_id, {}):
                        setattr(agent, method_name, self._original_methods[session_id][agent_key])
            
            # Restore task methods
            for task in self._monitored_tasks.get(session_id, set()):
                task_key = f"task_{id(task)}_execute"
                if task_key in self._original_methods.get(session_id, {}):
                    task.execute = self._original_methods[session_id][task_key]
            
            self.logger.debug(f"Restored original methods for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to restore original methods: {str(e)}")
    
    async def _process_workflow_context(self, event: AgentEvent) -> None:
        """Process workflow and collaboration context from events."""
        try:
            # Extract workflow patterns
            if event.event_type in [EventType.TASK_START, EventType.TASK_COMPLETE]:
                # Create workflow analysis event
                workflow_event = self.create_event(
                    event_type=EventType.CUSTOM,
                    agent_id=event.agent_id,
                    session_id=event.session_id,
                    message="Workflow pattern detected",
                    component="workflow",
                    operation="pattern_analysis",
                    data={
                        "task_id": event.data.get("task_id", ""),
                        "assigned_agent": event.data.get("assigned_agent", ""),
                        "agent_role": event.data.get("agent_role", ""),
                        "source_event_id": event.event_id
                    },
                    parent_event_id=event.event_id
                )
                workflow_event.add_tag("workflow")
                await self._queue_event(workflow_event)
            
            # Extract collaboration patterns
            if "collaboration" in event.tags:
                collab_event = self.create_event(
                    event_type=EventType.CUSTOM,
                    agent_id=event.agent_id,
                    session_id=event.session_id,
                    message="Collaboration pattern detected",
                    component="collaboration",
                    operation="pattern_analysis",
                    data={
                        "collaboration_type": "crew_coordination",
                        "participants": event.data.get("agent_utilization", {}).keys(),
                        "source_event_id": event.event_id
                    },
                    parent_event_id=event.event_id
                )
                collab_event.add_tag("collaboration")
                await self._queue_event(collab_event)
            
        except Exception as e:
            self.logger.error(f"Error processing workflow context: {str(e)}")
    
    def analyze_task_delegation_patterns(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze task delegation patterns for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Task delegation pattern analysis
        """
        try:
            monitor = self.get_workflow_monitor(session_id)
            if not monitor:
                return {}
            
            workflow_summary = monitor.get_workflow_summary()
            
            # Analyze task distribution
            task_assignments = workflow_summary.get("workflow_hierarchy", {})
            agent_performance = workflow_summary.get("agent_performance_summary", {})
            
            # Calculate delegation efficiency
            total_tasks = workflow_summary.get("total_tasks", 0)
            completed_tasks = workflow_summary.get("completed_tasks", 0)
            failed_tasks = workflow_summary.get("failed_tasks", 0)
            
            success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            # Analyze agent workload distribution
            agent_workloads: Dict[str, Dict[str, Any]] = {}
            for agent_name, perf in agent_performance.items():
                total_agent_tasks = perf.get("tasks_completed", 0) + perf.get("tasks_failed", 0)
                agent_workloads[agent_name] = total_agent_tasks
            
            # Calculate workload balance
            if agent_workloads:
                max_workload = max(agent_workloads.values())
                min_workload = min(agent_workloads.values())
                workload_balance = 1 - (max_workload - min_workload) / max_workload if max_workload > 0 else 1
            else:
                workload_balance = 0
            
            return {
                "total_tasks": total_tasks,
                "success_rate": success_rate,
                "agent_workloads": agent_workloads,
                "workload_balance": workload_balance,
                "delegation_efficiency": "high" if success_rate > 0.8 and workload_balance > 0.7 else "medium" if success_rate > 0.6 else "low",
                "task_assignments": task_assignments
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing task delegation patterns: {str(e)}")
            return {}
    
    def analyze_crew_collaboration_patterns(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze crew collaboration patterns for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Crew collaboration pattern analysis
        """
        try:
            monitor = self.get_workflow_monitor(session_id)
            if not monitor:
                return {}
            
            workflow_summary = monitor.get_workflow_summary()
            collaboration_patterns = workflow_summary.get("collaboration_patterns", [])
            
            if not collaboration_patterns:
                return {"total_patterns": 0}
            
            # Analyze collaboration metrics
            total_patterns = len(collaboration_patterns)
            
            # Calculate average collaboration scores
            collaboration_scores = []
            for pattern in collaboration_patterns:
                agent_utilization = pattern.get("agent_utilization", {})
                if agent_utilization:
                    # Calculate collaboration score based on agent utilization balance
                    utilization_values = [util.get("success_rate", 0) for util in agent_utilization.values()]
                    if utilization_values:
                        avg_success = sum(utilization_values) / len(utilization_values)
                        collaboration_scores.append(avg_success)
            
            avg_collaboration_score = sum(collaboration_scores) / len(collaboration_scores) if collaboration_scores else 0
            
            # Analyze skill utilization
            all_skills = set()
            for pattern in collaboration_patterns:
                agent_utilization = pattern.get("agent_utilization", {})
                for agent_data in agent_utilization.values():
                    all_skills.update(agent_data.get("skills_count", 0) for _ in range(1))
            
            return {
                "total_patterns": total_patterns,
                "average_collaboration_score": avg_collaboration_score,
                "collaboration_quality": "excellent" if avg_collaboration_score > 0.9 else "good" if avg_collaboration_score > 0.7 else "needs_improvement",
                "unique_skills_utilized": len(all_skills),
                "patterns_analyzed": collaboration_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing crew collaboration patterns: {str(e)}")
            return {}
    
    def get_role_based_performance_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get role-based performance metrics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Role-based performance metrics
        """
        try:
            monitor = self.get_workflow_monitor(session_id)
            if not monitor:
                return {}
            
            workflow_summary = monitor.get_workflow_summary()
            agent_performance = workflow_summary.get("agent_performance_summary", {})
            
            # Group performance by role
            role_performance: Dict[str, Dict[str, Any]] = {}
            for agent_name, perf in agent_performance.items():
                # Extract role from agent performance data (assuming it's stored)
                role = perf.get("role", "unknown_role")
                
                if role not in role_performance:
                    role_performance[role] = {
                        "agents": [],
                        "total_tasks": 0,
                        "total_completed": 0,
                        "total_failed": 0,
                        "total_execution_time": 0.0,
                        "skills_used": set()
                    }
                
                role_data = role_performance[role]
                role_data["agents"].append(agent_name)
                role_data["total_tasks"] += perf.get("tasks_completed", 0) + perf.get("tasks_failed", 0)
                role_data["total_completed"] += perf.get("tasks_completed", 0)
                role_data["total_failed"] += perf.get("tasks_failed", 0)
                role_data["total_execution_time"] += perf.get("average_execution_time", 0) * (perf.get("tasks_completed", 0) + perf.get("tasks_failed", 0))
                role_data["skills_used"].update(range(perf.get("skills_count", 0)))  # Placeholder for actual skills
            
            # Calculate role metrics
            for role, data in role_performance.items():
                total_tasks = data["total_tasks"]
                data["success_rate"] = data["total_completed"] / total_tasks if total_tasks > 0 else 0
                data["average_execution_time"] = data["total_execution_time"] / total_tasks if total_tasks > 0 else 0
                data["agent_count"] = len(data["agents"])
                data["skills_count"] = len(data["skills_used"])
                
                # Convert set to list for JSON serialization
                data["skills_used"] = list(data["skills_used"])
            
            return {
                "role_performance": role_performance,
                "total_roles": len(role_performance),
                "best_performing_role": max(role_performance.keys(), key=lambda r: role_performance[r]["success_rate"]) if role_performance else None,
                "most_utilized_role": max(role_performance.keys(), key=lambda r: role_performance[r]["total_tasks"]) if role_performance else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting role-based performance metrics: {str(e)}")
            return {}