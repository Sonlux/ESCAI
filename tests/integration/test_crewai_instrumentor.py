"""
Integration tests for CrewAI instrumentor.

These tests verify the integration with CrewAI's workflow system
and proper event capture during crew execution.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from escai_framework.instrumentation.crewai_instrumentor import (
    CrewAIInstrumentor, 
    CrewAIWorkflowMonitor,
    CREWAI_AVAILABLE
)
from escai_framework.instrumentation.events import EventType, EventSeverity, AgentEvent


# Mock CrewAI classes for testing when CrewAI is not available
class MockAgent:
    def __init__(self, role="test_agent", name=None):
        self.role = role
        self.name = name or role
        self._execute_calls = []
        self._perform_calls = []
    
    def execute_task(self, task):
        self._execute_calls.append(task)
        return f"Task executed by {self.role}"
    
    def perform_task(self, task):
        self._perform_calls.append(task)
        return f"Task performed by {self.role}"


class MockTask:
    def __init__(self, description="test task", task_id=None, tools=None):
        self.description = description
        self.id = task_id or f"task_{id(self)}"
        self.tools = tools or []
        self._execute_calls = []
    
    def execute(self, agent):
        self._execute_calls.append(agent)
        return f"Task '{self.description}' executed by {agent.role}"


class MockCrew:
    def __init__(self, name="test_crew", agents=None, tasks=None):
        self.name = name
        self.agents = agents or []
        self.tasks = tasks or []
        self._kickoff_calls = []
    
    def kickoff(self):
        self._kickoff_calls.append(time.time())
        return f"Crew {self.name} executed with {len(self.agents)} agents and {len(self.tasks)} tasks"


@pytest.fixture
def instrumentor():
    """Create a CrewAI instrumentor for testing."""
    if not CREWAI_AVAILABLE:
        pytest.skip("CrewAI not available")
    
    return CrewAIInstrumentor()


@pytest.fixture
def mock_instrumentor():
    """Create a mock CrewAI instrumentor for testing without CrewAI."""
    with patch('escai_framework.instrumentation.crewai_instrumentor.CREWAI_AVAILABLE', True):
        return CrewAIInstrumentor()


@pytest.fixture
def workflow_monitor(mock_instrumentor):
    """Create a workflow monitor for testing."""
    return CrewAIWorkflowMonitor(
        instrumentor=mock_instrumentor,
        session_id="test_session",
        agent_id="test_agent"
    )


@pytest.fixture
def mock_agents():
    """Create mock CrewAI agents for testing."""
    return [
        MockAgent("developer", "dev_agent"),
        MockAgent("reviewer", "review_agent"),
        MockAgent("manager", "mgr_agent")
    ]


@pytest.fixture
def mock_tasks():
    """Create mock CrewAI tasks for testing."""
    return [
        MockTask("Write code", "task_1"),
        MockTask("Review code", "task_2"),
        MockTask("Deploy code", "task_3")
    ]


@pytest.fixture
def mock_crew(mock_agents, mock_tasks):
    """Create a mock crew for testing."""
    return MockCrew("development_crew", mock_agents, mock_tasks)


class TestCrewAIInstrumentor:
    """Test cases for CrewAI instrumentor."""
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, mock_instrumentor, mock_crew):
        """Test starting monitoring for CrewAI workflows."""
        agent_id = "test_workflow"
        config = {
            "crews": [mock_crew],
            "monitor_collaboration": True,
            "monitor_performance": True
        }
        
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        assert session_id is not None
        assert len(session_id) > 0
        
        # Verify session was created
        session = mock_instrumentor._get_session(session_id)
        assert session is not None
        assert session.agent_id == agent_id
        assert session.framework == "crewai"
        
        # Verify workflow monitor was created
        monitor = mock_instrumentor.get_workflow_monitor(session_id)
        assert monitor is not None
        assert monitor.agent_id == agent_id
        assert monitor.session_id == session_id
        
        # Verify crew is being tracked
        assert session_id in mock_instrumentor._monitored_crews
        assert len(mock_instrumentor._monitored_crews[session_id]) == 1
    
    @pytest.mark.asyncio
    async def test_start_monitoring_with_individual_components(self, mock_instrumentor, mock_agents, mock_tasks):
        """Test starting monitoring with individual agents and tasks."""
        agent_id = "test_components"
        config = {
            "agents": mock_agents,
            "tasks": mock_tasks,
            "monitor_collaboration": True,
            "monitor_performance": True
        }
        
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        # Verify agents are being tracked
        assert session_id in mock_instrumentor._monitored_agents
        assert len(mock_instrumentor._monitored_agents[session_id]) == len(mock_agents)
        
        # Verify tasks are being tracked
        assert session_id in mock_instrumentor._monitored_tasks
        assert len(mock_instrumentor._monitored_tasks[session_id]) == len(mock_tasks)
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, mock_instrumentor, mock_crew):
        """Test stopping monitoring for CrewAI workflows."""
        agent_id = "test_workflow"
        config = {"crews": [mock_crew]}
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        # Add some test events
        test_event = AgentEvent(
            event_type=EventType.TASK_START,
            agent_id=agent_id,
            session_id=session_id,
            message="Test task",
            framework="crewai"
        )
        await mock_instrumentor.capture_event(test_event)
        
        # Stop monitoring
        summary = await mock_instrumentor.stop_monitoring(session_id)
        
        assert summary is not None
        assert summary.session_id == session_id
        assert summary.agent_id == agent_id
        assert summary.framework == "crewai"
        assert summary.total_events >= 1  # At least the test event
        
        # Verify cleanup
        monitor = mock_instrumentor.get_workflow_monitor(session_id)
        assert monitor is None
        assert session_id not in mock_instrumentor._monitored_crews
        assert session_id not in mock_instrumentor._monitored_agents
        assert session_id not in mock_instrumentor._monitored_tasks
    
    @pytest.mark.asyncio
    async def test_capture_event(self, mock_instrumentor, mock_crew):
        """Test capturing events."""
        agent_id = "test_workflow"
        config = {"crews": [mock_crew]}
        
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        
        # Create test event
        event = AgentEvent(
            event_type=EventType.TASK_START,
            agent_id=agent_id,
            session_id=session_id,
            message="Test task execution",
            framework="crewai",
            component="task",
            operation="execute",
            data={
                "task_id": "task_1",
                "assigned_agent": "developer",
                "agent_role": "developer"
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
        assert EventType.TASK_START in supported_events
        assert EventType.TASK_COMPLETE in supported_events
        assert EventType.TASK_FAIL in supported_events
        assert EventType.ACTION_START in supported_events
        assert EventType.ACTION_COMPLETE in supported_events
        assert EventType.DECISION_START in supported_events
        assert EventType.DECISION_COMPLETE in supported_events
    
    def test_get_framework_name(self, mock_instrumentor):
        """Test getting framework name."""
        assert mock_instrumentor.get_framework_name() == "crewai"


class TestCrewAIWorkflowMonitor:
    """Test cases for CrewAI workflow monitor."""
    
    def test_monitor_task_execution(self, workflow_monitor, mock_tasks, mock_agents):
        """Test monitoring task execution."""
        task = mock_tasks[0]
        agent = mock_agents[0]
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(workflow_monitor, '_queue_event_safe', side_effect=capture_event):
            # Call monitor method
            result = workflow_monitor.monitor_task_execution(
                task.execute, task, agent
            )
            
            # Verify original method was called
            assert len(task._execute_calls) == 1
            assert task._execute_calls[0] == agent
            
            # Verify events were captured
            assert len(events_captured) >= 2  # Start and complete events
            
            start_event = events_captured[0]
            assert start_event.event_type == EventType.TASK_START
            assert start_event.component == "task"
            assert start_event.operation == "execute"
            assert start_event.data["task_id"] == task.id
            assert start_event.data["assigned_agent"] == agent.role
            
            complete_event = events_captured[1]
            assert complete_event.event_type == EventType.TASK_COMPLETE
            assert complete_event.component == "task"
            assert complete_event.operation == "complete"
            assert complete_event.duration_ms is not None
    
    def test_monitor_crew_kickoff(self, workflow_monitor, mock_crew):
        """Test monitoring crew kickoff."""
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(workflow_monitor, '_queue_event_safe', side_effect=capture_event):
            # Call monitor method
            result = workflow_monitor.monitor_crew_kickoff(
                mock_crew.kickoff, mock_crew
            )
            
            # Verify original method was called
            assert len(mock_crew._kickoff_calls) == 1
            
            # Verify events were captured
            assert len(events_captured) >= 2  # Start and complete events
            
            start_event = events_captured[0]
            assert start_event.event_type == EventType.AGENT_START
            assert start_event.component == "crew"
            assert start_event.operation == "kickoff"
            assert start_event.data["crew_name"] == mock_crew.name
            assert start_event.data["agents_count"] == len(mock_crew.agents)
            assert start_event.data["tasks_count"] == len(mock_crew.tasks)
            
            # Find the crew completion event (may not be the second event due to collaboration analysis)
            complete_events = [e for e in events_captured 
                             if e.event_type == EventType.AGENT_STOP and e.component == "crew"]
            assert len(complete_events) >= 1
            
            complete_event = complete_events[0]
            assert complete_event.component == "crew"
            assert complete_event.operation == "complete"
            assert complete_event.duration_ms is not None
    
    def test_monitor_agent_action(self, workflow_monitor, mock_agents, mock_tasks):
        """Test monitoring agent actions."""
        agent = mock_agents[0]
        task = mock_tasks[0]
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(workflow_monitor, '_queue_event_safe', side_effect=capture_event):
            # Call monitor method
            result = workflow_monitor.monitor_agent_action(
                agent.execute_task, agent, task
            )
            
            # Verify original method was called
            assert len(agent._execute_calls) == 1
            assert agent._execute_calls[0] == task
            
            # Verify events were captured
            assert len(events_captured) >= 2  # Start and complete events
            
            start_event = events_captured[0]
            assert start_event.event_type == EventType.ACTION_START
            assert start_event.component == "agent"
            assert start_event.operation == "action"
            assert start_event.data["agent_name"] == agent.role
            assert start_event.data["task_id"] == task.id
            
            complete_event = events_captured[1]
            assert complete_event.event_type == EventType.ACTION_COMPLETE
            assert complete_event.component == "agent"
            assert complete_event.operation == "complete"
            assert complete_event.duration_ms is not None
    
    def test_workflow_summary_tracking(self, workflow_monitor, mock_tasks, mock_agents):
        """Test workflow summary generation."""
        # Mock event queuing
        with patch.object(workflow_monitor, '_queue_event_safe'):
            # Execute some tasks
            for i, (task, agent) in enumerate(zip(mock_tasks, mock_agents)):
                workflow_monitor.monitor_task_execution(
                    task.execute, task, agent
                )
        
        # Get workflow summary
        summary = workflow_monitor.get_workflow_summary()
        
        assert summary["total_tasks"] == len(mock_tasks)
        assert summary["completed_tasks"] == len(mock_tasks)
        assert summary["failed_tasks"] == 0
        assert summary["total_agents"] == len(mock_agents)
        
        # Check agent performance summary
        assert len(summary["agent_performance_summary"]) == len(mock_agents)
        for agent in mock_agents:
            assert agent.role in summary["agent_performance_summary"]
            perf = summary["agent_performance_summary"][agent.role]
            assert perf["tasks_completed"] == 1
            assert perf["tasks_failed"] == 0
    
    def test_error_handling_in_task_execution(self, workflow_monitor, mock_tasks, mock_agents):
        """Test error handling during task execution."""
        task = mock_tasks[0]
        agent = mock_agents[0]
        
        # Create a task execution method that raises an exception
        def failing_execute(agent):
            raise RuntimeError("Task execution failed")
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock the event queuing
        with patch.object(workflow_monitor, '_queue_event_safe', side_effect=capture_event):
            # Call monitor method and expect exception
            with pytest.raises(RuntimeError):
                workflow_monitor.monitor_task_execution(
                    failing_execute, task, agent
                )
            
            # Verify error event was captured
            error_events = [e for e in events_captured if e.severity == EventSeverity.ERROR]
            assert len(error_events) >= 1
            
            error_event = error_events[0]
            assert error_event.event_type == EventType.TASK_FAIL
            assert error_event.component == "task"
            assert error_event.operation == "error"
            assert error_event.error_type == "RuntimeError"
            assert error_event.error_message == "Task execution failed"


class TestCrewAIIntegration:
    """Integration tests with mock CrewAI workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_crew_workflow(self, mock_instrumentor, mock_crew):
        """Test a complete crew workflow execution."""
        agent_id = "integration_test"
        config = {
            "crews": [mock_crew],
            "monitor_collaboration": True,
            "monitor_performance": True
        }
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        monitor = mock_instrumentor.get_workflow_monitor(session_id)
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock event queuing to capture events
        with patch.object(monitor, '_queue_event_safe', side_effect=capture_event):
            # Simulate crew execution
            crew_result = monitor.monitor_crew_kickoff(
                mock_crew.kickoff, mock_crew
            )
            
            # Simulate individual task executions within the crew
            for i, (task, agent) in enumerate(zip(mock_crew.tasks, mock_crew.agents)):
                task_result = monitor.monitor_task_execution(
                    task.execute, task, agent
                )
                
                # Simulate agent actions
                action_result = monitor.monitor_agent_action(
                    agent.execute_task, agent, task
                )
        
        # Verify workflow events were captured
        assert len(events_captured) >= 8  # Crew start/stop + 3 tasks + 3 actions
        
        # Check for different event types
        event_types = [event.event_type for event in events_captured]
        assert EventType.AGENT_START in event_types  # Crew start
        assert EventType.AGENT_STOP in event_types   # Crew stop
        assert EventType.TASK_START in event_types   # Task starts
        assert EventType.TASK_COMPLETE in event_types # Task completions
        assert EventType.ACTION_START in event_types  # Agent actions
        assert EventType.ACTION_COMPLETE in event_types # Action completions
        
        # Verify crew workflow structure
        crew_events = [e for e in events_captured if e.component == "crew"]
        assert len(crew_events) >= 2  # Start and stop
        
        task_events = [e for e in events_captured if e.component == "task"]
        assert len(task_events) >= 6  # 3 tasks × 2 events each (start + complete)
        
        action_events = [e for e in events_captured if e.component == "agent"]
        assert len(action_events) >= 6  # 3 actions × 2 events each (start + complete)
        
        # Get workflow summary
        summary = monitor.get_workflow_summary()
        assert summary["total_tasks"] >= 3
        assert summary["total_agents"] >= 3
        assert summary["completed_tasks"] >= 3
        
        # Stop monitoring
        monitoring_summary = await mock_instrumentor.stop_monitoring(session_id)
        assert "workflow_summary" in monitoring_summary.performance_metrics
    
    @pytest.mark.asyncio
    async def test_collaboration_pattern_analysis(self, mock_instrumentor, mock_crew):
        """Test collaboration pattern analysis."""
        agent_id = "collaboration_test"
        config = {
            "crews": [mock_crew],
            "monitor_collaboration": True,
            "monitor_performance": True
        }
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        monitor = mock_instrumentor.get_workflow_monitor(session_id)
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock event queuing to capture events
        with patch.object(monitor, '_queue_event_safe', side_effect=capture_event):
            # Simulate crew execution with task distribution
            crew_result = monitor.monitor_crew_kickoff(
                mock_crew.kickoff, mock_crew
            )
            
            # Simulate uneven task distribution (some agents get more tasks)
            for i in range(2):  # Agent 0 gets 2 tasks
                task_result = monitor.monitor_task_execution(
                    mock_crew.tasks[0].execute, mock_crew.tasks[0], mock_crew.agents[0]
                )
            
            for i in range(1):  # Agent 1 gets 1 task
                task_result = monitor.monitor_task_execution(
                    mock_crew.tasks[1].execute, mock_crew.tasks[1], mock_crew.agents[1]
                )
            
            # No tasks for agent 2 to test uneven distribution
        
        # Verify collaboration analysis events
        collaboration_events = [e for e in events_captured 
                              if "collaboration" in e.tags or e.component == "collaboration"]
        
        assert len(collaboration_events) >= 1
        
        # Check collaboration pattern data
        for event in collaboration_events:
            if event.component == "collaboration":
                assert "task_distribution" in event.data
                assert "agent_utilization" in event.data
                assert "total_agents" in event.data
                assert "total_tasks" in event.data
        
        # Get workflow summary and check collaboration metrics
        summary = monitor.get_workflow_summary()
        assert "agent_performance_summary" in summary
        
        # Verify task distribution analysis
        perf_summary = summary["agent_performance_summary"]
        assert mock_crew.agents[0].role in perf_summary
        assert mock_crew.agents[1].role in perf_summary
        
        # Agent 0 should have more completed tasks
        assert perf_summary[mock_crew.agents[0].role]["tasks_completed"] >= 2
        assert perf_summary[mock_crew.agents[1].role]["tasks_completed"] >= 1
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, mock_instrumentor, mock_crew):
        """Test error handling and recovery in workflows."""
        agent_id = "error_recovery_test"
        config = {"crews": [mock_crew]}
        
        # Start monitoring
        session_id = await mock_instrumentor.start_monitoring(agent_id, config)
        monitor = mock_instrumentor.get_workflow_monitor(session_id)
        
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock event queuing to capture events
        with patch.object(monitor, '_queue_event_safe', side_effect=capture_event):
            # Simulate crew start
            crew_result = monitor.monitor_crew_kickoff(
                mock_crew.kickoff, mock_crew
            )
            
            # Simulate successful task
            task_result = monitor.monitor_task_execution(
                mock_crew.tasks[0].execute, mock_crew.tasks[0], mock_crew.agents[0]
            )
            
            # Simulate failed task
            def failing_execute(agent):
                raise ValueError("Task validation failed")
            
            with pytest.raises(ValueError):
                monitor.monitor_task_execution(
                    failing_execute, mock_crew.tasks[1], mock_crew.agents[1]
                )
            
            # Simulate recovery with another successful task
            task_result = monitor.monitor_task_execution(
                mock_crew.tasks[2].execute, mock_crew.tasks[2], mock_crew.agents[2]
            )
        
        # Verify error and recovery events
        error_events = [e for e in events_captured if e.severity == EventSeverity.ERROR]
        success_events = [e for e in events_captured 
                         if e.event_type == EventType.TASK_COMPLETE]
        
        assert len(error_events) >= 1  # At least one failure
        assert len(success_events) >= 2  # At least two successes
        
        # Verify workflow summary reflects mixed results
        summary = monitor.get_workflow_summary()
        assert summary["completed_tasks"] >= 2
        assert summary["failed_tasks"] >= 1
        
        # Check agent performance reflects the failure
        perf_summary = summary["agent_performance_summary"]
        failed_agent = mock_crew.agents[1].role
        if failed_agent in perf_summary:
            assert perf_summary[failed_agent]["tasks_failed"] >= 1


if __name__ == "__main__":
    pytest.main([__file__])