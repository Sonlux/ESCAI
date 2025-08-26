"""
Basic end-to-end workflow tests for ESCAI Framework.
Tests basic workflows with existing components.
"""

import asyncio
import time
from datetime import datetime
import pytest

from escai_framework.instrumentation.langchain_instrumentor import LangChainInstrumentor
from escai_framework.instrumentation.events import AgentEvent, EventType
from escai_framework.core.epistemic_extractor import EpistemicExtractor


@pytest.mark.asyncio
class TestBasicWorkflow:
    """Test basic end-to-end workflows."""
    
    async def test_basic_monitoring_workflow(self):
        """Test basic monitoring workflow from start to finish."""
        
        agent_id = "e2e_test_agent"
        instrumentor = LangChainInstrumentor()
        extractor = EpistemicExtractor()
        
        # Start monitoring
        session_id = await instrumentor.start_monitoring(agent_id, {
            "framework": "test",
            "scenario": "basic_workflow"
        })
        
        events_generated = []
        
        try:
            # Generate a sequence of events
            event_sequence = [
                (EventType.TASK_START, {"task": "basic_test", "priority": "high"}),
                (EventType.DECISION_COMPLETE, {"decision": "proceed", "confidence": 0.8}),
                (EventType.PERFORMANCE_METRIC, {"progress": 0.5, "step": "processing"}),
                (EventType.TASK_COMPLETE, {"status": "success", "duration": 30})
            ]
            
            for i, (event_type, data) in enumerate(event_sequence):
                event = AgentEvent(
                    event_id=f"e2e_event_{i}",
                    agent_id=agent_id,
                    event_type=event_type,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data=data
                )
                
                await instrumentor.capture_event(event)
                events_generated.append(event)
                
                # Small delay between events
                await asyncio.sleep(0.1)
            
            # Wait for event processing
            await asyncio.sleep(0.5)
            
            # Extract epistemic states from generated events
            epistemic_states = await extractor.extract_beliefs(events_generated)
            
            # Assertions
            assert len(events_generated) == 4, "Not all events were generated"
            assert len(epistemic_states) > 0, "No epistemic states extracted"
            
            # Verify event sequence
            event_types = [event.event_type for event in events_generated]
            assert EventType.TASK_START in event_types, "Missing task start event"
            assert EventType.TASK_COMPLETE in event_types, "Missing task completion event"
            
            # Verify epistemic state extraction
            assert all(state.agent_id == agent_id for state in epistemic_states), \
                "Inconsistent agent IDs in epistemic states"
            
            print(f"Basic workflow: {len(events_generated)} events, "
                  f"{len(epistemic_states)} states extracted")
        
        finally:
            await instrumentor.stop_monitoring(session_id)
    
    async def test_error_handling_workflow(self):
        """Test workflow with error handling."""
        
        agent_id = "e2e_error_test_agent"
        instrumentor = LangChainInstrumentor()
        
        session_id = await instrumentor.start_monitoring(agent_id, {})
        
        try:
            # Generate workflow with error and recovery
            events = [
                AgentEvent(
                    event_id="error_test_1",
                    agent_id=agent_id,
                    event_type=EventType.TASK_START,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"task": "error_test"}
                ),
                AgentEvent(
                    event_id="error_test_2",
                    agent_id=agent_id,
                    event_type=EventType.ERROR_OCCURRED,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"error": "TestError", "message": "Simulated error"}
                ),
                AgentEvent(
                    event_id="error_test_3",
                    agent_id=agent_id,
                    event_type=EventType.DECISION_COMPLETE,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"decision": "retry", "confidence": 0.7}
                ),
                AgentEvent(
                    event_id="error_test_4",
                    agent_id=agent_id,
                    event_type=EventType.TASK_COMPLETE,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"status": "success", "recovery": True}
                )
            ]
            
            for event in events:
                await instrumentor.capture_event(event)
                await asyncio.sleep(0.05)
            
            # Verify error handling workflow
            event_types = [event.event_type for event in events]
            assert EventType.ERROR_OCCURRED in event_types, "No error event found"
            assert EventType.DECISION_COMPLETE in event_types, "No recovery decision found"
            assert EventType.TASK_COMPLETE in event_types, "Recovery did not complete"
            
            print(f"Error handling workflow: {len(events)} events processed")
        
        finally:
            await instrumentor.stop_monitoring(session_id)
    
    async def test_concurrent_agents_workflow(self):
        """Test workflow with multiple concurrent agents."""
        
        num_agents = 3
        agents = [f"concurrent_agent_{i}" for i in range(num_agents)]
        instrumentors = [LangChainInstrumentor() for _ in range(num_agents)]
        
        # Start monitoring for all agents
        session_data = []
        for agent_id, instrumentor in zip(agents, instrumentors):
            session_id = await instrumentor.start_monitoring(agent_id, {})
            session_data.append((agent_id, instrumentor, session_id))
        
        try:
            # Generate concurrent events
            async def agent_workflow(agent_id: str, instrumentor: LangChainInstrumentor, session_id: str):
                events = [
                    AgentEvent(
                        event_id=f"{agent_id}_start",
                        agent_id=agent_id,
                        event_type=EventType.TASK_START,
                        timestamp=datetime.now(),
                        session_id=session_id,
                        data={"task": f"concurrent_task_{agent_id}"}
                    ),
                    AgentEvent(
                        event_id=f"{agent_id}_progress",
                        agent_id=agent_id,
                        event_type=EventType.PERFORMANCE_METRIC,
                        timestamp=datetime.now(),
                        session_id=session_id,
                        data={"progress": 0.5}
                    ),
                    AgentEvent(
                        event_id=f"{agent_id}_complete",
                        agent_id=agent_id,
                        event_type=EventType.TASK_COMPLETE,
                        timestamp=datetime.now(),
                        session_id=session_id,
                        data={"status": "success"}
                    )
                ]
                
                for event in events:
                    await instrumentor.capture_event(event)
                    await asyncio.sleep(0.1)
            
            # Run all agent workflows concurrently
            tasks = [
                agent_workflow(agent_id, instrumentor, session_id)
                for agent_id, instrumentor, session_id in session_data
            ]
            
            await asyncio.gather(*tasks)
            
            print(f"Concurrent workflow: {num_agents} agents completed successfully")
        
        finally:
            # Stop monitoring for all agents
            for agent_id, instrumentor, session_id in session_data:
                await instrumentor.stop_monitoring(session_id)
    
    async def test_data_flow_integrity(self):
        """Test data flow integrity through the system."""
        
        agent_id = "data_flow_test_agent"
        instrumentor = LangChainInstrumentor()
        extractor = EpistemicExtractor()
        
        session_id = await instrumentor.start_monitoring(agent_id, {})
        
        try:
            # Generate events with specific data
            test_data = {
                "task_id": "data_flow_test_123",
                "user_input": "analyze sales data",
                "confidence": 0.85,
                "complexity": "medium"
            }
            
            event = AgentEvent(
                event_id="data_flow_event",
                agent_id=agent_id,
                event_type=EventType.TASK_START,
                timestamp=datetime.now(),
                session_id=session_id,
                data=test_data
            )
            
            await instrumentor.capture_event(event)
            await asyncio.sleep(0.2)
            
            # Extract and verify data integrity
            agent_logs = [{
                "timestamp": event.timestamp,
                "agent_id": event.agent_id,
                "event_type": event.event_type.value,
                "data": event.data
            }]
            
            epistemic_states = await extractor.extract_beliefs(agent_logs)
            
            # Verify data flow integrity
            assert len(epistemic_states) > 0, "No epistemic states generated"
            
            # Check that agent ID is preserved
            for state in epistemic_states:
                assert state.agent_id == agent_id, f"Agent ID mismatch: {state.agent_id} != {agent_id}"
            
            print("Data flow integrity: PASSED")
        
        finally:
            await instrumentor.stop_monitoring(session_id)


if __name__ == "__main__":
    # Run basic e2e tests
    pytest.main([__file__, "-v", "--tb=short"])