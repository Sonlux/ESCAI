"""
Basic performance tests for ESCAI Framework.
Tests basic performance characteristics of existing components.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime
import pytest
import numpy as np

from escai_framework.instrumentation.langchain_instrumentor import LangChainInstrumentor
from escai_framework.instrumentation.events import AgentEvent, EventType
from escai_framework.core.epistemic_extractor import EpistemicExtractor


class MockAgent:
    """Mock agent for performance testing."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.task_count = 0
    
    async def execute_task(self, complexity: int = 1) -> Dict[str, Any]:
        """Simulate agent task execution."""
        # Simulate work based on complexity
        work_time = 0.01 * complexity  # Base work time
        await asyncio.sleep(work_time)
        
        self.task_count += 1
        return {
            "task_id": f"task_{self.task_count}",
            "result": f"Completed task {self.task_count}",
            "complexity": complexity,
            "execution_time": work_time
        }


@pytest.mark.asyncio
class TestBasicPerformance:
    """Test basic performance characteristics."""
    
    async def test_instrumentor_overhead(self, performance_config):
        """Test basic instrumentor overhead."""
        agent = MockAgent("test_agent")
        instrumentor = LangChainInstrumentor()
        
        # Measure baseline performance (10 iterations for speed)
        baseline_times = []
        for i in range(10):
            start_time = time.perf_counter()
            await agent.execute_task(complexity=1)
            end_time = time.perf_counter()
            baseline_times.append(end_time - start_time)
        
        # Reset agent
        agent = MockAgent("test_agent")
        
        # Measure with monitoring
        session_id = await instrumentor.start_monitoring(agent.agent_id, {})
        
        try:
            monitored_times = []
            for i in range(10):
                start_time = time.perf_counter()
                
                # Capture start event
                await instrumentor.capture_event(AgentEvent(
                    event_id=f"event_{i}",
                    agent_id=agent.agent_id,
                    event_type=EventType.TASK_START,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"task_id": f"task_{i}"}
                ))
                
                # Execute task
                await agent.execute_task(complexity=1)
                
                # Capture end event
                await instrumentor.capture_event(AgentEvent(
                    event_id=f"event_{i}_end",
                    agent_id=agent.agent_id,
                    event_type=EventType.TASK_COMPLETE,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"task_id": f"task_{i}"}
                ))
                
                end_time = time.perf_counter()
                monitored_times.append(end_time - start_time)
        
        finally:
            await instrumentor.stop_monitoring(session_id)
        
        # Calculate overhead
        baseline_avg = statistics.mean(baseline_times)
        monitored_avg = statistics.mean(monitored_times)
        
        overhead_percentage = ((monitored_avg - baseline_avg) / baseline_avg) * 100
        
        # Assert overhead is reasonable (allow up to 50% for this basic test)
        assert overhead_percentage <= 50, \
            f"Monitoring overhead {overhead_percentage:.2f}% too high"
        
        print(f"Basic monitoring overhead: {overhead_percentage:.2f}%")
    
    async def test_epistemic_extractor_performance(self):
        """Test epistemic extractor performance."""
        extractor = EpistemicExtractor()
        
        # Generate test events
        from escai_framework.instrumentation.events import AgentEvent, EventType
        
        logs = []
        for i in range(50):  # Smaller dataset for basic test
            logs.append(AgentEvent(
                event_id=f"perf_test_event_{i}",
                agent_id="test_agent",
                event_type=EventType.DECISION_START,
                timestamp=datetime.now(),
                session_id="perf_test_session",
                message=f"Test message {i}",
                data={"task_id": f"task_{i}"}
            ))
        
        # Measure processing time
        start_time = time.perf_counter()
        beliefs = await extractor.extract_beliefs(logs)
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        
        # Assert reasonable processing time (5 seconds for 50 events, allowing for NLP processing)
        assert processing_time <= 5.0, \
            f"Processing 50 events took {processing_time:.3f}s, expected <= 5.0s"
        
        print(f"Epistemic extraction time: {processing_time:.3f}s for 50 events")
    
    async def test_concurrent_monitoring(self):
        """Test concurrent monitoring performance."""
        num_agents = 5  # Smaller number for basic test
        agents = [MockAgent(f"agent_{i}") for i in range(num_agents)]
        instrumentors = [LangChainInstrumentor() for _ in range(num_agents)]
        
        # Start monitoring for all agents
        session_data = []
        for agent, instrumentor in zip(agents, instrumentors):
            session_id = await instrumentor.start_monitoring(agent.agent_id, {})
            session_data.append((agent, instrumentor, session_id))
        
        start_time = time.perf_counter()
        
        try:
            # Execute tasks concurrently
            tasks = []
            for agent, instrumentor, session_id in session_data:
                async def agent_task(a, i, sid):
                    for j in range(5):  # 5 tasks per agent
                        await i.capture_event(AgentEvent(
                            event_id=f"{a.agent_id}_task_{j}",
                            agent_id=a.agent_id,
                            event_type=EventType.TASK_START,
                            timestamp=datetime.now(),
                            session_id=sid,
                            data={"task": j}
                        ))
                        await a.execute_task()
                
                tasks.append(agent_task(agent, instrumentor, session_id))
            
            await asyncio.gather(*tasks)
        
        finally:
            # Stop monitoring
            for agent, instrumentor, session_id in session_data:
                await instrumentor.stop_monitoring(session_id)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        total_tasks = num_agents * 5
        tasks_per_second = total_tasks / execution_time
        
        # Assert reasonable throughput (at least 10 tasks/second)
        assert tasks_per_second >= 10, \
            f"Throughput {tasks_per_second:.1f} tasks/s below minimum 10"
        
        print(f"Concurrent monitoring throughput: {tasks_per_second:.1f} tasks/s")


if __name__ == "__main__":
    # Run basic performance tests
    pytest.main([__file__, "-v", "--tb=short"])