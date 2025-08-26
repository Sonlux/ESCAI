"""
Basic load tests for ESCAI Framework.
Tests basic load handling with existing components.
"""

import asyncio
import time
from datetime import datetime
import pytest

from escai_framework.instrumentation.langchain_instrumentor import LangChainInstrumentor
from escai_framework.instrumentation.events import AgentEvent, EventType


@pytest.mark.asyncio
class TestBasicLoad:
    """Test basic load handling."""
    
    async def test_multiple_events_load(self):
        """Test handling multiple events in sequence."""
        
        agent_id = "load_test_agent"
        instrumentor = LangChainInstrumentor()
        
        session_id = await instrumentor.start_monitoring(agent_id, {})
        
        try:
            num_events = 50  # Moderate load for basic test
            events_processed = 0
            
            start_time = time.perf_counter()
            
            # Generate events in sequence
            for i in range(num_events):
                event = AgentEvent(
                    event_id=f"load_event_{i}",
                    agent_id=agent_id,
                    event_type=EventType.PERFORMANCE_METRIC,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"sequence": i, "progress": i / num_events}
                )
                
                await instrumentor.capture_event(event)
                events_processed += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            # Calculate throughput
            events_per_second = events_processed / processing_time
            
            # Assert reasonable throughput (at least 20 events/second)
            assert events_per_second >= 20, \
                f"Throughput {events_per_second:.1f} events/s below minimum 20"
            
            print(f"Sequential load test: {events_per_second:.1f} events/s")
        
        finally:
            await instrumentor.stop_monitoring(session_id)
    
    async def test_concurrent_agents_load(self):
        """Test load with multiple concurrent agents."""
        
        num_agents = 10  # Moderate concurrency for basic test
        events_per_agent = 10
        
        agents = [f"concurrent_load_agent_{i}" for i in range(num_agents)]
        instrumentors = [LangChainInstrumentor() for _ in range(num_agents)]
        
        # Start monitoring for all agents
        session_data = []
        for agent_id, instrumentor in zip(agents, instrumentors):
            session_id = await instrumentor.start_monitoring(agent_id, {})
            session_data.append((agent_id, instrumentor, session_id))
        
        start_time = time.perf_counter()
        
        try:
            # Generate events concurrently
            async def agent_load_test(agent_id: str, instrumentor: LangChainInstrumentor, session_id: str):
                for i in range(events_per_agent):
                    event = AgentEvent(
                        event_id=f"{agent_id}_event_{i}",
                        agent_id=agent_id,
                        event_type=EventType.DECISION_COMPLETE,
                        timestamp=datetime.now(),
                        session_id=session_id,
                        data={"decision": f"decision_{i}", "agent": agent_id}
                    )
                    
                    await instrumentor.capture_event(event)
                    await asyncio.sleep(0.02)  # Small delay
            
            # Run all agent load tests concurrently
            tasks = [
                agent_load_test(agent_id, instrumentor, session_id)
                for agent_id, instrumentor, session_id in session_data
            ]
            
            await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            total_events = num_agents * events_per_agent
            events_per_second = total_events / total_time
            
            # Assert reasonable concurrent throughput
            assert events_per_second >= 50, \
                f"Concurrent throughput {events_per_second:.1f} events/s below minimum 50"
            
            print(f"Concurrent load test: {num_agents} agents, {events_per_second:.1f} events/s")
        
        finally:
            # Stop monitoring for all agents
            for agent_id, instrumentor, session_id in session_data:
                await instrumentor.stop_monitoring(session_id)
    
    async def test_burst_load_handling(self):
        """Test handling burst loads."""
        
        agent_id = "burst_load_agent"
        instrumentor = LangChainInstrumentor()
        
        session_id = await instrumentor.start_monitoring(agent_id, {})
        
        try:
            # Generate burst of events
            burst_size = 20
            events = []
            
            # Create all events first
            for i in range(burst_size):
                event = AgentEvent(
                    event_id=f"burst_event_{i}",
                    agent_id=agent_id,
                    event_type=EventType.TASK_START,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"burst_sequence": i}
                )
                events.append(event)
            
            # Send all events in rapid succession
            start_time = time.perf_counter()
            
            tasks = [instrumentor.capture_event(event) for event in events]
            await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            burst_time = end_time - start_time
            
            # Assert burst handling
            assert burst_time <= 2.0, f"Burst processing took {burst_time:.3f}s, expected <= 2.0s"
            
            burst_throughput = burst_size / burst_time
            assert burst_throughput >= 10, \
                f"Burst throughput {burst_throughput:.1f} events/s below minimum 10"
            
            print(f"Burst load test: {burst_size} events in {burst_time:.3f}s "
                  f"({burst_throughput:.1f} events/s)")
        
        finally:
            await instrumentor.stop_monitoring(session_id)
    
    async def test_sustained_load(self):
        """Test sustained load over time."""
        
        agent_id = "sustained_load_agent"
        instrumentor = LangChainInstrumentor()
        
        session_id = await instrumentor.start_monitoring(agent_id, {})
        
        try:
            # Run sustained load for 10 seconds
            duration = 10  # seconds
            events_per_second = 5  # Moderate sustained rate
            
            start_time = time.perf_counter()
            events_sent = 0
            
            while time.perf_counter() - start_time < duration:
                event = AgentEvent(
                    event_id=f"sustained_event_{events_sent}",
                    agent_id=agent_id,
                    event_type=EventType.PERFORMANCE_METRIC,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"sustained_sequence": events_sent}
                )
                
                await instrumentor.capture_event(event)
                events_sent += 1
                
                # Maintain target rate
                await asyncio.sleep(1.0 / events_per_second)
            
            end_time = time.perf_counter()
            actual_duration = end_time - start_time
            actual_rate = events_sent / actual_duration
            
            # Assert sustained performance
            expected_events = duration * events_per_second
            assert events_sent >= expected_events * 0.9, \
                f"Sent {events_sent} events, expected at least {expected_events * 0.9:.0f}"
            
            print(f"Sustained load test: {events_sent} events over {actual_duration:.1f}s "
                  f"({actual_rate:.1f} events/s)")
        
        finally:
            await instrumentor.stop_monitoring(session_id)


if __name__ == "__main__":
    # Run basic load tests
    pytest.main([__file__, "-v", "--tb=short"])