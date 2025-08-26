"""
Performance tests for monitoring overhead measurement.
Tests ensure that ESCAI monitoring adds minimal overhead to agent execution.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
import pytest
import numpy as np

from escai_framework.instrumentation.base_instrumentor import BaseInstrumentor
from escai_framework.instrumentation.langchain_instrumentor import LangChainInstrumentor
from escai_framework.instrumentation.events import AgentEvent, EventType
from escai_framework.core.epistemic_extractor import EpistemicExtractor
from escai_framework.analytics.pattern_mining import PatternMiner


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


class PerformanceTestSuite:
    """Performance test suite for monitoring overhead."""
    
    def __init__(self):
        self.baseline_times: List[float] = []
        self.monitored_times: List[float] = []
        self.overhead_threshold = 0.10  # 10% maximum overhead
    
    async def measure_baseline_performance(self, agent: MockAgent, iterations: int = 100) -> List[float]:
        """Measure baseline agent performance without monitoring."""
        times = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            await agent.execute_task(complexity=np.random.randint(1, 5))
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return times
    
    async def measure_monitored_performance(
        self, 
        agent: MockAgent, 
        instrumentor: BaseInstrumentor,
        iterations: int = 100
    ) -> List[float]:
        """Measure agent performance with monitoring enabled."""
        times = []
        
        # Start monitoring
        session_id = await instrumentor.start_monitoring(agent.agent_id, {})
        
        try:
            for i in range(iterations):
                start_time = time.perf_counter()
                
                # Simulate monitoring events
                await instrumentor.capture_event(AgentEvent(
                    event_id=f"event_{i}",
                    agent_id=agent.agent_id,
                    event_type=EventType.TASK_START,
                    timestamp=start_time,
                    data={"task_id": f"task_{i}"}
                ))
                
                # Execute task
                result = await agent.execute_task(complexity=np.random.randint(1, 5))
                
                # Capture completion event
                await instrumentor.capture_event(AgentEvent(
                    event_id=f"event_{i}_complete",
                    agent_id=agent.agent_id,
                    event_type=EventType.TASK_COMPLETE,
                    timestamp=time.perf_counter(),
                    data=result
                ))
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        finally:
            await instrumentor.stop_monitoring(session_id)
        
        return times
    
    def calculate_overhead(self, baseline_times: List[float], monitored_times: List[float]) -> Dict[str, float]:
        """Calculate monitoring overhead statistics."""
        baseline_avg = statistics.mean(baseline_times)
        monitored_avg = statistics.mean(monitored_times)
        
        overhead_absolute = monitored_avg - baseline_avg
        overhead_percentage = (overhead_absolute / baseline_avg) * 100
        
        return {
            "baseline_avg_ms": baseline_avg * 1000,
            "monitored_avg_ms": monitored_avg * 1000,
            "overhead_absolute_ms": overhead_absolute * 1000,
            "overhead_percentage": overhead_percentage,
            "baseline_std_ms": statistics.stdev(baseline_times) * 1000,
            "monitored_std_ms": statistics.stdev(monitored_times) * 1000
        }


@pytest.mark.asyncio
class TestMonitoringOverhead:
    """Test monitoring overhead across different scenarios."""
    
    async def test_langchain_instrumentor_overhead(self, performance_config):
        """Test LangChain instrumentor monitoring overhead."""
        suite = PerformanceTestSuite()
        agent = MockAgent("test_langchain_agent")
        instrumentor = LangChainInstrumentor()
        
        # Measure baseline performance
        baseline_times = await suite.measure_baseline_performance(agent, iterations=50)
        
        # Reset agent state
        agent = MockAgent("test_langchain_agent")
        
        # Measure monitored performance
        monitored_times = await suite.measure_monitored_performance(agent, instrumentor, iterations=50)
        
        # Calculate overhead
        overhead_stats = suite.calculate_overhead(baseline_times, monitored_times)
        
        # Assert overhead is within acceptable limits
        assert overhead_stats["overhead_percentage"] <= performance_config["max_monitoring_overhead"] * 100, \
            f"Monitoring overhead {overhead_stats['overhead_percentage']:.2f}% exceeds limit"
        
        print(f"LangChain Instrumentor Overhead: {overhead_stats['overhead_percentage']:.2f}%")
    
    async def test_epistemic_extractor_performance(self, performance_config, test_data_generator):
        """Test epistemic extractor performance with various data sizes."""
        extractor = EpistemicExtractor()
        
        data_sizes = [10, 50, 100, 500, 1000]
        processing_times = []
        
        for size in data_sizes:
            # Generate test logs
            logs = test_data_generator.generate_agent_logs(size)
            
            # Measure processing time
            start_time = time.perf_counter()
            epistemic_state = await extractor.extract_beliefs(logs)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # Assert processing time is reasonable
            max_time = size * 0.001  # 1ms per log entry
            assert processing_time <= max_time, \
                f"Processing {size} logs took {processing_time:.3f}s, expected <= {max_time:.3f}s"
        
        # Check that processing time scales reasonably
        time_per_log = [t / s for t, s in zip(processing_times, data_sizes)]
        avg_time_per_log = statistics.mean(time_per_log)
        
        assert avg_time_per_log <= 0.001, \
            f"Average time per log {avg_time_per_log:.6f}s exceeds 1ms limit"
        
        print(f"Average processing time per log: {avg_time_per_log * 1000:.3f}ms")
    
    async def test_pattern_analyzer_performance(self, performance_config, test_data_generator):
        """Test behavioral pattern analyzer performance."""
        analyzer = PatternMiner()
        
        # Generate test behavioral sequences
        sequences = test_data_generator.generate_behavioral_sequences(100)
        
        # Measure pattern mining performance
        start_time = time.perf_counter()
        patterns = await analyzer.mine_patterns(sequences)
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        
        # Assert reasonable processing time
        assert processing_time <= 5.0, \
            f"Pattern mining took {processing_time:.3f}s, expected <= 5.0s"
        
        print(f"Pattern mining: {processing_time:.3f}s")
    
    async def test_concurrent_monitoring_performance(self, performance_config):
        """Test performance with multiple concurrent agents."""
        num_agents = 10
        tasks_per_agent = 20
        
        agents = [MockAgent(f"agent_{i}") for i in range(num_agents)]
        instrumentors = [LangChainInstrumentor() for _ in range(num_agents)]
        
        async def monitor_agent(agent: MockAgent, instrumentor: BaseInstrumentor):
            """Monitor a single agent."""
            session_id = await instrumentor.start_monitoring(agent.agent_id, {})
            
            try:
                for i in range(tasks_per_agent):
                    await instrumentor.capture_event(AgentEvent(
                        event_id=f"{agent.agent_id}_event_{i}",
                        agent_id=agent.agent_id,
                        event_type=EventType.TASK_START,
                        timestamp=time.perf_counter(),
                        data={"task_id": f"task_{i}"}
                    ))
                    
                    await agent.execute_task()
            finally:
                await instrumentor.stop_monitoring(session_id)
        
        # Measure concurrent monitoring performance
        start_time = time.perf_counter()
        
        tasks = [monitor_agent(agent, instrumentor) 
                for agent, instrumentor in zip(agents, instrumentors)]
        await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate throughput
        total_events = num_agents * tasks_per_agent
        events_per_second = total_events / total_time
        
        # Assert minimum throughput
        min_throughput = performance_config["min_throughput_events_per_sec"]
        assert events_per_second >= min_throughput, \
            f"Throughput {events_per_second:.1f} events/s below minimum {min_throughput}"
        
        print(f"Concurrent monitoring throughput: {events_per_second:.1f} events/s")
    
    async def test_memory_usage_monitoring(self, performance_config):
        """Test memory usage during extended monitoring."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        agent = MockAgent("memory_test_agent")
        instrumentor = LangChainInstrumentor()
        
        session_id = await instrumentor.start_monitoring(agent.agent_id, {})
        
        try:
            # Generate many events to test memory usage
            for i in range(1000):
                await instrumentor.capture_event(AgentEvent(
                    event_id=f"memory_test_event_{i}",
                    agent_id=agent.agent_id,
                    event_type=EventType.DECISION_MADE,
                    timestamp=time.perf_counter(),
                    data={"decision": f"decision_{i}", "data": "x" * 1000}  # 1KB per event
                ))
                
                if i % 100 == 0:
                    await agent.execute_task()
        
        finally:
            await instrumentor.stop_monitoring(session_id)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Assert reasonable memory usage (should not exceed 100MB for 1000 events)
        assert memory_increase <= 100, \
            f"Memory usage increased by {memory_increase:.1f}MB, expected <= 100MB"
        
        print(f"Memory usage increase: {memory_increase:.1f}MB for 1000 events")


@pytest.mark.asyncio
class TestScalabilityPerformance:
    """Test system scalability under various loads."""
    
    async def test_agent_scaling(self, performance_config):
        """Test performance scaling with increasing number of agents."""
        max_agents = performance_config["max_concurrent_agents"]
        agent_counts = [1, 5, 10, 25, 50, max_agents]
        
        results = []
        
        for count in agent_counts:
            agents = [MockAgent(f"scale_agent_{i}") for i in range(count)]
            instrumentors = [LangChainInstrumentor() for _ in range(count)]
            
            start_time = time.perf_counter()
            
            # Start monitoring for all agents
            session_ids = []
            for agent, instrumentor in zip(agents, instrumentors):
                session_id = await instrumentor.start_monitoring(agent.agent_id, {})
                session_ids.append((instrumentor, session_id))
            
            # Execute tasks concurrently
            tasks = []
            for agent, instrumentor in zip(agents, instrumentors):
                async def agent_task(a, i):
                    for j in range(10):  # 10 tasks per agent
                        await i.capture_event(AgentEvent(
                            event_id=f"{a.agent_id}_task_{j}",
                            agent_id=a.agent_id,
                            event_type=EventType.TASK_START,
                            timestamp=time.perf_counter(),
                            data={"task": j}
                        ))
                        await a.execute_task()
                
                tasks.append(agent_task(agent, instrumentor))
            
            await asyncio.gather(*tasks)
            
            # Stop monitoring
            for instrumentor, session_id in session_ids:
                await instrumentor.stop_monitoring(session_id)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            results.append({
                "agent_count": count,
                "execution_time": execution_time,
                "tasks_per_second": (count * 10) / execution_time
            })
            
            print(f"Agents: {count}, Time: {execution_time:.2f}s, TPS: {results[-1]['tasks_per_second']:.1f}")
        
        # Verify that performance doesn't degrade significantly with scale
        baseline_tps = results[0]["tasks_per_second"]
        max_tps = results[-1]["tasks_per_second"]
        
        # Allow for some degradation but not more than 50%
        assert max_tps >= baseline_tps * 0.5, \
            f"Performance degraded from {baseline_tps:.1f} to {max_tps:.1f} TPS"
    
    async def test_event_throughput_scaling(self, performance_config):
        """Test event processing throughput under high load."""
        agent = MockAgent("throughput_test_agent")
        instrumentor = LangChainInstrumentor()
        
        session_id = await instrumentor.start_monitoring(agent.agent_id, {})
        
        try:
            event_counts = [100, 500, 1000, 5000, 10000]
            
            for count in event_counts:
                events = [
                    AgentEvent(
                        event_id=f"throughput_event_{i}",
                        agent_id=agent.agent_id,
                        event_type=EventType.DECISION_MADE,
                        timestamp=time.perf_counter(),
                        data={"decision": f"decision_{i}"}
                    )
                    for i in range(count)
                ]
                
                start_time = time.perf_counter()
                
                # Process events concurrently
                tasks = [instrumentor.capture_event(event) for event in events]
                await asyncio.gather(*tasks)
                
                end_time = time.perf_counter()
                processing_time = end_time - start_time
                events_per_second = count / processing_time
                
                min_throughput = performance_config["min_throughput_events_per_sec"]
                assert events_per_second >= min_throughput, \
                    f"Throughput {events_per_second:.1f} events/s below minimum {min_throughput}"
                
                print(f"Events: {count}, Throughput: {events_per_second:.1f} events/s")
        
        finally:
            await instrumentor.stop_monitoring(session_id)


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short"])