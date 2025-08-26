"""
Load tests for concurrent agent monitoring scenarios.
Tests system behavior under high concurrent load with stress testing.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any, Tuple
import pytest
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import json

from escai_framework.instrumentation.langchain_instrumentor import LangChainInstrumentor
from escai_framework.instrumentation.events import AgentEvent, EventType
from escai_framework.api.main import create_app


class LoadTestAgent:
    """Agent simulator for load testing."""
    
    def __init__(self, agent_id: str, behavior_profile: str = "normal"):
        self.agent_id = agent_id
        self.behavior_profile = behavior_profile
        self.task_count = 0
        self.errors = []
    
    async def simulate_workload(self, duration_seconds: int, tasks_per_minute: int = 60):
        """Simulate agent workload for specified duration."""
        end_time = time.time() + duration_seconds
        task_interval = 60.0 / tasks_per_minute  # seconds between tasks
        
        while time.time() < end_time:
            try:
                await self._execute_simulated_task()
                await asyncio.sleep(task_interval + np.random.exponential(0.1))
            except Exception as e:
                self.errors.append(str(e))
    
    async def _execute_simulated_task(self):
        """Execute a simulated task based on behavior profile."""
        self.task_count += 1
        
        if self.behavior_profile == "heavy":
            # Heavy workload with complex operations
            await asyncio.sleep(np.random.uniform(0.1, 0.5))
        elif self.behavior_profile == "bursty":
            # Bursty workload with occasional heavy operations
            if np.random.random() < 0.1:  # 10% chance of heavy operation
                await asyncio.sleep(np.random.uniform(0.5, 2.0))
            else:
                await asyncio.sleep(np.random.uniform(0.01, 0.05))
        else:  # normal
            # Normal workload
            await asyncio.sleep(np.random.uniform(0.02, 0.1))
        
        # Simulate occasional failures
        if np.random.random() < 0.05:  # 5% failure rate
            raise Exception(f"Simulated task failure for {self.agent_id}")


class APILoadTester:
    """Load tester for API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.response_times = []
        self.error_count = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, endpoint: str, data: Dict = None) -> Tuple[int, float]:
        """Make HTTP request and measure response time."""
        start_time = time.perf_counter()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    status = response.status
                    await response.text()
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    status = response.status
                    await response.text()
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            end_time = time.perf_counter()
            response_time = end_time - start_time
            self.response_times.append(response_time)
            
            return status, response_time
        
        except Exception as e:
            self.error_count += 1
            end_time = time.perf_counter()
            return 500, end_time - start_time
    
    async def load_test_endpoint(
        self, 
        method: str, 
        endpoint: str, 
        concurrent_users: int,
        duration_seconds: int,
        data_generator=None
    ) -> Dict[str, Any]:
        """Load test a specific endpoint."""
        
        async def user_simulation():
            """Simulate a single user's requests."""
            end_time = time.time() + duration_seconds
            requests_made = 0
            
            while time.time() < end_time:
                data = data_generator() if data_generator else None
                status, response_time = await self.make_request(method, endpoint, data)
                requests_made += 1
                
                # Add some randomness to request timing
                await asyncio.sleep(np.random.exponential(0.1))
            
            return requests_made
        
        # Start concurrent user simulations
        start_time = time.time()
        tasks = [user_simulation() for _ in range(concurrent_users)]
        requests_per_user = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_requests = sum(requests_per_user)
        actual_duration = end_time - start_time
        
        return {
            "endpoint": endpoint,
            "concurrent_users": concurrent_users,
            "duration": actual_duration,
            "total_requests": total_requests,
            "requests_per_second": total_requests / actual_duration,
            "average_response_time": statistics.mean(self.response_times) if self.response_times else 0,
            "p95_response_time": np.percentile(self.response_times, 95) if self.response_times else 0,
            "p99_response_time": np.percentile(self.response_times, 99) if self.response_times else 0,
            "error_rate": self.error_count / total_requests if total_requests > 0 else 0
        }


@pytest.mark.asyncio
class TestConcurrentMonitoring:
    """Test concurrent agent monitoring under load."""
    
    async def test_multiple_agent_monitoring(self, load_test_config):
        """Test monitoring multiple agents concurrently."""
        num_agents = 50
        test_duration = 60  # seconds
        
        agents = [LoadTestAgent(f"load_agent_{i}") for i in range(num_agents)]
        instrumentors = [LangChainInstrumentor() for _ in range(num_agents)]
        
        # Start monitoring for all agents
        session_ids = []
        for agent, instrumentor in zip(agents, instrumentors):
            session_id = await instrumentor.start_monitoring(agent.agent_id, {})
            session_ids.append((instrumentor, session_id))
        
        start_time = time.time()
        
        try:
            # Start agent workload simulations
            workload_tasks = []
            for agent in agents:
                task = asyncio.create_task(agent.simulate_workload(test_duration))
                workload_tasks.append(task)
            
            # Generate monitoring events concurrently
            async def generate_monitoring_events():
                event_count = 0
                while time.time() - start_time < test_duration:
                    for agent, instrumentor in zip(agents, instrumentors):
                        if np.random.random() < 0.1:  # 10% chance per iteration
                            await instrumentor.capture_event(AgentEvent(
                                event_id=f"load_event_{event_count}",
                                agent_id=agent.agent_id,
                                event_type=np.random.choice(list(EventType)),
                                timestamp=time.time(),
                                data={"event_count": event_count}
                            ))
                            event_count += 1
                    
                    await asyncio.sleep(0.01)  # 10ms between event generation cycles
            
            monitoring_task = asyncio.create_task(generate_monitoring_events())
            
            # Wait for all tasks to complete
            await asyncio.gather(*workload_tasks, monitoring_task)
        
        finally:
            # Stop monitoring for all agents
            for instrumentor, session_id in session_ids:
                await instrumentor.stop_monitoring(session_id)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Collect results
        total_tasks = sum(agent.task_count for agent in agents)
        total_errors = sum(len(agent.errors) for agent in agents)
        
        # Assertions
        assert actual_duration <= test_duration + 5, "Test took too long to complete"
        assert total_tasks > 0, "No tasks were executed"
        
        error_rate = total_errors / total_tasks if total_tasks > 0 else 0
        assert error_rate <= 0.1, f"Error rate {error_rate:.2%} too high"
        
        tasks_per_second = total_tasks / actual_duration
        print(f"Concurrent monitoring: {num_agents} agents, {tasks_per_second:.1f} tasks/s, {error_rate:.2%} errors")
    
    async def test_stress_monitoring_limits(self, load_test_config):
        """Test system behavior at monitoring limits."""
        # Gradually increase load until system shows stress
        agent_counts = [10, 25, 50, 100, 200]
        results = []
        
        for num_agents in agent_counts:
            agents = [LoadTestAgent(f"stress_agent_{i}", "heavy") for i in range(num_agents)]
            instrumentors = [LangChainInstrumentor() for _ in range(num_agents)]
            
            test_start = time.time()
            
            # Start monitoring
            session_ids = []
            try:
                for agent, instrumentor in zip(agents, instrumentors):
                    session_id = await instrumentor.start_monitoring(agent.agent_id, {})
                    session_ids.append((instrumentor, session_id))
                
                # Run stress test for shorter duration
                stress_duration = 30
                
                # Generate high-frequency events
                async def stress_event_generator():
                    event_count = 0
                    while time.time() - test_start < stress_duration:
                        tasks = []
                        for agent, instrumentor in zip(agents, instrumentors):
                            event = AgentEvent(
                                event_id=f"stress_event_{event_count}_{agent.agent_id}",
                                agent_id=agent.agent_id,
                                event_type=EventType.DECISION_MADE,
                                timestamp=time.time(),
                                data={"stress_test": True, "event_count": event_count}
                            )
                            tasks.append(instrumentor.capture_event(event))
                            event_count += 1
                        
                        await asyncio.gather(*tasks, return_exceptions=True)
                        await asyncio.sleep(0.001)  # 1ms between batches
                
                await stress_event_generator()
                
                test_end = time.time()
                duration = test_end - test_start
                
                results.append({
                    "agent_count": num_agents,
                    "duration": duration,
                    "completed": True
                })
                
                print(f"Stress test: {num_agents} agents completed in {duration:.1f}s")
            
            except Exception as e:
                results.append({
                    "agent_count": num_agents,
                    "duration": time.time() - test_start,
                    "completed": False,
                    "error": str(e)
                })
                print(f"Stress test: {num_agents} agents failed: {e}")
            
            finally:
                # Clean up monitoring sessions
                for instrumentor, session_id in session_ids:
                    try:
                        await instrumentor.stop_monitoring(session_id)
                    except:
                        pass  # Ignore cleanup errors during stress test
        
        # Verify that system handled reasonable load
        successful_tests = [r for r in results if r["completed"]]
        assert len(successful_tests) >= 3, "System failed under moderate load"
        
        max_successful_agents = max(r["agent_count"] for r in successful_tests)
        assert max_successful_agents >= 50, f"System only handled {max_successful_agents} agents"
    
    async def test_burst_load_handling(self, load_test_config):
        """Test system behavior under burst loads."""
        num_agents = 20
        burst_duration = 10  # seconds
        normal_duration = 30  # seconds
        
        agents = [LoadTestAgent(f"burst_agent_{i}", "bursty") for i in range(num_agents)]
        instrumentors = [LangChainInstrumentor() for _ in range(num_agents)]
        
        # Start monitoring
        session_ids = []
        for agent, instrumentor in zip(agents, instrumentors):
            session_id = await instrumentor.start_monitoring(agent.agent_id, {})
            session_ids.append((instrumentor, session_id))
        
        try:
            # Phase 1: Normal load
            print("Starting normal load phase...")
            normal_tasks = [agent.simulate_workload(normal_duration, 30) for agent in agents]
            
            # Phase 2: Burst load (after 15 seconds)
            await asyncio.sleep(15)
            print("Starting burst load phase...")
            
            # Generate burst of events
            async def burst_event_generator():
                for i in range(1000):  # 1000 events in burst
                    agent = np.random.choice(agents)
                    instrumentor = instrumentors[agents.index(agent)]
                    
                    await instrumentor.capture_event(AgentEvent(
                        event_id=f"burst_event_{i}",
                        agent_id=agent.agent_id,
                        event_type=EventType.DECISION_MADE,
                        timestamp=time.time(),
                        data={"burst_event": True, "sequence": i}
                    ))
                    
                    if i % 100 == 0:
                        await asyncio.sleep(0.01)  # Small pause every 100 events
            
            burst_task = asyncio.create_task(burst_event_generator())
            
            # Wait for normal load to complete and burst to finish
            await asyncio.gather(*normal_tasks, burst_task)
        
        finally:
            # Stop monitoring
            for instrumentor, session_id in session_ids:
                await instrumentor.stop_monitoring(session_id)
        
        # Verify system handled burst load
        total_tasks = sum(agent.task_count for agent in agents)
        total_errors = sum(len(agent.errors) for agent in agents)
        
        assert total_tasks > 0, "No tasks completed during burst test"
        
        error_rate = total_errors / total_tasks if total_tasks > 0 else 0
        assert error_rate <= 0.15, f"Error rate {error_rate:.2%} too high during burst"
        
        print(f"Burst test: {total_tasks} tasks, {error_rate:.2%} error rate")


@pytest.mark.asyncio
class TestAPILoadTesting:
    """Load test API endpoints under concurrent access."""
    
    async def test_monitoring_api_load(self, load_test_config):
        """Load test monitoring API endpoints."""
        
        def start_monitoring_data():
            return {
                "agent_id": f"api_test_agent_{np.random.randint(1000)}",
                "config": {"framework": "test", "level": "debug"}
            }
        
        async with APILoadTester() as tester:
            # Test different concurrent user levels
            for concurrent_users in [1, 5, 10, 25]:
                print(f"Testing with {concurrent_users} concurrent users...")
                
                result = await tester.load_test_endpoint(
                    method="POST",
                    endpoint="/api/v1/monitor/start",
                    concurrent_users=concurrent_users,
                    duration_seconds=30,
                    data_generator=start_monitoring_data
                )
                
                # Assertions
                assert result["error_rate"] <= 0.05, f"Error rate {result['error_rate']:.2%} too high"
                assert result["average_response_time"] <= 1.0, f"Average response time {result['average_response_time']:.3f}s too slow"
                assert result["p95_response_time"] <= 2.0, f"P95 response time {result['p95_response_time']:.3f}s too slow"
                
                print(f"  RPS: {result['requests_per_second']:.1f}, "
                      f"Avg RT: {result['average_response_time']*1000:.0f}ms, "
                      f"P95: {result['p95_response_time']*1000:.0f}ms, "
                      f"Errors: {result['error_rate']:.2%}")
    
    async def test_analysis_api_load(self, load_test_config):
        """Load test analysis API endpoints."""
        
        async with APILoadTester() as tester:
            endpoints_to_test = [
                ("GET", "/api/v1/epistemic/test_agent/current"),
                ("GET", "/api/v1/patterns/test_agent/analyze"),
                ("POST", "/api/v1/causal/analyze")
            ]
            
            for method, endpoint in endpoints_to_test:
                print(f"Load testing {method} {endpoint}...")
                
                data_generator = None
                if method == "POST":
                    data_generator = lambda: {
                        "agent_id": "test_agent",
                        "time_range": {"start": "2024-01-01", "end": "2024-01-02"}
                    }
                
                result = await tester.load_test_endpoint(
                    method=method,
                    endpoint=endpoint,
                    concurrent_users=10,
                    duration_seconds=30,
                    data_generator=data_generator
                )
                
                # More lenient assertions for analysis endpoints (they do more work)
                assert result["error_rate"] <= 0.10, f"Error rate {result['error_rate']:.2%} too high"
                assert result["average_response_time"] <= 2.0, f"Average response time {result['average_response_time']:.3f}s too slow"
                
                print(f"  {endpoint}: {result['requests_per_second']:.1f} RPS, "
                      f"{result['average_response_time']*1000:.0f}ms avg")


@pytest.mark.asyncio
class TestMemoryAndResourceUsage:
    """Test memory and resource usage under load."""
    
    async def test_memory_leak_detection(self, load_test_config):
        """Test for memory leaks during extended operation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run extended monitoring simulation
        num_agents = 10
        duration = 120  # 2 minutes
        
        agents = [LoadTestAgent(f"memory_agent_{i}") for i in range(num_agents)]
        instrumentors = [LangChainInstrumentor() for _ in range(num_agents)]
        
        session_ids = []
        for agent, instrumentor in zip(agents, instrumentors):
            session_id = await instrumentor.start_monitoring(agent.agent_id, {})
            session_ids.append((instrumentor, session_id))
        
        memory_samples = []
        
        try:
            start_time = time.time()
            
            # Monitor memory usage every 10 seconds
            async def memory_monitor():
                while time.time() - start_time < duration:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    await asyncio.sleep(10)
            
            # Generate continuous events
            async def event_generator():
                event_count = 0
                while time.time() - start_time < duration:
                    for agent, instrumentor in zip(agents, instrumentors):
                        await instrumentor.capture_event(AgentEvent(
                            event_id=f"memory_event_{event_count}",
                            agent_id=agent.agent_id,
                            event_type=EventType.DECISION_MADE,
                            timestamp=time.time(),
                            data={"memory_test": True, "count": event_count}
                        ))
                        event_count += 1
                    
                    await asyncio.sleep(0.1)
            
            # Run monitoring and event generation
            await asyncio.gather(
                memory_monitor(),
                event_generator(),
                *[agent.simulate_workload(duration, 60) for agent in agents]
            )
        
        finally:
            for instrumentor, session_id in session_ids:
                await instrumentor.stop_monitoring(session_id)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Check for memory leaks
        if len(memory_samples) > 1:
            # Calculate memory growth trend
            time_points = list(range(len(memory_samples)))
            memory_trend = np.polyfit(time_points, memory_samples, 1)[0]  # slope
            
            # Memory should not grow more than 1MB per minute
            max_growth_rate = 1.0  # MB per minute
            actual_growth_rate = memory_trend * 6  # samples every 10s, so *6 for per minute
            
            assert actual_growth_rate <= max_growth_rate, \
                f"Memory growing at {actual_growth_rate:.2f}MB/min, limit is {max_growth_rate}MB/min"
        
        # Total memory increase should be reasonable
        assert memory_increase <= 200, \
            f"Total memory increase {memory_increase:.1f}MB exceeds 200MB limit"
        
        print(f"Memory usage: initial={initial_memory:.1f}MB, final={final_memory:.1f}MB, "
              f"increase={memory_increase:.1f}MB")


if __name__ == "__main__":
    # Run load tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])