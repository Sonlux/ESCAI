"""
Load testing for WebSocket real-time interface.
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock
import pytest

from escai_framework.api.websocket import (
    manager, broadcast_epistemic_update, broadcast_system_alert,
    SubscriptionRequest
)
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState, BeliefType, GoalStatus


class LoadTestMetrics:
    """Collect and analyze load test metrics."""
    
    def __init__(self):
        self.connection_times = []
        self.message_send_times = []
        self.message_receive_times = []
        self.broadcast_times = []
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    def start_test(self):
        """Start timing the test."""
        self.start_time = time.time()
    
    def end_test(self):
        """End timing the test."""
        self.end_time = time.time()
    
    def add_connection_time(self, duration: float):
        """Add connection establishment time."""
        self.connection_times.append(duration)
    
    def add_message_send_time(self, duration: float):
        """Add message send time."""
        self.message_send_times.append(duration)
    
    def add_message_receive_time(self, duration: float):
        """Add message receive time."""
        self.message_receive_times.append(duration)
    
    def add_broadcast_time(self, duration: float):
        """Add broadcast time."""
        self.broadcast_times.append(duration)
    
    def add_error(self, error: str):
        """Add error."""
        self.errors.append(error)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        def safe_stats(data: List[float]) -> Dict[str, float]:
            if not data:
                return {"min": 0, "max": 0, "avg": 0, "p95": 0, "p99": 0}
            return {
                "min": min(data),
                "max": max(data),
                "avg": statistics.mean(data),
                "p95": statistics.quantiles(data, n=20)[18] if len(data) >= 20 else max(data),
                "p99": statistics.quantiles(data, n=100)[98] if len(data) >= 100 else max(data)
            }
        
        return {
            "total_time": total_time,
            "connection_stats": safe_stats(self.connection_times),
            "message_send_stats": safe_stats(self.message_send_times),
            "message_receive_stats": safe_stats(self.message_receive_times),
            "broadcast_stats": safe_stats(self.broadcast_times),
            "error_count": len(self.errors),
            "errors": self.errors[:10]  # First 10 errors
        }


@pytest.fixture
async def clean_manager_load():
    """Clean connection manager for load tests."""
    # Clear all connections
    manager.connections.clear()
    manager.user_connections.clear()
    manager.agent_subscribers.clear()
    manager.total_connections = 0
    manager.total_messages_sent = 0
    manager.total_messages_received = 0
    
    # Stop heartbeat if running
    if manager.heartbeat_manager.running:
        await manager.heartbeat_manager.stop()
    
    # Increase connection limit for load testing
    original_limit = manager.max_connections
    manager.max_connections = 1000
    
    yield manager
    
    # Cleanup after test
    manager.connections.clear()
    manager.user_connections.clear()
    manager.agent_subscribers.clear()
    manager.max_connections = original_limit
    if manager.heartbeat_manager.running:
        await manager.heartbeat_manager.stop()


class TestWebSocketLoadPerformance:
    """Test WebSocket performance under load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_connections_load(self, clean_manager_load):
        """Test performance with many concurrent connections."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        connection_count = 50
        connections = []
        
        # Create connections concurrently
        async def create_connection(i: int):
            start_time = time.time()
            try:
                mock_ws = AsyncMock()
                success = await manager.connect(mock_ws, f"conn_{i}", f"user_{i}")
                if success:
                    connections.append((f"conn_{i}", mock_ws))
                    metrics.add_connection_time(time.time() - start_time)
                else:
                    metrics.add_error(f"Failed to connect conn_{i}")
            except Exception as e:
                metrics.add_error(f"Connection error conn_{i}: {str(e)}")
        
        # Create connections in batches to avoid overwhelming
        batch_size = 10
        for i in range(0, connection_count, batch_size):
            batch_tasks = [
                create_connection(j) 
                for j in range(i, min(i + batch_size, connection_count))
            ]
            await asyncio.gather(*batch_tasks)
            await asyncio.sleep(0.1)  # Small delay between batches
        
        metrics.end_test()
        
        # Verify results
        assert len(connections) >= connection_count * 0.9  # Allow 10% failure rate
        assert len(manager.connections) == len(connections)
        
        summary = metrics.get_summary()
        print(f"Connection Load Test Summary: {summary}")
        
        # Performance assertions
        assert summary["connection_stats"]["avg"] < 0.1  # Average connection time < 100ms
        assert summary["error_count"] < connection_count * 0.1  # Less than 10% errors
    
    @pytest.mark.asyncio
    async def test_message_throughput_load(self, clean_manager_load):
        """Test message throughput under load."""
        metrics = LoadTestMetrics()
        
        # Setup connections
        connection_count = 20
        connections = []
        
        for i in range(connection_count):
            mock_ws = AsyncMock()
            await manager.connect(mock_ws, f"conn_{i}", f"user_{i}")
            connections.append((f"conn_{i}", mock_ws))
            
            # Subscribe to system alerts
            sub = SubscriptionRequest(type="system_alerts")
            manager.add_subscription(f"conn_{i}", sub)
        
        metrics.start_test()
        
        # Send messages rapidly
        message_count = 100
        
        async def send_message(i: int):
            start_time = time.time()
            try:
                sent_count = await broadcast_system_alert("load_test", f"Message {i}")
                metrics.add_broadcast_time(time.time() - start_time)
                if sent_count != connection_count:
                    metrics.add_error(f"Message {i} sent to {sent_count}/{connection_count} connections")
            except Exception as e:
                metrics.add_error(f"Broadcast error message {i}: {str(e)}")
        
        # Send messages in batches
        batch_size = 10
        for i in range(0, message_count, batch_size):
            batch_tasks = [
                send_message(j) 
                for j in range(i, min(i + batch_size, message_count))
            ]
            await asyncio.gather(*batch_tasks)
        
        metrics.end_test()
        
        summary = metrics.get_summary()
        print(f"Message Throughput Test Summary: {summary}")
        
        # Performance assertions
        assert summary["broadcast_stats"]["avg"] < 0.05  # Average broadcast time < 50ms
        assert summary["error_count"] < message_count * 0.05  # Less than 5% errors
        
        # Verify all connections received all messages
        for conn_id, mock_ws in connections:
            assert mock_ws.send_text.call_count == message_count
    
    @pytest.mark.asyncio
    async def test_subscription_filtering_load(self, clean_manager_load):
        """Test performance of subscription filtering under load."""
        metrics = LoadTestMetrics()
        
        # Setup connections with different filters
        connection_count = 30
        agent_count = 5
        
        for i in range(connection_count):
            mock_ws = AsyncMock()
            await manager.connect(mock_ws, f"conn_{i}", f"user_{i}")
            
            # Subscribe to different agents with different filters
            agent_id = f"agent_{i % agent_count}"
            confidence_threshold = 0.5 + (i % 5) * 0.1  # 0.5 to 0.9
            
            sub = SubscriptionRequest(
                type="epistemic_updates",
                agent_id=agent_id,
                filters={"confidence_level": {"min": confidence_threshold}}
            )
            manager.add_subscription(f"conn_{i}", sub)
        
        metrics.start_test()
        
        # Send epistemic updates with varying confidence levels
        update_count = 50
        
        for i in range(update_count):
            agent_id = f"agent_{i % agent_count}"
            confidence = 0.4 + (i % 10) * 0.06  # 0.4 to 1.0
            
            epistemic_state = EpistemicState(
                agent_id=agent_id,
                timestamp=datetime.utcnow(),
                belief_states=[BeliefState(content=f"belief_{i}", belief_type=BeliefType.FACTUAL, confidence=confidence)],
                knowledge_state=KnowledgeState(facts=[f"fact_{i}"], concepts={f"concept_{i}": f"value_{i}"}),
                goal_states=[GoalState(description=f"goal_{i}", status=GoalStatus.ACTIVE, priority=5, progress=0.5, sub_goals=[f"sub_{i}"])],
                confidence_level=confidence,
                uncertainty_score=1.0 - confidence,
                decision_context={"update_id": i}
            )
            
            start_time = time.time()
            try:
                filters = {"confidence_level": confidence}
                sent_count = await broadcast_epistemic_update(agent_id, epistemic_state, filters)
                metrics.add_broadcast_time(time.time() - start_time)
            except Exception as e:
                metrics.add_error(f"Broadcast error update {i}: {str(e)}")
        
        metrics.end_test()
        
        summary = metrics.get_summary()
        print(f"Subscription Filtering Test Summary: {summary}")
        
        # Performance assertions
        assert summary["broadcast_stats"]["avg"] < 0.1  # Average broadcast time < 100ms
        assert summary["error_count"] == 0  # No errors expected
    
    @pytest.mark.asyncio
    async def test_connection_churn_load(self, clean_manager_load):
        """Test performance with high connection churn."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        # Simulate connections coming and going
        churn_cycles = 10
        connections_per_cycle = 20
        
        for cycle in range(churn_cycles):
            # Connect phase
            connections = []
            for i in range(connections_per_cycle):
                start_time = time.time()
                try:
                    mock_ws = AsyncMock()
                    conn_id = f"cycle_{cycle}_conn_{i}"
                    success = await manager.connect(mock_ws, conn_id, f"user_{cycle}_{i}")
                    if success:
                        connections.append(conn_id)
                        metrics.add_connection_time(time.time() - start_time)
                    else:
                        metrics.add_error(f"Failed to connect {conn_id}")
                except Exception as e:
                    metrics.add_error(f"Connection error {conn_id}: {str(e)}")
            
            # Brief activity
            await asyncio.sleep(0.1)
            
            # Disconnect phase
            for conn_id in connections:
                try:
                    await manager.disconnect(conn_id)
                except Exception as e:
                    metrics.add_error(f"Disconnect error {conn_id}: {str(e)}")
            
            # Verify cleanup
            assert len(manager.connections) == 0
            assert len(manager.user_connections) == 0
        
        metrics.end_test()
        
        summary = metrics.get_summary()
        print(f"Connection Churn Test Summary: {summary}")
        
        # Performance assertions
        assert summary["connection_stats"]["avg"] < 0.05  # Average connection time < 50ms
        assert summary["error_count"] < churn_cycles * connections_per_cycle * 0.05  # Less than 5% errors
    
    @pytest.mark.asyncio
    async def test_heartbeat_load(self, clean_manager_load):
        """Test heartbeat performance with many connections."""
        # Setup many connections
        connection_count = 50
        connections = []
        
        for i in range(connection_count):
            mock_ws = AsyncMock()
            await manager.connect(mock_ws, f"conn_{i}", f"user_{i}")
            connections.append((f"conn_{i}", mock_ws))
        
        # Start heartbeat with shorter interval for testing
        original_interval = manager.heartbeat_manager.ping_interval
        manager.heartbeat_manager.ping_interval = 1  # 1 second
        
        try:
            # Let heartbeat run for a few cycles
            await asyncio.sleep(3.5)  # Should trigger 3 heartbeat cycles
            
            # Verify all connections received pings
            for conn_id, mock_ws in connections:
                # Should have received at least 2 pings (allowing for timing variations)
                assert mock_ws.send_text.call_count >= 2
                
                # Verify ping messages
                calls = mock_ws.send_text.call_args_list
                for call in calls:
                    message = json.loads(call[0][0])
                    assert message["type"] == "ping"
        
        finally:
            manager.heartbeat_manager.ping_interval = original_interval
    
    @pytest.mark.asyncio
    async def test_memory_usage_load(self, clean_manager_load):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many connections with subscriptions
        connection_count = 100
        
        for i in range(connection_count):
            mock_ws = AsyncMock()
            await manager.connect(mock_ws, f"conn_{i}", f"user_{i}")
            
            # Add multiple subscriptions per connection
            for j in range(3):
                sub = SubscriptionRequest(
                    type=["epistemic_updates", "pattern_alerts", "system_alerts"][j],
                    agent_id=f"agent_{i % 10}" if j < 2 else None,
                    filters={"test": f"filter_{i}_{j}"}
                )
                manager.add_subscription(f"conn_{i}", sub)
        
        # Send many messages
        for i in range(50):
            await broadcast_system_alert("memory_test", f"Message {i}")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50, f"Memory increase too high: {memory_increase:.1f}MB"
        
        # Cleanup and verify memory is released
        manager.connections.clear()
        manager.user_connections.clear()
        manager.agent_subscribers.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        await asyncio.sleep(0.1)  # Allow cleanup
        
        cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_released = final_memory - cleanup_memory
        
        print(f"Memory after cleanup: {cleanup_memory:.1f}MB (released {memory_released:.1f}MB)")
        
        # Should release most of the memory (at least 80%)
        assert memory_released > memory_increase * 0.8, f"Insufficient memory cleanup: {memory_released:.1f}MB"


class TestWebSocketStressTest:
    """Stress tests for WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_extreme_connection_count(self, clean_manager_load):
        """Test with extreme number of connections."""
        # This test pushes the limits - may need adjustment based on system capabilities
        connection_count = 200
        successful_connections = 0
        
        # Increase limits for stress test
        manager.max_connections = 500
        
        try:
            # Create connections in small batches to avoid overwhelming
            batch_size = 20
            for i in range(0, connection_count, batch_size):
                batch_tasks = []
                for j in range(i, min(i + batch_size, connection_count)):
                    async def create_conn(idx=j):
                        try:
                            mock_ws = AsyncMock()
                            success = await manager.connect(mock_ws, f"stress_conn_{idx}", f"stress_user_{idx}")
                            return success
                        except Exception:
                            return False
                    
                    batch_tasks.append(create_conn())
                
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                successful_connections += sum(1 for r in results if r is True)
                
                # Small delay between batches
                await asyncio.sleep(0.05)
            
            print(f"Stress test: {successful_connections}/{connection_count} connections successful")
            
            # Should achieve at least 80% success rate
            success_rate = successful_connections / connection_count
            assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2%}"
            
            # Test broadcasting to all connections
            if successful_connections > 0:
                # Subscribe all to system alerts
                for i in range(successful_connections):
                    try:
                        sub = SubscriptionRequest(type="system_alerts")
                        manager.add_subscription(f"stress_conn_{i}", sub)
                    except Exception:
                        pass  # Some may fail, that's ok in stress test
                
                # Broadcast message
                sent_count = await broadcast_system_alert("stress_test", "Stress test message")
                print(f"Stress broadcast: {sent_count} recipients")
                
                # Should reach most connections
                assert sent_count >= successful_connections * 0.8
        
        finally:
            # Cleanup
            manager.connections.clear()
            manager.user_connections.clear()
            manager.agent_subscribers.clear()
    
    @pytest.mark.asyncio
    async def test_message_flood(self, clean_manager_load):
        """Test handling message flood."""
        # Setup moderate number of connections
        connection_count = 30
        
        for i in range(connection_count):
            mock_ws = AsyncMock()
            await manager.connect(mock_ws, f"flood_conn_{i}", f"flood_user_{i}")
            
            sub = SubscriptionRequest(type="system_alerts")
            manager.add_subscription(f"flood_conn_{i}", sub)
        
        # Send flood of messages
        message_count = 500
        start_time = time.time()
        
        # Send messages as fast as possible
        tasks = []
        for i in range(message_count):
            task = broadcast_system_alert("flood_test", f"Flood message {i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_broadcasts = sum(1 for r in results if isinstance(r, int) and r > 0)
        total_time = end_time - start_time
        messages_per_second = message_count / total_time
        
        print(f"Message flood: {successful_broadcasts}/{message_count} successful in {total_time:.2f}s ({messages_per_second:.1f} msg/s)")
        
        # Should handle at least 90% of messages successfully
        success_rate = successful_broadcasts / message_count
        assert success_rate >= 0.9, f"Message flood success rate too low: {success_rate:.2%}"
        
        # Should achieve reasonable throughput (at least 50 messages/second)
        assert messages_per_second >= 50, f"Throughput too low: {messages_per_second:.1f} msg/s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])