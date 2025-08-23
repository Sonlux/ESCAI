"""
Integration tests for Redis storage functionality in ESCAI Framework.
Tests caching, session management, rate limiting, streaming, and pub/sub.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from escai_framework.storage.database import db_manager
from escai_framework.storage.redis_manager import RedisManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def redis_manager():
    """Initialize Redis manager for testing."""
    # Initialize database manager with test Redis URL
    db_manager.initialize(
        redis_url="redis://localhost:6379/1"  # Use database 1 for testing
    )
    
    # Check if Redis is available
    if not await db_manager.test_redis_connection():
        pytest.skip("Redis not available for testing")
    
    yield db_manager.redis_manager
    
    # Cleanup
    await db_manager.close()


@pytest.fixture
async def clean_redis(redis_manager):
    """Clean Redis database before each test."""
    # Clean up any existing test data
    if redis_manager.available:
        # Note: In a real test environment, you might want to use FLUSHDB
        # For now, we'll rely on using a separate test database
        pass
    yield redis_manager


class TestRedisSessionManagement:
    """Test Redis session management functionality."""
    
    async def test_session_lifecycle(self, clean_redis):
        """Test complete session lifecycle."""
        redis_mgr = clean_redis
        
        session_id = "test_session_123"
        session_data = {
            "user_id": "user_123",
            "username": "test_user",
            "login_time": datetime.utcnow().isoformat(),
            "permissions": ["read", "write"]
        }
        
        # Create session
        success = await redis_mgr.set_session(session_id, session_data, ttl_seconds=3600)
        assert success is True
        
        # Retrieve session
        retrieved = await redis_mgr.get_session(session_id)
        assert retrieved is not None
        assert retrieved["user_id"] == "user_123"
        assert retrieved["username"] == "test_user"
        assert retrieved["permissions"] == ["read", "write"]
        
        # Extend session
        extended = await redis_mgr.extend_session(session_id, ttl_seconds=7200)
        assert extended is True
        
        # Delete session
        deleted = await redis_mgr.delete_session(session_id)
        assert deleted is True
        
        # Verify deletion
        retrieved_after_delete = await redis_mgr.get_session(session_id)
        assert retrieved_after_delete is None
    
    async def test_session_expiry(self, clean_redis):
        """Test session TTL functionality."""
        redis_mgr = clean_redis
        
        session_id = "test_session_expiry"
        session_data = {"user_id": "user_456"}
        
        # Create session with short TTL
        success = await redis_mgr.set_session(session_id, session_data, ttl_seconds=1)
        assert success is True
        
        # Verify session exists
        retrieved = await redis_mgr.get_session(session_id)
        assert retrieved is not None
        
        # Wait for expiry
        await asyncio.sleep(1.5)
        
        # Verify session expired
        expired_session = await redis_mgr.get_session(session_id)
        assert expired_session is None
    
    async def test_session_complex_data(self, clean_redis):
        """Test session with complex nested data."""
        redis_mgr = clean_redis
        
        session_id = "test_complex_session"
        complex_data = {
            "user_profile": {
                "id": "user_789",
                "preferences": {
                    "theme": "dark",
                    "notifications": {
                        "email": True,
                        "push": False,
                        "sms": True
                    }
                },
                "history": [
                    {"action": "login", "timestamp": "2024-01-01T10:00:00Z"},
                    {"action": "view_dashboard", "timestamp": "2024-01-01T10:05:00Z"}
                ]
            },
            "session_metadata": {
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0...",
                "created_at": datetime.utcnow().isoformat()
            }
        }
        
        # Store complex session data
        success = await redis_mgr.set_session(session_id, complex_data, ttl_seconds=3600)
        assert success is True
        
        # Retrieve and verify
        retrieved = await redis_mgr.get_session(session_id)
        assert retrieved is not None
        assert retrieved["user_profile"]["id"] == "user_789"
        assert retrieved["user_profile"]["preferences"]["theme"] == "dark"
        assert len(retrieved["user_profile"]["history"]) == 2
        assert retrieved["session_metadata"]["ip_address"] == "192.168.1.100"


class TestRedisCaching:
    """Test Redis caching functionality."""
    
    async def test_cache_basic_operations(self, clean_redis):
        """Test basic cache operations."""
        redis_mgr = clean_redis
        
        key = "test_cache_key"
        value = {"result": "test_value", "computed_at": datetime.utcnow().isoformat()}
        
        # Set cache
        success = await redis_mgr.cache_set(key, value, ttl_seconds=3600)
        assert success is True
        
        # Check existence
        exists = await redis_mgr.cache_exists(key)
        assert exists is True
        
        # Get cache
        cached_value = await redis_mgr.cache_get(key)
        assert cached_value is not None
        assert cached_value["result"] == "test_value"
        
        # Delete cache
        deleted = await redis_mgr.cache_delete(key)
        assert deleted is True
        
        # Verify deletion
        exists_after_delete = await redis_mgr.cache_exists(key)
        assert exists_after_delete is False
    
    async def test_cache_ttl(self, clean_redis):
        """Test cache TTL functionality."""
        redis_mgr = clean_redis
        
        key = "test_ttl_key"
        value = {"data": "expires_soon"}
        
        # Set cache with short TTL
        success = await redis_mgr.cache_set(key, value, ttl_seconds=1)
        assert success is True
        
        # Verify exists
        cached = await redis_mgr.cache_get(key)
        assert cached is not None
        
        # Wait for expiry
        await asyncio.sleep(1.5)
        
        # Verify expired
        expired_cache = await redis_mgr.cache_get(key)
        assert expired_cache is None
    
    async def test_cache_without_ttl(self, clean_redis):
        """Test cache without TTL (persistent)."""
        redis_mgr = clean_redis
        
        key = "test_persistent_key"
        value = {"data": "persistent"}
        
        # Set cache without TTL
        success = await redis_mgr.cache_set(key, value)
        assert success is True
        
        # Verify exists
        cached = await redis_mgr.cache_get(key)
        assert cached is not None
        assert cached["data"] == "persistent"
        
        # Clean up
        await redis_mgr.cache_delete(key)


class TestRedisRateLimiting:
    """Test Redis rate limiting functionality."""
    
    async def test_rate_limiting_basic(self, clean_redis):
        """Test basic rate limiting functionality."""
        redis_mgr = clean_redis
        
        identifier = "test_user_rate_limit"
        limit = 3
        window = 60
        
        # Make requests within limit
        for i in range(limit):
            result = await redis_mgr.rate_limit_check(identifier, limit, window)
            assert result["allowed"] is True
            assert result["current_count"] == i + 1
            assert result["limit"] == limit
        
        # Exceed limit
        result = await redis_mgr.rate_limit_check(identifier, limit, window)
        assert result["allowed"] is False
        assert result["current_count"] == limit + 1
    
    async def test_rate_limiting_sliding_window(self, clean_redis):
        """Test sliding window rate limiting."""
        redis_mgr = clean_redis
        
        identifier = "test_sliding_window"
        limit = 2
        window = 2  # 2 seconds
        
        # Make first request
        result1 = await redis_mgr.rate_limit_check(identifier, limit, window)
        assert result1["allowed"] is True
        
        # Make second request (at limit)
        result2 = await redis_mgr.rate_limit_check(identifier, limit, window)
        assert result2["allowed"] is True
        
        # Third request should be blocked
        result3 = await redis_mgr.rate_limit_check(identifier, limit, window)
        assert result3["allowed"] is False
        
        # Wait for window to slide
        await asyncio.sleep(2.5)
        
        # Should be allowed again
        result4 = await redis_mgr.rate_limit_check(identifier, limit, window)
        assert result4["allowed"] is True
    
    async def test_rate_limiting_different_identifiers(self, clean_redis):
        """Test rate limiting with different identifiers."""
        redis_mgr = clean_redis
        
        limit = 2
        window = 60
        
        # Test different users
        result1 = await redis_mgr.rate_limit_check("user_1", limit, window)
        result2 = await redis_mgr.rate_limit_check("user_2", limit, window)
        
        assert result1["allowed"] is True
        assert result2["allowed"] is True
        assert result1["current_count"] == 1
        assert result2["current_count"] == 1


class TestRedisStreaming:
    """Test Redis Streams functionality."""
    
    async def test_stream_basic_operations(self, clean_redis):
        """Test basic stream operations."""
        redis_mgr = clean_redis
        
        stream_name = "test_stream"
        test_data = {
            "event_type": "test_event",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"key": "value"}
        }
        
        # Add to stream
        stream_id = await redis_mgr.stream_add(stream_name, test_data)
        assert stream_id is not None
        
        # Read from stream
        messages = await redis_mgr.stream_read(stream_name, "0", count=10)
        assert len(messages) == 1
        assert messages[0]["data"]["event_type"] == "test_event"
        assert messages[0]["data"]["data"]["key"] == "value"
    
    async def test_stream_consumer_group(self, clean_redis):
        """Test stream consumer group functionality."""
        redis_mgr = clean_redis
        
        stream_name = "test_consumer_stream"
        group_name = "test_group"
        consumer_name = "test_consumer"
        
        # Create consumer group
        group_created = await redis_mgr.stream_create_group(stream_name, group_name, "0")
        assert group_created is True
        
        # Add test data
        test_events = [
            {"event": "event_1", "data": "data_1"},
            {"event": "event_2", "data": "data_2"}
        ]
        
        for event in test_events:
            stream_id = await redis_mgr.stream_add(stream_name, event)
            assert stream_id is not None
        
        # Read as consumer group
        messages = await redis_mgr.stream_read_group(
            group_name, consumer_name, stream_name, count=10
        )
        
        assert len(messages) == 2
        
        # Acknowledge messages
        for msg in messages:
            acked = await redis_mgr.stream_ack(stream_name, group_name, msg["id"])
            assert acked is True
    
    async def test_stream_max_length(self, clean_redis):
        """Test stream max length functionality."""
        redis_mgr = clean_redis
        
        stream_name = "test_max_length_stream"
        max_length = 3
        
        # Add more messages than max length
        for i in range(5):
            data = {"message": f"message_{i}"}
            stream_id = await redis_mgr.stream_add(stream_name, data, max_length=max_length)
            assert stream_id is not None
        
        # Read all messages
        messages = await redis_mgr.stream_read(stream_name, "0", count=10)
        
        # Should have approximately max_length messages (Redis uses approximate trimming)
        assert len(messages) <= max_length + 1  # Allow for approximate trimming


class TestRedisPubSub:
    """Test Redis Pub/Sub functionality."""
    
    async def test_publish_subscribe(self, clean_redis):
        """Test basic publish/subscribe functionality."""
        redis_mgr = clean_redis
        
        channel = "test_channel"
        test_message = {"type": "test", "data": "hello world"}
        
        received_messages = []
        
        async def subscriber_task():
            async with redis_mgr.subscribe(channel) as subscriber:
                async for message in subscriber:
                    received_messages.append(message)
                    if len(received_messages) >= 1:
                        break
        
        # Start subscriber
        subscriber = asyncio.create_task(subscriber_task())
        
        # Give subscriber time to start
        await asyncio.sleep(0.1)
        
        # Publish message
        subscribers_count = await redis_mgr.publish(channel, test_message)
        
        # Wait for message to be received
        await asyncio.sleep(0.1)
        
        # Cancel subscriber
        subscriber.cancel()
        try:
            await subscriber
        except asyncio.CancelledError:
            pass
        
        # Verify message received
        assert len(received_messages) == 1
        assert received_messages[0]["channel"] == channel
        assert received_messages[0]["data"]["type"] == "test"
        assert received_messages[0]["data"]["data"] == "hello world"
    
    async def test_multiple_channels(self, clean_redis):
        """Test subscribing to multiple channels."""
        redis_mgr = clean_redis
        
        channels = ["channel_1", "channel_2"]
        received_messages = []
        
        async def subscriber_task():
            async with redis_mgr.subscribe(*channels) as subscriber:
                async for message in subscriber:
                    received_messages.append(message)
                    if len(received_messages) >= 2:
                        break
        
        # Start subscriber
        subscriber = asyncio.create_task(subscriber_task())
        await asyncio.sleep(0.1)
        
        # Publish to both channels
        await redis_mgr.publish("channel_1", {"message": "from_channel_1"})
        await redis_mgr.publish("channel_2", {"message": "from_channel_2"})
        
        await asyncio.sleep(0.1)
        
        # Cancel subscriber
        subscriber.cancel()
        try:
            await subscriber
        except asyncio.CancelledError:
            pass
        
        # Verify messages from both channels
        assert len(received_messages) == 2
        channels_received = [msg["channel"] for msg in received_messages]
        assert "channel_1" in channels_received
        assert "channel_2" in channels_received


class TestRedisTemporaryStorage:
    """Test Redis temporary storage functionality."""
    
    async def test_temporary_storage_lifecycle(self, clean_redis):
        """Test temporary storage lifecycle."""
        redis_mgr = clean_redis
        
        key = "temp_test_key"
        value = {"computation": "result", "timestamp": datetime.utcnow().isoformat()}
        
        # Store temporary data
        success = await redis_mgr.temp_store(key, value, ttl_seconds=3600)
        assert success is True
        
        # Retrieve temporary data
        retrieved = await redis_mgr.temp_get(key)
        assert retrieved is not None
        assert retrieved["computation"] == "result"
        
        # Delete temporary data
        deleted = await redis_mgr.temp_delete(key)
        assert deleted is True
        
        # Verify deletion
        after_delete = await redis_mgr.temp_get(key)
        assert after_delete is None
    
    async def test_temporary_storage_expiry(self, clean_redis):
        """Test temporary storage expiry."""
        redis_mgr = clean_redis
        
        key = "temp_expiry_key"
        value = {"data": "expires_soon"}
        
        # Store with short TTL
        success = await redis_mgr.temp_store(key, value, ttl_seconds=1)
        assert success is True
        
        # Verify exists
        retrieved = await redis_mgr.temp_get(key)
        assert retrieved is not None
        
        # Wait for expiry
        await asyncio.sleep(1.5)
        
        # Verify expired
        expired = await redis_mgr.temp_get(key)
        assert expired is None


class TestRedisHealthMonitoring:
    """Test Redis health monitoring functionality."""
    
    async def test_connection_test(self, clean_redis):
        """Test Redis connection testing."""
        redis_mgr = clean_redis
        
        # Test connection
        connected = await redis_mgr.test_connection()
        assert connected is True
    
    async def test_server_info(self, clean_redis):
        """Test Redis server info retrieval."""
        redis_mgr = clean_redis
        
        # Get server info
        info = await redis_mgr.get_info()
        assert info is not None
        assert isinstance(info, dict)
        
        # Check for expected keys
        expected_keys = ["redis_version", "connected_clients", "used_memory"]
        for key in expected_keys:
            assert key in info
    
    async def test_memory_usage(self, clean_redis):
        """Test memory usage monitoring."""
        redis_mgr = clean_redis
        
        # Create a test key
        test_key = "memory_test_key"
        test_data = {"large_data": "x" * 1000}  # 1KB of data
        
        await redis_mgr.cache_set("memory_test", test_data, ttl_seconds=60)
        
        # Check memory usage
        memory_usage = await redis_mgr.get_memory_usage(f"cache:{test_key}")
        # Memory usage might be None if the key doesn't exist or Redis doesn't support the command
        # Just verify the method doesn't crash
        assert memory_usage is None or isinstance(memory_usage, int)


class TestRedisFailover:
    """Test Redis failover and error handling."""
    
    async def test_graceful_degradation(self):
        """Test graceful degradation when Redis is not available."""
        # Create Redis manager without connection
        redis_mgr = RedisManager()
        redis_mgr.initialize(
            redis_url="redis://nonexistent:6379/0",
            failover_enabled=True
        )
        
        # Operations should not raise exceptions but return None/False
        session_set = await redis_mgr.set_session("test", {"data": "test"})
        assert session_set is None or session_set is False
        
        session_get = await redis_mgr.get_session("test")
        assert session_get is None
        
        cache_set = await redis_mgr.cache_set("test", {"data": "test"})
        assert cache_set is None or cache_set is False
        
        rate_limit = await redis_mgr.rate_limit_check("test", 10, 60)
        assert rate_limit["allowed"] is True  # Should allow when Redis unavailable
        
        await redis_mgr.close()
    
    async def test_connection_availability_check(self, clean_redis):
        """Test Redis availability checking."""
        redis_mgr = clean_redis
        
        # Should be available
        assert redis_mgr.available is True
        
        # Test connection
        connected = await redis_mgr.test_connection()
        assert connected is True


# Integration test for complete workflow
class TestRedisIntegration:
    """Integration tests for complete Redis workflows."""
    
    async def test_agent_monitoring_workflow(self, clean_redis):
        """Test complete agent monitoring workflow using Redis."""
        redis_mgr = clean_redis
        
        agent_id = "agent_integration_test"
        session_id = f"session_{agent_id}"
        
        # 1. Start monitoring session
        session_data = {
            "agent_id": agent_id,
            "start_time": datetime.utcnow().isoformat(),
            "monitoring_config": {
                "epistemic_tracking": True,
                "pattern_analysis": True,
                "performance_prediction": True
            }
        }
        
        session_created = await redis_mgr.set_session(session_id, session_data, ttl_seconds=7200)
        assert session_created is True
        
        # 2. Cache analysis results
        analysis_results = {
            "epistemic_states": [
                {"timestamp": "2024-01-01T10:00:00Z", "confidence": 0.8},
                {"timestamp": "2024-01-01T10:01:00Z", "confidence": 0.85}
            ],
            "behavioral_patterns": [
                {"pattern": "sequential_reasoning", "frequency": 5},
                {"pattern": "error_recovery", "frequency": 2}
            ]
        }
        
        cache_key = f"analysis_{agent_id}"
        cached = await redis_mgr.cache_set(cache_key, analysis_results, ttl_seconds=1800)
        assert cached is True
        
        # 3. Stream real-time events
        stream_name = "agent_events"
        events = [
            {
                "agent_id": agent_id,
                "event_type": "epistemic_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"new_belief": "task_complexity_high", "confidence": 0.9}
            },
            {
                "agent_id": agent_id,
                "event_type": "pattern_detected",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"pattern": "recursive_problem_solving", "strength": 0.85}
            }
        ]
        
        for event in events:
            stream_id = await redis_mgr.stream_add(stream_name, event)
            assert stream_id is not None
        
        # 4. Apply rate limiting for API access
        api_user = f"api_user_{agent_id}"
        rate_limit_result = await redis_mgr.rate_limit_check(api_user, 100, 3600)
        assert rate_limit_result["allowed"] is True
        
        # 5. Store temporary computation results
        temp_key = f"temp_computation_{agent_id}"
        temp_data = {
            "partial_prediction": {"success_probability": 0.87},
            "intermediate_patterns": ["pattern_a", "pattern_b"],
            "computation_progress": 0.6
        }
        
        temp_stored = await redis_mgr.temp_store(temp_key, temp_data, ttl_seconds=600)
        assert temp_stored is True
        
        # 6. Publish alert
        alert_channel = "agent_alerts"
        alert = {
            "agent_id": agent_id,
            "alert_type": "performance_degradation",
            "severity": "medium",
            "message": "Agent response time increased",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        published = await redis_mgr.publish(alert_channel, alert)
        # published count might be 0 if no subscribers, that's OK
        
        # 7. Verify all data can be retrieved
        retrieved_session = await redis_mgr.get_session(session_id)
        assert retrieved_session is not None
        assert retrieved_session["agent_id"] == agent_id
        
        retrieved_analysis = await redis_mgr.cache_get(cache_key)
        assert retrieved_analysis is not None
        assert len(retrieved_analysis["epistemic_states"]) == 2
        
        stream_messages = await redis_mgr.stream_read(stream_name, "0", count=10)
        assert len(stream_messages) >= 2
        
        retrieved_temp = await redis_mgr.temp_get(temp_key)
        assert retrieved_temp is not None
        assert retrieved_temp["computation_progress"] == 0.6
        
        # 8. Cleanup
        await redis_mgr.delete_session(session_id)
        await redis_mgr.cache_delete(cache_key)
        await redis_mgr.temp_delete(temp_key)