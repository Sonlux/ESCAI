"""
Unit tests for Redis manager functionality.
Tests Redis manager initialization, configuration, and error handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
from datetime import datetime

from escai_framework.storage.redis_manager import RedisManager
from redis.exceptions import ConnectionError, TimeoutError, RedisError


class TestRedisManagerInitialization:
    """Test Redis manager initialization and configuration."""
    
    def test_redis_manager_init(self):
        """Test Redis manager initialization."""
        redis_mgr = RedisManager()
        
        assert redis_mgr._redis_client is None
        assert redis_mgr._connection_pool is None
        assert redis_mgr._initialized is False
        assert redis_mgr._max_retries == 3
        assert redis_mgr._retry_delay == 1.0
    
    @patch('escai_framework.storage.redis_manager.ConnectionPool')
    @patch('escai_framework.storage.redis_manager.Redis')
    def test_redis_manager_initialize_success(self, mock_redis, mock_pool):
        """Test successful Redis manager initialization."""
        # Mock connection pool and Redis client
        mock_pool_instance = Mock()
        mock_pool.from_url.return_value = mock_pool_instance
        
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        redis_mgr = RedisManager()
        redis_mgr.initialize(
            redis_url="redis://localhost:6379/0",
            max_connections=20,
            max_retries=5,
            retry_delay=2.0
        )
        
        # Verify initialization
        assert redis_mgr._initialized is True
        assert redis_mgr._max_retries == 5
        assert redis_mgr._retry_delay == 2.0
        assert redis_mgr._redis_client == mock_redis_instance
        assert redis_mgr._connection_pool == mock_pool_instance
        
        # Verify connection pool creation
        mock_pool.from_url.assert_called_once_with(
            "redis://localhost:6379/0",
            max_connections=20,
            retry_on_timeout=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            health_check_interval=30,
            decode_responses=True
        )
    
    @patch('escai_framework.storage.redis_manager.ConnectionPool')
    def test_redis_manager_initialize_failure_with_failover(self, mock_pool):
        """Test Redis manager initialization failure with failover enabled."""
        # Mock connection pool to raise exception
        mock_pool.from_url.side_effect = ConnectionError("Connection failed")
        
        redis_mgr = RedisManager()
        
        # Should not raise exception with failover enabled
        redis_mgr.initialize(
            redis_url="redis://invalid:6379/0",
            failover_enabled=True
        )
        
        # Should continue without Redis
        assert redis_mgr._redis_client is None
        assert redis_mgr._connection_pool is None
        assert redis_mgr.available is False
    
    @patch('escai_framework.storage.redis_manager.ConnectionPool')
    def test_redis_manager_initialize_failure_without_failover(self, mock_pool):
        """Test Redis manager initialization failure without failover."""
        # Mock connection pool to raise exception
        mock_pool.from_url.side_effect = ConnectionError("Connection failed")
        
        redis_mgr = RedisManager()
        
        # Should raise exception without failover
        with pytest.raises(ConnectionError):
            redis_mgr.initialize(
                redis_url="redis://invalid:6379/0",
                failover_enabled=False
            )
    
    def test_redis_manager_double_initialization(self):
        """Test double initialization warning."""
        redis_mgr = RedisManager()
        
        with patch('escai_framework.storage.redis_manager.ConnectionPool'):
            with patch('escai_framework.storage.redis_manager.Redis'):
                redis_mgr.initialize()
                
                # Second initialization should log warning
                with patch('escai_framework.storage.redis_manager.logger') as mock_logger:
                    redis_mgr.initialize()
                    mock_logger.warning.assert_called_with("Redis manager already initialized")


class TestRedisManagerConnectionTesting:
    """Test Redis connection testing functionality."""
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful connection test."""
        redis_mgr = RedisManager()
        
        # Mock Redis client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        redis_mgr._redis_client = mock_client
        
        result = await redis_mgr.test_connection()
        
        assert result is True
        mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test connection test failure."""
        redis_mgr = RedisManager()
        
        # Mock Redis client to raise exception
        mock_client = AsyncMock()
        mock_client.ping.side_effect = ConnectionError("Connection failed")
        redis_mgr._redis_client = mock_client
        
        result = await redis_mgr.test_connection()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_test_connection_no_client(self):
        """Test connection test with no client."""
        redis_mgr = RedisManager()
        
        result = await redis_mgr.test_connection()
        
        assert result is False
    
    def test_available_property(self):
        """Test available property."""
        redis_mgr = RedisManager()
        
        # No client
        assert redis_mgr.available is False
        
        # With client
        redis_mgr._redis_client = Mock()
        assert redis_mgr.available is True


class TestRedisManagerRetryLogic:
    """Test Redis manager retry logic and error handling."""
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """Test successful operation without retry."""
        redis_mgr = RedisManager()
        redis_mgr._redis_client = AsyncMock()
        redis_mgr._max_retries = 3
        redis_mgr._retry_delay = 0.1
        
        # Mock operation
        mock_operation = AsyncMock(return_value="success")
        
        result = await redis_mgr._execute_with_retry(mock_operation, "arg1", key="value")
        
        assert result == "success"
        mock_operation.assert_called_once_with("arg1", key="value")
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_failure_with_failover(self):
        """Test operation failure with failover enabled."""
        redis_mgr = RedisManager()
        redis_mgr._redis_client = AsyncMock()
        redis_mgr._max_retries = 2
        redis_mgr._retry_delay = 0.01  # Short delay for testing
        redis_mgr._failover_enabled = True
        
        # Mock operation to always fail
        mock_operation = AsyncMock(side_effect=ConnectionError("Connection failed"))
        
        result = await redis_mgr._execute_with_retry(mock_operation)
        
        assert result is None
        assert mock_operation.call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_failure_without_failover(self):
        """Test operation failure without failover."""
        redis_mgr = RedisManager()
        redis_mgr._redis_client = AsyncMock()
        redis_mgr._max_retries = 1
        redis_mgr._retry_delay = 0.01
        redis_mgr._failover_enabled = False
        
        # Mock operation to always fail
        mock_operation = AsyncMock(side_effect=TimeoutError("Timeout"))
        
        with pytest.raises(TimeoutError):
            await redis_mgr._execute_with_retry(mock_operation)
        
        assert mock_operation.call_count == 2  # Initial + 1 retry
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_no_client_with_failover(self):
        """Test operation with no client and failover enabled."""
        redis_mgr = RedisManager()
        redis_mgr._redis_client = None
        redis_mgr._failover_enabled = True
        
        mock_operation = AsyncMock()
        
        result = await redis_mgr._execute_with_retry(mock_operation)
        
        assert result is None
        mock_operation.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_no_client_without_failover(self):
        """Test operation with no client and failover disabled."""
        redis_mgr = RedisManager()
        redis_mgr._redis_client = None
        redis_mgr._failover_enabled = False
        
        mock_operation = AsyncMock()
        
        with pytest.raises(RuntimeError, match="Redis not available"):
            await redis_mgr._execute_with_retry(mock_operation)
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_exponential_backoff(self):
        """Test exponential backoff in retry logic."""
        redis_mgr = RedisManager()
        redis_mgr._redis_client = AsyncMock()
        redis_mgr._max_retries = 2
        redis_mgr._retry_delay = 0.1
        redis_mgr._failover_enabled = True
        
        # Mock operation to fail first two times, succeed third time
        mock_operation = AsyncMock(side_effect=[
            ConnectionError("Fail 1"),
            ConnectionError("Fail 2"),
            "success"
        ])
        
        with patch('asyncio.sleep') as mock_sleep:
            result = await redis_mgr._execute_with_retry(mock_operation)
        
        assert result == "success"
        assert mock_operation.call_count == 3
        
        # Verify exponential backoff
        expected_delays = [0.1, 0.2]  # 0.1 * 2^0, 0.1 * 2^1
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays


class TestRedisManagerDataSerialization:
    """Test Redis manager data serialization and deserialization."""
    
    @pytest.mark.asyncio
    async def test_session_data_serialization(self):
        """Test session data serialization/deserialization."""
        redis_mgr = RedisManager()
        
        # Mock Redis client
        mock_client = AsyncMock()
        redis_mgr._redis_client = mock_client
        
        session_data = {
            "user_id": "user_123",
            "login_time": datetime.utcnow(),
            "permissions": ["read", "write"],
            "metadata": {
                "ip": "192.168.1.1",
                "user_agent": "Mozilla/5.0"
            }
        }
        
        # Test set session
        await redis_mgr.set_session("test_session", session_data, 3600)
        
        # Verify setex was called with serialized data
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        
        assert call_args[0][0] == "session:test_session"
        assert call_args[0][1] == 3600
        
        # Verify data was serialized as JSON
        serialized_data = call_args[0][2]
        import json
        deserialized = json.loads(serialized_data)
        assert deserialized["user_id"] == "user_123"
        assert deserialized["permissions"] == ["read", "write"]
    
    @pytest.mark.asyncio
    async def test_cache_complex_data_serialization(self):
        """Test caching complex data structures."""
        redis_mgr = RedisManager()
        
        # Mock Redis client
        mock_client = AsyncMock()
        redis_mgr._redis_client = mock_client
        
        complex_data = {
            "analysis_results": {
                "patterns": [
                    {"name": "pattern_1", "frequency": 5, "confidence": 0.9},
                    {"name": "pattern_2", "frequency": 3, "confidence": 0.7}
                ],
                "statistics": {
                    "total_events": 100,
                    "success_rate": 0.85,
                    "average_duration": 45.6
                }
            },
            "metadata": {
                "computed_at": datetime.utcnow(),
                "version": "1.0.0"
            }
        }
        
        # Test cache set
        await redis_mgr.cache_set("complex_key", complex_data, 1800)
        
        # Verify setex was called
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        
        assert call_args[0][0] == "cache:complex_key"
        assert call_args[0][1] == 1800
        
        # Verify data serialization
        serialized_data = call_args[0][2]
        import json
        deserialized = json.loads(serialized_data)
        assert len(deserialized["analysis_results"]["patterns"]) == 2
        assert deserialized["analysis_results"]["statistics"]["total_events"] == 100


class TestRedisManagerCleanup:
    """Test Redis manager cleanup and resource management."""
    
    @pytest.mark.asyncio
    async def test_close_connections(self):
        """Test closing Redis connections."""
        redis_mgr = RedisManager()
        
        # Mock Redis client and connection pool
        mock_client = AsyncMock()
        mock_pool = AsyncMock()
        
        redis_mgr._redis_client = mock_client
        redis_mgr._connection_pool = mock_pool
        redis_mgr._initialized = True
        
        await redis_mgr.close()
        
        # Verify cleanup
        mock_client.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()
        assert redis_mgr._initialized is False
    
    @pytest.mark.asyncio
    async def test_close_no_connections(self):
        """Test closing when no connections exist."""
        redis_mgr = RedisManager()
        
        # Should not raise exception
        await redis_mgr.close()
        
        assert redis_mgr._initialized is False


class TestRedisManagerEnvironmentConfiguration:
    """Test Redis manager environment variable configuration."""
    
    @patch.dict('os.environ', {'ESCAI_REDIS_URL': 'redis://env-host:6379/2'})
    @patch('escai_framework.storage.redis_manager.ConnectionPool')
    @patch('escai_framework.storage.redis_manager.Redis')
    def test_environment_variable_configuration(self, mock_redis, mock_pool):
        """Test configuration from environment variables."""
        redis_mgr = RedisManager()
        redis_mgr.initialize()
        
        # Verify environment variable was used
        mock_pool.from_url.assert_called_once()
        call_args = mock_pool.from_url.call_args[0]
        assert call_args[0] == 'redis://env-host:6379/2'
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('escai_framework.storage.redis_manager.ConnectionPool')
    @patch('escai_framework.storage.redis_manager.Redis')
    def test_default_configuration(self, mock_redis, mock_pool):
        """Test default configuration when no environment variables."""
        redis_mgr = RedisManager()
        redis_mgr.initialize()
        
        # Verify default URL was used
        mock_pool.from_url.assert_called_once()
        call_args = mock_pool.from_url.call_args[0]
        assert call_args[0] == 'redis://localhost:6379/0'


class TestRedisManagerEdgeCases:
    """Test Redis manager edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_with_redis_unavailable(self):
        """Test rate limiting when Redis is unavailable."""
        redis_mgr = RedisManager()
        redis_mgr._redis_client = None
        redis_mgr._failover_enabled = True
        
        result = await redis_mgr.rate_limit_check("user_123", 10, 60)
        
        # Should return allowing request when Redis unavailable
        assert result["allowed"] is True
        assert result["current_count"] == 0
        assert result["limit"] == 10
        assert result["window_seconds"] == 60
    
    @pytest.mark.asyncio
    async def test_stream_operations_with_redis_unavailable(self):
        """Test stream operations when Redis is unavailable."""
        redis_mgr = RedisManager()
        redis_mgr._redis_client = None
        redis_mgr._failover_enabled = True
        
        # Stream add should return None
        stream_id = await redis_mgr.stream_add("test_stream", {"data": "test"})
        assert stream_id is None
        
        # Stream read should return empty list
        messages = await redis_mgr.stream_read("test_stream")
        assert messages == []
    
    @pytest.mark.asyncio
    async def test_pubsub_with_redis_unavailable(self):
        """Test pub/sub when Redis is unavailable."""
        redis_mgr = RedisManager()
        redis_mgr._redis_client = None
        redis_mgr._failover_enabled = True
        
        # Publish should return 0
        count = await redis_mgr.publish("test_channel", {"message": "test"})
        assert count == 0
        
        # Subscribe should work as context manager but yield no messages
        async with redis_mgr.subscribe("test_channel") as subscriber:
            # subscriber should be an empty async generator
            assert subscriber is not None
            # Try to iterate - should not yield anything
            message_count = 0
            try:
                async for message in subscriber:
                    message_count += 1
                    if message_count > 0:  # Safety break
                        break
            except (StopAsyncIteration, TypeError):
                # Expected when Redis unavailable
                pass
            assert message_count == 0
    
    @pytest.mark.asyncio
    async def test_health_monitoring_with_redis_unavailable(self):
        """Test health monitoring when Redis is unavailable."""
        redis_mgr = RedisManager()
        redis_mgr._redis_client = None
        redis_mgr._failover_enabled = True
        
        # Get info should return empty dict
        info = await redis_mgr.get_info()
        assert info == {}
        
        # Memory usage should return None
        memory_usage = await redis_mgr.get_memory_usage("test_key")
        assert memory_usage is None