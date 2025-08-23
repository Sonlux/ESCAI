"""
Redis connection management and operations for ESCAI Framework.
Handles caching, session management, real-time data streaming, and rate limiting.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

logger = logging.getLogger(__name__)


class RedisManager:
    """Manages Redis connections and operations with connection pooling and failover."""
    
    def __init__(self):
        self._redis_client: Optional[Redis] = None
        self._connection_pool: Optional[ConnectionPool] = None
        self._initialized = False
        self._redis_url = None
        self._failover_enabled = False
        self._max_retries = 3
        self._retry_delay = 1.0
    
    def initialize(
        self,
        redis_url: Optional[str] = None,
        max_connections: int = 20,
        retry_on_timeout: bool = True,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        health_check_interval: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        failover_enabled: bool = True
    ):
        """Initialize Redis connection with connection pooling and failover."""
        if self._initialized:
            logger.warning("Redis manager already initialized")
            return
        
        # Use environment variable if URL not provided
        if not redis_url:
            redis_url = os.getenv('ESCAI_REDIS_URL', 'redis://localhost:6379/0')
        
        self._redis_url = redis_url
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._failover_enabled = failover_enabled
        
        try:
            # Create connection pool
            self._connection_pool = ConnectionPool.from_url(
                redis_url,
                max_connections=max_connections,
                retry_on_timeout=retry_on_timeout,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                health_check_interval=health_check_interval,
                decode_responses=True
            )
            
            # Create Redis client
            self._redis_client = Redis(connection_pool=self._connection_pool)
            
            self._initialized = True
            logger.info("Redis manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            if not failover_enabled:
                raise
            # Continue without Redis for graceful degradation
            self._redis_client = None
            self._connection_pool = None
    
    async def test_connection(self) -> bool:
        """Test Redis connection."""
        if not self._redis_client:
            return False
        
        try:
            await self._redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            return False
    
    @property
    def available(self) -> bool:
        """Check if Redis is available."""
        return self._redis_client is not None
    
    async def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute Redis operation with retry logic."""
        if not self._redis_client:
            if self._failover_enabled:
                logger.warning("Redis not available, operation skipped")
                return None
            raise RuntimeError("Redis not available")
        
        last_exception = None
        for attempt in range(self._max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < self._max_retries:
                    wait_time = self._retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Redis operation failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Redis operation failed after {self._max_retries + 1} attempts: {e}")
        
        if self._failover_enabled:
            logger.warning("Redis operation failed, continuing without Redis")
            return None
        raise last_exception
    
    # Session Management
    async def set_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> bool:
        """Store session data with TTL."""
        async def _set_session():
            serialized_data = json.dumps(session_data, default=str)
            return await self._redis_client.setex(
                f"session:{session_id}",
                ttl_seconds,
                serialized_data
            )
        
        result = await self._execute_with_retry(_set_session)
        return result is not None
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data."""
        async def _get_session():
            data = await self._redis_client.get(f"session:{session_id}")
            if data:
                return json.loads(data)
            return None
        
        return await self._execute_with_retry(_get_session)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session data."""
        async def _delete_session():
            return await self._redis_client.delete(f"session:{session_id}")
        
        result = await self._execute_with_retry(_delete_session)
        return result is not None and result > 0
    
    async def extend_session(self, session_id: str, ttl_seconds: int = 3600) -> bool:
        """Extend session TTL."""
        async def _extend_session():
            return await self._redis_client.expire(f"session:{session_id}", ttl_seconds)
        
        result = await self._execute_with_retry(_extend_session)
        return result is not None and result
    
    # Caching
    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set cache value with optional TTL."""
        async def _cache_set():
            serialized_value = json.dumps(value, default=str)
            if ttl_seconds:
                return await self._redis_client.setex(f"cache:{key}", ttl_seconds, serialized_value)
            else:
                return await self._redis_client.set(f"cache:{key}", serialized_value)
        
        result = await self._execute_with_retry(_cache_set)
        return result is not None
    
    async def cache_get(self, key: str) -> Any:
        """Get cache value."""
        async def _cache_get():
            data = await self._redis_client.get(f"cache:{key}")
            if data:
                return json.loads(data)
            return None
        
        return await self._execute_with_retry(_cache_get)
    
    async def cache_delete(self, key: str) -> bool:
        """Delete cache value."""
        async def _cache_delete():
            return await self._redis_client.delete(f"cache:{key}")
        
        result = await self._execute_with_retry(_cache_delete)
        return result is not None and result > 0
    
    async def cache_exists(self, key: str) -> bool:
        """Check if cache key exists."""
        async def _cache_exists():
            return await self._redis_client.exists(f"cache:{key}")
        
        result = await self._execute_with_retry(_cache_exists)
        return result is not None and result > 0
    
    # Rate Limiting
    async def rate_limit_check(
        self,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> Dict[str, Any]:
        """Check rate limit using sliding window counter."""
        async def _rate_limit_check():
            key = f"rate_limit:{identifier}"
            current_time = int(datetime.utcnow().timestamp())
            window_start = current_time - window_seconds
            
            # Use pipeline for atomic operations
            pipe = self._redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiry for cleanup
            pipe.expire(key, window_seconds + 1)
            
            results = await pipe.execute()
            current_count = results[1] + 1  # +1 for the request we just added
            
            return {
                'allowed': current_count <= limit,
                'current_count': current_count,
                'limit': limit,
                'window_seconds': window_seconds,
                'reset_time': current_time + window_seconds
            }
        
        result = await self._execute_with_retry(_rate_limit_check)
        if result is None:
            # Fallback when Redis is not available
            return {
                'allowed': True,
                'current_count': 0,
                'limit': limit,
                'window_seconds': window_seconds,
                'reset_time': int(datetime.utcnow().timestamp()) + window_seconds
            }
        return result
    
    # Real-time Data Streaming with Redis Streams
    async def stream_add(
        self,
        stream_name: str,
        data: Dict[str, Any],
        max_length: Optional[int] = 10000
    ) -> Optional[str]:
        """Add data to Redis stream."""
        async def _stream_add():
            # Serialize complex data types
            serialized_data = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    serialized_data[key] = json.dumps(value, default=str)
                else:
                    serialized_data[key] = str(value)
            
            stream_id = await self._redis_client.xadd(
                stream_name,
                serialized_data,
                maxlen=max_length,
                approximate=True
            )
            return stream_id
        
        return await self._execute_with_retry(_stream_add)
    
    async def stream_read(
        self,
        stream_name: str,
        last_id: str = "0",
        count: int = 100,
        block: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Read from Redis stream."""
        async def _stream_read():
            streams = {stream_name: last_id}
            
            if block is not None:
                result = await self._redis_client.xread(streams, count=count, block=block)
            else:
                result = await self._redis_client.xread(streams, count=count)
            
            messages = []
            for stream, stream_messages in result:
                for message_id, fields in stream_messages:
                    # Deserialize data
                    deserialized_fields = {}
                    for key, value in fields.items():
                        try:
                            # Try to parse as JSON first
                            deserialized_fields[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            # Keep as string if not JSON
                            deserialized_fields[key] = value
                    
                    messages.append({
                        'id': message_id,
                        'stream': stream,
                        'data': deserialized_fields
                    })
            
            return messages
        
        result = await self._execute_with_retry(_stream_read)
        return result or []
    
    async def stream_create_group(
        self,
        stream_name: str,
        group_name: str,
        start_id: str = "0"
    ) -> bool:
        """Create consumer group for stream."""
        async def _stream_create_group():
            try:
                await self._redis_client.xgroup_create(
                    stream_name,
                    group_name,
                    start_id,
                    mkstream=True
                )
                return True
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # Group already exists
                    return True
                raise
        
        result = await self._execute_with_retry(_stream_create_group)
        return result is not None and result
    
    async def stream_read_group(
        self,
        group_name: str,
        consumer_name: str,
        stream_name: str,
        count: int = 100,
        block: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Read from stream as part of consumer group."""
        async def _stream_read_group():
            streams = {stream_name: ">"}
            
            if block is not None:
                result = await self._redis_client.xreadgroup(
                    group_name,
                    consumer_name,
                    streams,
                    count=count,
                    block=block
                )
            else:
                result = await self._redis_client.xreadgroup(
                    group_name,
                    consumer_name,
                    streams,
                    count=count
                )
            
            messages = []
            for stream, stream_messages in result:
                for message_id, fields in stream_messages:
                    # Deserialize data
                    deserialized_fields = {}
                    for key, value in fields.items():
                        try:
                            deserialized_fields[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            deserialized_fields[key] = value
                    
                    messages.append({
                        'id': message_id,
                        'stream': stream,
                        'data': deserialized_fields
                    })
            
            return messages
        
        result = await self._execute_with_retry(_stream_read_group)
        return result or []
    
    async def stream_ack(
        self,
        stream_name: str,
        group_name: str,
        message_id: str
    ) -> bool:
        """Acknowledge message processing."""
        async def _stream_ack():
            return await self._redis_client.xack(stream_name, group_name, message_id)
        
        result = await self._execute_with_retry(_stream_ack)
        return result is not None and result > 0
    
    # Pub/Sub functionality
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        async def _publish():
            serialized_message = json.dumps(message, default=str)
            return await self._redis_client.publish(channel, serialized_message)
        
        result = await self._execute_with_retry(_publish)
        return result or 0
    
    @asynccontextmanager
    async def subscribe(self, *channels: str) -> AsyncGenerator[Any, None]:
        """Subscribe to channels and yield messages."""
        if not self._redis_client:
            logger.warning("Redis not available for subscription")
            # Create empty async generator for context manager compatibility
            async def empty_generator():
                return
                yield  # This will never execute but makes it a generator
            
            yield empty_generator()
            return
        
        pubsub = self._redis_client.pubsub()
        try:
            await pubsub.subscribe(*channels)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        # Try to deserialize JSON
                        data = json.loads(message['data'])
                        yield {
                            'channel': message['channel'],
                            'data': data
                        }
                    except (json.JSONDecodeError, TypeError):
                        # Keep as string if not JSON
                        yield {
                            'channel': message['channel'],
                            'data': message['data']
                        }
        finally:
            await pubsub.unsubscribe(*channels)
            await pubsub.close()
    
    # Temporary storage
    async def temp_store(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 300
    ) -> bool:
        """Store temporary data with TTL."""
        return await self.cache_set(f"temp:{key}", value, ttl_seconds)
    
    async def temp_get(self, key: str) -> Any:
        """Get temporary data."""
        return await self.cache_get(f"temp:{key}")
    
    async def temp_delete(self, key: str) -> bool:
        """Delete temporary data."""
        return await self.cache_delete(f"temp:{key}")
    
    # Health and monitoring
    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        async def _get_info():
            return await self._redis_client.info()
        
        result = await self._execute_with_retry(_get_info)
        return result or {}
    
    async def get_memory_usage(self, key: str) -> Optional[int]:
        """Get memory usage of a key."""
        async def _get_memory_usage():
            return await self._redis_client.memory_usage(key)
        
        return await self._execute_with_retry(_get_memory_usage)
    
    async def close(self):
        """Close Redis connections."""
        if self._redis_client:
            await self._redis_client.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
        self._initialized = False
        logger.info("Redis connections closed")


# Global Redis manager instance
redis_manager = RedisManager()