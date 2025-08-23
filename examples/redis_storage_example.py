"""
Example demonstrating Redis functionality in ESCAI Framework.
Shows caching, session management, rate limiting, and real-time streaming.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

from escai_framework.storage.database import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_session_management():
    """Demonstrate Redis session management."""
    print("\n=== Session Management Demo ===")
    
    redis_mgr = db_manager.redis_manager
    
    # Create session
    session_id = "user_123_session"
    session_data = {
        "user_id": "user_123",
        "username": "john_doe",
        "login_time": datetime.utcnow().isoformat(),
        "permissions": ["read", "write"],
        "preferences": {
            "theme": "dark",
            "language": "en"
        }
    }
    
    success = await redis_mgr.set_session(session_id, session_data, ttl_seconds=3600)
    print(f"Session created: {success}")
    
    # Retrieve session
    retrieved_session = await redis_mgr.get_session(session_id)
    print(f"Retrieved session: {retrieved_session}")
    
    # Extend session
    extended = await redis_mgr.extend_session(session_id, ttl_seconds=7200)
    print(f"Session extended: {extended}")
    
    # Delete session
    deleted = await redis_mgr.delete_session(session_id)
    print(f"Session deleted: {deleted}")


async def demonstrate_caching():
    """Demonstrate Redis caching functionality."""
    print("\n=== Caching Demo ===")
    
    redis_mgr = db_manager.redis_manager
    
    # Cache some analysis results
    analysis_key = "agent_123_behavioral_analysis"
    analysis_result = {
        "agent_id": "agent_123",
        "patterns_found": 15,
        "success_rate": 0.87,
        "common_patterns": [
            "sequential_reasoning",
            "error_recovery",
            "goal_refinement"
        ],
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "confidence_score": 0.92
    }
    
    # Set cache with TTL
    cached = await redis_mgr.cache_set(analysis_key, analysis_result, ttl_seconds=1800)
    print(f"Analysis cached: {cached}")
    
    # Check if exists
    exists = await redis_mgr.cache_exists(analysis_key)
    print(f"Cache exists: {exists}")
    
    # Retrieve from cache
    cached_result = await redis_mgr.cache_get(analysis_key)
    print(f"Retrieved from cache: {cached_result}")
    
    # Cache prediction results
    prediction_key = "agent_123_performance_prediction"
    prediction_data = {
        "predicted_success_rate": 0.91,
        "risk_factors": ["high_complexity", "time_pressure"],
        "recommended_interventions": ["add_checkpoints", "reduce_scope"],
        "confidence_interval": [0.85, 0.97]
    }
    
    await redis_mgr.cache_set(prediction_key, prediction_data, ttl_seconds=600)
    print(f"Prediction cached for 10 minutes")


async def demonstrate_rate_limiting():
    """Demonstrate Redis rate limiting."""
    print("\n=== Rate Limiting Demo ===")
    
    redis_mgr = db_manager.redis_manager
    
    # Simulate API requests from a user
    user_id = "api_user_456"
    limit = 5  # 5 requests per minute
    window = 60  # 60 seconds
    
    print(f"Rate limit: {limit} requests per {window} seconds")
    
    for i in range(7):  # Try 7 requests (should exceed limit)
        result = await redis_mgr.rate_limit_check(user_id, limit, window)
        
        status = "ALLOWED" if result['allowed'] else "BLOCKED"
        print(f"Request {i+1}: {status} - Count: {result['current_count']}/{result['limit']}")
        
        if not result['allowed']:
            print(f"Rate limit exceeded. Reset at: {result['reset_time']}")
        
        await asyncio.sleep(0.1)  # Small delay between requests


async def demonstrate_streaming():
    """Demonstrate Redis Streams for real-time data."""
    print("\n=== Real-time Streaming Demo ===")
    
    redis_mgr = db_manager.redis_manager
    stream_name = "agent_events"
    
    # Create consumer group
    group_created = await redis_mgr.stream_create_group(stream_name, "monitoring_group", "0")
    print(f"Consumer group created: {group_created}")
    
    # Add some events to the stream
    events = [
        {
            "event_type": "epistemic_update",
            "agent_id": "agent_789",
            "timestamp": datetime.utcnow().isoformat(),
            "belief_change": {
                "old_confidence": 0.7,
                "new_confidence": 0.85,
                "reason": "new_evidence_found"
            }
        },
        {
            "event_type": "pattern_detected",
            "agent_id": "agent_789",
            "timestamp": datetime.utcnow().isoformat(),
            "pattern": {
                "name": "recursive_problem_solving",
                "frequency": 3,
                "success_rate": 0.9
            }
        },
        {
            "event_type": "prediction_alert",
            "agent_id": "agent_789",
            "timestamp": datetime.utcnow().isoformat(),
            "alert": {
                "type": "high_failure_risk",
                "probability": 0.75,
                "recommended_action": "intervention_required"
            }
        }
    ]
    
    # Add events to stream
    for event in events:
        stream_id = await redis_mgr.stream_add(stream_name, event)
        print(f"Added event to stream: {stream_id}")
    
    # Read from stream
    print("\nReading from stream:")
    messages = await redis_mgr.stream_read(stream_name, "0", count=10)
    for msg in messages:
        print(f"Message ID: {msg['id']}")
        print(f"Event Type: {msg['data']['event_type']}")
        print(f"Agent ID: {msg['data']['agent_id']}")
        print("---")
    
    # Read as consumer group
    print("\nReading as consumer group:")
    group_messages = await redis_mgr.stream_read_group(
        "monitoring_group",
        "consumer_1",
        stream_name,
        count=5
    )
    
    for msg in group_messages:
        print(f"Processing message: {msg['id']}")
        print(f"Event: {msg['data']['event_type']}")
        
        # Acknowledge message processing
        acked = await redis_mgr.stream_ack(stream_name, "monitoring_group", msg['id'])
        print(f"Message acknowledged: {acked}")


async def demonstrate_pubsub():
    """Demonstrate Redis Pub/Sub functionality."""
    print("\n=== Pub/Sub Demo ===")
    
    redis_mgr = db_manager.redis_manager
    
    # Start subscriber in background task
    async def subscriber_task():
        print("Starting subscriber...")
        async with redis_mgr.subscribe("agent_alerts", "system_notifications") as subscriber:
            async for message in subscriber:
                print(f"Received on {message['channel']}: {message['data']}")
    
    # Start subscriber
    subscriber = asyncio.create_task(subscriber_task())
    
    # Give subscriber time to start
    await asyncio.sleep(0.1)
    
    # Publish some messages
    alerts = [
        {
            "type": "performance_degradation",
            "agent_id": "agent_456",
            "severity": "medium",
            "message": "Agent response time increased by 40%"
        },
        {
            "type": "anomaly_detected",
            "agent_id": "agent_789",
            "severity": "high",
            "message": "Unusual behavioral pattern detected"
        }
    ]
    
    for alert in alerts:
        subscribers_count = await redis_mgr.publish("agent_alerts", alert)
        print(f"Published alert to {subscribers_count} subscribers")
        await asyncio.sleep(0.1)
    
    # Publish system notification
    notification = {
        "type": "system_maintenance",
        "scheduled_time": "2024-01-15T02:00:00Z",
        "duration": "30 minutes"
    }
    
    await redis_mgr.publish("system_notifications", notification)
    print("Published system notification")
    
    # Let subscriber process messages
    await asyncio.sleep(0.5)
    
    # Cancel subscriber
    subscriber.cancel()
    try:
        await subscriber
    except asyncio.CancelledError:
        pass


async def demonstrate_temporary_storage():
    """Demonstrate temporary storage functionality."""
    print("\n=== Temporary Storage Demo ===")
    
    redis_mgr = db_manager.redis_manager
    
    # Store temporary computation results
    temp_key = "computation_result_xyz"
    temp_data = {
        "computation_id": "xyz",
        "intermediate_results": [0.1, 0.3, 0.7, 0.9],
        "partial_analysis": {
            "patterns_identified": 3,
            "confidence_scores": [0.8, 0.6, 0.9]
        },
        "expires_at": datetime.utcnow().isoformat()
    }
    
    # Store for 5 minutes
    stored = await redis_mgr.temp_store(temp_key, temp_data, ttl_seconds=300)
    print(f"Temporary data stored: {stored}")
    
    # Retrieve temporary data
    retrieved = await redis_mgr.temp_get(temp_key)
    print(f"Retrieved temporary data: {retrieved is not None}")
    
    if retrieved:
        print(f"Computation ID: {retrieved['computation_id']}")
        print(f"Patterns identified: {retrieved['partial_analysis']['patterns_identified']}")


async def demonstrate_health_monitoring():
    """Demonstrate Redis health monitoring."""
    print("\n=== Health Monitoring Demo ===")
    
    redis_mgr = db_manager.redis_manager
    
    # Test connection
    connected = await redis_mgr.test_connection()
    print(f"Redis connection healthy: {connected}")
    
    # Get server info
    info = await redis_mgr.get_info()
    if info:
        print(f"Redis version: {info.get('redis_version', 'Unknown')}")
        print(f"Connected clients: {info.get('connected_clients', 'Unknown')}")
        print(f"Used memory: {info.get('used_memory_human', 'Unknown')}")
        print(f"Total commands processed: {info.get('total_commands_processed', 'Unknown')}")
    
    # Check memory usage of a key
    test_key = "cache:test_memory_usage"
    await redis_mgr.cache_set("test_memory_usage", {"test": "data"}, ttl_seconds=60)
    
    memory_usage = await redis_mgr.get_memory_usage(test_key)
    if memory_usage:
        print(f"Memory usage of test key: {memory_usage} bytes")


async def main():
    """Main demonstration function."""
    print("ESCAI Framework Redis Storage Example")
    print("=====================================")
    
    # Initialize database manager
    db_manager.initialize(
        redis_url="redis://localhost:6379/0"  # Use default Redis URL
    )
    
    # Test Redis connection
    if not await db_manager.test_redis_connection():
        print("Redis connection failed. Please ensure Redis is running on localhost:6379")
        return
    
    print("Redis connection successful!")
    
    try:
        # Run all demonstrations
        await demonstrate_session_management()
        await demonstrate_caching()
        await demonstrate_rate_limiting()
        await demonstrate_streaming()
        await demonstrate_pubsub()
        await demonstrate_temporary_storage()
        await demonstrate_health_monitoring()
        
        print("\n=== Demo Complete ===")
        print("All Redis functionality demonstrated successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Clean up
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())