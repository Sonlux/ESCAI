"""
Integration tests for WebSocket real-time interface.
"""

import asyncio
import json
import pytest
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, patch
import jwt

from fastapi.testclient import TestClient
from fastapi import FastAPI

from escai_framework.api.websocket import (
    manager, websocket_router, broadcast_epistemic_update,
    broadcast_pattern_alert, broadcast_prediction_update,
    broadcast_system_alert, send_user_notification
)
from escai_framework.api.auth import SECRET_KEY, ALGORITHM
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState
from escai_framework.models.behavioral_pattern import BehavioralPattern, ExecutionSequence
from escai_framework.models.prediction_result import PredictionResult


class WebSocketTestClient:
    """Test client for WebSocket connections."""
    
    def __init__(self, uri: str, token: str):
        self.uri = uri
        self.token = token
        self.websocket = None
        self.messages = []
        self.connected = False
    
    async def connect(self):
        """Connect to WebSocket."""
        try:
            self.websocket = await websockets.connect(f"{self.uri}?token={self.token}")
            self.connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket."""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
    
    async def receive_message(self, timeout: float = 1.0):
        """Receive message from WebSocket."""
        if self.websocket:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
                parsed_message = json.loads(message)
                self.messages.append(parsed_message)
                return parsed_message
            except asyncio.TimeoutError:
                return None
        return None
    
    async def receive_messages(self, count: int, timeout: float = 5.0):
        """Receive multiple messages."""
        messages = []
        start_time = asyncio.get_event_loop().time()
        
        while len(messages) < count:
            if asyncio.get_event_loop().time() - start_time > timeout:
                break
            
            message = await self.receive_message(0.1)
            if message:
                messages.append(message)
        
        return messages


def create_test_token(user_id: str, roles: List[str] = None, permissions: List[str] = None) -> str:
    """Create test JWT token."""
    payload = {
        "sub": user_id,
        "roles": roles or ["user"],
        "permissions": permissions or ["websocket:connect", "websocket:subscribe"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_admin_token(user_id: str) -> str:
    """Create admin JWT token."""
    return create_test_token(
        user_id,
        roles=["admin"],
        permissions=["websocket:connect", "websocket:subscribe", "websocket:admin"]
    )


@pytest.fixture
async def websocket_app():
    """Create test FastAPI app with WebSocket routes."""
    app = FastAPI()
    app.include_router(websocket_router, prefix="/ws")
    return app


@pytest.fixture
async def clean_manager():
    """Clean connection manager before each test."""
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
    
    yield manager
    
    # Cleanup after test
    manager.connections.clear()
    manager.user_connections.clear()
    manager.agent_subscribers.clear()
    if manager.heartbeat_manager.running:
        await manager.heartbeat_manager.stop()


class TestWebSocketConnection:
    """Test WebSocket connection management."""
    
    @pytest.mark.asyncio
    async def test_successful_connection(self, clean_manager):
        """Test successful WebSocket connection."""
        token = create_test_token("test_user")
        
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        
        # Test connection
        success = await manager.connect(mock_websocket, "conn_1", "test_user")
        
        assert success is True
        assert "conn_1" in manager.connections
        assert "test_user" in manager.user_connections
        assert manager.total_connections == 1
        mock_websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connection_limit(self, clean_manager):
        """Test connection limit enforcement."""
        # Set low connection limit for testing
        original_limit = manager.max_connections
        manager.max_connections = 2
        
        try:
            token = create_test_token("test_user")
            
            # Create mock WebSockets
            mock_ws1 = AsyncMock()
            mock_ws2 = AsyncMock()
            mock_ws3 = AsyncMock()
            
            # Connect up to limit
            success1 = await manager.connect(mock_ws1, "conn_1", "user_1")
            success2 = await manager.connect(mock_ws2, "conn_2", "user_2")
            
            assert success1 is True
            assert success2 is True
            assert len(manager.connections) == 2
            
            # Try to exceed limit
            success3 = await manager.connect(mock_ws3, "conn_3", "user_3")
            
            assert success3 is False
            assert len(manager.connections) == 2
            mock_ws3.close.assert_called_once()
        
        finally:
            manager.max_connections = original_limit
    
    @pytest.mark.asyncio
    async def test_connection_disconnect(self, clean_manager):
        """Test WebSocket disconnection."""
        mock_websocket = AsyncMock()
        
        # Connect
        await manager.connect(mock_websocket, "conn_1", "test_user")
        assert len(manager.connections) == 1
        
        # Disconnect
        await manager.disconnect("conn_1")
        assert len(manager.connections) == 0
        assert len(manager.user_connections) == 0
    
    @pytest.mark.asyncio
    async def test_heartbeat_management(self, clean_manager):
        """Test heartbeat functionality."""
        mock_websocket = AsyncMock()
        
        # Connect - should start heartbeat
        await manager.connect(mock_websocket, "conn_1", "test_user")
        
        # Wait a bit for heartbeat to start
        await asyncio.sleep(0.1)
        assert manager.heartbeat_manager.running is True
        
        # Disconnect - should stop heartbeat
        await manager.disconnect("conn_1")
        await asyncio.sleep(0.1)
        assert manager.heartbeat_manager.running is False


class TestWebSocketSubscriptions:
    """Test WebSocket subscription management."""
    
    @pytest.mark.asyncio
    async def test_add_subscription(self, clean_manager):
        """Test adding subscriptions."""
        mock_websocket = AsyncMock()
        await manager.connect(mock_websocket, "conn_1", "test_user")
        
        from escai_framework.api.websocket import SubscriptionRequest
        
        # Add subscription
        subscription = SubscriptionRequest(
            type="epistemic_updates",
            agent_id="agent_1",
            filters={"confidence": {"min": 0.8}}
        )
        
        success = manager.add_subscription("conn_1", subscription)
        assert success is True
        
        # Check subscription was added
        conn_info = manager.connections["conn_1"]
        assert len(conn_info.subscriptions) == 1
        assert conn_info.subscriptions[0].type == "epistemic_updates"
        assert conn_info.subscriptions[0].agent_id == "agent_1"
        
        # Check agent subscribers tracking
        assert "agent_1" in manager.agent_subscribers
        assert "conn_1" in manager.agent_subscribers["agent_1"]
    
    @pytest.mark.asyncio
    async def test_remove_subscription(self, clean_manager):
        """Test removing subscriptions."""
        mock_websocket = AsyncMock()
        await manager.connect(mock_websocket, "conn_1", "test_user")
        
        from escai_framework.api.websocket import SubscriptionRequest
        
        # Add subscription
        subscription = SubscriptionRequest(
            type="epistemic_updates",
            agent_id="agent_1"
        )
        manager.add_subscription("conn_1", subscription)
        
        # Remove subscription
        success = manager.remove_subscription("conn_1", "epistemic_updates", "agent_1")
        assert success is True
        
        # Check subscription was removed
        conn_info = manager.connections["conn_1"]
        assert len(conn_info.subscriptions) == 0
        
        # Check agent subscribers tracking
        assert "agent_1" not in manager.agent_subscribers
    
    @pytest.mark.asyncio
    async def test_duplicate_subscription(self, clean_manager):
        """Test handling duplicate subscriptions."""
        mock_websocket = AsyncMock()
        await manager.connect(mock_websocket, "conn_1", "test_user")
        
        from escai_framework.api.websocket import SubscriptionRequest
        
        subscription = SubscriptionRequest(
            type="epistemic_updates",
            agent_id="agent_1"
        )
        
        # Add subscription twice
        success1 = manager.add_subscription("conn_1", subscription)
        success2 = manager.add_subscription("conn_1", subscription)
        
        assert success1 is True
        assert success2 is False  # Duplicate should be rejected
        
        conn_info = manager.connections["conn_1"]
        assert len(conn_info.subscriptions) == 1


class TestWebSocketBroadcasting:
    """Test WebSocket broadcasting functionality."""
    
    @pytest.mark.asyncio
    async def test_broadcast_epistemic_update(self, clean_manager):
        """Test broadcasting epistemic state updates."""
        # Setup connections with subscriptions
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        await manager.connect(mock_ws1, "conn_1", "user_1")
        await manager.connect(mock_ws2, "conn_2", "user_2")
        
        from escai_framework.api.websocket import SubscriptionRequest
        
        # Subscribe to epistemic updates
        sub1 = SubscriptionRequest(type="epistemic_updates", agent_id="agent_1")
        sub2 = SubscriptionRequest(type="epistemic_updates", agent_id="agent_1")
        
        manager.add_subscription("conn_1", sub1)
        manager.add_subscription("conn_2", sub2)
        
        # Create test epistemic state
        from escai_framework.models.epistemic_state import BeliefType, GoalStatus
        
        epistemic_state = EpistemicState(
            agent_id="agent_1",
            timestamp=datetime.utcnow(),
            belief_states=[BeliefState(content="test belief", belief_type=BeliefType.FACTUAL, confidence=0.9)],
            knowledge_state=KnowledgeState(facts=["fact1"], concepts={"concept1": "value1"}),
            goal_states=[GoalState(description="test goal", status=GoalStatus.ACTIVE, priority=5, progress=0.5, sub_goals=["sub1"])],
            confidence_level=0.85,
            uncertainty_score=0.15,
            decision_context={"context": "test"}
        )
        
        # Broadcast update
        sent_count = await broadcast_epistemic_update("agent_1", epistemic_state)
        
        assert sent_count == 2
        assert mock_ws1.send_text.call_count == 1
        assert mock_ws2.send_text.call_count == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_with_filters(self, clean_manager):
        """Test broadcasting with subscription filters."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        await manager.connect(mock_ws1, "conn_1", "user_1")
        await manager.connect(mock_ws2, "conn_2", "user_2")
        
        from escai_framework.api.websocket import SubscriptionRequest
        
        # Subscribe with different filters
        sub1 = SubscriptionRequest(
            type="epistemic_updates",
            agent_id="agent_1",
            filters={"confidence_level": {"min": 0.8}}
        )
        sub2 = SubscriptionRequest(
            type="epistemic_updates",
            agent_id="agent_1",
            filters={"confidence_level": {"min": 0.9}}
        )
        
        manager.add_subscription("conn_1", sub1)
        manager.add_subscription("conn_2", sub2)
        
        # Create epistemic state with confidence 0.85
        from escai_framework.models.epistemic_state import BeliefType, GoalStatus
        
        epistemic_state = EpistemicState(
            agent_id="agent_1",
            timestamp=datetime.utcnow(),
            belief_states=[],
            knowledge_state=KnowledgeState(facts=[], concepts={}),
            goal_states=[GoalState(description="test", status=GoalStatus.ACTIVE, priority=5, progress=0.5, sub_goals=[])],
            confidence_level=0.85,
            uncertainty_score=0.15,
            decision_context={}
        )
        
        # Broadcast with filters
        filters = {"confidence_level": 0.85}
        sent_count = await broadcast_epistemic_update("agent_1", epistemic_state, filters)
        
        # Only conn_1 should receive (min 0.8), conn_2 requires min 0.9
        assert sent_count == 1
        assert mock_ws1.send_text.call_count == 1
        assert mock_ws2.send_text.call_count == 0
    
    @pytest.mark.asyncio
    async def test_broadcast_system_alert(self, clean_manager):
        """Test broadcasting system alerts."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        await manager.connect(mock_ws1, "conn_1", "user_1")
        await manager.connect(mock_ws2, "conn_2", "user_2")
        
        from escai_framework.api.websocket import SubscriptionRequest
        
        # Subscribe to system alerts
        sub1 = SubscriptionRequest(type="system_alerts")
        sub2 = SubscriptionRequest(type="epistemic_updates", agent_id="agent_1")  # Different type
        
        manager.add_subscription("conn_1", sub1)
        manager.add_subscription("conn_2", sub2)
        
        # Broadcast system alert
        sent_count = await broadcast_system_alert("maintenance", "System maintenance scheduled", "warning")
        
        # Only conn_1 should receive (subscribed to system_alerts)
        assert sent_count == 1
        assert mock_ws1.send_text.call_count == 1
        assert mock_ws2.send_text.call_count == 0


class TestWebSocketMessageHandling:
    """Test WebSocket message handling."""
    
    @pytest.mark.asyncio
    async def test_ping_pong(self, clean_manager):
        """Test ping/pong mechanism."""
        mock_websocket = AsyncMock()
        await manager.connect(mock_websocket, "conn_1", "test_user")
        
        from escai_framework.api.websocket import handle_websocket_message
        
        user_info = {"user_id": "test_user", "roles": ["user"], "permissions": ["websocket:connect"]}
        
        # Send ping
        await handle_websocket_message("conn_1", user_info, {"type": "ping"})
        
        # Should respond with pong
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_subscription_message(self, clean_manager):
        """Test subscription message handling."""
        mock_websocket = AsyncMock()
        await manager.connect(mock_websocket, "conn_1", "test_user")
        
        from escai_framework.api.websocket import handle_websocket_message
        
        user_info = {
            "user_id": "test_user",
            "roles": ["user"],
            "permissions": ["websocket:connect", "websocket:subscribe", "agent:agent_1:monitor"]
        }
        
        # Send subscription request
        message = {
            "type": "subscribe",
            "data": {
                "type": "epistemic_updates",
                "agent_id": "agent_1",
                "filters": {"confidence": {"min": 0.8}}
            }
        }
        
        await handle_websocket_message("conn_1", user_info, message)
        
        # Should confirm subscription
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "subscription_confirmed"
        assert response["data"]["subscription_type"] == "epistemic_updates"
        assert response["data"]["agent_id"] == "agent_1"
    
    @pytest.mark.asyncio
    async def test_permission_denied(self, clean_manager):
        """Test permission denied handling."""
        mock_websocket = AsyncMock()
        await manager.connect(mock_websocket, "conn_1", "test_user")
        
        from escai_framework.api.websocket import handle_websocket_message
        
        user_info = {
            "user_id": "test_user",
            "roles": ["user"],
            "permissions": ["websocket:connect"]  # Missing subscribe permission
        }
        
        # Send subscription request
        message = {
            "type": "subscribe",
            "data": {
                "type": "epistemic_updates",
                "agent_id": "agent_1"
            }
        }
        
        await handle_websocket_message("conn_1", user_info, message)
        
        # Should return permission denied error
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "error"
        assert response["code"] == "PERMISSION_DENIED"


class TestWebSocketLoadTesting:
    """Test WebSocket under load conditions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_connections(self, clean_manager):
        """Test handling multiple concurrent connections."""
        connections = []
        
        # Create multiple connections
        for i in range(10):
            mock_ws = AsyncMock()
            success = await manager.connect(mock_ws, f"conn_{i}", f"user_{i}")
            assert success is True
            connections.append(mock_ws)
        
        assert len(manager.connections) == 10
        assert len(manager.user_connections) == 10
        
        # Test broadcasting to all
        from escai_framework.api.websocket import SubscriptionRequest
        
        # Subscribe all to system alerts
        for i in range(10):
            sub = SubscriptionRequest(type="system_alerts")
            manager.add_subscription(f"conn_{i}", sub)
        
        # Broadcast system alert
        sent_count = await broadcast_system_alert("test", "Test message")
        assert sent_count == 10
        
        # Verify all connections received message
        for mock_ws in connections:
            mock_ws.send_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_message_throughput(self, clean_manager):
        """Test message throughput."""
        mock_websocket = AsyncMock()
        await manager.connect(mock_websocket, "conn_1", "test_user")
        
        from escai_framework.api.websocket import SubscriptionRequest
        
        # Subscribe to system alerts
        sub = SubscriptionRequest(type="system_alerts")
        manager.add_subscription("conn_1", sub)
        
        # Send many messages rapidly
        message_count = 100
        for i in range(message_count):
            await broadcast_system_alert("test", f"Message {i}")
        
        # Verify all messages were sent
        assert mock_websocket.send_text.call_count == message_count
        assert manager.total_messages_sent == message_count
    
    @pytest.mark.asyncio
    async def test_connection_recovery(self, clean_manager):
        """Test connection recovery after failures."""
        mock_websocket = AsyncMock()
        
        # Simulate send failure
        mock_websocket.send_text.side_effect = Exception("Connection lost")
        
        await manager.connect(mock_websocket, "conn_1", "test_user")
        
        from escai_framework.api.websocket import SubscriptionRequest
        
        # Subscribe to system alerts
        sub = SubscriptionRequest(type="system_alerts")
        manager.add_subscription("conn_1", sub)
        
        # Try to send message (should fail and remove connection)
        sent_count = await broadcast_system_alert("test", "Test message")
        
        assert sent_count == 0
        assert len(manager.connections) == 0  # Connection should be removed


class TestWebSocketSecurity:
    """Test WebSocket security features."""
    
    @pytest.mark.asyncio
    async def test_token_validation(self, clean_manager):
        """Test JWT token validation."""
        from escai_framework.api.websocket import authenticate_websocket
        
        # Valid token
        valid_token = create_test_token("test_user")
        user_info = await authenticate_websocket(valid_token)
        assert user_info is not None
        assert user_info["user_id"] == "test_user"
        
        # Invalid token
        invalid_token = "invalid.token.here"
        user_info = await authenticate_websocket(invalid_token)
        assert user_info is None
        
        # Expired token
        expired_payload = {
            "sub": "test_user",
            "exp": datetime.utcnow() - timedelta(hours=1)
        }
        expired_token = jwt.encode(expired_payload, SECRET_KEY, algorithm=ALGORITHM)
        user_info = await authenticate_websocket(expired_token)
        assert user_info is None
    
    @pytest.mark.asyncio
    async def test_permission_checking(self, clean_manager):
        """Test permission checking."""
        from escai_framework.api.websocket import check_websocket_permission
        
        # Admin user
        admin_info = {"user_id": "admin", "roles": ["admin"], "permissions": []}
        assert check_websocket_permission(admin_info, "any:permission") is True
        
        # User with specific permission
        user_info = {"user_id": "user", "roles": ["user"], "permissions": ["websocket:connect"]}
        assert check_websocket_permission(user_info, "websocket:connect") is True
        assert check_websocket_permission(user_info, "websocket:admin") is False
        
        # User without permission
        limited_user = {"user_id": "limited", "roles": ["user"], "permissions": []}
        assert check_websocket_permission(limited_user, "websocket:connect") is False
    
    @pytest.mark.asyncio
    async def test_agent_access_control(self, clean_manager):
        """Test agent-specific access control."""
        mock_websocket = AsyncMock()
        await manager.connect(mock_websocket, "conn_1", "test_user")
        
        from escai_framework.api.websocket import handle_websocket_message
        
        # User with access to specific agent
        user_info = {
            "user_id": "test_user",
            "roles": ["user"],
            "permissions": ["websocket:connect", "websocket:subscribe", "agent:agent_1:monitor"]
        }
        
        # Should allow subscription to agent_1
        message = {
            "type": "subscribe",
            "data": {
                "type": "epistemic_updates",
                "agent_id": "agent_1"
            }
        }
        
        await handle_websocket_message("conn_1", user_info, message)
        
        # Should confirm subscription
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "subscription_confirmed"
        
        # Reset mock
        mock_websocket.reset_mock()
        
        # Try to subscribe to agent_2 (no permission)
        message["data"]["agent_id"] = "agent_2"
        await handle_websocket_message("conn_1", user_info, message)
        
        # Should return access denied error
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        assert response["type"] == "error"
        assert response["code"] == "AGENT_ACCESS_DENIED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])