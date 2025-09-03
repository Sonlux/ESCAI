"""
WebSocket endpoints for real-time communication in ESCAI Framework API.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Union
from uuid import uuid4
import time
import weakref

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status, Depends, Query
from pydantic import BaseModel, ValidationError
import jwt

from .auth import auth_manager, SECRET_KEY, ALGORITHM, get_current_user
from ..models.epistemic_state import EpistemicState
from ..models.behavioral_pattern import BehavioralPattern
from ..models.prediction_result import PredictionResult
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Router
websocket_router = APIRouter()

# WebSocket message models
class WebSocketMessage(BaseModel):
    """Base WebSocket message."""
    type: str
    timestamp: datetime
    data: Dict[str, Any]

class SubscriptionRequest(BaseModel):
    """WebSocket subscription request."""
    type: str  # epistemic_updates, pattern_alerts, prediction_updates, system_alerts
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

class ConnectionInfo:
    """Information about a WebSocket connection."""
    
    def __init__(self, connection_id: str, user_id: str, websocket: WebSocket, 
                 connected_at: datetime, last_ping: datetime, subscriptions: List[SubscriptionRequest] = None):
        self.connection_id = connection_id
        self.user_id = user_id
        self.websocket = websocket
        self.connected_at = connected_at
        self.last_ping = last_ping
        self.subscriptions = subscriptions or []
        self.is_alive = True

class HeartbeatManager:
    """Manages heartbeat for WebSocket connections."""
    
    def __init__(self, ping_interval: int = 30, pong_timeout: int = 10):
        self.ping_interval = ping_interval  # seconds
        self.pong_timeout = pong_timeout    # seconds
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    async def start(self, connection_manager):
        """Start heartbeat monitoring."""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._heartbeat_loop(connection_manager))
    
    async def stop(self):
        """Stop heartbeat monitoring."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
    
    async def _heartbeat_loop(self, connection_manager):
        """Main heartbeat loop."""
        while self.running:
            try:
                await asyncio.sleep(self.ping_interval)
                await self._send_pings(connection_manager)
                await asyncio.sleep(self.pong_timeout)
                await self._check_pongs(connection_manager)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
    
    async def _send_pings(self, connection_manager):
        """Send ping messages to all connections."""
        current_time = datetime.utcnow()
        ping_message = json.dumps({
            "type": "ping",
            "timestamp": current_time.isoformat()
        })
        
        for connection_id, conn_info in connection_manager.connections.items():
            if conn_info.is_alive:
                try:
                    await conn_info.websocket.send_text(ping_message)
                except Exception as e:
                    logger.warning(f"Failed to send ping to {connection_id}: {e}")
                    conn_info.is_alive = False
    
    async def _check_pongs(self, connection_manager):
        """Check for connections that didn't respond to ping."""
        current_time = datetime.utcnow()
        timeout_threshold = current_time - timedelta(seconds=self.pong_timeout)
        
        dead_connections = []
        for connection_id, conn_info in connection_manager.connections.items():
            if conn_info.is_alive and conn_info.last_ping < timeout_threshold:
                logger.warning(f"Connection {connection_id} failed heartbeat check")
                dead_connections.append(connection_id)
        
        # Remove dead connections
        for connection_id in dead_connections:
            await connection_manager._remove_dead_connection(connection_id)

class ConnectionManager:
    """Manages WebSocket connections with enhanced features."""
    
    def __init__(self, max_connections: int = 100):
        self.connections: Dict[str, ConnectionInfo] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.agent_subscribers: Dict[str, Set[str]] = {}  # agent_id -> connection_ids
        self.max_connections = max_connections
        self.heartbeat_manager = HeartbeatManager()
        self._connection_lock = asyncio.Lock()
        
        # Statistics
        self.total_connections = 0
        self.total_messages_sent = 0
        self.total_messages_received = 0
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str) -> bool:
        """Accept WebSocket connection with connection limits."""
        async with self._connection_lock:
            # Check connection limits
            if len(self.connections) >= self.max_connections:
                logger.warning(f"Connection limit reached ({self.max_connections}), rejecting connection")
                await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
                return False
            
            await websocket.accept()
            
            # Create connection info
            current_time = datetime.utcnow()
            conn_info = ConnectionInfo(
                connection_id=connection_id,
                user_id=user_id,
                websocket=websocket,
                connected_at=current_time,
                last_ping=current_time,
                subscriptions=[]
            )
            
            self.connections[connection_id] = conn_info
            
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
            
            self.total_connections += 1
            
            # Start heartbeat manager if this is the first connection
            if len(self.connections) == 1:
                await self.heartbeat_manager.start(self)
            
            logger.info(f"WebSocket connection {connection_id} established for user {user_id}")
            return True
    
    async def disconnect(self, connection_id: str):
        """Remove WebSocket connection."""
        async with self._connection_lock:
            if connection_id not in self.connections:
                return
            
            conn_info = self.connections[connection_id]
            user_id = conn_info.user_id
            
            # Remove from connections
            del self.connections[connection_id]
            
            # Remove from user connections
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Remove subscriptions
            for sub in conn_info.subscriptions:
                if sub.agent_id and sub.agent_id in self.agent_subscribers:
                    self.agent_subscribers[sub.agent_id].discard(connection_id)
                    if not self.agent_subscribers[sub.agent_id]:
                        del self.agent_subscribers[sub.agent_id]
            
            # Stop heartbeat manager if no connections left
            if len(self.connections) == 0:
                await self.heartbeat_manager.stop()
            
            logger.info(f"WebSocket connection {connection_id} disconnected for user {user_id}")
    
    async def _remove_dead_connection(self, connection_id: str):
        """Remove a dead connection."""
        if connection_id in self.connections:
            conn_info = self.connections[connection_id]
            try:
                await conn_info.websocket.close()
            except Exception:
                logger.info(f"Connection {connection_id} was already closed.")
            await self.disconnect(connection_id)
    
    async def send_personal_message(self, message: str, connection_id: str) -> bool:
        """Send message to specific connection."""
        if connection_id not in self.connections:
            return False
        
        conn_info = self.connections[connection_id]
        if not conn_info.is_alive:
            return False
        
        try:
            await conn_info.websocket.send_text(message)
            self.total_messages_sent += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {e}")
            conn_info.is_alive = False
            await self._remove_dead_connection(connection_id)
            return False
    
    async def send_to_user(self, message: str, user_id: str) -> int:
        """Send message to all connections of a user. Returns number of successful sends."""
        if user_id not in self.user_connections:
            return 0
        
        successful_sends = 0
        for connection_id in self.user_connections[user_id].copy():
            if await self.send_personal_message(message, connection_id):
                successful_sends += 1
        
        return successful_sends
    
    async def broadcast_to_subscribers(self, message: str, agent_id: str, message_type: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """Broadcast message to subscribers of specific agent with filtering. Returns number of successful sends."""
        if agent_id not in self.agent_subscribers:
            return 0
        
        successful_sends = 0
        message_data = json.loads(message)
        
        # Make a copy to avoid modification during iteration
        subscriber_ids = self.agent_subscribers[agent_id].copy()
        
        for connection_id in subscriber_ids:
            if connection_id not in self.connections:
                continue
            
            conn_info = self.connections[connection_id]
            
            # Check if connection has matching subscription
            matching_subscription = None
            for sub in conn_info.subscriptions:
                if sub.type == message_type and (sub.agent_id is None or sub.agent_id == agent_id):
                    matching_subscription = sub
                    break
            
            if not matching_subscription:
                continue
            
            # Apply subscription filters if specified
            if matching_subscription.filters:
                # Check message data against subscription filters
                # For epistemic updates, check the data field which contains the epistemic state
                filter_data = message_data.get("data", {}) if "data" in message_data else message_data
                if not self._message_matches_filters(filter_data, matching_subscription.filters):
                    continue
            
            if await self.send_personal_message(message, connection_id):
                successful_sends += 1
        
        return successful_sends
    
    def _message_matches_filters(self, message_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if message matches subscription filters."""
        for filter_key, filter_value in filters.items():
            if filter_key not in message_data:
                return False
            
            message_value = message_data[filter_key]
            
            # Support different filter types
            if isinstance(filter_value, dict):
                if "min" in filter_value and message_value < filter_value["min"]:
                    return False
                if "max" in filter_value and message_value > filter_value["max"]:
                    return False
                if "equals" in filter_value and message_value != filter_value["equals"]:
                    return False
                if "contains" in filter_value and filter_value["contains"] not in str(message_value):
                    return False
            else:
                if message_value != filter_value:
                    return False
        
        return True
    
    async def broadcast_system_alert(self, message: str) -> int:
        """Broadcast system alert to all connections. Returns number of successful sends."""
        successful_sends = 0
        
        # Make a copy to avoid dictionary changed size during iteration
        connections_copy = dict(self.connections)
        
        for connection_id, conn_info in connections_copy.items():
            # Check if connection still exists and subscribes to system alerts
            if (connection_id in self.connections and 
                any(sub.type == "system_alerts" for sub in conn_info.subscriptions)):
                if await self.send_personal_message(message, connection_id):
                    successful_sends += 1
        
        return successful_sends
    
    def add_subscription(self, connection_id: str, subscription: SubscriptionRequest) -> bool:
        """Add subscription for connection."""
        if connection_id not in self.connections:
            return False
        
        conn_info = self.connections[connection_id]
        
        # Check for duplicate subscriptions
        for existing_sub in conn_info.subscriptions:
            if (existing_sub.type == subscription.type and 
                existing_sub.agent_id == subscription.agent_id):
                return False  # Subscription already exists
        
        conn_info.subscriptions.append(subscription)
        
        # Track agent subscribers
        if subscription.agent_id:
            if subscription.agent_id not in self.agent_subscribers:
                self.agent_subscribers[subscription.agent_id] = set()
            self.agent_subscribers[subscription.agent_id].add(connection_id)
        
        return True
    
    def remove_subscription(self, connection_id: str, subscription_type: str, agent_id: Optional[str] = None) -> bool:
        """Remove subscription for connection."""
        if connection_id not in self.connections:
            return False
        
        conn_info = self.connections[connection_id]
        original_count = len(conn_info.subscriptions)
        
        conn_info.subscriptions = [
            sub for sub in conn_info.subscriptions
            if not (sub.type == subscription_type and sub.agent_id == agent_id)
        ]
        
        # Update agent subscribers
        if agent_id and agent_id in self.agent_subscribers:
            self.agent_subscribers[agent_id].discard(connection_id)
            if not self.agent_subscribers[agent_id]:
                del self.agent_subscribers[agent_id]
        
        return len(conn_info.subscriptions) < original_count
    
    def update_last_ping(self, connection_id: str):
        """Update last ping time for connection."""
        if connection_id in self.connections:
            self.connections[connection_id].last_ping = datetime.utcnow()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        active_connections = sum(1 for conn in self.connections.values() if conn.is_alive)
        total_subscriptions = sum(len(conn.subscriptions) for conn in self.connections.values())
        
        return {
            "active_connections": active_connections,
            "total_connections_ever": self.total_connections,
            "active_users": len(self.user_connections),
            "total_subscriptions": total_subscriptions,
            "messages_sent": self.total_messages_sent,
            "messages_received": self.total_messages_received,
            "agent_subscribers": len(self.agent_subscribers)
        }

# Global connection manager
manager = ConnectionManager()

async def authenticate_websocket(token: str) -> Optional[Dict[str, Any]]:
    """Authenticate WebSocket connection using JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        
        # Check token expiration
        exp = payload.get("exp")
        if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
            return None
        
        return {
            "user_id": user_id,
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", [])
        }
    except jwt.PyJWTError:
        return None

def check_websocket_permission(user_info: Dict[str, Any], required_permission: str) -> bool:
    """Check if user has required permission for WebSocket operation."""
    permissions = user_info.get("permissions", [])
    roles = user_info.get("roles", [])
    
    # Admin role has all permissions
    if "admin" in roles:
        return True
    
    # Check specific permission
    return required_permission in permissions

@websocket_router.websocket("/monitor/{session_id}")
async def websocket_monitor(websocket: WebSocket, session_id: str, token: str = Query(...)):
    """WebSocket endpoint for real-time monitoring data."""
    # Authenticate
    user_info = await authenticate_websocket(token)
    if not user_info:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return
    
    user_id = user_info["user_id"]
    
    # Check basic WebSocket permission
    if not check_websocket_permission(user_info, "websocket:connect"):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Insufficient permissions")
        return
    
    connection_id = str(uuid4())
    
    try:
        # Attempt to connect with connection limits
        if not await manager.connect(websocket, connection_id, user_id):
            return  # Connection rejected due to limits
        
        # Send connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connection_established",
                "connection_id": connection_id,
                "session_id": session_id,
                "user_id": user_id,
                "server_time": datetime.utcnow().isoformat(),
                "heartbeat_interval": manager.heartbeat_manager.ping_interval
            }),
            connection_id
        )
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                manager.total_messages_received += 1
                message_data = json.loads(data)
                
                await handle_websocket_message(connection_id, user_info, message_data)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket {connection_id} disconnected normally")
                break
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "code": "INVALID_JSON",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    connection_id
                )
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    connection_id
                )
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        await manager.disconnect(connection_id)

async def handle_websocket_message(connection_id: str, user_info: Dict[str, Any], message_data: Dict[str, Any]):
    """Handle incoming WebSocket message with enhanced security and features."""
    message_type = message_data.get("type")
    user_id = user_info["user_id"]
    
    if message_type == "subscribe":
        # Handle subscription request
        if not check_websocket_permission(user_info, "websocket:subscribe"):
            await manager.send_personal_message(
                json.dumps({
                    "type": "error",
                    "code": "PERMISSION_DENIED",
                    "message": "Insufficient permissions to subscribe",
                    "timestamp": datetime.utcnow().isoformat()
                }),
                connection_id
            )
            return
        
        try:
            subscription_data = message_data.get("data", {})
            subscription = SubscriptionRequest(**subscription_data)
            
            # Check agent-specific permissions
            if subscription.agent_id and not check_websocket_permission(user_info, f"agent:{subscription.agent_id}:monitor"):
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "code": "AGENT_ACCESS_DENIED",
                        "message": f"No permission to monitor agent {subscription.agent_id}",
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    connection_id
                )
                return
            
            if manager.add_subscription(connection_id, subscription):
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscription_confirmed",
                        "data": {
                            "subscription_type": subscription.type,
                            "agent_id": subscription.agent_id,
                            "filters": subscription.filters
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    connection_id
                )
            else:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "code": "SUBSCRIPTION_EXISTS",
                        "message": "Subscription already exists",
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    connection_id
                )
            
        except ValidationError as e:
            await manager.send_personal_message(
                json.dumps({
                    "type": "error",
                    "code": "VALIDATION_ERROR",
                    "message": f"Invalid subscription request: {e}",
                    "timestamp": datetime.utcnow().isoformat()
                }),
                connection_id
            )
    
    elif message_type == "unsubscribe":
        # Handle unsubscription request
        data = message_data.get("data", {})
        subscription_type = data.get("subscription_type")
        agent_id = data.get("agent_id")
        
        if manager.remove_subscription(connection_id, subscription_type, agent_id):
            await manager.send_personal_message(
                json.dumps({
                    "type": "unsubscription_confirmed",
                    "data": {
                        "subscription_type": subscription_type,
                        "agent_id": agent_id
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }),
                connection_id
            )
        else:
            await manager.send_personal_message(
                json.dumps({
                    "type": "error",
                    "code": "SUBSCRIPTION_NOT_FOUND",
                    "message": "Subscription not found",
                    "timestamp": datetime.utcnow().isoformat()
                }),
                connection_id
            )
    
    elif message_type == "pong":
        # Handle pong response to ping
        manager.update_last_ping(connection_id)
        # No response needed for pong
    
    elif message_type == "ping":
        # Handle manual ping from client
        await manager.send_personal_message(
            json.dumps({
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            }),
            connection_id
        )
    
    elif message_type == "get_subscriptions":
        # Handle request for current subscriptions
        if connection_id in manager.connections:
            conn_info = manager.connections[connection_id]
            await manager.send_personal_message(
                json.dumps({
                    "type": "subscriptions_list",
                    "data": {
                        "subscriptions": [sub.dict() for sub in conn_info.subscriptions]
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }),
                connection_id
            )
    
    elif message_type == "get_stats":
        # Handle request for connection statistics (admin only)
        if not check_websocket_permission(user_info, "websocket:admin"):
            await manager.send_personal_message(
                json.dumps({
                    "type": "error",
                    "code": "PERMISSION_DENIED",
                    "message": "Admin permission required",
                    "timestamp": datetime.utcnow().isoformat()
                }),
                connection_id
            )
            return
        
        stats = manager.get_connection_stats()
        await manager.send_personal_message(
            json.dumps({
                "type": "stats",
                "data": stats,
                "timestamp": datetime.utcnow().isoformat()
            }),
            connection_id
        )
    
    else:
        await manager.send_personal_message(
            json.dumps({
                "type": "error",
                "code": "UNKNOWN_MESSAGE_TYPE",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.utcnow().isoformat()
            }),
            connection_id
        )

# Functions to broadcast updates (called by other components)
async def broadcast_epistemic_update(agent_id: str, epistemic_state: EpistemicState, filters: Optional[Dict[str, Any]] = None) -> int:
    """Broadcast epistemic state update. Returns number of successful sends."""
    message = json.dumps({
        "type": "epistemic_update",
        "agent_id": agent_id,
        "data": epistemic_state.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return await manager.broadcast_to_subscribers(message, agent_id, "epistemic_updates", filters)

async def broadcast_pattern_alert(agent_id: str, pattern: BehavioralPattern, alert_type: str, severity: str = "info", filters: Optional[Dict[str, Any]] = None) -> int:
    """Broadcast behavioral pattern alert. Returns number of successful sends."""
    message = json.dumps({
        "type": "pattern_alert",
        "agent_id": agent_id,
        "alert_type": alert_type,
        "severity": severity,
        "data": pattern.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return await manager.broadcast_to_subscribers(message, agent_id, "pattern_alerts", filters)

async def broadcast_prediction_update(agent_id: str, prediction: PredictionResult, filters: Optional[Dict[str, Any]] = None) -> int:
    """Broadcast prediction update. Returns number of successful sends."""
    message = json.dumps({
        "type": "prediction_update",
        "agent_id": agent_id,
        "data": prediction.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return await manager.broadcast_to_subscribers(message, agent_id, "prediction_updates", filters)

async def broadcast_system_alert(alert_type: str, message: str, severity: str = "info", metadata: Optional[Dict[str, Any]] = None) -> int:
    """Broadcast system-wide alert. Returns number of successful sends."""
    alert_message = json.dumps({
        "type": "system_alert",
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return await manager.broadcast_system_alert(alert_message)

# Additional utility functions
async def send_user_notification(user_id: str, notification_type: str, message: str, data: Optional[Dict[str, Any]] = None) -> int:
    """Send notification to specific user. Returns number of successful sends."""
    notification = json.dumps({
        "type": "user_notification",
        "notification_type": notification_type,
        "message": message,
        "data": data or {},
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return await manager.send_to_user(notification, user_id)

async def broadcast_agent_status_change(agent_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> int:
    """Broadcast agent status change. Returns number of successful sends."""
    message = json.dumps({
        "type": "agent_status_change",
        "agent_id": agent_id,
        "status": status,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return await manager.broadcast_to_subscribers(message, agent_id, "agent_status")

# Health check and management endpoints
@websocket_router.get("/health")
async def websocket_health():
    """WebSocket service health check with detailed statistics."""
    stats = manager.get_connection_stats()
    
    # Determine health status
    status = "healthy"
    if stats["active_connections"] == 0:
        status = "idle"
    elif stats["active_connections"] >= manager.max_connections * 0.9:
        status = "near_capacity"
    
    return {
        "status": status,
        "max_connections": manager.max_connections,
        "heartbeat_interval": manager.heartbeat_manager.ping_interval,
        "heartbeat_timeout": manager.heartbeat_manager.pong_timeout,
        "heartbeat_running": manager.heartbeat_manager.running,
        **stats,
        "timestamp": datetime.utcnow().isoformat()
    }

@websocket_router.get("/connections")
async def list_connections(current_user: dict = Depends(get_current_user)):
    """List active WebSocket connections (admin only)."""
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    connections_info = []
    for connection_id, conn_info in manager.connections.items():
        connections_info.append({
            "connection_id": connection_id,
            "user_id": conn_info.user_id,
            "connected_at": conn_info.connected_at.isoformat(),
            "last_ping": conn_info.last_ping.isoformat(),
            "is_alive": conn_info.is_alive,
            "subscriptions": [sub.dict() for sub in conn_info.subscriptions]
        })
    
    return {
        "connections": connections_info,
        "total": len(connections_info),
        "timestamp": datetime.utcnow().isoformat()
    }

@websocket_router.post("/broadcast/test")
async def test_broadcast(
    message: str,
    message_type: str = "test",
    agent_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Test broadcast functionality (admin only)."""
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    if agent_id:
        # Broadcast to specific agent subscribers
        sent_count = await manager.broadcast_to_subscribers(
            json.dumps({
                "type": message_type,
                "message": message,
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }),
            agent_id,
            message_type
        )
    else:
        # System-wide broadcast
        sent_count = await broadcast_system_alert("test", message, "info")
    
    return {
        "message": "Broadcast sent",
        "recipients": sent_count,
        "timestamp": datetime.utcnow().isoformat()
    }