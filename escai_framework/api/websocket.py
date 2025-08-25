"""
WebSocket endpoints for real-time communication in ESCAI Framework API.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status, Depends, Query
from pydantic import BaseModel, ValidationError
import jwt

from .auth import auth_manager, SECRET_KEY, ALGORITHM
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

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.subscriptions: Dict[str, List[SubscriptionRequest]] = {}  # connection_id -> subscriptions
        self.agent_subscribers: Dict[str, Set[str]] = {}  # agent_id -> connection_ids
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        self.subscriptions[connection_id] = []
        
        logger.info(f"WebSocket connection {connection_id} established for user {user_id}")
    
    def disconnect(self, connection_id: str, user_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Remove subscriptions
        if connection_id in self.subscriptions:
            subscriptions = self.subscriptions[connection_id]
            for sub in subscriptions:
                if sub.agent_id and sub.agent_id in self.agent_subscribers:
                    self.agent_subscribers[sub.agent_id].discard(connection_id)
                    if not self.agent_subscribers[sub.agent_id]:
                        del self.agent_subscribers[sub.agent_id]
            del self.subscriptions[connection_id]
        
        logger.info(f"WebSocket connection {connection_id} disconnected for user {user_id}")
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to connection {connection_id}: {e}")
                # Remove broken connection
                await self._remove_broken_connection(connection_id)
    
    async def send_to_user(self, message: str, user_id: str):
        """Send message to all connections of a user."""
        if user_id in self.user_connections:
            broken_connections = []
            for connection_id in self.user_connections[user_id].copy():
                try:
                    await self.active_connections[connection_id].send_text(message)
                except Exception as e:
                    logger.error(f"Failed to send message to connection {connection_id}: {e}")
                    broken_connections.append(connection_id)
            
            # Remove broken connections
            for connection_id in broken_connections:
                await self._remove_broken_connection(connection_id)
    
    async def broadcast_to_subscribers(self, message: str, agent_id: str, message_type: str):
        """Broadcast message to subscribers of specific agent."""
        if agent_id not in self.agent_subscribers:
            return
        
        broken_connections = []
        for connection_id in self.agent_subscribers[agent_id].copy():
            # Check if connection has matching subscription
            if connection_id in self.subscriptions:
                subscriptions = self.subscriptions[connection_id]
                if any(sub.type == message_type and (sub.agent_id is None or sub.agent_id == agent_id) 
                       for sub in subscriptions):
                    try:
                        await self.active_connections[connection_id].send_text(message)
                    except Exception as e:
                        logger.error(f"Failed to send message to connection {connection_id}: {e}")
                        broken_connections.append(connection_id)
        
        # Remove broken connections
        for connection_id in broken_connections:
            await self._remove_broken_connection(connection_id)
    
    async def broadcast_system_alert(self, message: str):
        """Broadcast system alert to all connections."""
        broken_connections = []
        for connection_id, websocket in self.active_connections.items():
            # Check if connection subscribes to system alerts
            if connection_id in self.subscriptions:
                subscriptions = self.subscriptions[connection_id]
                if any(sub.type == "system_alerts" for sub in subscriptions):
                    try:
                        await websocket.send_text(message)
                    except Exception as e:
                        logger.error(f"Failed to send system alert to connection {connection_id}: {e}")
                        broken_connections.append(connection_id)
        
        # Remove broken connections
        for connection_id in broken_connections:
            await self._remove_broken_connection(connection_id)
    
    def add_subscription(self, connection_id: str, subscription: SubscriptionRequest):
        """Add subscription for connection."""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].append(subscription)
            
            # Track agent subscribers
            if subscription.agent_id:
                if subscription.agent_id not in self.agent_subscribers:
                    self.agent_subscribers[subscription.agent_id] = set()
                self.agent_subscribers[subscription.agent_id].add(connection_id)
    
    def remove_subscription(self, connection_id: str, subscription_type: str, agent_id: Optional[str] = None):
        """Remove subscription for connection."""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id] = [
                sub for sub in self.subscriptions[connection_id]
                if not (sub.type == subscription_type and sub.agent_id == agent_id)
            ]
            
            # Update agent subscribers
            if agent_id and agent_id in self.agent_subscribers:
                self.agent_subscribers[agent_id].discard(connection_id)
                if not self.agent_subscribers[agent_id]:
                    del self.agent_subscribers[agent_id]
    
    async def _remove_broken_connection(self, connection_id: str):
        """Remove broken connection."""
        # Find user_id for this connection
        user_id = None
        for uid, conn_ids in self.user_connections.items():
            if connection_id in conn_ids:
                user_id = uid
                break
        
        if user_id:
            self.disconnect(connection_id, user_id)

# Global connection manager
manager = ConnectionManager()

async def authenticate_websocket(token: str) -> Optional[str]:
    """Authenticate WebSocket connection using JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except jwt.PyJWTError:
        return None

@websocket_router.websocket("/monitor/{session_id}")
async def websocket_monitor(websocket: WebSocket, session_id: str, token: str = Query(...)):
    """WebSocket endpoint for real-time monitoring data."""
    # Authenticate
    user_id = await authenticate_websocket(token)
    if not user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    connection_id = str(uuid4())
    
    try:
        await manager.connect(websocket, connection_id, user_id)
        
        # Send connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connection_established",
                "connection_id": connection_id,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }),
            connection_id
        )
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                await handle_websocket_message(connection_id, user_id, message_data)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
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
                        "message": "Internal server error",
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    connection_id
                )
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        manager.disconnect(connection_id, user_id)

async def handle_websocket_message(connection_id: str, user_id: str, message_data: Dict[str, Any]):
    """Handle incoming WebSocket message."""
    message_type = message_data.get("type")
    
    if message_type == "subscribe":
        # Handle subscription request
        try:
            subscription = SubscriptionRequest(**message_data.get("data", {}))
            manager.add_subscription(connection_id, subscription)
            
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
            
        except ValidationError as e:
            await manager.send_personal_message(
                json.dumps({
                    "type": "error",
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
        
        manager.remove_subscription(connection_id, subscription_type, agent_id)
        
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
    
    elif message_type == "ping":
        # Handle ping/pong for connection health
        await manager.send_personal_message(
            json.dumps({
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            }),
            connection_id
        )
    
    else:
        await manager.send_personal_message(
            json.dumps({
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.utcnow().isoformat()
            }),
            connection_id
        )

# Functions to broadcast updates (called by other components)
async def broadcast_epistemic_update(agent_id: str, epistemic_state: EpistemicState):
    """Broadcast epistemic state update."""
    message = json.dumps({
        "type": "epistemic_update",
        "agent_id": agent_id,
        "data": epistemic_state.dict(),
        "timestamp": datetime.utcnow().isoformat()
    })
    
    await manager.broadcast_to_subscribers(message, agent_id, "epistemic_updates")

async def broadcast_pattern_alert(agent_id: str, pattern: BehavioralPattern, alert_type: str):
    """Broadcast behavioral pattern alert."""
    message = json.dumps({
        "type": "pattern_alert",
        "agent_id": agent_id,
        "alert_type": alert_type,
        "data": pattern.dict(),
        "timestamp": datetime.utcnow().isoformat()
    })
    
    await manager.broadcast_to_subscribers(message, agent_id, "pattern_alerts")

async def broadcast_prediction_update(agent_id: str, prediction: PredictionResult):
    """Broadcast prediction update."""
    message = json.dumps({
        "type": "prediction_update",
        "agent_id": agent_id,
        "data": prediction.dict(),
        "timestamp": datetime.utcnow().isoformat()
    })
    
    await manager.broadcast_to_subscribers(message, agent_id, "prediction_updates")

async def broadcast_system_alert(alert_type: str, message: str, severity: str = "info"):
    """Broadcast system-wide alert."""
    alert_message = json.dumps({
        "type": "system_alert",
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    await manager.broadcast_system_alert(alert_message)

# Health check for WebSocket connections
@websocket_router.get("/health")
async def websocket_health():
    """WebSocket service health check."""
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "active_users": len(manager.user_connections),
        "total_subscriptions": sum(len(subs) for subs in manager.subscriptions.values()),
        "timestamp": datetime.utcnow().isoformat()
    }