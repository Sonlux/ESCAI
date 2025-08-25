"""
Example demonstrating WebSocket real-time interface usage.
"""

import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, Any

from escai_framework.api.websocket import (
    broadcast_epistemic_update, broadcast_system_alert,
    broadcast_pattern_alert, send_user_notification
)
from escai_framework.models.epistemic_state import (
    EpistemicState, BeliefState, KnowledgeState, GoalState,
    BeliefType, GoalStatus
)
from escai_framework.models.behavioral_pattern import (
    BehavioralPattern, ExecutionSequence
)


class WebSocketClient:
    """Simple WebSocket client for testing."""
    
    def __init__(self, uri: str, token: str):
        self.uri = uri
        self.token = token
        self.websocket = None
        self.running = False
    
    async def connect(self):
        """Connect to WebSocket server."""
        try:
            self.websocket = await websockets.connect(f"{self.uri}?token={self.token}")
            self.running = True
            print(f"Connected to {self.uri}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from WebSocket server")
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to server."""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
            print(f"Sent: {message}")
    
    async def listen(self):
        """Listen for messages from server."""
        while self.running and self.websocket:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                await self.handle_message(data)
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server")
                break
            except Exception as e:
                print(f"Error receiving message: {e}")
                break
    
    async def handle_message(self, data: Dict[str, Any]):
        """Handle incoming message."""
        message_type = data.get("type")
        timestamp = data.get("timestamp", "")
        
        if message_type == "connection_established":
            print(f"✅ Connection established: {data.get('connection_id')}")
            print(f"   Heartbeat interval: {data.get('heartbeat_interval')}s")
        
        elif message_type == "subscription_confirmed":
            sub_data = data.get("data", {})
            print(f"✅ Subscription confirmed: {sub_data.get('subscription_type')} for agent {sub_data.get('agent_id')}")
        
        elif message_type == "epistemic_update":
            agent_id = data.get("agent_id")
            epistemic_data = data.get("data", {})
            confidence = epistemic_data.get("confidence_level", 0)
            print(f"🧠 Epistemic update for {agent_id}: confidence={confidence:.2f}")
        
        elif message_type == "pattern_alert":
            agent_id = data.get("agent_id")
            alert_type = data.get("alert_type")
            severity = data.get("severity", "info")
            print(f"⚠️  Pattern alert for {agent_id}: {alert_type} ({severity})")
        
        elif message_type == "prediction_update":
            agent_id = data.get("agent_id")
            prediction_data = data.get("data", {})
            success_probability = prediction_data.get("success_probability", 0)
            print(f"🔮 Prediction update for {agent_id}: success={success_probability:.2f}")
        
        elif message_type == "system_alert":
            alert_type = data.get("alert_type")
            message = data.get("message")
            severity = data.get("severity", "info")
            print(f"🚨 System alert ({severity}): {alert_type} - {message}")
        
        elif message_type == "ping":
            # Respond to ping with pong
            await self.send_message({"type": "pong"})
        
        elif message_type == "pong":
            print("💓 Heartbeat pong received")
        
        elif message_type == "error":
            error_code = data.get("code", "UNKNOWN")
            error_message = data.get("message", "Unknown error")
            print(f"❌ Error ({error_code}): {error_message}")
        
        else:
            print(f"📨 Received: {data}")
    
    async def subscribe_to_agent(self, agent_id: str, subscription_types: list = None):
        """Subscribe to agent updates."""
        if subscription_types is None:
            subscription_types = ["epistemic_updates", "pattern_alerts", "prediction_updates"]
        
        for sub_type in subscription_types:
            await self.send_message({
                "type": "subscribe",
                "data": {
                    "type": sub_type,
                    "agent_id": agent_id,
                    "filters": {"confidence_level": {"min": 0.7}} if sub_type == "epistemic_updates" else None
                }
            })
    
    async def subscribe_to_system_alerts(self):
        """Subscribe to system-wide alerts."""
        await self.send_message({
            "type": "subscribe",
            "data": {
                "type": "system_alerts"
            }
        })


async def simulate_agent_activity():
    """Simulate agent activity by broadcasting updates."""
    print("\n🤖 Starting agent activity simulation...")
    
    # Create sample epistemic state
    epistemic_state = EpistemicState(
        agent_id="demo_agent",
        timestamp=datetime.utcnow(),
        belief_states=[
            BeliefState(
                content="The user wants to analyze sales data",
                belief_type=BeliefType.FACTUAL,
                confidence=0.9
            )
        ],
        knowledge_state=KnowledgeState(
            facts=["Sales data is in CSV format", "Data contains 12 months of records"],
            concepts={"data_analysis": "statistical analysis of business metrics"}
        ),
        goal_states=[
            GoalState(
                description="Complete sales data analysis",
                status=GoalStatus.ACTIVE,
                priority=8,
                progress=0.3,
                sub_goals=["Load data", "Clean data", "Generate insights"]
            )
        ],
        confidence_level=0.85,
        uncertainty_score=0.15,
        decision_context={"task_type": "data_analysis", "urgency": "high"}
    )
    
    # Broadcast epistemic update
    print("📡 Broadcasting epistemic update...")
    sent_count = await broadcast_epistemic_update("demo_agent", epistemic_state)
    print(f"   Sent to {sent_count} subscribers")
    
    await asyncio.sleep(2)
    
    # Create sample behavioral pattern
    pattern = BehavioralPattern(
        pattern_id="data_loading_pattern",
        pattern_name="Data Loading Sequence",
        execution_sequences=[
            ExecutionSequence(
                sequence_id="seq_1",
                agent_id="demo_agent",
                steps=["validate_file", "load_csv", "check_schema"],
                timestamps=[datetime.utcnow() for _ in range(3)],
                success=True,
                duration_ms=1500
            )
        ],
        frequency=5,
        success_rate=0.95,
        average_duration=1200.0,
        common_triggers=["csv_file_detected"],
        failure_modes=["invalid_schema", "file_not_found"]
    )
    
    # Broadcast pattern alert
    print("📡 Broadcasting pattern alert...")
    sent_count = await broadcast_pattern_alert("demo_agent", pattern, "pattern_detected", "info")
    print(f"   Sent to {sent_count} subscribers")
    
    await asyncio.sleep(2)
    
    # Broadcast system alert
    print("📡 Broadcasting system alert...")
    sent_count = await broadcast_system_alert("maintenance", "Scheduled maintenance in 30 minutes", "warning")
    print(f"   Sent to {sent_count} subscribers")


async def run_client_example():
    """Run WebSocket client example."""
    # Note: This would need a real JWT token in practice
    token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJkZW1vX3VzZXIiLCJyb2xlcyI6WyJ1c2VyIl0sInBlcm1pc3Npb25zIjpbIndlYnNvY2tldDpjb25uZWN0Iiwid2Vic29ja2V0OnN1YnNjcmliZSIsImFnZW50OmRlbW9fYWdlbnQ6bW9uaXRvciJdLCJleHAiOjk5OTk5OTk5OTl9.demo_token"
    
    client = WebSocketClient("ws://localhost:8000/ws/monitor/demo_session", token)
    
    try:
        # Connect to server
        if not await client.connect():
            return
        
        # Start listening for messages
        listen_task = asyncio.create_task(client.listen())
        
        # Wait a bit for connection to establish
        await asyncio.sleep(1)
        
        # Subscribe to agent updates
        print("\n📝 Subscribing to agent updates...")
        await client.subscribe_to_agent("demo_agent")
        
        # Subscribe to system alerts
        print("📝 Subscribing to system alerts...")
        await client.subscribe_to_system_alerts()
        
        # Wait for subscriptions to be confirmed
        await asyncio.sleep(1)
        
        # Simulate some agent activity
        await simulate_agent_activity()
        
        # Send a ping
        print("\n💓 Sending ping...")
        await client.send_message({"type": "ping"})
        
        # Wait for responses
        await asyncio.sleep(3)
        
        # Get current subscriptions
        print("\n📋 Getting current subscriptions...")
        await client.send_message({"type": "get_subscriptions"})
        
        await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        await client.disconnect()
        if not listen_task.done():
            listen_task.cancel()


async def main():
    """Main example function."""
    print("🚀 WebSocket Real-time Interface Example")
    print("=" * 50)
    
    print("\nThis example demonstrates:")
    print("• WebSocket connection management")
    print("• Real-time subscription system")
    print("• Event broadcasting with filtering")
    print("• Heartbeat mechanism")
    print("• Authentication and authorization")
    
    print("\n📋 To run this example:")
    print("1. Start the ESCAI API server: python -m escai_framework.api.main")
    print("2. Run this example: python examples/websocket_example.py")
    print("3. The client will connect and demonstrate various features")
    
    # Note: In a real scenario, you would uncomment the following line
    # await run_client_example()
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())