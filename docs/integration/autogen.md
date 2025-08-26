# AutoGen Integration Guide

This guide shows how to integrate ESCAI monitoring with Microsoft AutoGen multi-agent conversations and workflows.

## Quick Start

### 1. Installation

```bash
pip install escai-framework pyautogen
```

### 2. Basic Integration

```python
import autogen
from escai_framework import monitor_agent

# Configure AutoGen agents
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-openai-api-key",
    }
]

llm_config = {"config_list": config_list, "temperature": 0}

# Create agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "coding"},
)

# Monitor the multi-agent conversation
with monitor_agent(
    agent_id="autogen-conversation",
    framework="autogen",
    config={
        "monitor_conversations": True,
        "capture_role_changes": True,
        "track_group_decisions": True,
        "monitor_agent_interactions": True
    }
) as session:
    # Start the conversation
    user_proxy.initiate_chat(
        assistant,
        message="Write a Python function to calculate the factorial of a number."
    )

    # Access conversation analysis
    conversation_analysis = session.get_conversation_analysis()
    print(f"Total messages: {conversation_analysis['message_count']}")
    print(f"Agents involved: {conversation_analysis['agent_count']}")
    print(f"Conversation turns: {conversation_analysis['turn_count']}")
```

## Detailed Integration

### 1. Multi-Agent Group Chat Monitoring

Monitor complex group conversations with multiple agents:

```python
import autogen
from escai_framework.instrumentation import AutoGenInstrumentor

# Create multiple agents
researcher = autogen.AssistantAgent(
    name="researcher",
    llm_config=llm_config,
    system_message="You are a research specialist. Gather and analyze information."
)

writer = autogen.AssistantAgent(
    name="writer",
    llm_config=llm_config,
    system_message="You are a content writer. Create engaging and informative content."
)

critic = autogen.AssistantAgent(
    name="critic",
    llm_config=llm_config,
    system_message="You are a critic. Review and provide feedback on content quality."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

# Create group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, writer, critic],
    messages=[],
    max_round=20
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# Initialize instrumentor for detailed monitoring
instrumentor = AutoGenInstrumentor()

session_id = instrumentor.start_monitoring(
    agent_id="research-writing-team",
    config={
        "monitor_conversations": True,
        "capture_role_changes": True,
        "track_group_decisions": True,
        "monitor_agent_coordination": True,
        "conversation_history_limit": 100,
        "capture_agent_personalities": True,
        "track_consensus_building": True
    }
)

try:
    # Start group conversation
    user_proxy.initiate_chat(
        manager,
        message="Research and write a comprehensive article about renewable energy trends in 2024."
    )

    # Analyze group dynamics
    group_analysis = instrumentor.get_group_dynamics_analysis(session_id)
    print("Group Dynamics Analysis:")
    print(f"  Participation balance: {group_analysis['participation_balance']:.2f}")
    print(f"  Consensus level: {group_analysis['consensus_level']:.2f}")
    print(f"  Conflict instances: {group_analysis['conflict_count']}")

    # Agent-specific analysis
    for agent_name, stats in group_analysis['agent_stats'].items():
        print(f"  {agent_name}:")
        print(f"    Messages sent: {stats['message_count']}")
        print(f"    Influence score: {stats['influence_score']:.2f}")
        print(f"    Collaboration rating: {stats['collaboration_rating']:.2f}")

finally:
    summary = instrumentor.stop_monitoring(session_id)
    print(f"Conversation summary: {summary}")
```

### 2. Custom Agent Monitoring

Monitor custom AutoGen agents with specific behaviors:

```python
class DataAnalystAgent(autogen.AssistantAgent):
    """Custom agent for data analysis tasks."""

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.analysis_history = []

    def analyze_data(self, data):
        """Custom method for data analysis."""
        # Custom analysis logic
        analysis_result = f"Analysis of {len(data)} data points completed"
        self.analysis_history.append(analysis_result)
        return analysis_result

    def get_analysis_summary(self):
        """Get summary of all analyses performed."""
        return {
            "total_analyses": len(self.analysis_history),
            "latest_analysis": self.analysis_history[-1] if self.analysis_history else None
        }

class ProjectManagerAgent(autogen.AssistantAgent):
    """Custom agent for project management."""

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.tasks = []
        self.completed_tasks = []

    def assign_task(self, task, assignee):
        """Assign a task to an agent."""
        task_info = {
            "task": task,
            "assignee": assignee,
            "assigned_at": time.time()
        }
        self.tasks.append(task_info)
        return f"Task '{task}' assigned to {assignee}"

    def complete_task(self, task):
        """Mark a task as completed."""
        for t in self.tasks:
            if t["task"] == task:
                t["completed_at"] = time.time()
                self.completed_tasks.append(t)
                self.tasks.remove(t)
                break

# Create custom agents
data_analyst = DataAnalystAgent(
    name="data_analyst",
    llm_config=llm_config,
    system_message="You are a data analyst. Analyze data and provide insights."
)

project_manager = ProjectManagerAgent(
    name="project_manager",
    llm_config=llm_config,
    system_message="You are a project manager. Coordinate tasks and manage workflow."
)

# Monitor custom agent behaviors
with monitor_agent(
    agent_id="custom-agent-workflow",
    framework="autogen",
    config={
        "monitor_conversations": True,
        "track_custom_methods": True,
        "capture_agent_state": True,
        "monitor_task_delegation": True,
        "track_workflow_patterns": True
    }
) as session:

    # Create group chat with custom agents
    groupchat = autogen.GroupChat(
        agents=[user_proxy, project_manager, data_analyst],
        messages=[],
        max_round=15
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )

    # Start workflow
    user_proxy.initiate_chat(
        manager,
        message="We need to analyze sales data and create a report. Please coordinate the work."
    )

    # Analyze custom agent behaviors
    custom_analysis = session.get_custom_agent_analysis()
    print("Custom Agent Analysis:")
    print(f"  Task assignments: {custom_analysis['task_assignments']}")
    print(f"  Data analyses performed: {custom_analysis['data_analyses']}")
    print(f"  Workflow efficiency: {custom_analysis['workflow_efficiency']:.2f}")
```

### 3. Conversation Flow Analysis

Analyze conversation patterns and decision-making processes:

```python
from escai_framework import ESCAIClient
import autogen

# Set up agents for decision-making scenario
decision_maker = autogen.AssistantAgent(
    name="decision_maker",
    llm_config=llm_config,
    system_message="You make final decisions based on input from advisors."
)

financial_advisor = autogen.AssistantAgent(
    name="financial_advisor",
    llm_config=llm_config,
    system_message="You provide financial analysis and recommendations."
)

technical_advisor = autogen.AssistantAgent(
    name="technical_advisor",
    llm_config=llm_config,
    system_message="You provide technical feasibility analysis."
)

market_advisor = autogen.AssistantAgent(
    name="market_advisor",
    llm_config=llm_config,
    system_message="You provide market analysis and competitive insights."
)

# Initialize ESCAI client
client = ESCAIClient(
    base_url="http://localhost:8000",
    username="your_username",
    password="your_password"
)

# Start monitoring with conversation flow analysis
session = client.start_monitoring(
    agent_id="decision-making-team",
    framework="autogen",
    config={
        "monitor_conversations": True,
        "track_decision_processes": True,
        "analyze_influence_patterns": True,
        "capture_consensus_building": True,
        "monitor_information_flow": True,
        "track_opinion_changes": True
    }
)

try:
    # Create decision-making group
    groupchat = autogen.GroupChat(
        agents=[user_proxy, decision_maker, financial_advisor, technical_advisor, market_advisor],
        messages=[],
        max_round=25
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )

    # Start decision-making conversation
    user_proxy.initiate_chat(
        manager,
        message="We're considering launching a new AI-powered mobile app. Please analyze the opportunity and make a recommendation."
    )

    # Analyze conversation flow
    flow_analysis = client.get_conversation_flow_analysis(session.session_id)
    print("Conversation Flow Analysis:")
    print(f"  Decision points: {len(flow_analysis['decision_points'])}")
    print(f"  Information requests: {flow_analysis['info_requests']}")
    print(f"  Opinion changes: {flow_analysis['opinion_changes']}")
    print(f"  Consensus reached: {flow_analysis['consensus_reached']}")

    # Analyze agent influence
    influence_analysis = client.get_influence_analysis(session.session_id)
    print("\nAgent Influence Analysis:")
    for agent, influence in influence_analysis.items():
        print(f"  {agent}: {influence['influence_score']:.2f}")
        print(f"    Key contributions: {influence['key_contributions']}")
        print(f"    Decision impact: {influence['decision_impact']:.2f}")

finally:
    summary = client.stop_monitoring(session.session_id)
```

### 4. Real-time Conversation Monitoring

Monitor conversations in real-time with WebSocket updates:

```python
import asyncio
import websockets
import json
from escai_framework import AsyncESCAIClient

async def monitor_realtime_conversation():
    # Initialize async client
    client = AsyncESCAIClient(
        base_url="http://localhost:8000",
        username="your_username",
        password="your_password"
    )

    # Start monitoring
    session = await client.start_monitoring(
        agent_id="realtime-autogen-chat",
        framework="autogen",
        config={
            "monitor_conversations": True,
            "real_time_analysis": True,
            "capture_sentiment": True,
            "track_engagement_levels": True
        }
    )

    # Connect to WebSocket for real-time updates
    ws_url = f"ws://localhost:8000/ws/monitor/{session.session_id}"

    async def handle_updates():
        async with websockets.connect(ws_url) as websocket:
            async for message in websocket:
                data = json.loads(message)

                if data["type"] == "conversation_update":
                    update = data["payload"]
                    print(f"New message from {update['agent_name']}: {update['message'][:50]}...")
                    print(f"Sentiment: {update['sentiment']}")
                    print(f"Engagement: {update['engagement_level']}")

                elif data["type"] == "pattern_detected":
                    pattern = data["payload"]
                    print(f"Pattern detected: {pattern['pattern_name']}")
                    print(f"Confidence: {pattern['confidence']}")

                elif data["type"] == "alert":
                    alert = data["payload"]
                    print(f"ALERT: {alert['message']}")

    # Start real-time monitoring
    update_task = asyncio.create_task(handle_updates())

    # Run AutoGen conversation (in a separate thread or process)
    # This would be your normal AutoGen conversation code

    # Wait for updates
    await update_task

    # Stop monitoring
    await client.stop_monitoring(session.session_id)

# Run real-time monitoring
asyncio.run(monitor_realtime_conversation())
```

## Configuration Options

### AutoGen-Specific Configuration

```python
autogen_config = {
    # Conversation monitoring
    "monitor_conversations": True,          # Track all conversation messages
    "capture_message_metadata": True,       # Store message timestamps, roles, etc.
    "track_conversation_flow": True,        # Analyze conversation patterns
    "conversation_history_limit": 100,      # Max messages to keep in memory

    # Agent behavior monitoring
    "capture_role_changes": True,           # Monitor when agents change roles
    "track_agent_personalities": True,      # Analyze agent communication styles
    "monitor_agent_coordination": True,     # Track how agents coordinate
    "capture_agent_state": True,            # Monitor internal agent state

    # Group dynamics
    "track_group_decisions": True,          # Monitor group decision processes
    "analyze_influence_patterns": True,     # Track agent influence on decisions
    "capture_consensus_building": True,     # Monitor consensus formation
    "monitor_conflict_resolution": True,    # Track how conflicts are resolved

    # Decision analysis
    "track_decision_processes": True,       # Monitor decision-making steps
    "capture_reasoning_chains": True,       # Store reasoning processes
    "monitor_information_flow": True,       # Track information sharing
    "track_opinion_changes": True,          # Monitor opinion evolution

    # Custom agent monitoring
    "track_custom_methods": True,           # Monitor custom agent methods
    "monitor_task_delegation": True,        # Track task assignments
    "capture_workflow_patterns": True,      # Analyze workflow efficiency
    "track_skill_utilization": True,        # Monitor skill usage

    # Real-time analysis
    "real_time_analysis": True,             # Enable real-time insights
    "capture_sentiment": True,              # Analyze message sentiment
    "track_engagement_levels": True,        # Monitor agent engagement
    "detect_anomalies": True,               # Detect unusual patterns

    # Performance settings
    "sampling_rate": 1.0,                   # Fraction of events to capture
    "batch_size": 50,                       # Messages per processing batch
    "async_processing": True,               # Use async processing
    "buffer_size": 500,                     # Message buffer size

    # Storage settings
    "store_full_conversations": True,       # Keep complete conversation history
    "compress_messages": False,             # Compress stored messages
    "retention_days": 30,                   # How long to keep data
}
```

## Advanced Features

### 1. Conversation Analytics Dashboard

```python
from escai_framework.visualization import AutoGenDashboard

# Create conversation analytics dashboard
dashboard = AutoGenDashboard(session_id=session.session_id)

# Configure dashboard widgets
dashboard.add_widget("conversation_flow", {
    "show_message_timeline": True,
    "highlight_decision_points": True,
    "show_agent_interactions": True
})

dashboard.add_widget("agent_performance", {
    "show_participation_balance": True,
    "track_influence_scores": True,
    "monitor_response_times": True
})

dashboard.add_widget("group_dynamics", {
    "show_consensus_building": True,
    "track_conflict_resolution": True,
    "monitor_collaboration_patterns": True
})

# Start dashboard
dashboard.start(port=8081)
print("AutoGen dashboard available at http://localhost:8081")
```

### 2. Custom Conversation Patterns

```python
from escai_framework.patterns import ConversationPattern

# Define custom conversation patterns
brainstorming_pattern = ConversationPattern(
    name="brainstorming_session",
    description="Collaborative idea generation pattern",
    triggers=[
        "multiple agents contributing ideas",
        "building on previous suggestions",
        "creative problem solving"
    ],
    indicators=[
        lambda msgs: len(set(msg.sender for msg in msgs)) >= 3,
        lambda msgs: any("idea" in msg.content.lower() for msg in msgs),
        lambda msgs: any("what if" in msg.content.lower() for msg in msgs)
    ]
)

decision_making_pattern = ConversationPattern(
    name="decision_making_process",
    description="Structured decision-making pattern",
    triggers=[
        "options being evaluated",
        "pros and cons discussion",
        "final decision reached"
    ],
    indicators=[
        lambda msgs: any("option" in msg.content.lower() for msg in msgs),
        lambda msgs: any("decide" in msg.content.lower() for msg in msgs),
        lambda msgs: any("recommend" in msg.content.lower() for msg in msgs)
    ]
)

# Add patterns to monitoring
config["custom_patterns"] = [brainstorming_pattern, decision_making_pattern]

with monitor_agent(agent_id, "autogen", config) as session:
    # Your AutoGen conversation
    # Custom patterns will be automatically detected
    pass
```

### 3. Agent Performance Metrics

```python
from escai_framework.metrics import AgentMetric

# Define custom agent metrics
response_quality_metric = AgentMetric(
    name="response_quality",
    description="Quality of agent responses",
    calculator=lambda msg: calculate_response_quality(msg.content)
)

collaboration_metric = AgentMetric(
    name="collaboration_score",
    description="How well agent collaborates",
    calculator=lambda msgs: calculate_collaboration_score(msgs)
)

# Add metrics to configuration
config["agent_metrics"] = [response_quality_metric, collaboration_metric]

with monitor_agent(agent_id, "autogen", config) as session:
    # Run conversation
    # Metrics are automatically calculated

    # Get agent performance report
    performance = session.get_agent_performance_report()
    for agent_name, metrics in performance.items():
        print(f"{agent_name} Performance:")
        print(f"  Response Quality: {metrics['response_quality']:.2f}")
        print(f"  Collaboration Score: {metrics['collaboration_score']:.2f}")
```

## Best Practices

### 1. Efficient Group Chat Monitoring

```python
# For large group chats, optimize configuration
large_group_config = {
    "monitor_conversations": True,
    "conversation_history_limit": 50,       # Limit history for performance
    "sampling_rate": 0.5,                   # Sample 50% of messages
    "batch_size": 100,                      # Larger batches
    "async_processing": True,               # Better performance
    "store_full_conversations": False,      # Reduce storage
}
```

### 2. Handling Long Conversations

```python
# For long-running conversations
long_conversation_config = {
    "conversation_history_limit": 200,      # Keep more history
    "enable_conversation_summarization": True,  # Summarize old parts
    "checkpoint_interval": 50,              # Save state every 50 messages
    "auto_archive_threshold": 1000,         # Archive after 1000 messages
}
```

### 3. Error Handling

```python
from escai_framework.exceptions import AutoGenInstrumentationError

try:
    with monitor_agent(agent_id, "autogen", config) as session:
        user_proxy.initiate_chat(assistant, message=user_message)
except AutoGenInstrumentationError as e:
    print(f"Monitoring error: {e}")
    # Fallback to unmonitored conversation
    user_proxy.initiate_chat(assistant, message=user_message)
except Exception as e:
    print(f"Conversation error: {e}")
    # Handle conversation errors
```

## Troubleshooting

### Common Issues

1. **Message Capture Issues**

   ```python
   # Ensure proper message handling
   config = {
       "monitor_conversations": True,
       "capture_message_metadata": True,
       "conversation_history_limit": 100  # Ensure sufficient limit
   }
   ```

2. **Group Chat Monitoring Problems**

   ```python
   # Verify all agents are properly tracked
   groupchat = autogen.GroupChat(
       agents=[user_proxy, assistant1, assistant2],
       messages=[],
       max_round=20,
       speaker_selection_method="auto"  # Ensure proper speaker selection
   )
   ```

3. **Performance Issues with Large Groups**
   ```python
   # Optimize for large groups
   config = {
       "sampling_rate": 0.3,           # Reduce sampling
       "batch_size": 200,              # Increase batch size
       "async_processing": True,       # Use async processing
       "store_full_conversations": False  # Reduce storage overhead
   }
   ```

### Performance Monitoring

```python
# Monitor conversation performance
import time

start_time = time.time()

with monitor_agent(agent_id, "autogen", config) as session:
    conversation_start = time.time()

    user_proxy.initiate_chat(assistant, message=user_message)

    conversation_end = time.time()

    conversation_time = conversation_end - conversation_start
    total_time = time.time() - start_time
    overhead = (total_time - conversation_time) / conversation_time * 100

    print(f"Conversation time: {conversation_time:.2f}s")
    print(f"Monitoring overhead: {overhead:.1f}%")

    # Get detailed performance metrics
    perf_metrics = session.get_performance_metrics()
    print(f"Messages processed: {perf_metrics['messages_processed']}")
    print(f"Processing rate: {perf_metrics['messages_per_second']:.1f} msg/s")
```

## Next Steps

1. **Explore Group Dynamics**: Experiment with different group compositions
2. **Custom Patterns**: Define patterns specific to your use cases
3. **Real-time Monitoring**: Set up live conversation monitoring
4. **Performance Optimization**: Tune configuration for your conversation patterns
5. **Integration**: Combine with other monitoring tools and dashboards

For more examples, see the [AutoGen examples directory](../../examples/autogen/) in the repository.
