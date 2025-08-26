# ESCAI Framework Integration Guide

This guide provides step-by-step instructions for integrating ESCAI monitoring with different agent frameworks.

## Supported Frameworks

- [LangChain](#langchain-integration)
- [AutoGen](#autogen-integration)
- [CrewAI](#crewai-integration)
- [OpenAI Assistants](#openai-assistants-integration)

## Prerequisites

Before integrating with any framework, ensure you have:

1. **ESCAI Framework installed**:

   ```bash
   pip install escai-framework
   ```

2. **ESCAI server running**:

   ```bash
   escai server start
   ```

3. **Authentication credentials** configured

## Quick Integration Checklist

For any framework integration:

- [ ] Install ESCAI Framework
- [ ] Import the appropriate instrumentor
- [ ] Initialize monitoring with your agent configuration
- [ ] Start your agent with ESCAI monitoring enabled
- [ ] Access real-time insights through the API or CLI

## Framework-Specific Guides

### [LangChain Integration](./langchain.md)

Monitor LangChain agents, chains, and tools with comprehensive reasoning trace capture.

### [AutoGen Integration](./autogen.md)

Track multi-agent conversations, role assignments, and group decision-making processes.

### [CrewAI Integration](./crewai.md)

Monitor crew workflows, task delegation, and collaborative agent interactions.

### [OpenAI Assistants Integration](./openai.md)

Observe OpenAI Assistant interactions, function calls, and thread conversations.

## Common Integration Patterns

### 1. Basic Monitoring Setup

```python
from escai_framework import ESCAIClient
from escai_framework.instrumentation import get_instrumentor

# Initialize ESCAI client
escai_client = ESCAIClient(
    base_url="http://localhost:8000",
    username="your_username",
    password="your_password"
)

# Get framework-specific instrumentor
instrumentor = get_instrumentor("langchain")  # or "autogen", "crewai", "openai"

# Start monitoring
session = escai_client.start_monitoring(
    agent_id="my-agent-001",
    framework="langchain",
    config={
        "capture_reasoning": True,
        "monitor_memory": True,
        "sampling_rate": 1.0
    }
)

# Your agent code here
# ...

# Stop monitoring
summary = escai_client.stop_monitoring(session.session_id)
```

### 2. Context Manager Pattern

```python
from escai_framework import monitor_agent

# Automatic session management
with monitor_agent(
    agent_id="my-agent-001",
    framework="langchain",
    config={"capture_reasoning": True}
) as session:
    # Your agent execution code
    result = your_agent.run(task)

    # Access real-time insights
    current_state = session.get_current_epistemic_state()
    print(f"Agent confidence: {current_state.confidence_level}")

# Monitoring automatically stopped
```

### 3. Async Integration

```python
import asyncio
from escai_framework import AsyncESCAIClient

async def monitor_async_agent():
    client = AsyncESCAIClient(
        base_url="http://localhost:8000",
        username="your_username",
        password="your_password"
    )

    session = await client.start_monitoring(
        agent_id="async-agent-001",
        framework="langchain",
        config={"capture_reasoning": True}
    )

    # Stream real-time updates
    async for update in client.stream_updates(session.session_id):
        if update.type == "epistemic_update":
            print(f"New state: {update.payload.confidence_level}")
        elif update.type == "alert":
            print(f"Alert: {update.payload.message}")

    await client.stop_monitoring(session.session_id)

# Run async monitoring
asyncio.run(monitor_async_agent())
```

## Configuration Options

### Common Configuration Parameters

All framework instrumentors support these common configuration options:

```python
config = {
    # Core monitoring settings
    "capture_reasoning": True,      # Capture reasoning traces
    "monitor_memory": True,         # Track memory usage
    "monitor_tools": True,          # Monitor tool usage
    "capture_errors": True,         # Capture error states

    # Performance settings
    "sampling_rate": 1.0,           # 0.0-1.0, fraction of events to capture
    "batch_size": 100,              # Events per batch for processing
    "buffer_size": 1000,            # Maximum events in buffer

    # Filtering settings
    "event_filters": [              # Only capture specific event types
        "decision_made",
        "tool_used",
        "error_occurred"
    ],
    "exclude_patterns": [           # Exclude events matching patterns
        "debug_*",
        "internal_*"
    ],

    # Analysis settings
    "enable_pattern_detection": True,   # Real-time pattern detection
    "enable_causal_analysis": True,     # Causal relationship discovery
    "enable_predictions": True,         # Performance predictions

    # Storage settings
    "retention_days": 30,           # How long to keep data
    "compress_logs": True,          # Compress stored logs

    # Alert settings
    "alert_thresholds": {
        "low_confidence": 0.3,      # Alert when confidence drops below
        "high_uncertainty": 0.8,    # Alert when uncertainty exceeds
        "pattern_anomaly": 0.9      # Alert on anomalous patterns
    }
}
```

### Framework-Specific Configuration

Each framework may have additional specific options:

#### LangChain

```python
langchain_config = {
    "capture_chain_steps": True,        # Capture individual chain steps
    "monitor_llm_calls": True,          # Monitor LLM API calls
    "track_token_usage": True,          # Track token consumption
    "capture_retrieval": True,          # Monitor retrieval operations
    "memory_tracking_depth": 5          # How many memory items to track
}
```

#### AutoGen

```python
autogen_config = {
    "monitor_conversations": True,      # Track agent conversations
    "capture_role_changes": True,       # Monitor role assignments
    "track_group_decisions": True,      # Monitor group decision processes
    "conversation_history_limit": 100   # Max conversation items to track
}
```

#### CrewAI

```python
crewai_config = {
    "monitor_task_delegation": True,    # Track task assignments
    "capture_crew_coordination": True,  # Monitor crew interactions
    "track_skill_usage": True,          # Monitor skill utilization
    "workflow_depth_limit": 10          # Max workflow depth to track
}
```

#### OpenAI Assistants

```python
openai_config = {
    "monitor_function_calls": True,     # Track function calls
    "capture_thread_context": True,     # Monitor thread conversations
    "track_tool_usage": True,           # Monitor tool interactions
    "thread_history_limit": 50          # Max thread messages to track
}
```

## Best Practices

### 1. Performance Optimization

- **Use appropriate sampling rates** for production environments
- **Filter events** to capture only what you need
- **Batch processing** for high-throughput scenarios

```python
# Production configuration
production_config = {
    "sampling_rate": 0.1,           # Sample 10% of events
    "event_filters": [              # Only critical events
        "error_occurred",
        "task_completed",
        "decision_made"
    ],
    "batch_size": 500,              # Larger batches for efficiency
    "enable_predictions": False     # Disable expensive predictions
}
```

### 2. Error Handling

Always implement proper error handling:

```python
from escai_framework.exceptions import (
    ESCAIConnectionError,
    ESCAIAuthenticationError,
    ESCAIConfigurationError
)

try:
    session = escai_client.start_monitoring(agent_id, framework, config)
except ESCAIConnectionError:
    print("Cannot connect to ESCAI server")
    # Fallback to non-monitored execution
except ESCAIAuthenticationError:
    print("Authentication failed")
    # Handle authentication issues
except ESCAIConfigurationError as e:
    print(f"Configuration error: {e}")
    # Fix configuration and retry
```

### 3. Resource Management

Properly manage monitoring resources:

```python
# Use try/finally for cleanup
session = None
try:
    session = escai_client.start_monitoring(agent_id, framework, config)
    # Your agent code
    result = agent.run(task)
finally:
    if session:
        escai_client.stop_monitoring(session.session_id)

# Or use context managers (recommended)
with monitor_agent(agent_id, framework, config) as session:
    result = agent.run(task)
    # Automatic cleanup
```

### 4. Testing Integration

Test your integration thoroughly:

```python
import pytest
from escai_framework.testing import MockESCAIClient

def test_agent_with_monitoring():
    # Use mock client for testing
    with MockESCAIClient() as mock_client:
        session = mock_client.start_monitoring(
            agent_id="test-agent",
            framework="langchain",
            config={"capture_reasoning": True}
        )

        # Run your agent
        result = your_agent.run("test task")

        # Verify monitoring captured events
        events = mock_client.get_captured_events(session.session_id)
        assert len(events) > 0
        assert any(e.type == "decision_made" for e in events)
```

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```python
   # Wrong
   from escai_framework.langchain import LangChainInstrumentor

   # Correct
   from escai_framework.instrumentation import get_instrumentor
   instrumentor = get_instrumentor("langchain")
   ```

2. **Configuration Errors**

   ```python
   # Validate configuration before use
   from escai_framework.utils import validate_config

   config = {"capture_reasoning": True}
   if validate_config(config, framework="langchain"):
       session = client.start_monitoring(agent_id, "langchain", config)
   ```

3. **Performance Issues**

   ```python
   # Monitor overhead
   import time

   start_time = time.time()
   # Your agent code
   end_time = time.time()

   overhead = session.get_monitoring_overhead()
   print(f"Monitoring overhead: {overhead:.2%}")
   ```

### Getting Help

- Check the framework-specific integration guides
- Review the [troubleshooting guide](../troubleshooting/README.md)
- Join our [community forum](https://community.escai.dev)
- Report issues on [GitHub](https://github.com/escai-framework/escai/issues)

## Next Steps

1. Choose your framework and follow the specific integration guide
2. Start with basic monitoring configuration
3. Gradually add more advanced features as needed
4. Monitor performance and adjust configuration
5. Explore the [visualization dashboard](../visualization/README.md) for insights
