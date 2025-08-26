# ESCAI Framework API Reference

The ESCAI Framework provides a comprehensive REST API and WebSocket interface for monitoring autonomous agent cognition in real-time.

## Quick Start

### Authentication

All API endpoints require JWT authentication. First, obtain a token:

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

Response:

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Basic Usage

```python
import requests

# Set up authentication
headers = {"Authorization": "Bearer YOUR_JWT_TOKEN"}

# Start monitoring an agent
response = requests.post(
    "http://localhost:8000/api/v1/monitor/start",
    json={
        "agent_id": "my-langchain-agent",
        "framework": "langchain",
        "config": {
            "capture_reasoning": True,
            "monitor_memory": True,
            "sampling_rate": 1.0
        }
    },
    headers=headers
)

session_id = response.json()["session_id"]
print(f"Monitoring session started: {session_id}")

# Get current epistemic state
response = requests.get(
    f"http://localhost:8000/api/v1/epistemic/my-langchain-agent/current",
    headers=headers
)

epistemic_state = response.json()
print(f"Agent confidence: {epistemic_state['confidence_level']}")
print(f"Current goals: {epistemic_state['goal_state']['primary_goal']}")
```

## API Endpoints Overview

### Monitoring Endpoints

| Endpoint                              | Method | Description                   |
| ------------------------------------- | ------ | ----------------------------- |
| `/api/v1/monitor/start`               | POST   | Start monitoring an agent     |
| `/api/v1/monitor/{session_id}/status` | GET    | Get monitoring session status |
| `/api/v1/monitor/{session_id}/stop`   | POST   | Stop monitoring session       |

### Analysis Endpoints

| Endpoint                                 | Method | Description                 |
| ---------------------------------------- | ------ | --------------------------- |
| `/api/v1/epistemic/{agent_id}/current`   | GET    | Get current epistemic state |
| `/api/v1/patterns/{agent_id}/analyze`    | GET    | Analyze behavioral patterns |
| `/api/v1/causal/analyze`                 | POST   | Perform causal analysis     |
| `/api/v1/predictions/{agent_id}/current` | GET    | Get performance predictions |
| `/api/v1/explain/{agent_id}/behavior`    | GET    | Get behavior explanations   |

## WebSocket API

### Real-time Monitoring

Connect to the WebSocket endpoint for real-time updates:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/monitor/SESSION_ID");

ws.onmessage = function (event) {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "epistemic_update":
      console.log("New epistemic state:", data.payload);
      break;
    case "pattern_detected":
      console.log("Pattern detected:", data.payload);
      break;
    case "prediction_update":
      console.log("New prediction:", data.payload);
      break;
    case "alert":
      console.log("Alert:", data.payload);
      break;
  }
};
```

### WebSocket Message Types

- `epistemic_update`: Real-time epistemic state changes
- `pattern_detected`: New behavioral patterns identified
- `prediction_update`: Updated performance predictions
- `alert`: Important notifications (anomalies, failures)
- `heartbeat`: Connection health check

## Error Handling

The API uses standard HTTP status codes and provides detailed error information:

```json
{
  "error": "ValidationError",
  "message": "Invalid agent_id format",
  "details": {
    "field": "agent_id",
    "expected": "string with length 3-50",
    "received": "ab"
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Rate Limiting

API endpoints are rate-limited to ensure system stability:

- **Authentication**: 10 requests per minute
- **Monitoring**: 100 requests per minute
- **Analysis**: 50 requests per minute
- **WebSocket**: 1000 messages per minute

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Data Models

### EpistemicState

Represents an agent's cognitive state at a point in time:

```json
{
  "agent_id": "my-agent-001",
  "timestamp": "2024-01-15T10:30:00Z",
  "belief_states": [
    {
      "belief_id": "belief_001",
      "content": "The user wants to analyze sales data",
      "confidence": 0.95,
      "evidence": ["user_input", "context_analysis"],
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "knowledge_state": {
    "facts": ["Sales data is in CSV format", "Data contains 12 months"],
    "concepts": ["data_analysis", "sales_metrics"],
    "relationships": [
      {
        "subject": "sales_data",
        "predicate": "contains",
        "object": "monthly_totals"
      }
    ]
  },
  "goal_state": {
    "primary_goal": "Generate sales analysis report",
    "sub_goals": ["Load data", "Clean data", "Calculate metrics"],
    "progress": 0.3,
    "status": "active"
  },
  "confidence_level": 0.87,
  "uncertainty_score": 0.13,
  "decision_context": {
    "task_complexity": "medium",
    "available_tools": ["pandas", "matplotlib"],
    "time_constraints": "none"
  }
}
```

### BehavioralPattern

Represents identified patterns in agent behavior:

```json
{
  "pattern_id": "pattern_001",
  "pattern_name": "Data Analysis Workflow",
  "execution_sequences": [
    "load_data -> validate_data -> analyze_data -> generate_report"
  ],
  "frequency": 15,
  "success_rate": 0.93,
  "average_duration": 45000,
  "common_triggers": ["data_analysis_request", "csv_file_provided"],
  "failure_modes": ["invalid_data_format", "missing_columns"],
  "statistical_significance": 0.95
}
```

### CausalRelationship

Represents discovered causal relationships:

```json
{
  "cause_event": "data_validation_failed",
  "effect_event": "error_handling_triggered",
  "strength": 0.89,
  "confidence": 0.92,
  "delay_ms": 150,
  "evidence": ["temporal_correlation", "logical_dependency"],
  "statistical_significance": 0.98,
  "causal_mechanism": "Exception propagation through validation pipeline"
}
```

## Interactive Examples

### Python SDK

```python
from escai_framework import ESCAIClient

# Initialize client
client = ESCAIClient(
    base_url="http://localhost:8000",
    username="your_username",
    password="your_password"
)

# Start monitoring
session = client.start_monitoring(
    agent_id="my-agent",
    framework="langchain",
    config={
        "capture_reasoning": True,
        "monitor_memory": True
    }
)

# Get real-time updates
for update in client.stream_updates(session.session_id):
    if update.type == "epistemic_update":
        print(f"Confidence: {update.payload.confidence_level}")
    elif update.type == "alert":
        print(f"Alert: {update.payload.message}")

# Analyze patterns
patterns = client.analyze_patterns(
    agent_id="my-agent",
    time_window="24h"
)

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_name}")
    print(f"Success rate: {pattern.success_rate}")

# Stop monitoring
summary = client.stop_monitoring(session.session_id)
print(f"Captured {summary.total_events} events")
```

### JavaScript/Node.js

```javascript
const ESCAIClient = require("escai-framework-js");

const client = new ESCAIClient({
  baseUrl: "http://localhost:8000",
  username: "your_username",
  password: "your_password",
});

// Start monitoring
const session = await client.startMonitoring({
  agentId: "my-agent",
  framework: "openai",
  config: {
    captureReasoning: true,
    monitorTools: true,
  },
});

// Real-time monitoring
client.onEpistemicUpdate((state) => {
  console.log(`Agent confidence: ${state.confidenceLevel}`);
});

client.onPatternDetected((pattern) => {
  console.log(`New pattern: ${pattern.patternName}`);
});

// Get predictions
const predictions = await client.getPredictions("my-agent");
predictions.forEach((pred) => {
  console.log(`${pred.predictionType}: ${pred.predictedValue}`);
});
```

## Best Practices

### 1. Efficient Monitoring

- Use appropriate sampling rates to balance insight and performance
- Monitor only necessary events to reduce overhead
- Use time windows effectively for analysis queries

```python
# Good: Targeted monitoring
config = {
    "capture_reasoning": True,
    "monitor_memory": False,  # Skip if not needed
    "sampling_rate": 0.1,     # Sample 10% of events
    "event_filters": ["decision", "error", "completion"]
}

# Avoid: Over-monitoring
config = {
    "capture_everything": True,  # Too much overhead
    "sampling_rate": 1.0,        # 100% sampling
    "no_filters": True           # Captures noise
}
```

### 2. Error Handling

Always implement proper error handling:

```python
try:
    session = client.start_monitoring(agent_id, framework, config)
except ESCAIAuthenticationError:
    # Handle authentication issues
    client.refresh_token()
    session = client.start_monitoring(agent_id, framework, config)
except ESCAIRateLimitError as e:
    # Handle rate limiting
    time.sleep(e.retry_after)
    session = client.start_monitoring(agent_id, framework, config)
except ESCAIValidationError as e:
    # Handle validation errors
    print(f"Invalid config: {e.details}")
```

### 3. Resource Management

Properly manage monitoring sessions:

```python
# Use context managers
with client.monitor_agent(agent_id, framework, config) as session:
    # Monitoring is active
    for update in client.stream_updates(session.session_id):
        process_update(update)
# Monitoring automatically stopped

# Or manual management
session = client.start_monitoring(agent_id, framework, config)
try:
    # Your monitoring logic
    pass
finally:
    client.stop_monitoring(session.session_id)
```

## Troubleshooting

### Common Issues

1. **Authentication Failures**

   - Check token expiration
   - Verify credentials
   - Ensure proper header format

2. **Rate Limiting**

   - Implement exponential backoff
   - Use appropriate request intervals
   - Consider upgrading plan limits

3. **WebSocket Disconnections**

   - Implement reconnection logic
   - Handle connection state properly
   - Use heartbeat monitoring

4. **High Monitoring Overhead**
   - Reduce sampling rate
   - Filter unnecessary events
   - Use batch processing

### Getting Help

- **Documentation**: [https://docs.escai.dev](https://docs.escai.dev)
- **GitHub Issues**: [https://github.com/escai-framework/escai/issues](https://github.com/escai-framework/escai/issues)
- **Community Forum**: [https://community.escai.dev](https://community.escai.dev)
- **Support Email**: support@escai.dev

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:

- **YAML**: [openapi.yaml](./openapi.yaml)
- **Interactive Docs**: http://localhost:8000/docs (when server is running)
- **ReDoc**: http://localhost:8000/redoc (alternative documentation view)
