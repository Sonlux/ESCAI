# ESCAI Framework Examples

This directory contains comprehensive examples demonstrating how to use the ESCAI Framework with different agent frameworks and real-world scenarios.

## Quick Start Examples

### Basic Monitoring

- [Basic Agent Monitoring](./basic_monitoring.py) - Simple monitoring setup
- [Configuration Examples](./configuration_examples.py) - Different configuration options
- [Error Handling](./error_handling_examples.py) - Proper error handling patterns

### Framework Integration

- [LangChain Examples](./langchain/) - LangChain integration examples
- [AutoGen Examples](./autogen/) - AutoGen multi-agent examples
- [CrewAI Examples](./crewai/) - CrewAI workflow examples
- [OpenAI Assistants Examples](./openai/) - OpenAI Assistants integration

## Real-World Scenarios

### Business Applications

- [Customer Service Bot](./business/customer_service_bot.py) - Customer support automation
- [Data Analysis Agent](./business/data_analysis_agent.py) - Business intelligence automation
- [Content Generation Pipeline](./business/content_generation.py) - Automated content creation
- [Research Assistant](./business/research_assistant.py) - Research and analysis automation

### Technical Applications

- [Code Review Agent](./technical/code_review_agent.py) - Automated code review
- [DevOps Assistant](./technical/devops_assistant.py) - Infrastructure management
- [Testing Coordinator](./technical/testing_coordinator.py) - Test automation coordination
- [Documentation Generator](./technical/documentation_generator.py) - Automated documentation

### Educational Applications

- [Tutoring System](./educational/tutoring_system.py) - Personalized tutoring
- [Study Group Facilitator](./educational/study_group.py) - Group learning coordination
- [Assessment Generator](./educational/assessment_generator.py) - Automated assessment creation

## Advanced Examples

### Multi-Agent Workflows

- [Project Management Team](./advanced/project_management_team.py) - Complex project coordination
- [Research Collaboration](./advanced/research_collaboration.py) - Multi-agent research
- [Decision Making Committee](./advanced/decision_committee.py) - Group decision processes

### Custom Integrations

- [Custom Framework Integration](./advanced/custom_framework.py) - Integrating custom agent frameworks
- [External Tool Integration](./advanced/external_tools.py) - Integrating external APIs and tools
- [Workflow Orchestration](./advanced/workflow_orchestration.py) - Complex workflow management

### Performance and Scaling

- [High-Throughput Monitoring](./performance/high_throughput.py) - Monitoring many agents
- [Real-time Analytics](./performance/realtime_analytics.py) - Live performance monitoring
- [Load Testing](./performance/load_testing.py) - Testing monitoring overhead

## Interactive Demos

### Jupyter Notebooks

- [Getting Started Notebook](./notebooks/getting_started.ipynb) - Interactive introduction
- [Pattern Analysis Demo](./notebooks/pattern_analysis.ipynb) - Behavioral pattern exploration
- [Causal Analysis Demo](./notebooks/causal_analysis.ipynb) - Causal relationship discovery
- [Prediction Models Demo](./notebooks/prediction_models.ipynb) - Performance prediction examples

### Web Applications

- [Monitoring Dashboard](./web/monitoring_dashboard/) - Real-time monitoring interface
- [Agent Comparison Tool](./web/agent_comparison/) - Compare agent performance
- [Pattern Explorer](./web/pattern_explorer/) - Interactive pattern analysis

## Running the Examples

### Prerequisites

1. **Install ESCAI Framework**:

   ```bash
   pip install escai-framework
   ```

2. **Start ESCAI Server**:

   ```bash
   escai server start
   ```

3. **Set up Authentication**:
   ```bash
   escai auth login --username your_username --password your_password
   ```

### Basic Example

```python
# Run a basic monitoring example
python examples/basic_monitoring.py
```

### Framework-Specific Examples

```bash
# LangChain examples
python examples/langchain/simple_chain_monitoring.py
python examples/langchain/agent_with_tools.py

# AutoGen examples
python examples/autogen/group_chat_monitoring.py
python examples/autogen/decision_making_team.py

# CrewAI examples
python examples/crewai/workflow_monitoring.py
python examples/crewai/task_delegation.py

# OpenAI Assistants examples
python examples/openai/assistant_monitoring.py
python examples/openai/function_calling.py
```

### Real-World Scenarios

```bash
# Business applications
python examples/business/customer_service_bot.py
python examples/business/data_analysis_agent.py

# Technical applications
python examples/technical/code_review_agent.py
python examples/technical/devops_assistant.py
```

## Example Structure

Each example follows a consistent structure:

```python
"""
Example: [Example Name]
Description: [Brief description of what this example demonstrates]
Framework: [Agent framework used]
Complexity: [Basic/Intermediate/Advanced]
"""

import os
from escai_framework import monitor_agent
# Other imports...

def main():
    """Main example function."""

    # Configuration
    config = {
        # Example-specific configuration
    }

    # Set up agent/framework components
    # ...

    # Monitor execution
    with monitor_agent(
        agent_id="example-agent",
        framework="framework_name",
        config=config
    ) as session:

        # Execute agent task
        result = agent.run(task)

        # Analyze results
        analysis = session.get_analysis()
        print_results(analysis)

    return result

def print_results(analysis):
    """Print analysis results in a readable format."""
    print("=== ESCAI Analysis Results ===")
    print(f"Confidence Level: {analysis.confidence_level:.2f}")
    print(f"Execution Time: {analysis.execution_time_ms}ms")
    # More analysis output...

if __name__ == "__main__":
    main()
```

## Configuration Examples

### Development Configuration

```python
dev_config = {
    "capture_reasoning": True,
    "monitor_memory": True,
    "sampling_rate": 1.0,
    "enable_pattern_detection": True,
    "enable_predictions": True,
    "alert_thresholds": {
        "low_confidence": 0.3,
        "high_uncertainty": 0.8
    }
}
```

### Production Configuration

```python
prod_config = {
    "capture_reasoning": False,      # Reduce overhead
    "monitor_memory": False,         # Skip detailed memory tracking
    "sampling_rate": 0.1,           # Sample 10% of events
    "enable_pattern_detection": True, # Keep pattern detection
    "enable_predictions": False,     # Disable expensive predictions
    "batch_size": 500,              # Larger batches for efficiency
    "async_processing": True        # Better performance
}
```

### Testing Configuration

```python
test_config = {
    "capture_reasoning": True,
    "monitor_memory": True,
    "sampling_rate": 1.0,
    "store_full_conversations": True,
    "enable_detailed_logging": True,
    "mock_external_services": True   # For testing
}
```

## Best Practices Demonstrated

### 1. Error Handling

All examples demonstrate proper error handling:

```python
from escai_framework.exceptions import ESCAIError

try:
    with monitor_agent(agent_id, framework, config) as session:
        result = agent.run(task)
except ESCAIError as e:
    print(f"Monitoring error: {e}")
    # Fallback to unmonitored execution
    result = agent.run(task)
```

### 2. Resource Management

Examples show proper resource cleanup:

```python
# Using context managers (recommended)
with monitor_agent(agent_id, framework, config) as session:
    result = agent.run(task)
    # Automatic cleanup

# Manual management (when context managers aren't suitable)
session = None
try:
    session = client.start_monitoring(agent_id, framework, config)
    result = agent.run(task)
finally:
    if session:
        client.stop_monitoring(session.session_id)
```

### 3. Performance Monitoring

Examples include performance monitoring:

```python
import time

start_time = time.time()
with monitor_agent(agent_id, framework, config) as session:
    result = agent.run(task)

    # Check monitoring overhead
    overhead = session.get_monitoring_overhead()
    if overhead > 0.1:  # 10% threshold
        print(f"Warning: High monitoring overhead ({overhead:.1%})")
```

### 4. Configuration Validation

Examples validate configuration before use:

```python
from escai_framework.utils import validate_config

if not validate_config(config, framework="langchain"):
    print("Invalid configuration, using defaults")
    config = get_default_config("langchain")
```

## Testing Examples

Each example includes test cases:

```python
import pytest
from escai_framework.testing import MockESCAIClient

def test_example():
    """Test the example with mock client."""
    with MockESCAIClient() as mock_client:
        # Run example with mock client
        result = run_example_with_client(mock_client)

        # Verify results
        assert result is not None

        # Verify monitoring captured events
        events = mock_client.get_captured_events()
        assert len(events) > 0
```

## Contributing Examples

We welcome contributions of new examples! Please follow these guidelines:

1. **Use the standard example structure**
2. **Include comprehensive documentation**
3. **Add error handling and resource management**
4. **Include test cases**
5. **Follow the existing naming conventions**

### Example Template

```python
"""
Example: [Your Example Name]
Description: [What this example demonstrates]
Framework: [Agent framework used]
Complexity: [Basic/Intermediate/Advanced]
Author: [Your name]
"""

# Your example code here
```

## Getting Help

- **Documentation**: Check the [main documentation](../README.md)
- **Issues**: Report problems on [GitHub Issues](https://github.com/escai-framework/escai/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/escai-framework/escai/discussions)
- **Community**: Visit our [community forum](https://community.escai.dev)

## Next Steps

1. **Start with Basic Examples**: Try the basic monitoring examples first
2. **Choose Your Framework**: Focus on examples for your agent framework
3. **Explore Real-World Scenarios**: Look at business or technical applications
4. **Advanced Features**: Try multi-agent workflows and custom integrations
5. **Performance Testing**: Test monitoring overhead with your workloads
