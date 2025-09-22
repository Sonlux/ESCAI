# ESCAI CLI Comprehensive Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Quick Start](#quick-start)
4. [Command Reference](#command-reference)
5. [Workflows and Use Cases](#workflows-and-use-cases)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Introduction

The ESCAI CLI provides a comprehensive command-line interface for monitoring, analyzing, and understanding autonomous agent behavior. It offers both interactive menu-driven navigation and direct command execution for maximum flexibility.

### Key Features

- **Real-time Monitoring**: Track agent execution with live updates
- **Behavioral Analysis**: Identify patterns and causal relationships
- **Multi-framework Support**: Works with LangChain, AutoGen, CrewAI, and OpenAI
- **Publication-ready Output**: Generate academic-quality reports and visualizations
- **Session Management**: Track, replay, and compare monitoring sessions
- **Interactive Interface**: Menu-driven navigation for ease of use

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- One or more supported agent frameworks:
  - LangChain: `pip install langchain`
  - AutoGen: `pip install autogen`
  - CrewAI: `pip install crewai`
  - OpenAI: `pip install openai`

### Installation

```bash
# Install ESCAI Framework
pip install escai-framework

# Verify installation
escai --version
```

### Initial Configuration

```bash
# Run interactive setup
escai config setup

# Test configuration
escai config test

# Show current configuration
escai config show
```

## Quick Start

### 1. Start Monitoring

```bash
# Monitor a LangChain agent
escai monitor start --framework langchain --agent-id my_agent

# Monitor with custom configuration
escai monitor start --framework langchain --agent-id my_agent \
  --capture-epistemic --capture-behavioral --capture-performance
```

### 2. View Real-time Status

```bash
# Check monitoring status
escai monitor status

# Live dashboard
escai monitor live

# View detailed logs
escai monitor logs --agent-id my_agent
```

### 3. Analyze Behavior

```bash
# Analyze patterns
escai analyze patterns --agent-id my_agent

# Causal analysis
escai analyze causal --agent-id my_agent

# Generate visualizations
escai analyze visualize --type patterns --format heatmap
```

### 4. Generate Reports

```bash
# Statistical report
escai publication generate --type statistical --agent-id my_agent

# Academic paper format
escai publication generate --type academic --format latex
```

## Command Reference

### Monitor Commands

| Command          | Description               | Example                                                         |
| ---------------- | ------------------------- | --------------------------------------------------------------- |
| `monitor start`  | Start monitoring an agent | `escai monitor start --framework langchain --agent-id my_agent` |
| `monitor stop`   | Stop monitoring session   | `escai monitor stop --session-id abc123`                        |
| `monitor status` | View monitoring status    | `escai monitor status`                                          |
| `monitor live`   | Real-time dashboard       | `escai monitor live --agent-id my_agent`                        |
| `monitor logs`   | View monitoring logs      | `escai monitor logs --level info`                               |

### Analyze Commands

| Command             | Description                  | Example                                           |
| ------------------- | ---------------------------- | ------------------------------------------------- |
| `analyze patterns`  | Analyze behavioral patterns  | `escai analyze patterns --timeframe 24h`          |
| `analyze causal`    | Causal relationship analysis | `escai analyze causal --confidence-threshold 0.8` |
| `analyze stats`     | Statistical analysis         | `escai analyze stats --agent-id my_agent`         |
| `analyze visualize` | Generate visualizations      | `escai analyze visualize --type patterns`         |
| `analyze export`    | Export analysis results      | `escai analyze export --format json`              |

### Session Commands

| Command           | Description              | Example                                          |
| ----------------- | ------------------------ | ------------------------------------------------ |
| `session list`    | List monitoring sessions | `escai session list --active`                    |
| `session details` | Show session details     | `escai session details --session-id abc123`      |
| `session replay`  | Replay session           | `escai session replay --session-id abc123`       |
| `session export`  | Export session data      | `escai session export --format csv`              |
| `session compare` | Compare sessions         | `escai session compare --sessions abc123,def456` |

### Configuration Commands

| Command        | Description               | Example                                    |
| -------------- | ------------------------- | ------------------------------------------ |
| `config setup` | Interactive configuration | `escai config setup`                       |
| `config show`  | Show current config       | `escai config show`                        |
| `config test`  | Test configuration        | `escai config test --framework langchain`  |
| `config set`   | Set configuration value   | `escai config set database.host localhost` |
| `config reset` | Reset configuration       | `escai config reset --confirm`             |

## Workflows and Use Cases

### Basic Agent Monitoring

```bash
# 1. Configure ESCAI
escai config setup

# 2. Start monitoring
escai monitor start --framework langchain --agent-id research_agent

# 3. Let agent run for some time...

# 4. Check status
escai monitor status

# 5. Stop monitoring
escai monitor stop --agent-id research_agent
```

### Pattern Analysis Workflow

```bash
# 1. Start monitoring with pattern capture
escai monitor start --framework langchain --agent-id my_agent --capture-behavioral

# 2. Run agent for sufficient time (10+ minutes recommended)

# 3. Analyze patterns
escai analyze patterns --agent-id my_agent --timeframe 1h

# 4. Visualize patterns
escai analyze visualize --type patterns --format heatmap

# 5. Export results
escai analyze export --format json --output patterns.json
```

### Multi-Agent Comparison

```bash
# 1. Monitor multiple agents
escai monitor start --framework autogen --agent-id agent1
escai monitor start --framework autogen --agent-id agent2

# 2. Let agents run...

# 3. Compare performance
escai analyze compare --agents agent1,agent2

# 4. Generate comparative report
escai publication generate --type comparative --agents agent1,agent2
```

### Academic Research Workflow

```bash
# 1. Start comprehensive monitoring
escai monitor start --framework langchain --agent-id research_subject \
  --capture-epistemic --capture-behavioral --capture-performance

# 2. Run experiments...

# 3. Comprehensive analysis
escai analyze patterns --agent-id research_subject
escai analyze causal --agent-id research_subject
escai analyze stats --agent-id research_subject

# 4. Generate publication-ready output
escai publication generate --type academic --format latex \
  --include-citations --include-methodology

# 5. Export data for further analysis
escai session export --session-id <session> --format csv
```

## Advanced Features

### Interactive Mode

```bash
# Launch interactive menu system
escai --interactive

# Navigate using numbered options
# Use 'b' to go back, 'q' to quit
# Use 'h' for help at any menu level
```

### Debug Mode

```bash
# Enable debug mode for troubleshooting
escai --debug monitor start --framework langchain --agent-id my_agent

# View debug logs
escai logs show --level debug --recent
```

### Configuration Profiles

```bash
# Create named configuration profile
escai config-mgmt create-profile --name production

# Switch between profiles
escai config-mgmt switch-profile --name production

# Export profile for sharing
escai config-mgmt export-profile --name production --output prod-config.json
```

### Session Management

```bash
# Tag sessions for organization
escai session tag --session-id abc123 --tags experiment1,baseline

# Search sessions by tags
escai session search --tags experiment1

# Replay session with modifications
escai session replay --session-id abc123 --speed 2x --filter errors
```

## Troubleshooting

### Common Issues

#### Framework Not Found

```
Error: Framework 'langchain' is not available
```

**Solution:**

```bash
# Install the framework
pip install langchain

# Verify installation
escai config test --framework langchain
```

#### Database Connection Issues

```
Error: Cannot connect to database
```

**Solutions:**

```bash
# Reconfigure database
escai config setup --reset

# Test connection
escai config test

# Check database status
escai config check
```

#### No Monitoring Data

```
Warning: No monitoring data captured
```

**Solutions:**

```bash
# Ensure agent is running during monitoring
# Check framework integration
escai config test --framework <framework>

# Enable debug mode
escai --debug monitor start --framework <framework> --agent-id <agent>
```

### Performance Issues

#### Slow CLI Response

```bash
# Clear caches
escai session cleanup --cache

# Optimize database
escai config optimize

# Check system resources
escai logs system
```

#### High Memory Usage

```bash
# Reduce monitoring frequency
escai monitor start --max-events-per-second 10

# Limit buffer size
escai monitor start --buffer-size 1000

# Stop unnecessary sessions
escai session list --active
escai session stop --session-id <id>
```

### Getting Help

```bash
# Comprehensive help system
escai help

# Command-specific help
escai help monitor

# Topic help
escai help getting_started

# Search help content
escai help --search patterns

# Workflow guides
escai help workflow basic_monitoring
```

## Best Practices

### Monitoring

1. **Start monitoring before agent execution** to capture complete behavior
2. **Use descriptive agent IDs** for better organization
3. **Monitor for sufficient duration** (10+ minutes) for meaningful pattern detection
4. **Stop monitoring cleanly** to preserve all data
5. **Use appropriate capture settings** based on your analysis needs

### Analysis

1. **Collect sufficient data** before analysis (minimum 10 minutes of monitoring)
2. **Use appropriate timeframes** for pattern analysis
3. **Set confidence thresholds** to filter noise in causal analysis
4. **Combine multiple analysis types** for comprehensive insights
5. **Export results** for reproducibility and further analysis

### Performance

1. **Use session management** to organize long-running experiments
2. **Clean up old sessions** regularly to maintain performance
3. **Use configuration profiles** for different research contexts
4. **Monitor system resources** during intensive analysis
5. **Use sampling** for very large datasets

### Research and Publication

1. **Document methodology** using session metadata
2. **Include confidence intervals** in statistical reports
3. **Use consistent agent IDs** across related experiments
4. **Export raw data** alongside processed results
5. **Generate reproducible reports** with version information

### Collaboration

1. **Share configuration profiles** with team members
2. **Use consistent naming conventions** for agents and sessions
3. **Export session data** for collaborative analysis
4. **Document experimental procedures** in session tags
5. **Use version control** for configuration files

## Integration Examples

### LangChain Integration

```python
from langchain.agents import initialize_agent
from escai_framework.instrumentation.langchain_instrumentor import LangChainInstrumentor

# Initialize instrumentor
instrumentor = LangChainInstrumentor()

# Start monitoring
session_id = await instrumentor.start_monitoring(
    agent_id="langchain_agent",
    config={
        "capture_epistemic_states": True,
        "capture_behavioral_patterns": True
    }
)

# Create and run agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
result = agent.run("What is the weather like today?")

# Stop monitoring
await instrumentor.stop_monitoring(session_id)
```

### AutoGen Integration

```python
import autogen
from escai_framework.instrumentation.autogen_instrumentor import AutoGenInstrumentor

# Initialize instrumentor
instrumentor = AutoGenInstrumentor()

# Start monitoring
session_id = await instrumentor.start_monitoring(
    agent_id="autogen_group",
    config={
        "capture_conversations": True,
        "capture_agent_interactions": True
    }
)

# Create and run multi-agent system
user_proxy = autogen.UserProxyAgent("user_proxy")
assistant = autogen.AssistantAgent("assistant")

user_proxy.initiate_chat(assistant, message="Solve this problem...")

# Stop monitoring
await instrumentor.stop_monitoring(session_id)
```

## API Integration

The CLI can also be used programmatically through the API client:

```python
from escai_framework.cli.services.api_client import ESCAIAPIClient

# Initialize client
client = ESCAIAPIClient(base_url="http://localhost:8000")

# Start monitoring via API
result = await client.start_monitoring(
    agent_id="api_agent",
    framework="langchain",
    config={"capture_epistemic_states": True}
)

# Get agent status
status = await client.get_agent_status("api_agent")

# Analyze patterns
patterns = await client.analyze_patterns(agent_id="api_agent")
```

## Conclusion

The ESCAI CLI provides a comprehensive toolkit for monitoring and analyzing autonomous agent behavior. By following the workflows and best practices outlined in this guide, you can gain deep insights into agent cognition and generate publication-ready research outputs.

For additional help:

- Use `escai help` for interactive assistance
- Check the troubleshooting section for common issues
- Refer to the API documentation for programmatic usage
- Join the community forums for support and discussions
