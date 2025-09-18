# Framework Integration Troubleshooting

This guide helps resolve common issues when integrating ESCAI CLI with agent frameworks.

## Quick Diagnostics

### Check Framework Availability

```bash
# Validate all frameworks
escai monitor validate

# Validate specific framework
escai monitor validate --framework langchain
```

### Common Error Messages

#### "Framework 'X' is not available"

**Cause**: The framework is not installed or not properly configured.

**Solutions**:

1. Install the framework:

   ```bash
   pip install langchain    # For LangChain
   pip install pyautogen    # For AutoGen
   pip install crewai       # For CrewAI
   pip install openai       # For OpenAI
   ```

2. Verify installation:
   ```bash
   python -c "import langchain; print('LangChain OK')"
   python -c "import autogen; print('AutoGen OK')"
   python -c "import crewai; print('CrewAI OK')"
   python -c "import openai; print('OpenAI OK')"
   ```

#### "Failed to start monitoring"

**Cause**: Instrumentor initialization failed.

**Solutions**:

1. Check framework-specific requirements
2. Verify agent configuration
3. Check system resources

#### "Session not found"

**Cause**: Trying to operate on a non-existent monitoring session.

**Solutions**:

1. List active sessions:
   ```bash
   escai monitor status
   ```
2. Use correct session ID from the start command output

## Framework-Specific Issues

### LangChain Integration

#### Issue: "LangChain callbacks not working"

**Symptoms**:

- No events captured during chain execution
- Empty monitoring results

**Solutions**:

1. Ensure callback handler is properly attached:

   ```python
   from escai_framework.cli.integration import get_framework_connector

   # Get the callback handler
   connector = get_framework_connector()
   # Use with your LangChain chains
   ```

2. Check LangChain version compatibility:
   ```bash
   pip show langchain
   # Ensure version >= 0.0.200
   ```

#### Issue: "Memory tracking not working"

**Solutions**:

1. Enable memory monitoring in config:

   ```bash
   escai monitor start --agent-id my_agent --framework langchain --capture-performance
   ```

2. Check memory configuration:
   ```python
   config = {
       'monitor_memory': True,
       'monitor_context': True
   }
   ```

### AutoGen Integration

#### Issue: "Multi-agent conversations not captured"

**Symptoms**:

- Only seeing single agent events
- Missing conversation flow

**Solutions**:

1. Provide agents in configuration:

   ```python
   config = {
       'agents': [agent1, agent2, agent3],
       'monitor_conversations': True
   }
   ```

2. Ensure GroupChat is properly configured:
   ```python
   config = {
       'group_chats': [group_chat],
       'monitor_decisions': True
   }
   ```

#### Issue: "Speaker selection not monitored"

**Solutions**:

1. Enable group decision monitoring:

   ```bash
   escai monitor start --agent-id system --framework autogen
   ```

2. Check GroupChat instrumentation:
   ```python
   # Ensure GroupChat has select_speaker_msg method
   assert hasattr(group_chat, 'select_speaker_msg')
   ```

### CrewAI Integration

#### Issue: "Task delegation not visible"

**Solutions**:

1. Enable task monitoring:

   ```python
   config = {
       'monitor_task_delegation': True,
       'capture_behavioral_patterns': True
   }
   ```

2. Check CrewAI version:
   ```bash
   pip show crewai
   # Ensure compatible version
   ```

### OpenAI Integration

#### Issue: "Tool usage not captured"

**Solutions**:

1. Enable tool monitoring:

   ```python
   config = {
       'monitor_tool_usage': True,
       'capture_performance_metrics': True
   }
   ```

2. Check OpenAI client configuration:
   ```python
   # Ensure proper API key setup
   import openai
   assert openai.api_key is not None
   ```

## Performance Issues

### High Monitoring Overhead

**Symptoms**:

- Slow agent execution
- High CPU/memory usage
- Dropped events

**Solutions**:

1. Reduce event capture rate:

   ```python
   config = {
       'max_events_per_second': 50,  # Reduce from default 100
       'buffer_size': 500           # Reduce from default 1000
   }
   ```

2. Disable unnecessary monitoring:

   ```bash
   escai monitor start --agent-id my_agent --framework langchain \
     --no-capture-epistemic --no-capture-behavioral
   ```

3. Check system resources:
   ```bash
   # Monitor CPU and memory usage
   top -p $(pgrep -f escai)
   ```

### Memory Leaks

**Symptoms**:

- Increasing memory usage over time
- System slowdown during long monitoring sessions

**Solutions**:

1. Restart monitoring sessions periodically:

   ```bash
   # Stop all sessions
   escai monitor stop --all

   # Restart specific session
   escai monitor start --agent-id my_agent --framework langchain
   ```

2. Check for circular references:
   ```python
   # Use weak references in custom event handlers
   import weakref
   ```

## Network and Connectivity Issues

### API Connection Failures

**Symptoms**:

- "Failed to connect to ESCAI backend"
- Timeout errors

**Solutions**:

1. Check API service status:

   ```bash
   curl http://localhost:8000/health
   ```

2. Verify configuration:

   ```bash
   cat ~/.escai/config.json
   ```

3. Test network connectivity:
   ```bash
   ping localhost
   telnet localhost 8000
   ```

### WebSocket Connection Issues

**Solutions**:

1. Check firewall settings
2. Verify WebSocket support:
   ```bash
   # Test WebSocket connection
   wscat -c ws://localhost:8000/ws
   ```

## Configuration Issues

### Invalid Configuration

**Symptoms**:

- "Configuration validation failed"
- Missing required parameters

**Solutions**:

1. Use configuration templates:

   ```bash
   escai config show --template langchain
   ```

2. Validate configuration:

   ```python
   from escai_framework.cli.integration import FrameworkConnector

   connector = FrameworkConnector()
   connector._validate_monitoring_config(config, 'langchain')
   ```

### Environment Variables

**Required Environment Variables**:

```bash
# Optional: ESCAI API configuration
export ESCAI_API_HOST=localhost
export ESCAI_API_PORT=8000
export ESCAI_API_KEY=your_api_key

# Framework-specific variables
export OPENAI_API_KEY=your_openai_key
export LANGCHAIN_API_KEY=your_langchain_key
```

## Debugging Tools

### Enable Debug Logging

```bash
# Set debug level
export ESCAI_LOG_LEVEL=DEBUG

# Run with verbose output
escai --verbose monitor start --agent-id my_agent --framework langchain
```

### Capture Debug Information

```python
import logging

# Enable debug logging for ESCAI components
logging.getLogger('escai_framework').setLevel(logging.DEBUG)
logging.getLogger('escai_framework.cli').setLevel(logging.DEBUG)
logging.getLogger('escai_framework.instrumentation').setLevel(logging.DEBUG)
```

### Test Framework Integration

```python
# Test script for framework integration
import asyncio
from escai_framework.cli.integration import get_framework_connector

async def test_integration():
    connector = get_framework_connector()

    # Test each framework
    for framework in ['langchain', 'autogen', 'crewai', 'openai']:
        print(f"Testing {framework}...")
        result = await connector.validate_framework_integration(framework)

        if result['errors']:
            print(f"❌ {framework}: {result['errors']}")
        else:
            print(f"✅ {framework}: OK")

# Run test
asyncio.run(test_integration())
```

## Getting Help

### Collect Diagnostic Information

```bash
# System information
python --version
pip list | grep -E "(escai|langchain|autogen|crewai|openai)"

# ESCAI version and status
escai --version
escai monitor validate

# Active sessions
escai monitor status
```

### Report Issues

When reporting issues, include:

1. **System Information**:

   - Operating system and version
   - Python version
   - Framework versions

2. **Error Details**:

   - Complete error message
   - Stack trace
   - Steps to reproduce

3. **Configuration**:

   - Monitoring configuration
   - Environment variables
   - CLI command used

4. **Logs**:
   - Debug logs (with sensitive data removed)
   - System logs if relevant

### Community Resources

- **Documentation**: [ESCAI Framework Docs](https://escai-framework.readthedocs.io)
- **GitHub Issues**: [Report bugs and feature requests](https://github.com/escai-framework/escai/issues)
- **Discussions**: [Community discussions](https://github.com/escai-framework/escai/discussions)

## Best Practices

### Monitoring Setup

1. **Start Simple**: Begin with basic monitoring before enabling all features
2. **Test Locally**: Validate integration in development before production
3. **Monitor Resources**: Keep an eye on system resource usage
4. **Regular Cleanup**: Stop unused monitoring sessions

### Configuration Management

1. **Use Profiles**: Create named configuration profiles for different scenarios
2. **Version Control**: Store configurations in version control
3. **Environment Separation**: Use different configs for dev/staging/prod
4. **Security**: Keep API keys and sensitive data secure

### Performance Optimization

1. **Selective Monitoring**: Only monitor what you need
2. **Batch Processing**: Use appropriate buffer sizes
3. **Resource Limits**: Set reasonable limits on event processing
4. **Regular Maintenance**: Clean up old sessions and data
