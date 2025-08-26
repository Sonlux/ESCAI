# ESCAI Framework Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the ESCAI Framework.

## Quick Diagnostics

### System Health Check

```bash
# Check overall system status
escai status

# Detailed health check
escai health --detailed

# Check specific components
escai health --component api
escai health --component database
escai health --component monitoring
```

### Log Analysis

```bash
# View recent logs
escai logs --tail 100

# Filter by severity
escai logs --level ERROR --tail 50

# Component-specific logs
escai logs --component api --tail 100
escai logs --component instrumentation --tail 100

# Real-time log monitoring
escai logs --follow
```

## Common Issues and Solutions

### 1. Installation and Setup Issues

#### Issue: `pip install escai-framework` fails

**Symptoms:**

- Package not found errors
- Dependency conflicts
- Permission errors

**Solutions:**

```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Use virtual environment
python -m venv escai-env
source escai-env/bin/activate  # On Windows: escai-env\Scripts\activate
pip install escai-framework

# Install from source if package not available
git clone https://github.com/escai-framework/escai.git
cd escai
pip install -e .

# Fix permission issues (Linux/Mac)
pip install --user escai-framework

# Clear pip cache if corrupted
pip cache purge
pip install escai-framework
```

#### Issue: Database connection failures during setup

**Symptoms:**

- "Connection refused" errors
- "Authentication failed" errors
- Timeout errors

**Solutions:**

```bash
# Check if databases are running
docker-compose ps

# Start databases if not running
docker-compose up -d postgres mongodb redis influxdb neo4j

# Check database connectivity
escai config test-connections

# Reset database passwords
docker-compose down
docker volume rm escai_postgres_data escai_mongodb_data
docker-compose up -d

# Manual connection test
psql -h localhost -U escai_user -d escai_dev
mongo localhost:27017/escai_logs
redis-cli ping
```

### 2. API and Server Issues

#### Issue: API server won't start

**Symptoms:**

- "Port already in use" errors
- "Permission denied" errors
- Server crashes on startup

**Solutions:**

```bash
# Check if port is in use
netstat -tulpn | grep :8000
lsof -i :8000

# Kill process using the port
kill -9 $(lsof -t -i:8000)

# Use different port
escai server start --port 8001

# Check configuration
escai config validate

# Start with debug mode
escai server start --debug --reload

# Check permissions
sudo chown -R $USER:$USER ~/.escai/
chmod 755 ~/.escai/
```

#### Issue: API requests timing out

**Symptoms:**

- Slow response times
- Gateway timeout errors
- Connection timeouts

**Solutions:**

```bash
# Check server load
escai status --performance

# Increase timeout settings
export ESCAI_REQUEST_TIMEOUT=60
export ESCAI_DB_TIMEOUT=30

# Scale up workers
escai server start --workers 8

# Check database performance
escai health --component database --detailed

# Monitor resource usage
top -p $(pgrep -f escai)
```

#### Issue: Authentication failures

**Symptoms:**

- "Invalid credentials" errors
- "Token expired" errors
- "Unauthorized" responses

**Solutions:**

```bash
# Check authentication configuration
escai config show auth

# Reset authentication
escai auth reset

# Generate new JWT secret
escai auth generate-secret

# Login with correct credentials
escai auth login --username admin --password your_password

# Check token expiration
escai auth status

# Refresh token
escai auth refresh
```

### 3. Monitoring and Instrumentation Issues

#### Issue: Agent monitoring not working

**Symptoms:**

- No events captured
- Empty monitoring sessions
- Instrumentation errors

**Solutions:**

```python
# Check instrumentor registration
from escai_framework.instrumentation import get_instrumentor

try:
    instrumentor = get_instrumentor("langchain")
    print("Instrumentor loaded successfully")
except Exception as e:
    print(f"Instrumentor error: {e}")

# Verify framework integration
import langchain
print(f"LangChain version: {langchain.__version__}")

# Test basic monitoring
from escai_framework import monitor_agent

with monitor_agent("test-agent", "langchain", {"sampling_rate": 1.0}) as session:
    print(f"Session ID: {session.session_id}")
    # Your agent code here
```

```bash
# Check monitoring configuration
escai config show monitoring

# Test monitoring connectivity
escai monitor test --framework langchain

# Check for framework compatibility
escai monitor check-compatibility --framework langchain

# Enable debug logging for monitoring
export ESCAI_LOG_LEVEL=DEBUG
escai server start
```

#### Issue: High monitoring overhead

**Symptoms:**

- Slow agent execution
- High CPU/memory usage
- Performance degradation

**Solutions:**

```python
# Reduce sampling rate
config = {
    "sampling_rate": 0.1,  # Sample 10% of events
    "batch_size": 500,     # Larger batches
    "async_processing": True,
    "capture_reasoning": False,  # Disable expensive features
    "monitor_memory": False
}

# Use performance monitoring
import time

start_time = time.time()
with monitor_agent("agent", "framework", config) as session:
    # Your code
    pass
overhead = session.get_monitoring_overhead()
print(f"Overhead: {overhead:.1%}")
```

```bash
# Monitor system resources
escai monitor performance --agent-id your-agent

# Optimize configuration
escai config optimize --for-performance

# Check for memory leaks
escai monitor memory-usage --detailed
```

### 4. Database Issues

#### Issue: PostgreSQL connection problems

**Symptoms:**

- "Connection refused" errors
- "Too many connections" errors
- Slow query performance

**Solutions:**

```bash
# Check PostgreSQL status
docker-compose exec postgres pg_isready

# Check connection limits
docker-compose exec postgres psql -U escai_user -c "SHOW max_connections;"

# Optimize connection pool
export ESCAI_DB_POOL_SIZE=10
export ESCAI_DB_MAX_OVERFLOW=20

# Check for long-running queries
docker-compose exec postgres psql -U escai_user -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Restart PostgreSQL
docker-compose restart postgres

# Check disk space
df -h
docker system df
```

#### Issue: MongoDB performance problems

**Symptoms:**

- Slow write operations
- High memory usage
- Connection timeouts

**Solutions:**

```bash
# Check MongoDB status
docker-compose exec mongodb mongo --eval "db.adminCommand('ismaster')"

# Check database size
docker-compose exec mongodb mongo escai_logs --eval "db.stats()"

# Create indexes for better performance
docker-compose exec mongodb mongo escai_logs --eval "
db.events.createIndex({timestamp: 1});
db.events.createIndex({agent_id: 1, timestamp: 1});
"

# Check slow operations
docker-compose exec mongodb mongo escai_logs --eval "db.setProfilingLevel(2, {slowms: 100})"

# Compact database
docker-compose exec mongodb mongo escai_logs --eval "db.runCommand({compact: 'events'})"
```

#### Issue: Redis memory issues

**Symptoms:**

- "Out of memory" errors
- Slow cache operations
- Connection refused

**Solutions:**

```bash
# Check Redis memory usage
docker-compose exec redis redis-cli info memory

# Set memory limit
docker-compose exec redis redis-cli config set maxmemory 512mb
docker-compose exec redis redis-cli config set maxmemory-policy allkeys-lru

# Clear cache if needed
docker-compose exec redis redis-cli flushall

# Check for memory leaks
docker-compose exec redis redis-cli memory usage session:*
```

### 5. Performance Issues

#### Issue: Slow API responses

**Symptoms:**

- High response times
- Timeout errors
- Poor user experience

**Diagnostic Steps:**

```bash
# Measure API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/v1/health

# Check database query performance
escai performance analyze --component database

# Monitor resource usage
escai performance monitor --duration 300

# Profile API endpoints
escai performance profile --endpoint /api/v1/epistemic/*/current
```

**Solutions:**

```bash
# Optimize database queries
escai database optimize-indexes
escai database analyze-slow-queries

# Increase worker processes
escai server start --workers 8

# Enable caching
export ESCAI_CACHE_ENABLED=true
export ESCAI_CACHE_TTL=300

# Optimize configuration
escai config optimize --for-performance
```

#### Issue: High memory usage

**Symptoms:**

- Out of memory errors
- System slowdown
- Process crashes

**Solutions:**

```bash
# Monitor memory usage
escai monitor memory --detailed

# Optimize memory settings
export ESCAI_MAX_MEMORY_MB=4096
export ESCAI_GC_THRESHOLD=0.8

# Reduce batch sizes
export ESCAI_BATCH_SIZE=100
export ESCAI_BUFFER_SIZE=500

# Enable memory profiling
escai profile memory --duration 300
```

### 6. Framework Integration Issues

#### Issue: LangChain integration problems

**Symptoms:**

- Callbacks not working
- Missing chain steps
- Tool usage not captured

**Solutions:**

```python
# Verify LangChain version compatibility
import langchain
print(f"LangChain version: {langchain.__version__}")

# Check callback registration
from langchain.callbacks import get_openai_callback
from escai_framework.instrumentation import LangChainInstrumentor

instrumentor = LangChainInstrumentor()

# Ensure callbacks are properly set
with get_openai_callback() as cb:
    with monitor_agent("agent", "langchain", config) as session:
        # Your LangChain code
        result = agent_executor.invoke({"input": "test"})
        print(f"Tokens used: {cb.total_tokens}")

# Check for callback conflicts
import langchain.callbacks
print("Registered callbacks:", langchain.callbacks.manager._callbacks)
```

#### Issue: AutoGen conversation monitoring fails

**Symptoms:**

- Messages not captured
- Group chat issues
- Agent coordination problems

**Solutions:**

```python
# Verify AutoGen setup
import autogen
print(f"AutoGen version: {autogen.__version__}")

# Check agent configuration
agents = [user_proxy, assistant]
for agent in agents:
    print(f"Agent {agent.name}: {type(agent)}")

# Test basic conversation
with monitor_agent("autogen-test", "autogen", config) as session:
    user_proxy.initiate_chat(
        assistant,
        message="Hello, this is a test message."
    )

    # Check captured events
    events = session.get_captured_events()
    print(f"Captured {len(events)} events")
```

### 7. WebSocket Issues

#### Issue: WebSocket connections failing

**Symptoms:**

- Connection refused
- Frequent disconnections
- No real-time updates

**Solutions:**

```bash
# Test WebSocket connectivity
wscat -c ws://localhost:8000/ws/monitor/test-session

# Check WebSocket configuration
escai config show websocket

# Monitor WebSocket connections
escai monitor websocket --connections

# Check for proxy issues
export ESCAI_WEBSOCKET_PROXY_TIMEOUT=60
```

```javascript
// Client-side debugging
const ws = new WebSocket("ws://localhost:8000/ws/monitor/session-id");

ws.onopen = function (event) {
  console.log("WebSocket connected");
};

ws.onerror = function (error) {
  console.error("WebSocket error:", error);
};

ws.onclose = function (event) {
  console.log("WebSocket closed:", event.code, event.reason);
};
```

## Debugging Tools

### 1. Built-in Diagnostics

```bash
# Comprehensive system check
escai diagnose

# Component-specific diagnostics
escai diagnose --component api
escai diagnose --component database
escai diagnose --component monitoring

# Performance diagnostics
escai diagnose --performance

# Security diagnostics
escai diagnose --security
```

### 2. Log Analysis Tools

```bash
# Parse and analyze logs
escai logs analyze --file /path/to/escai.log

# Find error patterns
escai logs search --pattern "ERROR" --context 5

# Generate log summary
escai logs summary --last-24h

# Export logs for analysis
escai logs export --format json --output logs.json
```

### 3. Performance Profiling

```bash
# Profile API performance
escai profile api --duration 300

# Profile database queries
escai profile database --slow-queries

# Profile memory usage
escai profile memory --track-allocations

# Generate performance report
escai profile report --output performance-report.html
```

## Environment-Specific Issues

### Development Environment

```bash
# Common development issues
escai dev check-setup
escai dev fix-permissions
escai dev reset-databases

# Development-specific configuration
export ESCAI_ENV=development
export ESCAI_DEBUG=true
export ESCAI_LOG_LEVEL=DEBUG
```

### Production Environment

```bash
# Production health checks
escai prod health-check
escai prod security-check
escai prod performance-check

# Production optimization
escai prod optimize
escai prod tune-performance
```

### Docker Environment

```bash
# Docker-specific diagnostics
docker-compose logs escai-api
docker-compose exec escai-api escai status

# Container resource usage
docker stats escai_escai-api_1

# Network connectivity
docker-compose exec escai-api ping postgres
docker-compose exec escai-api telnet redis 6379
```

### Kubernetes Environment

```bash
# Kubernetes diagnostics
kubectl get pods -n escai
kubectl describe pod escai-api-xxx -n escai
kubectl logs escai-api-xxx -n escai

# Resource usage
kubectl top pods -n escai
kubectl top nodes

# Network policies
kubectl get networkpolicies -n escai
kubectl describe networkpolicy escai-network-policy -n escai
```

## Getting Help

### Self-Service Resources

1. **Documentation**: [https://docs.escai.dev](https://docs.escai.dev)
2. **FAQ**: [FAQ Section](#frequently-asked-questions)
3. **Examples**: [GitHub Examples](https://github.com/escai-framework/escai/tree/main/examples)
4. **Community Forum**: [https://community.escai.dev](https://community.escai.dev)

### Support Channels

1. **GitHub Issues**: [Report bugs and feature requests](https://github.com/escai-framework/escai/issues)
2. **GitHub Discussions**: [Ask questions and share ideas](https://github.com/escai-framework/escai/discussions)
3. **Discord Community**: [Real-time chat support](https://discord.gg/escai)
4. **Email Support**: support@escai.dev (Enterprise customers)

### When Reporting Issues

Please include:

1. **ESCAI Framework version**: `escai version`
2. **Python version**: `python --version`
3. **Operating system**: `uname -a` (Linux/Mac) or `systeminfo` (Windows)
4. **Configuration**: `escai config show` (remove sensitive data)
5. **Error logs**: Recent error messages and stack traces
6. **Steps to reproduce**: Detailed steps to reproduce the issue
7. **Expected vs actual behavior**: What you expected vs what happened

### Issue Template

```markdown
## Issue Description

Brief description of the issue

## Environment

- ESCAI Framework version:
- Python version:
- Operating System:
- Deployment method: (local/docker/kubernetes)

## Steps to Reproduce

1. Step 1
2. Step 2
3. Step 3

## Expected Behavior

What you expected to happen

## Actual Behavior

What actually happened

## Error Logs
```

Paste error logs here

```

## Additional Context
Any additional information that might be helpful
```

## Frequently Asked Questions

### General Questions

**Q: What are the minimum system requirements?**
A: 2 CPU cores, 4GB RAM, 20GB storage. See [deployment guide](../deployment/README.md) for details.

**Q: Which Python versions are supported?**
A: Python 3.8 and above. Python 3.9+ is recommended.

**Q: Can I use ESCAI with my existing agent framework?**
A: ESCAI supports LangChain, AutoGen, CrewAI, and OpenAI Assistants. Custom framework integration is possible.

### Installation Questions

**Q: Why does `pip install escai-framework` fail?**
A: Common causes include outdated pip, dependency conflicts, or network issues. Try updating pip and using a virtual environment.

**Q: How do I install from source?**
A: Clone the repository and run `pip install -e .` from the project root.

### Configuration Questions

**Q: Where should I put my configuration file?**
A: Place it at `~/.escai/config.yaml` or set `ESCAI_CONFIG_PATH` environment variable.

**Q: How do I configure multiple databases?**
A: Each database type has its own configuration section. See the [configuration guide](../configuration/README.md).

### Performance Questions

**Q: How much overhead does monitoring add?**
A: Typically 5-15% depending on configuration. Use sampling and filtering to reduce overhead.

**Q: Can I monitor multiple agents simultaneously?**
A: Yes, ESCAI supports monitoring multiple agents concurrently with proper resource allocation.

### Security Questions

**Q: How is sensitive data protected?**
A: ESCAI includes PII detection, data masking, encryption at rest and in transit, and audit logging.

**Q: Can I integrate with my existing authentication system?**
A: Yes, ESCAI supports JWT tokens and can integrate with external authentication providers.

### Troubleshooting Questions

**Q: What should I do if the API server won't start?**
A: Check port availability, configuration validity, database connectivity, and permissions.

**Q: How do I debug monitoring issues?**
A: Enable debug logging, check framework compatibility, verify configuration, and test with minimal examples.

**Q: Why are my WebSocket connections failing?**
A: Check network connectivity, proxy settings, firewall rules, and WebSocket configuration.

---

_Last updated: [Current Date]_
_For the most up-to-date troubleshooting information, visit [https://docs.escai.dev/troubleshooting](https://docs.escai.dev/troubleshooting)_
