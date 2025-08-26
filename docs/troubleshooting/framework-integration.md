# Framework Integration Troubleshooting Guide

This guide helps diagnose and resolve common issues with framework integration robustness in the ESCAI framework.

## Quick Diagnostics

### Health Check Script

Create a comprehensive health check script to diagnose integration issues:

```python
#!/usr/bin/env python3
"""
ESCAI Framework Integration Health Check

Run this script to diagnose framework integration issues.
"""

import asyncio
import logging
import sys
from datetime import datetime

from escai_framework.instrumentation.framework_compatibility import get_compatibility_manager
from escai_framework.instrumentation.adaptive_sampling import get_sampling_manager
from escai_framework.utils.circuit_breaker import get_all_circuit_breakers

async def run_health_check():
    """Run comprehensive health check."""
    print("ESCAI Framework Integration Health Check")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print()

    # Check framework compatibility
    print("1. Framework Compatibility Check")
    print("-" * 30)

    manager = get_compatibility_manager()
    frameworks = await manager.detect_all_frameworks()

    for name, info in frameworks.items():
        status_icon = "✓" if info.status.value == "available" else "✗"
        print(f"{status_icon} {name}: {info.status.value}")

        if info.version:
            print(f"  Version: {info.version}")

        if info.error_message:
            print(f"  Error: {info.error_message}")

        if info.available_features:
            print(f"  Features: {len(info.available_features)} available")

        if info.missing_features:
            print(f"  Missing: {len(info.missing_features)} features")

        print()

    # Check sampling system
    print("2. Adaptive Sampling Status")
    print("-" * 30)

    sampling_manager = get_sampling_manager()
    stats = sampling_manager.get_sampling_statistics()

    print(f"Strategy: {stats['strategy']}")
    print(f"Total events: {stats['total_events']}")
    print(f"Sampling rate: {stats['sampling_rate']:.2%}")
    print(f"Overhead per event: {stats['overhead_per_event_ms']:.2f}ms")
    print()

    # Check circuit breakers
    print("3. Circuit Breaker Status")
    print("-" * 30)

    breakers = get_all_circuit_breakers()
    for name, breaker in breakers.items():
        status = breaker.get_status()
        state_icon = "✓" if status['state'] == "closed" else "⚠" if status['state'] == "half_open" else "✗"

        print(f"{state_icon} {name}: {status['state']}")
        print(f"  Calls: {status['metrics']['total_calls']}")
        print(f"  Failures: {status['metrics']['failed_calls']}")
        print(f"  Success rate: {(1 - status['metrics']['failure_rate']) * 100:.1f}%")
        print()

    # System recommendations
    print("4. Recommendations")
    print("-" * 30)

    recommendations = []

    # Check for unavailable frameworks
    unavailable = [name for name, info in frameworks.items()
                  if info.status.value == "unavailable"]
    if unavailable:
        recommendations.append(f"Install missing frameworks: {', '.join(unavailable)}")

    # Check for high overhead
    if stats['overhead_per_event_ms'] > 10.0:
        recommendations.append("High monitoring overhead detected - consider reducing sampling rate")

    # Check for open circuit breakers
    open_breakers = [name for name, breaker in breakers.items()
                    if breaker.get_status()['state'] == 'open']
    if open_breakers:
        recommendations.append(f"Circuit breakers open: {', '.join(open_breakers)} - check for underlying issues")

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✓ All systems appear healthy")

    print()
    print("Health check complete.")

if __name__ == "__main__":
    asyncio.run(run_health_check())
```

## Common Issues and Solutions

### 1. Framework Not Detected

#### Symptoms

- `FrameworkNotSupportedError` exceptions
- Framework status shows as "unavailable"
- Import errors in logs

#### Diagnosis

```python
import sys
import importlib

# Check if framework is installed
frameworks_to_check = ["langchain", "autogen", "crewai", "openai"]

for framework in frameworks_to_check:
    try:
        module = importlib.import_module(framework)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {framework}: {version}")
    except ImportError as e:
        print(f"✗ {framework}: Not installed ({e})")
```

#### Solutions

1. **Install missing framework**:

   ```bash
   # LangChain
   pip install langchain

   # AutoGen
   pip install pyautogen

   # CrewAI
   pip install crewai

   # OpenAI
   pip install openai
   ```

2. **Check Python path**:

   ```python
   import sys
   print("Python path:")
   for path in sys.path:
       print(f"  {path}")
   ```

3. **Virtual environment issues**:
   ```bash
   # Ensure you're in the correct virtual environment
   which python
   pip list | grep -E "(langchain|autogen|crewai|openai)"
   ```

### 2. Version Compatibility Issues

#### Symptoms

- Framework status shows as "incompatible"
- Version-related error messages
- Missing features warnings

#### Diagnosis

```python
from escai_framework.instrumentation.framework_compatibility import get_compatibility_manager

async def check_version_compatibility():
    manager = get_compatibility_manager()

    # Get compatibility matrix
    matrix = manager.get_framework_matrix()

    for framework, requirements in matrix.items():
        print(f"{framework}:")
        print(f"  Required: {requirements['min_version']} - {requirements['max_version']}")
        print(f"  Current: {requirements['current_version']}")
        print(f"  Status: {requirements['status']}")
        print()

asyncio.run(check_version_compatibility())
```

#### Solutions

1. **Upgrade framework**:

   ```bash
   pip install --upgrade langchain
   pip install --upgrade pyautogen
   pip install --upgrade crewai
   pip install --upgrade openai
   ```

2. **Downgrade if necessary**:

   ```bash
   # Install specific version
   pip install langchain==0.1.0
   ```

3. **Check breaking changes**:
   ```python
   # Review breaking changes for your version
   framework_info = await manager.detect_framework("langchain")
   if framework_info.compatibility:
       breaking_changes = framework_info.compatibility.breaking_changes
       for version, description in breaking_changes.items():
           print(f"Version {version}: {description}")
   ```

### 3. High Monitoring Overhead

#### Symptoms

- Slow agent execution
- High CPU usage
- Memory consumption issues
- Overhead warnings in logs

#### Diagnosis

```python
from escai_framework.instrumentation.adaptive_sampling import get_sampling_manager

def diagnose_overhead():
    manager = get_sampling_manager()
    stats = manager.get_sampling_statistics()

    print(f"Sampling rate: {stats['sampling_rate']:.2%}")
    print(f"Overhead per event: {stats['overhead_per_event_ms']:.2f}ms")
    print(f"Total events: {stats['total_events']}")
    print(f"Sampled events: {stats['sampled_events']}")

    if stats['overhead_per_event_ms'] > 10.0:
        print("⚠ High overhead detected!")
        return True
    return False

diagnose_overhead()
```

#### Solutions

1. **Reduce sampling rate**:

   ```python
   from escai_framework.instrumentation.adaptive_sampling import SamplingConfig, SamplingStrategy

   # Configure lower sampling rate
   config = SamplingConfig(
       strategy=SamplingStrategy.ADAPTIVE_RATE,
       base_sampling_rate=0.5,  # Reduce to 50%
       min_sampling_rate=0.1,
       max_overhead_ms=5.0
   )

   manager = get_sampling_manager()
   manager.update_configuration(config)
   ```

2. **Use performance-aware sampling**:

   ```python
   config = SamplingConfig(
       strategy=SamplingStrategy.PERFORMANCE_AWARE,
       base_sampling_rate=1.0,
       max_overhead_ms=2.0  # Strict overhead limit
   )
   ```

3. **Enable importance-based sampling**:

   ```python
   from escai_framework.instrumentation.events import EventType

   config = SamplingConfig(
       strategy=SamplingStrategy.IMPORTANCE_BASED,
       high_priority_events=[
           EventType.AGENT_ERROR,
           EventType.TASK_FAIL,
           EventType.DECISION_START
       ]
   )
   ```

### 4. Circuit Breaker Issues

#### Symptoms

- "Circuit breaker is open" errors
- Intermittent monitoring failures
- Service unavailable messages

#### Diagnosis

```python
from escai_framework.utils.circuit_breaker import get_all_circuit_breakers

def diagnose_circuit_breakers():
    breakers = get_all_circuit_breakers()

    for name, breaker in breakers.items():
        status = breaker.get_status()

        print(f"Circuit Breaker: {name}")
        print(f"  State: {status['state']}")
        print(f"  Total calls: {status['metrics']['total_calls']}")
        print(f"  Failed calls: {status['metrics']['failed_calls']}")
        print(f"  Failure rate: {status['metrics']['failure_rate']:.2%}")
        print(f"  Last state change: {status['last_state_change']}")

        if status['state'] == 'open':
            print("  ⚠ Circuit breaker is OPEN - blocking calls")
        elif status['state'] == 'half_open':
            print("  ⚠ Circuit breaker is HALF-OPEN - testing recovery")

        print()

diagnose_circuit_breakers()
```

#### Solutions

1. **Reset circuit breaker**:

   ```python
   from escai_framework.utils.circuit_breaker import get_circuit_breaker

   # Reset specific circuit breaker
   breaker = get_circuit_breaker("langchain_main")
   breaker.reset()
   print("Circuit breaker reset")
   ```

2. **Adjust thresholds**:

   ```python
   from escai_framework.utils.circuit_breaker import CircuitBreakerConfig

   # More lenient configuration
   config = CircuitBreakerConfig(
       failure_threshold=10,  # Allow more failures
       recovery_timeout=30.0,  # Shorter recovery time
       timeout=60.0  # Longer operation timeout
   )
   ```

3. **Check underlying issues**:

   ```python
   # Look for root cause of failures
   import logging

   # Enable debug logging
   logging.getLogger('escai_framework.utils.circuit_breaker').setLevel(logging.DEBUG)
   ```

### 5. Recovery Failures

#### Symptoms

- Instrumentor stuck in "failed" state
- Repeated recovery attempts
- Degraded functionality not recovering

#### Diagnosis

```python
def diagnose_instrumentor_health(instrumentor):
    health = instrumentor.get_health_status()

    print(f"Instrumentor: {health['framework']}")
    print(f"State: {health['state']}")
    print(f"Error count: {health['error_count']}")
    print(f"Recovery attempts: {health['recovery_attempts']}")
    print(f"Success rate: {health['success_rate']:.2%}")
    print(f"Degradation level: {health['degradation_level']:.2f}")

    if health['last_error']:
        print(f"Last error: {health['last_error']}")

    if not health['is_healthy']:
        print("⚠ Instrumentor is not healthy!")

        if health['state'] == 'failed':
            print("  - Instrumentor has failed")
        elif health['state'] == 'degraded':
            print("  - Instrumentor is running in degraded mode")
        elif health['state'] == 'recovering':
            print("  - Instrumentor is attempting recovery")
```

#### Solutions

1. **Manual recovery**:

   ```python
   # Force recovery attempt
   await instrumentor._attempt_recovery()
   ```

2. **Adjust recovery configuration**:

   ```python
   from escai_framework.instrumentation.robust_instrumentor import RobustConfig

   config = RobustConfig(
       max_consecutive_errors=10,  # Allow more errors
       error_recovery_timeout=30.0,  # Faster recovery
       max_recovery_attempts=5  # More recovery attempts
   )
   ```

3. **Enable graceful degradation**:
   ```python
   config = RobustConfig(
       enable_degradation=True,
       degradation_threshold=0.5,  # Degrade at 50% error rate
       min_functionality_level=0.2  # Maintain 20% functionality
   )
   ```

### 6. Memory Leaks

#### Symptoms

- Gradually increasing memory usage
- Out of memory errors
- Slow performance over time

#### Diagnosis

```python
import psutil
import gc

def diagnose_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()

    print(f"RSS Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"VMS Memory: {memory_info.vms / 1024 / 1024:.2f} MB")

    # Check garbage collection
    print(f"Garbage collection counts: {gc.get_count()}")

    # Check sampling manager cache
    manager = get_sampling_manager()
    cache_stats = manager.get_cache_stats() if hasattr(manager, 'get_cache_stats') else {}
    print(f"Sampling cache: {cache_stats}")

diagnose_memory_usage()
```

#### Solutions

1. **Clear caches periodically**:

   ```python
   # Clear framework compatibility cache
   manager = get_compatibility_manager()
   manager.clear_cache()

   # Clear sampling cache
   sampling_manager = get_sampling_manager()
   sampling_manager.reset_metrics()
   ```

2. **Reduce cache sizes**:

   ```python
   # Configure smaller cache sizes in sampling
   config = SamplingConfig(
       adaptation_window=50,  # Smaller window
       batch_size=5  # Smaller batches
   )
   ```

3. **Force garbage collection**:

   ```python
   import gc

   # Force garbage collection
   collected = gc.collect()
   print(f"Collected {collected} objects")
   ```

## Performance Optimization

### 1. Optimal Sampling Configuration

```python
# Production-optimized sampling configuration
production_sampling = SamplingConfig(
    strategy=SamplingStrategy.HYBRID,
    base_sampling_rate=0.8,  # 80% sampling
    min_sampling_rate=0.05,  # Never go below 5%
    max_sampling_rate=1.0,
    max_overhead_ms=2.0,     # Strict overhead limit
    target_overhead_ms=1.0,  # Target 1ms per event
    adaptation_window=50,    # Smaller adaptation window
    high_priority_events=[
        EventType.AGENT_ERROR,
        EventType.TASK_FAIL,
        EventType.DECISION_START,
        EventType.DECISION_COMPLETE
    ]
)
```

### 2. Circuit Breaker Tuning

```python
# Optimized circuit breaker configuration
optimized_cb_config = CircuitBreakerConfig(
    failure_threshold=3,      # Quick failure detection
    recovery_timeout=15.0,    # Fast recovery attempts
    success_threshold=2,      # Quick recovery confirmation
    timeout=5.0,             # Short operation timeout
    max_response_time=2.0,   # Strict response time limit
    max_memory_usage=50 * 1024 * 1024,  # 50MB limit
    max_cpu_usage=0.5        # 50% CPU limit
)
```

### 3. Robust Instrumentor Tuning

```python
# Performance-optimized robust configuration
performance_config = RobustConfig(
    max_consecutive_errors=2,     # Quick error detection
    error_recovery_timeout=10.0,  # Fast recovery
    max_recovery_attempts=3,      # Limited recovery attempts
    max_overhead_percentage=1.0,  # Very strict overhead limit
    performance_check_interval=15.0,  # Frequent performance checks
    enable_degradation=True,
    degradation_threshold=0.2,    # Early degradation
    min_functionality_level=0.5,  # Maintain good functionality
    adaptive_sampling=True,
    sampling_config=production_sampling
)
```

## Monitoring and Alerting

### 1. Health Monitoring Script

```python
#!/usr/bin/env python3
"""
Continuous health monitoring for ESCAI framework integration.
"""

import asyncio
import time
import json
from datetime import datetime

async def continuous_monitoring(interval=60):
    """Monitor health continuously and log issues."""

    while True:
        try:
            # Collect health data
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "frameworks": {},
                "sampling": {},
                "circuit_breakers": {}
            }

            # Check frameworks
            manager = get_compatibility_manager()
            frameworks = await manager.detect_all_frameworks()

            for name, info in frameworks.items():
                health_data["frameworks"][name] = {
                    "status": info.status.value,
                    "version": str(info.version) if info.version else None,
                    "error": info.error_message
                }

            # Check sampling
            sampling_manager = get_sampling_manager()
            health_data["sampling"] = sampling_manager.get_sampling_statistics()

            # Check circuit breakers
            breakers = get_all_circuit_breakers()
            for name, breaker in breakers.items():
                status = breaker.get_status()
                health_data["circuit_breakers"][name] = {
                    "state": status["state"],
                    "failure_rate": status["metrics"]["failure_rate"]
                }

            # Log health data
            print(json.dumps(health_data, indent=2))

            # Check for issues and alert
            await check_and_alert(health_data)

        except Exception as e:
            print(f"Health monitoring error: {e}")

        await asyncio.sleep(interval)

async def check_and_alert(health_data):
    """Check for issues and generate alerts."""

    alerts = []

    # Check for unavailable frameworks
    for name, info in health_data["frameworks"].items():
        if info["status"] == "unavailable":
            alerts.append(f"Framework {name} is unavailable: {info['error']}")

    # Check for high overhead
    sampling = health_data["sampling"]
    if sampling.get("overhead_per_event_ms", 0) > 10.0:
        alerts.append(f"High sampling overhead: {sampling['overhead_per_event_ms']:.2f}ms")

    # Check for open circuit breakers
    for name, cb in health_data["circuit_breakers"].items():
        if cb["state"] == "open":
            alerts.append(f"Circuit breaker {name} is open (failure rate: {cb['failure_rate']:.2%})")

    # Send alerts (implement your alerting mechanism)
    for alert in alerts:
        print(f"ALERT: {alert}")
        # await send_alert(alert)  # Implement your alerting

if __name__ == "__main__":
    asyncio.run(continuous_monitoring())
```

### 2. Metrics Collection

```python
def collect_metrics():
    """Collect comprehensive metrics for monitoring."""

    metrics = {
        "timestamp": time.time(),
        "framework_compatibility": {},
        "sampling_performance": {},
        "circuit_breaker_status": {},
        "system_resources": {}
    }

    # Framework metrics
    manager = get_compatibility_manager()
    matrix = manager.get_framework_matrix()
    metrics["framework_compatibility"] = matrix

    # Sampling metrics
    sampling_manager = get_sampling_manager()
    metrics["sampling_performance"] = sampling_manager.get_sampling_statistics()

    # Circuit breaker metrics
    breakers = get_all_circuit_breakers()
    for name, breaker in breakers.items():
        metrics["circuit_breaker_status"][name] = breaker.get_status()

    # System resource metrics
    import psutil
    process = psutil.Process()
    metrics["system_resources"] = {
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "open_files": len(process.open_files()),
        "threads": process.num_threads()
    }

    return metrics
```

This troubleshooting guide provides comprehensive diagnostics and solutions for framework integration robustness issues. Use the health check script regularly and implement continuous monitoring to maintain optimal performance.
