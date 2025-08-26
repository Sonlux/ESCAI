# Framework Integration Robustness

This document describes the enhanced framework integration robustness features in the ESCAI framework, including version compatibility checking, automatic adaptation, graceful degradation, and monitoring overhead optimization.

## Overview

The ESCAI framework provides robust integration capabilities that ensure reliable monitoring across different agent frameworks, even when framework APIs change, versions are incompatible, or system resources are constrained.

## Key Features

### 1. Framework Compatibility Management

#### Version Detection and Validation

The framework automatically detects installed agent frameworks and validates their compatibility:

```python
from escai_framework.instrumentation.framework_compatibility import get_compatibility_manager

# Get compatibility manager
manager = get_compatibility_manager()

# Detect specific framework
framework_info = await manager.detect_framework("langchain")
print(f"Status: {framework_info.status}")
print(f"Version: {framework_info.version}")
print(f"Available features: {framework_info.available_features}")

# Detect all frameworks
all_frameworks = await manager.detect_all_frameworks()
compatible_frameworks = await manager.get_compatible_frameworks()
```

#### Compatibility Requirements

Each framework has defined compatibility requirements:

```python
# Get compatibility matrix
matrix = manager.get_framework_matrix()
print(matrix["langchain"]["min_version"])  # Minimum supported version
print(matrix["langchain"]["required_features"])  # Required features
```

#### Configuration Validation

Framework-specific configurations are validated before use:

```python
config = {
    "agents": [agent1, agent2],
    "monitor_conversations": True
}

is_valid, errors = await manager.validate_framework_configuration(
    "autogen", config
)

if not is_valid:
    print(f"Configuration errors: {errors}")
```

### 2. Adaptive Sampling System

#### Intelligent Sampling Strategies

The adaptive sampling system automatically adjusts monitoring frequency based on system performance and event importance:

```python
from escai_framework.instrumentation.adaptive_sampling import (
    SamplingConfig, SamplingStrategy, get_sampling_manager
)

# Configure adaptive sampling
config = SamplingConfig(
    strategy=SamplingStrategy.HYBRID,
    base_sampling_rate=1.0,
    min_sampling_rate=0.1,
    max_overhead_ms=10.0,
    high_priority_events=[EventType.AGENT_ERROR, EventType.TASK_FAIL]
)

manager = get_sampling_manager()
manager.update_configuration(config)

# Check if event should be sampled
decision = await manager.should_sample_event(event, context)
```

#### Sampling Strategies

1. **Fixed Rate**: Consistent sampling percentage
2. **Adaptive Rate**: Adjusts based on performance metrics
3. **Importance-Based**: Prioritizes critical events
4. **Performance-Aware**: Considers system load
5. **Hybrid**: Combines multiple strategies

#### Performance Optimization

The system automatically optimizes sampling rates to maintain performance:

```python
# Get sampling statistics
stats = manager.get_sampling_statistics()
print(f"Sampling rate: {stats['sampling_rate']:.2%}")
print(f"Overhead per event: {stats['overhead_per_event_ms']:.2f}ms")

# Manually optimize if needed
await manager.optimize_sampling_rate()
```

### 3. Robust Instrumentor Base Class

#### Enhanced Error Handling

The `RobustInstrumentor` class provides comprehensive error handling and recovery:

```python
from escai_framework.instrumentation.robust_instrumentor import (
    RobustInstrumentor, RobustConfig, RecoveryStrategy
)

class MyRobustInstrumentor(RobustInstrumentor):
    def __init__(self):
        config = RobustConfig(
            max_consecutive_errors=5,
            error_recovery_timeout=60.0,
            enable_degradation=True,
            recovery_strategies=[
                RecoveryStrategy.IMMEDIATE_RETRY,
                RecoveryStrategy.EXPONENTIAL_BACKOFF,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ]
        )
        super().__init__("my_framework", config)

    async def _initialize_framework_specific(self):
        # Framework-specific initialization
        pass

    async def _start_monitoring_protected(self, agent_id: str, config: dict) -> str:
        # Protected monitoring start implementation
        return "session_id"
```

#### Health Monitoring

Monitor instrumentor health and performance:

```python
# Get health status
health = instrumentor.get_health_status()
print(f"State: {health['state']}")
print(f"Success rate: {health['success_rate']:.2%}")
print(f"Error count: {health['error_count']}")
print(f"Degradation level: {health['degradation_level']:.2f}")
```

#### Recovery Strategies

Multiple recovery strategies are available:

1. **Immediate Retry**: Quick retry after failure
2. **Exponential Backoff**: Increasing delays between retries
3. **Graceful Degradation**: Reduced functionality mode
4. **Circuit Breaker**: Temporary service suspension
5. **Fallback Mode**: Alternative processing methods

### 4. Circuit Breaker Integration

#### Automatic Protection

Circuit breakers protect against cascading failures:

```python
from escai_framework.utils.circuit_breaker import get_circuit_breaker

# Get circuit breaker for framework
breaker = get_circuit_breaker("langchain_main")

# Check status
status = breaker.get_status()
print(f"State: {status['state']}")
print(f"Failure rate: {status['metrics']['failure_rate']:.2%}")
```

#### Configuration

Configure circuit breaker behavior:

```python
from escai_framework.utils.circuit_breaker import CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    max_response_time=5.0,
    max_memory_usage=100 * 1024 * 1024  # 100MB
)
```

### 5. Fallback Mechanisms

#### Graceful Degradation

When primary processing fails, fallback mechanisms ensure continued operation:

```python
from escai_framework.utils.fallback import get_fallback_manager

manager = get_fallback_manager()

# Execute with fallback support
result = await manager.execute_with_fallback(
    primary_function,
    input_data,
    cache_key="epistemic_extraction"
)

if result.success:
    print(f"Strategy used: {result.strategy_used}")
    print(f"Confidence: {result.confidence}")
else:
    print(f"All fallbacks failed: {result.error_message}")
```

#### Custom Fallback Providers

Create custom fallback providers:

```python
from escai_framework.utils.fallback import FallbackProvider, FallbackStrategy

class CustomFallbackProvider(FallbackProvider):
    def can_handle(self, input_data, error):
        return isinstance(error, SpecificError)

    async def execute_fallback(self, input_data, error):
        # Custom fallback logic
        return FallbackResult(
            success=True,
            result=processed_data,
            strategy_used=FallbackStrategy.RULE_BASED,
            confidence=0.7
        )

    @property
    def strategy(self):
        return FallbackStrategy.RULE_BASED

# Register custom provider
manager.register_provider(CustomFallbackProvider())
```

## Configuration Examples

### Basic Robust Configuration

```python
from escai_framework.instrumentation.robust_instrumentor import RobustConfig
from escai_framework.instrumentation.adaptive_sampling import SamplingConfig, SamplingStrategy

config = RobustConfig(
    # Error handling
    max_consecutive_errors=3,
    error_recovery_timeout=30.0,
    max_recovery_attempts=3,

    # Performance monitoring
    max_overhead_percentage=5.0,
    performance_check_interval=60.0,

    # Graceful degradation
    enable_degradation=True,
    degradation_threshold=0.3,
    min_functionality_level=0.2,

    # Adaptive sampling
    adaptive_sampling=True,
    sampling_config=SamplingConfig(
        strategy=SamplingStrategy.HYBRID,
        base_sampling_rate=1.0,
        min_sampling_rate=0.1,
        max_overhead_ms=5.0
    )
)
```

### Production Configuration

```python
# Production-ready configuration with conservative settings
production_config = RobustConfig(
    max_consecutive_errors=5,
    error_recovery_timeout=120.0,
    max_recovery_attempts=5,
    max_overhead_percentage=2.0,
    performance_check_interval=30.0,
    enable_degradation=True,
    degradation_threshold=0.2,
    min_functionality_level=0.3,
    adaptive_sampling=True,
    sampling_config=SamplingConfig(
        strategy=SamplingStrategy.PERFORMANCE_AWARE,
        base_sampling_rate=0.8,
        min_sampling_rate=0.05,
        max_overhead_ms=2.0,
        target_overhead_ms=1.0
    )
)
```

## Best Practices

### 1. Framework Integration

- Always validate framework compatibility before deployment
- Use version pinning in production environments
- Monitor framework update announcements for breaking changes
- Test with multiple framework versions during development

### 2. Error Handling

- Configure appropriate error thresholds for your environment
- Use exponential backoff for transient failures
- Implement graceful degradation for non-critical functionality
- Monitor error rates and recovery success rates

### 3. Performance Optimization

- Start with conservative sampling rates in production
- Monitor overhead metrics continuously
- Adjust sampling strategies based on system load patterns
- Use importance-based sampling for critical events

### 4. Monitoring and Alerting

- Set up alerts for instrumentor health state changes
- Monitor circuit breaker state transitions
- Track sampling rate changes over time
- Alert on excessive error rates or recovery failures

## Troubleshooting

### Common Issues

#### Framework Not Detected

```python
# Check framework installation
framework_info = await manager.detect_framework("langchain")
if framework_info.status == FrameworkStatus.UNAVAILABLE:
    print("Framework not installed or not in Python path")
    print(f"Error: {framework_info.error_message}")
```

#### High Monitoring Overhead

```python
# Check sampling statistics
stats = sampling_manager.get_sampling_statistics()
if stats['overhead_per_event_ms'] > 10.0:
    print("High overhead detected, optimizing sampling rate")
    await sampling_manager.optimize_sampling_rate()
```

#### Frequent Recovery Attempts

```python
# Check instrumentor health
health = instrumentor.get_health_status()
if health['recovery_attempts'] > 5:
    print("Frequent recovery attempts detected")
    print(f"Last error: {health['last_error']}")
    print("Consider adjusting error thresholds or investigating root cause")
```

### Debugging Tools

#### Enable Debug Logging

```python
import logging

# Enable debug logging for robustness components
logging.getLogger('escai_framework.instrumentation.framework_compatibility').setLevel(logging.DEBUG)
logging.getLogger('escai_framework.instrumentation.adaptive_sampling').setLevel(logging.DEBUG)
logging.getLogger('escai_framework.instrumentation.robust_instrumentor').setLevel(logging.DEBUG)
```

#### Health Check Endpoint

```python
# Create health check endpoint for monitoring
async def health_check():
    manager = get_compatibility_manager()
    frameworks = await manager.detect_all_frameworks()

    health_status = {
        "frameworks": {
            name: {
                "status": info.status.value,
                "version": str(info.version) if info.version else None
            }
            for name, info in frameworks.items()
        },
        "sampling": sampling_manager.get_sampling_statistics(),
        "timestamp": time.time()
    }

    return health_status
```

## Migration Guide

### Upgrading from Basic Instrumentors

1. **Replace base class**:

   ```python
   # Before
   class MyInstrumentor(BaseInstrumentor):
       pass

   # After
   class MyInstrumentor(RobustInstrumentor):
       def __init__(self):
           super().__init__("my_framework", RobustConfig())
   ```

2. **Implement required methods**:

   ```python
   async def _initialize_framework_specific(self):
       # Framework-specific initialization
       pass

   async def _start_monitoring_protected(self, agent_id: str, config: dict) -> str:
       # Protected monitoring implementation
       return await super().start_monitoring(agent_id, config)
   ```

3. **Update error handling**:
   ```python
   # Error handling is now automatic
   # Remove manual try/catch blocks and let RobustInstrumentor handle them
   ```

### Configuration Migration

```python
# Migrate existing configuration to robust configuration
old_config = {
    "max_overhead_percent": 10.0,
    "max_events_per_second": 1000
}

new_config = RobustConfig(
    max_overhead_percentage=old_config["max_overhead_percent"],
    sampling_config=SamplingConfig(
        base_sampling_rate=1.0,
        max_overhead_ms=1000.0 / old_config["max_events_per_second"]
    )
)
```

## Performance Considerations

### Memory Usage

- Sampling managers maintain limited history (configurable)
- Circuit breakers use bounded metrics storage
- Framework compatibility cache has automatic cleanup

### CPU Overhead

- Sampling decisions are optimized for speed
- Framework detection is cached
- Health monitoring runs in background threads

### Network Impact

- No additional network calls for robustness features
- All processing is local to the instrumentor

## Security Considerations

- Framework detection does not execute untrusted code
- Sampling decisions do not expose sensitive data
- Error messages are sanitized before logging
- Fallback mechanisms maintain data privacy

This robustness system ensures that the ESCAI framework can reliably monitor agent systems even in challenging environments with framework changes, resource constraints, or intermittent failures.
