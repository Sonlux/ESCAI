"""
Circuit breaker patterns for monitoring overhead protection.

This module implements circuit breaker patterns to protect against system overload
and excessive monitoring overhead with configurable thresholds and recovery mechanisms.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, List
from threading import Lock
from collections import deque

from .exceptions import MonitoringOverheadError, ServiceUnavailableError


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States of a circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls are blocked
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds to wait before trying half-open
    success_threshold: int = 3  # Successful calls needed to close circuit from half-open
    timeout: float = 30.0  # Timeout for individual calls
    monitoring_window: float = 300.0  # Time window for monitoring metrics (seconds)
    overhead_threshold: float = 0.1  # Maximum allowed monitoring overhead (10%)
    
    # Performance monitoring
    max_response_time: float = 5.0  # Maximum acceptable response time
    max_memory_usage: float = 1024 * 1024 * 100  # 100MB max memory usage
    max_cpu_usage: float = 0.8  # 80% max CPU usage


@dataclass
class CircuitMetrics:
    """Metrics tracked by the circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    
    # Performance metrics
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def reset(self):
        """Reset all metrics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.response_times.clear()
        self.memory_usage.clear()
        self.cpu_usage.clear()
    
    def get_failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    def get_average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_average_memory_usage(self) -> float:
        """Calculate average memory usage."""
        if not self.memory_usage:
            return 0.0
        return sum(self.memory_usage) / len(self.memory_usage)
    
    def get_average_cpu_usage(self) -> float:
        """Calculate average CPU usage."""
        if not self.cpu_usage:
            return 0.0
        return sum(self.cpu_usage) / len(self.cpu_usage)


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting against system overload.
    
    Monitors system performance and automatically opens the circuit when
    thresholds are exceeded, preventing further load on the system.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self._lock = Lock()
        self._last_state_change = time.time()
        
        logger.info(f"Circuit breaker '{name}' initialized in {self.state.value} state")
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on current metrics."""
        # Check failure threshold
        if self.metrics.consecutive_failures >= self.config.failure_threshold:
            return True
        
        # Check performance thresholds
        if self.metrics.get_average_response_time() > self.config.max_response_time:
            logger.warning(f"Circuit '{self.name}': Response time threshold exceeded")
            return True
        
        if self.metrics.get_average_memory_usage() > self.config.max_memory_usage:
            logger.warning(f"Circuit '{self.name}': Memory usage threshold exceeded")
            return True
        
        if self.metrics.get_average_cpu_usage() > self.config.max_cpu_usage:
            logger.warning(f"Circuit '{self.name}': CPU usage threshold exceeded")
            return True
        
        return False
    
    def _should_close_circuit(self) -> bool:
        """Determine if circuit should be closed from half-open state."""
        return self.metrics.consecutive_successes >= self.config.success_threshold
    
    def _can_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset."""
        return (time.time() - self._last_state_change) >= self.config.recovery_timeout
    
    def _record_success(self, response_time: float, memory_usage: float = 0, cpu_usage: float = 0):
        """Record a successful operation."""
        with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()
            
            # Record performance metrics
            self.metrics.response_times.append(response_time)
            if memory_usage > 0:
                self.metrics.memory_usage.append(memory_usage)
            if cpu_usage > 0:
                self.metrics.cpu_usage.append(cpu_usage)
            
            # State transitions
            if self.state == CircuitState.HALF_OPEN and self._should_close_circuit():
                self._transition_to_closed()
    
    def _record_failure(self, exception: Exception):
        """Record a failed operation."""
        with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = time.time()
            
            # State transitions
            if self.state == CircuitState.CLOSED and self._should_open_circuit():
                self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition circuit to open state."""
        self.state = CircuitState.OPEN
        self._last_state_change = time.time()
        logger.warning(
            f"Circuit breaker '{self.name}' opened. "
            f"Failures: {self.metrics.consecutive_failures}, "
            f"Failure rate: {self.metrics.get_failure_rate():.2%}"
        )
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self._last_state_change = time.time()
        self.metrics.consecutive_successes = 0
        logger.info(f"Circuit breaker '{self.name}' transitioned to half-open")
    
    def _transition_to_closed(self):
        """Transition circuit to closed state."""
        self.state = CircuitState.CLOSED
        self._last_state_change = time.time()
        logger.info(f"Circuit breaker '{self.name}' closed after successful recovery")
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute an async function through the circuit breaker."""
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._can_attempt_reset():
                self._transition_to_half_open()
            else:
                raise ServiceUnavailableError(
                    f"Circuit breaker '{self.name}' is open",
                    status_code=503
                )
        
        # Execute the function with monitoring
        start_time = time.time()
        try:
            # Add timeout to the call
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            self._record_success(response_time)
            return result
            
        except Exception as e:
            self._record_failure(e)
            raise e
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a synchronous function through the circuit breaker."""
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._can_attempt_reset():
                self._transition_to_half_open()
            else:
                raise ServiceUnavailableError(
                    f"Circuit breaker '{self.name}' is open",
                    status_code=503
                )
        
        # Execute the function with monitoring
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            self._record_success(response_time)
            return result
            
        except Exception as e:
            self._record_failure(e)
            raise e
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status and metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "failure_rate": self.metrics.get_failure_rate(),
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "average_response_time": self.metrics.get_average_response_time(),
                "average_memory_usage": self.metrics.get_average_memory_usage(),
                "average_cpu_usage": self.metrics.get_average_cpu_usage()
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "overhead_threshold": self.config.overhead_threshold
            },
            "last_state_change": self._last_state_change
        }
    
    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.metrics.reset()
            self._last_state_change = time.time()
            logger.info(f"Circuit breaker '{self.name}' manually reset")


class MonitoringCircuitBreaker(CircuitBreaker):
    """
    Specialized circuit breaker for monitoring overhead protection.
    
    Monitors the performance impact of the monitoring system itself
    and automatically disables monitoring when overhead exceeds thresholds.
    """
    
    def __init__(self, name: str = "monitoring", config: CircuitBreakerConfig = None):
        super().__init__(name, config)
        self._baseline_performance: Dict[str, float] = {}
        self._monitoring_enabled = True
    
    def set_baseline_performance(self, response_time: float, memory_usage: float, cpu_usage: float):
        """Set baseline performance metrics for overhead calculation."""
        self._baseline_performance = {
            "response_time": response_time,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage
        }
        logger.info(f"Baseline performance set for monitoring circuit breaker: {self._baseline_performance}")
    
    def calculate_overhead(self) -> Dict[str, float]:
        """Calculate current monitoring overhead compared to baseline."""
        if not self._baseline_performance:
            return {"response_time": 0.0, "memory_usage": 0.0, "cpu_usage": 0.0}
        
        current_response_time = self.metrics.get_average_response_time()
        current_memory = self.metrics.get_average_memory_usage()
        current_cpu = self.metrics.get_average_cpu_usage()
        
        overhead: Dict[str, float] = {}
        
        if self._baseline_performance["response_time"] > 0:
            overhead["response_time"] = (
                (current_response_time - self._baseline_performance["response_time"]) /
                self._baseline_performance["response_time"]
            )
        else:
            overhead["response_time"] = 0.0
        
        if self._baseline_performance["memory_usage"] > 0:
            overhead["memory_usage"] = (
                (current_memory - self._baseline_performance["memory_usage"]) /
                self._baseline_performance["memory_usage"]
            )
        else:
            overhead["memory_usage"] = 0.0
        
        if self._baseline_performance["cpu_usage"] > 0:
            overhead["cpu_usage"] = (
                (current_cpu - self._baseline_performance["cpu_usage"]) /
                self._baseline_performance["cpu_usage"]
            )
        else:
            overhead["cpu_usage"] = 0.0
        
        return overhead
    
    def check_monitoring_overhead(self) -> bool:
        """Check if monitoring overhead exceeds thresholds."""
        overhead = self.calculate_overhead()
        
        for metric, value in overhead.items():
            if value > self.config.overhead_threshold:
                logger.warning(
                    f"Monitoring overhead threshold exceeded for {metric}: "
                    f"{value:.2%} > {self.config.overhead_threshold:.2%}"
                )
                return True
        
        return False
    
    async def monitor_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with monitoring overhead protection."""
        if not self._monitoring_enabled or self.state == CircuitState.OPEN:
            # Execute without monitoring
            return await func(*args, **kwargs)
        
        # Check overhead before execution
        if self.check_monitoring_overhead():
            raise MonitoringOverheadError(
                max(self.calculate_overhead().values()),
                self.config.overhead_threshold
            )
        
        return await self.call_async(func, *args, **kwargs)
    
    def monitor_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with monitoring overhead protection."""
        if not self._monitoring_enabled or self.state == CircuitState.OPEN:
            # Execute without monitoring
            return func(*args, **kwargs)
        
        # Check overhead before execution
        if self.check_monitoring_overhead():
            raise MonitoringOverheadError(
                max(self.calculate_overhead().values()),
                self.config.overhead_threshold
            )
        
        return self.call_sync(func, *args, **kwargs)
    
    def disable_monitoring(self):
        """Temporarily disable monitoring to reduce overhead."""
        self._monitoring_enabled = False
        logger.warning("Monitoring temporarily disabled due to overhead concerns")
    
    def enable_monitoring(self):
        """Re-enable monitoring."""
        self._monitoring_enabled = True
        logger.info("Monitoring re-enabled")


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = Lock()


def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def get_monitoring_circuit_breaker(config: CircuitBreakerConfig = None) -> MonitoringCircuitBreaker:
    """Get or create the monitoring circuit breaker."""
    with _registry_lock:
        if "monitoring" not in _circuit_breakers:
            monitoring_breaker = MonitoringCircuitBreaker("monitoring", config)
            _circuit_breakers["monitoring"] = monitoring_breaker
        return _circuit_breakers["monitoring"]  # type: ignore


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    with _registry_lock:
        return _circuit_breakers.copy()


def reset_all_circuit_breakers():
    """Reset all registered circuit breakers."""
    with _registry_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")