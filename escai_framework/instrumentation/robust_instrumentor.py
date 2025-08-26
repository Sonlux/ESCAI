"""
Robust instrumentor base class with enhanced error handling and recovery.

This module provides a robust base class for framework instrumentors with
comprehensive error handling, automatic recovery, graceful degradation,
and adaptive monitoring capabilities.
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union
import threading
import weakref
from contextlib import asynccontextmanager

from .base_instrumentor import BaseInstrumentor, InstrumentationError, MonitoringSummary
from .framework_compatibility import get_compatibility_manager, FrameworkInfo, FrameworkStatus
from .adaptive_sampling import get_sampling_manager, SamplingDecision, SamplingConfig, SamplingStrategy
from .events import AgentEvent, EventType, EventSeverity
from ..utils.exceptions import (
    FrameworkNotSupportedError, MonitoringOverheadError, AgentConnectionError,
    ProcessingError, NetworkError, ServiceUnavailableError
)
from ..utils.retry import retry_async, RetryConfig, BackoffStrategy
from ..utils.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig, CircuitState
from ..utils.fallback import get_fallback_manager, FallbackResult, FallbackStrategy


logger = logging.getLogger(__name__)


class InstrumentorState(Enum):
    """States of the robust instrumentor."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    FAILED = "failed"
    DISABLED = "disabled"


class RecoveryStrategy(Enum):
    """Recovery strategies for instrumentor failures."""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK_MODE = "fallback_mode"


@dataclass
class InstrumentorHealth:
    """Health status of the instrumentor."""
    state: InstrumentorState = InstrumentorState.INITIALIZING
    last_error: Optional[Exception] = None
    error_count: int = 0
    recovery_attempts: int = 0
    last_recovery_attempt: Optional[float] = None
    degradation_level: float = 0.0  # 0.0 = full functionality, 1.0 = minimal functionality
    
    # Performance metrics
    success_rate: float = 1.0
    average_response_time: float = 0.0
    overhead_percentage: float = 0.0
    
    def is_healthy(self) -> bool:
        """Check if instrumentor is in healthy state."""
        return self.state in [InstrumentorState.HEALTHY, InstrumentorState.DEGRADED]
    
    def can_recover(self) -> bool:
        """Check if recovery can be attempted."""
        return self.state in [InstrumentorState.FAILED, InstrumentorState.DEGRADED]


@dataclass
class RobustConfig:
    """Configuration for robust instrumentor behavior."""
    # Error handling
    max_consecutive_errors: int = 5
    error_recovery_timeout: float = 60.0
    max_recovery_attempts: int = 3
    
    # Performance monitoring
    max_overhead_percentage: float = 10.0
    performance_check_interval: float = 30.0
    
    # Graceful degradation
    enable_degradation: bool = True
    degradation_threshold: float = 0.5  # Error rate threshold for degradation
    min_functionality_level: float = 0.2  # Minimum functionality to maintain
    
    # Sampling configuration
    adaptive_sampling: bool = True
    sampling_config: SamplingConfig = field(default_factory=lambda: SamplingConfig(
        strategy=SamplingStrategy.HYBRID,
        base_sampling_rate=1.0,
        min_sampling_rate=0.1
    ))
    
    # Recovery strategies
    recovery_strategies: List[RecoveryStrategy] = field(default_factory=lambda: [
        RecoveryStrategy.IMMEDIATE_RETRY,
        RecoveryStrategy.EXPONENTIAL_BACKOFF,
        RecoveryStrategy.GRACEFUL_DEGRADATION,
        RecoveryStrategy.FALLBACK_MODE
    ])


class RobustInstrumentor(BaseInstrumentor):
    """
    Robust instrumentor base class with enhanced error handling and recovery.
    
    This class extends BaseInstrumentor with comprehensive error handling,
    automatic recovery mechanisms, graceful degradation, and adaptive monitoring.
    """
    
    def __init__(self, framework_name: str, config: RobustConfig = None, **kwargs):
        super().__init__(**kwargs)
        
        self.framework_name = framework_name
        self.robust_config = config or RobustConfig()
        self.health = InstrumentorHealth()
        
        # Managers and utilities
        self.compatibility_manager = get_compatibility_manager()
        self.sampling_manager = get_sampling_manager()
        self.fallback_manager = get_fallback_manager()
        
        # Circuit breakers
        self.main_circuit_breaker = get_circuit_breaker(
            f"{framework_name}_main",
            CircuitBreakerConfig(
                failure_threshold=self.robust_config.max_consecutive_errors,
                recovery_timeout=self.robust_config.error_recovery_timeout
            )
        )
        
        # Performance monitoring
        self._performance_history = []
        self._last_performance_check = time.time()
        
        # Recovery management
        self._recovery_lock = threading.RLock()
        self._recovery_task: Optional[asyncio.Task] = None
        
        # Framework compatibility
        self._framework_info: Optional[FrameworkInfo] = None
        self._compatibility_validated = False
        
        logger.info(f"Robust instrumentor initialized for {framework_name}")
    
    async def initialize(self) -> bool:
        """
        Initialize the instrumentor with compatibility checking and validation.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.health.state = InstrumentorState.INITIALIZING
            
            # Check framework compatibility
            await self._validate_framework_compatibility()
            
            # Configure adaptive sampling if enabled
            if self.robust_config.adaptive_sampling:
                self.sampling_manager.update_configuration(self.robust_config.sampling_config)
            
            # Initialize framework-specific components
            await self._initialize_framework_specific()
            
            # Start background monitoring
            await self._start_background_monitoring()
            
            self.health.state = InstrumentorState.HEALTHY
            logger.info(f"Robust instrumentor for {self.framework_name} initialized successfully")
            return True
            
        except Exception as e:
            self.health.state = InstrumentorState.FAILED
            self.health.last_error = e
            logger.error(f"Failed to initialize robust instrumentor for {self.framework_name}: {e}")
            return False
    
    async def _validate_framework_compatibility(self):
        """Validate framework compatibility and version."""
        try:
            self._framework_info = await self.compatibility_manager.detect_framework(self.framework_name)
            
            if self._framework_info.status != FrameworkStatus.AVAILABLE:
                raise FrameworkNotSupportedError(
                    self.framework_name,
                    [info.name for info in await self.compatibility_manager.get_compatible_frameworks()]
                )
            
            self._compatibility_validated = True
            logger.info(f"Framework compatibility validated: {self._framework_info}")
            
        except Exception as e:
            logger.error(f"Framework compatibility validation failed: {e}")
            raise
    
    @abstractmethod
    async def _initialize_framework_specific(self):
        """Initialize framework-specific components. Must be implemented by subclasses."""
        pass
    
    async def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        # Start performance monitoring
        asyncio.create_task(self._monitor_performance())
        
        # Start health checking
        asyncio.create_task(self._monitor_health())
    
    async def start_monitoring(self, agent_id: str, config: Dict[str, Any]) -> str:
        """
        Start monitoring with robust error handling.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration parameters for monitoring
            
        Returns:
            session_id: Unique identifier for the monitoring session
        """
        try:
            # Validate configuration
            is_valid, errors = await self.compatibility_manager.validate_framework_configuration(
                self.framework_name, config
            )
            
            if not is_valid:
                raise InstrumentationError(f"Invalid configuration: {'; '.join(errors)}")
            
            # Check instrumentor health
            if not self.health.is_healthy():
                if self.health.can_recover():
                    await self._attempt_recovery()
                else:
                    raise InstrumentationError(f"Instrumentor is not healthy: {self.health.state.value}")
            
            # Start monitoring with circuit breaker protection
            session_id = await self.main_circuit_breaker.call_async(
                self._start_monitoring_protected,
                agent_id,
                config
            )
            
            return session_id
            
        except Exception as e:
            await self._handle_error(e, "start_monitoring")
            raise
    
    async def _start_monitoring_protected(self, agent_id: str, config: Dict[str, Any]) -> str:
        """Protected start monitoring implementation."""
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _start_monitoring_protected")
    
    async def stop_monitoring(self, session_id: str) -> MonitoringSummary:
        """
        Stop monitoring with robust error handling.
        
        Args:
            session_id: Identifier of the session to stop
            
        Returns:
            MonitoringSummary: Summary of the monitoring session
        """
        try:
            # Stop monitoring with circuit breaker protection
            summary = await self.main_circuit_breaker.call_async(
                self._stop_monitoring_protected,
                session_id
            )
            
            return summary
            
        except Exception as e:
            await self._handle_error(e, "stop_monitoring")
            raise
    
    async def _stop_monitoring_protected(self, session_id: str) -> MonitoringSummary:
        """Protected stop monitoring implementation."""
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _stop_monitoring_protected")
    
    async def capture_event(self, event: AgentEvent) -> None:
        """
        Capture an event with adaptive sampling and error handling.
        
        Args:
            event: The agent event to capture
        """
        try:
            # Check if we should sample this event
            if self.robust_config.adaptive_sampling:
                decision = await self.sampling_manager.should_sample_event(event)
                
                if decision == SamplingDecision.SKIP:
                    return
                elif decision == SamplingDecision.PRIORITY_SAMPLE:
                    event.add_tag("priority_sampled")
            
            # Capture event with circuit breaker protection
            await self.main_circuit_breaker.call_async(
                self._capture_event_protected,
                event
            )
            
        except Exception as e:
            await self._handle_error(e, "capture_event")
            
            # Try fallback event capture
            try:
                await self._capture_event_fallback(event)
            except Exception as fallback_error:
                logger.error(f"Fallback event capture also failed: {fallback_error}")
    
    async def _capture_event_protected(self, event: AgentEvent) -> None:
        """Protected event capture implementation."""
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _capture_event_protected")
    
    async def _capture_event_fallback(self, event: AgentEvent) -> None:
        """Fallback event capture when main capture fails."""
        # Simple fallback: just log the event
        logger.warning(f"Fallback capture for event: {event.event_type.value} - {event.message}")
    
    async def _handle_error(self, error: Exception, operation: str):
        """
        Handle errors with appropriate recovery strategies.
        
        Args:
            error: The exception that occurred
            operation: The operation that failed
        """
        with self._recovery_lock:
            self.health.error_count += 1
            self.health.last_error = error
            
            logger.error(f"Error in {operation}: {error}")
            
            # Update health state based on error count
            if self.health.error_count >= self.robust_config.max_consecutive_errors:
                if self.health.state == InstrumentorState.HEALTHY:
                    self.health.state = InstrumentorState.DEGRADED
                    logger.warning(f"Instrumentor degraded due to consecutive errors: {self.health.error_count}")
                elif self.health.state == InstrumentorState.DEGRADED:
                    self.health.state = InstrumentorState.FAILED
                    logger.error(f"Instrumentor failed due to excessive errors: {self.health.error_count}")
            
            # Trigger recovery if needed
            if self.health.can_recover() and not self._recovery_task:
                self._recovery_task = asyncio.create_task(self._attempt_recovery())
    
    async def _attempt_recovery(self):
        """Attempt to recover from failures using configured strategies."""
        try:
            with self._recovery_lock:
                if self.health.recovery_attempts >= self.robust_config.max_recovery_attempts:
                    logger.error("Maximum recovery attempts reached, disabling instrumentor")
                    self.health.state = InstrumentorState.DISABLED
                    return
                
                self.health.state = InstrumentorState.RECOVERING
                self.health.recovery_attempts += 1
                self.health.last_recovery_attempt = time.time()
            
            logger.info(f"Attempting recovery for {self.framework_name} instrumentor (attempt {self.health.recovery_attempts})")
            
            # Try recovery strategies in order
            for strategy in self.robust_config.recovery_strategies:
                try:
                    success = await self._execute_recovery_strategy(strategy)
                    if success:
                        logger.info(f"Recovery successful using strategy: {strategy.value}")
                        self.health.state = InstrumentorState.HEALTHY
                        self.health.error_count = 0
                        return
                except Exception as e:
                    logger.warning(f"Recovery strategy {strategy.value} failed: {e}")
                    continue
            
            # All recovery strategies failed
            logger.error("All recovery strategies failed")
            self.health.state = InstrumentorState.FAILED
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            self.health.state = InstrumentorState.FAILED
        finally:
            self._recovery_task = None
    
    async def _execute_recovery_strategy(self, strategy: RecoveryStrategy) -> bool:
        """
        Execute a specific recovery strategy.
        
        Args:
            strategy: The recovery strategy to execute
            
        Returns:
            True if recovery successful, False otherwise
        """
        if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
            return await self._immediate_retry_recovery()
        elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            return await self._exponential_backoff_recovery()
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation_recovery()
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._circuit_breaker_recovery()
        elif strategy == RecoveryStrategy.FALLBACK_MODE:
            return await self._fallback_mode_recovery()
        else:
            return False
    
    async def _immediate_retry_recovery(self) -> bool:
        """Immediate retry recovery strategy."""
        try:
            # Re-validate framework compatibility
            await self._validate_framework_compatibility()
            
            # Reset circuit breaker
            self.main_circuit_breaker.reset()
            
            # Test basic functionality
            await self._test_basic_functionality()
            
            return True
        except Exception as e:
            logger.warning(f"Immediate retry recovery failed: {e}")
            return False
    
    async def _exponential_backoff_recovery(self) -> bool:
        """Exponential backoff recovery strategy."""
        try:
            # Wait with exponential backoff
            wait_time = min(2 ** self.health.recovery_attempts, 60)
            await asyncio.sleep(wait_time)
            
            return await self._immediate_retry_recovery()
        except Exception as e:
            logger.warning(f"Exponential backoff recovery failed: {e}")
            return False
    
    async def _graceful_degradation_recovery(self) -> bool:
        """Graceful degradation recovery strategy."""
        try:
            # Enable degraded mode with reduced functionality
            self.health.degradation_level = min(
                self.health.degradation_level + 0.2,
                1.0 - self.robust_config.min_functionality_level
            )
            
            # Reduce sampling rate
            if self.robust_config.adaptive_sampling:
                current_config = self.sampling_manager.config
                new_config = SamplingConfig(
                    strategy=current_config.strategy,
                    base_sampling_rate=current_config.base_sampling_rate * 0.5,
                    min_sampling_rate=current_config.min_sampling_rate,
                    max_sampling_rate=current_config.max_sampling_rate
                )
                self.sampling_manager.update_configuration(new_config)
            
            logger.info(f"Enabled graceful degradation: level={self.health.degradation_level:.2f}")
            return True
            
        except Exception as e:
            logger.warning(f"Graceful degradation recovery failed: {e}")
            return False
    
    async def _circuit_breaker_recovery(self) -> bool:
        """Circuit breaker recovery strategy."""
        try:
            # Reset circuit breaker and test
            self.main_circuit_breaker.reset()
            
            # Test with a simple operation
            await self._test_basic_functionality()
            
            return True
        except Exception as e:
            logger.warning(f"Circuit breaker recovery failed: {e}")
            return False
    
    async def _fallback_mode_recovery(self) -> bool:
        """Fallback mode recovery strategy."""
        try:
            # Enable fallback mode for all operations
            logger.info("Enabling fallback mode for all operations")
            
            # This always succeeds as it just enables fallback processing
            return True
        except Exception as e:
            logger.warning(f"Fallback mode recovery failed: {e}")
            return False
    
    async def _test_basic_functionality(self):
        """Test basic functionality to verify recovery."""
        # This should be implemented by subclasses to test framework-specific functionality
        pass
    
    async def _monitor_performance(self):
        """Monitor performance metrics and adjust accordingly."""
        while True:
            try:
                await asyncio.sleep(self.robust_config.performance_check_interval)
                
                # Check overhead
                if self.robust_config.adaptive_sampling:
                    stats = self.sampling_manager.get_sampling_statistics()
                    overhead = stats.get('overhead_per_event_ms', 0)
                    
                    if overhead > self.robust_config.max_overhead_percentage:
                        logger.warning(f"High monitoring overhead detected: {overhead:.2f}ms")
                        await self.sampling_manager.optimize_sampling_rate()
                
                # Update health metrics
                self._update_performance_metrics()
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def _monitor_health(self):
        """Monitor overall health and trigger recovery if needed."""
        while True:
            try:
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
                # Check if recovery is needed
                if (self.health.state == InstrumentorState.FAILED and 
                    self.health.can_recover() and 
                    not self._recovery_task):
                    
                    # Check if enough time has passed since last recovery attempt
                    if (not self.health.last_recovery_attempt or 
                        time.time() - self.health.last_recovery_attempt > self.robust_config.error_recovery_timeout):
                        
                        self._recovery_task = asyncio.create_task(self._attempt_recovery())
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics in health status."""
        try:
            # Get circuit breaker metrics
            cb_status = self.main_circuit_breaker.get_status()
            cb_metrics = cb_status.get('metrics', {})
            
            self.health.success_rate = 1.0 - cb_metrics.get('failure_rate', 0.0)
            self.health.average_response_time = cb_metrics.get('average_response_time', 0.0)
            
            # Get sampling metrics if available
            if self.robust_config.adaptive_sampling:
                sampling_stats = self.sampling_manager.get_sampling_statistics()
                self.health.overhead_percentage = sampling_stats.get('overhead_per_event_ms', 0.0)
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the instrumentor."""
        return {
            "framework": self.framework_name,
            "state": self.health.state.value,
            "is_healthy": self.health.is_healthy(),
            "error_count": self.health.error_count,
            "recovery_attempts": self.health.recovery_attempts,
            "degradation_level": self.health.degradation_level,
            "success_rate": self.health.success_rate,
            "average_response_time": self.health.average_response_time,
            "overhead_percentage": self.health.overhead_percentage,
            "last_error": str(self.health.last_error) if self.health.last_error else None,
            "circuit_breaker_state": self.main_circuit_breaker.state.value,
            "framework_compatibility": {
                "status": self._framework_info.status.value if self._framework_info else "unknown",
                "version": str(self._framework_info.version) if self._framework_info and self._framework_info.version else "unknown"
            }
        }
    
    async def shutdown(self):
        """Shutdown the instrumentor gracefully."""
        try:
            logger.info(f"Shutting down robust instrumentor for {self.framework_name}")
            
            # Cancel recovery task if running
            if self._recovery_task and not self._recovery_task.done():
                self._recovery_task.cancel()
                try:
                    await self._recovery_task
                except asyncio.CancelledError:
                    pass
            
            # Call parent shutdown
            await super().stop()
            
            self.health.state = InstrumentorState.DISABLED
            logger.info(f"Robust instrumentor for {self.framework_name} shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")