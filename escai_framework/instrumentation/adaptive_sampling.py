"""
Adaptive sampling system for monitoring overhead optimization.

This module implements intelligent sampling strategies that automatically adjust
monitoring frequency based on system performance, agent behavior patterns,
and resource utilization to minimize overhead while maintaining observability.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
from collections import deque
import statistics
import threading

from ..utils.exceptions import MonitoringOverheadError
from ..utils.circuit_breaker import get_monitoring_circuit_breaker
from .events import AgentEvent, EventType, EventSeverity


logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Available sampling strategies."""
    FIXED_RATE = "fixed_rate"
    ADAPTIVE_RATE = "adaptive_rate"
    IMPORTANCE_BASED = "importance_based"
    PERFORMANCE_AWARE = "performance_aware"
    HYBRID = "hybrid"


class SamplingDecision(Enum):
    """Sampling decision outcomes."""
    SAMPLE = "sample"
    SKIP = "skip"
    PRIORITY_SAMPLE = "priority_sample"
    BATCH_SAMPLE = "batch_sample"


@dataclass
class SamplingMetrics:
    """Metrics for sampling performance."""
    total_events: int = 0
    sampled_events: int = 0
    skipped_events: int = 0
    priority_events: int = 0
    batch_events: int = 0
    overhead_ms: float = 0.0
    accuracy_score: float = 1.0
    
    def get_sampling_rate(self) -> float:
        """Calculate current sampling rate."""
        if self.total_events == 0:
            return 0.0
        return self.sampled_events / self.total_events
    
    def get_overhead_per_event(self) -> float:
        """Calculate overhead per sampled event."""
        if self.sampled_events == 0:
            return 0.0
        return self.overhead_ms / self.sampled_events


@dataclass
class SamplingConfig:
    """Configuration for adaptive sampling."""
    strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE_RATE
    base_sampling_rate: float = 1.0  # 100% by default
    min_sampling_rate: float = 0.1   # Never go below 10%
    max_sampling_rate: float = 1.0   # Never exceed 100%
    
    # Performance thresholds
    max_overhead_ms: float = 10.0    # Maximum overhead per event
    target_overhead_ms: float = 5.0  # Target overhead per event
    
    # Adaptation parameters
    adaptation_window: int = 100     # Events to consider for adaptation
    adaptation_factor: float = 0.1   # How aggressively to adapt (0.1 = 10% change)
    
    # Importance-based sampling
    high_priority_events: List[EventType] = field(default_factory=lambda: [
        EventType.AGENT_ERROR,
        EventType.TASK_FAIL,
        EventType.DECISION_START,
        EventType.DECISION_COMPLETE
    ])
    
    # Batch sampling
    batch_size: int = 10
    batch_timeout_ms: float = 1000.0


class SamplingDecisionMaker(ABC):
    """Abstract base class for sampling decision makers."""
    
    @abstractmethod
    async def should_sample(self, event: AgentEvent, context: Dict[str, Any]) -> SamplingDecision:
        """Decide whether to sample an event."""
        pass
    
    @abstractmethod
    def update_metrics(self, metrics: SamplingMetrics):
        """Update internal state based on current metrics."""
        pass
    
    @abstractmethod
    def get_current_rate(self) -> float:
        """Get current sampling rate."""
        pass


class FixedRateSampler(SamplingDecisionMaker):
    """Fixed rate sampling strategy."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.sampling_rate = config.base_sampling_rate
        self._counter = 0
    
    async def should_sample(self, event: AgentEvent, context: Dict[str, Any]) -> SamplingDecision:
        """Fixed rate sampling decision."""
        self._counter += 1
        
        # Always sample high priority events
        if event.event_type in self.config.high_priority_events:
            return SamplingDecision.PRIORITY_SAMPLE
        
        # Sample based on fixed rate
        if (self._counter % max(1, int(1.0 / self.sampling_rate))) == 0:
            return SamplingDecision.SAMPLE
        
        return SamplingDecision.SKIP
    
    def update_metrics(self, metrics: SamplingMetrics):
        """No adaptation for fixed rate sampler."""
        pass
    
    def get_current_rate(self) -> float:
        return self.sampling_rate


class AdaptiveRateSampler(SamplingDecisionMaker):
    """Adaptive rate sampling based on performance metrics."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.sampling_rate = config.base_sampling_rate
        self._recent_overheads: deque[float] = deque(maxlen=config.adaptation_window)
        self._counter = 0
        self._lock = threading.RLock()
    
    async def should_sample(self, event: AgentEvent, context: Dict[str, Any]) -> SamplingDecision:
        """Adaptive sampling decision."""
        with self._lock:
            self._counter += 1
            
            # Always sample high priority events
            if event.event_type in self.config.high_priority_events:
                return SamplingDecision.PRIORITY_SAMPLE
            
            # Sample based on current adaptive rate
            if (self._counter % max(1, int(1.0 / self.sampling_rate))) == 0:
                return SamplingDecision.SAMPLE
            
            return SamplingDecision.SKIP
    
    def update_metrics(self, metrics: SamplingMetrics):
        """Adapt sampling rate based on performance metrics."""
        with self._lock:
            overhead_per_event = metrics.get_overhead_per_event()
            self._recent_overheads.append(overhead_per_event)
            
            if len(self._recent_overheads) < 10:  # Need some data points
                return
            
            avg_overhead = statistics.mean(self._recent_overheads)
            
            # Adapt sampling rate based on overhead
            if avg_overhead > self.config.max_overhead_ms:
                # Reduce sampling rate
                new_rate = self.sampling_rate * (1 - self.config.adaptation_factor)
                self.sampling_rate = max(new_rate, self.config.min_sampling_rate)
                logger.info(f"Reduced sampling rate to {self.sampling_rate:.3f} due to high overhead ({avg_overhead:.2f}ms)")
                
            elif avg_overhead < self.config.target_overhead_ms:
                # Increase sampling rate
                new_rate = self.sampling_rate * (1 + self.config.adaptation_factor)
                self.sampling_rate = min(new_rate, self.config.max_sampling_rate)
                logger.debug(f"Increased sampling rate to {self.sampling_rate:.3f} due to low overhead ({avg_overhead:.2f}ms)")
    
    def get_current_rate(self) -> float:
        return self.sampling_rate


class ImportanceBasedSampler(SamplingDecisionMaker):
    """Importance-based sampling that prioritizes critical events."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.base_rate = config.base_sampling_rate
        self._event_importance = {
            EventType.AGENT_ERROR: 1.0,
            EventType.TASK_FAIL: 1.0,
            EventType.DECISION_START: 0.9,
            EventType.DECISION_COMPLETE: 0.9,
            EventType.TOOL_CALL: 0.7,
            EventType.TOOL_RESPONSE: 0.7,
            EventType.ACTION_START: 0.6,
            EventType.ACTION_COMPLETE: 0.6,
            EventType.AGENT_START: 0.8,
            EventType.AGENT_STOP: 0.8,
            EventType.CUSTOM: 0.3
        }
        self._counter = 0
    
    async def should_sample(self, event: AgentEvent, context: Dict[str, Any]) -> SamplingDecision:
        """Importance-based sampling decision."""
        self._counter += 1
        
        # Get importance score for this event type
        importance = self._event_importance.get(event.event_type, 0.5)
        
        # Adjust importance based on event severity
        if hasattr(event, 'severity'):
            if event.severity == EventSeverity.CRITICAL:
                importance = min(1.0, importance * 1.5)
            elif event.severity == EventSeverity.ERROR:
                importance = min(1.0, importance * 1.2)
            elif event.severity == EventSeverity.DEBUG:
                importance = importance * 0.5
        
        # Sample based on importance-weighted rate
        effective_rate = self.base_rate * importance
        
        if (self._counter % max(1, int(1.0 / effective_rate))) == 0:
            if importance >= 0.9:
                return SamplingDecision.PRIORITY_SAMPLE
            else:
                return SamplingDecision.SAMPLE
        
        return SamplingDecision.SKIP
    
    def update_metrics(self, metrics: SamplingMetrics):
        """Adjust base rate based on overall performance."""
        overhead_per_event = metrics.get_overhead_per_event()
        
        if overhead_per_event > self.config.max_overhead_ms:
            self.base_rate = max(
                self.base_rate * 0.9,
                self.config.min_sampling_rate
            )
        elif overhead_per_event < self.config.target_overhead_ms:
            self.base_rate = min(
                self.base_rate * 1.1,
                self.config.max_sampling_rate
            )
    
    def get_current_rate(self) -> float:
        return self.base_rate


class PerformanceAwareSampler(SamplingDecisionMaker):
    """Performance-aware sampling that considers system load."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.sampling_rate = config.base_sampling_rate
        self._system_load_history: deque[float] = deque(maxlen=50)
        self._counter = 0
        self._lock = threading.RLock()
    
    async def should_sample(self, event: AgentEvent, context: Dict[str, Any]) -> SamplingDecision:
        """Performance-aware sampling decision."""
        with self._lock:
            self._counter += 1
            
            # Always sample critical events
            if event.event_type in self.config.high_priority_events:
                return SamplingDecision.PRIORITY_SAMPLE
            
            # Get current system load from context
            system_load = context.get('system_load', 0.5)
            self._system_load_history.append(system_load)
            
            # Adjust sampling rate based on system load
            if system_load > 0.8:  # High load
                effective_rate = self.sampling_rate * 0.5
            elif system_load > 0.6:  # Medium load
                effective_rate = self.sampling_rate * 0.7
            else:  # Low load
                effective_rate = self.sampling_rate
            
            if (self._counter % max(1, int(1.0 / effective_rate))) == 0:
                return SamplingDecision.SAMPLE
            
            return SamplingDecision.SKIP
    
    def update_metrics(self, metrics: SamplingMetrics):
        """Adapt based on performance and system load."""
        with self._lock:
            if not self._system_load_history:
                return
            
            avg_load = statistics.mean(self._system_load_history)
            overhead_per_event = metrics.get_overhead_per_event()
            
            # Adapt sampling rate based on both load and overhead
            if overhead_per_event > self.config.max_overhead_ms or avg_load > 0.8:
                new_rate = self.sampling_rate * (1 - self.config.adaptation_factor)
                self.sampling_rate = max(new_rate, self.config.min_sampling_rate)
            elif overhead_per_event < self.config.target_overhead_ms and avg_load < 0.4:
                new_rate = self.sampling_rate * (1 + self.config.adaptation_factor)
                self.sampling_rate = min(new_rate, self.config.max_sampling_rate)
    
    def get_current_rate(self) -> float:
        return self.sampling_rate


class HybridSampler(SamplingDecisionMaker):
    """Hybrid sampler combining multiple strategies."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.adaptive_sampler = AdaptiveRateSampler(config)
        self.importance_sampler = ImportanceBasedSampler(config)
        self.performance_sampler = PerformanceAwareSampler(config)
    
    async def should_sample(self, event: AgentEvent, context: Dict[str, Any]) -> SamplingDecision:
        """Hybrid sampling decision using multiple strategies."""
        # Get decisions from all samplers
        adaptive_decision = await self.adaptive_sampler.should_sample(event, context)
        importance_decision = await self.importance_sampler.should_sample(event, context)
        performance_decision = await self.performance_sampler.should_sample(event, context)
        
        # Priority sampling if any sampler says so
        if any(d == SamplingDecision.PRIORITY_SAMPLE for d in [adaptive_decision, importance_decision, performance_decision]):
            return SamplingDecision.PRIORITY_SAMPLE
        
        # Sample if majority says sample
        sample_votes = sum(1 for d in [adaptive_decision, importance_decision, performance_decision] 
                          if d == SamplingDecision.SAMPLE)
        
        if sample_votes >= 2:
            return SamplingDecision.SAMPLE
        
        return SamplingDecision.SKIP
    
    def update_metrics(self, metrics: SamplingMetrics):
        """Update all component samplers."""
        self.adaptive_sampler.update_metrics(metrics)
        self.importance_sampler.update_metrics(metrics)
        self.performance_sampler.update_metrics(metrics)
    
    def get_current_rate(self) -> float:
        """Return average rate across samplers."""
        rates = [
            self.adaptive_sampler.get_current_rate(),
            self.importance_sampler.get_current_rate(),
            self.performance_sampler.get_current_rate()
        ]
        return statistics.mean(rates)


class AdaptiveSamplingManager:
    """Manages adaptive sampling for monitoring overhead optimization."""
    
    def __init__(self, config: SamplingConfig = None):
        self.config = config or SamplingConfig()
        self.metrics = SamplingMetrics()
        self._sampler = self._create_sampler()
        self._event_batch: List[AgentEvent] = []
        self._batch_timer = None
        self._lock = threading.RLock()
        self._circuit_breaker = get_monitoring_circuit_breaker()
        
        # Performance tracking
        self._last_update = time.time()
        self._update_interval = 10.0  # Update metrics every 10 seconds
    
    def _create_sampler(self) -> SamplingDecisionMaker:
        """Create appropriate sampler based on strategy."""
        if self.config.strategy == SamplingStrategy.FIXED_RATE:
            return FixedRateSampler(self.config)
        elif self.config.strategy == SamplingStrategy.ADAPTIVE_RATE:
            return AdaptiveRateSampler(self.config)
        elif self.config.strategy == SamplingStrategy.IMPORTANCE_BASED:
            return ImportanceBasedSampler(self.config)
        elif self.config.strategy == SamplingStrategy.PERFORMANCE_AWARE:
            return PerformanceAwareSampler(self.config)
        elif self.config.strategy == SamplingStrategy.HYBRID:
            return HybridSampler(self.config)
        else:
            return AdaptiveRateSampler(self.config)
    
    async def should_sample_event(self, event: AgentEvent, context: Dict[str, Any] = None) -> SamplingDecision:
        """
        Determine if an event should be sampled.
        
        Args:
            event: The event to consider for sampling
            context: Additional context (system load, etc.)
            
        Returns:
            SamplingDecision indicating what to do with the event
        """
        start_time = time.time()
        
        try:
            with self._lock:
                self.metrics.total_events += 1
                
                # Get sampling decision
                context = context or {}
                decision = await self._sampler.should_sample(event, context)
                
                # Update metrics based on decision
                if decision == SamplingDecision.SAMPLE:
                    self.metrics.sampled_events += 1
                elif decision == SamplingDecision.SKIP:
                    self.metrics.skipped_events += 1
                elif decision == SamplingDecision.PRIORITY_SAMPLE:
                    self.metrics.priority_events += 1
                    self.metrics.sampled_events += 1
                elif decision == SamplingDecision.BATCH_SAMPLE:
                    self.metrics.batch_events += 1
                    self.metrics.sampled_events += 1
                
                # Track overhead
                overhead_ms = (time.time() - start_time) * 1000
                self.metrics.overhead_ms += overhead_ms
                
                # Update sampler if enough time has passed
                if time.time() - self._last_update > self._update_interval:
                    self._sampler.update_metrics(self.metrics)
                    self._last_update = time.time()
                
                return decision
                
        except Exception as e:
            logger.error(f"Sampling decision failed: {e}")
            # Default to sampling on error
            return SamplingDecision.SAMPLE
    
    async def process_batch_sampling(self, events: List[AgentEvent]) -> List[AgentEvent]:
        """
        Process a batch of events for batch sampling.
        
        Args:
            events: List of events to process
            
        Returns:
            List of events that should be sampled
        """
        if not events:
            return []
        
        sampled_events = []
        
        for event in events:
            decision = await self.should_sample_event(event)
            if decision in [SamplingDecision.SAMPLE, SamplingDecision.PRIORITY_SAMPLE, SamplingDecision.BATCH_SAMPLE]:
                sampled_events.append(event)
        
        return sampled_events
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get current sampling statistics."""
        with self._lock:
            return {
                "total_events": self.metrics.total_events,
                "sampled_events": self.metrics.sampled_events,
                "skipped_events": self.metrics.skipped_events,
                "priority_events": self.metrics.priority_events,
                "batch_events": self.metrics.batch_events,
                "sampling_rate": self.metrics.get_sampling_rate(),
                "overhead_per_event_ms": self.metrics.get_overhead_per_event(),
                "current_sampler_rate": self._sampler.get_current_rate(),
                "strategy": self.config.strategy.value,
                "accuracy_score": self.metrics.accuracy_score
            }
    
    def update_configuration(self, new_config: SamplingConfig):
        """Update sampling configuration."""
        with self._lock:
            self.config = new_config
            self._sampler = self._create_sampler()
            logger.info(f"Updated sampling configuration: strategy={new_config.strategy.value}")
    
    def reset_metrics(self):
        """Reset sampling metrics."""
        with self._lock:
            self.metrics = SamplingMetrics()
            logger.info("Sampling metrics reset")
    
    def check_overhead_threshold(self) -> bool:
        """Check if overhead exceeds acceptable thresholds."""
        overhead_per_event = self.metrics.get_overhead_per_event()
        
        if overhead_per_event > self.config.max_overhead_ms:
            logger.warning(
                f"Sampling overhead threshold exceeded: {overhead_per_event:.2f}ms > {self.config.max_overhead_ms}ms"
            )
            return True
        
        return False
    
    async def optimize_sampling_rate(self):
        """Optimize sampling rate based on current performance."""
        if self.check_overhead_threshold():
            # Reduce sampling rate
            current_rate = self._sampler.get_current_rate()
            new_rate = max(current_rate * 0.8, self.config.min_sampling_rate)
            
            # Update configuration
            new_config = SamplingConfig(
                strategy=self.config.strategy,
                base_sampling_rate=new_rate,
                min_sampling_rate=self.config.min_sampling_rate,
                max_sampling_rate=self.config.max_sampling_rate,
                max_overhead_ms=self.config.max_overhead_ms,
                target_overhead_ms=self.config.target_overhead_ms
            )
            
            self.update_configuration(new_config)
            logger.info(f"Optimized sampling rate to {new_rate:.3f}")


# Global adaptive sampling manager
_sampling_manager = AdaptiveSamplingManager()


def get_sampling_manager() -> AdaptiveSamplingManager:
    """Get the global adaptive sampling manager."""
    return _sampling_manager


async def should_sample_event(event: AgentEvent, context: Dict[str, Any] = None) -> SamplingDecision:
    """Check if an event should be sampled using the global manager."""
    return await _sampling_manager.should_sample_event(event, context)


def configure_adaptive_sampling(config: SamplingConfig):
    """Configure adaptive sampling using the global manager."""
    _sampling_manager.update_configuration(config)