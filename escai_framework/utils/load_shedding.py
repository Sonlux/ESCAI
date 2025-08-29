"""
Graceful degradation and load shedding for system overload scenarios.

This module implements load shedding strategies to maintain system stability
during high load conditions by selectively reducing functionality.
"""

import asyncio
import logging
import psutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from threading import Lock
from collections import deque, defaultdict

from .exceptions import ESCAIBaseException, ErrorSeverity


logger = logging.getLogger(__name__)


class LoadLevel(Enum):
    """System load levels for load shedding decisions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Priority(Enum):
    """Priority levels for different operations."""
    CRITICAL = 1    # Core monitoring functionality
    HIGH = 2        # Important analysis features
    MEDIUM = 3      # Standard features
    LOW = 4         # Nice-to-have features
    OPTIONAL = 5    # Non-essential features


@dataclass
class SystemMetrics:
    """Current system performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: float = 0.0
    active_connections: int = 0
    queue_size: int = 0
    response_time: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class LoadThresholds:
    """Thresholds for different load levels."""
    cpu_medium: float = 70.0
    cpu_high: float = 85.0
    cpu_critical: float = 95.0
    
    memory_medium: float = 70.0
    memory_high: float = 85.0
    memory_critical: float = 95.0
    
    response_time_medium: float = 1.0
    response_time_high: float = 3.0
    response_time_critical: float = 10.0
    
    error_rate_medium: float = 0.05
    error_rate_high: float = 0.15
    error_rate_critical: float = 0.30
    
    queue_size_medium: int = 100
    queue_size_high: int = 500
    queue_size_critical: int = 1000


@dataclass
class LoadSheddingRule:
    """Rule for load shedding decisions."""
    name: str
    load_level: LoadLevel
    affected_priorities: Set[Priority]
    shed_percentage: float  # Percentage of requests to shed (0.0 to 1.0)
    description: str


class LoadSheddingError(ESCAIBaseException):
    """Raised when a request is shed due to high load."""
    
    def __init__(self, load_level: LoadLevel, priority: Priority):
        super().__init__(
            f"Request shed due to {load_level.value} load (priority: {priority.value})",
            severity=ErrorSeverity.LOW,
            recovery_hint="Retry request later when system load decreases"
        )
        self.load_level = load_level
        self.priority = priority


class SystemMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self._metrics_history: deque = deque(maxlen=100)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start_monitoring(self) -> None:
        """Start system monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = await self._collect_metrics()
                self._metrics_history.append(metrics)
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Network I/O (simplified)
        network_io = 0.0
        try:
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                bytes_sent_diff = net_io.bytes_sent - self._last_net_io.bytes_sent
                bytes_recv_diff = net_io.bytes_recv - self._last_net_io.bytes_recv
                network_io = (bytes_sent_diff + bytes_recv_diff) / self.monitoring_interval
            self._last_net_io = net_io
        except Exception:
            pass
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            active_connections=0,  # Will be updated by connection manager
            queue_size=0,  # Will be updated by queue manager
            response_time=0.0,  # Will be updated by request handler
            error_rate=0.0  # Will be updated by error tracker
        )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent metrics."""
        return self._metrics_history[-1] if self._metrics_history else None
    
    def get_average_metrics(self, window_size: int = 10) -> Optional[SystemMetrics]:
        """Get average metrics over a window."""
        if not self._metrics_history:
            return None
        
        recent_metrics = list(self._metrics_history)[-window_size:]
        
        return SystemMetrics(
            cpu_usage=sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            memory_usage=sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            disk_usage=sum(m.disk_usage for m in recent_metrics) / len(recent_metrics),
            network_io=sum(m.network_io for m in recent_metrics) / len(recent_metrics),
            active_connections=recent_metrics[-1].active_connections,
            queue_size=recent_metrics[-1].queue_size,
            response_time=sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            error_rate=sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        )


class LoadShedder:
    """Implements load shedding strategies for graceful degradation."""
    
    def __init__(self, thresholds: LoadThresholds = None):
        self.thresholds = thresholds or LoadThresholds()
        self.system_monitor = SystemMonitor()
        self._shed_counters: Dict[Priority, int] = defaultdict(int)
        self._total_requests: Dict[Priority, int] = defaultdict(int)
        self._lock = Lock()
        
        # Default load shedding rules
        self.rules = [
            LoadSheddingRule(
                name="critical_load_optional",
                load_level=LoadLevel.CRITICAL,
                affected_priorities={Priority.OPTIONAL, Priority.LOW},
                shed_percentage=1.0,
                description="Shed all optional and low priority requests during critical load"
            ),
            LoadSheddingRule(
                name="critical_load_medium",
                load_level=LoadLevel.CRITICAL,
                affected_priorities={Priority.MEDIUM},
                shed_percentage=0.8,
                description="Shed 80% of medium priority requests during critical load"
            ),
            LoadSheddingRule(
                name="high_load_optional",
                load_level=LoadLevel.HIGH,
                affected_priorities={Priority.OPTIONAL},
                shed_percentage=1.0,
                description="Shed all optional requests during high load"
            ),
            LoadSheddingRule(
                name="high_load_low",
                load_level=LoadLevel.HIGH,
                affected_priorities={Priority.LOW},
                shed_percentage=0.7,
                description="Shed 70% of low priority requests during high load"
            ),
            LoadSheddingRule(
                name="medium_load_optional",
                load_level=LoadLevel.MEDIUM,
                affected_priorities={Priority.OPTIONAL},
                shed_percentage=0.5,
                description="Shed 50% of optional requests during medium load"
            )
        ]
    
    async def start(self) -> None:
        """Start the load shedder."""
        await self.system_monitor.start_monitoring()
        logger.info("Load shedder started")
    
    async def stop(self) -> None:
        """Stop the load shedder."""
        await self.system_monitor.stop_monitoring()
        logger.info("Load shedder stopped")
    
    def determine_load_level(self, metrics: SystemMetrics) -> LoadLevel:
        """Determine current system load level based on metrics."""
        # Check critical thresholds
        if (metrics.cpu_usage >= self.thresholds.cpu_critical or
            metrics.memory_usage >= self.thresholds.memory_critical or
            metrics.response_time >= self.thresholds.response_time_critical or
            metrics.error_rate >= self.thresholds.error_rate_critical or
            metrics.queue_size >= self.thresholds.queue_size_critical):
            return LoadLevel.CRITICAL
        
        # Check high thresholds
        if (metrics.cpu_usage >= self.thresholds.cpu_high or
            metrics.memory_usage >= self.thresholds.memory_high or
            metrics.response_time >= self.thresholds.response_time_high or
            metrics.error_rate >= self.thresholds.error_rate_high or
            metrics.queue_size >= self.thresholds.queue_size_high):
            return LoadLevel.HIGH
        
        # Check medium thresholds
        if (metrics.cpu_usage >= self.thresholds.cpu_medium or
            metrics.memory_usage >= self.thresholds.memory_medium or
            metrics.response_time >= self.thresholds.response_time_medium or
            metrics.error_rate >= self.thresholds.error_rate_medium or
            metrics.queue_size >= self.thresholds.queue_size_medium):
            return LoadLevel.MEDIUM
        
        return LoadLevel.LOW
    
    def should_shed_request(self, priority: Priority) -> bool:
        """Determine if a request should be shed based on current load and priority."""
        metrics = self.system_monitor.get_current_metrics()
        if not metrics:
            return False
        
        load_level = self.determine_load_level(metrics)
        
        # Track total requests
        with self._lock:
            self._total_requests[priority] += 1
        
        # Find applicable rules
        applicable_rules = [
            rule for rule in self.rules
            if rule.load_level == load_level and priority in rule.affected_priorities
        ]
        
        if not applicable_rules:
            return False
        
        # Use the most restrictive rule
        max_shed_percentage = max(rule.shed_percentage for rule in applicable_rules)
        
        # Simple probabilistic shedding
        import random
        should_shed = random.random() < max_shed_percentage
        
        if should_shed:
            with self._lock:
                self._shed_counters[priority] += 1
            
            logger.warning(
                f"Shedding {priority.name} priority request due to {load_level.value} load "
                f"(CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%)"
            )
        
        return should_shed
    
    async def execute_with_load_shedding(
        self,
        func: Callable,
        priority: Priority,
        *args,
        **kwargs
    ) -> Any:
        """Execute a function with load shedding protection."""
        if self.should_shed_request(priority):
            metrics = self.system_monitor.get_current_metrics()
            load_level = self.determine_load_level(metrics) if metrics else LoadLevel.LOW
            raise LoadSheddingError(load_level, priority)
        
        return await func(*args, **kwargs)
    
    def get_shedding_stats(self) -> Dict[str, Any]:
        """Get load shedding statistics."""
        with self._lock:
            stats = {
                "total_requests": dict(self._total_requests),
                "shed_requests": dict(self._shed_counters),
                "shed_rates": {}
            }
            
            for priority in Priority:
                total = self._total_requests.get(priority, 0)
                shed = self._shed_counters.get(priority, 0)
                stats["shed_rates"][priority.name] = shed / total if total > 0 else 0.0
        
        # Add current system metrics
        current_metrics = self.system_monitor.get_current_metrics()
        if current_metrics:
            stats["current_load_level"] = self.determine_load_level(current_metrics).value
            stats["current_metrics"] = {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "response_time": current_metrics.response_time,
                "error_rate": current_metrics.error_rate,
                "queue_size": current_metrics.queue_size
            }
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset shedding statistics."""
        with self._lock:
            self._shed_counters.clear()
            self._total_requests.clear()
        logger.info("Load shedding statistics reset")
    
    def add_rule(self, rule: LoadSheddingRule) -> None:
        """Add a custom load shedding rule."""
        self.rules.append(rule)
        logger.info(f"Added load shedding rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> None:
        """Remove a load shedding rule by name."""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        logger.info(f"Removed load shedding rule: {rule_name}")


class GracefulDegradationManager:
    """Manages graceful degradation of system functionality."""
    
    def __init__(self):
        self.load_shedder = LoadShedder()
        self._degraded_features: Set[str] = set()
        self._feature_priorities: Dict[str, Priority] = {}
        self._lock = Lock()
    
    async def start(self):
        """Start the degradation manager."""
        await self.load_shedder.start()
        logger.info("Graceful degradation manager started")
    
    async def stop(self):
        """Stop the degradation manager."""
        await self.load_shedder.stop()
        logger.info("Graceful degradation manager stopped")
    
    def register_feature(self, feature_name: str, priority: Priority):
        """Register a feature with its priority level."""
        with self._lock:
            self._feature_priorities[feature_name] = priority
        logger.info(f"Registered feature '{feature_name}' with priority {priority.name}")
    
    def is_feature_available(self, feature_name: str) -> bool:
        """Check if a feature is currently available."""
        with self._lock:
            return feature_name not in self._degraded_features
    
    def degrade_feature(self, feature_name: str, reason: str = "High system load"):
        """Temporarily degrade a feature."""
        with self._lock:
            self._degraded_features.add(feature_name)
        logger.warning(f"Feature '{feature_name}' degraded: {reason}")
    
    def restore_feature(self, feature_name: str):
        """Restore a degraded feature."""
        with self._lock:
            self._degraded_features.discard(feature_name)
        logger.info(f"Feature '{feature_name}' restored")
    
    async def execute_feature(
        self,
        feature_name: str,
        func: Callable,
        fallback_func: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute a feature with degradation support."""
        # Check if feature is available
        if not self.is_feature_available(feature_name):
            if fallback_func:
                logger.info(f"Using fallback for degraded feature '{feature_name}'")
                return await fallback_func(*args, **kwargs)
            else:
                raise ESCAIBaseException(
                    f"Feature '{feature_name}' is currently unavailable due to system degradation",
                    severity=ErrorSeverity.MEDIUM,
                    recovery_hint="Try again later when system load decreases"
                )
        
        # Get feature priority
        priority = self._feature_priorities.get(feature_name, Priority.MEDIUM)
        
        # Execute with load shedding
        try:
            return await self.load_shedder.execute_with_load_shedding(
                func, priority, *args, **kwargs
            )
        except LoadSheddingError:
            # If load shedding occurs, degrade the feature temporarily
            self.degrade_feature(feature_name, "Load shedding triggered")
            
            if fallback_func:
                logger.info(f"Using fallback for load-shed feature '{feature_name}'")
                return await fallback_func(*args, **kwargs)
            else:
                raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        with self._lock:
            degraded_features = list(self._degraded_features)
        
        return {
            "degraded_features": degraded_features,
            "total_features": len(self._feature_priorities),
            "load_shedding_stats": self.load_shedder.get_shedding_stats(),
            "feature_priorities": {
                name: priority.name for name, priority in self._feature_priorities.items()
            }
        }


# Global degradation manager instance
_degradation_manager = GracefulDegradationManager()


def get_degradation_manager() -> GracefulDegradationManager:
    """Get the global degradation manager instance."""
    return _degradation_manager


async def execute_with_degradation(
    feature_name: str,
    func: Callable,
    fallback_func: Optional[Callable] = None,
    *args,
    **kwargs
) -> Any:
    """Execute a function with graceful degradation support."""
    return await _degradation_manager.execute_feature(
        feature_name, func, fallback_func, *args, **kwargs
    )