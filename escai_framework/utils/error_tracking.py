"""
Comprehensive logging and monitoring for error tracking and debugging.

This module provides structured error tracking, metrics collection, and
monitoring capabilities for debugging and system health monitoring.
"""

import asyncio
import json
import logging
import traceback
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from collections import defaultdict, deque
from threading import Lock
import uuid

from .exceptions import ESCAIBaseException, ErrorSeverity, ErrorCategory


class LogLevel(Enum):
    """Extended log levels for error tracking."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorEvent:
    """Structured error event for tracking."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    level: LogLevel = LogLevel.ERROR
    message: str = ""
    exception_type: str = ""
    exception_message: str = ""
    stack_trace: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Error classification
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.PROCESSING
    
    # System context
    component: str = ""
    function_name: str = ""
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Performance metrics
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    # Recovery information
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        data['severity'] = self.severity.value
        data['category'] = self.category.value
        return data
    
    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        context: Dict[str, Any] = None,
        component: str = "",
        function_name: str = ""
    ) -> 'ErrorEvent':
        """Create an ErrorEvent from an exception."""
        # Determine severity and category
        if isinstance(exception, ESCAIBaseException):
            severity = exception.severity
            category = exception.category or ErrorCategory.PROCESSING
        else:
            severity = ErrorSeverity.MEDIUM
            category = ErrorCategory.PROCESSING
        
        return cls(
            level=LogLevel.ERROR,
            message=str(exception),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            context=context or {},
            severity=severity,
            category=category,
            component=component,
            function_name=function_name
        )


@dataclass
class ErrorMetrics:
    """Aggregated error metrics."""
    total_errors: int = 0
    errors_by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_category: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_component: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_rate: float = 0.0
    mean_time_between_errors: float = 0.0
    last_error_time: Optional[datetime] = None
    
    def update_from_event(self, event: ErrorEvent):
        """Update metrics from an error event."""
        self.total_errors += 1
        self.errors_by_severity[event.severity.value] += 1
        self.errors_by_category[event.category.value] += 1
        if event.component:
            self.errors_by_component[event.component] += 1
        self.last_error_time = event.timestamp


class ErrorTracker:
    """Tracks and analyzes error patterns."""
    
    def __init__(self, max_events: int = 10000, metrics_window_hours: int = 24):
        self.max_events = max_events
        self.metrics_window_hours = metrics_window_hours
        
        self._events: deque = deque(maxlen=max_events)
        self._metrics = ErrorMetrics()
        self._lock = Lock()
        
        # Pattern detection
        self._error_patterns: Dict[str, List[ErrorEvent]] = defaultdict(list)
        self._recurring_errors: Dict[str, int] = defaultdict(int)
        
        # Alerting thresholds
        self.alert_thresholds = {
            "error_rate_per_minute": 10,
            "critical_errors_per_hour": 5,
            "recurring_error_threshold": 5
        }
        
        self._alert_callbacks: List[Callable] = []
    
    def track_error(self, event: ErrorEvent):
        """Track an error event."""
        with self._lock:
            self._events.append(event)
            self._metrics.update_from_event(event)
            
            # Pattern detection
            pattern_key = f"{event.exception_type}:{event.component}"
            self._error_patterns[pattern_key].append(event)
            self._recurring_errors[pattern_key] += 1
            
            # Check for alerts
            self._check_alerts(event)
    
    def track_exception(
        self,
        exception: Exception,
        context: Dict[str, Any] = None,
        component: str = "",
        function_name: str = ""
    ) -> str:
        """Track an exception and return the event ID."""
        event = ErrorEvent.from_exception(exception, context, component, function_name)
        self.track_error(event)
        return event.event_id
    
    def _check_alerts(self, event: ErrorEvent):
        """Check if any alert conditions are met."""
        # Check error rate
        recent_errors = self._get_recent_errors(minutes=1)
        if len(recent_errors) >= self.alert_thresholds["error_rate_per_minute"]:
            self._trigger_alert("high_error_rate", {
                "error_count": len(recent_errors),
                "threshold": self.alert_thresholds["error_rate_per_minute"],
                "window": "1 minute"
            })
        
        # Check critical errors
        if event.severity == ErrorSeverity.CRITICAL:
            recent_critical = self._get_recent_errors(hours=1, severity=ErrorSeverity.CRITICAL)
            if len(recent_critical) >= self.alert_thresholds["critical_errors_per_hour"]:
                self._trigger_alert("high_critical_errors", {
                    "critical_count": len(recent_critical),
                    "threshold": self.alert_thresholds["critical_errors_per_hour"],
                    "window": "1 hour"
                })
        
        # Check recurring errors
        pattern_key = f"{event.exception_type}:{event.component}"
        if self._recurring_errors[pattern_key] >= self.alert_thresholds["recurring_error_threshold"]:
            self._trigger_alert("recurring_error", {
                "pattern": pattern_key,
                "count": self._recurring_errors[pattern_key],
                "threshold": self.alert_thresholds["recurring_error_threshold"]
            })
    
    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger an alert."""
        alert_data = {
            "type": alert_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details
        }
        
        for callback in self._alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function."""
        self._alert_callbacks.append(callback)
    
    def _get_recent_errors(
        self,
        minutes: int = None,
        hours: int = None,
        severity: ErrorSeverity = None
    ) -> List[ErrorEvent]:
        """Get recent errors within a time window."""
        if minutes:
            cutoff = datetime.now(timezone.utc).timestamp() - (minutes * 60)
        elif hours:
            cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        else:
            return []
        
        recent_events = []
        for event in reversed(self._events):
            if event.timestamp.timestamp() < cutoff:
                break
            if severity is None or event.severity == severity:
                recent_events.append(event)
        
        return recent_events
    
    def get_error_patterns(self, min_occurrences: int = 3) -> Dict[str, Dict[str, Any]]:
        """Get detected error patterns."""
        patterns = {}
        
        for pattern_key, events in self._error_patterns.items():
            if len(events) >= min_occurrences:
                patterns[pattern_key] = {
                    "count": len(events),
                    "first_occurrence": events[0].timestamp.isoformat(),
                    "last_occurrence": events[-1].timestamp.isoformat(),
                    "components": list(set(e.component for e in events if e.component)),
                    "severity_distribution": {
                        severity.value: sum(1 for e in events if e.severity == severity)
                        for severity in ErrorSeverity
                    }
                }
        
        return patterns
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current error metrics."""
        with self._lock:
            return {
                "total_errors": self._metrics.total_errors,
                "errors_by_severity": dict(self._metrics.errors_by_severity),
                "errors_by_category": dict(self._metrics.errors_by_category),
                "errors_by_component": dict(self._metrics.errors_by_component),
                "error_rate_last_hour": len(self._get_recent_errors(hours=1)),
                "error_rate_last_minute": len(self._get_recent_errors(minutes=1)),
                "last_error_time": self._metrics.last_error_time.isoformat() if self._metrics.last_error_time else None,
                "patterns_detected": len(self.get_error_patterns())
            }
    
    def get_events(
        self,
        limit: int = 100,
        severity: ErrorSeverity = None,
        category: ErrorCategory = None,
        component: str = None
    ) -> List[ErrorEvent]:
        """Get error events with optional filtering."""
        events = []
        count = 0
        
        for event in reversed(self._events):
            if count >= limit:
                break
            
            if severity and event.severity != severity:
                continue
            if category and event.category != category:
                continue
            if component and event.component != component:
                continue
            
            events.append(event)
            count += 1
        
        return events
    
    def clear_events(self):
        """Clear all tracked events."""
        with self._lock:
            self._events.clear()
            self._error_patterns.clear()
            self._recurring_errors.clear()
            self._metrics = ErrorMetrics()


class StructuredLogger:
    """Structured logger with error tracking integration."""
    
    def __init__(self, name: str, error_tracker: ErrorTracker = None):
        self.logger = logging.getLogger(name)
        self.error_tracker = error_tracker
        self.component = name
        
        # Configure structured logging
        self._setup_structured_logging()
    
    def _setup_structured_logging(self):
        """Setup structured logging format."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _log_with_context(
        self,
        level: LogLevel,
        message: str,
        context: Dict[str, Any] = None,
        exception: Exception = None,
        function_name: str = ""
    ):
        """Log with structured context."""
        # Create log entry
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "component": self.component,
            "function": function_name,
            "message": message,
            "context": context or {}
        }
        
        if exception:
            log_data["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        # Log to standard logger
        log_message = json.dumps(log_data, default=str)
        
        if level == LogLevel.CRITICAL:
            self.logger.critical(log_message)
        elif level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif level == LogLevel.INFO:
            self.logger.info(log_message)
        elif level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == LogLevel.TRACE:
            self.logger.debug(f"TRACE: {log_message}")
        
        # Track errors
        if level in [LogLevel.ERROR, LogLevel.CRITICAL] and self.error_tracker:
            if exception:
                self.error_tracker.track_exception(
                    exception, context, self.component, function_name
                )
            else:
                # Create error event from log message
                event = ErrorEvent(
                    level=level,
                    message=message,
                    context=context or {},
                    component=self.component,
                    function_name=function_name,
                    severity=ErrorSeverity.CRITICAL if level == LogLevel.CRITICAL else ErrorSeverity.HIGH
                )
                self.error_tracker.track_error(event)
    
    def trace(self, message: str, context: Dict[str, Any] = None, function_name: str = ""):
        """Log trace message."""
        self._log_with_context(LogLevel.TRACE, message, context, function_name=function_name)
    
    def debug(self, message: str, context: Dict[str, Any] = None, function_name: str = ""):
        """Log debug message."""
        self._log_with_context(LogLevel.DEBUG, message, context, function_name=function_name)
    
    def info(self, message: str, context: Dict[str, Any] = None, function_name: str = ""):
        """Log info message."""
        self._log_with_context(LogLevel.INFO, message, context, function_name=function_name)
    
    def warning(self, message: str, context: Dict[str, Any] = None, function_name: str = ""):
        """Log warning message."""
        self._log_with_context(LogLevel.WARNING, message, context, function_name=function_name)
    
    def error(self, message: str, context: Dict[str, Any] = None, exception: Exception = None, function_name: str = ""):
        """Log error message."""
        self._log_with_context(LogLevel.ERROR, message, context, exception, function_name)
    
    def critical(self, message: str, context: Dict[str, Any] = None, exception: Exception = None, function_name: str = ""):
        """Log critical message."""
        self._log_with_context(LogLevel.CRITICAL, message, context, exception, function_name)
    
    def exception(self, message: str, exception: Exception, context: Dict[str, Any] = None, function_name: str = ""):
        """Log exception with full traceback."""
        self.error(message, context, exception, function_name)


class MonitoringDecorator:
    """Decorator for automatic error tracking and performance monitoring."""
    
    def __init__(self, error_tracker: ErrorTracker, component: str = ""):
        self.error_tracker = error_tracker
        self.component = component
    
    def __call__(self, func: Callable):
        """Decorate function with monitoring."""
        if asyncio.iscoroutinefunction(func):
            return self._wrap_async(func)
        else:
            return self._wrap_sync(func)
    
    def _wrap_async(self, func: Callable):
        """Wrap async function."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            
            try:
                result = await func(*args, **kwargs)
                
                # Track successful execution
                execution_time = time.time() - start_time
                event = ErrorEvent(
                    level=LogLevel.TRACE,
                    message=f"Function {function_name} completed successfully",
                    component=self.component,
                    function_name=function_name,
                    execution_time=execution_time
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Track error
                event_id = self.error_tracker.track_exception(
                    e,
                    context={
                        "function_args": str(args)[:200],  # Truncate for privacy
                        "function_kwargs": str(kwargs)[:200],
                        "execution_time": execution_time
                    },
                    component=self.component,
                    function_name=function_name
                )
                
                # Re-raise the exception
                raise e
        
        return wrapper
    
    def _wrap_sync(self, func: Callable):
        """Wrap sync function."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            
            try:
                result = func(*args, **kwargs)
                
                # Track successful execution
                execution_time = time.time() - start_time
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Track error
                event_id = self.error_tracker.track_exception(
                    e,
                    context={
                        "function_args": str(args)[:200],  # Truncate for privacy
                        "function_kwargs": str(kwargs)[:200],
                        "execution_time": execution_time
                    },
                    component=self.component,
                    function_name=function_name
                )
                
                # Re-raise the exception
                raise e
        
        return wrapper


# Global error tracker instance
_global_error_tracker = ErrorTracker()


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance."""
    return _global_error_tracker


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger with error tracking."""
    return StructuredLogger(name, _global_error_tracker)


def monitor_errors(component: str = ""):
    """Decorator for automatic error monitoring."""
    return MonitoringDecorator(_global_error_tracker, component)


def track_exception(
    exception: Exception,
    context: Dict[str, Any] = None,
    component: str = "",
    function_name: str = ""
) -> str:
    """Track an exception globally."""
    return _global_error_tracker.track_exception(exception, context, component, function_name)