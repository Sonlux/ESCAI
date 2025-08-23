"""
InfluxDB data models for time-series metrics.

This module defines structured models for different types of metrics
stored in InfluxDB, providing type safety and validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum


class MetricType(Enum):
    """Enumeration of supported metric types."""
    AGENT_PERFORMANCE = "agent_performance"
    API_METRICS = "api_metrics"
    SYSTEM_METRICS = "system_metrics"
    PREDICTION_METRICS = "prediction_metrics"
    PATTERN_METRICS = "pattern_metrics"
    CAUSAL_METRICS = "causal_metrics"
    EPISTEMIC_METRICS = "epistemic_metrics"


@dataclass
class BaseMetric:
    """Base class for all metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_influx_point(self) -> Dict[str, Any]:
        """Convert to InfluxDB point format."""
        return {
            "measurement": self.get_measurement_name(),
            "tags": self.tags,
            "fields": self.get_fields(),
            "time": self.timestamp
        }
    
    def get_measurement_name(self) -> str:
        """Get the measurement name for this metric."""
        raise NotImplementedError
    
    def get_fields(self) -> Dict[str, Union[float, int, str, bool]]:
        """Get the field values for this metric."""
        raise NotImplementedError


@dataclass
class AgentPerformanceMetric(BaseMetric):
    """Metric for agent execution performance."""
    agent_id: str = ""
    session_id: str = ""
    framework: str = ""
    task_type: str = ""
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    success: bool = False
    error_count: int = 0
    retry_count: int = 0
    
    def __post_init__(self):
        """Set tags after initialization."""
        self.tags.update({
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "framework": self.framework,
            "task_type": self.task_type
        })
    
    def get_measurement_name(self) -> str:
        return MetricType.AGENT_PERFORMANCE.value
    
    def get_fields(self) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "success": self.success,
            "error_count": self.error_count,
            "retry_count": self.retry_count
        }


@dataclass
class APIMetric(BaseMetric):
    """Metric for API endpoint performance."""
    endpoint: str = ""
    method: str = ""
    status_code: int = 200
    response_time_ms: float = 0.0
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    user_id: Optional[str] = None
    
    def __post_init__(self):
        """Set tags after initialization."""
        self.tags.update({
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": str(self.status_code)
        })
        if self.user_id:
            self.tags["user_id"] = self.user_id
    
    def get_measurement_name(self) -> str:
        return MetricType.API_METRICS.value
    
    def get_fields(self) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "response_time_ms": self.response_time_ms,
            "request_size_bytes": self.request_size_bytes,
            "response_size_bytes": self.response_size_bytes
        }


@dataclass
class SystemMetric(BaseMetric):
    """Metric for system resource usage."""
    component: str = ""
    instance: str = ""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes: int = 0
    active_connections: int = 0
    queue_size: int = 0
    
    def __post_init__(self):
        """Set tags after initialization."""
        self.tags.update({
            "component": self.component,
            "instance": self.instance
        })
    
    def get_measurement_name(self) -> str:
        return MetricType.SYSTEM_METRICS.value
    
    def get_fields(self) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_usage_percent": self.disk_usage_percent,
            "network_io_bytes": self.network_io_bytes,
            "active_connections": self.active_connections,
            "queue_size": self.queue_size
        }


@dataclass
class PredictionMetric(BaseMetric):
    """Metric for prediction model performance."""
    agent_id: str = ""
    model_type: str = ""
    prediction_type: str = ""
    accuracy: float = 0.0
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    feature_count: int = 0
    prediction_value: Optional[float] = None
    actual_value: Optional[float] = None
    
    def __post_init__(self):
        """Set tags after initialization."""
        self.tags.update({
            "agent_id": self.agent_id,
            "model_type": self.model_type,
            "prediction_type": self.prediction_type
        })
    
    def get_measurement_name(self) -> str:
        return MetricType.PREDICTION_METRICS.value
    
    def get_fields(self) -> Dict[str, Union[float, int, str, bool]]:
        fields = {
            "accuracy": self.accuracy,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "feature_count": self.feature_count
        }
        if self.prediction_value is not None:
            fields["prediction_value"] = self.prediction_value
        if self.actual_value is not None:
            fields["actual_value"] = self.actual_value
        return fields


@dataclass
class PatternMetric(BaseMetric):
    """Metric for behavioral pattern analysis."""
    agent_id: str = ""
    pattern_type: str = ""
    framework: str = ""
    pattern_frequency: int = 0
    success_rate: float = 0.0
    average_duration_ms: float = 0.0
    anomaly_score: float = 0.0
    pattern_id: Optional[str] = None
    
    def __post_init__(self):
        """Set tags after initialization."""
        self.tags.update({
            "agent_id": self.agent_id,
            "pattern_type": self.pattern_type,
            "framework": self.framework
        })
        if self.pattern_id:
            self.tags["pattern_id"] = self.pattern_id
    
    def get_measurement_name(self) -> str:
        return MetricType.PATTERN_METRICS.value
    
    def get_fields(self) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "pattern_frequency": self.pattern_frequency,
            "success_rate": self.success_rate,
            "average_duration_ms": self.average_duration_ms,
            "anomaly_score": self.anomaly_score
        }


@dataclass
class CausalMetric(BaseMetric):
    """Metric for causal relationship analysis."""
    agent_id: str = ""
    cause_event: str = ""
    effect_event: str = ""
    causal_strength: float = 0.0
    confidence: float = 0.0
    delay_ms: int = 0
    statistical_significance: float = 0.0
    
    def __post_init__(self):
        """Set tags after initialization."""
        self.tags.update({
            "agent_id": self.agent_id,
            "cause_event": self.cause_event,
            "effect_event": self.effect_event
        })
    
    def get_measurement_name(self) -> str:
        return MetricType.CAUSAL_METRICS.value
    
    def get_fields(self) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "causal_strength": self.causal_strength,
            "confidence": self.confidence,
            "delay_ms": self.delay_ms,
            "statistical_significance": self.statistical_significance
        }


@dataclass
class EpistemicMetric(BaseMetric):
    """Metric for epistemic state changes."""
    agent_id: str = ""
    session_id: str = ""
    belief_count: int = 0
    knowledge_items: int = 0
    goal_count: int = 0
    confidence_level: float = 0.0
    uncertainty_score: float = 0.0
    state_change_magnitude: float = 0.0
    
    def __post_init__(self):
        """Set tags after initialization."""
        self.tags.update({
            "agent_id": self.agent_id,
            "session_id": self.session_id
        })
    
    def get_measurement_name(self) -> str:
        return MetricType.EPISTEMIC_METRICS.value
    
    def get_fields(self) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "belief_count": self.belief_count,
            "knowledge_items": self.knowledge_items,
            "goal_count": self.goal_count,
            "confidence_level": self.confidence_level,
            "uncertainty_score": self.uncertainty_score,
            "state_change_magnitude": self.state_change_magnitude
        }


@dataclass
class MetricBatch:
    """Container for batch metric operations."""
    metrics: List[BaseMetric] = field(default_factory=list)
    
    def add_metric(self, metric: BaseMetric) -> None:
        """Add a metric to the batch."""
        self.metrics.append(metric)
    
    def add_metrics(self, metrics: List[BaseMetric]) -> None:
        """Add multiple metrics to the batch."""
        self.metrics.extend(metrics)
    
    def clear(self) -> None:
        """Clear all metrics from the batch."""
        self.metrics.clear()
    
    def size(self) -> int:
        """Get the number of metrics in the batch."""
        return len(self.metrics)
    
    def to_influx_points(self) -> List[Dict[str, Any]]:
        """Convert all metrics to InfluxDB point format."""
        return [metric.to_influx_point() for metric in self.metrics]
    
    def filter_by_type(self, metric_type: MetricType) -> List[BaseMetric]:
        """Filter metrics by type."""
        return [
            metric for metric in self.metrics
            if metric.get_measurement_name() == metric_type.value
        ]


# Utility functions for creating common metrics
def create_agent_performance_metric(
    agent_id: str,
    session_id: str,
    framework: str,
    task_type: str,
    execution_time_ms: float,
    memory_usage_mb: float,
    cpu_usage_percent: float,
    success: bool,
    error_count: int = 0,
    retry_count: int = 0,
    timestamp: Optional[datetime] = None
) -> AgentPerformanceMetric:
    """Create an agent performance metric."""
    return AgentPerformanceMetric(
        agent_id=agent_id,
        session_id=session_id,
        framework=framework,
        task_type=task_type,
        execution_time_ms=execution_time_ms,
        memory_usage_mb=memory_usage_mb,
        cpu_usage_percent=cpu_usage_percent,
        success=success,
        error_count=error_count,
        retry_count=retry_count,
        timestamp=timestamp or datetime.utcnow()
    )


def create_api_metric(
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: float,
    request_size_bytes: int = 0,
    response_size_bytes: int = 0,
    user_id: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> APIMetric:
    """Create an API performance metric."""
    return APIMetric(
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        response_time_ms=response_time_ms,
        request_size_bytes=request_size_bytes,
        response_size_bytes=response_size_bytes,
        user_id=user_id,
        timestamp=timestamp or datetime.utcnow()
    )


def create_system_metric(
    component: str,
    instance: str,
    cpu_percent: float,
    memory_percent: float,
    disk_usage_percent: float,
    network_io_bytes: int,
    active_connections: int = 0,
    queue_size: int = 0,
    timestamp: Optional[datetime] = None
) -> SystemMetric:
    """Create a system performance metric."""
    return SystemMetric(
        component=component,
        instance=instance,
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        disk_usage_percent=disk_usage_percent,
        network_io_bytes=network_io_bytes,
        active_connections=active_connections,
        queue_size=queue_size,
        timestamp=timestamp or datetime.utcnow()
    )


def create_prediction_metric(
    agent_id: str,
    model_type: str,
    prediction_type: str,
    accuracy: float,
    confidence: float,
    processing_time_ms: float,
    feature_count: int,
    prediction_value: Optional[float] = None,
    actual_value: Optional[float] = None,
    timestamp: Optional[datetime] = None
) -> PredictionMetric:
    """Create a prediction performance metric."""
    return PredictionMetric(
        agent_id=agent_id,
        model_type=model_type,
        prediction_type=prediction_type,
        accuracy=accuracy,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
        feature_count=feature_count,
        prediction_value=prediction_value,
        actual_value=actual_value,
        timestamp=timestamp or datetime.utcnow()
    )


def create_pattern_metric(
    agent_id: str,
    pattern_type: str,
    framework: str,
    pattern_frequency: int,
    success_rate: float,
    average_duration_ms: float,
    anomaly_score: float,
    pattern_id: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> PatternMetric:
    """Create a pattern analysis metric."""
    return PatternMetric(
        agent_id=agent_id,
        pattern_type=pattern_type,
        framework=framework,
        pattern_frequency=pattern_frequency,
        success_rate=success_rate,
        average_duration_ms=average_duration_ms,
        anomaly_score=anomaly_score,
        pattern_id=pattern_id,
        timestamp=timestamp or datetime.utcnow()
    )


def create_causal_metric(
    agent_id: str,
    cause_event: str,
    effect_event: str,
    causal_strength: float,
    confidence: float,
    delay_ms: int,
    statistical_significance: float,
    timestamp: Optional[datetime] = None
) -> CausalMetric:
    """Create a causal relationship metric."""
    return CausalMetric(
        agent_id=agent_id,
        cause_event=cause_event,
        effect_event=effect_event,
        causal_strength=causal_strength,
        confidence=confidence,
        delay_ms=delay_ms,
        statistical_significance=statistical_significance,
        timestamp=timestamp or datetime.utcnow()
    )


def create_epistemic_metric(
    agent_id: str,
    session_id: str,
    belief_count: int,
    knowledge_items: int,
    goal_count: int,
    confidence_level: float,
    uncertainty_score: float,
    state_change_magnitude: float,
    timestamp: Optional[datetime] = None
) -> EpistemicMetric:
    """Create an epistemic state metric."""
    return EpistemicMetric(
        agent_id=agent_id,
        session_id=session_id,
        belief_count=belief_count,
        knowledge_items=knowledge_items,
        goal_count=goal_count,
        confidence_level=confidence_level,
        uncertainty_score=uncertainty_score,
        state_change_magnitude=state_change_magnitude,
        timestamp=timestamp or datetime.utcnow()
    )