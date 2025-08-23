"""
Unit tests for InfluxDB data models.

These tests verify the data model classes and their conversion
to InfluxDB point format.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from escai_framework.storage.influx_models import (
    MetricType, BaseMetric, AgentPerformanceMetric, APIMetric, SystemMetric,
    PredictionMetric, PatternMetric, CausalMetric, EpistemicMetric, MetricBatch,
    create_agent_performance_metric, create_api_metric, create_system_metric,
    create_prediction_metric, create_pattern_metric, create_causal_metric,
    create_epistemic_metric
)


class TestMetricType:
    """Test MetricType enumeration."""

    def test_metric_type_values(self):
        """Test metric type enumeration values."""
        assert MetricType.AGENT_PERFORMANCE.value == "agent_performance"
        assert MetricType.API_METRICS.value == "api_metrics"
        assert MetricType.SYSTEM_METRICS.value == "system_metrics"
        assert MetricType.PREDICTION_METRICS.value == "prediction_metrics"
        assert MetricType.PATTERN_METRICS.value == "pattern_metrics"
        assert MetricType.CAUSAL_METRICS.value == "causal_metrics"
        assert MetricType.EPISTEMIC_METRICS.value == "epistemic_metrics"


class TestBaseMetric:
    """Test BaseMetric abstract base class."""

    def test_base_metric_abstract_methods(self):
        """Test that BaseMetric abstract methods raise NotImplementedError."""
        metric = BaseMetric()
        
        with pytest.raises(NotImplementedError):
            metric.get_measurement_name()
        
        with pytest.raises(NotImplementedError):
            metric.get_fields()

    def test_base_metric_to_influx_point(self):
        """Test BaseMetric to_influx_point method structure."""
        # Create a concrete implementation for testing
        class TestMetric(BaseMetric):
            def get_measurement_name(self) -> str:
                return "test_measurement"
            
            def get_fields(self) -> Dict[str, Any]:
                return {"test_field": 42.0}
        
        timestamp = datetime.utcnow()
        metric = TestMetric(timestamp=timestamp, tags={"test_tag": "value"})
        
        point = metric.to_influx_point()
        
        assert point["measurement"] == "test_measurement"
        assert point["tags"] == {"test_tag": "value"}
        assert point["fields"] == {"test_field": 42.0}
        assert point["time"] == timestamp


class TestAgentPerformanceMetric:
    """Test AgentPerformanceMetric model."""

    def test_agent_performance_metric_creation(self):
        """Test agent performance metric creation."""
        timestamp = datetime.utcnow()
        metric = AgentPerformanceMetric(
            agent_id="test-agent",
            session_id="test-session",
            framework="langchain",
            task_type="reasoning",
            execution_time_ms=1500.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            success=True,
            error_count=1,
            retry_count=2,
            timestamp=timestamp
        )
        
        assert metric.agent_id == "test-agent"
        assert metric.session_id == "test-session"
        assert metric.framework == "langchain"
        assert metric.task_type == "reasoning"
        assert metric.execution_time_ms == 1500.0
        assert metric.memory_usage_mb == 256.0
        assert metric.cpu_usage_percent == 45.0
        assert metric.success is True
        assert metric.error_count == 1
        assert metric.retry_count == 2
        assert metric.timestamp == timestamp

    def test_agent_performance_metric_tags(self):
        """Test agent performance metric tag assignment."""
        metric = AgentPerformanceMetric(
            agent_id="test-agent",
            session_id="test-session",
            framework="langchain",
            task_type="reasoning"
        )
        
        expected_tags = {
            "agent_id": "test-agent",
            "session_id": "test-session",
            "framework": "langchain",
            "task_type": "reasoning"
        }
        
        assert metric.tags == expected_tags

    def test_agent_performance_metric_measurement_name(self):
        """Test agent performance metric measurement name."""
        metric = AgentPerformanceMetric()
        assert metric.get_measurement_name() == "agent_performance"

    def test_agent_performance_metric_fields(self):
        """Test agent performance metric fields."""
        metric = AgentPerformanceMetric(
            execution_time_ms=1500.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            success=True,
            error_count=1,
            retry_count=2
        )
        
        expected_fields = {
            "execution_time_ms": 1500.0,
            "memory_usage_mb": 256.0,
            "cpu_usage_percent": 45.0,
            "success": True,
            "error_count": 1,
            "retry_count": 2
        }
        
        assert metric.get_fields() == expected_fields

    def test_agent_performance_metric_to_influx_point(self):
        """Test agent performance metric conversion to InfluxDB point."""
        timestamp = datetime.utcnow()
        metric = AgentPerformanceMetric(
            agent_id="test-agent",
            session_id="test-session",
            framework="langchain",
            task_type="reasoning",
            execution_time_ms=1500.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            success=True,
            timestamp=timestamp
        )
        
        point = metric.to_influx_point()
        
        assert point["measurement"] == "agent_performance"
        assert point["tags"]["agent_id"] == "test-agent"
        assert point["fields"]["execution_time_ms"] == 1500.0
        assert point["time"] == timestamp


class TestAPIMetric:
    """Test APIMetric model."""

    def test_api_metric_creation(self):
        """Test API metric creation."""
        metric = APIMetric(
            endpoint="/api/v1/test",
            method="POST",
            status_code=200,
            response_time_ms=125.0,
            request_size_bytes=1024,
            response_size_bytes=512,
            user_id="user123"
        )
        
        assert metric.endpoint == "/api/v1/test"
        assert metric.method == "POST"
        assert metric.status_code == 200
        assert metric.response_time_ms == 125.0
        assert metric.request_size_bytes == 1024
        assert metric.response_size_bytes == 512
        assert metric.user_id == "user123"

    def test_api_metric_tags_with_user_id(self):
        """Test API metric tags with user ID."""
        metric = APIMetric(
            endpoint="/api/v1/test",
            method="GET",
            status_code=200,
            user_id="user123"
        )
        
        expected_tags = {
            "endpoint": "/api/v1/test",
            "method": "GET",
            "status_code": "200",
            "user_id": "user123"
        }
        
        assert metric.tags == expected_tags

    def test_api_metric_tags_without_user_id(self):
        """Test API metric tags without user ID."""
        metric = APIMetric(
            endpoint="/api/v1/test",
            method="GET",
            status_code=404
        )
        
        expected_tags = {
            "endpoint": "/api/v1/test",
            "method": "GET",
            "status_code": "404"
        }
        
        assert metric.tags == expected_tags

    def test_api_metric_measurement_name(self):
        """Test API metric measurement name."""
        metric = APIMetric()
        assert metric.get_measurement_name() == "api_metrics"

    def test_api_metric_fields(self):
        """Test API metric fields."""
        metric = APIMetric(
            response_time_ms=125.0,
            request_size_bytes=1024,
            response_size_bytes=512
        )
        
        expected_fields = {
            "response_time_ms": 125.0,
            "request_size_bytes": 1024,
            "response_size_bytes": 512
        }
        
        assert metric.get_fields() == expected_fields


class TestSystemMetric:
    """Test SystemMetric model."""

    def test_system_metric_creation(self):
        """Test system metric creation."""
        metric = SystemMetric(
            component="api-server",
            instance="server-1",
            cpu_percent=65.0,
            memory_percent=78.0,
            disk_usage_percent=45.0,
            network_io_bytes=1048576,
            active_connections=25,
            queue_size=10
        )
        
        assert metric.component == "api-server"
        assert metric.instance == "server-1"
        assert metric.cpu_percent == 65.0
        assert metric.memory_percent == 78.0
        assert metric.disk_usage_percent == 45.0
        assert metric.network_io_bytes == 1048576
        assert metric.active_connections == 25
        assert metric.queue_size == 10

    def test_system_metric_tags(self):
        """Test system metric tags."""
        metric = SystemMetric(
            component="database",
            instance="db-primary"
        )
        
        expected_tags = {
            "component": "database",
            "instance": "db-primary"
        }
        
        assert metric.tags == expected_tags

    def test_system_metric_measurement_name(self):
        """Test system metric measurement name."""
        metric = SystemMetric()
        assert metric.get_measurement_name() == "system_metrics"

    def test_system_metric_fields(self):
        """Test system metric fields."""
        metric = SystemMetric(
            cpu_percent=65.0,
            memory_percent=78.0,
            disk_usage_percent=45.0,
            network_io_bytes=1048576,
            active_connections=25,
            queue_size=10
        )
        
        expected_fields = {
            "cpu_percent": 65.0,
            "memory_percent": 78.0,
            "disk_usage_percent": 45.0,
            "network_io_bytes": 1048576,
            "active_connections": 25,
            "queue_size": 10
        }
        
        assert metric.get_fields() == expected_fields


class TestPredictionMetric:
    """Test PredictionMetric model."""

    def test_prediction_metric_creation(self):
        """Test prediction metric creation."""
        metric = PredictionMetric(
            agent_id="test-agent",
            model_type="lstm",
            prediction_type="success",
            accuracy=0.85,
            confidence=0.92,
            processing_time_ms=150.0,
            feature_count=25,
            prediction_value=0.8,
            actual_value=0.75
        )
        
        assert metric.agent_id == "test-agent"
        assert metric.model_type == "lstm"
        assert metric.prediction_type == "success"
        assert metric.accuracy == 0.85
        assert metric.confidence == 0.92
        assert metric.processing_time_ms == 150.0
        assert metric.feature_count == 25
        assert metric.prediction_value == 0.8
        assert metric.actual_value == 0.75

    def test_prediction_metric_fields_with_optional_values(self):
        """Test prediction metric fields with optional values."""
        metric = PredictionMetric(
            accuracy=0.85,
            confidence=0.92,
            processing_time_ms=150.0,
            feature_count=25,
            prediction_value=0.8,
            actual_value=0.75
        )
        
        fields = metric.get_fields()
        
        assert "prediction_value" in fields
        assert "actual_value" in fields
        assert fields["prediction_value"] == 0.8
        assert fields["actual_value"] == 0.75

    def test_prediction_metric_fields_without_optional_values(self):
        """Test prediction metric fields without optional values."""
        metric = PredictionMetric(
            accuracy=0.85,
            confidence=0.92,
            processing_time_ms=150.0,
            feature_count=25
        )
        
        fields = metric.get_fields()
        
        assert "prediction_value" not in fields
        assert "actual_value" not in fields


class TestPatternMetric:
    """Test PatternMetric model."""

    def test_pattern_metric_creation(self):
        """Test pattern metric creation."""
        metric = PatternMetric(
            agent_id="test-agent",
            pattern_type="sequential",
            framework="langchain",
            pattern_frequency=15,
            success_rate=0.85,
            average_duration_ms=2500.0,
            anomaly_score=0.15,
            pattern_id="pattern-123"
        )
        
        assert metric.agent_id == "test-agent"
        assert metric.pattern_type == "sequential"
        assert metric.framework == "langchain"
        assert metric.pattern_frequency == 15
        assert metric.success_rate == 0.85
        assert metric.average_duration_ms == 2500.0
        assert metric.anomaly_score == 0.15
        assert metric.pattern_id == "pattern-123"

    def test_pattern_metric_tags_with_pattern_id(self):
        """Test pattern metric tags with pattern ID."""
        metric = PatternMetric(
            agent_id="test-agent",
            pattern_type="sequential",
            framework="langchain",
            pattern_id="pattern-123"
        )
        
        expected_tags = {
            "agent_id": "test-agent",
            "pattern_type": "sequential",
            "framework": "langchain",
            "pattern_id": "pattern-123"
        }
        
        assert metric.tags == expected_tags

    def test_pattern_metric_tags_without_pattern_id(self):
        """Test pattern metric tags without pattern ID."""
        metric = PatternMetric(
            agent_id="test-agent",
            pattern_type="sequential",
            framework="langchain"
        )
        
        expected_tags = {
            "agent_id": "test-agent",
            "pattern_type": "sequential",
            "framework": "langchain"
        }
        
        assert metric.tags == expected_tags


class TestCausalMetric:
    """Test CausalMetric model."""

    def test_causal_metric_creation(self):
        """Test causal metric creation."""
        metric = CausalMetric(
            agent_id="test-agent",
            cause_event="decision_made",
            effect_event="action_taken",
            causal_strength=0.75,
            confidence=0.85,
            delay_ms=250,
            statistical_significance=0.95
        )
        
        assert metric.agent_id == "test-agent"
        assert metric.cause_event == "decision_made"
        assert metric.effect_event == "action_taken"
        assert metric.causal_strength == 0.75
        assert metric.confidence == 0.85
        assert metric.delay_ms == 250
        assert metric.statistical_significance == 0.95

    def test_causal_metric_measurement_name(self):
        """Test causal metric measurement name."""
        metric = CausalMetric()
        assert metric.get_measurement_name() == "causal_metrics"


class TestEpistemicMetric:
    """Test EpistemicMetric model."""

    def test_epistemic_metric_creation(self):
        """Test epistemic metric creation."""
        metric = EpistemicMetric(
            agent_id="test-agent",
            session_id="test-session",
            belief_count=5,
            knowledge_items=12,
            goal_count=3,
            confidence_level=0.85,
            uncertainty_score=0.25,
            state_change_magnitude=0.45
        )
        
        assert metric.agent_id == "test-agent"
        assert metric.session_id == "test-session"
        assert metric.belief_count == 5
        assert metric.knowledge_items == 12
        assert metric.goal_count == 3
        assert metric.confidence_level == 0.85
        assert metric.uncertainty_score == 0.25
        assert metric.state_change_magnitude == 0.45

    def test_epistemic_metric_measurement_name(self):
        """Test epistemic metric measurement name."""
        metric = EpistemicMetric()
        assert metric.get_measurement_name() == "epistemic_metrics"


class TestMetricBatch:
    """Test MetricBatch functionality."""

    def test_metric_batch_creation(self):
        """Test metric batch creation."""
        batch = MetricBatch()
        assert batch.size() == 0
        assert len(batch.metrics) == 0

    def test_metric_batch_add_metric(self):
        """Test adding single metric to batch."""
        batch = MetricBatch()
        metric = AgentPerformanceMetric(agent_id="test")
        
        batch.add_metric(metric)
        
        assert batch.size() == 1
        assert batch.metrics[0] == metric

    def test_metric_batch_add_metrics(self):
        """Test adding multiple metrics to batch."""
        batch = MetricBatch()
        metrics = [
            AgentPerformanceMetric(agent_id="test1"),
            APIMetric(endpoint="/test1"),
            SystemMetric(component="test1")
        ]
        
        batch.add_metrics(metrics)
        
        assert batch.size() == 3
        assert batch.metrics == metrics

    def test_metric_batch_clear(self):
        """Test clearing batch."""
        batch = MetricBatch()
        batch.add_metric(AgentPerformanceMetric(agent_id="test"))
        
        assert batch.size() == 1
        
        batch.clear()
        
        assert batch.size() == 0
        assert len(batch.metrics) == 0

    def test_metric_batch_to_influx_points(self):
        """Test converting batch to InfluxDB points."""
        batch = MetricBatch()
        metrics = [
            AgentPerformanceMetric(agent_id="test1"),
            APIMetric(endpoint="/test1")
        ]
        batch.add_metrics(metrics)
        
        points = batch.to_influx_points()
        
        assert len(points) == 2
        assert points[0]["measurement"] == "agent_performance"
        assert points[1]["measurement"] == "api_metrics"

    def test_metric_batch_filter_by_type(self):
        """Test filtering batch by metric type."""
        batch = MetricBatch()
        metrics = [
            AgentPerformanceMetric(agent_id="test1"),
            APIMetric(endpoint="/test1"),
            AgentPerformanceMetric(agent_id="test2"),
            SystemMetric(component="test1")
        ]
        batch.add_metrics(metrics)
        
        agent_metrics = batch.filter_by_type(MetricType.AGENT_PERFORMANCE)
        api_metrics = batch.filter_by_type(MetricType.API_METRICS)
        
        assert len(agent_metrics) == 2
        assert len(api_metrics) == 1
        assert all(isinstance(m, AgentPerformanceMetric) for m in agent_metrics)
        assert all(isinstance(m, APIMetric) for m in api_metrics)


class TestUtilityFunctions:
    """Test utility functions for creating metrics."""

    def test_create_agent_performance_metric(self):
        """Test create_agent_performance_metric function."""
        timestamp = datetime.utcnow()
        metric = create_agent_performance_metric(
            agent_id="test-agent",
            session_id="test-session",
            framework="langchain",
            task_type="reasoning",
            execution_time_ms=1500.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            success=True,
            error_count=1,
            retry_count=2,
            timestamp=timestamp
        )
        
        assert isinstance(metric, AgentPerformanceMetric)
        assert metric.agent_id == "test-agent"
        assert metric.execution_time_ms == 1500.0
        assert metric.timestamp == timestamp

    def test_create_api_metric(self):
        """Test create_api_metric function."""
        metric = create_api_metric(
            endpoint="/api/v1/test",
            method="POST",
            status_code=200,
            response_time_ms=125.0,
            request_size_bytes=1024,
            response_size_bytes=512,
            user_id="user123"
        )
        
        assert isinstance(metric, APIMetric)
        assert metric.endpoint == "/api/v1/test"
        assert metric.user_id == "user123"

    def test_create_system_metric(self):
        """Test create_system_metric function."""
        metric = create_system_metric(
            component="api-server",
            instance="server-1",
            cpu_percent=65.0,
            memory_percent=78.0,
            disk_usage_percent=45.0,
            network_io_bytes=1048576,
            active_connections=25,
            queue_size=10
        )
        
        assert isinstance(metric, SystemMetric)
        assert metric.component == "api-server"
        assert metric.cpu_percent == 65.0

    def test_create_prediction_metric(self):
        """Test create_prediction_metric function."""
        metric = create_prediction_metric(
            agent_id="test-agent",
            model_type="lstm",
            prediction_type="success",
            accuracy=0.85,
            confidence=0.92,
            processing_time_ms=150.0,
            feature_count=25,
            prediction_value=0.8,
            actual_value=0.75
        )
        
        assert isinstance(metric, PredictionMetric)
        assert metric.agent_id == "test-agent"
        assert metric.model_type == "lstm"
        assert metric.accuracy == 0.85

    def test_create_pattern_metric(self):
        """Test create_pattern_metric function."""
        metric = create_pattern_metric(
            agent_id="test-agent",
            pattern_type="sequential",
            framework="langchain",
            pattern_frequency=15,
            success_rate=0.85,
            average_duration_ms=2500.0,
            anomaly_score=0.15,
            pattern_id="pattern-123"
        )
        
        assert isinstance(metric, PatternMetric)
        assert metric.agent_id == "test-agent"
        assert metric.pattern_type == "sequential"
        assert metric.pattern_id == "pattern-123"

    def test_create_causal_metric(self):
        """Test create_causal_metric function."""
        metric = create_causal_metric(
            agent_id="test-agent",
            cause_event="decision_made",
            effect_event="action_taken",
            causal_strength=0.75,
            confidence=0.85,
            delay_ms=250,
            statistical_significance=0.95
        )
        
        assert isinstance(metric, CausalMetric)
        assert metric.agent_id == "test-agent"
        assert metric.cause_event == "decision_made"
        assert metric.causal_strength == 0.75

    def test_create_epistemic_metric(self):
        """Test create_epistemic_metric function."""
        metric = create_epistemic_metric(
            agent_id="test-agent",
            session_id="test-session",
            belief_count=5,
            knowledge_items=12,
            goal_count=3,
            confidence_level=0.85,
            uncertainty_score=0.25,
            state_change_magnitude=0.45
        )
        
        assert isinstance(metric, EpistemicMetric)
        assert metric.agent_id == "test-agent"
        assert metric.session_id == "test-session"
        assert metric.belief_count == 5

    def test_create_metric_with_default_timestamp(self):
        """Test creating metric with default timestamp."""
        before = datetime.utcnow()
        metric = create_agent_performance_metric(
            agent_id="test",
            session_id="test",
            framework="test",
            task_type="test",
            execution_time_ms=100.0,
            memory_usage_mb=50.0,
            cpu_usage_percent=25.0,
            success=True
        )
        after = datetime.utcnow()
        
        assert before <= metric.timestamp <= after


if __name__ == "__main__":
    pytest.main([__file__, "-v"])