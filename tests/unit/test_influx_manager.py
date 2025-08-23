"""
Unit tests for InfluxDB manager functionality.

These tests focus on testing the InfluxDB manager logic without requiring
a real InfluxDB instance.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

from escai_framework.storage.influx_manager import (
    InfluxDBManager, MetricPoint, RetentionPolicy,
    write_agent_performance_metric, write_api_metric, write_system_metric
)


class TestInfluxDBManager:
    """Test InfluxDB manager functionality."""

    def test_init(self):
        """Test InfluxDB manager initialization."""
        manager = InfluxDBManager(
            url="http://test:8086",
            token="test-token",
            org="test-org",
            bucket="test-bucket",
            timeout=5000,
            batch_size=500
        )
        
        assert manager.url == "http://test:8086"
        assert manager.token == "test-token"
        assert manager.org == "test-org"
        assert manager.bucket == "test-bucket"
        assert manager.timeout == 5000
        assert manager.batch_size == 500

    def test_init_without_influxdb(self):
        """Test initialization when InfluxDB client is not available."""
        with patch('escai_framework.storage.influx_manager.INFLUXDB_AVAILABLE', False):
            with pytest.raises(ImportError, match="influxdb-client package is required"):
                InfluxDBManager()

    def test_metric_schemas(self):
        """Test metric schema definitions."""
        manager = InfluxDBManager()
        
        # Check that all expected schemas are defined
        expected_schemas = [
            "agent_performance",
            "api_metrics", 
            "system_metrics",
            "prediction_metrics",
            "pattern_metrics"
        ]
        
        for schema_name in expected_schemas:
            assert schema_name in manager.metric_schemas
            schema = manager.metric_schemas[schema_name]
            assert "tags" in schema
            assert "fields" in schema
            assert isinstance(schema["tags"], list)
            assert isinstance(schema["fields"], list)

    def test_validate_metric_point_valid(self):
        """Test metric point validation with valid data."""
        manager = InfluxDBManager()
        
        # Valid agent performance metric
        point = MetricPoint(
            measurement="agent_performance",
            tags={
                "agent_id": "test-agent",
                "session_id": "test-session",
                "framework": "langchain",
                "task_type": "reasoning"
            },
            fields={
                "execution_time_ms": 1500.0,
                "memory_usage_mb": 256.0,
                "cpu_usage_percent": 45.0,
                "success": True
            }
        )
        
        assert manager._validate_metric_point(point) is True

    def test_validate_metric_point_missing_tags(self):
        """Test metric point validation with missing required tags."""
        manager = InfluxDBManager()
        
        # Missing required tags
        point = MetricPoint(
            measurement="agent_performance",
            tags={"agent_id": "test-agent"},  # Missing other required tags
            fields={
                "execution_time_ms": 1500.0,
                "memory_usage_mb": 256.0,
                "cpu_usage_percent": 45.0,
                "success": True
            }
        )
        
        assert manager._validate_metric_point(point) is False

    def test_validate_metric_point_missing_fields(self):
        """Test metric point validation with missing required fields."""
        manager = InfluxDBManager()
        
        # Missing required fields
        point = MetricPoint(
            measurement="agent_performance",
            tags={
                "agent_id": "test-agent",
                "session_id": "test-session",
                "framework": "langchain",
                "task_type": "reasoning"
            },
            fields={"execution_time_ms": 1500.0}  # Missing other required fields
        )
        
        assert manager._validate_metric_point(point) is False

    def test_validate_metric_point_unknown_measurement(self):
        """Test metric point validation with unknown measurement."""
        manager = InfluxDBManager()
        
        # Unknown measurement (should pass with warning)
        point = MetricPoint(
            measurement="unknown_measurement",
            tags={"test": "value"},
            fields={"value": 42.0}
        )
        
        assert manager._validate_metric_point(point) is True

    @patch('escai_framework.storage.influx_manager.Point')
    def test_create_influx_point(self, mock_point_class):
        """Test creation of InfluxDB Point from MetricPoint."""
        manager = InfluxDBManager()
        mock_point = Mock()
        mock_point.tag.return_value = mock_point
        mock_point.field.return_value = mock_point
        mock_point.time.return_value = mock_point
        mock_point_class.return_value = mock_point
        
        timestamp = datetime.utcnow()
        metric_point = MetricPoint(
            measurement="test_measurement",
            tags={"tag1": "value1", "tag2": "value2"},
            fields={"field1": 42.0, "field2": "string_value"},
            timestamp=timestamp
        )
        
        result = manager._create_influx_point(metric_point)
        
        # Verify Point was created with correct measurement
        mock_point_class.assert_called_once_with("test_measurement")
        
        # Verify tags were added
        assert mock_point.tag.call_count == 2
        mock_point.tag.assert_any_call("tag1", "value1")
        mock_point.tag.assert_any_call("tag2", "value2")
        
        # Verify fields were added
        assert mock_point.field.call_count == 2
        mock_point.field.assert_any_call("field1", 42.0)
        mock_point.field.assert_any_call("field2", "string_value")
        
        # Verify timestamp was set
        mock_point.time.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_metric_not_connected(self):
        """Test writing metric when not connected."""
        manager = InfluxDBManager()
        
        with pytest.raises(RuntimeError, match="InfluxDB not connected"):
            await manager.write_metric(
                measurement="test",
                tags={"test": "value"},
                fields={"value": 42.0}
            )

    @pytest.mark.asyncio
    async def test_write_metric_invalid_point(self):
        """Test writing invalid metric point."""
        manager = InfluxDBManager()
        manager._write_api = Mock()
        
        # Mock validation to return False
        with patch.object(manager, '_validate_metric_point', return_value=False):
            with pytest.raises(ValueError, match="Invalid metric point"):
                await manager.write_metric(
                    measurement="agent_performance",
                    tags={"incomplete": "tags"},
                    fields={"incomplete": "fields"}
                )

    @pytest.mark.asyncio
    async def test_write_metrics_batch_empty(self):
        """Test writing empty metrics batch."""
        manager = InfluxDBManager()
        manager._write_api = Mock()
        
        # Should handle empty list gracefully
        await manager.write_metrics_batch([])
        
        # Write API should not be called
        manager._write_api.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_metrics_batch_with_invalid_metrics(self):
        """Test writing batch with some invalid metrics."""
        manager = InfluxDBManager()
        manager._write_api = Mock()
        
        metrics = [
            MetricPoint("valid", {"tag": "value"}, {"field": 1.0}),
            MetricPoint("invalid", {}, {}),  # Will be invalid
            MetricPoint("valid2", {"tag": "value"}, {"field": 2.0})
        ]
        
        # Mock validation to return True for valid, False for invalid
        def mock_validate(point):
            return point.measurement != "invalid"
        
        with patch.object(manager, '_validate_metric_point', side_effect=mock_validate):
            with patch.object(manager, '_create_influx_point') as mock_create:
                mock_create.return_value = Mock()
                
                await manager.write_metrics_batch(metrics)
                
                # Should only create points for valid metrics
                assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_query_metrics_not_connected(self):
        """Test querying metrics when not connected."""
        manager = InfluxDBManager()
        
        with pytest.raises(RuntimeError, match="InfluxDB not connected"):
            await manager.query_metrics("test query")

    @pytest.mark.asyncio
    async def test_query_metrics_with_bucket_injection(self):
        """Test query metrics with automatic bucket injection."""
        manager = InfluxDBManager(bucket="test-bucket")
        manager._query_api = Mock()
        
        # Mock query result
        mock_table = Mock()
        mock_record = Mock()
        mock_record.get_measurement.return_value = "test_measurement"
        mock_record.get_time.return_value = datetime.utcnow()
        mock_record.get_value.return_value = 42.0
        mock_record.get_field.return_value = "test_field"
        mock_record.values = {"tag1": "value1"}
        mock_table.records = [mock_record]
        manager._query_api.query.return_value = [mock_table]
        
        query = 'range(start: -1h) |> filter(fn: (r) => r._measurement == "test")'
        
        result = await manager.query_metrics(query)
        
        # Verify bucket was injected
        expected_query = f'from(bucket: "test-bucket") |> {query}'
        manager._query_api.query.assert_called_once_with(query=expected_query, org=manager.org)
        
        # Verify result structure
        assert len(result) == 1
        assert result[0]["measurement"] == "test_measurement"
        assert result[0]["value"] == 42.0

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self):
        """Test health check when not connected."""
        manager = InfluxDBManager()
        
        health = await manager.health_check()
        
        assert health["status"] == "disconnected"
        assert "not initialized" in health["message"]

    @pytest.mark.asyncio
    async def test_health_check_with_exception(self):
        """Test health check when client raises exception."""
        manager = InfluxDBManager()
        manager._client = Mock()
        manager._client.health.side_effect = Exception("Connection failed")
        
        health = await manager.health_check()
        
        assert health["status"] == "error"
        assert "Connection failed" in health["message"]

    def test_retention_policy(self):
        """Test retention policy data structure."""
        policy = RetentionPolicy(
            name="test_policy",
            duration_seconds=7 * 24 * 3600,  # 7 days
            shard_group_duration_seconds=3600,  # 1 hour
            replication_factor=2
        )
        
        assert policy.name == "test_policy"
        assert policy.duration_seconds == 7 * 24 * 3600
        assert policy.shard_group_duration_seconds == 3600
        assert policy.replication_factor == 2


class TestConvenienceFunctions:
    """Test convenience functions for common metric types."""

    @pytest.mark.asyncio
    async def test_write_agent_performance_metric(self):
        """Test agent performance metric convenience function."""
        manager = Mock()
        manager.write_metric = AsyncMock()
        
        await write_agent_performance_metric(
            manager=manager,
            agent_id="test-agent",
            session_id="test-session",
            framework="langchain",
            task_type="reasoning",
            execution_time_ms=1500.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            success=True
        )
        
        manager.write_metric.assert_called_once()
        call_args = manager.write_metric.call_args
        
        assert call_args[1]["measurement"] == "agent_performance"
        assert call_args[1]["tags"]["agent_id"] == "test-agent"
        assert call_args[1]["fields"]["execution_time_ms"] == 1500.0

    @pytest.mark.asyncio
    async def test_write_api_metric(self):
        """Test API metric convenience function."""
        manager = Mock()
        manager.write_metric = AsyncMock()
        
        await write_api_metric(
            manager=manager,
            endpoint="/api/v1/test",
            method="POST",
            status_code=200,
            response_time_ms=125.0,
            request_size_bytes=1024,
            response_size_bytes=512
        )
        
        manager.write_metric.assert_called_once()
        call_args = manager.write_metric.call_args
        
        assert call_args[1]["measurement"] == "api_metrics"
        assert call_args[1]["tags"]["endpoint"] == "/api/v1/test"
        assert call_args[1]["fields"]["response_time_ms"] == 125.0

    @pytest.mark.asyncio
    async def test_write_system_metric(self):
        """Test system metric convenience function."""
        manager = Mock()
        manager.write_metric = AsyncMock()
        
        await write_system_metric(
            manager=manager,
            component="api-server",
            instance="server-1",
            cpu_percent=65.0,
            memory_percent=78.0,
            disk_usage_percent=45.0,
            network_io_bytes=1048576
        )
        
        manager.write_metric.assert_called_once()
        call_args = manager.write_metric.call_args
        
        assert call_args[1]["measurement"] == "system_metrics"
        assert call_args[1]["tags"]["component"] == "api-server"
        assert call_args[1]["fields"]["cpu_percent"] == 65.0


class TestMetricPoint:
    """Test MetricPoint data structure."""

    def test_metric_point_creation(self):
        """Test MetricPoint creation."""
        timestamp = datetime.utcnow()
        point = MetricPoint(
            measurement="test_measurement",
            tags={"tag1": "value1"},
            fields={"field1": 42.0},
            timestamp=timestamp
        )
        
        assert point.measurement == "test_measurement"
        assert point.tags == {"tag1": "value1"}
        assert point.fields == {"field1": 42.0}
        assert point.timestamp == timestamp

    def test_metric_point_default_timestamp(self):
        """Test MetricPoint with default timestamp."""
        point = MetricPoint(
            measurement="test",
            tags={},
            fields={"value": 1.0}
        )
        
        # Should have None timestamp (will be set by InfluxDB)
        assert point.timestamp is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])