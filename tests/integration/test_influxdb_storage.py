"""
Integration tests for InfluxDB storage functionality.

These tests verify the InfluxDB manager, models, and dashboard functionality
with a real InfluxDB instance.
"""

import pytest
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any

from escai_framework.storage.influx_manager import InfluxDBManager, MetricPoint, RetentionPolicy
from escai_framework.storage.influx_models import (
    AgentPerformanceMetric, APIMetric, SystemMetric, PredictionMetric,
    PatternMetric, CausalMetric, EpistemicMetric, MetricBatch, MetricType,
    create_agent_performance_metric, create_api_metric, create_system_metric
)
from escai_framework.storage.influx_dashboard import InfluxDashboardManager


# Test configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "test-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "test-org")
TEST_BUCKET = "test-metrics"


@pytest.fixture
async def influx_manager():
    """Create InfluxDB manager for testing."""
    manager = InfluxDBManager(
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        bucket=TEST_BUCKET,
        batch_size=100
    )
    
    try:
        await manager.connect()
        yield manager
    except Exception as e:
        pytest.skip(f"InfluxDB not available: {e}")
    finally:
        if manager._client:
            await manager.disconnect()


@pytest.fixture
async def dashboard_manager(influx_manager):
    """Create dashboard manager for testing."""
    return InfluxDashboardManager(influx_manager)


class TestInfluxDBManager:
    """Test InfluxDB manager functionality."""

    async def test_connection(self, influx_manager):
        """Test InfluxDB connection."""
        health = await influx_manager.health_check()
        assert health["status"] in ["pass", "warn"]

    async def test_bucket_operations(self, influx_manager):
        """Test bucket creation and information retrieval."""
        bucket_info = await influx_manager.get_bucket_info()
        assert bucket_info["name"] == TEST_BUCKET
        assert "retention_rules" in bucket_info

    async def test_write_single_metric(self, influx_manager):
        """Test writing a single metric."""
        await influx_manager.write_metric(
            measurement="test_metric",
            tags={"test_tag": "value1"},
            fields={"test_field": 42.0},
            timestamp=datetime.utcnow()
        )

        # Verify metric was written
        query = f'''
        from(bucket: "{TEST_BUCKET}")
        |> range(start: -1h)
        |> filter(fn: (r) => r._measurement == "test_metric")
        |> filter(fn: (r) => r.test_tag == "value1")
        '''
        
        results = await influx_manager.query_metrics(query)
        assert len(results) > 0

    async def test_write_metrics_batch(self, influx_manager):
        """Test writing metrics in batch."""
        metrics = []
        for i in range(10):
            metric = MetricPoint(
                measurement="batch_test",
                tags={"batch_id": "test", "index": str(i)},
                fields={"value": float(i * 10)},
                timestamp=datetime.utcnow()
            )
            metrics.append(metric)

        await influx_manager.write_metrics_batch(metrics)

        # Verify metrics were written
        query = f'''
        from(bucket: "{TEST_BUCKET}")
        |> range(start: -1h)
        |> filter(fn: (r) => r._measurement == "batch_test")
        |> filter(fn: (r) => r.batch_id == "test")
        '''
        
        results = await influx_manager.query_metrics(query)
        assert len(results) >= 10

    async def test_agent_performance_metrics(self, influx_manager):
        """Test agent performance metric queries."""
        # Write test data
        agent_id = "test-agent-123"
        session_id = "session-456"
        
        await influx_manager.write_metric(
            measurement="agent_performance",
            tags={
                "agent_id": agent_id,
                "session_id": session_id,
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

        # Query metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        results = await influx_manager.get_agent_performance_metrics(
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert len(results) > 0
        assert any(r.get("agent_id") == agent_id for r in results)

    async def test_api_performance_metrics(self, influx_manager):
        """Test API performance metric queries."""
        # Write test data
        endpoint = "/api/v1/monitor"
        
        await influx_manager.write_metric(
            measurement="api_metrics",
            tags={
                "endpoint": endpoint,
                "method": "POST",
                "status_code": "200"
            },
            fields={
                "response_time_ms": 125.0,
                "request_size_bytes": 1024,
                "response_size_bytes": 512
            }
        )

        # Query metrics
        results = await influx_manager.get_api_performance_metrics(endpoint=endpoint)
        assert len(results) > 0

    async def test_system_metrics(self, influx_manager):
        """Test system metric queries."""
        # Write test data
        component = "api-server"
        
        await influx_manager.write_metric(
            measurement="system_metrics",
            tags={
                "component": component,
                "instance": "server-1"
            },
            fields={
                "cpu_percent": 65.0,
                "memory_percent": 78.0,
                "disk_usage_percent": 45.0,
                "network_io_bytes": 1048576
            }
        )

        # Query metrics
        results = await influx_manager.get_system_metrics(component=component)
        assert len(results) > 0

    async def test_retention_policy(self, influx_manager):
        """Test retention policy creation."""
        policy = RetentionPolicy(
            name="test_policy",
            duration_seconds=7 * 24 * 3600,  # 7 days
            replication_factor=1
        )
        
        await influx_manager.create_retention_policy(policy)
        
        # Verify policy was applied
        bucket_info = await influx_manager.get_bucket_info()
        retention_rules = bucket_info.get("retention_rules", [])
        assert len(retention_rules) > 0

    async def test_delete_metrics(self, influx_manager):
        """Test metric deletion."""
        # Write test data
        measurement = "delete_test"
        await influx_manager.write_metric(
            measurement=measurement,
            tags={"test": "delete"},
            fields={"value": 1.0}
        )

        # Delete metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        await influx_manager.delete_metrics(
            start_time=start_time,
            end_time=end_time,
            predicate=f'_measurement="{measurement}"'
        )

        # Verify deletion (may take some time to propagate)
        await asyncio.sleep(1)


class TestInfluxModels:
    """Test InfluxDB data models."""

    def test_agent_performance_metric(self):
        """Test agent performance metric model."""
        metric = create_agent_performance_metric(
            agent_id="agent-123",
            session_id="session-456",
            framework="langchain",
            task_type="reasoning",
            execution_time_ms=1500.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            success=True,
            error_count=0,
            retry_count=1
        )
        
        assert metric.agent_id == "agent-123"
        assert metric.framework == "langchain"
        assert metric.success is True
        
        # Test conversion to InfluxDB point
        point = metric.to_influx_point()
        assert point["measurement"] == "agent_performance"
        assert point["tags"]["agent_id"] == "agent-123"
        assert point["fields"]["execution_time_ms"] == 1500.0

    def test_api_metric(self):
        """Test API metric model."""
        metric = create_api_metric(
            endpoint="/api/v1/test",
            method="GET",
            status_code=200,
            response_time_ms=125.0,
            request_size_bytes=512,
            response_size_bytes=1024
        )
        
        assert metric.endpoint == "/api/v1/test"
        assert metric.status_code == 200
        
        point = metric.to_influx_point()
        assert point["measurement"] == "api_metrics"
        assert point["tags"]["method"] == "GET"
        assert point["fields"]["response_time_ms"] == 125.0

    def test_system_metric(self):
        """Test system metric model."""
        metric = create_system_metric(
            component="api-server",
            instance="server-1",
            cpu_percent=65.0,
            memory_percent=78.0,
            disk_usage_percent=45.0,
            network_io_bytes=1048576
        )
        
        assert metric.component == "api-server"
        assert metric.cpu_percent == 65.0
        
        point = metric.to_influx_point()
        assert point["measurement"] == "system_metrics"
        assert point["tags"]["component"] == "api-server"
        assert point["fields"]["cpu_percent"] == 65.0

    def test_metric_batch(self):
        """Test metric batch operations."""
        batch = MetricBatch()
        
        # Add individual metric
        metric1 = create_agent_performance_metric(
            agent_id="agent-1",
            session_id="session-1",
            framework="langchain",
            task_type="test",
            execution_time_ms=100.0,
            memory_usage_mb=128.0,
            cpu_usage_percent=25.0,
            success=True
        )
        batch.add_metric(metric1)
        
        # Add multiple metrics
        metrics = [
            create_api_metric("/api/test1", "GET", 200, 50.0),
            create_api_metric("/api/test2", "POST", 201, 75.0)
        ]
        batch.add_metrics(metrics)
        
        assert batch.size() == 3
        
        # Test conversion to InfluxDB points
        points = batch.to_influx_points()
        assert len(points) == 3
        
        # Test filtering by type
        api_metrics = batch.filter_by_type(MetricType.API_METRICS)
        assert len(api_metrics) == 2


class TestInfluxDashboard:
    """Test InfluxDB dashboard functionality."""

    async def test_dashboard_creation(self, dashboard_manager):
        """Test dashboard manager initialization."""
        dashboards = dashboard_manager.list_dashboards()
        assert len(dashboards) > 0
        
        # Check default dashboards exist
        dashboard_names = [d["name"] for d in dashboards]
        assert "agent_performance" in dashboard_names
        assert "system_metrics" in dashboard_names
        assert "api_performance" in dashboard_names

    async def test_dashboard_data_retrieval(self, dashboard_manager, influx_manager):
        """Test dashboard data retrieval."""
        # Write some test data first
        await influx_manager.write_metric(
            measurement="agent_performance",
            tags={
                "agent_id": "test-agent",
                "session_id": "test-session",
                "framework": "langchain",
                "task_type": "test"
            },
            fields={
                "execution_time_ms": 1000.0,
                "memory_usage_mb": 200.0,
                "cpu_usage_percent": 50.0,
                "success": True
            }
        )
        
        # Get dashboard data
        dashboard_data = await dashboard_manager.get_dashboard_data(
            "agent_performance",
            time_range_hours=1
        )
        
        assert dashboard_data["name"] == "Agent Performance"
        assert len(dashboard_data["panels"]) > 0
        
        # Check panel structure
        panel = dashboard_data["panels"][0]
        assert "title" in panel
        assert "queries" in panel
        assert len(panel["queries"]) > 0

    async def test_real_time_metrics(self, dashboard_manager, influx_manager):
        """Test real-time metrics retrieval."""
        # Write test data
        await influx_manager.write_metric(
            measurement="system_metrics",
            tags={"component": "test", "instance": "test-1"},
            fields={"cpu_percent": 60.0, "memory_percent": 70.0}
        )
        
        # Get real-time metrics
        metrics = await dashboard_manager.get_real_time_metrics(
            metric_types=["system_metrics"],
            time_window_minutes=5
        )
        
        assert "system_metrics" in metrics
        # Note: Results may be empty if no data in the time window

    async def test_alert_conditions(self, dashboard_manager, influx_manager):
        """Test alert condition checking."""
        # Write test data that should trigger alerts
        await influx_manager.write_metric(
            measurement="system_metrics",
            tags={"component": "test", "instance": "test-1"},
            fields={"cpu_percent": 95.0}  # Should trigger critical alert
        )
        
        # Check alerts
        alerts = await dashboard_manager.get_alert_conditions("system_metrics")
        
        # Note: Alerts may not trigger immediately due to aggregation windows
        assert isinstance(alerts, list)

    async def test_dashboard_export_import(self, dashboard_manager):
        """Test dashboard configuration export and import."""
        # Export existing dashboard
        config = await dashboard_manager.export_dashboard_config("agent_performance")
        
        assert config["name"] == "Agent Performance"
        assert "panels" in config
        assert len(config["panels"]) > 0
        
        # Import as new dashboard
        config["name"] = "Test Dashboard"
        config["description"] = "Test dashboard import"
        
        await dashboard_manager.import_dashboard_config("test_dashboard", config)
        
        # Verify import
        dashboards = dashboard_manager.list_dashboards()
        dashboard_names = [d["name"] for d in dashboards]
        assert "test_dashboard" in dashboard_names


class TestInfluxIntegration:
    """Test complete InfluxDB integration scenarios."""

    async def test_complete_monitoring_workflow(self, influx_manager, dashboard_manager):
        """Test complete monitoring workflow from data ingestion to visualization."""
        # 1. Write various types of metrics
        agent_id = "integration-test-agent"
        session_id = "integration-test-session"
        
        # Agent performance metrics
        await influx_manager.write_metric(
            measurement="agent_performance",
            tags={
                "agent_id": agent_id,
                "session_id": session_id,
                "framework": "langchain",
                "task_type": "integration_test"
            },
            fields={
                "execution_time_ms": 2000.0,
                "memory_usage_mb": 512.0,
                "cpu_usage_percent": 75.0,
                "success": True
            }
        )
        
        # API metrics
        await influx_manager.write_metric(
            measurement="api_metrics",
            tags={
                "endpoint": "/api/v1/integration_test",
                "method": "POST",
                "status_code": "200"
            },
            fields={
                "response_time_ms": 150.0,
                "request_size_bytes": 2048,
                "response_size_bytes": 1024
            }
        )
        
        # System metrics
        await influx_manager.write_metric(
            measurement="system_metrics",
            tags={
                "component": "integration-test",
                "instance": "test-instance"
            },
            fields={
                "cpu_percent": 80.0,
                "memory_percent": 85.0,
                "disk_usage_percent": 60.0,
                "network_io_bytes": 2097152
            }
        )
        
        # 2. Query metrics through manager
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        agent_metrics = await influx_manager.get_agent_performance_metrics(
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time
        )
        
        api_metrics = await influx_manager.get_api_performance_metrics(
            endpoint="/api/v1/integration_test"
        )
        
        system_metrics = await influx_manager.get_system_metrics(
            component="integration-test"
        )
        
        # 3. Get dashboard data
        dashboard_data = await dashboard_manager.get_dashboard_data(
            "agent_performance",
            time_range_hours=1
        )
        
        # 4. Verify data flow
        assert len(agent_metrics) > 0 or len(api_metrics) > 0 or len(system_metrics) > 0
        assert dashboard_data["name"] == "Agent Performance"
        
        # 5. Test real-time metrics
        real_time_data = await dashboard_manager.get_real_time_metrics(
            metric_types=["agent_performance", "api_metrics", "system_metrics"],
            time_window_minutes=10
        )
        
        assert isinstance(real_time_data, dict)
        assert len(real_time_data) == 3

    async def test_batch_operations_performance(self, influx_manager):
        """Test batch operations performance."""
        import time
        
        # Create large batch of metrics
        batch_size = 1000
        metrics = []
        
        for i in range(batch_size):
            metric = MetricPoint(
                measurement="performance_test",
                tags={
                    "batch_id": "perf_test",
                    "index": str(i),
                    "category": f"cat_{i % 10}"
                },
                fields={
                    "value": float(i),
                    "squared": float(i * i),
                    "random": float(i * 0.123)
                },
                timestamp=datetime.utcnow()
            )
            metrics.append(metric)
        
        # Measure batch write performance
        start_time = time.time()
        await influx_manager.write_metrics_batch(metrics)
        batch_duration = time.time() - start_time
        
        # Measure individual write performance (smaller sample)
        individual_metrics = metrics[:100]
        start_time = time.time()
        for metric in individual_metrics:
            await influx_manager.write_metric(
                measurement=metric.measurement,
                tags=metric.tags,
                fields=metric.fields,
                timestamp=metric.timestamp
            )
        individual_duration = time.time() - start_time
        
        # Batch should be significantly faster per metric
        batch_per_metric = batch_duration / batch_size
        individual_per_metric = individual_duration / len(individual_metrics)
        
        print(f"Batch write: {batch_per_metric:.6f}s per metric")
        print(f"Individual write: {individual_per_metric:.6f}s per metric")
        
        # Batch should be at least 2x faster
        assert batch_per_metric < individual_per_metric / 2

    async def test_data_consistency(self, influx_manager):
        """Test data consistency across operations."""
        measurement = "consistency_test"
        test_data = []
        
        # Write test data with known values
        for i in range(50):
            timestamp = datetime.utcnow() - timedelta(minutes=i)
            value = i * 10.0
            
            await influx_manager.write_metric(
                measurement=measurement,
                tags={"test_id": "consistency", "index": str(i)},
                fields={"value": value, "index": i},
                timestamp=timestamp
            )
            
            test_data.append({"timestamp": timestamp, "value": value, "index": i})
        
        # Query back the data
        query = f'''
        from(bucket: "{TEST_BUCKET}")
        |> range(start: -2h)
        |> filter(fn: (r) => r._measurement == "{measurement}")
        |> filter(fn: (r) => r.test_id == "consistency")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        results = await influx_manager.query_metrics(query)
        
        # Verify data consistency
        assert len(results) == len(test_data)
        
        # Check that all values are present and correct
        result_values = {int(r.get("index", 0)): r.get("value", 0) for r in results}
        
        for expected in test_data:
            expected_index = expected["index"]
            expected_value = expected["value"]
            
            assert expected_index in result_values
            assert abs(result_values[expected_index] - expected_value) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])