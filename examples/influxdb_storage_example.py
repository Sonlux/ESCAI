"""
Example demonstrating InfluxDB time-series storage functionality.

This example shows how to:
1. Connect to InfluxDB
2. Write various types of metrics
3. Query time-series data
4. Use dashboard functionality
5. Set up retention policies
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

from escai_framework.storage.influx_manager import InfluxDBManager, RetentionPolicy
from escai_framework.storage.influx_models import (
    create_agent_performance_metric, create_api_metric, create_system_metric,
    create_prediction_metric, create_pattern_metric, MetricBatch
)
from escai_framework.storage.influx_dashboard import InfluxDashboardManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_influxdb_connection() -> InfluxDBManager:
    """Set up InfluxDB connection."""
    # Configuration (use environment variables in production)
    config = {
        "url": os.getenv("INFLUXDB_URL", "http://localhost:8086"),
        "token": os.getenv("INFLUXDB_TOKEN", "your-token-here"),
        "org": os.getenv("INFLUXDB_ORG", "escai"),
        "bucket": "escai-metrics",
        "batch_size": 1000
    }
    
    manager = InfluxDBManager(**config)
    
    try:
        await manager.connect()
        logger.info("Connected to InfluxDB successfully")
        
        # Check health
        health = await manager.health_check()
        logger.info(f"InfluxDB health: {health['status']}")
        
        return manager
        
    except Exception as e:
        logger.error(f"Failed to connect to InfluxDB: {e}")
        logger.info("Make sure InfluxDB is running and credentials are correct")
        raise


async def write_sample_agent_metrics(manager: InfluxDBManager) -> None:
    """Write sample agent performance metrics."""
    logger.info("Writing sample agent performance metrics...")
    
    agents = ["langchain-agent-1", "autogen-agent-2", "crewai-agent-3"]
    frameworks = ["langchain", "autogen", "crewai"]
    task_types = ["reasoning", "planning", "execution", "analysis"]
    
    # Write metrics for the last 24 hours
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=24)
    
    batch = MetricBatch()
    
    for i in range(100):
        # Generate random timestamp within the last 24 hours
        timestamp = start_time + timedelta(
            seconds=random.randint(0, int((end_time - start_time).total_seconds()))
        )
        
        agent_id = random.choice(agents)
        framework = frameworks[agents.index(agent_id) % len(frameworks)]
        task_type = random.choice(task_types)
        
        # Generate realistic performance metrics
        execution_time = random.uniform(500, 5000)  # 0.5-5 seconds
        memory_usage = random.uniform(100, 1000)    # 100MB-1GB
        cpu_usage = random.uniform(20, 90)          # 20-90%
        success = random.random() > 0.15            # 85% success rate
        
        metric = create_agent_performance_metric(
            agent_id=agent_id,
            session_id=f"session-{i // 10}",
            framework=framework,
            task_type=task_type,
            execution_time_ms=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            success=success,
            error_count=0 if success else random.randint(1, 3),
            retry_count=random.randint(0, 2),
            timestamp=timestamp
        )
        
        batch.add_metric(metric)
    
    # Write batch
    await manager.write_metrics_batch(batch.metrics)
    logger.info(f"Wrote {batch.size()} agent performance metrics")


async def write_sample_api_metrics(manager: InfluxDBManager) -> None:
    """Write sample API performance metrics."""
    logger.info("Writing sample API performance metrics...")
    
    endpoints = [
        "/api/v1/monitor/start",
        "/api/v1/monitor/stop",
        "/api/v1/analysis/epistemic",
        "/api/v1/analysis/patterns",
        "/api/v1/predictions/current"
    ]
    
    methods = ["GET", "POST", "PUT", "DELETE"]
    status_codes = [200, 201, 400, 404, 500]
    status_weights = [0.7, 0.15, 0.08, 0.05, 0.02]  # Mostly successful
    
    batch = MetricBatch()
    
    for i in range(200):
        endpoint = random.choice(endpoints)
        method = random.choice(methods)
        status_code = random.choices(status_codes, weights=status_weights)[0]
        
        # Response time varies by status code
        if status_code < 300:
            response_time = random.uniform(50, 500)
        elif status_code < 500:
            response_time = random.uniform(100, 800)
        else:
            response_time = random.uniform(1000, 5000)
        
        metric = create_api_metric(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time,
            request_size_bytes=random.randint(100, 10000),
            response_size_bytes=random.randint(200, 50000),
            user_id=f"user-{random.randint(1, 20)}" if random.random() > 0.3 else None
        )
        
        batch.add_metric(metric)
    
    await manager.write_metrics_batch(batch.metrics)
    logger.info(f"Wrote {batch.size()} API performance metrics")


async def write_sample_system_metrics(manager: InfluxDBManager) -> None:
    """Write sample system performance metrics."""
    logger.info("Writing sample system performance metrics...")
    
    components = ["api-server", "database", "cache", "queue", "worker"]
    instances = ["primary", "secondary", "worker-1", "worker-2"]
    
    batch = MetricBatch()
    
    # Write metrics for the last hour with 1-minute intervals
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    
    current_time = start_time
    while current_time <= end_time:
        for component in components:
            for instance in instances[:2]:  # Limit instances
                # Generate realistic system metrics
                cpu_percent = random.uniform(20, 80)
                memory_percent = random.uniform(40, 90)
                disk_usage = random.uniform(30, 70)
                network_io = random.randint(1000, 1000000)
                
                metric = create_system_metric(
                    component=component,
                    instance=instance,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    disk_usage_percent=disk_usage,
                    network_io_bytes=network_io,
                    active_connections=random.randint(10, 100),
                    queue_size=random.randint(0, 50),
                    timestamp=current_time
                )
                
                batch.add_metric(metric)
        
        current_time += timedelta(minutes=1)
    
    await manager.write_metrics_batch(batch.metrics)
    logger.info(f"Wrote {batch.size()} system performance metrics")


async def write_sample_prediction_metrics(manager: InfluxDBManager) -> None:
    """Write sample prediction performance metrics."""
    logger.info("Writing sample prediction performance metrics...")
    
    agents = ["agent-1", "agent-2", "agent-3"]
    model_types = ["lstm", "random_forest", "xgboost", "ensemble"]
    prediction_types = ["success", "failure", "completion_time", "resource_usage"]
    
    batch = MetricBatch()
    
    for i in range(50):
        agent_id = random.choice(agents)
        model_type = random.choice(model_types)
        prediction_type = random.choice(prediction_types)
        
        # Generate realistic prediction metrics
        accuracy = random.uniform(0.7, 0.95)
        confidence = random.uniform(0.6, 0.99)
        processing_time = random.uniform(10, 500)
        
        metric = create_prediction_metric(
            agent_id=agent_id,
            model_type=model_type,
            prediction_type=prediction_type,
            accuracy=accuracy,
            confidence=confidence,
            processing_time_ms=processing_time,
            feature_count=random.randint(5, 50),
            prediction_value=random.uniform(0, 1) if random.random() > 0.5 else None,
            actual_value=random.uniform(0, 1) if random.random() > 0.5 else None
        )
        
        batch.add_metric(metric)
    
    await manager.write_metrics_batch(batch.metrics)
    logger.info(f"Wrote {batch.size()} prediction performance metrics")


async def query_metrics_examples(manager: InfluxDBManager) -> None:
    """Demonstrate various metric queries."""
    logger.info("Demonstrating metric queries...")
    
    # 1. Query agent performance metrics
    logger.info("1. Querying agent performance metrics...")
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=6)
    
    agent_metrics = await manager.get_agent_performance_metrics(
        agent_id="langchain-agent-1",
        start_time=start_time,
        end_time=end_time
    )
    logger.info(f"Found {len(agent_metrics)} agent performance records")
    
    # 2. Query API metrics
    logger.info("2. Querying API performance metrics...")
    api_metrics = await manager.get_api_performance_metrics(
        endpoint="/api/v1/monitor/start",
        start_time=start_time,
        end_time=end_time
    )
    logger.info(f"Found {len(api_metrics)} API performance records")
    
    # 3. Query system metrics
    logger.info("3. Querying system performance metrics...")
    system_metrics = await manager.get_system_metrics(
        component="api-server",
        start_time=start_time,
        end_time=end_time
    )
    logger.info(f"Found {len(system_metrics)} system performance records")
    
    # 4. Custom query - Average response time by endpoint
    logger.info("4. Custom query - Average response time by endpoint...")
    custom_query = f'''
    from(bucket: "{manager.bucket}")
    |> range(start: -6h)
    |> filter(fn: (r) => r._measurement == "api_metrics")
    |> filter(fn: (r) => r._field == "response_time_ms")
    |> group(columns: ["endpoint"])
    |> mean()
    |> yield(name: "avg_response_time")
    '''
    
    results = await manager.query_metrics(custom_query)
    logger.info(f"Average response times: {len(results)} endpoints")
    for result in results[:5]:  # Show first 5
        endpoint = result.get("endpoint", "unknown")
        avg_time = result.get("value", 0)
        logger.info(f"  {endpoint}: {avg_time:.2f}ms")


async def demonstrate_dashboard_functionality(manager: InfluxDBManager) -> None:
    """Demonstrate dashboard functionality."""
    logger.info("Demonstrating dashboard functionality...")
    
    dashboard_manager = InfluxDashboardManager(manager)
    
    # 1. List available dashboards
    dashboards = dashboard_manager.list_dashboards()
    logger.info(f"Available dashboards: {len(dashboards)}")
    for dashboard in dashboards:
        logger.info(f"  - {dashboard['name']}: {dashboard['description']}")
    
    # 2. Get dashboard data
    logger.info("Getting agent performance dashboard data...")
    try:
        dashboard_data = await dashboard_manager.get_dashboard_data(
            "agent_performance",
            time_range_hours=6
        )
        
        logger.info(f"Dashboard: {dashboard_data['name']}")
        logger.info(f"Panels: {len(dashboard_data['panels'])}")
        
        for panel in dashboard_data['panels']:
            logger.info(f"  Panel: {panel['title']}")
            for query in panel['queries']:
                data_points = len(query.get('data', []))
                logger.info(f"    Query '{query['name']}': {data_points} data points")
                
    except Exception as e:
        logger.warning(f"Dashboard data retrieval failed: {e}")
    
    # 3. Get real-time metrics
    logger.info("Getting real-time metrics...")
    try:
        real_time_data = await dashboard_manager.get_real_time_metrics(
            metric_types=["agent_performance", "api_metrics", "system_metrics"],
            time_window_minutes=10
        )
        
        for metric_type, data in real_time_data.items():
            logger.info(f"  {metric_type}: {len(data)} recent data points")
            
    except Exception as e:
        logger.warning(f"Real-time metrics retrieval failed: {e}")
    
    # 4. Check alert conditions
    logger.info("Checking alert conditions...")
    try:
        alerts = await dashboard_manager.get_alert_conditions("system_metrics")
        
        if alerts:
            logger.warning(f"Found {len(alerts)} alerts:")
            for alert in alerts:
                logger.warning(f"  {alert['level'].upper()}: {alert['message']}")
        else:
            logger.info("No alerts triggered")
            
    except Exception as e:
        logger.warning(f"Alert checking failed: {e}")


async def demonstrate_retention_policies(manager: InfluxDBManager) -> None:
    """Demonstrate retention policy management."""
    logger.info("Demonstrating retention policy management...")
    
    # 1. Get current bucket info
    bucket_info = await manager.get_bucket_info()
    logger.info(f"Current bucket: {bucket_info['name']}")
    logger.info(f"Retention rules: {len(bucket_info.get('retention_rules', []))}")
    
    # 2. Create a retention policy
    logger.info("Creating retention policy...")
    policy = RetentionPolicy(
        name="short_term",
        duration_seconds=7 * 24 * 3600,  # 7 days
        replication_factor=1
    )
    
    try:
        await manager.create_retention_policy(policy)
        logger.info("Retention policy created successfully")
        
        # Verify policy was applied
        updated_info = await manager.get_bucket_info()
        logger.info(f"Updated retention rules: {len(updated_info.get('retention_rules', []))}")
        
    except Exception as e:
        logger.warning(f"Retention policy creation failed: {e}")


async def demonstrate_batch_operations(manager: InfluxDBManager) -> None:
    """Demonstrate batch operations performance."""
    logger.info("Demonstrating batch operations...")
    
    import time
    
    # Create a large batch of metrics
    batch_size = 1000
    logger.info(f"Creating batch of {batch_size} metrics...")
    
    batch = MetricBatch()
    
    for i in range(batch_size):
        metric = create_agent_performance_metric(
            agent_id=f"batch-agent-{i % 10}",
            session_id=f"batch-session-{i // 100}",
            framework="test",
            task_type="batch_test",
            execution_time_ms=random.uniform(100, 1000),
            memory_usage_mb=random.uniform(50, 500),
            cpu_usage_percent=random.uniform(10, 80),
            success=random.random() > 0.1
        )
        batch.add_metric(metric)
    
    # Measure batch write performance
    logger.info("Writing batch...")
    start_time = time.time()
    await manager.write_metrics_batch(batch.metrics)
    batch_duration = time.time() - start_time
    
    logger.info(f"Batch write completed in {batch_duration:.2f} seconds")
    logger.info(f"Rate: {batch_size / batch_duration:.0f} metrics/second")
    
    # Verify data was written
    query = f'''
    from(bucket: "{manager.bucket}")
    |> range(start: -1h)
    |> filter(fn: (r) => r._measurement == "agent_performance")
    |> filter(fn: (r) => r.task_type == "batch_test")
    |> count()
    '''
    
    results = await manager.query_metrics(query)
    if results:
        count = results[0].get("value", 0)
        logger.info(f"Verified: {count} batch metrics written")


async def cleanup_test_data(manager: InfluxDBManager) -> None:
    """Clean up test data."""
    logger.info("Cleaning up test data...")
    
    try:
        # Delete test metrics from the last day
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)
        
        # Delete batch test data
        await manager.delete_metrics(
            start_time=start_time,
            end_time=end_time,
            predicate='task_type="batch_test"'
        )
        
        logger.info("Test data cleanup completed")
        
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


async def main():
    """Main example function."""
    logger.info("Starting InfluxDB storage example...")
    
    try:
        # 1. Set up connection
        manager = await setup_influxdb_connection()
        
        # 2. Write sample data
        await write_sample_agent_metrics(manager)
        await write_sample_api_metrics(manager)
        await write_sample_system_metrics(manager)
        await write_sample_prediction_metrics(manager)
        
        # 3. Query examples
        await query_metrics_examples(manager)
        
        # 4. Dashboard functionality
        await demonstrate_dashboard_functionality(manager)
        
        # 5. Retention policies
        await demonstrate_retention_policies(manager)
        
        # 6. Batch operations
        await demonstrate_batch_operations(manager)
        
        # 7. Cleanup (optional)
        # await cleanup_test_data(manager)
        
        logger.info("InfluxDB storage example completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
    
    finally:
        # Clean up connection
        if 'manager' in locals():
            await manager.disconnect()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())