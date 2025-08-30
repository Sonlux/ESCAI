"""
InfluxDB manager for time-series metrics storage and querying.

This module provides functionality for:
- Time-series data ingestion with batch operations
- Performance and timing metrics storage
- Retention policies and data aggregation
- System monitoring metrics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import json

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
    from influxdb_client.client.query_api import QueryApi
    from influxdb_client.client.delete_api import DeleteApi
    from influxdb_client.domain.bucket_retention_rules import BucketRetentionRules
    from influxdb_client.domain.bucket import Bucket
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    # Mock classes for when InfluxDB is not available
    InfluxDBClient = type(None)  # type: ignore
    Point = type(None)  # type: ignore
    WritePrecision = type(None)  # type: ignore
    SYNCHRONOUS = None
    ASYNCHRONOUS = None


logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    measurement: str
    tags: Dict[str, str]
    fields: Dict[str, Union[float, int, str, bool]]
    timestamp: Optional[datetime] = None


@dataclass
class RetentionPolicy:
    """Represents a retention policy configuration."""
    name: str
    duration_seconds: int
    shard_group_duration_seconds: Optional[int] = None
    replication_factor: int = 1


class InfluxDBManager:
    """
    Manager for InfluxDB operations including time-series data ingestion,
    querying, and retention policy management.
    """

    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "",
        org: str = "escai",
        bucket: str = "metrics",
        timeout: int = 10000,
        batch_size: int = 1000
    ):
        """
        Initialize InfluxDB manager.

        Args:
            url: InfluxDB server URL
            token: Authentication token
            org: Organization name
            bucket: Default bucket name
            timeout: Connection timeout in milliseconds
            batch_size: Batch size for bulk operations
        """
        if not INFLUXDB_AVAILABLE:
            raise ImportError("influxdb-client package is required for InfluxDB support")

        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.timeout = timeout
        self.batch_size = batch_size
        
        self._client: Optional[InfluxDBClient] = None
        self._write_api: Optional[Any] = None
        self._query_api: Optional[Any] = None
        self._delete_api: Optional[Any] = None
        self._buckets_api: Optional[Any] = None
        self._buckets_api = None
        
        # Metric schemas for validation
        self.metric_schemas = {
            "agent_performance": {
                "tags": ["agent_id", "session_id", "framework", "task_type"],
                "fields": ["execution_time_ms", "memory_usage_mb", "cpu_usage_percent", "success"]
            },
            "api_metrics": {
                "tags": ["endpoint", "method", "status_code"],
                "fields": ["response_time_ms", "request_size_bytes", "response_size_bytes"]
            },
            "system_metrics": {
                "tags": ["component", "instance"],
                "fields": ["cpu_percent", "memory_percent", "disk_usage_percent", "network_io_bytes"]
            },
            "prediction_metrics": {
                "tags": ["agent_id", "model_type", "prediction_type"],
                "fields": ["accuracy", "confidence", "processing_time_ms", "feature_count"]
            },
            "pattern_metrics": {
                "tags": ["agent_id", "pattern_type", "framework"],
                "fields": ["pattern_frequency", "success_rate", "average_duration_ms", "anomaly_score"]
            }
        }

    async def connect(self) -> None:
        """Establish connection to InfluxDB."""
        try:
            self._client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=self.timeout
            )
            
            # Test connection
            await self._test_connection()
            
            # Initialize APIs
            self._write_api = self._client.write_api(write_options=ASYNCHRONOUS)
            self._query_api = self._client.query_api()
            self._delete_api = self._client.delete_api()
            self._buckets_api = self._client.buckets_api()
            
            # Ensure bucket exists
            await self._ensure_bucket_exists()
            
            logger.info(f"Connected to InfluxDB at {self.url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection to InfluxDB."""
        if self._client:
            self._client.close()
            self._client = None
            self._write_api = None
            self._query_api = None
            self._delete_api = None
            self._buckets_api = None
            logger.info("Disconnected from InfluxDB")

    async def _test_connection(self) -> None:
        """Test InfluxDB connection."""
        if not self._client:
            raise RuntimeError("Client not initialized")
        
        try:
            # Test with a simple query
            health = self._client.health()
            if health.status != "pass":
                raise RuntimeError(f"InfluxDB health check failed: {health.message}")
        except Exception as e:
            raise RuntimeError(f"InfluxDB connection test failed: {e}")

    async def _ensure_bucket_exists(self) -> None:
        """Ensure the default bucket exists."""
        try:
            existing_buckets = self._buckets_api.find_buckets()
            bucket_names = [b.name for b in existing_buckets.buckets] if existing_buckets.buckets else []
            
            if self.bucket not in bucket_names:
                # Create bucket with default retention policy (30 days)
                retention_rules = [BucketRetentionRules(type="expire", every_seconds=30 * 24 * 3600)]
                bucket_obj = Bucket(name=self.bucket, retention_rules=retention_rules)
                self._buckets_api.create_bucket(bucket=bucket_obj)
                logger.info(f"Created bucket: {self.bucket}")
            else:
                logger.info(f"Bucket {self.bucket} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure bucket exists: {e}")
            raise

    def _validate_metric_point(self, point: MetricPoint) -> bool:
        """Validate metric point against schema."""
        schema = self.metric_schemas.get(point.measurement)
        if not schema:
            logger.warning(f"No schema defined for measurement: {point.measurement}")
            return True  # Allow unknown measurements
        
        # Check required tags
        required_tags = set(schema["tags"])
        provided_tags = set(point.tags.keys())
        if not required_tags.issubset(provided_tags):
            missing_tags = required_tags - provided_tags
            logger.error(f"Missing required tags for {point.measurement}: {missing_tags}")
            return False
        
        # Check required fields
        required_fields = set(schema["fields"])
        provided_fields = set(point.fields.keys())
        if not required_fields.issubset(provided_fields):
            missing_fields = required_fields - provided_fields
            logger.error(f"Missing required fields for {point.measurement}: {missing_fields}")
            return False
        
        return True

    def _create_influx_point(self, metric_point: MetricPoint) -> Point:
        """Convert MetricPoint to InfluxDB Point."""
        point = Point(metric_point.measurement)
        
        # Add tags
        for key, value in metric_point.tags.items():
            point = point.tag(key, str(value))
        
        # Add fields
        for field_name, field_val in metric_point.fields.items():
            point = point.field(field_name, field_val)
        
        # Add timestamp if provided
        if metric_point.timestamp:
            point = point.time(metric_point.timestamp, WritePrecision.MS)
        
        return point

    async def write_metric(
        self,
        measurement: str,
        tags: Dict[str, str],
        fields: Dict[str, Union[float, int, str, bool]],
        timestamp: Optional[datetime] = None,
        bucket: Optional[str] = None
    ) -> None:
        """
        Write a single metric point to InfluxDB.

        Args:
            measurement: Measurement name
            tags: Tag key-value pairs
            fields: Field key-value pairs
            timestamp: Optional timestamp (defaults to current time)
            bucket: Optional bucket name (defaults to default bucket)
        """
        if not self._write_api:
            raise RuntimeError("InfluxDB not connected")

        metric_point = MetricPoint(
            measurement=measurement,
            tags=tags,
            fields=fields,
            timestamp=timestamp
        )
        
        if not self._validate_metric_point(metric_point):
            raise ValueError(f"Invalid metric point for measurement: {measurement}")

        point = self._create_influx_point(metric_point)
        
        try:
            self._write_api.write(
                bucket=bucket or self.bucket,
                org=self.org,
                record=point
            )
            logger.debug(f"Wrote metric: {measurement}")
        except Exception as e:
            logger.error(f"Failed to write metric {measurement}: {e}")
            raise

    async def write_metrics_batch(
        self,
        metrics: List[MetricPoint],
        bucket: Optional[str] = None
    ) -> None:
        """
        Write multiple metrics in batch for better performance.

        Args:
            metrics: List of metric points
            bucket: Optional bucket name
        """
        if not self._write_api:
            raise RuntimeError("InfluxDB not connected")

        if not metrics:
            return

        # Validate all points
        valid_metrics = []
        for metric in metrics:
            if self._validate_metric_point(metric):
                valid_metrics.append(metric)
            else:
                logger.warning(f"Skipping invalid metric: {metric.measurement}")

        if not valid_metrics:
            logger.warning("No valid metrics to write")
            return

        # Convert to InfluxDB points
        points = [self._create_influx_point(metric) for metric in valid_metrics]
        
        # Write in batches
        for i in range(0, len(points), self.batch_size):
            batch = points[i:i + self.batch_size]
            try:
                self._write_api.write(
                    bucket=bucket or self.bucket,
                    org=self.org,
                    record=batch
                )
                logger.debug(f"Wrote batch of {len(batch)} metrics")
            except Exception as e:
                logger.error(f"Failed to write metrics batch: {e}")
                raise

    async def query_metrics(
        self,
        query: str,
        bucket: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Flux query and return results.

        Args:
            query: Flux query string
            bucket: Optional bucket name

        Returns:
            List of query results as dictionaries
        """
        if not self._query_api:
            raise RuntimeError("InfluxDB not connected")

        try:
            # If bucket not specified in query, add it
            if bucket and 'from(bucket:' not in query:
                query = f'from(bucket: "{bucket}") |> {query}'
            elif not bucket and 'from(bucket:' not in query:
                query = f'from(bucket: "{self.bucket}") |> {query}'
            
            result = self._query_api.query(query=query, org=self.org)
            
            # Convert to list of dictionaries
            records = []
            for table in result:
                for record in table.records:
                    record_dict = {
                        "measurement": record.get_measurement(),
                        "time": record.get_time(),
                        "value": record.get_value(),
                        "field": record.get_field(),
                        "tags": record.values
                    }
                    records.append(record_dict)
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to query metrics: {e}")
            raise

    async def get_agent_performance_metrics(
        self,
        agent_id: str,
        start_time: datetime,
        end_time: datetime,
        bucket: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get performance metrics for a specific agent."""
        query = f'''
        from(bucket: "{bucket or self.bucket}")
        |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
        |> filter(fn: (r) => r._measurement == "agent_performance")
        |> filter(fn: (r) => r.agent_id == "{agent_id}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        return await self.query_metrics(query)

    async def get_api_performance_metrics(
        self,
        endpoint: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        bucket: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get API performance metrics."""
        # Default to last hour if no time range specified
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.utcnow()

        query = f'''
        from(bucket: "{bucket or self.bucket}")
        |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
        |> filter(fn: (r) => r._measurement == "api_metrics")
        '''
        
        if endpoint:
            query += f'|> filter(fn: (r) => r.endpoint == "{endpoint}")'
        
        query += '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        
        return await self.query_metrics(query)

    async def get_system_metrics(
        self,
        component: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        bucket: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get system performance metrics."""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.utcnow()

        query = f'''
        from(bucket: "{bucket or self.bucket}")
        |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
        |> filter(fn: (r) => r._measurement == "system_metrics")
        '''
        
        if component:
            query += f'|> filter(fn: (r) => r.component == "{component}")'
        
        query += '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        
        return await self.query_metrics(query)

    async def create_retention_policy(
        self,
        policy: RetentionPolicy,
        bucket: Optional[str] = None
    ) -> None:
        """Create or update retention policy for a bucket."""
        try:
            target_bucket = bucket or self.bucket
            bucket_obj = self._buckets_api.find_bucket_by_name(target_bucket)
            
            if not bucket_obj:
                raise ValueError(f"Bucket {target_bucket} not found")

            # Create retention rule
            retention_rule = BucketRetentionRules(
                type="expire",
                every_seconds=policy.duration_seconds,
                shard_group_duration_seconds=policy.shard_group_duration_seconds
            )
            
            # Update bucket with new retention policy
            bucket_obj.retention_rules = [retention_rule]
            self._buckets_api.update_bucket(bucket=bucket_obj)
            
            logger.info(f"Updated retention policy for bucket {target_bucket}: {policy.duration_seconds}s")
            
        except Exception as e:
            logger.error(f"Failed to create retention policy: {e}")
            raise

    async def delete_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        predicate: Optional[str] = None,
        bucket: Optional[str] = None
    ) -> None:
        """
        Delete metrics within a time range.

        Args:
            start_time: Start of deletion range
            end_time: End of deletion range
            predicate: Optional predicate for filtering (e.g., '_measurement="agent_performance"')
            bucket: Optional bucket name
        """
        if not self._delete_api:
            raise RuntimeError("InfluxDB not connected")

        try:
            self._delete_api.delete(
                start=start_time,
                stop=end_time,
                predicate=predicate,
                bucket=bucket or self.bucket,
                org=self.org
            )
            logger.info(f"Deleted metrics from {start_time} to {end_time}")
            
        except Exception as e:
            logger.error(f"Failed to delete metrics: {e}")
            raise

    async def get_bucket_info(self, bucket: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a bucket."""
        try:
            target_bucket = bucket or self.bucket
            bucket_obj = self._buckets_api.find_bucket_by_name(target_bucket)
            
            if not bucket_obj:
                raise ValueError(f"Bucket {target_bucket} not found")

            return {
                "name": bucket_obj.name,
                "id": bucket_obj.id,
                "org_id": bucket_obj.org_id,
                "retention_rules": [
                    {
                        "type": rule.type,
                        "every_seconds": rule.every_seconds
                    }
                    for rule in (bucket_obj.retention_rules or [])
                ],
                "created_at": bucket_obj.created_at,
                "updated_at": bucket_obj.updated_at
            }
            
        except Exception as e:
            logger.error(f"Failed to get bucket info: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on InfluxDB connection."""
        try:
            if not self._client:
                return {"status": "disconnected", "message": "Client not initialized"}

            health = self._client.health()
            
            return {
                "status": health.status,
                "message": health.message or "OK",
                "version": health.version,
                "commit": health.commit
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


# Convenience functions for common metric types
async def write_agent_performance_metric(
    manager: InfluxDBManager,
    agent_id: str,
    session_id: str,
    framework: str,
    task_type: str,
    execution_time_ms: float,
    memory_usage_mb: float,
    cpu_usage_percent: float,
    success: bool,
    timestamp: Optional[datetime] = None
) -> None:
    """Write agent performance metric."""
    await manager.write_metric(
        measurement="agent_performance",
        tags={
            "agent_id": agent_id,
            "session_id": session_id,
            "framework": framework,
            "task_type": task_type
        },
        fields={
            "execution_time_ms": execution_time_ms,
            "memory_usage_mb": memory_usage_mb,
            "cpu_usage_percent": cpu_usage_percent,
            "success": success
        },
        timestamp=timestamp
    )


async def write_api_metric(
    manager: InfluxDBManager,
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: float,
    request_size_bytes: int = 0,
    response_size_bytes: int = 0,
    timestamp: Optional[datetime] = None
) -> None:
    """Write API performance metric."""
    await manager.write_metric(
        measurement="api_metrics",
        tags={
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        },
        fields={
            "response_time_ms": response_time_ms,
            "request_size_bytes": request_size_bytes,
            "response_size_bytes": response_size_bytes
        },
        timestamp=timestamp
    )


async def write_system_metric(
    manager: InfluxDBManager,
    component: str,
    instance: str,
    cpu_percent: float,
    memory_percent: float,
    disk_usage_percent: float,
    network_io_bytes: int,
    timestamp: Optional[datetime] = None
) -> None:
    """Write system performance metric."""
    await manager.write_metric(
        measurement="system_metrics",
        tags={
            "component": component,
            "instance": instance
        },
        fields={
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_usage_percent": disk_usage_percent,
            "network_io_bytes": network_io_bytes
        },
        timestamp=timestamp
    )