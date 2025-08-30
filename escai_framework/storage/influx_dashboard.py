"""
InfluxDB monitoring dashboard utilities.

This module provides functionality for creating monitoring dashboards
and visualizations using InfluxDB time-series data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

from .influx_manager import InfluxDBManager
from .influx_models import MetricType


logger = logging.getLogger(__name__)


@dataclass
class DashboardQuery:
    """Represents a dashboard query configuration."""
    name: str
    description: str
    query: str
    refresh_interval_seconds: int = 30
    chart_type: str = "line"  # line, bar, gauge, stat
    unit: str = ""
    thresholds: Optional[Dict[str, float]] = None


@dataclass
class DashboardPanel:
    """Represents a dashboard panel configuration."""
    title: str
    queries: List[DashboardQuery]
    width: int = 12  # Grid width (1-12)
    height: int = 6  # Grid height
    position: Tuple[int, int] = (0, 0)  # (x, y) position


@dataclass
class Dashboard:
    """Represents a complete dashboard configuration."""
    name: str
    description: str
    panels: List[DashboardPanel]
    refresh_interval_seconds: int = 30
    time_range_hours: int = 24


class InfluxDashboardManager:
    """
    Manager for creating and managing InfluxDB monitoring dashboards.
    """

    def __init__(self, influx_manager: InfluxDBManager):
        """
        Initialize dashboard manager.

        Args:
            influx_manager: InfluxDB manager instance
        """
        self.influx_manager = influx_manager
        self.dashboards: Dict[str, Dashboard] = {}
        self._initialize_default_dashboards()

    def _initialize_default_dashboards(self) -> None:
        """Initialize default dashboard configurations."""
        # Agent Performance Dashboard
        agent_dashboard = self._create_agent_performance_dashboard()
        self.dashboards["agent_performance"] = agent_dashboard

        # System Metrics Dashboard
        system_dashboard = self._create_system_metrics_dashboard()
        self.dashboards["system_metrics"] = system_dashboard

        # API Performance Dashboard
        api_dashboard = self._create_api_performance_dashboard()
        self.dashboards["api_performance"] = api_dashboard

        # Prediction Analytics Dashboard
        prediction_dashboard = self._create_prediction_analytics_dashboard()
        self.dashboards["prediction_analytics"] = prediction_dashboard

    def _create_agent_performance_dashboard(self) -> Dashboard:
        """Create agent performance monitoring dashboard."""
        # Execution Time Panel
        execution_time_panel = DashboardPanel(
            title="Agent Execution Time",
            queries=[
                DashboardQuery(
                    name="avg_execution_time",
                    description="Average execution time by framework",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "agent_performance")
                    |> filter(fn: (r) => r._field == "execution_time_ms")
                    |> group(columns: ["framework"])
                    |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
                    |> yield(name: "mean")
                    ''',
                    chart_type="line",
                    unit="ms"
                )
            ],
            width=6,
            height=6,
            position=(0, 0)
        )

        # Success Rate Panel
        success_rate_panel = DashboardPanel(
            title="Agent Success Rate",
            queries=[
                DashboardQuery(
                    name="success_rate",
                    description="Success rate by framework",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "agent_performance")
                    |> filter(fn: (r) => r._field == "success")
                    |> group(columns: ["framework"])
                    |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
                    |> map(fn: (r) => ({{ r with _value: r._value * 100.0 }}))
                    |> yield(name: "success_rate")
                    ''',
                    chart_type="line",
                    unit="%",
                    thresholds={"warning": 80.0, "critical": 60.0}
                )
            ],
            width=6,
            height=6,
            position=(6, 0)
        )

        # Memory Usage Panel
        memory_panel = DashboardPanel(
            title="Agent Memory Usage",
            queries=[
                DashboardQuery(
                    name="memory_usage",
                    description="Memory usage by agent",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "agent_performance")
                    |> filter(fn: (r) => r._field == "memory_usage_mb")
                    |> group(columns: ["agent_id"])
                    |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
                    |> yield(name: "memory")
                    ''',
                    chart_type="line",
                    unit="MB"
                )
            ],
            width=12,
            height=6,
            position=(0, 6)
        )

        return Dashboard(
            name="Agent Performance",
            description="Monitor agent execution performance and resource usage",
            panels=[execution_time_panel, success_rate_panel, memory_panel],
            refresh_interval_seconds=30,
            time_range_hours=24
        )

    def _create_system_metrics_dashboard(self) -> Dashboard:
        """Create system metrics monitoring dashboard."""
        # CPU Usage Panel
        cpu_panel = DashboardPanel(
            title="System CPU Usage",
            queries=[
                DashboardQuery(
                    name="cpu_usage",
                    description="CPU usage by component",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "system_metrics")
                    |> filter(fn: (r) => r._field == "cpu_percent")
                    |> group(columns: ["component"])
                    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
                    |> yield(name: "cpu")
                    ''',
                    chart_type="line",
                    unit="%",
                    thresholds={"warning": 70.0, "critical": 90.0}
                )
            ],
            width=6,
            height=6,
            position=(0, 0)
        )

        # Memory Usage Panel
        memory_panel = DashboardPanel(
            title="System Memory Usage",
            queries=[
                DashboardQuery(
                    name="memory_usage",
                    description="Memory usage by component",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "system_metrics")
                    |> filter(fn: (r) => r._field == "memory_percent")
                    |> group(columns: ["component"])
                    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
                    |> yield(name: "memory")
                    ''',
                    chart_type="line",
                    unit="%",
                    thresholds={"warning": 80.0, "critical": 95.0}
                )
            ],
            width=6,
            height=6,
            position=(6, 0)
        )

        # Network I/O Panel
        network_panel = DashboardPanel(
            title="Network I/O",
            queries=[
                DashboardQuery(
                    name="network_io",
                    description="Network I/O by component",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "system_metrics")
                    |> filter(fn: (r) => r._field == "network_io_bytes")
                    |> group(columns: ["component"])
                    |> aggregateWindow(every: 1m, fn: sum, createEmpty: false)
                    |> derivative(unit: 1s, nonNegative: true)
                    |> yield(name: "network_rate")
                    ''',
                    chart_type="line",
                    unit="bytes/s"
                )
            ],
            width=12,
            height=6,
            position=(0, 6)
        )

        return Dashboard(
            name="System Metrics",
            description="Monitor system resource usage and performance",
            panels=[cpu_panel, memory_panel, network_panel],
            refresh_interval_seconds=15,
            time_range_hours=6
        )

    def _create_api_performance_dashboard(self) -> Dashboard:
        """Create API performance monitoring dashboard."""
        # Response Time Panel
        response_time_panel = DashboardPanel(
            title="API Response Time",
            queries=[
                DashboardQuery(
                    name="response_time",
                    description="Average response time by endpoint",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "api_metrics")
                    |> filter(fn: (r) => r._field == "response_time_ms")
                    |> group(columns: ["endpoint"])
                    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
                    |> yield(name: "response_time")
                    ''',
                    chart_type="line",
                    unit="ms",
                    thresholds={"warning": 500.0, "critical": 1000.0}
                )
            ],
            width=8,
            height=6,
            position=(0, 0)
        )

        # Request Rate Panel
        request_rate_panel = DashboardPanel(
            title="Request Rate",
            queries=[
                DashboardQuery(
                    name="request_rate",
                    description="Requests per second",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "api_metrics")
                    |> filter(fn: (r) => r._field == "response_time_ms")
                    |> aggregateWindow(every: 1m, fn: count, createEmpty: false)
                    |> map(fn: (r) => ({{ r with _value: r._value / 60.0 }}))
                    |> yield(name: "rps")
                    ''',
                    chart_type="stat",
                    unit="req/s"
                )
            ],
            width=4,
            height=6,
            position=(8, 0)
        )

        # Error Rate Panel
        error_rate_panel = DashboardPanel(
            title="API Error Rate",
            queries=[
                DashboardQuery(
                    name="error_rate",
                    description="Error rate by status code",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "api_metrics")
                    |> filter(fn: (r) => r._field == "response_time_ms")
                    |> group(columns: ["status_code"])
                    |> aggregateWindow(every: 5m, fn: count, createEmpty: false)
                    |> yield(name: "requests")
                    ''',
                    chart_type="bar",
                    unit="count"
                )
            ],
            width=12,
            height=6,
            position=(0, 6)
        )

        return Dashboard(
            name="API Performance",
            description="Monitor API endpoint performance and usage",
            panels=[response_time_panel, request_rate_panel, error_rate_panel],
            refresh_interval_seconds=30,
            time_range_hours=12
        )

    def _create_prediction_analytics_dashboard(self) -> Dashboard:
        """Create prediction analytics monitoring dashboard."""
        # Prediction Accuracy Panel
        accuracy_panel = DashboardPanel(
            title="Prediction Accuracy",
            queries=[
                DashboardQuery(
                    name="prediction_accuracy",
                    description="Prediction accuracy by model type",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "prediction_metrics")
                    |> filter(fn: (r) => r._field == "accuracy")
                    |> group(columns: ["model_type"])
                    |> aggregateWindow(every: 10m, fn: mean, createEmpty: false)
                    |> map(fn: (r) => ({{ r with _value: r._value * 100.0 }}))
                    |> yield(name: "accuracy")
                    ''',
                    chart_type="line",
                    unit="%",
                    thresholds={"warning": 80.0, "critical": 70.0}
                )
            ],
            width=6,
            height=6,
            position=(0, 0)
        )

        # Processing Time Panel
        processing_time_panel = DashboardPanel(
            title="Prediction Processing Time",
            queries=[
                DashboardQuery(
                    name="processing_time",
                    description="Processing time by prediction type",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "prediction_metrics")
                    |> filter(fn: (r) => r._field == "processing_time_ms")
                    |> group(columns: ["prediction_type"])
                    |> aggregateWindow(every: 10m, fn: mean, createEmpty: false)
                    |> yield(name: "processing_time")
                    ''',
                    chart_type="line",
                    unit="ms"
                )
            ],
            width=6,
            height=6,
            position=(6, 0)
        )

        # Pattern Analysis Panel
        pattern_panel = DashboardPanel(
            title="Pattern Analysis Metrics",
            queries=[
                DashboardQuery(
                    name="pattern_frequency",
                    description="Pattern frequency by type",
                    query='''
                    from(bucket: "{bucket}")
                    |> range(start: -{time_range}h)
                    |> filter(fn: (r) => r._measurement == "pattern_metrics")
                    |> filter(fn: (r) => r._field == "pattern_frequency")
                    |> group(columns: ["pattern_type"])
                    |> aggregateWindow(every: 15m, fn: sum, createEmpty: false)
                    |> yield(name: "frequency")
                    ''',
                    chart_type="bar",
                    unit="count"
                )
            ],
            width=12,
            height=6,
            position=(0, 6)
        )

        return Dashboard(
            name="Prediction Analytics",
            description="Monitor prediction model performance and pattern analysis",
            panels=[accuracy_panel, processing_time_panel, pattern_panel],
            refresh_interval_seconds=60,
            time_range_hours=48
        )

    async def get_dashboard_data(
        self,
        dashboard_name: str,
        time_range_hours: Optional[int] = None,
        bucket: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get data for a specific dashboard.

        Args:
            dashboard_name: Name of the dashboard
            time_range_hours: Time range in hours (overrides dashboard default)
            bucket: Bucket name (overrides default)

        Returns:
            Dictionary containing dashboard data
        """
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard '{dashboard_name}' not found")

        dashboard = self.dashboards[dashboard_name]
        time_range = time_range_hours or dashboard.time_range_hours
        target_bucket = bucket or self.influx_manager.bucket

        dashboard_data: Dict[str, Any] = {
            "name": dashboard.name,
            "description": dashboard.description,
            "refresh_interval": dashboard.refresh_interval_seconds,
            "time_range_hours": time_range,
            "panels": []
        }

        for panel in dashboard.panels:
            panel_data: Dict[str, Any] = {
                "title": panel.title,
                "width": panel.width,
                "height": panel.height,
                "position": panel.position,
                "queries": []
            }

            for query in panel.queries:
                # Format query with parameters
                formatted_query = query.query.format(
                    bucket=target_bucket,
                    time_range=time_range
                )

                try:
                    # Execute query
                    results = await self.influx_manager.query_metrics(formatted_query)
                    
                    query_data = {
                        "name": query.name,
                        "description": query.description,
                        "chart_type": query.chart_type,
                        "unit": query.unit,
                        "thresholds": query.thresholds,
                        "data": results
                    }
                    
                    panel_data["queries"].append(query_data)
                    
                except Exception as e:
                    logger.error(f"Failed to execute query {query.name}: {e}")
                    # Add empty result for failed queries
                    query_data = {
                        "name": query.name,
                        "description": query.description,
                        "chart_type": query.chart_type,
                        "unit": query.unit,
                        "thresholds": query.thresholds,
                        "data": [],
                        "error": str(e)
                    }
                    panel_data["queries"].append(query_data)

            dashboard_data["panels"].append(panel_data)

        return dashboard_data

    async def get_real_time_metrics(
        self,
        metric_types: List[str],
        time_window_minutes: int = 5,
        bucket: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get real-time metrics for specified types.

        Args:
            metric_types: List of metric types to retrieve
            time_window_minutes: Time window in minutes
            bucket: Bucket name

        Returns:
            Dictionary of metric type to data points
        """
        results: Dict[str, List[Dict[str, Any]]] = {}
        target_bucket = bucket or self.influx_manager.bucket

        for metric_type in metric_types:
            query = f'''
            from(bucket: "{target_bucket}")
            |> range(start: -{time_window_minutes}m)
            |> filter(fn: (r) => r._measurement == "{metric_type}")
            |> aggregateWindow(every: 30s, fn: mean, createEmpty: false)
            |> yield(name: "realtime")
            '''

            try:
                data = await self.influx_manager.query_metrics(query)
                results[metric_type] = data
            except Exception as e:
                logger.error(f"Failed to get real-time metrics for {metric_type}: {e}")
                results[metric_type] = []

        return results

    async def get_alert_conditions(
        self,
        dashboard_name: str,
        bucket: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Check alert conditions for a dashboard.

        Args:
            dashboard_name: Name of the dashboard
            bucket: Bucket name

        Returns:
            List of alert conditions that are triggered
        """
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard '{dashboard_name}' not found")

        dashboard = self.dashboards[dashboard_name]
        alerts = []

        for panel in dashboard.panels:
            for query in panel.queries:
                if not query.thresholds:
                    continue

                # Get latest value for the query
                formatted_query = query.query.format(
                    bucket=bucket or self.influx_manager.bucket,
                    time_range=1  # Last hour for alert checking
                )

                try:
                    results = await self.influx_manager.query_metrics(formatted_query)
                    
                    if results:
                        latest_value = results[-1].get("value", 0)
                        
                        # Check thresholds
                        for threshold_type, threshold_value in query.thresholds.items():
                            if threshold_type == "critical" and latest_value >= threshold_value:
                                alerts.append({
                                    "panel": panel.title,
                                    "query": query.name,
                                    "level": "critical",
                                    "value": latest_value,
                                    "threshold": threshold_value,
                                    "unit": query.unit,
                                    "message": f"{query.description} is {latest_value}{query.unit} (critical: {threshold_value}{query.unit})"
                                })
                            elif threshold_type == "warning" and latest_value >= threshold_value:
                                alerts.append({
                                    "panel": panel.title,
                                    "query": query.name,
                                    "level": "warning",
                                    "value": latest_value,
                                    "threshold": threshold_value,
                                    "unit": query.unit,
                                    "message": f"{query.description} is {latest_value}{query.unit} (warning: {threshold_value}{query.unit})"
                                })

                except Exception as e:
                    logger.error(f"Failed to check alert condition for {query.name}: {e}")

        return alerts

    def list_dashboards(self) -> List[Dict[str, str]]:
        """List available dashboards."""
        return [
            {
                "name": name,
                "title": dashboard.name,
                "description": dashboard.description
            }
            for name, dashboard in self.dashboards.items()
        ]

    def add_custom_dashboard(self, name: str, dashboard: Dashboard) -> None:
        """Add a custom dashboard configuration."""
        self.dashboards[name] = dashboard
        logger.info(f"Added custom dashboard: {name}")

    def remove_dashboard(self, name: str) -> None:
        """Remove a dashboard configuration."""
        if name in self.dashboards:
            del self.dashboards[name]
            logger.info(f"Removed dashboard: {name}")

    async def export_dashboard_config(self, dashboard_name: str) -> Dict[str, Any]:
        """Export dashboard configuration as JSON."""
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard '{dashboard_name}' not found")

        dashboard = self.dashboards[dashboard_name]
        
        return {
            "name": dashboard.name,
            "description": dashboard.description,
            "refresh_interval_seconds": dashboard.refresh_interval_seconds,
            "time_range_hours": dashboard.time_range_hours,
            "panels": [
                {
                    "title": panel.title,
                    "width": panel.width,
                    "height": panel.height,
                    "position": panel.position,
                    "queries": [
                        {
                            "name": query.name,
                            "description": query.description,
                            "query": query.query,
                            "refresh_interval_seconds": query.refresh_interval_seconds,
                            "chart_type": query.chart_type,
                            "unit": query.unit,
                            "thresholds": query.thresholds
                        }
                        for query in panel.queries
                    ]
                }
                for panel in dashboard.panels
            ]
        }

    async def import_dashboard_config(self, name: str, config: Dict[str, Any]) -> None:
        """Import dashboard configuration from JSON."""
        panels = []
        
        for panel_config in config.get("panels", []):
            queries = []
            
            for query_config in panel_config.get("queries", []):
                query = DashboardQuery(
                    name=query_config["name"],
                    description=query_config["description"],
                    query=query_config["query"],
                    refresh_interval_seconds=query_config.get("refresh_interval_seconds", 30),
                    chart_type=query_config.get("chart_type", "line"),
                    unit=query_config.get("unit", ""),
                    thresholds=query_config.get("thresholds")
                )
                queries.append(query)
            
            panel = DashboardPanel(
                title=panel_config["title"],
                queries=queries,
                width=panel_config.get("width", 12),
                height=panel_config.get("height", 6),
                position=tuple(panel_config.get("position", [0, 0]))
            )
            panels.append(panel)
        
        dashboard = Dashboard(
            name=config["name"],
            description=config["description"],
            panels=panels,
            refresh_interval_seconds=config.get("refresh_interval_seconds", 30),
            time_range_hours=config.get("time_range_hours", 24)
        )
        
        self.dashboards[name] = dashboard
        logger.info(f"Imported dashboard: {name}")