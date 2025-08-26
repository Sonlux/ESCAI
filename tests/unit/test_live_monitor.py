"""
Unit tests for live monitoring components
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from escai_framework.cli.utils.live_monitor import (
    MonitoringMetric, AgentStatus, LiveDataSource, LiveDashboard,
    StreamingLogViewer
)


class TestMonitoringMetric:
    """Test monitoring metric functionality"""
    
    def test_metric_creation(self):
        """Test metric creation"""
        metric = MonitoringMetric(
            name="CPU Usage",
            current_value=50.0,
            unit="%",
            threshold_warning=70.0,
            threshold_critical=90.0
        )
        
        assert metric.name == "CPU Usage"
        assert metric.current_value == 50.0
        assert metric.unit == "%"
        assert metric.threshold_warning == 70.0
        assert metric.threshold_critical == 90.0
        assert metric.history == []
    
    def test_add_value(self):
        """Test adding values to metric"""
        metric = MonitoringMetric("Test", 0.0, max_history=5)
        
        # Add values
        for i in range(10):
            metric.add_value(float(i))
        
        assert metric.current_value == 9.0
        assert len(metric.history) == 5  # Should be limited by max_history
        assert metric.history == [5.0, 6.0, 7.0, 8.0, 9.0]
    
    def test_get_trend(self):
        """Test trend calculation"""
        metric = MonitoringMetric("Test", 0.0)
        
        # Test stable trend
        for i in range(10):
            metric.add_value(50.0)
        assert metric.get_trend() == "stable"
        
        # Test increasing trend
        metric = MonitoringMetric("Test", 0.0)
        for i in range(10):
            metric.add_value(float(i * 10))
        assert metric.get_trend() == "increasing"
        
        # Test decreasing trend
        metric = MonitoringMetric("Test", 0.0)
        for i in range(10):
            metric.add_value(float(100 - i * 10))
        assert metric.get_trend() == "decreasing"
    
    def test_get_status(self):
        """Test status calculation based on thresholds"""
        metric = MonitoringMetric(
            "Test", 0.0,
            threshold_warning=70.0,
            threshold_critical=90.0
        )
        
        # Normal status
        metric.current_value = 50.0
        assert metric.get_status() == "normal"
        
        # Warning status
        metric.current_value = 75.0
        assert metric.get_status() == "warning"
        
        # Critical status
        metric.current_value = 95.0
        assert metric.get_status() == "critical"


class TestAgentStatus:
    """Test agent status functionality"""
    
    def test_agent_creation(self):
        """Test agent status creation"""
        agent = AgentStatus(
            agent_id="test_agent",
            status="active",
            framework="LangChain"
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.status == "active"
        assert agent.framework == "LangChain"
        assert agent.events_processed == 0
        assert agent.error_count == 0
    
    def test_update_activity(self):
        """Test activity update"""
        agent = AgentStatus("test_agent")
        old_time = agent.last_activity
        
        time.sleep(0.01)  # Small delay
        agent.update_activity()
        
        assert agent.last_activity > old_time
    
    def test_increment_events(self):
        """Test event increment"""
        agent = AgentStatus("test_agent")
        old_time = agent.last_activity
        
        time.sleep(0.01)
        agent.increment_events()
        
        assert agent.events_processed == 1
        assert agent.last_activity > old_time
    
    def test_add_error(self):
        """Test error addition"""
        agent = AgentStatus("test_agent")
        old_time = agent.last_activity
        
        time.sleep(0.01)
        agent.add_error()
        
        assert agent.error_count == 1
        assert agent.last_activity > old_time


class TestLiveDataSource:
    """Test live data source functionality"""
    
    def test_initialization(self):
        """Test data source initialization"""
        source = LiveDataSource()
        
        assert len(source.agents) == 5  # Should create 5 sample agents
        assert len(source.metrics) == 6  # Should create 6 sample metrics
        assert not source.running
    
    def test_agent_initialization(self):
        """Test sample agent initialization"""
        source = LiveDataSource()
        
        for agent_id, agent in source.agents.items():
            assert agent_id.startswith("agent_")
            assert agent.agent_id == agent_id
            assert agent.framework in ["LangChain", "AutoGen", "CrewAI", "OpenAI"]
            assert agent.status in ["active", "idle", "processing", "error"]
            assert 0 <= agent.success_rate <= 1
    
    def test_metric_initialization(self):
        """Test sample metric initialization"""
        source = LiveDataSource()
        
        expected_metrics = [
            "cpu_usage", "memory_usage", "active_agents",
            "events_per_second", "success_rate", "response_time"
        ]
        
        for metric_name in expected_metrics:
            assert metric_name in source.metrics
            metric = source.metrics[metric_name]
            assert isinstance(metric, MonitoringMetric)
    
    def test_start_stop(self):
        """Test starting and stopping data source"""
        source = LiveDataSource()
        
        # Test start
        source.start()
        assert source.running
        assert source.update_thread is not None
        
        # Test stop
        source.stop()
        assert not source.running
    
    def test_get_latest_data(self):
        """Test getting latest data snapshot"""
        source = LiveDataSource()
        
        data = source.get_latest_data()
        
        assert "agents" in data
        assert "metrics" in data
        assert "timestamp" in data
        assert len(data["agents"]) == 5
        assert len(data["metrics"]) == 6
        assert isinstance(data["timestamp"], datetime)


class TestLiveDashboard:
    """Test live dashboard functionality"""
    
    def setup_method(self):
        """Set up test dashboard"""
        self.data_source = LiveDataSource()
        self.dashboard = LiveDashboard(self.data_source)
    
    def test_dashboard_creation(self):
        """Test dashboard creation"""
        assert self.dashboard.data_source == self.data_source
        assert not self.dashboard.running
        assert self.dashboard.refresh_rate == 1.0
    
    def test_alert_thresholds(self):
        """Test alert threshold configuration"""
        expected_thresholds = {
            "cpu_usage": 80,
            "memory_usage": 85,
            "response_time": 600
        }
        
        assert self.dashboard.alert_thresholds == expected_thresholds
        assert len(self.dashboard.active_alerts) == 0
    
    def test_update_alerts(self):
        """Test alert updating"""
        # Create metrics that exceed thresholds
        metrics = {
            "cpu_usage": MonitoringMetric("CPU Usage", 85.0, unit="%"),
            "memory_usage": MonitoringMetric("Memory Usage", 90.0, unit="%"),
            "response_time": MonitoringMetric("Response Time", 700.0, unit="ms")
        }
        
        self.dashboard._update_alerts(metrics)
        
        # Should have alerts for all three metrics
        assert len(self.dashboard.active_alerts) == 3
        
        # Check alert messages
        alert_messages = list(self.dashboard.active_alerts)
        assert any("CPU Usage" in msg for msg in alert_messages)
        assert any("Memory Usage" in msg for msg in alert_messages)
        assert any("Response Time" in msg for msg in alert_messages)
    
    def test_layout_creation(self):
        """Test dashboard layout creation"""
        layout = self.dashboard._create_layout()
        
        # Check that layout has expected sections
        child_names = [child.name for child in layout._children]
        assert "header" in child_names
        assert "main" in child_names
        assert "footer" in child_names
    
    @patch('escai_framework.cli.utils.live_monitor.datetime')
    def test_header_creation(self, mock_datetime):
        """Test header panel creation"""
        mock_datetime.now.return_value.strftime.return_value = "2024-01-15 14:30:25"
        
        header = self.dashboard._create_header()
        
        # Check that header is a Panel object
        from rich.panel import Panel
        assert isinstance(header, Panel)
        # We can't easily test the content without rendering, so just check type
    
    def test_metrics_panel_creation(self):
        """Test metrics panel creation"""
        panel = self.dashboard._create_metrics_panel()
        
        # Check that panel is a Panel object
        from rich.panel import Panel
        assert isinstance(panel, Panel)
    
    def test_agents_panel_creation(self):
        """Test agents panel creation"""
        panel = self.dashboard._create_agents_panel()
        
        # Check that panel is a Panel object
        from rich.panel import Panel
        assert isinstance(panel, Panel)


class TestStreamingLogViewer:
    """Test streaming log viewer functionality"""
    
    def test_viewer_creation(self):
        """Test log viewer creation"""
        viewer = StreamingLogViewer()
        
        assert len(viewer.log_buffer) == 0
        assert viewer.max_lines == 100
        assert not viewer.running
        assert len(viewer.filters) == 0
        assert len(viewer.highlight_patterns) == 0
    
    def test_add_filter(self):
        """Test adding log filters"""
        viewer = StreamingLogViewer()
        
        viewer.add_filter("ERROR")
        viewer.add_filter("WARN")
        
        assert len(viewer.filters) == 2
        assert "ERROR" in viewer.filters
        assert "WARN" in viewer.filters
    
    def test_add_highlight(self):
        """Test adding highlight patterns"""
        viewer = StreamingLogViewer()
        
        viewer.add_highlight("ERROR", "bold red")
        viewer.add_highlight("INFO", "bold blue")
        
        assert len(viewer.highlight_patterns) == 2
        assert ("ERROR", "bold red") in viewer.highlight_patterns
        assert ("INFO", "bold blue") in viewer.highlight_patterns
    
    def test_log_display_creation(self):
        """Test log display panel creation"""
        viewer = StreamingLogViewer()
        
        # Add some test logs
        viewer.log_buffer = [
            "[14:30:25.123] INFO  Agent    - Processing request",
            "[14:30:26.456] ERROR Database - Connection failed",
            "[14:30:27.789] WARN  Cache    - Cache miss"
        ]
        
        panel = viewer._create_log_display()
        
        # Check that panel is a Panel object
        from rich.panel import Panel
        assert isinstance(panel, Panel)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_live_dashboard(self):
        """Test live dashboard creation utility"""
        from escai_framework.cli.utils.live_monitor import create_live_dashboard
        
        dashboard = create_live_dashboard()
        
        assert isinstance(dashboard, LiveDashboard)
        assert isinstance(dashboard.data_source, LiveDataSource)
    
    def test_create_streaming_logs(self):
        """Test streaming log viewer creation utility"""
        from escai_framework.cli.utils.live_monitor import create_streaming_logs
        
        viewer = create_streaming_logs()
        
        assert isinstance(viewer, StreamingLogViewer)
        # Should have default highlights
        assert len(viewer.highlight_patterns) == 4
        
        # Check for expected highlights
        patterns = [pattern for pattern, _ in viewer.highlight_patterns]
        assert "ERROR" in patterns
        assert "WARN" in patterns
        assert "INFO" in patterns
        assert "SUCCESS" in patterns


if __name__ == '__main__':
    pytest.main([__file__])