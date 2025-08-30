"""
Real-time live monitoring displays for ESCAI CLI
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Thread, Event
import queue
import random

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.align import Align

from .console import get_console
from .ascii_viz import ASCIISparkline, ASCIIProgressBar, ASCIILineChart, ChartConfig


@dataclass
class MonitoringMetric:
    """Represents a monitoring metric with history"""
    name: str
    current_value: float
    history: List[float] = field(default_factory=list)
    max_history: int = 100
    unit: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    def add_value(self, value: float):
        """Add a new value to the metric"""
        self.current_value = value
        self.history.append(value)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_trend(self) -> str:
        """Get trend direction"""
        if len(self.history) < 2:
            return "stable"
        
        recent_avg = sum(self.history[-5:]) / min(5, len(self.history))
        older_avg = sum(self.history[-10:-5]) / min(5, len(self.history) - 5) if len(self.history) > 5 else recent_avg
        
        if recent_avg > older_avg * 1.05:
            return "increasing"
        elif recent_avg < older_avg * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def get_status(self) -> str:
        """Get status based on thresholds"""
        if self.threshold_critical and self.current_value >= self.threshold_critical:
            return "critical"
        elif self.threshold_warning and self.current_value >= self.threshold_warning:
            return "warning"
        else:
            return "normal"


@dataclass
class AgentStatus:
    """Represents the status of an agent"""
    agent_id: str
    status: str = "unknown"
    framework: str = ""
    uptime: timedelta = field(default_factory=lambda: timedelta())
    events_processed: int = 0
    success_rate: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    epistemic_state: Dict[str, Any] = field(default_factory=dict)
    current_task: str = ""
    error_count: int = 0
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def increment_events(self):
        """Increment event counter"""
        self.events_processed += 1
        self.update_activity()
    
    def add_error(self):
        """Add error to count"""
        self.error_count += 1
        self.update_activity()


class LiveDataSource:
    """Simulates real-time data source"""
    
    def __init__(self):
        self.agents: Dict[str, AgentStatus] = {}
        self.metrics: Dict[str, MonitoringMetric] = {}
        self.running = False
        self.update_thread = None
        self.data_queue = queue.Queue()
        
        # Initialize sample agents
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize sample agents and metrics"""
        frameworks = ["LangChain", "AutoGen", "CrewAI", "OpenAI"]
        statuses = ["active", "idle", "processing", "error"]
        
        for i in range(5):
            agent_id = f"agent_{i+1:03d}"
            self.agents[agent_id] = AgentStatus(
                agent_id=agent_id,
                status=random.choice(statuses),
                framework=random.choice(frameworks),
                uptime=timedelta(hours=random.randint(0, 24), minutes=random.randint(0, 59)),
                events_processed=random.randint(100, 5000),
                success_rate=random.uniform(0.7, 0.98),
                current_task=f"Task {random.randint(1, 100)}"
            )
        
        # Initialize metrics
        self.metrics = {
            "cpu_usage": MonitoringMetric("CPU Usage", 0.0, unit="%", threshold_warning=70, threshold_critical=90),
            "memory_usage": MonitoringMetric("Memory Usage", 0.0, unit="%", threshold_warning=80, threshold_critical=95),
            "active_agents": MonitoringMetric("Active Agents", 0.0, unit=""),
            "events_per_second": MonitoringMetric("Events/sec", 0.0, unit="eps"),
            "success_rate": MonitoringMetric("Success Rate", 0.0, unit="%"),
            "response_time": MonitoringMetric("Response Time", 0.0, unit="ms", threshold_warning=500, threshold_critical=1000)
        }
    
    def start(self):
        """Start the data source"""
        if not self.running:
            self.running = True
            self.update_thread = Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
    
    def stop(self):
        """Stop the data source"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
    
    def _update_loop(self):
        """Main update loop for generating data"""
        while self.running:
            try:
                # Update metrics with simulated data
                self.metrics["cpu_usage"].add_value(random.uniform(10, 95))
                self.metrics["memory_usage"].add_value(random.uniform(30, 90))
                self.metrics["active_agents"].add_value(len([a for a in self.agents.values() if a.status == "active"]))
                self.metrics["events_per_second"].add_value(random.uniform(50, 200))
                self.metrics["success_rate"].add_value(random.uniform(85, 98))
                self.metrics["response_time"].add_value(random.uniform(100, 800))
                
                # Update agent statuses
                for agent in self.agents.values():
                    if random.random() < 0.1:  # 10% chance to change status
                        agent.status = random.choice(["active", "idle", "processing"])
                    
                    if random.random() < 0.3:  # 30% chance to process event
                        agent.increment_events()
                        agent.success_rate = random.uniform(0.8, 0.99)
                    
                    if random.random() < 0.05:  # 5% chance of error
                        agent.add_error()
                        agent.status = "error"
                
                # Put update notification in queue
                self.data_queue.put({"type": "update", "timestamp": datetime.now()})
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"Error in data update loop: {e}")
                time.sleep(1)
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get latest data snapshot"""
        return {
            "agents": dict(self.agents),
            "metrics": dict(self.metrics),
            "timestamp": datetime.now()
        }


class LiveDashboard:
    """Real-time dashboard with auto-refreshing panels"""
    
    def __init__(self, data_source: LiveDataSource):
        self.data_source = data_source
        self.console = get_console()
        self.running = False
        self.refresh_rate = 1.0  # seconds
        self.sparkline = ASCIISparkline()
        self.progress_bar = ASCIIProgressBar(width=30)
        
        # Alert settings
        self.alert_thresholds = {
            "cpu_usage": 80,
            "memory_usage": 85,
            "response_time": 600
        }
        self.active_alerts: Set[str] = set()
    
    def run(self):
        """Run the live dashboard"""
        self.running = True
        self.data_source.start()
        
        try:
            with Live(self._create_layout(), refresh_per_second=1/self.refresh_rate, screen=True) as live:
                while self.running:
                    try:
                        # Check for data updates
                        try:
                            update = self.data_source.data_queue.get_nowait()
                            live.update(self._create_layout())
                        except queue.Empty:
                            pass
                        
                        time.sleep(0.1)  # Small sleep to prevent busy waiting
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        self.console.print(f"[red]Dashboard error: {e}[/red]")
                        time.sleep(1)
        
        finally:
            self.running = False
            self.data_source.stop()
    
    def _create_layout(self) -> Layout:
        """Create the dashboard layout"""
        layout = Layout()
        
        # Split into main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=5)
        )
        
        # Split main area
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left area
        layout["left"].split_column(
            Layout(name="metrics", ratio=1),
            Layout(name="agents", ratio=1)
        )
        
        # Split right area
        layout["right"].split_column(
            Layout(name="alerts", size=10),
            Layout(name="trends", ratio=1)
        )
        
        # Populate sections
        layout["header"].update(self._create_header())
        layout["metrics"].update(self._create_metrics_panel())
        layout["agents"].update(self._create_agents_panel())
        layout["alerts"].update(self._create_alerts_panel())
        layout["trends"].update(self._create_trends_panel())
        layout["footer"].update(self._create_footer())
        
        return layout
    
    def _create_header(self) -> Panel:
        """Create header panel"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = self.data_source.get_latest_data()
        
        active_agents = len([a for a in data["agents"].values() if a.status == "active"])
        total_agents = len(data["agents"])
        
        header_text = (
            f"[bold cyan]ESCAI Live Monitor[/bold cyan] | "
            f"Time: {current_time} | "
            f"Active Agents: {active_agents}/{total_agents} | "
            f"Alerts: {len(self.active_alerts)}"
        )
        
        return Panel(header_text, style="bold blue")
    
    def _create_metrics_panel(self) -> Panel:
        """Create system metrics panel"""
        data = self.data_source.get_latest_data()
        metrics = data["metrics"]
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Current", style="white", width=10)
        table.add_column("Trend", style="blue", width=20)
        table.add_column("Status", style="green", width=10)
        
        for name, metric in metrics.items():
            # Create sparkline for trend
            trend_spark = self.sparkline.create(metric.history[-20:], 15) if metric.history else "─" * 15
            
            # Determine status color
            status = metric.get_status()
            status_color = {"normal": "green", "warning": "yellow", "critical": "red"}.get(status, "white")
            
            # Format current value
            current_val = f"{metric.current_value:.1f}{metric.unit}"
            
            table.add_row(
                metric.name,
                current_val,
                trend_spark,
                f"[{status_color}]{status.upper()}[/{status_color}]"
            )
        
        return Panel(table, title="System Metrics", border_style="green")
    
    def _create_agents_panel(self) -> Panel:
        """Create agents status panel"""
        data = self.data_source.get_latest_data()
        agents = data["agents"]
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Agent ID", style="cyan", width=12)
        table.add_column("Status", style="white", width=10)
        table.add_column("Framework", style="blue", width=10)
        table.add_column("Events", style="yellow", width=8)
        table.add_column("Success", style="green", width=8)
        table.add_column("Task", style="white", width=15)
        
        for agent in list(agents.values())[:10]:  # Show top 10 agents
            # Status with color
            status_colors = {
                "active": "green",
                "idle": "yellow", 
                "processing": "blue",
                "error": "red"
            }
            status_color = status_colors.get(agent.status, "white")
            status_text = f"[{status_color}]{agent.status}[/{status_color}]"
            
            # Success rate with progress bar
            success_bar = self.progress_bar.create(agent.success_rate)
            
            table.add_row(
                agent.agent_id,
                status_text,
                agent.framework,
                str(agent.events_processed),
                f"{agent.success_rate:.1%}",
                agent.current_task[:15] + "..." if len(agent.current_task) > 15 else agent.current_task
            )
        
        return Panel(table, title="Agent Status", border_style="blue")
    
    def _create_alerts_panel(self) -> Panel:
        """Create alerts panel"""
        data = self.data_source.get_latest_data()
        metrics = data["metrics"]
        
        # Check for new alerts
        self._update_alerts(metrics)
        
        alert_lines = []
        if self.active_alerts:
            for alert in list(self.active_alerts)[:5]:  # Show top 5 alerts
                alert_lines.append(f"🚨 {alert}")
        else:
            alert_lines.append("✅ No active alerts")
        
        # Add recent events
        alert_lines.append("")
        alert_lines.append("[bold]Recent Events:[/bold]")
        alert_lines.append("• Agent_003 completed task")
        alert_lines.append("• High memory usage detected")
        alert_lines.append("• New agent connected")
        
        content = "\n".join(alert_lines)
        return Panel(content, title="Alerts & Events", border_style="red")
    
    def _create_trends_panel(self) -> Panel:
        """Create trends visualization panel"""
        data = self.data_source.get_latest_data()
        metrics = data["metrics"]
        
        # Create mini charts for key metrics
        lines = []
        
        # CPU Usage trend
        cpu_metric = metrics.get("cpu_usage")
        if cpu_metric and cpu_metric.history:
            config = ChartConfig(width=25, height=6, title="CPU Usage")
            chart = ASCIILineChart(config)
            cpu_chart = chart.create(cpu_metric.history[-20:])
            lines.append(cpu_chart)
            lines.append("")
        
        # Success Rate sparkline
        success_metric = metrics.get("success_rate")
        if success_metric and success_metric.history:
            spark = self.sparkline.create(success_metric.history[-20:], 25)
            lines.append(f"Success Rate: {spark}")
            lines.append("")
        
        # Response Time sparkline
        response_metric = metrics.get("response_time")
        if response_metric and response_metric.history:
            spark = self.sparkline.create(response_metric.history[-20:], 25)
            lines.append(f"Response Time: {spark}")
        
        content = "\n".join(lines) if lines else "No trend data available"
        return Panel(content, title="Trends", border_style="magenta")
    
    def _create_footer(self) -> Panel:
        """Create footer with controls"""
        footer_text = (
            "[bold]Controls:[/bold] Ctrl+C to exit | "
            f"Refresh Rate: {self.refresh_rate}s | "
            f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
        )
        return Panel(footer_text, style="dim")
    
    def _update_alerts(self, metrics: Dict[str, MonitoringMetric]):
        """Update active alerts based on current metrics"""
        new_alerts = set()
        
        for name, metric in metrics.items():
            if name in self.alert_thresholds:
                threshold = self.alert_thresholds[name]
                if metric.current_value > threshold:
                    alert_msg = f"{metric.name} is {metric.current_value:.1f}{metric.unit} (threshold: {threshold}{metric.unit})"
                    new_alerts.add(alert_msg)
        
        # Add agent-specific alerts
        data = self.data_source.get_latest_data()
        error_agents = [a for a in data["agents"].values() if a.status == "error"]
        for agent in error_agents:
            new_alerts.add(f"Agent {agent.agent_id} in error state")
        
        self.active_alerts = new_alerts


class StreamingLogViewer:
    """Real-time log streaming viewer"""
    
    def __init__(self):
        self.console = get_console()
        self.log_buffer = []
        self.max_lines = 100
        self.running = False
        self.filters = []
        self.highlight_patterns = []
    
    def add_filter(self, pattern: str):
        """Add log filter pattern"""
        self.filters.append(pattern)
    
    def add_highlight(self, pattern: str, style: str = "bold yellow"):
        """Add highlight pattern"""
        self.highlight_patterns.append((pattern, style))
    
    def start_streaming(self):
        """Start streaming logs"""
        self.running = True
        
        # Simulate log generation
        def generate_logs():
            log_levels = ["INFO", "WARN", "ERROR", "DEBUG"]
            components = ["Agent", "API", "Database", "Cache", "Monitor"]
            messages = [
                "Processing request",
                "Connection established",
                "Task completed successfully",
                "Validation failed",
                "Timeout occurred",
                "Cache miss",
                "Database query executed",
                "Authentication successful"
            ]
            
            while self.running:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                level = random.choice(log_levels)
                component = random.choice(components)
                message = random.choice(messages)
                
                log_entry = f"[{timestamp}] {level:5} {component:8} - {message}"
                
                # Apply filters
                if not self.filters or any(f in log_entry for f in self.filters):
                    self.log_buffer.append(log_entry)
                    
                    # Keep buffer size manageable
                    if len(self.log_buffer) > self.max_lines:
                        self.log_buffer.pop(0)
                
                time.sleep(random.uniform(0.1, 0.5))
        
        log_thread = Thread(target=generate_logs, daemon=True)
        log_thread.start()
        
        try:
            with Live(self._create_log_display(), refresh_per_second=5, screen=True) as live:
                while self.running:
                    live.update(self._create_log_display())
                    time.sleep(0.2)
        except KeyboardInterrupt:
            self.running = False
    
    def _create_log_display(self) -> Panel:
        """Create log display panel"""
        # Get recent logs
        recent_logs = self.log_buffer[-30:] if self.log_buffer else ["No logs available"]
        
        # Apply highlighting
        formatted_logs = []
        for log in recent_logs:
            formatted_log = log
            for pattern, style in self.highlight_patterns:
                if pattern in log:
                    formatted_log = formatted_log.replace(pattern, f"[{style}]{pattern}[/{style}]")
            formatted_logs.append(formatted_log)
        
        content = "\n".join(formatted_logs)
        
        title = f"Live Logs ({len(self.log_buffer)} total)"
        if self.filters:
            title += f" - Filtered: {', '.join(self.filters)}"
        
        return Panel(content, title=title, border_style="cyan")


# Utility functions for creating live monitoring interfaces

def create_live_dashboard() -> LiveDashboard:
    """Create and return a live dashboard instance"""
    data_source = LiveDataSource()
    return LiveDashboard(data_source)


def create_streaming_logs() -> StreamingLogViewer:
    """Create and return a streaming log viewer"""
    viewer = StreamingLogViewer()
    
    # Add some default highlights
    viewer.add_highlight("ERROR", "bold red")
    viewer.add_highlight("WARN", "bold yellow")
    viewer.add_highlight("INFO", "bold blue")
    viewer.add_highlight("SUCCESS", "bold green")
    
    return viewer