"""
Monitoring commands for ESCAI CLI
"""

import asyncio
import time
from typing import Optional

import click
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from ..utils.console import get_console
from ..utils.formatters import (
    format_agent_status_table, 
    format_epistemic_state,
    create_progress_bar
)
from ..services.api_client import ESCAIAPIClient

console = get_console()

@click.group(name='monitor')
def monitor_group():
    """Real-time monitoring commands"""
    pass

@monitor_group.command()
@click.option('--agent-id', required=True, help='Agent ID to monitor')
@click.option('--framework', type=click.Choice(['langchain', 'autogen', 'crewai', 'openai']), 
              required=True, help='Agent framework')
@click.option('--config', help='Configuration file path')
def start(agent_id: str, framework: str, config: Optional[str]):
    """Start monitoring an agent"""
    console.print(f"[info]Starting monitoring for agent: {agent_id}[/info]")
    
    with create_progress_bar("Initializing monitoring...") as progress:
        task = progress.add_task("Setting up instrumentor...", total=100)
        
        # Simulate setup steps
        for i in range(0, 101, 20):
            time.sleep(0.1)
            progress.update(task, completed=i)
    
    console.print(f"[success]✅ Monitoring started for {agent_id} ({framework})[/success]")
    console.print(f"Session ID: [highlight]session_{int(time.time())}[/highlight]")
    console.print("\nUse [cyan]escai monitor status[/cyan] to view real-time updates")

@monitor_group.command()
@click.option('--session-id', help='Specific session to stop')
@click.option('--all', is_flag=True, help='Stop all monitoring sessions')
def stop(session_id: Optional[str], all: bool):
    """Stop monitoring sessions"""
    if all:
        console.print("[warning]Stopping all monitoring sessions...[/warning]")
        console.print("[success]✅ All sessions stopped[/success]")
    elif session_id:
        console.print(f"[warning]Stopping session: {session_id}[/warning]")
        console.print(f"[success]✅ Session {session_id} stopped[/success]")
    else:
        console.print("[error]Please specify --session-id or --all[/error]")

@monitor_group.command()
@click.option('--refresh', default=2, help='Refresh interval in seconds')
@click.option('--agent-id', help='Filter by specific agent ID')
def status(refresh: int, agent_id: Optional[str]):
    """Display real-time agent status"""
    
    def generate_status_display():
        # Mock data for demonstration
        agents = [
            {
                'id': 'agent_001',
                'status': 'active',
                'framework': 'langchain',
                'uptime': '2h 15m',
                'event_count': 1247,
                'last_activity': '2s ago'
            },
            {
                'id': 'agent_002', 
                'status': 'idle',
                'framework': 'autogen',
                'uptime': '45m',
                'event_count': 892,
                'last_activity': '1m ago'
            }
        ]
        
        if agent_id:
            agents = [a for a in agents if a['id'] == agent_id]
        
        layout = Layout()
        layout.split_column(
            Layout(Panel("🔄 Real-time Agent Monitoring", style="bold cyan"), size=3),
            Layout(format_agent_status_table(agents))
        )
        
        return layout
    
    console.print("[info]Starting real-time status display (Press Ctrl+C to exit)[/info]")
    
    try:
        with Live(generate_status_display(), refresh_per_second=1/refresh, console=console) as live:
            while True:
                time.sleep(refresh)
                live.update(generate_status_display())
    except KeyboardInterrupt:
        console.print("\n[yellow]Status monitoring stopped[/yellow]")

@monitor_group.command()
@click.option('--agent-id', required=True, help='Agent ID to monitor')
@click.option('--refresh', default=3, help='Refresh interval in seconds')
def epistemic(agent_id: str, refresh: int):
    """Monitor epistemic state in real-time"""
    
    def generate_epistemic_display():
        # Mock epistemic state data
        state = {
            'agent_id': agent_id,
            'beliefs': [
                {'content': 'User wants to analyze sales data', 'confidence': 0.95},
                {'content': 'Data is in CSV format', 'confidence': 0.87},
                {'content': 'Analysis should include trends', 'confidence': 0.72},
                {'content': 'Output format should be visual', 'confidence': 0.68}
            ],
            'knowledge': {
                'fact_count': 156,
                'concept_count': 43,
                'relationship_count': 89
            },
            'goals': [
                {'description': 'Load and validate data', 'progress': 1.0},
                {'description': 'Perform trend analysis', 'progress': 0.65},
                {'description': 'Generate visualizations', 'progress': 0.23}
            ],
            'uncertainty_score': 0.34
        }
        
        return format_epistemic_state(state)
    
    console.print(f"[info]Monitoring epistemic state for {agent_id} (Press Ctrl+C to exit)[/info]")
    
    try:
        with Live(generate_epistemic_display(), refresh_per_second=1/refresh, console=console) as live:
            while True:
                time.sleep(refresh)
                live.update(generate_epistemic_display())
    except KeyboardInterrupt:
        console.print("\n[yellow]Epistemic monitoring stopped[/yellow]")


@monitor_group.command()
@click.option('--refresh-rate', default=1.0, help='Dashboard refresh rate in seconds')
def dashboard(refresh_rate: float):
    """Launch real-time monitoring dashboard"""
    
    console.print("[info]Starting live monitoring dashboard...[/info]")
    console.print("Press Ctrl+C to exit")
    
    from ..utils.live_monitor import create_live_dashboard
    
    try:
        dashboard = create_live_dashboard()
        dashboard.refresh_rate = refresh_rate
        dashboard.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Dashboard error: {str(e)}[/red]")


@monitor_group.command()
@click.option('--filter', 'log_filter', help='Filter logs by pattern')
@click.option('--highlight', multiple=True, help='Highlight patterns in logs')
def logs(log_filter: str, highlight: tuple):
    """Stream live agent logs with filtering and highlighting"""
    
    console.print("[info]Starting live log streaming...[/info]")
    console.print("Press Ctrl+C to exit")
    
    from ..utils.live_monitor import create_streaming_logs
    
    try:
        viewer = create_streaming_logs()
        
        # Add filter if specified
        if log_filter:
            viewer.add_filter(log_filter)
            console.print(f"[info]Filtering logs by: {log_filter}[/info]")
        
        # Add highlights if specified
        for pattern in highlight:
            viewer.add_highlight(pattern, "bold yellow")
            console.print(f"[info]Highlighting: {pattern}[/info]")
        
        viewer.start_streaming()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Log streaming stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Log streaming error: {str(e)}[/red]")


@monitor_group.command()
@click.option('--agent-id', help='Monitor specific agent')
@click.option('--metric', help='Monitor specific metric')
def live(agent_id: str, metric: str):
    """Live monitoring with real-time updates"""
    
    console.print("[info]Starting live monitoring...[/info]")
    
    from ..utils.live_monitor import LiveDataSource, MonitoringMetric
    from ..utils.ascii_viz import ASCIISparkline, ASCIIProgressBar
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    import time
    
    # Create data source
    data_source = LiveDataSource()
    data_source.start()
    
    sparkline = ASCIISparkline()
    progress_bar = ASCIIProgressBar(width=40)
    
    def create_live_display():
        """Create live display panel"""
        data = data_source.get_latest_data()
        
        if agent_id:
            # Show specific agent
            if agent_id in data["agents"]:
                agent = data["agents"][agent_id]
                content = (
                    f"[bold]Agent ID:[/bold] {agent.agent_id}\n"
                    f"[bold]Status:[/bold] {agent.status}\n"
                    f"[bold]Framework:[/bold] {agent.framework}\n"
                    f"[bold]Events Processed:[/bold] {agent.events_processed}\n"
                    f"[bold]Success Rate:[/bold] {agent.success_rate:.1%}\n"
                    f"[bold]Current Task:[/bold] {agent.current_task}\n"
                    f"[bold]Last Activity:[/bold] {agent.last_activity.strftime('%H:%M:%S')}"
                )
                return Panel(content, title=f"Agent {agent_id} - Live Status", border_style="green")
            else:
                return Panel(f"Agent {agent_id} not found", title="Error", border_style="red")
        
        elif metric:
            # Show specific metric
            if metric in data["metrics"]:
                metric_obj = data["metrics"][metric]
                
                # Create sparkline for history
                spark = sparkline.create(metric_obj.history[-30:], 40) if metric_obj.history else "─" * 40
                
                # Create progress bar for current value (if percentage)
                if metric_obj.unit == "%":
                    progress = progress_bar.create(metric_obj.current_value / 100)
                else:
                    progress = f"Current: {metric_obj.current_value:.2f}{metric_obj.unit}"
                
                content = (
                    f"[bold]Metric:[/bold] {metric_obj.name}\n"
                    f"[bold]Current Value:[/bold] {metric_obj.current_value:.2f}{metric_obj.unit}\n"
                    f"[bold]Trend:[/bold] {metric_obj.get_trend()}\n"
                    f"[bold]Status:[/bold] {metric_obj.get_status()}\n\n"
                    f"[bold]History (30 points):[/bold]\n{spark}\n\n"
                    f"[bold]Progress:[/bold]\n{progress}"
                )
                
                return Panel(content, title=f"Metric: {metric_obj.name}", border_style="blue")
            else:
                return Panel(f"Metric {metric} not found", title="Error", border_style="red")
        
        else:
            # Show overview
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            table.add_column("Trend", style="blue")
            table.add_column("Status", style="green")
            
            for name, metric_obj in data["metrics"].items():
                spark = sparkline.create(metric_obj.history[-10:], 15) if metric_obj.history else "─" * 15
                status = metric_obj.get_status()
                status_color = {"normal": "green", "warning": "yellow", "critical": "red"}.get(status, "white")
                
                table.add_row(
                    metric_obj.name,
                    f"{metric_obj.current_value:.1f}{metric_obj.unit}",
                    spark,
                    f"[{status_color}]{status.upper()}[/{status_color}]"
                )
            
            return Panel(table, title="Live Metrics Overview", border_style="cyan")
    
    try:
        with Live(create_live_display(), refresh_per_second=2, screen=True) as live:
            while True:
                live.update(create_live_display())
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Live monitoring stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Live monitoring error: {str(e)}[/red]")
    finally:
        data_source.stop()