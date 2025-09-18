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
from ..integration.framework_connector import get_framework_connector, framework_context
from ..utils.error_handling import (
    CLIErrorHandler, FrameworkError, ValidationError, NetworkError,
    ErrorSeverity, ErrorSuggestion
)
from ..utils.error_decorators import (
    handle_cli_errors, require_framework, retry_on_network_error,
    graceful_degradation, create_monitoring_context
)

console = get_console()
error_handler = CLIErrorHandler(console=console)

@click.group(name='monitor')
def monitor_group():
    """Real-time monitoring commands"""
    pass

async def _start_monitoring_impl(agent_id: str, framework: str, config: Optional[str], 
                               capture_epistemic: bool, capture_behavioral: bool, capture_performance: bool):
    """Implementation of start monitoring functionality."""
    try:
        # Get framework connector
        connector = get_framework_connector()
        
        # Check framework availability
        available_frameworks = connector.get_available_frameworks()
        if framework not in available_frameworks:
            raise FrameworkError(
                f"Framework '{framework}' is not available",
                framework=framework,
                suggestions=[
                    ErrorSuggestion(
                        action="Install framework",
                        description=f"Install the {framework} framework",
                        command_example=f"pip install {framework}"
                    ),
                    ErrorSuggestion(
                        action="Check available frameworks",
                        description=f"Available frameworks: {', '.join(available_frameworks)}",
                        command_example="escai monitor --help"
                    )
                ]
            )
        
        # Prepare monitoring configuration
        monitoring_config = {
            'capture_epistemic_states': capture_epistemic,
            'capture_behavioral_patterns': capture_behavioral,
            'capture_performance_metrics': capture_performance,
            'max_events_per_second': 100,
            'buffer_size': 1000
        }
        
        # Load additional config from file if provided
        if config:
            import json
            from pathlib import Path
            config_path = Path(config)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    monitoring_config.update(file_config)
            else:
                console.print(f"[warning]Config file not found: {config}[/warning]")
        
        with create_progress_bar("Initializing monitoring...") as progress:
            task = progress.add_task("Setting up instrumentor...", total=100)
            
            progress.update(task, completed=20, description="Validating framework...")
            await asyncio.sleep(0.1)
            
            progress.update(task, completed=40, description="Creating instrumentor...")
            await asyncio.sleep(0.1)
            
            progress.update(task, completed=60, description="Starting monitoring session...")
            
            # Start actual monitoring
            session_id = await connector.start_monitoring(
                agent_id=agent_id,
                framework=framework,
                config=monitoring_config
            )
            
            progress.update(task, completed=80, description="Configuring event capture...")
            await asyncio.sleep(0.1)
            
            progress.update(task, completed=100, description="Monitoring active!")
        
        console.print(f"[success]✅ Monitoring started for {agent_id} ({framework})[/success]")
        console.print(f"Session ID: [highlight]{session_id}[/highlight]")
        console.print(f"Configuration: epistemic={capture_epistemic}, behavioral={capture_behavioral}, performance={capture_performance}")
        console.print("\nUse [cyan]escai monitor status[/cyan] to view real-time updates")
        console.print(f"Use [cyan]escai monitor stop --session-id {session_id}[/cyan] to stop monitoring")
        
    except Exception as e:
        # Re-raise with additional context
        if not isinstance(e, (ValidationError, NetworkError, FrameworkError)):
            raise FrameworkError(
                f"Failed to initialize monitoring for {framework}: {str(e)}",
                framework=framework,
                severity=ErrorSeverity.HIGH
            )
        raise


@monitor_group.command()
@click.option('--agent-id', required=True, help='Agent ID to monitor')
@click.option('--framework', type=click.Choice(['langchain', 'autogen', 'crewai', 'openai']), 
              required=True, help='Agent framework')
@click.option('--config', help='Configuration file path')
@click.option('--capture-epistemic', is_flag=True, default=True, help='Capture epistemic states')
@click.option('--capture-behavioral', is_flag=True, default=True, help='Capture behavioral patterns')
@click.option('--capture-performance', is_flag=True, default=True, help='Capture performance metrics')
def start(agent_id: str, framework: str, config: Optional[str], 
          capture_epistemic: bool, capture_behavioral: bool, capture_performance: bool):
    """Start monitoring an agent using ESCAI framework instrumentors"""
    try:
        # Validate inputs
        if not agent_id or not agent_id.strip():
            console.print("[error]Agent ID cannot be empty[/error]")
            return
        
        if len(agent_id) > 100:
            console.print("[error]Agent ID must be less than 100 characters[/error]")
            return
        
        console.print(f"[info]Starting monitoring for agent: {agent_id}[/info]")
        
        # Run async function
        asyncio.run(_start_monitoring_impl(
            agent_id, framework, config, 
            capture_epistemic, capture_behavioral, capture_performance
        ))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring startup cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[error]Failed to start monitoring: {str(e)}[/error]")
        if isinstance(e, FrameworkError):
            if hasattr(e, 'suggestions') and e.suggestions:
                console.print("\n[info]Suggestions:[/info]")
                for suggestion in e.suggestions:
                    console.print(f"  • {suggestion.description}")
                    if hasattr(suggestion, 'command_example') and suggestion.command_example:
                        console.print(f"    Example: [cyan]{suggestion.command_example}[/cyan]")


@monitor_group.command()
@click.option('--framework', help='Validate specific framework (optional)')
@handle_cli_errors(error_handler=error_handler, context_factory=create_monitoring_context("validate"))
def validate(framework: Optional[str]):
    """Validate framework integration and availability"""
    
    async def _validate_frameworks():
        try:
            connector = get_framework_connector()
            
            if framework:
                # Validate specific framework
                console.print(f"[info]Validating {framework} framework integration...[/info]")
                
                result = await connector.validate_framework_integration(framework)
                
                # Display results
                console.print(f"\n[bold]Validation Results for {framework}:[/bold]")
                console.print(f"Available: {'✅' if result['available'] else '❌'}")
                console.print(f"Instrumentor Created: {'✅' if result['instrumentor_created'] else '❌'}")
                console.print(f"Test Monitoring: {'✅' if result['test_monitoring'] else '❌'}")
                
                if result['events_supported']:
                    console.print(f"Supported Events: {', '.join(result['events_supported'])}")
                
                if result['errors']:
                    console.print(f"\n[red]Errors:[/red]")
                    for error in result['errors']:
                        console.print(f"  • {error}")
                
                if result['warnings']:
                    console.print(f"\n[yellow]Warnings:[/yellow]")
                    for warning in result['warnings']:
                        console.print(f"  • {warning}")
                
                if result['available'] and result['instrumentor_created'] and result['test_monitoring']:
                    console.print(f"\n[success]✅ {framework} integration is working correctly![/success]")
                else:
                    console.print(f"\n[error]❌ {framework} integration has issues[/error]")
            
            else:
                # Validate all frameworks
                console.print("[info]Validating all framework integrations...[/info]")
                
                frameworks = ['langchain', 'autogen', 'crewai', 'openai']
                results = {}
                
                for fw in frameworks:
                    console.print(f"Validating {fw}...")
                    results[fw] = await connector.validate_framework_integration(fw)
                
                # Display summary
                console.print("\n[bold]Framework Integration Summary:[/bold]")
                
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Framework", style="cyan")
                table.add_column("Available", style="white")
                table.add_column("Instrumentor", style="white")
                table.add_column("Test Monitoring", style="white")
                table.add_column("Status", style="white")
                
                for fw, result in results.items():
                    available = "✅" if result['available'] else "❌"
                    instrumentor = "✅" if result['instrumentor_created'] else "❌"
                    test_monitoring = "✅" if result['test_monitoring'] else "❌"
                    
                    if result['available'] and result['instrumentor_created'] and result['test_monitoring']:
                        status = "[green]Working[/green]"
                    elif result['available']:
                        status = "[yellow]Partial[/yellow]"
                    else:
                        status = "[red]Not Available[/red]"
                    
                    table.add_row(fw, available, instrumentor, test_monitoring, status)
                
                console.print(table)
                
                # Show available frameworks
                available_frameworks = connector.get_available_frameworks()
                if available_frameworks:
                    console.print(f"\n[success]Available frameworks: {', '.join(available_frameworks)}[/success]")
                else:
                    console.print("\n[error]No frameworks are currently available[/error]")
                    console.print("Install frameworks with: pip install langchain autogen crewai openai")
        
        except Exception as e:
            console.print(f"[error]Validation failed: {str(e)}[/error]")
    
    # Run async function
    asyncio.run(_validate_frameworks())

@monitor_group.command()
@click.option('--session-id', help='Specific session to stop')
@click.option('--all', is_flag=True, help='Stop all monitoring sessions')
@handle_cli_errors(error_handler=error_handler, context_factory=create_monitoring_context("stop"))
def stop(session_id: Optional[str], all: bool):
    """Stop monitoring sessions using ESCAI framework instrumentors"""
    
    async def _stop_monitoring():
        try:
            connector = get_framework_connector()
            
            if all:
                console.print("[warning]Stopping all monitoring sessions...[/warning]")
                
                # Get all active sessions
                sessions = await connector.get_session_status()
                
                if not sessions:
                    console.print("[info]No active monitoring sessions found[/info]")
                    return
                
                stopped_count = 0
                for session in sessions:
                    try:
                        sid = session['session_id']
                        result = await connector.stop_monitoring(sid)
                        console.print(f"[success]✅ Stopped session {sid} ({session.get('framework', 'unknown')})[/success]")
                        stopped_count += 1
                    except Exception as e:
                        console.print(f"[error]Failed to stop session {session.get('session_id', 'unknown')}: {str(e)}[/error]")
                
                console.print(f"[success]✅ Stopped {stopped_count} monitoring sessions[/success]")
                
            elif session_id:
                console.print(f"[warning]Stopping session: {session_id}[/warning]")
                
                result = await connector.stop_monitoring(session_id)
                
                console.print(f"[success]✅ Session {session_id} stopped[/success]")
                
                # Display summary if available
                if 'summary' in result:
                    summary = result['summary']
                    console.print(f"[info]Session Summary:[/info]")
                    console.print(f"  Agent: {result.get('agent_id', 'unknown')}")
                    console.print(f"  Framework: {result.get('framework', 'unknown')}")
                    console.print(f"  Duration: {summary.get('total_duration_ms', 0)}ms")
                    console.print(f"  Events: {summary.get('total_events', 0)}")
                
            else:
                console.print("[error]Please specify --session-id or --all[/error]")
                
                # Show available sessions
                sessions = await connector.get_session_status()
                if sessions:
                    console.print("\n[info]Active sessions:[/info]")
                    for session in sessions:
                        console.print(f"  {session['session_id']} - {session.get('agent_id', 'unknown')} ({session.get('framework', 'unknown')})")
        
        except Exception as e:
            if isinstance(e, FrameworkError):
                raise
            raise FrameworkError(f"Failed to stop monitoring: {str(e)}", framework="unknown")
    
    # Run async function
    asyncio.run(_stop_monitoring())

@monitor_group.command()
@click.option('--refresh', default=2, help='Refresh interval in seconds')
@click.option('--agent-id', help='Filter by specific agent ID')
@handle_cli_errors(error_handler=error_handler, context_factory=create_monitoring_context("status"))
def status(refresh: int, agent_id: Optional[str]):
    """Display real-time agent status from ESCAI framework instrumentors"""
    
    async def get_current_status():
        """Get current status from framework connector"""
        try:
            connector = get_framework_connector()
            sessions = await connector.get_session_status()
            
            # Filter by agent_id if specified
            if agent_id:
                sessions = [s for s in sessions if s.get('agent_id') == agent_id]
            
            # Convert to display format
            agents = []
            for session in sessions:
                monitoring_stats = session.get('monitoring_stats', {})
                perf_metrics = session.get('performance_metrics', {})
                
                agent_data = {
                    'id': session.get('agent_id', 'unknown'),
                    'session_id': session.get('session_id', 'unknown'),
                    'status': session.get('status', 'unknown'),
                    'framework': session.get('framework', 'unknown'),
                    'uptime': session.get('uptime_formatted', '0s'),
                    'event_count': monitoring_stats.get('events_captured', 0),
                    'last_activity': monitoring_stats.get('last_activity', 'unknown'),
                    'errors': perf_metrics.get('errors_encountered', 0),
                    'performance_overhead': monitoring_stats.get('performance_overhead', 0.0)
                }
                agents.append(agent_data)
            
            return agents
            
        except Exception as e:
            console.print(f"[error]Failed to get status: {str(e)}[/error]")
            return []
    
    def generate_status_display():
        """Generate the status display layout"""
        # Get current data (this will be called synchronously in the display loop)
        agents = []
        try:
            # We need to run the async function in a new event loop
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an event loop, we can't use asyncio.run()
                    # This is a limitation - in a real implementation, we'd need to
                    # restructure this to work properly with async
                    agents = []
                else:
                    agents = asyncio.run(get_current_status())
            except RuntimeError:
                # No event loop running
                agents = asyncio.run(get_current_status())
        except Exception as e:
            console.print(f"[error]Status update failed: {str(e)}[/error]")
            agents = []
        
        if not agents:
            return Panel(
                "[yellow]No active monitoring sessions found[/yellow]\n\n"
                "Use [cyan]escai monitor start --agent-id <id> --framework <framework>[/cyan] to start monitoring",
                title="🔄 Real-time Agent Monitoring",
                style="bold cyan"
            )
        
        layout = Layout()
        layout.split_column(
            Layout(Panel("🔄 Real-time Agent Monitoring", style="bold cyan"), size=3),
            Layout(format_agent_status_table(agents))
        )
        
        return layout
    
    console.print("[info]Starting real-time status display (Press Ctrl+C to exit)[/info]")
    console.print("[info]Note: Status updates every {} seconds[/info]".format(refresh))
    
    try:
        # Initial display
        display = generate_status_display()
        
        with Live(display, refresh_per_second=1/refresh, console=console) as live:
            while True:
                time.sleep(refresh)
                live.update(generate_status_display())
    except KeyboardInterrupt:
        console.print("\n[yellow]Status monitoring stopped[/yellow]")

@monitor_group.command()
@click.option('--agent-id', required=True, help='Agent ID to monitor')
@click.option('--session-id', help='Specific session ID (optional)')
@click.option('--refresh', default=3, help='Refresh interval in seconds')
@handle_cli_errors(error_handler=error_handler, context_factory=create_monitoring_context("epistemic"))
def epistemic(agent_id: str, session_id: Optional[str], refresh: int):
    """Monitor epistemic state in real-time using ESCAI framework"""
    
    async def get_current_epistemic_state():
        """Get current epistemic state from framework connector"""
        try:
            connector = get_framework_connector()
            state = await connector.get_epistemic_state(agent_id, session_id)
            return state
        except Exception as e:
            console.print(f"[error]Failed to get epistemic state: {str(e)}[/error]")
            return {
                'agent_id': agent_id,
                'status': 'error',
                'error': str(e)
            }
    
    def generate_epistemic_display():
        """Generate the epistemic state display"""
        try:
            # Get current epistemic state
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Fallback for when already in event loop
                    state = {
                        'agent_id': agent_id,
                        'status': 'updating',
                        'message': 'Fetching epistemic state...'
                    }
                else:
                    state = asyncio.run(get_current_epistemic_state())
            except RuntimeError:
                state = asyncio.run(get_current_epistemic_state())
            
            # Handle error states
            if state.get('status') == 'error':
                return Panel(
                    f"[red]Error getting epistemic state:[/red]\n{state.get('error', 'Unknown error')}",
                    title=f"Epistemic State - {agent_id}",
                    style="red"
                )
            elif state.get('status') == 'not_monitored':
                return Panel(
                    f"[yellow]Agent {agent_id} is not currently being monitored[/yellow]\n\n"
                    f"Use [cyan]escai monitor start --agent-id {agent_id} --framework <framework>[/cyan] to start monitoring",
                    title=f"Epistemic State - {agent_id}",
                    style="yellow"
                )
            
            return format_epistemic_state(state)
            
        except Exception as e:
            return Panel(
                f"[red]Display error:[/red] {str(e)}",
                title=f"Epistemic State - {agent_id}",
                style="red"
            )
    
    console.print(f"[info]Monitoring epistemic state for {agent_id} (Press Ctrl+C to exit)[/info]")
    if session_id:
        console.print(f"[info]Using session: {session_id}[/info]")
    console.print(f"[info]Refresh interval: {refresh} seconds[/info]")
    
    try:
        # Initial display
        display = generate_epistemic_display()
        
        with Live(display, refresh_per_second=1/refresh, console=console) as live:
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