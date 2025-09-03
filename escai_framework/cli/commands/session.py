"""
Session management commands for ESCAI CLI
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import click
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

from ..utils.console import get_console

console = get_console()

SESSIONS_DIR = Path.home() / '.escai' / 'sessions'

@click.group(name='session')
def session_group():
    """Session management commands"""
    pass

def get_sessions() -> List[Dict[str, Any]]:
    """Get all monitoring sessions"""
    if not SESSIONS_DIR.exists():
        return []
    
    sessions = []
    for session_file in SESSIONS_DIR.glob('*.json'):
        try:
            with open(session_file, 'r') as f:
                session = json.load(f)
                sessions.append(session)
        except Exception:
            logging.warning(f"Could not load session file {session_file}", exc_info=True)
            continue
    
    return sorted(sessions, key=lambda x: x.get('start_time', ''), reverse=True)

def save_session(session: Dict[str, Any]):
    """Save a monitoring session"""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_file = SESSIONS_DIR / f"{session['session_id']}.json"
    
    with open(session_file, 'w') as f:
        json.dump(session, f, indent=2)

@session_group.command()
def list():
    """List all monitoring sessions"""
    
    sessions = get_sessions()
    
    if not sessions:
        console.print("[info]No monitoring sessions found.[/info]")
        return
    
    table = Table(title="Monitoring Sessions", show_header=True, header_style="bold cyan")
    
    table.add_column("Session ID", style="yellow", no_wrap=True)
    table.add_column("Agent ID", style="blue")
    table.add_column("Framework", style="green")
    table.add_column("Status", justify="center")
    table.add_column("Start Time", style="muted")
    table.add_column("Duration", style="white")
    table.add_column("Events", justify="right", style="magenta")
    
    for session in sessions:
        status = session.get('status', 'unknown')
        status_style = "green" if status == 'active' else "red" if status == 'stopped' else "yellow"
        status_icon = "🟢" if status == 'active' else "🔴" if status == 'stopped' else "🟡"
        
        # Calculate duration
        start_time = datetime.fromisoformat(session.get('start_time', ''))
        if status == 'active':
            duration = datetime.now() - start_time
        else:
            end_time = datetime.fromisoformat(session.get('end_time', session.get('start_time', '')))
            duration = end_time - start_time
        
        duration_str = str(duration).split('.')[0]  # Remove microseconds
        
        table.add_row(
            session.get('session_id', 'N/A'),
            session.get('agent_id', 'N/A'),
            session.get('framework', 'N/A'),
            f"{status_icon} [{status_style}]{status}[/{status_style}]",
            session.get('start_time', 'N/A'),
            duration_str,
            str(session.get('event_count', 0))
        )
    
    console.print(table)

@session_group.command()
@click.argument('session_id')
def show(session_id: str):
    """Show detailed session information"""
    
    session_file = SESSIONS_DIR / f"{session_id}.json"
    
    if not session_file.exists():
        console.print(f"[error]Session {session_id} not found.[/error]")
        return
    
    with open(session_file, 'r') as f:
        session = json.load(f)
    
    # Session overview
    overview_content = []
    overview_content.append(f"[bold]Session ID:[/bold] {session.get('session_id', 'N/A')}")
    overview_content.append(f"[bold]Agent ID:[/bold] {session.get('agent_id', 'N/A')}")
    overview_content.append(f"[bold]Framework:[/bold] {session.get('framework', 'N/A')}")
    overview_content.append(f"[bold]Status:[/bold] {session.get('status', 'N/A')}")
    overview_content.append(f"[bold]Start Time:[/bold] {session.get('start_time', 'N/A')}")
    
    if session.get('end_time'):
        overview_content.append(f"[bold]End Time:[/bold] {session.get('end_time')}")
    
    overview_content.append(f"[bold]Event Count:[/bold] {session.get('event_count', 0)}")
    
    overview_panel = Panel(
        "\n".join(overview_content),
        title="Session Overview",
        border_style="blue"
    )
    console.print(overview_panel)
    
    # Configuration
    if 'config' in session:
        config_content = []
        for key, value in session['config'].items():
            config_content.append(f"[bold]{key}:[/bold] {value}")
        
        config_panel = Panel(
            "\n".join(config_content),
            title="Configuration",
            border_style="green"
        )
        console.print(config_panel)
    
    # Statistics
    if 'statistics' in session:
        stats = session['statistics']
        stats_content = []
        stats_content.append(f"[bold]Total Events:[/bold] {stats.get('total_events', 0)}")
        stats_content.append(f"[bold]Epistemic Updates:[/bold] {stats.get('epistemic_updates', 0)}")
        stats_content.append(f"[bold]Pattern Matches:[/bold] {stats.get('pattern_matches', 0)}")
        stats_content.append(f"[bold]Predictions Generated:[/bold] {stats.get('predictions', 0)}")
        stats_content.append(f"[bold]Average Processing Time:[/bold] {stats.get('avg_processing_time', 'N/A')}")
        
        stats_panel = Panel(
            "\n".join(stats_content),
            title="Statistics",
            border_style="magenta"
        )
        console.print(stats_panel)

@session_group.command()
@click.argument('session_id')
def stop(session_id: str):
    """Stop a monitoring session"""
    
    session_file = SESSIONS_DIR / f"{session_id}.json"
    
    if not session_file.exists():
        console.print(f"[error]Session {session_id} not found.[/error]")
        return
    
    with open(session_file, 'r') as f:
        session = json.load(f)
    
    if session.get('status') != 'active':
        console.print(f"[warning]Session {session_id} is not active.[/warning]")
        return
    
    # Update session status
    session['status'] = 'stopped'
    session['end_time'] = datetime.now().isoformat()
    
    save_session(session)
    
    console.print(f"[success]✅ Session {session_id} stopped.[/success]")

@session_group.command()
@click.option('--older-than', default='7d', help='Remove sessions older than (1h, 1d, 7d, 30d)')
@click.option('--status', help='Remove sessions with specific status')
@click.option('--force', is_flag=True, help='Skip confirmation prompt')
def cleanup(older_than: str, status: str, force: bool):
    """Clean up old monitoring sessions"""
    
    sessions = get_sessions()
    
    if not sessions:
        console.print("[info]No sessions to clean up.[/info]")
        return
    
    # Parse time threshold
    time_units = {'h': 'hours', 'd': 'days', 'w': 'weeks'}
    unit = older_than[-1]
    value = int(older_than[:-1])
    
    if unit not in time_units:
        console.print("[error]Invalid time format. Use format like '7d', '24h', '2w'[/error]")
        return
    
    threshold = datetime.now() - timedelta(**{time_units[unit]: value})
    
    # Filter sessions to remove
    sessions_to_remove = []
    for session in sessions:
        session_time = datetime.fromisoformat(session.get('start_time', ''))
        
        should_remove = session_time < threshold
        
        if status and session.get('status') != status:
            should_remove = False
        
        if should_remove:
            sessions_to_remove.append(session)
    
    if not sessions_to_remove:
        console.print("[info]No sessions match the cleanup criteria.[/info]")
        return
    
    # Show sessions to be removed
    console.print(f"[warning]Found {len(sessions_to_remove)} sessions to remove:[/warning]")
    
    for session in sessions_to_remove[:5]:  # Show first 5
        console.print(f"  • {session.get('session_id')} ({session.get('start_time')})")
    
    if len(sessions_to_remove) > 5:
        console.print(f"  ... and {len(sessions_to_remove) - 5} more")
    
    if not force:
        if not Confirm.ask("Proceed with cleanup?", default=False):
            console.print("[info]Cleanup cancelled.[/info]")
            return
    
    # Remove sessions
    removed_count = 0
    for session in sessions_to_remove:
        session_file = SESSIONS_DIR / f"{session['session_id']}.json"
        try:
            session_file.unlink()
            removed_count += 1
        except Exception as e:
            console.print(f"[error]Failed to remove {session['session_id']}: {e}[/error]")
    
    console.print(f"[success]✅ Removed {removed_count} sessions.[/success]")

@session_group.command()
@click.argument('session_id')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']), default='json')
@click.option('--output', help='Output file path')
def export(session_id: str, output_format: str, output: str):
    """Export session data"""
    
    session_file = SESSIONS_DIR / f"{session_id}.json"
    
    if not session_file.exists():
        console.print(f"[error]Session {session_id} not found.[/error]")
        return
    
    with open(session_file, 'r') as f:
        session = json.load(f)
    
    if not output:
        output = f"{session_id}.{output_format}"
    
    if output_format == 'json':
        with open(output, 'w') as f:
            json.dump(session, f, indent=2)
    elif output_format == 'csv':
        import csv
        with open(output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Key', 'Value'])
            for key, value in session.items():
                writer.writerow([key, str(value)])
    
    console.print(f"[success]✅ Session data exported to {output}[/success]")