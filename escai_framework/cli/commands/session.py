"""
Enhanced session management commands for ESCAI CLI
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import click
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..utils.console import get_console
from ..session_storage import SessionStorage

console = get_console()

# Global session storage instance
session_storage = SessionStorage()

@click.group(name='session')
def session_group():
    """Session management commands"""
    pass

# Legacy functions for backward compatibility - now use SessionStorage

@session_group.command()
@click.option('--agent-id', help='Filter by agent ID')
@click.option('--framework', help='Filter by framework')
@click.option('--status', help='Filter by status')
@click.option('--tags', help='Filter by tags (comma-separated)')
@click.option('--limit', default=50, help='Maximum number of sessions to show')
@click.option('--active', is_flag=True, help='Show only active sessions')
def list(agent_id: Optional[str], framework: Optional[str], status: Optional[str], 
         tags: Optional[str], limit: int, active: bool):
    """List monitoring sessions with advanced filtering"""
    
    # Parse tags
    tag_list = [tag.strip() for tag in tags.split(',')] if tags else None
    
    # If active flag is set, override status filter
    if active:
        status = 'active'
    
    sessions = session_storage.list_sessions(
        agent_id=agent_id,
        framework=framework,
        status=status,
        tags=tag_list,
        limit=limit
    )
    
    if not sessions:
        console.print("[info]No monitoring sessions found.[/info]")
        return
    
    table = Table(title="Monitoring Sessions", show_header=True, header_style="bold cyan")
    
    table.add_column("Session ID", style="yellow", no_wrap=True, max_width=12)
    table.add_column("Agent ID", style="blue", max_width=15)
    table.add_column("Framework", style="green")
    table.add_column("Status", justify="center")
    table.add_column("Start Time", style="muted")
    table.add_column("Duration", style="white")
    table.add_column("Commands", justify="right", style="magenta")
    table.add_column("Tags", style="dim")
    
    for session in sessions:
        status_val = session.get('status', 'unknown')
        status_style = "green" if status_val == 'active' else "red" if status_val == 'completed' else "yellow"
        status_icon = "🟢" if status_val == 'active' else "🔴" if status_val == 'completed' else "🟡"
        
        # Calculate duration
        start_time = datetime.fromisoformat(session.get('start_time', ''))
        if status_val == 'active':
            duration = datetime.now() - start_time
        else:
            end_time_str = session.get('end_time')
            if end_time_str:
                end_time = datetime.fromisoformat(end_time_str)
                duration = end_time - start_time
            else:
                duration = datetime.now() - start_time
        
        duration_str = str(duration).split('.')[0]  # Remove microseconds
        
        # Get command count
        command_history = session_storage.get_command_history(session['session_id'])
        command_count = len(command_history)
        
        # Format tags
        tags_display = ", ".join(session.get('tags', [])[:2])
        if len(session.get('tags', [])) > 2:
            tags_display += "..."
        
        table.add_row(
            session.get('session_id', 'N/A')[:12] + "...",
            session.get('agent_id', 'N/A'),
            session.get('framework', 'N/A'),
            f"{status_icon} [{status_style}]{status_val}[/{status_style}]",
            session.get('start_time', 'N/A')[:16],
            duration_str,
            str(command_count),
            tags_display
        )
    
    console.print(table)
    
    # Show summary statistics
    stats = session_storage.get_session_statistics()
    summary = Panel(
        f"[bold]Total Sessions:[/bold] {stats['total_sessions']} | "
        f"[bold]Active:[/bold] {stats['active_sessions']} | "
        f"[bold]Completed:[/bold] {stats['completed_sessions']} | "
        f"[bold]Unique Agents:[/bold] {stats['unique_agents']}",
        title="Summary",
        border_style="dim"
    )
    console.print(summary)

@session_group.command()
@click.argument('session_id')
@click.option('--show-commands', is_flag=True, help='Show command history')
@click.option('--show-config', is_flag=True, help='Show configuration details')
def show(session_id: str, show_commands: bool, show_config: bool):
    """Show detailed session information"""
    
    session = session_storage.get_session(session_id)
    
    if not session:
        console.print(f"[error]Session {session_id} not found.[/error]")
        return
    
    # Session overview
    overview_content = []
    overview_content.append(f"[bold]Session ID:[/bold] {session.get('session_id', 'N/A')}")
    overview_content.append(f"[bold]Agent ID:[/bold] {session.get('agent_id', 'N/A')}")
    overview_content.append(f"[bold]Framework:[/bold] {session.get('framework', 'N/A')}")
    overview_content.append(f"[bold]Status:[/bold] {session.get('status', 'N/A')}")
    overview_content.append(f"[bold]Start Time:[/bold] {session.get('start_time', 'N/A')}")
    
    if session.get('end_time'):
        overview_content.append(f"[bold]End Time:[/bold] {session.get('end_time')}")
    
    if session.get('description'):
        overview_content.append(f"[bold]Description:[/bold] {session.get('description')}")
    
    if session.get('tags'):
        overview_content.append(f"[bold]Tags:[/bold] {', '.join(session.get('tags', []))}")
    
    # Get command count
    command_history = session_storage.get_command_history(session['session_id'])
    overview_content.append(f"[bold]Commands Executed:[/bold] {len(command_history)}")
    
    overview_panel = Panel(
        "\n".join(overview_content),
        title="Session Overview",
        border_style="blue"
    )
    console.print(overview_panel)
    
    # Configuration
    if show_config and session.get('config'):
        config_content = []
        for key, value in session['config'].items():
            config_content.append(f"[bold]{key}:[/bold] {value}")
        
        config_panel = Panel(
            "\n".join(config_content),
            title="Configuration",
            border_style="green"
        )
        console.print(config_panel)
    
    # Command History
    if show_commands and command_history:
        cmd_table = Table(title="Command History", show_header=True, header_style="bold magenta")
        cmd_table.add_column("Timestamp", style="muted")
        cmd_table.add_column("Command", style="cyan")
        cmd_table.add_column("Success", justify="center")
        cmd_table.add_column("Execution Time", justify="right")
        
        for cmd in command_history[-10:]:  # Show last 10 commands
            success_icon = "✅" if cmd['success'] else "❌"
            exec_time = f"{cmd['execution_time']:.3f}s" if cmd['execution_time'] else "N/A"
            
            cmd_table.add_row(
                cmd['timestamp'][:19],
                cmd['command'],
                success_icon,
                exec_time
            )
        
        console.print(cmd_table)
        
        if len(command_history) > 10:
            console.print(f"[dim]... and {len(command_history) - 10} more commands[/dim]")
    
    # Statistics
    if session.get('statistics'):
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
    
    session = session_storage.get_session(session_id)
    
    if not session:
        console.print(f"[error]Session {session_id} not found.[/error]")
        return
    
    if session.get('status') != 'active':
        console.print(f"[warning]Session {session_id} is not active.[/warning]")
        return
    
    # End the session
    session_storage.end_session(session_id)
    
    console.print(f"[success]✅ Session {session_id} stopped.[/success]")

@session_group.command()
@click.option('--older-than', default='7d', help='Remove sessions older than (1h, 1d, 7d, 30d)')
@click.option('--status', help='Remove sessions with specific status')
@click.option('--force', is_flag=True, help='Skip confirmation prompt')
def cleanup(older_than: str, status: str, force: bool):
    """Clean up old monitoring sessions"""
    
    # Parse time threshold
    time_units = {'h': 1/24, 'd': 1, 'w': 7}
    unit = older_than[-1]
    
    try:
        value = int(older_than[:-1])
    except ValueError:
        console.print("[error]Invalid time format. Use format like '7d', '24h', '2w'[/error]")
        return
    
    if unit not in time_units:
        console.print("[error]Invalid time format. Use format like '7d', '24h', '2w'[/error]")
        return
    
    days = value * time_units[unit]
    
    # Preview what will be removed
    sessions_to_preview = session_storage.list_sessions(status=status, limit=1000)
    cutoff_date = datetime.now() - timedelta(days=days)
    
    sessions_to_remove = [
        s for s in sessions_to_preview 
        if datetime.fromisoformat(s['start_time']) < cutoff_date
    ]
    
    if not sessions_to_remove:
        console.print("[info]No sessions match the cleanup criteria.[/info]")
        return
    
    # Show sessions to be removed
    console.print(f"[warning]Found {len(sessions_to_remove)} sessions to remove:[/warning]")
    
    for session in sessions_to_remove[:5]:  # Show first 5
        console.print(f"  • {session['session_id'][:12]}... ({session['start_time'][:16]})")
    
    if len(sessions_to_remove) > 5:
        console.print(f"  ... and {len(sessions_to_remove) - 5} more")
    
    if not force:
        if not Confirm.ask("Proceed with cleanup?", default=False):
            console.print("[info]Cleanup cancelled.[/info]")
            return
    
    # Remove sessions using storage backend
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Cleaning up sessions...", total=len(sessions_to_remove))
        
        removed_count = 0
        for session in sessions_to_remove:
            try:
                session_storage.delete_session(session['session_id'])
                removed_count += 1
            except Exception as e:
                console.print(f"[error]Failed to remove {session['session_id']}: {e}[/error]")
            
            progress.advance(task)
    
    console.print(f"[success]✅ Removed {removed_count} sessions.[/success]")

@session_group.command()
@click.argument('session_id')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']), default='json')
@click.option('--output', help='Output file path')
@click.option('--include-commands', is_flag=True, help='Include command history in export')
def export(session_id: str, output_format: str, output: str, include_commands: bool):
    """Export session data with enhanced options"""
    
    session = session_storage.get_session(session_id)
    
    if not session:
        console.print(f"[error]Session {session_id} not found.[/error]")
        return
    
    if not output:
        output = f"{session_id[:8]}.{output_format}"
    
    # Prepare export data
    export_data = session.copy()
    
    if include_commands:
        command_history = session_storage.get_command_history(session_id)
        export_data['command_history'] = command_history
    
    if output_format == 'json':
        with open(output, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    elif output_format == 'csv':
        import csv
        with open(output, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Session metadata
            writer.writerow(['Section', 'Key', 'Value'])
            for key, value in session.items():
                if key not in ['command_history']:
                    writer.writerow(['session', key, str(value)])
            
            # Command history if included
            if include_commands:
                command_history = session_storage.get_command_history(session_id)
                for cmd in command_history:
                    for key, value in cmd.items():
                        writer.writerow(['command', key, str(value)])
    
    console.print(f"[success]✅ Session data exported to {output}[/success]")


@session_group.command()
@click.argument('query')
@click.option('--fields', default='agent_id,framework,description', 
              help='Fields to search (comma-separated)')
def search(query: str, fields: str):
    """Search sessions by text query"""
    
    search_fields = [field.strip() for field in fields.split(',')]
    sessions = session_storage.search_sessions(query, search_fields)
    
    if not sessions:
        console.print(f"[info]No sessions found matching '{query}'[/info]")
        return
    
    console.print(f"[info]Found {len(sessions)} sessions matching '{query}':[/info]\n")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Session ID", style="yellow", max_width=12)
    table.add_column("Agent ID", style="blue")
    table.add_column("Framework", style="green")
    table.add_column("Description", style="white")
    table.add_column("Start Time", style="muted")
    
    for session in sessions:
        table.add_row(
            session['session_id'][:12] + "...",
            session.get('agent_id', 'N/A'),
            session.get('framework', 'N/A'),
            session.get('description', 'N/A')[:30] + "..." if len(session.get('description', '')) > 30 else session.get('description', 'N/A'),
            session.get('start_time', 'N/A')[:16]
        )
    
    console.print(table)


@session_group.command()
@click.argument('session_id')
@click.option('--dry-run', is_flag=True, help='Show commands without executing')
@click.option('--interactive', is_flag=True, help='Confirm each command before execution')
def replay(session_id: str, dry_run: bool, interactive: bool):
    """Replay commands from a session"""
    
    session = session_storage.get_session(session_id)
    if not session:
        console.print(f"[error]Session {session_id} not found.[/error]")
        return
    
    command_history = session_storage.get_command_history(session_id)
    if not command_history:
        console.print(f"[info]No commands found in session {session_id}[/info]")
        return
    
    console.print(f"[info]Replaying {len(command_history)} commands from session {session_id}[/info]")
    
    if dry_run:
        console.print("[yellow]DRY RUN - Commands will not be executed[/yellow]\n")
    
    for i, cmd in enumerate(command_history, 1):
        console.print(f"\n[bold cyan]Command {i}/{len(command_history)}:[/bold cyan]")
        console.print(f"[white]{cmd['command']}[/white]")
        
        if cmd['arguments']:
            console.print(f"[dim]Arguments: {json.dumps(cmd['arguments'], indent=2)}[/dim]")
        
        if dry_run:
            continue
        
        if interactive:
            if not Confirm.ask("Execute this command?", default=True):
                console.print("[yellow]Skipped[/yellow]")
                continue
        
        # Here you would implement actual command execution
        # For now, we'll simulate it
        console.print("[green]✅ Command executed (simulated)[/green]")
        
        if cmd['execution_time']:
            time.sleep(min(cmd['execution_time'], 1.0))  # Simulate execution time (max 1s)
    
    if not dry_run:
        console.print(f"\n[success]✅ Replay completed for session {session_id}[/success]")


@session_group.command()
@click.argument('session_ids', nargs=-1, required=True)
@click.option('--output', help='Output file for comparison report')
def compare(session_ids: tuple, output: Optional[str]):
    """Compare multiple sessions"""
    
    if len(session_ids) < 2:
        console.print("[error]At least 2 session IDs are required for comparison[/error]")
        return
    
    sessions = []
    for session_id in session_ids:
        session = session_storage.get_session(session_id)
        if not session:
            console.print(f"[error]Session {session_id} not found.[/error]")
            return
        sessions.append(session)
    
    console.print(f"[info]Comparing {len(sessions)} sessions[/info]\n")
    
    # Basic comparison table
    comparison_table = Table(title="Session Comparison", show_header=True, header_style="bold cyan")
    comparison_table.add_column("Metric", style="bold")
    
    for session in sessions:
        comparison_table.add_column(f"{session['session_id'][:8]}...", style="white")
    
    # Add comparison rows
    metrics = [
        ("Agent ID", lambda s: s.get('agent_id', 'N/A')),
        ("Framework", lambda s: s.get('framework', 'N/A')),
        ("Status", lambda s: s.get('status', 'N/A')),
        ("Start Time", lambda s: s.get('start_time', 'N/A')[:16]),
        ("Commands", lambda s: str(len(session_storage.get_command_history(s['session_id'])))),
        ("Tags", lambda s: ', '.join(s.get('tags', [])[:2])),
    ]
    
    for metric_name, metric_func in metrics:
        row = [metric_name]
        for session in sessions:
            row.append(metric_func(session))
        comparison_table.add_row(*row)
    
    console.print(comparison_table)
    
    # Command comparison
    console.print("\n[bold]Command History Comparison:[/bold]")
    
    for session in sessions:
        commands = session_storage.get_command_history(session['session_id'])
        console.print(f"\n[cyan]{session['session_id'][:12]}...[/cyan]: {len(commands)} commands")
        
        if commands:
            # Show command frequency
            command_counts: Dict[str, int] = {}
            for cmd in commands:
                command_counts[cmd['command']] = command_counts.get(cmd['command'], 0) + 1
            
            for cmd, count in sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                console.print(f"  • {cmd}: {count}x")
    
    if output:
        # Generate detailed comparison report
        report = {
            'comparison_timestamp': datetime.now().isoformat(),
            'sessions': sessions,
            'command_histories': {
                s['session_id']: session_storage.get_command_history(s['session_id']) 
                for s in sessions
            }
        }
        
        with open(output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        console.print(f"\n[success]✅ Detailed comparison report saved to {output}[/success]")


@session_group.command()
@click.argument('session_id')
@click.option('--tags', help='Tags to add (comma-separated)')
@click.option('--description', help='Update description')
@click.option('--remove-tags', help='Tags to remove (comma-separated)')
def tag(session_id: str, tags: Optional[str], description: Optional[str], remove_tags: Optional[str]):
    """Add tags and update session metadata"""
    
    session = session_storage.get_session(session_id)
    if not session:
        console.print(f"[error]Session {session_id} not found.[/error]")
        return
    
    updates = {}
    
    if tags:
        new_tags = [tag.strip() for tag in tags.split(',')]
        current_tags = session.get('tags', [])
        updated_tags = list(set(current_tags + new_tags))
        updates['tags'] = updated_tags
        console.print(f"[info]Added tags: {', '.join(new_tags)}[/info]")
    
    if remove_tags:
        tags_to_remove = [tag.strip() for tag in remove_tags.split(',')]
        current_tags = session.get('tags', [])
        updated_tags = [tag for tag in current_tags if tag not in tags_to_remove]
        updates['tags'] = updated_tags
        console.print(f"[info]Removed tags: {', '.join(tags_to_remove)}[/info]")
    
    if description:
        updates['description'] = description
        console.print(f"[info]Updated description[/info]")
    
    if updates:
        session_storage.update_session(session_id, **updates)
        console.print(f"[success]✅ Session {session_id} updated[/success]")
    else:
        console.print("[info]No updates specified[/info]")


@session_group.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table')
def stats(output_format: str):
    """Show session statistics"""
    
    stats = session_storage.get_session_statistics()
    
    if output_format == 'json':
        console.print(json.dumps(stats, indent=2))
        return
    
    # Display as formatted panels
    overview_content = [
        f"[bold]Total Sessions:[/bold] {stats['total_sessions']}",
        f"[bold]Active Sessions:[/bold] {stats['active_sessions']}",
        f"[bold]Completed Sessions:[/bold] {stats['completed_sessions']}",
        f"[bold]Unique Agents:[/bold] {stats['unique_agents']}",
        f"[bold]Frameworks Used:[/bold] {stats['frameworks_used']}"
    ]
    
    overview_panel = Panel(
        "\n".join(overview_content),
        title="Session Overview",
        border_style="blue"
    )
    console.print(overview_panel)
    
    command_content = [
        f"[bold]Total Commands:[/bold] {stats['total_commands']}",
        f"[bold]Successful Commands:[/bold] {stats['successful_commands']}",
        f"[bold]Success Rate:[/bold] {stats['successful_commands'] / max(stats['total_commands'], 1) * 100:.1f}%"
    ]
    
    if stats['avg_execution_time']:
        command_content.append(f"[bold]Average Execution Time:[/bold] {stats['avg_execution_time']:.3f}s")
    
    command_panel = Panel(
        "\n".join(command_content),
        title="Command Statistics",
        border_style="green"
    )
    console.print(command_panel)