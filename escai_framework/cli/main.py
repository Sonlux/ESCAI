"""
Main CLI application for ESCAI Framework
"""

import asyncio
import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .commands.monitor import monitor_group
from .commands.analyze import analyze_group
from .commands.config import config_group
from .commands.session import session_group
from .utils.logo import display_logo
from .utils.console import get_console

console = get_console()

@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version information')
@click.pass_context
def cli(ctx, version):
    """
    ESCAI Framework - Epistemic State and Causal Analysis Intelligence
    
    Monitor autonomous agent cognition in real-time with deep insights
    into how AI agents think, decide, and behave during task execution.
    """
    if version:
        click.echo("ESCAI Framework v1.0.0")
        return
    
    if ctx.invoked_subcommand is None:
        display_logo()
        console.print("\n[bold cyan]Welcome to ESCAI Framework![/bold cyan]")
        console.print("Use [bold]escai --help[/bold] to see available commands.\n")
        
        # Show quick start guide
        quick_start = Panel(
            "[bold]Quick Start:[/bold]\n\n"
            "• [cyan]escai monitor start[/cyan] - Start monitoring an agent\n"
            "• [cyan]escai analyze patterns[/cyan] - Analyze behavioral patterns\n"
            "• [cyan]escai analyze causal[/cyan] - Explore causal relationships\n"
            "• [cyan]escai config setup[/cyan] - Configure database connections\n"
            "• [cyan]escai session list[/cyan] - View active sessions",
            title="Getting Started",
            border_style="blue"
        )
        console.print(quick_start)

# Add command groups
cli.add_command(monitor_group)
cli.add_command(analyze_group)
cli.add_command(config_group)
cli.add_command(session_group)

def main():
    """Entry point for the CLI application"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    main()