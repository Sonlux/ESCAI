"""
Main CLI application for ESCAI Framework
"""

import sys

import click
from rich.panel import Panel

from .commands.monitor import monitor_group
from .commands.analyze import analyze_group
from .commands.config import config_group
from .commands.session import session_group
from .commands.config_mgmt import config_mgmt_group
from .utils.logo import display_logo
from .utils.console import get_console
from .utils.interactive_menu import launch_interactive_menu
from .utils.error_handling import CLIErrorHandler, ErrorCategory, ErrorSeverity
from .utils.error_decorators import handle_cli_errors, create_command_context

console = get_console()
error_handler = CLIErrorHandler(console=console)

@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--interactive', '-i', is_flag=True, help='Launch interactive menu system')
@click.pass_context
@handle_cli_errors(error_handler=error_handler, context_factory=create_command_context("main"))
def cli(ctx, version, interactive):
    """
    ESCAI Framework - Epistemic State and Causal Analysis Intelligence
    
    Monitor autonomous agent cognition in real-time with deep insights
    into how AI agents think, decide, and behave during task execution.
    
    Use 'escai' without arguments to see quick start guide, or use 'escai --interactive' 
    to launch the interactive menu system.
    """
    if version:
        click.echo("ESCAI Framework v1.0.0")
        return
    
    if interactive:
        launch_interactive_menu()
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
            "• [cyan]escai session list[/cyan] - View active sessions\n\n"
            "• [yellow]escai --interactive[/yellow] - Launch interactive menu system",
            title="Getting Started",
            border_style="blue"
        )
        console.print(quick_start)

# Add command groups
cli.add_command(monitor_group)
cli.add_command(analyze_group)
cli.add_command(config_group)
cli.add_command(session_group)
cli.add_command(config_mgmt_group)

def main():
    """Entry point for the CLI application"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        # Use the error handler for unhandled exceptions
        error_handler.handle_error(e)
        
        # Check if it's a critical error
        if error_handler.is_feature_degraded("core_functionality"):
            sys.exit(1)
        else:
            # Non-critical error, continue with graceful degradation
            console.print("\n[yellow]Continuing with limited functionality...[/yellow]")
            sys.exit(0)

if __name__ == '__main__':
    main()