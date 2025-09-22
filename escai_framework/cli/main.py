"""
Main CLI application for ESCAI Framework
"""

import sys
import time
from pathlib import Path

import click
from rich.panel import Panel

from .commands.monitor import monitor_group
from .commands.analyze import analyze_group
from .commands.config import config_group
from .commands.session import session_group
from .commands.publication import publication_group
from .commands.config_mgmt import config_mgmt_group
from .commands.logs import logs_group
from .commands.help import help_group
from .utils.logo import display_logo
from .utils.console import get_console
from .utils.interactive_menu import launch_interactive_menu
from .utils.error_handling import CLIErrorHandler
from .utils.error_decorators import handle_cli_errors, create_command_context
from .utils.logging_system import initialize_logging, get_logger, set_debug_mode
from .utils.debug_mode import enable_debug, disable_debug, is_debug_enabled
from .utils.error_tracker import initialize_error_tracking, error_tracking_context
from .utils.log_management import initialize_log_management
from .integration.framework_connector import get_framework_connector
from .utils.startup_optimizer import optimize_cli_startup, finalize_cli_startup

console = get_console()
error_handler = CLIErrorHandler(console=console)

# Initialize logging system
log_dir = Path.home() / ".escai" / "logs"
log_manager = initialize_logging(log_dir, debug_mode=False)
error_tracker = initialize_error_tracking()
log_file_manager = initialize_log_management(log_dir)
logger = get_logger("cli.main")

@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--interactive', '-i', is_flag=True, help='Launch interactive menu system')
@click.option('--debug', is_flag=True, help='Enable debug mode with verbose output')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
@handle_cli_errors(error_handler=error_handler, context_factory=create_command_context("main"))
def cli(ctx, version, interactive, debug, verbose):
    """
    ESCAI Framework - Epistemic State and Causal Analysis Intelligence
    
    Monitor autonomous agent cognition in real-time with deep insights
    into how AI agents think, decide, and behave during task execution.
    
    Use 'escai' without arguments to see quick start guide, or use 'escai --interactive' 
    to launch the interactive menu system.
    """
    # Configure logging based on options
    if debug:
        enable_debug(verbose=True, interactive=True)
        set_debug_mode(True)
        logger.info("Debug mode enabled")
    elif verbose:
        set_debug_mode(True)
        logger.info("Verbose logging enabled")
    
    # Store context for error tracking
    ctx.ensure_object(dict)
    ctx.obj['session_id'] = f"cli_{int(time.time())}"
    
    if version:
        logger.user_action("Version information requested")
        click.echo("ESCAI Framework v1.0.0")
        return
    
    if interactive:
        logger.user_action("Interactive menu launched")
        with error_tracking_context(session_id=ctx.obj['session_id'], command="interactive"):
            launch_interactive_menu()
        return
    
    if ctx.invoked_subcommand is None:
        logger.user_action("CLI started without subcommand")
        display_logo()
        console.print("\n[bold cyan]Welcome to ESCAI Framework![/bold cyan]")
        console.print("Use [bold]escai --help[/bold] to see available commands.\n")
        
        # Show debug status if enabled
        if is_debug_enabled():
            console.print("[dim]Debug mode is enabled[/dim]\n")
        
        # Show quick start guide
        quick_start = Panel(
            "[bold]Quick Start:[/bold]\n\n"
            "• [cyan]escai monitor start[/cyan] - Start monitoring an agent\n"
            "• [cyan]escai analyze patterns[/cyan] - Analyze behavioral patterns\n"
            "• [cyan]escai analyze causal[/cyan] - Explore causal relationships\n"
            "• [cyan]escai config setup[/cyan] - Configure database connections\n"
            "• [cyan]escai session list[/cyan] - View active sessions\n\n"
            "• [yellow]escai --interactive[/yellow] - Launch interactive menu system\n"
            "• [dim]escai --debug[/dim] - Enable debug mode with verbose output",
            title="Getting Started",
            border_style="blue"
        )
        console.print(quick_start)

# Add command groups
cli.add_command(monitor_group)
cli.add_command(analyze_group)
cli.add_command(config_group)
cli.add_command(session_group)
cli.add_command(publication_group)
cli.add_command(config_mgmt_group)
cli.add_command(logs_group)
cli.add_command(help_group)

def main():
    """Entry point for the CLI application"""
    # Optimize startup performance
    optimize_cli_startup()
    
    try:
        logger.info("CLI application starting")
        cli()
        logger.info("CLI application completed successfully")
    except KeyboardInterrupt:
        logger.user_action("Operation cancelled by user")
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        # Track the error
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_type and exc_value and isinstance(exc_value, Exception):
            from .utils.error_tracker import track_error
            error_id = track_error(exc_type, exc_value, exc_traceback, command="main")
            logger.critical(f"Unhandled exception in CLI main: {e}", extra={"error_id": error_id})
        else:
            logger.critical(f"Unhandled exception in CLI main: {e}")
        
        # Use the error handler for unhandled exceptions
        error_handler.handle_error(e)
        
        # Check if it's a critical error
        if error_handler.is_feature_degraded("core_functionality"):
            logger.critical("Core functionality degraded, exiting")
            sys.exit(1)
        else:
            # Non-critical error, continue with graceful degradation
            logger.warning("Continuing with limited functionality")
            console.print("\n[yellow]Continuing with limited functionality...[/yellow]")
            sys.exit(0)
    finally:
        # Finalize startup optimization
        finalize_cli_startup()
        
        # Clean up logging resources
        if is_debug_enabled():
            disable_debug()

if __name__ == '__main__':
    main()