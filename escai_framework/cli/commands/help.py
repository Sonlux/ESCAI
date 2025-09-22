"""
Help command for ESCAI CLI.

Provides comprehensive help with cross-references and contextual guidance.
"""

import click
from typing import Optional

from ..utils.help_system import get_help_system
from ..utils.console import get_console
from ..utils.error_decorators import handle_cli_errors, create_command_context
from ..utils.error_handling import CLIErrorHandler

console = get_console()
error_handler = CLIErrorHandler(console=console)


@click.group(name='help', invoke_without_command=True)
@click.argument('topic', required=False)
@click.option('--search', '-s', help='Search help content')
@click.pass_context
@handle_cli_errors(error_handler=error_handler, context_factory=create_command_context("help"))
def help_group(ctx, topic: Optional[str], search: Optional[str]):
    """
    Comprehensive help system with cross-references.
    
    Get help for commands, topics, and workflows with contextual guidance
    and cross-references to related functionality.
    
    Examples:
        escai help                    # Show quick reference
        escai help monitor            # Show command help
        escai help getting_started    # Show topic help
        escai help --search patterns  # Search help content
    """
    help_system = get_help_system()
    
    if search:
        help_system.search_help(search)
        return
    
    if topic:
        # Try to show command help first, then topic help
        if '.' in topic:
            # Handle subcommand help (e.g., "monitor.start")
            command, subcommand = topic.split('.', 1)
            help_system.show_command_help(command, subcommand)
        else:
            # Try as topic first, then as command
            if topic in help_system._topics:
                help_system.show_topic_help(topic)
            elif topic in help_system._commands:
                help_system.show_command_help(topic)
            else:
                # Try both and let them handle the error
                try:
                    help_system.show_topic_help(topic)
                except:
                    help_system.show_command_help(topic)
        return
    
    if ctx.invoked_subcommand is None:
        # Show quick reference by default
        help_system.show_quick_reference()


@help_group.command()
@click.argument('workflow_name')
@handle_cli_errors(error_handler=error_handler, context_factory=create_command_context("help.workflow"))
def workflow(workflow_name: str):
    """
    Show help for a specific workflow.
    
    Workflows are step-by-step guides for common tasks.
    
    Examples:
        escai help workflow basic_monitoring
        escai help workflow pattern_analysis
        escai help workflow causal_analysis
    """
    help_system = get_help_system()
    help_system.show_workflow_help(workflow_name)


@help_group.command()
@click.argument('command_name')
@click.argument('subcommand_name', required=False)
@handle_cli_errors(error_handler=error_handler, context_factory=create_command_context("help.command"))
def command(command_name: str, subcommand_name: Optional[str]):
    """
    Show detailed help for a specific command.
    
    Provides comprehensive information including usage, examples,
    related commands, and common workflows.
    
    Examples:
        escai help command monitor
        escai help command monitor start
        escai help command analyze patterns
    """
    help_system = get_help_system()
    help_system.show_command_help(command_name, subcommand_name)


@help_group.command()
@click.argument('topic_name')
@handle_cli_errors(error_handler=error_handler, context_factory=create_command_context("help.topic"))
def topic(topic_name: str):
    """
    Show help for a specific topic.
    
    Topics provide in-depth guidance on concepts, features,
    and best practices.
    
    Examples:
        escai help topic getting_started
        escai help topic frameworks
        escai help topic troubleshooting
    """
    help_system = get_help_system()
    help_system.show_topic_help(topic_name)


@help_group.command()
@click.argument('query')
@handle_cli_errors(error_handler=error_handler, context_factory=create_command_context("help.search"))
def search(query: str):
    """
    Search help content for a query.
    
    Searches across commands, topics, and workflows to find
    relevant help information.
    
    Examples:
        escai help search monitoring
        escai help search patterns
        escai help search database
    """
    help_system = get_help_system()
    help_system.search_help(query)


@help_group.command()
@handle_cli_errors(error_handler=error_handler, context_factory=create_command_context("help.topics"))
def topics():
    """
    List all available help topics.
    
    Shows all topics available in the help system with
    brief descriptions.
    """
    help_system = get_help_system()
    
    from rich.table import Table
    
    table = Table(title="Available Help Topics")
    table.add_column("Topic", style="cyan", width=20)
    table.add_column("Title", style="white")
    table.add_column("Category", style="dim", width=15)
    
    for topic in help_system._topics.values():
        table.add_row(
            topic.name,
            topic.title,
            topic.category.replace('_', ' ').title()
        )
    
    console.print(table)
    
    console.print("\n[dim]Use 'escai help <topic>' to view detailed help for any topic[/dim]")


@help_group.command()
@handle_cli_errors(error_handler=error_handler, context_factory=create_command_context("help.workflows"))
def workflows():
    """
    List all available workflows.
    
    Shows all workflows available in the help system with
    brief descriptions.
    """
    help_system = get_help_system()
    
    from rich.table import Table
    
    table = Table(title="Available Workflows")
    table.add_column("Workflow", style="cyan", width=25)
    table.add_column("Description", style="white")
    table.add_column("Steps", style="dim", width=8)
    
    for workflow_name, steps in help_system._workflows.items():
        description = workflow_name.replace('_', ' ').title()
        table.add_row(
            workflow_name,
            description,
            str(len(steps))
        )
    
    console.print(table)
    
    console.print("\n[dim]Use 'escai help workflow <name>' to view detailed workflow steps[/dim]")