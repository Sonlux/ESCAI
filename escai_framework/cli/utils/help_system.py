"""
Comprehensive help system with cross-references fo    
    def __init__(self) -> None:
        \"\"\"Initialize the help system.\"\"\"
        self.console = get_console()
        self._topics: Dict[str, HelpTopic] = {}
        self._commands: Dict[str, CommandReference] = {}
        self._workflows: Dict[str, List[str]] = {}
        self._initialize_help_content()CLI.

This module provides a unified help system that generates contextual help
with cross-references between commands and features.
"""

import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

from .console import get_console


@dataclass
class HelpTopic:
    """Represents a help topic with metadata and content."""
    name: str
    title: str
    content: str
    category: str
    tags: List[str]
    related_commands: List[str]
    related_topics: List[str]
    examples: List[str]
    prerequisites: List[str]


@dataclass
class CommandReference:
    """Cross-reference information for a command."""
    command: str
    description: str
    category: str
    related_commands: List[str]
    common_workflows: List[str]


class HelpSystem:
    """
    Comprehensive help system with cross-references and contextual guidance.
    
    This system provides:
    - Command help with cross-references
    - Workflow guidance
    - Troubleshooting assistance
    - Quick reference guides
    """
    
    def __init__(self):
        """Initialize the help system."""
        self.console = get_console()
        self._topics: Dict[str, HelpTopic] = {}
        self._commands: Dict[str, CommandReference] = {}
        self._workflows: Dict[str, List[str]] = {}
        self._initialize_help_content()
    
    def _initialize_help_content(self) -> None:
        """Initialize help content and cross-references."""
        # Define command references
        self._commands = {
            'monitor': CommandReference(
                command='monitor',
                description='Monitor agent execution in real-time',
                category='monitoring',
                related_commands=['analyze', 'session', 'config'],
                common_workflows=['basic_monitoring', 'multi_agent_monitoring']
            ),
            'analyze': CommandReference(
                command='analyze',
                description='Analyze agent behavior and patterns',
                category='analysis',
                related_commands=['monitor', 'publication'],
                common_workflows=['pattern_analysis', 'causal_analysis']
            ),
            'config': CommandReference(
                command='config',
                description='Configure ESCAI framework settings',
                category='configuration',
                related_commands=['monitor', 'analyze'],
                common_workflows=['initial_setup', 'database_setup']
            ),
            'session': CommandReference(
                command='session',
                description='Manage monitoring sessions',
                category='session_management',
                related_commands=['monitor', 'analyze'],
                common_workflows=['session_replay', 'session_comparison']
            ),
            'publication': CommandReference(
                command='publication',
                description='Generate publication-ready outputs',
                category='output',
                related_commands=['analyze'],
                common_workflows=['academic_reporting', 'statistical_reporting']
            )
        }
        
        # Define common workflows
        self._workflows = {
            'basic_monitoring': [
                'config setup',
                'monitor start --framework langchain --agent-id my_agent',
                'monitor status',
                'monitor stop'
            ],
            'pattern_analysis': [
                'monitor start --framework langchain --agent-id my_agent',
                'analyze patterns --agent-id my_agent',
                'analyze visualize --type patterns',
                'publication generate --type patterns'
            ],
            'causal_analysis': [
                'monitor start --framework langchain --agent-id my_agent',
                'analyze causal --agent-id my_agent',
                'analyze visualize --type causal',
                'publication generate --type causal'
            ],
            'session_management': [
                'session list',
                'session details --session-id <id>',
                'session replay --session-id <id>',
                'session export --session-id <id>'
            ]
        }
        
        # Define help topics
        self._topics = {
            'getting_started': HelpTopic(
                name='getting_started',
                title='Getting Started with ESCAI CLI',
                content=self._get_getting_started_content(),
                category='basics',
                tags=['beginner', 'setup', 'quickstart'],
                related_commands=['config', 'monitor'],
                related_topics=['frameworks', 'configuration'],
                examples=['basic_monitoring'],
                prerequisites=[]
            ),
            'frameworks': HelpTopic(
                name='frameworks',
                title='Supported Agent Frameworks',
                content=self._get_frameworks_content(),
                category='integration',
                tags=['langchain', 'autogen', 'crewai', 'openai'],
                related_commands=['monitor', 'config'],
                related_topics=['getting_started', 'troubleshooting'],
                examples=['basic_monitoring', 'multi_agent_monitoring'],
                prerequisites=['getting_started']
            ),
            'monitoring': HelpTopic(
                name='monitoring',
                title='Agent Monitoring Guide',
                content=self._get_monitoring_content(),
                category='monitoring',
                tags=['real-time', 'epistemic', 'behavioral'],
                related_commands=['monitor'],
                related_topics=['analysis', 'session_management'],
                examples=['basic_monitoring'],
                prerequisites=['getting_started', 'frameworks']
            ),
            'analysis': HelpTopic(
                name='analysis',
                title='Behavioral Analysis Guide',
                content=self._get_analysis_content(),
                category='analysis',
                tags=['patterns', 'causal', 'statistical'],
                related_commands=['analyze'],
                related_topics=['monitoring', 'publication'],
                examples=['pattern_analysis', 'causal_analysis'],
                prerequisites=['monitoring']
            ),
            'troubleshooting': HelpTopic(
                name='troubleshooting',
                title='Troubleshooting Common Issues',
                content=self._get_troubleshooting_content(),
                category='support',
                tags=['errors', 'debugging', 'performance'],
                related_commands=['config', 'logs'],
                related_topics=['frameworks', 'configuration'],
                examples=[],
                prerequisites=[]
            )
        }
    
    def show_command_help(self, command: str, subcommand: Optional[str] = None) -> None:
        """
        Show comprehensive help for a command with cross-references.
        
        Args:
            command: Main command name
            subcommand: Optional subcommand name
        """
        full_command = f"{command}.{subcommand}" if subcommand else command
        
        # Get command reference
        cmd_ref = self._commands.get(command)
        if not cmd_ref:
            self.console.print(f"[red]No help available for command: {command}[/red]")
            return
        
        # Create help panel
        help_content = []
        
        # Command description
        help_content.append(f"[bold]{cmd_ref.description}[/bold]\n")
        
        # Usage examples
        if subcommand:
            help_content.append(f"[cyan]Usage:[/cyan] escai {command} {subcommand} [OPTIONS]\n")
        else:
            help_content.append(f"[cyan]Usage:[/cyan] escai {command} [SUBCOMMAND] [OPTIONS]\n")
        
        # Related commands
        if cmd_ref.related_commands:
            help_content.append("[cyan]Related Commands:[/cyan]")
            for related in cmd_ref.related_commands:
                related_ref = self._commands.get(related)
                if related_ref:
                    help_content.append(f"  • [yellow]escai {related}[/yellow] - {related_ref.description}")
            help_content.append("")
        
        # Common workflows
        workflows = [w for w in cmd_ref.common_workflows if w in self._workflows]
        if workflows:
            help_content.append("[cyan]Common Workflows:[/cyan]")
            for workflow in workflows:
                help_content.append(f"  • [green]{workflow.replace('_', ' ').title()}[/green]")
                help_content.append(f"    Use: [dim]escai help workflow {workflow}[/dim]")
            help_content.append("")
        
        # Cross-references
        related_topics = [topic for topic in self._topics.values() 
                         if command in topic.related_commands]
        if related_topics:
            help_content.append("[cyan]See Also:[/cyan]")
            for topic in related_topics:
                help_content.append(f"  • [blue]escai help {topic.name}[/blue] - {topic.title}")
        
        # Display help panel
        panel = Panel(
            "\n".join(help_content),
            title=f"Help: {command}" + (f" {subcommand}" if subcommand else ""),
            border_style="blue"
        )
        self.console.print(panel)
    
    def show_topic_help(self, topic: str) -> None:
        """
        Show help for a specific topic.
        
        Args:
            topic: Topic name to show help for
        """
        help_topic = self._topics.get(topic)
        if not help_topic:
            self.console.print(f"[red]No help available for topic: {topic}[/red]")
            self._suggest_topics(topic)
            return
        
        # Show topic content
        self.console.print(f"\n[bold cyan]{help_topic.title}[/bold cyan]\n")
        
        # Show prerequisites if any
        if help_topic.prerequisites:
            prereq_panel = Panel(
                "Read these topics first:\n" + 
                "\n".join([f"• [blue]escai help {p}[/blue]" for p in help_topic.prerequisites]),
                title="Prerequisites",
                border_style="yellow"
            )
            self.console.print(prereq_panel)
        
        # Show main content
        self.console.print(Markdown(help_topic.content))
        
        # Show related information
        if help_topic.related_commands or help_topic.related_topics:
            self._show_related_info(help_topic)
        
        # Show examples
        if help_topic.examples:
            self._show_topic_examples(help_topic)
    
    def show_workflow_help(self, workflow: str) -> None:
        """
        Show help for a specific workflow.
        
        Args:
            workflow: Workflow name to show help for
        """
        if workflow not in self._workflows:
            self.console.print(f"[red]No workflow found: {workflow}[/red]")
            self._suggest_workflows(workflow)
            return
        
        steps = self._workflows[workflow]
        
        # Create workflow table
        table = Table(title=f"Workflow: {workflow.replace('_', ' ').title()}")
        table.add_column("Step", style="cyan", width=6)
        table.add_column("Command", style="green")
        table.add_column("Description", style="dim")
        
        for i, step in enumerate(steps, 1):
            # Parse command and description
            if ' - ' in step:
                command, description = step.split(' - ', 1)
            else:
                command = step
                description = ""
            
            table.add_row(str(i), command, description)
        
        self.console.print(table)
        
        # Show workflow notes
        workflow_notes = self._get_workflow_notes(workflow)
        if workflow_notes:
            notes_panel = Panel(
                workflow_notes,
                title="Notes",
                border_style="yellow"
            )
            self.console.print(notes_panel)
    
    def show_quick_reference(self) -> None:
        """Show a quick reference guide for all commands."""
        # Create command reference table
        table = Table(title="ESCAI CLI Quick Reference")
        table.add_column("Command", style="cyan", width=15)
        table.add_column("Description", style="white")
        table.add_column("Category", style="dim", width=12)
        
        # Sort commands by category
        sorted_commands = sorted(self._commands.values(), key=lambda x: (x.category, x.command))
        
        for cmd_ref in sorted_commands:
            table.add_row(
                f"escai {cmd_ref.command}",
                cmd_ref.description,
                cmd_ref.category.replace('_', ' ').title()
            )
        
        self.console.print(table)
        
        # Show help usage
        help_panel = Panel(
            "[cyan]Getting Help:[/cyan]\n"
            "• [yellow]escai help[/yellow] - Show this quick reference\n"
            "• [yellow]escai help <command>[/yellow] - Show command help\n"
            "• [yellow]escai help <topic>[/yellow] - Show topic help\n"
            "• [yellow]escai help workflow <name>[/yellow] - Show workflow guide\n"
            "• [yellow]escai --help[/yellow] - Show CLI options\n\n"
            "[cyan]Available Topics:[/cyan]\n" +
            " • ".join([f"[blue]{topic}[/blue]" for topic in self._topics.keys()]),
            title="Help System",
            border_style="green"
        )
        self.console.print(help_panel)
    
    def search_help(self, query: str) -> None:
        """
        Search help content for a query.
        
        Args:
            query: Search query
        """
        query_lower = query.lower()
        results = []
        
        # Search commands
        for cmd_ref in self._commands.values():
            if (query_lower in cmd_ref.command.lower() or 
                query_lower in cmd_ref.description.lower() or
                any(query_lower in tag.lower() for tag in [cmd_ref.category])):
                results.append(('command', cmd_ref.command, cmd_ref.description))
        
        # Search topics
        for topic in self._topics.values():
            if (query_lower in topic.name.lower() or 
                query_lower in topic.title.lower() or
                query_lower in topic.content.lower() or
                any(query_lower in tag.lower() for tag in topic.tags)):
                results.append(('topic', topic.name, topic.title))
        
        # Search workflows
        for workflow_name, steps in self._workflows.items():
            if (query_lower in workflow_name.lower() or
                any(query_lower in step.lower() for step in steps)):
                results.append(('workflow', workflow_name, f"Workflow: {workflow_name.replace('_', ' ').title()}"))
        
        if not results:
            self.console.print(f"[yellow]No help found for: {query}[/yellow]")
            return
        
        # Display results
        table = Table(title=f"Help Search Results: '{query}'")
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Name", style="green", width=20)
        table.add_column("Description", style="white")
        table.add_column("Command", style="dim")
        
        for result_type, name, description in results:
            if result_type == 'command':
                command = f"escai help {name}"
            elif result_type == 'topic':
                command = f"escai help {name}"
            else:  # workflow
                command = f"escai help workflow {name}"
            
            table.add_row(result_type.title(), name, description, command)
        
        self.console.print(table)
    
    def _show_related_info(self, topic: HelpTopic) -> None:
        """Show related commands and topics for a help topic."""
        related_content = []
        
        if topic.related_commands:
            related_content.append("[cyan]Related Commands:[/cyan]")
            for cmd in topic.related_commands:
                cmd_ref = self._commands.get(cmd)
                if cmd_ref:
                    related_content.append(f"  • [yellow]escai {cmd}[/yellow] - {cmd_ref.description}")
        
        if topic.related_topics:
            if related_content:
                related_content.append("")
            related_content.append("[cyan]Related Topics:[/cyan]")
            for topic_name in topic.related_topics:
                related_topic = self._topics.get(topic_name)
                if related_topic:
                    related_content.append(f"  • [blue]escai help {topic_name}[/blue] - {related_topic.title}")
        
        if related_content:
            panel = Panel(
                "\n".join(related_content),
                title="Related Information",
                border_style="blue"
            )
            self.console.print(panel)
    
    def _show_topic_examples(self, topic: HelpTopic) -> None:
        """Show examples for a help topic."""
        if not topic.examples:
            return
        
        examples_content = []
        for example in topic.examples:
            if example in self._workflows:
                examples_content.append(f"[green]{example.replace('_', ' ').title()}:[/green]")
                for step in self._workflows[example]:
                    examples_content.append(f"  [dim]$[/dim] [yellow]{step}[/yellow]")
                examples_content.append("")
        
        if examples_content:
            panel = Panel(
                "\n".join(examples_content[:-1]),  # Remove last empty line
                title="Examples",
                border_style="green"
            )
            self.console.print(panel)
    
    def _suggest_topics(self, query: str) -> None:
        """Suggest similar topics for a failed query."""
        suggestions = []
        query_lower = query.lower()
        
        for topic_name in self._topics.keys():
            if query_lower in topic_name.lower():
                suggestions.append(topic_name)
        
        if suggestions:
            self.console.print(f"\n[yellow]Did you mean:[/yellow]")
            for suggestion in suggestions:
                self.console.print(f"  • [blue]escai help {suggestion}[/blue]")
        else:
            self.console.print(f"\n[dim]Use 'escai help' to see available topics[/dim]")
    
    def _suggest_workflows(self, query: str) -> None:
        """Suggest similar workflows for a failed query."""
        suggestions = []
        query_lower = query.lower()
        
        for workflow_name in self._workflows.keys():
            if query_lower in workflow_name.lower():
                suggestions.append(workflow_name)
        
        if suggestions:
            self.console.print(f"\n[yellow]Did you mean:[/yellow]")
            for suggestion in suggestions:
                self.console.print(f"  • [blue]escai help workflow {suggestion}[/blue]")
        else:
            available = ", ".join(self._workflows.keys())
            self.console.print(f"\n[dim]Available workflows: {available}[/dim]")
    
    def _get_workflow_notes(self, workflow: str) -> Optional[str]:
        """Get additional notes for a workflow."""
        notes = {
            'basic_monitoring': (
                "• Ensure your agent framework is installed before starting\n"
                "• Use --debug flag for verbose output during troubleshooting\n"
                "• Monitor status shows real-time updates every 2 seconds"
            ),
            'pattern_analysis': (
                "• Requires at least 10 minutes of monitoring data for meaningful patterns\n"
                "• Use --timeframe option to analyze specific time periods\n"
                "• Visualization works best with terminal width > 100 characters"
            ),
            'causal_analysis': (
                "• Causal analysis requires multiple agent interactions\n"
                "• Results improve with longer monitoring sessions\n"
                "• Use --confidence-threshold to filter weak relationships"
            )
        }
        return notes.get(workflow)
    
    def _get_getting_started_content(self) -> str:
        """Get getting started content."""
        return """
# Getting Started with ESCAI CLI

ESCAI (Epistemic State and Causal Analysis Intelligence) provides deep insights into how AI agents think, decide, and behave during task execution.

## Quick Setup

1. **Configure Database Connection**
   ```bash
   escai config setup
   ```

2. **Start Monitoring an Agent**
   ```bash
   escai monitor start --framework langchain --agent-id my_agent
   ```

3. **View Real-time Status**
   ```bash
   escai monitor status
   ```

4. **Analyze Patterns**
   ```bash
   escai analyze patterns --agent-id my_agent
   ```

## Key Concepts

- **Epistemic States**: What the agent believes, knows, and goals it pursues
- **Behavioral Patterns**: Recurring decision-making and execution strategies  
- **Causal Analysis**: Understanding cause-effect relationships in agent behavior
- **Performance Prediction**: Forecasting task outcomes and identifying potential failures

## Next Steps

- Learn about [supported frameworks](escai help frameworks)
- Explore [monitoring capabilities](escai help monitoring)
- Try [analysis features](escai help analysis)
"""
    
    def _get_frameworks_content(self) -> str:
        """Get frameworks content."""
        return """
# Supported Agent Frameworks

ESCAI supports monitoring multiple agent frameworks with deep integration for each.

## LangChain
- **Installation**: `pip install langchain`
- **Features**: Chain execution tracking, tool usage monitoring, memory analysis
- **Best For**: Complex reasoning chains, tool-using agents

## AutoGen
- **Installation**: `pip install autogen`  
- **Features**: Multi-agent conversation tracking, role-based analysis
- **Best For**: Multi-agent systems, collaborative problem solving

## CrewAI
- **Installation**: `pip install crewai`
- **Features**: Task delegation monitoring, crew performance analysis
- **Best For**: Structured team workflows, hierarchical agent systems

## OpenAI Assistants
- **Installation**: `pip install openai`
- **Features**: Assistant API monitoring, tool usage tracking, reasoning analysis
- **Best For**: OpenAI-powered assistants, function calling workflows

## Framework Selection

Choose based on your agent architecture:
- Single reasoning agent → LangChain
- Multiple collaborating agents → AutoGen  
- Structured team workflows → CrewAI
- OpenAI Assistant API → OpenAI

## Integration Status

Check framework availability:
```bash
escai config test --framework <framework_name>
```
"""
    
    def _get_monitoring_content(self) -> str:
        """Get monitoring content."""
        return """
# Agent Monitoring Guide

Real-time monitoring provides insights into agent cognition as it happens.

## Starting Monitoring

```bash
# Basic monitoring
escai monitor start --framework langchain --agent-id my_agent

# With custom configuration
escai monitor start --framework langchain --agent-id my_agent \\
  --capture-epistemic --capture-behavioral --capture-performance
```

## Monitoring Features

### Real-time Display
- Live epistemic state updates
- Behavioral pattern detection
- Performance metrics tracking
- Error and anomaly alerts

### Data Capture
- **Epistemic States**: Beliefs, knowledge, goals, uncertainty
- **Behavioral Patterns**: Decision strategies, execution patterns
- **Performance Metrics**: Response times, success rates, resource usage

## Monitoring Commands

- `escai monitor status` - View active sessions
- `escai monitor live` - Real-time dashboard
- `escai monitor logs` - View detailed logs
- `escai monitor stop` - Stop monitoring session

## Best Practices

1. **Start monitoring before agent execution**
2. **Use descriptive agent IDs for organization**
3. **Monitor for sufficient duration (10+ minutes) for pattern detection**
4. **Use --debug flag for troubleshooting**
5. **Stop monitoring cleanly to preserve data**
"""
    
    def _get_analysis_content(self) -> str:
        """Get analysis content."""
        return """
# Behavioral Analysis Guide

Analyze captured monitoring data to understand agent behavior and performance.

## Pattern Analysis

Identify recurring behavioral patterns:

```bash
# Analyze patterns for specific agent
escai analyze patterns --agent-id my_agent

# Analyze patterns across timeframe
escai analyze patterns --timeframe 24h --confidence-threshold 0.8
```

### Pattern Types
- **Decision Patterns**: How agents make choices
- **Execution Patterns**: How agents carry out tasks
- **Error Patterns**: Common failure modes
- **Performance Patterns**: Resource usage trends

## Causal Analysis

Understand cause-effect relationships:

```bash
# Analyze causal relationships
escai analyze causal --agent-id my_agent

# Focus on specific outcomes
escai analyze causal --outcome success --confidence-threshold 0.9
```

### Causal Features
- **Event Causality**: What events lead to outcomes
- **State Causality**: How epistemic states influence behavior
- **Performance Causality**: What affects agent performance

## Statistical Analysis

Generate statistical insights:

```bash
# Statistical summary
escai analyze stats --agent-id my_agent

# Comparative analysis
escai analyze compare --agents agent1,agent2
```

## Visualization

Create visual representations:

```bash
# Pattern heatmap
escai analyze visualize --type patterns --format heatmap

# Causal network
escai analyze visualize --type causal --format network

# Performance trends
escai analyze visualize --type performance --format timeline
```

## Export Results

Save analysis for further use:

```bash
# Export to JSON
escai analyze export --format json --output analysis.json

# Export to CSV for spreadsheet analysis
escai analyze export --format csv --output patterns.csv
```
"""
    
    def _get_troubleshooting_content(self) -> str:
        """Get troubleshooting content."""
        return """
# Troubleshooting Common Issues

## Framework Integration Issues

### Framework Not Found
```
Error: Framework 'langchain' is not available
```
**Solution**: Install the framework
```bash
pip install langchain  # or autogen, crewai, openai
```

### Import Errors
```
Error: Cannot import framework modules
```
**Solutions**:
1. Check virtual environment activation
2. Verify framework installation: `pip list | grep <framework>`
3. Try reinstalling: `pip uninstall <framework> && pip install <framework>`

## Database Connection Issues

### Connection Failed
```
Error: Cannot connect to database
```
**Solutions**:
1. Run database setup: `escai config setup`
2. Check database status: `escai config test`
3. Verify connection settings: `escai config show`

### Permission Errors
```
Error: Permission denied accessing database
```
**Solutions**:
1. Check file permissions on database directory
2. Run with appropriate user permissions
3. Reconfigure database: `escai config setup --reset`

## Monitoring Issues

### No Data Captured
```
Warning: No monitoring data captured
```
**Solutions**:
1. Ensure agent is actually running during monitoring
2. Check framework integration: `escai config test --framework <name>`
3. Enable debug mode: `escai --debug monitor start ...`

### High Memory Usage
```
Warning: High memory usage detected
```
**Solutions**:
1. Reduce monitoring frequency: `--max-events-per-second 10`
2. Limit buffer size: `--buffer-size 1000`
3. Stop unnecessary monitoring sessions

## Performance Issues

### Slow CLI Response
**Solutions**:
1. Clear session cache: `escai session cleanup`
2. Optimize database: `escai config optimize`
3. Check system resources: `escai logs system`

### Analysis Timeouts
**Solutions**:
1. Reduce analysis timeframe: `--timeframe 1h`
2. Increase timeout: `--timeout 300`
3. Use sampling: `--sample-rate 0.1`

## Getting Help

1. **Enable Debug Mode**: `escai --debug <command>`
2. **Check Logs**: `escai logs show --level error`
3. **System Status**: `escai config check`
4. **Framework Status**: `escai config test --all-frameworks`

## Reporting Issues

When reporting issues, include:
1. ESCAI version: `escai --version`
2. Framework versions: `pip list | grep -E "(langchain|autogen|crewai|openai)"`
3. Error logs: `escai logs show --recent`
4. System info: `escai config system-info`
"""


# Global help system instance
_help_system: Optional[HelpSystem] = None


def get_help_system() -> HelpSystem:
    """Get the global help system instance."""
    global _help_system
    if _help_system is None:
        _help_system = HelpSystem()
    return _help_system