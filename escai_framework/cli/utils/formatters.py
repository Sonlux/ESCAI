"""
Formatting utilities for CLI output
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from .console import get_console

console = get_console()

def format_agent_status_table(agents: List[Dict[str, Any]]) -> Table:
    """Format agent status as a rich table"""
    table = Table(title="Agent Status", show_header=True, header_style="bold magenta")
    
    table.add_column("Agent ID", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Framework", style="blue")
    table.add_column("Uptime", style="green")
    table.add_column("Events", justify="right", style="yellow")
    table.add_column("Last Activity", style="muted")
    
    for agent in agents:
        status_style = "green" if agent.get('status') == 'active' else "red"
        status_icon = "🟢" if agent.get('status') == 'active' else "🔴"
        
        table.add_row(
            agent.get('id', 'N/A'),
            f"{status_icon} {agent.get('status', 'unknown')}",
            agent.get('framework', 'N/A'),
            agent.get('uptime', 'N/A'),
            str(agent.get('event_count', 0)),
            agent.get('last_activity', 'N/A')
        )
    
    return table

def format_epistemic_state(state: Dict[str, Any]) -> Panel:
    """Format epistemic state as a rich panel"""
    content = []
    
    # Beliefs section
    beliefs = state.get('beliefs', [])
    if beliefs:
        content.append("[bold cyan]Beliefs:[/bold cyan]")
        for belief in beliefs[:5]:  # Show top 5
            confidence = belief.get('confidence', 0)
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            content.append(f"  • {belief.get('content', 'N/A')} [{confidence_bar}] {confidence:.2f}")
    
    # Knowledge section
    knowledge = state.get('knowledge', {})
    if knowledge:
        content.append("\n[bold green]Knowledge:[/bold green]")
        content.append(f"  • Facts: {knowledge.get('fact_count', 0)}")
        content.append(f"  • Concepts: {knowledge.get('concept_count', 0)}")
        content.append(f"  • Relationships: {knowledge.get('relationship_count', 0)}")
    
    # Goals section
    goals = state.get('goals', [])
    if goals:
        content.append("\n[bold yellow]Goals:[/bold yellow]")
        for goal in goals[:3]:  # Show top 3
            progress = goal.get('progress', 0)
            progress_bar = "█" * int(progress * 10) + "░" * (10 - int(progress * 10))
            content.append(f"  • {goal.get('description', 'N/A')} [{progress_bar}] {progress:.1%}")
    
    # Uncertainty
    uncertainty = state.get('uncertainty_score', 0)
    uncertainty_color = "red" if uncertainty > 0.7 else "yellow" if uncertainty > 0.4 else "green"
    content.append(f"\n[bold]Uncertainty:[/bold] [{uncertainty_color}]{uncertainty:.2f}[/{uncertainty_color}]")
    
    return Panel(
        "\n".join(content) if content else "No epistemic state data available",
        title=f"Epistemic State - {state.get('agent_id', 'Unknown')}",
        border_style="blue"
    )

def format_behavioral_patterns(patterns: List[Dict[str, Any]]) -> Table:
    """Format behavioral patterns as a table"""
    table = Table(title="Behavioral Patterns", show_header=True, header_style="bold magenta")
    
    table.add_column("Pattern", style="cyan")
    table.add_column("Frequency", justify="right", style="yellow")
    table.add_column("Success Rate", justify="center")
    table.add_column("Avg Duration", justify="right", style="green")
    table.add_column("Significance", justify="center", style="blue")
    
    for pattern in patterns:
        success_rate = pattern.get('success_rate', 0)
        success_color = "green" if success_rate > 0.8 else "yellow" if success_rate > 0.5 else "red"
        success_icon = "✅" if success_rate > 0.8 else "⚠️" if success_rate > 0.5 else "❌"
        
        significance = pattern.get('statistical_significance', 0)
        sig_stars = "★" * min(5, int(significance * 5))
        
        table.add_row(
            pattern.get('pattern_name', 'Unknown'),
            str(pattern.get('frequency', 0)),
            f"{success_icon} [{success_color}]{success_rate:.1%}[/{success_color}]",
            pattern.get('average_duration', 'N/A'),
            sig_stars
        )
    
    return table

def format_causal_tree(relationships: List[Dict[str, Any]]) -> Tree:
    """Format causal relationships as a tree structure"""
    tree = Tree("🔗 [bold cyan]Causal Relationships[/bold cyan]")
    
    # Group by cause events
    cause_groups = {}
    for rel in relationships:
        cause = rel.get('cause_event', 'Unknown')
        if cause not in cause_groups:
            cause_groups[cause] = []
        cause_groups[cause].append(rel)
    
    for cause, effects in cause_groups.items():
        cause_branch = tree.add(f"[yellow]📤 {cause}[/yellow]")
        
        for effect_rel in effects:
            effect = effect_rel.get('effect_event', 'Unknown')
            strength = effect_rel.get('strength', 0)
            confidence = effect_rel.get('confidence', 0)
            
            strength_bar = "█" * int(strength * 5) + "░" * (5 - int(strength * 5))
            confidence_color = "green" if confidence > 0.8 else "yellow" if confidence > 0.5 else "red"
            
            cause_branch.add(
                f"[blue]📥 {effect}[/blue] "
                f"[{confidence_color}]({confidence:.2f})[/{confidence_color}] "
                f"[{strength_bar}]"
            )
    
    return tree

def format_predictions(predictions: List[Dict[str, Any]]) -> Panel:
    """Format performance predictions"""
    content = []
    
    for pred in predictions:
        outcome = pred.get('predicted_outcome', 'unknown')
        confidence = pred.get('confidence', 0)
        
        # Outcome icon and color
        if outcome == 'success':
            icon, color = "✅", "green"
        elif outcome == 'failure':
            icon, color = "❌", "red"
        else:
            icon, color = "❓", "yellow"
        
        # Confidence visualization
        conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        
        # Risk factors
        risk_factors = pred.get('risk_factors', [])
        risk_text = ", ".join(risk_factors[:3]) if risk_factors else "None identified"
        
        content.append(f"{icon} [{color}]{outcome.title()}[/{color}] [{conf_bar}] {confidence:.1%}")
        content.append(f"   Risk Factors: {risk_text}")
        
        # Trend arrow
        trend = pred.get('trend', 'stable')
        trend_icons = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}
        content.append(f"   Trend: {trend_icons.get(trend, '➡️')} {trend}")
        content.append("")
    
    return Panel(
        "\n".join(content) if content else "No predictions available",
        title="Performance Predictions",
        border_style="magenta"
    )

def create_progress_bar(description: str) -> Progress:
    """Create a styled progress bar"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    )

def format_ascii_chart(data: List[float], title: str, width: int = 50, height: int = 10) -> str:
    """Create a simple ASCII chart"""
    if not data:
        return f"{title}: No data available"
    
    # Normalize data to chart height
    max_val = max(data)
    min_val = min(data)
    range_val = max_val - min_val if max_val != min_val else 1
    
    normalized = [(val - min_val) / range_val * (height - 1) for val in data]
    
    # Create chart
    chart_lines = []
    for row in range(height - 1, -1, -1):
        line = ""
        for i, val in enumerate(normalized):
            if i < width and val >= row:
                line += "█"
            else:
                line += " " if i < width else ""
        chart_lines.append(line)
    
    # Add title and axis
    result = f"{title}\n"
    result += "┌" + "─" * width + "┐\n"
    for line in chart_lines:
        result += "│" + line.ljust(width) + "│\n"
    result += "└" + "─" * width + "┘\n"
    result += f"Min: {min_val:.2f}, Max: {max_val:.2f}"
    
    return result