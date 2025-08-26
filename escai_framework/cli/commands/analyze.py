"""
Analysis commands for ESCAI CLI
"""

import click
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from ..utils.console import get_console
from ..utils.formatters import (
    format_behavioral_patterns,
    format_causal_tree,
    format_predictions,
    format_ascii_chart
)

console = get_console()

@click.group(name='analyze')
def analyze_group():
    """Analysis and exploration commands"""
    pass

@analyze_group.command()
@click.option('--agent-id', help='Filter by specific agent ID')
@click.option('--timeframe', default='24h', help='Analysis timeframe (1h, 24h, 7d, 30d)')
@click.option('--min-frequency', default=5, help='Minimum pattern frequency')
@click.option('--interactive', is_flag=True, help='Interactive pattern exploration')
def patterns(agent_id: str, timeframe: str, min_frequency: int, interactive: bool):
    """Analyze behavioral patterns"""
    
    console.print(f"[info]Analyzing behavioral patterns (timeframe: {timeframe})[/info]")
    
    # Mock pattern data
    patterns_data = [
        {
            'pattern_name': 'Sequential Data Processing',
            'frequency': 45,
            'success_rate': 0.89,
            'average_duration': '2.3s',
            'statistical_significance': 0.95
        },
        {
            'pattern_name': 'Error Recovery Loop',
            'frequency': 12,
            'success_rate': 0.67,
            'average_duration': '5.1s',
            'statistical_significance': 0.78
        },
        {
            'pattern_name': 'Optimization Iteration',
            'frequency': 28,
            'success_rate': 0.92,
            'average_duration': '1.8s',
            'statistical_significance': 0.88
        }
    ]
    
    # Filter by frequency
    filtered_patterns = [p for p in patterns_data if p['frequency'] >= min_frequency]
    
    table = format_behavioral_patterns(filtered_patterns)
    console.print(table)
    
    if interactive:
        console.print("\n[cyan]Interactive Pattern Explorer[/cyan]")
        
        while True:
            pattern_names = [p['pattern_name'] for p in filtered_patterns]
            pattern_names.append("Exit")
            
            choice = Prompt.ask(
                "Select a pattern to explore",
                choices=pattern_names,
                default="Exit"
            )
            
            if choice == "Exit":
                break
                
            selected_pattern = next(p for p in filtered_patterns if p['pattern_name'] == choice)
            
            # Show detailed pattern analysis
            detail_panel = Panel(
                f"[bold]Pattern:[/bold] {selected_pattern['pattern_name']}\n"
                f"[bold]Frequency:[/bold] {selected_pattern['frequency']} occurrences\n"
                f"[bold]Success Rate:[/bold] {selected_pattern['success_rate']:.1%}\n"
                f"[bold]Average Duration:[/bold] {selected_pattern['average_duration']}\n"
                f"[bold]Statistical Significance:[/bold] {selected_pattern['statistical_significance']:.2f}\n\n"
                f"[bold]Common Triggers:[/bold]\n"
                f"• Data validation requirements\n"
                f"• Complex query processing\n"
                f"• Multi-step transformations\n\n"
                f"[bold]Failure Modes:[/bold]\n"
                f"• Timeout on large datasets\n"
                f"• Memory constraints\n"
                f"• Invalid input formats",
                title=f"Pattern Details: {choice}",
                border_style="blue"
            )
            console.print(detail_panel)

@analyze_group.command()
@click.option('--agent-id', help='Filter by specific agent ID')
@click.option('--min-strength', default=0.5, help='Minimum causal strength threshold')
@click.option('--interactive', is_flag=True, help='Interactive causal exploration')
def causal(agent_id: str, min_strength: float, interactive: bool):
    """Explore causal relationships"""
    
    console.print("[info]Analyzing causal relationships...[/info]")
    
    # Mock causal relationship data
    relationships = [
        {
            'cause_event': 'Data Validation Error',
            'effect_event': 'Retry Mechanism Triggered',
            'strength': 0.87,
            'confidence': 0.92,
            'delay_ms': 150
        },
        {
            'cause_event': 'Large Dataset Detected',
            'effect_event': 'Batch Processing Initiated',
            'strength': 0.94,
            'confidence': 0.89,
            'delay_ms': 50
        },
        {
            'cause_event': 'Memory Usage High',
            'effect_event': 'Garbage Collection Triggered',
            'strength': 0.78,
            'confidence': 0.85,
            'delay_ms': 200
        },
        {
            'cause_event': 'API Rate Limit Hit',
            'effect_event': 'Exponential Backoff Started',
            'strength': 0.96,
            'confidence': 0.98,
            'delay_ms': 100
        }
    ]
    
    # Filter by strength
    filtered_relationships = [r for r in relationships if r['strength'] >= min_strength]
    
    tree = format_causal_tree(filtered_relationships)
    console.print(tree)
    
    if interactive:
        console.print("\n[cyan]Interactive Causal Explorer[/cyan]")
        
        while True:
            causes = list(set(r['cause_event'] for r in filtered_relationships))
            causes.append("Exit")
            
            choice = Prompt.ask(
                "Select a cause event to explore",
                choices=causes,
                default="Exit"
            )
            
            if choice == "Exit":
                break
            
            # Show effects for selected cause
            effects = [r for r in filtered_relationships if r['cause_event'] == choice]
            
            effects_panel = Panel(
                "\n".join([
                    f"[bold]Effect:[/bold] {e['effect_event']}\n"
                    f"  Strength: {e['strength']:.2f} | "
                    f"Confidence: {e['confidence']:.2f} | "
                    f"Delay: {e['delay_ms']}ms\n"
                    for e in effects
                ]),
                title=f"Effects of: {choice}",
                border_style="green"
            )
            console.print(effects_panel)

@analyze_group.command()
@click.option('--agent-id', help='Filter by specific agent ID')
@click.option('--horizon', default='1h', help='Prediction horizon (15m, 1h, 4h, 24h)')
def predictions(agent_id: str, horizon: str):
    """View performance predictions"""
    
    console.print(f"[info]Generating predictions (horizon: {horizon})[/info]")
    
    # Mock prediction data
    predictions_data = [
        {
            'predicted_outcome': 'success',
            'confidence': 0.87,
            'risk_factors': ['High memory usage', 'Complex query'],
            'trend': 'improving'
        },
        {
            'predicted_outcome': 'failure',
            'confidence': 0.72,
            'risk_factors': ['API rate limits', 'Network latency', 'Data size'],
            'trend': 'declining'
        }
    ]
    
    predictions_panel = format_predictions(predictions_data)
    console.print(predictions_panel)
    
    # Show trend chart
    console.print("\n[bold]Success Rate Trend:[/bold]")
    trend_data = [0.85, 0.87, 0.82, 0.89, 0.91, 0.88, 0.92, 0.87, 0.85, 0.89]
    chart = format_ascii_chart(trend_data, "Success Rate Over Time", width=40, height=8)
    console.print(f"[muted]{chart}[/muted]")

@analyze_group.command()
@click.option('--agent-id', help='Filter by specific agent ID')
@click.option('--event-type', help='Filter by event type')
@click.option('--limit', default=10, help='Number of recent events to show')
def events(agent_id: str, event_type: str, limit: int):
    """View recent agent events"""
    
    console.print(f"[info]Showing {limit} recent events[/info]")
    
    # Mock event data
    events_data = [
        {'timestamp': '2024-01-15 14:30:25', 'type': 'decision', 'description': 'Selected optimization strategy A'},
        {'timestamp': '2024-01-15 14:30:22', 'type': 'belief_update', 'description': 'Updated confidence in data quality'},
        {'timestamp': '2024-01-15 14:30:18', 'type': 'goal_progress', 'description': 'Completed data validation phase'},
        {'timestamp': '2024-01-15 14:30:15', 'type': 'error', 'description': 'Temporary API connection failure'},
        {'timestamp': '2024-01-15 14:30:12', 'type': 'pattern_match', 'description': 'Detected sequential processing pattern'},
    ]
    
    from rich.table import Table
    table = Table(title="Recent Agent Events", show_header=True, header_style="bold magenta")
    
    table.add_column("Timestamp", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Description", style="white")
    
    for event in events_data[:limit]:
        event_type_style = {
            'decision': 'green',
            'belief_update': 'yellow', 
            'goal_progress': 'blue',
            'error': 'red',
            'pattern_match': 'magenta'
        }.get(event['type'], 'white')
        
        table.add_row(
            event['timestamp'],
            f"[{event_type_style}]{event['type']}[/{event_type_style}]",
            event['description']
        )
    
    console.print(table)