"""
Analysis commands for ESCAI CLI
"""

import click
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

from ..utils.console import get_console
from ..utils.formatters import (
    format_behavioral_patterns,
    format_causal_tree,
    format_predictions,
    format_ascii_chart
)
from ..utils.ascii_viz import (
    ASCIIBarChart, ASCIILineChart, ASCIIHistogram, ASCIIScatterPlot,
    ASCIIHeatmap, ASCIISparkline, ChartConfig,
    create_epistemic_state_chart, create_pattern_frequency_heatmap,
    create_causal_strength_scatter
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
    
    # Show trend chart using new ASCII visualization
    console.print("\n[bold]Success Rate Trend:[/bold]")
    trend_data = [0.85, 0.87, 0.82, 0.89, 0.91, 0.88, 0.92, 0.87, 0.85, 0.89]
    
    config = ChartConfig(width=50, height=8, title="Success Rate Over Time")
    line_chart = ASCIILineChart(config)
    chart = line_chart.create(trend_data)
    console.print(f"[muted]{chart}[/muted]")
    
    # Show sparkline for compact view
    sparkline = ASCIISparkline()
    spark = sparkline.create(trend_data, 30)
    console.print(f"\nCompact trend: {spark} (Current: {trend_data[-1]:.2f})")

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


@analyze_group.command()
@click.option('--agent-id', help='Filter by specific agent ID')
@click.option('--chart-type', type=click.Choice(['bar', 'line', 'histogram', 'scatter', 'heatmap']), 
              default='bar', help='Type of visualization')
@click.option('--metric', default='confidence', help='Metric to visualize')
def visualize(agent_id: str, chart_type: str, metric: str):
    """Create advanced ASCII visualizations of agent data"""
    
    console.print(f"[info]Creating {chart_type} chart for {metric}[/info]")
    
    # Mock data based on metric
    if metric == 'confidence':
        data = [0.85, 0.92, 0.78, 0.89, 0.95, 0.82, 0.91, 0.87, 0.93, 0.86]
        labels = [f"Belief {i+1}" for i in range(len(data))]
        title = "Belief Confidence Levels"
    elif metric == 'performance':
        data = [1.2, 2.1, 1.8, 2.5, 1.9, 2.3, 1.7, 2.0, 1.6, 2.2]
        labels = [f"Task {i+1}" for i in range(len(data))]
        title = "Task Performance (seconds)"
    else:
        data = [10, 25, 18, 32, 28, 15, 22, 19, 26, 21]
        labels = [f"Item {i+1}" for i in range(len(data))]
        title = f"{metric.title()} Distribution"
    
    config = ChartConfig(width=60, height=12, title=title)
    
    if chart_type == 'bar':
        chart = ASCIIBarChart(config)
        result = chart.create(data, labels)
    elif chart_type == 'line':
        chart = ASCIILineChart(config)
        result = chart.create(data)
    elif chart_type == 'histogram':
        chart = ASCIIHistogram(config)
        result = chart.create(data)
    elif chart_type == 'scatter':
        # Create scatter plot with data vs index
        x_data = list(range(len(data)))
        chart = ASCIIScatterPlot(config)
        result = chart.create(x_data, data)
    elif chart_type == 'heatmap':
        # Create 2D heatmap from 1D data
        rows = 3
        cols = len(data) // rows + (1 if len(data) % rows else 0)
        heatmap_data = []
        for i in range(rows):
            row = data[i*cols:(i+1)*cols]
            if len(row) < cols:
                row.extend([0] * (cols - len(row)))
            heatmap_data.append(row)
        
        chart = ASCIIHeatmap(config)
        result = chart.create(heatmap_data)
    
    console.print(result)


@analyze_group.command()
@click.option('--agent-id', help='Filter by specific agent ID')
def epistemic(agent_id: str):
    """Visualize epistemic state evolution"""
    
    console.print("[info]Analyzing epistemic state data...[/info]")
    
    # Mock epistemic state data
    epistemic_data = {
        'beliefs': [
            {'content': 'Data quality is high', 'confidence': 0.89},
            {'content': 'Processing will succeed', 'confidence': 0.76},
            {'content': 'User intent is clear', 'confidence': 0.92},
            {'content': 'Resources are sufficient', 'confidence': 0.68},
            {'content': 'Timeline is achievable', 'confidence': 0.84}
        ],
        'uncertainty_history': [0.15, 0.22, 0.18, 0.25, 0.19, 0.16, 0.21, 0.17, 0.14, 0.20],
        'knowledge': {
            'fact_count': 127,
            'concept_count': 45,
            'relationship_count': 89
        },
        'goals': [
            {'description': 'Complete data processing', 'progress': 0.75},
            {'description': 'Validate results', 'progress': 0.45},
            {'description': 'Generate report', 'progress': 0.20}
        ]
    }
    
    # Create comprehensive epistemic visualization
    result = create_epistemic_state_chart(epistemic_data)
    console.print(result)
    
    # Additional detailed visualizations
    console.print("\n[bold cyan]Belief Confidence Distribution:[/bold cyan]")
    confidences = [b['confidence'] for b in epistemic_data['beliefs']]
    config = ChartConfig(width=50, height=8, title="Belief Confidence Histogram")
    hist_chart = ASCIIHistogram(config)
    hist_result = hist_chart.create(confidences, bins=5)
    console.print(hist_result)
    
    console.print("\n[bold cyan]Goal Progress Overview:[/bold cyan]")
    goal_progress = [g['progress'] for g in epistemic_data['goals']]
    goal_labels = [g['description'][:20] + '...' if len(g['description']) > 20 
                   else g['description'] for g in epistemic_data['goals']]
    config = ChartConfig(width=60, height=6, title="Goal Progress")
    bar_chart = ASCIIBarChart(config)
    bar_result = bar_chart.create(goal_progress, goal_labels)
    console.print(bar_result)


@analyze_group.command()
@click.option('--timeframe', default='24h', help='Analysis timeframe')
def heatmap(timeframe: str):
    """Generate pattern frequency heatmap"""
    
    console.print(f"[info]Generating pattern frequency heatmap ({timeframe})[/info]")
    
    # Mock pattern frequency data
    pattern_data = [
        {'pattern_name': 'Data Validation', 'time_period': 'Morning', 'frequency': 15},
        {'pattern_name': 'Data Validation', 'time_period': 'Afternoon', 'frequency': 22},
        {'pattern_name': 'Data Validation', 'time_period': 'Evening', 'frequency': 8},
        {'pattern_name': 'Error Recovery', 'time_period': 'Morning', 'frequency': 5},
        {'pattern_name': 'Error Recovery', 'time_period': 'Afternoon', 'frequency': 12},
        {'pattern_name': 'Error Recovery', 'time_period': 'Evening', 'frequency': 18},
        {'pattern_name': 'Optimization', 'time_period': 'Morning', 'frequency': 28},
        {'pattern_name': 'Optimization', 'time_period': 'Afternoon', 'frequency': 35},
        {'pattern_name': 'Optimization', 'time_period': 'Evening', 'frequency': 20},
        {'pattern_name': 'Decision Making', 'time_period': 'Morning', 'frequency': 42},
        {'pattern_name': 'Decision Making', 'time_period': 'Afternoon', 'frequency': 38},
        {'pattern_name': 'Decision Making', 'time_period': 'Evening', 'frequency': 25}
    ]
    
    result = create_pattern_frequency_heatmap(pattern_data)
    console.print(result)


@analyze_group.command()
@click.option('--min-strength', default=0.3, help='Minimum causal strength')
def causal_scatter(min_strength: float):
    """Visualize causal relationships as scatter plot"""
    
    console.print("[info]Creating causal relationship scatter plot...[/info]")
    
    # Mock causal relationship data
    causal_data = [
        {'strength': 0.87, 'confidence': 0.92, 'cause': 'Data Error', 'effect': 'Retry'},
        {'strength': 0.94, 'confidence': 0.89, 'cause': 'Large Dataset', 'effect': 'Batch Process'},
        {'strength': 0.78, 'confidence': 0.85, 'cause': 'High Memory', 'effect': 'GC Trigger'},
        {'strength': 0.96, 'confidence': 0.98, 'cause': 'Rate Limit', 'effect': 'Backoff'},
        {'strength': 0.65, 'confidence': 0.72, 'cause': 'Network Lag', 'effect': 'Timeout'},
        {'strength': 0.82, 'confidence': 0.88, 'cause': 'Invalid Input', 'effect': 'Validation'},
        {'strength': 0.45, 'confidence': 0.55, 'cause': 'Low Battery', 'effect': 'Power Save'},
        {'strength': 0.91, 'confidence': 0.94, 'cause': 'Cache Miss', 'effect': 'DB Query'}
    ]
    
    # Filter by minimum strength
    filtered_data = [c for c in causal_data if c['strength'] >= min_strength]
    
    result = create_causal_strength_scatter(filtered_data)
    console.print(result)
    
    # Show detailed table
    console.print("\n[bold]Causal Relationship Details:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Cause", style="cyan")
    table.add_column("Effect", style="blue")
    table.add_column("Strength", justify="center", style="green")
    table.add_column("Confidence", justify="center", style="yellow")
    
    for rel in filtered_data:
        table.add_row(
            rel['cause'],
            rel['effect'],
            f"{rel['strength']:.2f}",
            f"{rel['confidence']:.2f}"
        )
    
    console.print(table)


@analyze_group.command()
@click.option('--agent-id', help='Filter by specific agent ID')
@click.option('--max-depth', default=5, help='Maximum tree depth to display')
def tree(agent_id: str, max_depth: int):
    """Display causal relationships as expandable tree"""
    
    console.print("[info]Building causal relationship tree...[/info]")
    
    from ..utils.ascii_viz import ASCIITreeView
    
    # Mock hierarchical causal data
    tree_data = {
        'name': 'Agent Decision Process',
        'value': '100%',
        'children': [
            {
                'name': 'Data Processing Phase',
                'value': '45%',
                'children': [
                    {
                        'name': 'Input Validation',
                        'value': '15%',
                        'children': [
                            {'name': 'Schema Check', 'value': '5%'},
                            {'name': 'Type Validation', 'value': '6%'},
                            {'name': 'Range Validation', 'value': '4%'}
                        ]
                    },
                    {
                        'name': 'Data Transformation',
                        'value': '20%',
                        'children': [
                            {'name': 'Normalization', 'value': '8%'},
                            {'name': 'Encoding', 'value': '7%'},
                            {'name': 'Aggregation', 'value': '5%'}
                        ]
                    },
                    {
                        'name': 'Quality Assessment',
                        'value': '10%',
                        'children': [
                            {'name': 'Completeness Check', 'value': '4%'},
                            {'name': 'Consistency Check', 'value': '3%'},
                            {'name': 'Accuracy Check', 'value': '3%'}
                        ]
                    }
                ]
            },
            {
                'name': 'Decision Making Phase',
                'value': '35%',
                'children': [
                    {
                        'name': 'Strategy Selection',
                        'value': '15%',
                        'children': [
                            {'name': 'Performance Analysis', 'value': '6%'},
                            {'name': 'Resource Assessment', 'value': '5%'},
                            {'name': 'Risk Evaluation', 'value': '4%'}
                        ]
                    },
                    {
                        'name': 'Parameter Optimization',
                        'value': '12%',
                        'children': [
                            {'name': 'Hyperparameter Tuning', 'value': '7%'},
                            {'name': 'Threshold Adjustment', 'value': '5%'}
                        ]
                    },
                    {
                        'name': 'Confidence Calculation',
                        'value': '8%',
                        'children': [
                            {'name': 'Uncertainty Quantification', 'value': '4%'},
                            {'name': 'Evidence Weighting', 'value': '4%'}
                        ]
                    }
                ]
            },
            {
                'name': 'Execution Phase',
                'value': '20%',
                'children': [
                    {
                        'name': 'Action Planning',
                        'value': '8%',
                        'children': [
                            {'name': 'Step Sequencing', 'value': '4%'},
                            {'name': 'Resource Allocation', 'value': '4%'}
                        ]
                    },
                    {
                        'name': 'Monitoring Setup',
                        'value': '7%',
                        'children': [
                            {'name': 'Progress Tracking', 'value': '4%'},
                            {'name': 'Error Detection', 'value': '3%'}
                        ]
                    },
                    {
                        'name': 'Result Validation',
                        'value': '5%',
                        'children': [
                            {'name': 'Output Verification', 'value': '3%'},
                            {'name': 'Quality Metrics', 'value': '2%'}
                        ]
                    }
                ]
            }
        ]
    }
    
    tree_view = ASCIITreeView()
    result = tree_view.create(tree_data, max_depth)
    console.print(result)
    
    # Show tree statistics
    console.print(f"\n[bold]Tree Statistics:[/bold]")
    console.print(f"Max Depth: {max_depth}")
    console.print(f"Total Nodes: {_count_tree_nodes(tree_data)}")
    console.print(f"Leaf Nodes: {_count_leaf_nodes(tree_data)}")


def _count_tree_nodes(node: dict) -> int:
    """Count total nodes in tree"""
    count = 1
    for child in node.get('children', []):
        count += _count_tree_nodes(child)
    return count


def _count_leaf_nodes(node: dict) -> int:
    """Count leaf nodes in tree"""
    children = node.get('children', [])
    if not children:
        return 1
    
    count = 0
    for child in children:
        count += _count_leaf_nodes(child)
    return count


@analyze_group.command()
@click.option('--metric', default='all', help='Metric to show progress for')
def progress(metric: str):
    """Display progress bars for various metrics"""
    
    console.print("[info]Displaying progress metrics...[/info]")
    
    from ..utils.ascii_viz import ASCIIProgressBar
    from datetime import timedelta
    
    progress_bar = ASCIIProgressBar(width=50)
    
    # Mock progress data
    metrics = {
        'data_processing': {'progress': 0.75, 'status': 'Processing batch 3/4', 'eta': timedelta(minutes=2, seconds=30), 'rate': 125.5},
        'model_training': {'progress': 0.42, 'status': 'Epoch 42/100', 'eta': timedelta(hours=1, minutes=15), 'rate': 2.3},
        'validation': {'progress': 0.89, 'status': 'Validating results', 'eta': timedelta(seconds=45), 'rate': 89.2},
        'optimization': {'progress': 0.33, 'status': 'Tuning parameters', 'eta': timedelta(minutes=8), 'rate': 15.7},
        'deployment': {'progress': 0.95, 'status': 'Finalizing deployment', 'eta': timedelta(seconds=12), 'rate': 5.1}
    }
    
    if metric == 'all':
        for name, data in metrics.items():
            console.print(f"\n[bold cyan]{name.replace('_', ' ').title()}:[/bold cyan]")
            progress_str = progress_bar.create(
                data['progress'], 
                data['status'], 
                data['eta'], 
                data['rate']
            )
            console.print(progress_str)
    else:
        if metric in metrics:
            data = metrics[metric]
            console.print(f"\n[bold cyan]{metric.replace('_', ' ').title()}:[/bold cyan]")
            progress_str = progress_bar.create(
                data['progress'], 
                data['status'], 
                data['eta'], 
                data['rate']
            )
            console.print(progress_str)
        else:
            console.print(f"[error]Unknown metric: {metric}[/error]")
            console.print(f"Available metrics: {', '.join(metrics.keys())}")


@analyze_group.command()
@click.option('--agent-id', help='Filter by specific agent ID')
def interactive(agent_id: str):
    """Launch interactive data exploration interface"""
    
    console.print("[info]Launching interactive exploration interface...[/info]")
    
    from ..utils.interactive import create_interactive_table, create_interactive_tree, TableColumn
    
    # Mock data for interactive exploration
    agent_data = [
        {
            'agent_id': 'agent_001',
            'status': 'active',
            'framework': 'LangChain',
            'uptime': '2h 15m',
            'events': 1247,
            'success_rate': 0.89,
            'last_activity': '2024-01-15 14:30:25'
        },
        {
            'agent_id': 'agent_002',
            'status': 'active',
            'framework': 'AutoGen',
            'uptime': '1h 42m',
            'events': 892,
            'success_rate': 0.92,
            'last_activity': '2024-01-15 14:29:18'
        },
        {
            'agent_id': 'agent_003',
            'status': 'idle',
            'framework': 'CrewAI',
            'uptime': '3h 8m',
            'events': 2156,
            'success_rate': 0.87,
            'last_activity': '2024-01-15 14:25:42'
        },
        {
            'agent_id': 'agent_004',
            'status': 'error',
            'framework': 'OpenAI',
            'uptime': '0h 23m',
            'events': 156,
            'success_rate': 0.45,
            'last_activity': '2024-01-15 14:28:03'
        },
        {
            'agent_id': 'agent_005',
            'status': 'active',
            'framework': 'LangChain',
            'uptime': '4h 56m',
            'events': 3421,
            'success_rate': 0.94,
            'last_activity': '2024-01-15 14:30:01'
        }
    ]
    
    # Define table columns
    columns = [
        TableColumn('Agent ID', 'agent_id', width=12),
        TableColumn('Status', 'status', width=10),
        TableColumn('Framework', 'framework', width=12),
        TableColumn('Uptime', 'uptime', width=10),
        TableColumn('Events', 'events', width=8),
        TableColumn('Success Rate', 'success_rate', width=12),
        TableColumn('Last Activity', 'last_activity', width=20)
    ]
    
    console.print("\n[bold cyan]Interactive Agent Explorer[/bold cyan]")
    console.print("Use vim-like navigation (hjkl), Space to select, Enter to view details")
    console.print("Press F1 for help, q to quit\n")
    
    # Launch interactive table
    try:
        selected_agent = create_interactive_table(agent_data, columns)
        
        if selected_agent:
            console.print(f"\n[bold green]Selected Agent:[/bold green] {selected_agent['agent_id']}")
            
            # Show detailed view
            detail_panel = Panel(
                f"[bold]Agent ID:[/bold] {selected_agent['agent_id']}\n"
                f"[bold]Status:[/bold] {selected_agent['status']}\n"
                f"[bold]Framework:[/bold] {selected_agent['framework']}\n"
                f"[bold]Uptime:[/bold] {selected_agent['uptime']}\n"
                f"[bold]Total Events:[/bold] {selected_agent['events']}\n"
                f"[bold]Success Rate:[/bold] {selected_agent['success_rate']:.1%}\n"
                f"[bold]Last Activity:[/bold] {selected_agent['last_activity']}",
                title="Agent Details",
                border_style="green"
            )
            console.print(detail_panel)
        else:
            console.print("[yellow]No agent selected[/yellow]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interactive session cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error in interactive mode: {str(e)}[/red]")


@analyze_group.command()
def tree_explorer():
    """Launch interactive tree explorer for causal relationships"""
    
    console.print("[info]Launching interactive tree explorer...[/info]")
    
    from ..utils.interactive import create_interactive_tree
    
    # Mock hierarchical causal data
    tree_data = {
        'name': 'Agent Behavior Analysis',
        'value': 'Root',
        'children': [
            {
                'name': 'Input Processing',
                'value': '35%',
                'children': [
                    {
                        'name': 'Data Validation',
                        'value': '15%',
                        'children': [
                            {'name': 'Schema Validation', 'value': '5%'},
                            {'name': 'Type Checking', 'value': '6%'},
                            {'name': 'Range Validation', 'value': '4%'}
                        ]
                    },
                    {
                        'name': 'Data Preprocessing',
                        'value': '20%',
                        'children': [
                            {'name': 'Cleaning', 'value': '8%'},
                            {'name': 'Normalization', 'value': '7%'},
                            {'name': 'Feature Extraction', 'value': '5%'}
                        ]
                    }
                ]
            },
            {
                'name': 'Decision Making',
                'value': '40%',
                'children': [
                    {
                        'name': 'Strategy Selection',
                        'value': '20%',
                        'children': [
                            {'name': 'Performance Analysis', 'value': '8%'},
                            {'name': 'Resource Assessment', 'value': '7%'},
                            {'name': 'Risk Evaluation', 'value': '5%'}
                        ]
                    },
                    {
                        'name': 'Parameter Optimization',
                        'value': '20%',
                        'children': [
                            {'name': 'Hyperparameter Tuning', 'value': '12%'},
                            {'name': 'Threshold Adjustment', 'value': '8%'}
                        ]
                    }
                ]
            },
            {
                'name': 'Output Generation',
                'value': '25%',
                'children': [
                    {
                        'name': 'Result Formatting',
                        'value': '15%',
                        'children': [
                            {'name': 'Data Serialization', 'value': '8%'},
                            {'name': 'Response Structuring', 'value': '7%'}
                        ]
                    },
                    {
                        'name': 'Quality Assurance',
                        'value': '10%',
                        'children': [
                            {'name': 'Output Validation', 'value': '6%'},
                            {'name': 'Consistency Check', 'value': '4%'}
                        ]
                    }
                ]
            }
        ]
    }
    
    console.print("\n[bold cyan]Interactive Tree Explorer[/bold cyan]")
    console.print("Use jk to navigate, hl to expand/collapse, Enter to toggle, Space to select")
    console.print("Press e to expand all, c to collapse all, F1 for help, q to quit\n")
    
    try:
        selected_node = create_interactive_tree(tree_data)
        
        if selected_node:
            console.print(f"\n[bold green]Selected Node:[/bold green] {selected_node['name']}")
            
            # Show node details
            detail_panel = Panel(
                f"[bold]Name:[/bold] {selected_node['name']}\n"
                f"[bold]Value:[/bold] {selected_node.get('value', 'N/A')}\n"
                f"[bold]Children:[/bold] {len(selected_node.get('children', []))}\n"
                f"[bold]Type:[/bold] {'Branch' if selected_node.get('children') else 'Leaf'}",
                title="Node Details",
                border_style="green"
            )
            console.print(detail_panel)
        else:
            console.print("[yellow]No node selected[/yellow]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interactive session cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error in interactive mode: {str(e)}[/red]")


@analyze_group.command()
@click.option('--data-file', help='JSON file containing data to analyze')
@click.option('--interactive', is_flag=True, help='Interactive query builder')
def query(data_file: str, interactive: bool):
    """Advanced query builder for data filtering"""
    
    console.print("[info]Advanced Query Builder[/info]")
    
    from ..utils.analysis_tools import QueryBuilder, QueryOperator
    
    # Mock data for demonstration
    sample_data = [
        {"agent_id": "agent_001", "status": "active", "success_rate": 0.89, "events": 1247, "framework": "LangChain"},
        {"agent_id": "agent_002", "status": "idle", "success_rate": 0.92, "events": 892, "framework": "AutoGen"},
        {"agent_id": "agent_003", "status": "active", "success_rate": 0.87, "events": 2156, "framework": "CrewAI"},
        {"agent_id": "agent_004", "status": "error", "success_rate": 0.45, "events": 156, "framework": "OpenAI"},
        {"agent_id": "agent_005", "status": "active", "success_rate": 0.94, "events": 3421, "framework": "LangChain"}
    ]
    
    if data_file:
        try:
            import json
            with open(data_file, 'r') as f:
                sample_data = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading data file: {e}[/red]")
            return
    
    builder = QueryBuilder()
    
    if interactive:
        console.print("\n[bold cyan]Interactive Query Builder[/bold cyan]")
        console.print("Build complex queries with multiple conditions")
        
        while True:
            console.print(f"\nCurrent conditions: {len(builder.conditions)}")
            for i, condition in enumerate(builder.conditions):
                console.print(f"  {i+1}. {condition.field} {condition.operator.value} {condition.value}")
            
            console.print("\nOptions:")
            console.print("1. Add condition")
            console.print("2. Remove condition")
            console.print("3. Set logic operator (AND/OR)")
            console.print("4. Execute query")
            console.print("5. Clear all conditions")
            console.print("6. Exit")
            
            choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4", "5", "6"])
            
            if choice == "1":
                # Add condition
                field = Prompt.ask("Field name")
                
                operators = [op.value for op in QueryOperator]
                operator_str = Prompt.ask("Operator", choices=operators)
                operator = QueryOperator(operator_str)
                
                value = Prompt.ask("Value")
                
                # Try to convert to appropriate type
                try:
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        value = float(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                except:
                    pass  # Keep as string
                
                case_sensitive = Confirm.ask("Case sensitive?", default=True)
                builder.add_condition(field, operator, value, case_sensitive)
                console.print("[green]Condition added[/green]")
            
            elif choice == "2":
                # Remove condition
                if builder.conditions:
                    index = IntPrompt.ask("Condition number to remove", default=1) - 1
                    if 0 <= index < len(builder.conditions):
                        builder.remove_condition(index)
                        console.print("[green]Condition removed[/green]")
                    else:
                        console.print("[red]Invalid condition number[/red]")
                else:
                    console.print("[yellow]No conditions to remove[/yellow]")
            
            elif choice == "3":
                # Set logic operator
                logic_op = Prompt.ask("Logic operator", choices=["AND", "OR"], default="AND")
                builder.logic_operator = logic_op
                console.print(f"[green]Logic operator set to {logic_op}[/green]")
            
            elif choice == "4":
                # Execute query
                results = builder.execute(sample_data)
                console.print(f"\n[bold]Query Results: {len(results)} records[/bold]")
                
                if results:
                    table = Table(show_header=True, header_style="bold magenta")
                    
                    # Add columns based on first result
                    for key in results[0].keys():
                        table.add_column(key, style="cyan")
                    
                    # Add rows (limit to first 10)
                    for result in results[:10]:
                        table.add_row(*[str(result.get(key, '')) for key in results[0].keys()])
                    
                    console.print(table)
                    
                    if len(results) > 10:
                        console.print(f"[dim]... and {len(results) - 10} more records[/dim]")
                else:
                    console.print("[yellow]No records match the query[/yellow]")
            
            elif choice == "5":
                # Clear conditions
                builder.clear_conditions()
                console.print("[green]All conditions cleared[/green]")
            
            elif choice == "6":
                break
    
    else:
        # Non-interactive mode - show example
        console.print("\n[bold]Example Query:[/bold]")
        builder.add_condition("status", QueryOperator.EQUALS, "active")
        builder.add_condition("success_rate", QueryOperator.GREATER_THAN, 0.85)
        
        results = builder.execute(sample_data)
        console.print(f"Found {len(results)} active agents with success rate > 85%")
        
        for result in results:
            console.print(f"  • {result['agent_id']}: {result['success_rate']:.1%} success rate")


@analyze_group.command()
@click.option('--field', required=True, help='Field to analyze statistically')
@click.option('--data-file', help='JSON file containing data')
def stats(field: str, data_file: str):
    """Statistical analysis of numeric data"""
    
    console.print(f"[info]Statistical Analysis of '{field}'[/info]")
    
    from ..utils.analysis_tools import StatisticalAnalysis
    
    # Mock data
    sample_data = [
        {"response_time": 120, "success_rate": 0.89, "events": 1247},
        {"response_time": 95, "success_rate": 0.92, "events": 892},
        {"response_time": 150, "success_rate": 0.87, "events": 2156},
        {"response_time": 300, "success_rate": 0.45, "events": 156},
        {"response_time": 80, "success_rate": 0.94, "events": 3421},
        {"response_time": 110, "success_rate": 0.91, "events": 1876},
        {"response_time": 200, "success_rate": 0.78, "events": 945}
    ]
    
    if data_file:
        try:
            import json
            with open(data_file, 'r') as f:
                sample_data = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading data file: {e}[/red]")
            return
    
    # Extract numeric values
    values = []
    for record in sample_data:
        if field in record:
            try:
                values.append(float(record[field]))
            except (ValueError, TypeError):
                continue
    
    if not values:
        console.print(f"[red]No numeric data found for field '{field}'[/red]")
        return
    
    # Perform statistical analysis
    stats_analyzer = StatisticalAnalysis()
    descriptive_stats = stats_analyzer.descriptive_stats(values)
    
    # Display results
    console.print(f"\n[bold cyan]Descriptive Statistics for '{field}':[/bold cyan]")
    
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Statistic", style="cyan")
    stats_table.add_column("Value", style="white")
    
    stats_table.add_row("Count", str(descriptive_stats["count"]))
    stats_table.add_row("Mean", f"{descriptive_stats['mean']:.3f}")
    stats_table.add_row("Median", f"{descriptive_stats['median']:.3f}")
    stats_table.add_row("Mode", f"{descriptive_stats['mode']:.3f}")
    stats_table.add_row("Std Dev", f"{descriptive_stats['std_dev']:.3f}")
    stats_table.add_row("Variance", f"{descriptive_stats['variance']:.3f}")
    stats_table.add_row("Min", f"{descriptive_stats['min']:.3f}")
    stats_table.add_row("Max", f"{descriptive_stats['max']:.3f}")
    stats_table.add_row("Range", f"{descriptive_stats['range']:.3f}")
    stats_table.add_row("Q1", f"{descriptive_stats['q1']:.3f}")
    stats_table.add_row("Q3", f"{descriptive_stats['q3']:.3f}")
    stats_table.add_row("IQR", f"{descriptive_stats['iqr']:.3f}")
    stats_table.add_row("CV", f"{descriptive_stats['cv']:.3f}")
    
    console.print(stats_table)
    
    # Create histogram
    console.print(f"\n[bold]Distribution of '{field}':[/bold]")
    from ..utils.ascii_viz import ASCIIHistogram, ChartConfig
    config = ChartConfig(width=60, height=10, title=f"{field} Distribution")
    histogram = ASCIIHistogram(config)
    hist_chart = histogram.create(values, bins=8)
    console.print(hist_chart)


@analyze_group.command()
@click.option('--x-field', required=True, help='X-axis field')
@click.option('--y-field', required=True, help='Y-axis field')
@click.option('--data-file', help='JSON file containing data')
def correlate(x_field: str, y_field: str, data_file: str):
    """Analyze correlation between two numeric fields"""
    
    console.print(f"[info]Correlation Analysis: {x_field} vs {y_field}[/info]")
    
    from ..utils.analysis_tools import DataCorrelationExplorer
    
    # Mock data
    sample_data = [
        {"response_time": 120, "success_rate": 0.89, "events": 1247, "cpu_usage": 45},
        {"response_time": 95, "success_rate": 0.92, "events": 892, "cpu_usage": 38},
        {"response_time": 150, "success_rate": 0.87, "events": 2156, "cpu_usage": 62},
        {"response_time": 300, "success_rate": 0.45, "events": 156, "cpu_usage": 85},
        {"response_time": 80, "success_rate": 0.94, "events": 3421, "cpu_usage": 32},
        {"response_time": 110, "success_rate": 0.91, "events": 1876, "cpu_usage": 41},
        {"response_time": 200, "success_rate": 0.78, "events": 945, "cpu_usage": 68}
    ]
    
    if data_file:
        try:
            import json
            with open(data_file, 'r') as f:
                sample_data = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading data file: {e}[/red]")
            return
    
    explorer = DataCorrelationExplorer()
    
    # Create scatter plot
    scatter_plot = explorer.create_scatter_plot(sample_data, x_field, y_field)
    console.print(f"\n[bold]Scatter Plot: {x_field} vs {y_field}[/bold]")
    console.print(scatter_plot)
    
    # Calculate correlation
    from ..utils.analysis_tools import StatisticalAnalysis
    stats = StatisticalAnalysis()
    
    # Extract values
    x_values, y_values = [], []
    for record in sample_data:
        try:
            x_val = float(record.get(x_field, 0))
            y_val = float(record.get(y_field, 0))
            x_values.append(x_val)
            y_values.append(y_val)
        except (ValueError, TypeError):
            continue
    
    if len(x_values) >= 2:
        correlation_result = stats.correlation_analysis(x_values, y_values)
        
        console.print(f"\n[bold cyan]Correlation Analysis:[/bold cyan]")
        console.print(f"Correlation Coefficient: {correlation_result['correlation']:.3f}")
        console.print(f"Strength: {correlation_result['strength']}")
        console.print(f"R-squared: {correlation_result['r_squared']:.3f}")
        console.print(f"Sample Size: {correlation_result['sample_size']}")
        
        # Interpretation
        corr = correlation_result['correlation']
        if corr > 0.7:
            interpretation = "Strong positive correlation"
        elif corr > 0.3:
            interpretation = "Moderate positive correlation"
        elif corr > -0.3:
            interpretation = "Weak or no correlation"
        elif corr > -0.7:
            interpretation = "Moderate negative correlation"
        else:
            interpretation = "Strong negative correlation"
        
        console.print(f"Interpretation: [bold]{interpretation}[/bold]")
    else:
        console.print("[red]Insufficient data for correlation analysis[/red]")


@analyze_group.command()
@click.option('--time-field', required=True, help='Time field name')
@click.option('--value-field', required=True, help='Value field name')
@click.option('--data-file', help='JSON file containing time series data')
def timeseries(time_field: str, value_field: str, data_file: str):
    """Time series analysis with trend detection"""
    
    console.print(f"[info]Time Series Analysis: {value_field} over {time_field}[/info]")
    
    from ..utils.analysis_tools import TimeSeriesAnalyzer
    from datetime import datetime, timedelta
    
    # Mock time series data
    base_time = datetime(2024, 1, 1)
    sample_data = []
    for i in range(30):
        # Create trend with some noise
        trend_value = i * 2 + 50 + (i % 7) * 5  # Weekly pattern
        sample_data.append({
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "response_time": trend_value + (i % 3) * 10,  # Some variation
            "success_rate": 0.9 - (i % 10) * 0.02,  # Slight decline
            "events": 1000 + i * 50 + (i % 5) * 100
        })
    
    if data_file:
        try:
            import json
            with open(data_file, 'r') as f:
                sample_data = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading data file: {e}[/red]")
            return
    
    analyzer = TimeSeriesAnalyzer()
    
    # Perform analysis
    analysis_result = analyzer.analyze_time_series(sample_data, time_field, value_field)
    
    if "error" in analysis_result:
        console.print(f"[red]Error: {analysis_result['error']}[/red]")
        return
    
    # Display results
    console.print(f"\n[bold cyan]Time Series Analysis Results:[/bold cyan]")
    
    # Basic statistics
    basic_stats = analysis_result["basic_stats"]
    console.print(f"\n[bold]Basic Statistics:[/bold]")
    console.print(f"Data Points: {basic_stats['count']}")
    console.print(f"Mean: {basic_stats['mean']:.2f}")
    console.print(f"Std Dev: {basic_stats['std_dev']:.2f}")
    console.print(f"Min: {basic_stats['min']:.2f}")
    console.print(f"Max: {basic_stats['max']:.2f}")
    
    # Trend analysis
    trend = analysis_result["trend"]
    console.print(f"\n[bold]Trend Analysis:[/bold]")
    console.print(f"Direction: {trend['direction']}")
    console.print(f"Slope: {trend['slope']:.4f}")
    console.print(f"R-squared: {trend['r_squared']:.3f}")
    console.print(f"Strength: {trend['strength']}")
    
    # Seasonality
    seasonality = analysis_result["seasonality"]
    console.print(f"\n[bold]Seasonality:[/bold]")
    if seasonality["detected"]:
        console.print(f"Seasonal pattern detected with period {seasonality['period']}")
        console.print(f"Confidence: {seasonality['confidence']}")
    else:
        console.print("No clear seasonal pattern detected")
    
    # Volatility
    volatility = analysis_result["volatility"]
    console.print(f"\n[bold]Volatility:[/bold]")
    console.print(f"Price Volatility: {volatility['price_volatility']:.2f}")
    console.print(f"Max Drawdown: {volatility['max_drawdown']:.1%}")
    
    # Create time series chart
    chart = analyzer.create_time_series_chart(sample_data, time_field, value_field)
    console.print(f"\n[bold]Time Series Chart:[/bold]")
    console.print(chart)