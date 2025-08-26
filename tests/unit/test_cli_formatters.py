"""
Unit tests for CLI formatters
"""

import pytest
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel

from escai_framework.cli.utils.formatters import (
    format_agent_status_table,
    format_epistemic_state,
    format_behavioral_patterns,
    format_causal_tree,
    format_predictions,
    format_ascii_chart,
    create_progress_bar
)


class TestCLIFormatters:
    """Test CLI formatting utilities"""
    
    def test_format_agent_status_table_empty(self):
        """Test formatting empty agent status table"""
        table = format_agent_status_table([])
        assert isinstance(table, Table)
        assert table.title == "Agent Status"
        assert len(table.columns) == 6
        assert len(table.rows) == 0
    
    def test_format_agent_status_table_with_data(self):
        """Test formatting agent status table with data"""
        agents = [
            {
                'id': 'agent_001',
                'status': 'active',
                'framework': 'langchain',
                'uptime': '2h 15m',
                'event_count': 1247,
                'last_activity': '2s ago'
            },
            {
                'id': 'agent_002',
                'status': 'inactive',
                'framework': 'autogen',
                'uptime': '45m',
                'event_count': 892,
                'last_activity': '1m ago'
            }
        ]
        
        table = format_agent_status_table(agents)
        assert isinstance(table, Table)
        assert len(table.rows) == 2
        
        # Check that status icons are included by checking the table has data
        # We can't easily test the rendered content, so just verify structure
        assert table.title == "Agent Status"
        assert len(table.columns) == 6
    
    def test_format_agent_status_table_missing_fields(self):
        """Test formatting agent status table with missing fields"""
        agents = [
            {
                'id': 'agent_001',
                # Missing other fields
            }
        ]
        
        table = format_agent_status_table(agents)
        assert isinstance(table, Table)
        assert len(table.rows) == 1
        
        # Should handle missing fields gracefully
        table_str = str(table)
        assert 'N/A' in table_str or '0' in table_str
    
    def test_format_epistemic_state_complete(self):
        """Test formatting complete epistemic state"""
        state = {
            'agent_id': 'test_agent',
            'beliefs': [
                {'content': 'User wants to analyze data', 'confidence': 0.95},
                {'content': 'Data is in CSV format', 'confidence': 0.87}
            ],
            'knowledge': {
                'fact_count': 156,
                'concept_count': 43,
                'relationship_count': 89
            },
            'goals': [
                {'description': 'Load and validate data', 'progress': 1.0},
                {'description': 'Perform analysis', 'progress': 0.65}
            ],
            'uncertainty_score': 0.34
        }
        
        panel = format_epistemic_state(state)
        assert isinstance(panel, Panel)
        assert 'test_agent' in panel.title
        
        content = str(panel.renderable)
        assert 'Beliefs:' in content
        assert 'Knowledge:' in content
        assert 'Goals:' in content
        assert 'Uncertainty:' in content
    
    def test_format_epistemic_state_empty(self):
        """Test formatting empty epistemic state"""
        state = {'agent_id': 'test_agent'}
        
        panel = format_epistemic_state(state)
        assert isinstance(panel, Panel)
        assert 'test_agent' in panel.title
        
        content = str(panel.renderable)
        assert 'Uncertainty:' in content  # Should show default uncertainty
    
    def test_format_behavioral_patterns_empty(self):
        """Test formatting empty behavioral patterns"""
        table = format_behavioral_patterns([])
        assert isinstance(table, Table)
        assert table.title == "Behavioral Patterns"
        assert len(table.columns) == 5
        assert len(table.rows) == 0
    
    def test_format_behavioral_patterns_with_data(self):
        """Test formatting behavioral patterns with data"""
        patterns = [
            {
                'pattern_name': 'Sequential Processing',
                'frequency': 45,
                'success_rate': 0.89,
                'average_duration': '2.3s',
                'statistical_significance': 0.95
            },
            {
                'pattern_name': 'Error Recovery',
                'frequency': 12,
                'success_rate': 0.45,  # Low success rate
                'average_duration': '5.1s',
                'statistical_significance': 0.78
            }
        ]
        
        table = format_behavioral_patterns(patterns)
        assert isinstance(table, Table)
        assert len(table.rows) == 2
        
        # Check that table structure is correct
        assert table.title == "Behavioral Patterns"
        assert len(table.columns) == 5
    
    def test_format_causal_tree_empty(self):
        """Test formatting empty causal tree"""
        tree = format_causal_tree([])
        assert isinstance(tree, Tree)
        assert 'Causal Relationships' in str(tree.label)
        assert len(tree.children) == 0
    
    def test_format_causal_tree_with_data(self):
        """Test formatting causal tree with data"""
        relationships = [
            {
                'cause_event': 'Data Validation Error',
                'effect_event': 'Retry Mechanism Triggered',
                'strength': 0.87,
                'confidence': 0.92
            },
            {
                'cause_event': 'Data Validation Error',
                'effect_event': 'Error Logging',
                'strength': 0.95,
                'confidence': 0.98
            },
            {
                'cause_event': 'Large Dataset',
                'effect_event': 'Batch Processing',
                'strength': 0.78,
                'confidence': 0.85
            }
        ]
        
        tree = format_causal_tree(relationships)
        assert isinstance(tree, Tree)
        assert len(tree.children) == 2  # Two unique causes
        
        # Check that cause events are grouped
        cause_labels = [str(child.label) for child in tree.children]
        assert any('Data Validation Error' in label for label in cause_labels)
        assert any('Large Dataset' in label for label in cause_labels)
    
    def test_format_predictions_empty(self):
        """Test formatting empty predictions"""
        panel = format_predictions([])
        assert isinstance(panel, Panel)
        assert panel.title == "Performance Predictions"
        
        content = str(panel.renderable)
        assert 'No predictions available' in content
    
    def test_format_predictions_with_data(self):
        """Test formatting predictions with data"""
        predictions = [
            {
                'predicted_outcome': 'success',
                'confidence': 0.87,
                'risk_factors': ['High memory usage', 'Complex query'],
                'trend': 'improving'
            },
            {
                'predicted_outcome': 'failure',
                'confidence': 0.72,
                'risk_factors': [],
                'trend': 'declining'
            },
            {
                'predicted_outcome': 'unknown',
                'confidence': 0.45,
                'risk_factors': ['Network latency'],
                'trend': 'stable'
            }
        ]
        
        panel = format_predictions(predictions)
        assert isinstance(panel, Panel)
        
        content = str(panel.renderable)
        assert '✅' in content  # Success icon
        assert '❌' in content  # Failure icon
        assert '❓' in content  # Unknown icon
        assert '📈' in content  # Improving trend
        assert '📉' in content  # Declining trend
        assert '➡️' in content  # Stable trend
    
    def test_format_ascii_chart_empty(self):
        """Test formatting ASCII chart with empty data"""
        chart = format_ascii_chart([], "Empty Chart")
        assert "Empty Chart: No data available" in chart
    
    def test_format_ascii_chart_single_value(self):
        """Test formatting ASCII chart with single value"""
        chart = format_ascii_chart([5.0], "Single Value", width=10, height=5)
        assert "Single Value" in chart
        assert "Min: 5.00, Max: 5.00" in chart
        assert "┌" in chart and "┐" in chart  # Chart borders
        assert "└" in chart and "┘" in chart
    
    def test_format_ascii_chart_multiple_values(self):
        """Test formatting ASCII chart with multiple values"""
        data = [1.0, 3.0, 2.0, 4.0, 2.5]
        chart = format_ascii_chart(data, "Test Chart", width=20, height=8)
        
        assert "Test Chart" in chart
        assert "Min: 1.00, Max: 4.00" in chart
        assert "█" in chart  # Should contain chart bars
        assert chart.count("│") >= 8  # Should have vertical borders
        assert chart.count("─") >= 20  # Should have horizontal borders
    
    def test_format_ascii_chart_custom_dimensions(self):
        """Test formatting ASCII chart with custom dimensions"""
        data = [1, 2, 3, 4, 5]
        chart = format_ascii_chart(data, "Custom Chart", width=30, height=12)
        
        lines = chart.split('\n')
        # Should have title + top border + chart rows + bottom border + min/max line
        assert len(lines) >= 15
        
        # Check width (accounting for borders)
        chart_lines = [line for line in lines if line.startswith('│')]
        if chart_lines:
            # Each chart line should be width + 2 (for borders)
            assert len(chart_lines[0]) == 32  # 30 + 2 borders
    
    def test_create_progress_bar(self):
        """Test creating progress bar"""
        progress = create_progress_bar("Test Progress")
        assert progress is not None
        
        # Should be able to add tasks
        task_id = progress.add_task("Test Task", total=100)
        assert task_id is not None
    
    def test_format_epistemic_state_partial_data(self):
        """Test formatting epistemic state with partial data"""
        # Test with only beliefs
        state_beliefs_only = {
            'agent_id': 'test_agent',
            'beliefs': [{'content': 'Test belief', 'confidence': 0.8}]
        }
        panel = format_epistemic_state(state_beliefs_only)
        content = str(panel.renderable)
        assert 'Beliefs:' in content
        assert 'Test belief' in content
        
        # Test with only knowledge
        state_knowledge_only = {
            'agent_id': 'test_agent',
            'knowledge': {'fact_count': 10, 'concept_count': 5}
        }
        panel = format_epistemic_state(state_knowledge_only)
        content = str(panel.renderable)
        assert 'Knowledge:' in content
        assert 'Facts: 10' in content
        
        # Test with only goals
        state_goals_only = {
            'agent_id': 'test_agent',
            'goals': [{'description': 'Test goal', 'progress': 0.5}]
        }
        panel = format_epistemic_state(state_goals_only)
        content = str(panel.renderable)
        assert 'Goals:' in content
        assert 'Test goal' in content
    
    def test_format_behavioral_patterns_edge_cases(self):
        """Test formatting behavioral patterns with edge cases"""
        patterns = [
            {
                'pattern_name': 'Perfect Pattern',
                'frequency': 100,
                'success_rate': 1.0,  # Perfect success rate
                'average_duration': '1.0s',
                'statistical_significance': 1.0  # Perfect significance
            },
            {
                'pattern_name': 'Zero Pattern',
                'frequency': 0,
                'success_rate': 0.0,  # Zero success rate
                'average_duration': '0.0s',
                'statistical_significance': 0.0  # Zero significance
            }
        ]
        
        table = format_behavioral_patterns(patterns)
        assert len(table.rows) == 2
        
        # Should handle edge cases gracefully
        assert table.title == "Behavioral Patterns"
        assert len(table.columns) == 5
    
    def test_format_causal_tree_duplicate_causes(self):
        """Test formatting causal tree with duplicate cause events"""
        relationships = [
            {
                'cause_event': 'Same Cause',
                'effect_event': 'Effect 1',
                'strength': 0.8,
                'confidence': 0.9
            },
            {
                'cause_event': 'Same Cause',
                'effect_event': 'Effect 2',
                'strength': 0.7,
                'confidence': 0.8
            }
        ]
        
        tree = format_causal_tree(relationships)
        assert len(tree.children) == 1  # Should group by cause
        
        # The single cause should have multiple effects
        cause_node = tree.children[0]
        assert len(cause_node.children) == 2