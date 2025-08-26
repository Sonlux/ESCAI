"""
Unit tests for CLI visualization enhancements
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from escai_framework.cli.utils.ascii_viz import (
    ASCIIChart, ASCIIBarChart, ASCIILineChart, ASCIIHistogram,
    ASCIIScatterPlot, ASCIIHeatmap, ASCIISparkline, ASCIIProgressBar,
    ASCIITreeView, ChartConfig, create_epistemic_state_chart,
    create_pattern_frequency_heatmap, create_causal_strength_scatter
)
from escai_framework.cli.utils.data_filters import (
    DataFilter, FilterCondition, FilterOperator, SearchQuery,
    create_data_filter, interactive_data_explorer
)
from escai_framework.cli.utils.console import (
    get_console, set_color_scheme, get_available_schemes,
    create_themed_console
)
from escai_framework.cli.utils.reporting import DataExporter, ReportFormat


class TestASCIIVisualization:
    """Test ASCII visualization components"""
    
    def test_chart_config_creation(self):
        """Test chart configuration creation"""
        config = ChartConfig(
            width=80,
            height=20,
            title="Test Chart",
            color_scheme="dark",
            border_style="double"
        )
        
        assert config.width == 80
        assert config.height == 20
        assert config.title == "Test Chart"
        assert config.color_scheme == "dark"
        assert config.border_style == "double"
    
    def test_ascii_chart_base_class(self):
        """Test ASCII chart base class"""
        config = ChartConfig(unicode_chars=True, border_style="rounded")
        chart = ASCIIChart(config)
        
        assert chart.config == config
        assert 'full' in chart.chars
        assert 'corner_tl' in chart.chars
        assert 'primary' in chart.colors
    
    def test_bar_chart_creation(self):
        """Test ASCII bar chart creation"""
        config = ChartConfig(width=40, height=10, title="Test Bar Chart")
        chart = ASCIIBarChart(config)
        
        data = [10, 25, 15, 30, 20]
        labels = ["A", "B", "C", "D", "E"]
        
        result = chart.create(data, labels)
        
        assert isinstance(result, str)
        assert "Test Bar Chart" in result
        assert len(result.split('\n')) > 5  # Should have multiple lines
    
    def test_line_chart_creation(self):
        """Test ASCII line chart creation"""
        config = ChartConfig(width=50, height=8, title="Test Line Chart")
        chart = ASCIILineChart(config)
        
        data = [1, 3, 2, 5, 4, 6, 5, 7]
        
        result = chart.create(data)
        
        assert isinstance(result, str)
        assert "Test Line Chart" in result
        assert "●" in result or "·" in result  # Should contain plot points
    
    def test_histogram_creation(self):
        """Test ASCII histogram creation"""
        config = ChartConfig(width=60, height=10, title="Test Histogram")
        chart = ASCIIHistogram(config)
        
        data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
        
        result = chart.create(data, bins=5)
        
        assert isinstance(result, str)
        assert "Test Histogram" in result
    
    def test_scatter_plot_creation(self):
        """Test ASCII scatter plot creation"""
        config = ChartConfig(width=40, height=10, title="Test Scatter")
        chart = ASCIIScatterPlot(config)
        
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 1, 5, 3]
        
        result = chart.create(x_data, y_data)
        
        assert isinstance(result, str)
        assert "Test Scatter" in result
        assert "●" in result  # Should contain plot points
    
    def test_heatmap_creation(self):
        """Test ASCII heatmap creation"""
        config = ChartConfig(width=50, height=10, title="Test Heatmap")
        chart = ASCIIHeatmap(config)
        
        data = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        row_labels = ["Row1", "Row2", "Row3"]
        col_labels = ["Col1", "Col2", "Col3"]
        
        result = chart.create(data, row_labels, col_labels)
        
        assert isinstance(result, str)
        assert "Test Heatmap" in result
        assert "Row1" in result
        assert "Col1" in result
    
    def test_sparkline_creation(self):
        """Test ASCII sparkline creation"""
        sparkline = ASCIISparkline(unicode_chars=True)
        
        data = [1, 3, 2, 5, 4, 6, 5, 7, 6, 8]
        
        result = sparkline.create(data, width=20)
        
        assert isinstance(result, str)
        assert len(result) == 20
    
    def test_progress_bar_creation(self):
        """Test ASCII progress bar creation"""
        progress_bar = ASCIIProgressBar(width=40, unicode_chars=True)
        
        result = progress_bar.create(
            progress=0.75,
            status="Processing...",
            eta=timedelta(minutes=2, seconds=30),
            rate=125.5
        )
        
        assert isinstance(result, str)
        assert "75.0%" in result
        assert "Processing..." in result
        assert "ETA:" in result
        assert "125.5/s" in result
    
    def test_tree_view_creation(self):
        """Test ASCII tree view creation"""
        tree_view = ASCIITreeView(unicode_chars=True)
        
        tree_data = {
            'name': 'Root',
            'value': '100%',
            'children': [
                {
                    'name': 'Child1',
                    'value': '50%',
                    'children': [
                        {'name': 'Grandchild1', 'value': '25%'}
                    ]
                },
                {'name': 'Child2', 'value': '50%'}
            ]
        }
        
        result = tree_view.create(tree_data, max_depth=3)
        
        assert isinstance(result, str)
        assert "Root" in result
        assert "Child1" in result
        assert "Grandchild1" in result
    
    def test_epistemic_state_chart(self):
        """Test epistemic state chart creation"""
        epistemic_data = {
            'beliefs': [
                {'content': 'Test belief 1', 'confidence': 0.8},
                {'content': 'Test belief 2', 'confidence': 0.6}
            ],
            'uncertainty_history': [0.2, 0.3, 0.25, 0.15, 0.1]
        }
        
        result = create_epistemic_state_chart(epistemic_data)
        
        assert isinstance(result, str)
        assert "Belief Confidences" in result
        assert "Uncertainty Trend" in result
    
    def test_pattern_frequency_heatmap(self):
        """Test pattern frequency heatmap creation"""
        pattern_data = [
            {'pattern_name': 'Pattern1', 'time_period': 'Morning', 'frequency': 10},
            {'pattern_name': 'Pattern1', 'time_period': 'Evening', 'frequency': 5},
            {'pattern_name': 'Pattern2', 'time_period': 'Morning', 'frequency': 8}
        ]
        
        result = create_pattern_frequency_heatmap(pattern_data)
        
        assert isinstance(result, str)
        assert "Pattern Frequency Heatmap" in result
    
    def test_causal_strength_scatter(self):
        """Test causal strength scatter plot creation"""
        causal_data = [
            {'strength': 0.8, 'confidence': 0.9},
            {'strength': 0.6, 'confidence': 0.7},
            {'strength': 0.9, 'confidence': 0.8}
        ]
        
        result = create_causal_strength_scatter(causal_data)
        
        assert isinstance(result, str)
        assert "Causal Relationships" in result


class TestDataFiltering:
    """Test data filtering and search functionality"""
    
    def test_data_filter_creation(self):
        """Test data filter creation"""
        filter_engine = create_data_filter()
        
        assert isinstance(filter_engine, DataFilter)
        assert filter_engine.saved_queries == {}
        assert filter_engine.filter_history == []
    
    def test_filter_condition_creation(self):
        """Test filter condition creation"""
        condition = FilterCondition(
            field="status",
            operator=FilterOperator.EQUALS,
            value="active",
            case_sensitive=False
        )
        
        assert condition.field == "status"
        assert condition.operator == FilterOperator.EQUALS
        assert condition.value == "active"
        assert condition.case_sensitive is False
    
    def test_search_query_creation(self):
        """Test search query creation"""
        conditions = [
            FilterCondition("status", FilterOperator.EQUALS, "active"),
            FilterCondition("confidence", FilterOperator.GREATER_THAN, 0.8)
        ]
        
        query = SearchQuery(
            conditions=conditions,
            logic="AND",
            limit=10,
            sort_by="confidence",
            sort_desc=True
        )
        
        assert len(query.conditions) == 2
        assert query.logic == "AND"
        assert query.limit == 10
        assert query.sort_by == "confidence"
        assert query.sort_desc is True
    
    def test_apply_filter_equals(self):
        """Test applying equals filter"""
        filter_engine = DataFilter()
        
        data = [
            {'status': 'active', 'confidence': 0.9},
            {'status': 'inactive', 'confidence': 0.7},
            {'status': 'active', 'confidence': 0.8}
        ]
        
        query = SearchQuery(
            conditions=[FilterCondition("status", FilterOperator.EQUALS, "active")]
        )
        
        result = filter_engine.apply_filter(data, query)
        
        assert len(result) == 2
        assert all(item['status'] == 'active' for item in result)
    
    def test_apply_filter_greater_than(self):
        """Test applying greater than filter"""
        filter_engine = DataFilter()
        
        data = [
            {'confidence': 0.9},
            {'confidence': 0.7},
            {'confidence': 0.8}
        ]
        
        query = SearchQuery(
            conditions=[FilterCondition("confidence", FilterOperator.GREATER_THAN, 0.75)]
        )
        
        result = filter_engine.apply_filter(data, query)
        
        assert len(result) == 2
        assert all(item['confidence'] > 0.75 for item in result)
    
    def test_apply_filter_contains(self):
        """Test applying contains filter"""
        filter_engine = DataFilter()
        
        data = [
            {'name': 'test_agent_001'},
            {'name': 'production_agent'},
            {'name': 'test_agent_002'}
        ]
        
        query = SearchQuery(
            conditions=[FilterCondition("name", FilterOperator.CONTAINS, "test")]
        )
        
        result = filter_engine.apply_filter(data, query)
        
        assert len(result) == 2
        assert all("test" in item['name'] for item in result)
    
    def test_apply_filter_and_logic(self):
        """Test applying filter with AND logic"""
        filter_engine = DataFilter()
        
        data = [
            {'status': 'active', 'confidence': 0.9},
            {'status': 'active', 'confidence': 0.7},
            {'status': 'inactive', 'confidence': 0.9}
        ]
        
        query = SearchQuery(
            conditions=[
                FilterCondition("status", FilterOperator.EQUALS, "active"),
                FilterCondition("confidence", FilterOperator.GREATER_THAN, 0.8)
            ],
            logic="AND"
        )
        
        result = filter_engine.apply_filter(data, query)
        
        assert len(result) == 1
        assert result[0]['status'] == 'active'
        assert result[0]['confidence'] == 0.9
    
    def test_apply_filter_or_logic(self):
        """Test applying filter with OR logic"""
        filter_engine = DataFilter()
        
        data = [
            {'status': 'active', 'confidence': 0.7},
            {'status': 'inactive', 'confidence': 0.9},
            {'status': 'inactive', 'confidence': 0.6}
        ]
        
        query = SearchQuery(
            conditions=[
                FilterCondition("status", FilterOperator.EQUALS, "active"),
                FilterCondition("confidence", FilterOperator.GREATER_THAN, 0.8)
            ],
            logic="OR"
        )
        
        result = filter_engine.apply_filter(data, query)
        
        assert len(result) == 2
    
    def test_nested_field_access(self):
        """Test accessing nested fields"""
        filter_engine = DataFilter()
        
        data = [
            {'metadata': {'priority': 'high', 'tags': ['urgent']}},
            {'metadata': {'priority': 'low', 'tags': ['normal']}}
        ]
        
        query = SearchQuery(
            conditions=[FilterCondition("metadata.priority", FilterOperator.EQUALS, "high")]
        )
        
        result = filter_engine.apply_filter(data, query)
        
        assert len(result) == 1
        assert result[0]['metadata']['priority'] == 'high'
    
    def test_quick_search(self):
        """Test quick text search"""
        filter_engine = DataFilter()
        
        data = [
            {'name': 'agent_001', 'description': 'Processing agent'},
            {'name': 'agent_002', 'description': 'Analysis agent'},
            {'name': 'agent_003', 'description': 'Monitoring system'}
        ]
        
        result = filter_engine.quick_search(data, "agent")
        
        assert len(result) == 3  # All contain "agent"
        
        result = filter_engine.quick_search(data, "processing")
        
        assert len(result) == 1
        assert "Processing" in result[0]['description']
    
    def test_fuzzy_search(self):
        """Test fuzzy search functionality"""
        filter_engine = DataFilter()
        
        data = [
            {'name': 'processing_agent'},
            {'name': 'process_monitor'},
            {'name': 'data_analyzer'}
        ]
        
        result = filter_engine.fuzzy_search(data, "process", "name", threshold=0.3)
        
        assert len(result) >= 2  # Should match processing_agent and process_monitor
    
    def test_save_and_load_query(self):
        """Test saving and loading queries"""
        filter_engine = DataFilter()
        
        query = SearchQuery(
            conditions=[FilterCondition("status", FilterOperator.EQUALS, "active")]
        )
        
        filter_engine.save_query("test_query", query)
        
        assert "test_query" in filter_engine.saved_queries
        
        loaded_query = filter_engine.load_query("test_query")
        
        assert loaded_query is not None
        assert len(loaded_query.conditions) == 1
        assert loaded_query.conditions[0].field == "status"
    
    def test_filter_summary_creation(self):
        """Test filter summary panel creation"""
        filter_engine = DataFilter()
        
        query = SearchQuery(
            conditions=[
                FilterCondition("status", FilterOperator.EQUALS, "active"),
                FilterCondition("confidence", FilterOperator.GREATER_THAN, 0.8)
            ],
            logic="AND",
            sort_by="confidence",
            sort_desc=True,
            limit=10
        )
        
        summary = filter_engine.create_filter_summary(query, 5)
        
        assert summary.title == "🔍 Filter Summary"
        assert "Conditions (AND)" in summary.renderable
        assert "Sort: confidence (DESC)" in summary.renderable
        assert "Results: 5 items" in summary.renderable


class TestColorSchemes:
    """Test color scheme functionality"""
    
    def test_get_available_schemes(self):
        """Test getting available color schemes"""
        schemes = get_available_schemes()
        
        assert isinstance(schemes, list)
        assert "default" in schemes
        assert "dark" in schemes
        assert "light" in schemes
        assert "high_contrast" in schemes
        assert "monochrome" in schemes
    
    def test_set_color_scheme(self):
        """Test setting color scheme"""
        result = set_color_scheme("dark")
        assert result is True
        
        result = set_color_scheme("invalid_scheme")
        assert result is False
    
    def test_create_themed_console(self):
        """Test creating themed console"""
        console = create_themed_console("dark")
        
        assert console is not None
        assert hasattr(console, 'print')
    
    def test_get_console_with_theme(self):
        """Test getting console with specific theme"""
        console = get_console("light")
        
        assert console is not None
        assert hasattr(console, 'print')


class TestDataExporter:
    """Test data export functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from rich.console import Console
        self.console = Console()
        self.exporter = DataExporter(self.console)
        
        self.test_data = {
            'agents': [
                {'id': 'agent_001', 'status': 'active', 'confidence': 0.9},
                {'id': 'agent_002', 'status': 'idle', 'confidence': 0.7}
            ],
            'metadata': {
                'export_time': '2024-01-15T14:30:00',
                'total_count': 2
            }
        }
    
    def test_export_json(self, tmp_path):
        """Test JSON export"""
        output_path = tmp_path / "test_export"
        
        result_path = self.exporter._export_json(self.test_data, output_path)
        
        assert result_path.exists()
        assert result_path.suffix == '.json'
        
        import json
        with open(result_path, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data == self.test_data
    
    def test_export_csv(self, tmp_path):
        """Test CSV export"""
        output_path = tmp_path / "test_export"
        
        result_path = self.exporter._export_csv(self.test_data, output_path)
        
        assert result_path.exists()
        assert result_path.suffix == '.csv'
        
        # Check that file has content
        assert result_path.stat().st_size > 0
    
    def test_export_markdown(self, tmp_path):
        """Test Markdown export"""
        output_path = tmp_path / "test_export"
        
        result_path = self.exporter._export_markdown(self.test_data, output_path)
        
        assert result_path.exists()
        assert result_path.suffix == '.md'
        
        content = result_path.read_text()
        assert "# ESCAI Data Export" in content
        assert "Generated:" in content
    
    def test_export_txt(self, tmp_path):
        """Test plain text export"""
        output_path = tmp_path / "test_export"
        
        result_path = self.exporter._export_txt(self.test_data, output_path)
        
        assert result_path.exists()
        assert result_path.suffix == '.txt'
        
        content = result_path.read_text()
        assert "ESCAI Data Export" in content
        assert "Generated:" in content
    
    def test_flatten_data_for_csv(self):
        """Test data flattening for CSV export"""
        nested_data = {
            'agent': {
                'id': 'agent_001',
                'metadata': {'priority': 'high', 'tags': ['urgent', 'critical']}
            },
            'performance': {'success_rate': 0.9, 'avg_time': 1.2}
        }
        
        flattened = self.exporter._flatten_data_for_csv(nested_data)
        
        assert len(flattened) == 1
        assert 'agent_id' in flattened[0]
        assert 'agent_metadata_priority' in flattened[0]
        assert 'performance_success_rate' in flattened[0]


class TestCLICommands:
    """Test CLI command functionality"""
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_patterns_command(self, mock_console):
        """Test analyze patterns command"""
        from escai_framework.cli.commands.analyze import patterns
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(patterns, ['--timeframe', '1h'])
        
        assert result.exit_code == 0
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_search_command(self, mock_console):
        """Test analyze search command"""
        from escai_framework.cli.commands.analyze import search
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(search, ['--query', 'test'])
        
        assert result.exit_code == 0
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_export_command(self, mock_console):
        """Test analyze export command"""
        from escai_framework.cli.commands.analyze import export
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(export, ['--format', 'json'])
        
        assert result.exit_code == 0
    
    @patch('escai_framework.cli.commands.config.console')
    def test_config_theme_command(self, mock_console):
        """Test config theme command"""
        from escai_framework.cli.commands.config import theme
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(theme, ['--list'])
        
        assert result.exit_code == 0


class TestVisualizationIntegration:
    """Test integration between visualization components"""
    
    def test_chart_with_different_color_schemes(self):
        """Test charts with different color schemes"""
        data = [10, 20, 15, 25, 30]
        
        for scheme in ["default", "dark", "light"]:
            config = ChartConfig(
                width=40,
                height=8,
                title=f"Test Chart - {scheme}",
                color_scheme=scheme
            )
            
            chart = ASCIIBarChart(config)
            result = chart.create(data)
            
            assert isinstance(result, str)
            assert f"Test Chart - {scheme}" in result
    
    def test_chart_with_different_border_styles(self):
        """Test charts with different border styles"""
        data = [1, 2, 3, 4, 5]
        
        for style in ["rounded", "square", "double", "thick"]:
            config = ChartConfig(
                width=30,
                height=6,
                title=f"Test - {style}",
                border_style=style,
                unicode_chars=True
            )
            
            chart = ASCIILineChart(config)
            result = chart.create(data)
            
            assert isinstance(result, str)
    
    def test_filter_and_visualization_pipeline(self):
        """Test complete pipeline from filtering to visualization"""
        # Create test data
        data = [
            {'agent_id': 'agent_001', 'confidence': 0.9, 'success': True},
            {'agent_id': 'agent_002', 'confidence': 0.7, 'success': True},
            {'agent_id': 'agent_003', 'confidence': 0.5, 'success': False},
            {'agent_id': 'agent_004', 'confidence': 0.8, 'success': True}
        ]
        
        # Filter data
        filter_engine = DataFilter()
        query = SearchQuery(
            conditions=[FilterCondition("success", FilterOperator.EQUALS, True)]
        )
        
        filtered_data = filter_engine.apply_filter(data, query)
        
        # Extract confidence values for visualization
        confidence_values = [item['confidence'] for item in filtered_data]
        
        # Create visualization
        config = ChartConfig(width=40, height=8, title="Success Confidence")
        chart = ASCIIBarChart(config)
        result = chart.create(confidence_values)
        
        assert len(filtered_data) == 3  # Only successful items
        assert isinstance(result, str)
        assert "Success Confidence" in result


if __name__ == '__main__':
    pytest.main([__file__])