"""
Integration tests for CLI visualization and reporting enhancements
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from escai_framework.cli.main import cli
from escai_framework.cli.utils.console import get_console, set_color_scheme
from escai_framework.cli.utils.data_filters import DataFilter, SearchQuery, FilterCondition, FilterOperator
from escai_framework.cli.utils.reporting import DataExporter, ReportFormat


class TestCLIIntegration:
    """Test CLI command integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.console = get_console()
    
    def test_cli_main_help(self):
        """Test main CLI help command"""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "ESCAI Framework" in result.output
        assert "monitor" in result.output
        assert "analyze" in result.output
        assert "config" in result.output
    
    def test_cli_version(self):
        """Test CLI version command"""
        result = self.runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert "ESCAI Framework" in result.output
    
    @patch('escai_framework.cli.utils.logo.display_logo')
    def test_cli_no_command(self, mock_logo):
        """Test CLI with no command shows welcome"""
        result = self.runner.invoke(cli, [])
        
        assert result.exit_code == 0
        mock_logo.assert_called_once()


class TestAnalyzeCommands:
    """Test analyze command group"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
    
    def test_analyze_help(self):
        """Test analyze command help"""
        result = self.runner.invoke(cli, ['analyze', '--help'])
        
        assert result.exit_code == 0
        assert "Analysis and exploration commands" in result.output
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_patterns(self, mock_console):
        """Test analyze patterns command"""
        result = self.runner.invoke(cli, ['analyze', 'patterns'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_patterns_interactive(self, mock_console):
        """Test analyze patterns with interactive flag"""
        with patch('escai_framework.cli.commands.analyze.Prompt.ask') as mock_prompt:
            mock_prompt.return_value = "Exit"
            
            result = self.runner.invoke(cli, ['analyze', 'patterns', '--interactive'])
            
            assert result.exit_code == 0
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_causal(self, mock_console):
        """Test analyze causal command"""
        result = self.runner.invoke(cli, ['analyze', 'causal'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_predictions(self, mock_console):
        """Test analyze predictions command"""
        result = self.runner.invoke(cli, ['analyze', 'predictions'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_visualize(self, mock_console):
        """Test analyze visualize command"""
        result = self.runner.invoke(cli, ['analyze', 'visualize', '--chart-type', 'bar'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_epistemic(self, mock_console):
        """Test analyze epistemic command"""
        result = self.runner.invoke(cli, ['analyze', 'epistemic'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_heatmap(self, mock_console):
        """Test analyze heatmap command"""
        result = self.runner.invoke(cli, ['analyze', 'heatmap'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_search_help(self, mock_console):
        """Test analyze search command without parameters shows help"""
        result = self.runner.invoke(cli, ['analyze', 'search'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_search_with_query(self, mock_console):
        """Test analyze search command with query"""
        result = self.runner.invoke(cli, ['analyze', 'search', '--query', 'test'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_filter_examples(self, mock_console):
        """Test analyze filter command shows examples"""
        result = self.runner.invoke(cli, ['analyze', 'filter'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_export(self, mock_console):
        """Test analyze export command"""
        result = self.runner.invoke(cli, ['analyze', 'export', '--format', 'json'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_timeline(self, mock_console):
        """Test analyze timeline command"""
        result = self.runner.invoke(cli, ['analyze', 'timeline'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.analyze.console')
    def test_analyze_health(self, mock_console):
        """Test analyze health command"""
        result = self.runner.invoke(cli, ['analyze', 'health'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()


class TestConfigCommands:
    """Test config command group"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
    
    def test_config_help(self):
        """Test config command help"""
        result = self.runner.invoke(cli, ['config', '--help'])
        
        assert result.exit_code == 0
        assert "Configuration management commands" in result.output
    
    @patch('escai_framework.cli.commands.config.console')
    def test_config_theme_list(self, mock_console):
        """Test config theme list command"""
        result = self.runner.invoke(cli, ['config', 'theme', '--list'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.config.console')
    def test_config_theme_preview(self, mock_console):
        """Test config theme preview command"""
        result = self.runner.invoke(cli, ['config', 'theme', '--preview'])
        
        assert result.exit_code == 0
    
    @patch('escai_framework.cli.commands.config.console')
    def test_config_check(self, mock_console):
        """Test config check command"""
        result = self.runner.invoke(cli, ['config', 'check'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    def test_config_theme_set_valid(self):
        """Test setting valid color scheme"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.json'
            
            with patch('escai_framework.cli.commands.config.CONFIG_FILE', config_file):
                with patch('escai_framework.cli.commands.config.CONFIG_DIR', Path(temp_dir)):
                    result = self.runner.invoke(cli, ['config', 'theme', '--scheme', 'dark'])
                    
                    assert result.exit_code == 0
                    assert config_file.exists()
                    
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    assert config['ui']['color_scheme'] == 'dark'
    
    def test_config_theme_set_invalid(self):
        """Test setting invalid color scheme"""
        result = self.runner.invoke(cli, ['config', 'theme', '--scheme', 'invalid'])
        
        assert result.exit_code == 0  # Command succeeds but shows error


class TestMonitorCommands:
    """Test monitor command group"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
    
    def test_monitor_help(self):
        """Test monitor command help"""
        result = self.runner.invoke(cli, ['monitor', '--help'])
        
        assert result.exit_code == 0
        assert "Real-time monitoring commands" in result.output
    
    @patch('escai_framework.cli.commands.monitor.console')
    def test_monitor_start(self, mock_console):
        """Test monitor start command"""
        result = self.runner.invoke(cli, [
            'monitor', 'start',
            '--agent-id', 'test_agent',
            '--framework', 'langchain'
        ])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()
    
    @patch('escai_framework.cli.commands.monitor.console')
    def test_monitor_stop(self, mock_console):
        """Test monitor stop command"""
        result = self.runner.invoke(cli, ['monitor', 'stop', '--all'])
        
        assert result.exit_code == 0
        mock_console.print.assert_called()


class TestDataFilteringIntegration:
    """Test data filtering integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.filter_engine = DataFilter()
        self.test_data = [
            {
                'id': 'agent_001',
                'name': 'Processing Agent',
                'status': 'active',
                'confidence': 0.9,
                'metadata': {'priority': 'high', 'tags': ['urgent']}
            },
            {
                'id': 'agent_002',
                'name': 'Analysis Agent',
                'status': 'idle',
                'confidence': 0.7,
                'metadata': {'priority': 'medium', 'tags': ['normal']}
            },
            {
                'id': 'agent_003',
                'name': 'Monitor Agent',
                'status': 'error',
                'confidence': 0.4,
                'metadata': {'priority': 'low', 'tags': ['background']}
            }
        ]
    
    def test_complex_filter_query(self):
        """Test complex filter with multiple conditions"""
        query = SearchQuery(
            conditions=[
                FilterCondition("status", FilterOperator.NOT_EQUALS, "error"),
                FilterCondition("confidence", FilterOperator.GREATER_THAN, 0.6),
                FilterCondition("metadata.priority", FilterOperator.IN, ["high", "medium"])
            ],
            logic="AND",
            sort_by="confidence",
            sort_desc=True
        )
        
        result = self.filter_engine.apply_filter(self.test_data, query)
        
        assert len(result) == 2
        assert result[0]['confidence'] == 0.9  # Sorted descending
        assert result[1]['confidence'] == 0.7
    
    def test_search_and_filter_combination(self):
        """Test combining search and filtering"""
        # First, do a quick search
        search_results = self.filter_engine.quick_search(self.test_data, "agent")
        
        # Then apply filter to search results
        query = SearchQuery(
            conditions=[FilterCondition("status", FilterOperator.EQUALS, "active")]
        )
        
        filtered_results = self.filter_engine.apply_filter(search_results, query)
        
        assert len(filtered_results) == 1
        assert filtered_results[0]['id'] == 'agent_001'
    
    def test_save_load_query_workflow(self):
        """Test complete save/load query workflow"""
        # Create and save a query
        query = SearchQuery(
            conditions=[
                FilterCondition("status", FilterOperator.EQUALS, "active"),
                FilterCondition("confidence", FilterOperator.GREATER_THAN, 0.8)
            ],
            logic="AND"
        )
        
        self.filter_engine.save_query("high_confidence_active", query)
        
        # Load and apply the query
        loaded_query = self.filter_engine.load_query("high_confidence_active")
        result = self.filter_engine.apply_filter(self.test_data, loaded_query)
        
        assert len(result) == 1
        assert result[0]['id'] == 'agent_001'
    
    def test_fuzzy_search_integration(self):
        """Test fuzzy search integration"""
        # Test fuzzy matching on names
        results = self.filter_engine.fuzzy_search(
            self.test_data, 
            "process", 
            "name", 
            threshold=0.3
        )
        
        assert len(results) >= 1
        assert any("Processing" in item['name'] for item in results)


class TestVisualizationIntegration:
    """Test visualization component integration"""
    
    def test_chart_creation_pipeline(self):
        """Test complete chart creation pipeline"""
        from escai_framework.cli.utils.ascii_viz import ASCIIBarChart, ChartConfig
        
        # Simulate data processing pipeline
        raw_data = [
            {'agent': 'A', 'success_rate': 0.9},
            {'agent': 'B', 'success_rate': 0.7},
            {'agent': 'C', 'success_rate': 0.8}
        ]
        
        # Extract data for visualization
        values = [item['success_rate'] for item in raw_data]
        labels = [item['agent'] for item in raw_data]
        
        # Create chart
        config = ChartConfig(
            width=50,
            height=10,
            title="Agent Success Rates",
            color_scheme="default"
        )
        
        chart = ASCIIBarChart(config)
        result = chart.create(values, labels)
        
        assert isinstance(result, str)
        assert "Agent Success Rates" in result
        assert len(result.split('\n')) > 5
    
    def test_multiple_chart_types(self):
        """Test creating multiple chart types with same data"""
        from escai_framework.cli.utils.ascii_viz import (
            ASCIIBarChart, ASCIILineChart, ASCIIHistogram, ChartConfig
        )
        
        data = [1, 3, 2, 5, 4, 6, 5, 7, 6, 8]
        config = ChartConfig(width=40, height=8, title="Test Data")
        
        # Test different chart types
        charts = [
            ASCIIBarChart(config),
            ASCIILineChart(config),
            ASCIIHistogram(config)
        ]
        
        for chart in charts:
            if isinstance(chart, ASCIIHistogram):
                result = chart.create(data, bins=5)
            else:
                result = chart.create(data)
            
            assert isinstance(result, str)
            assert "Test Data" in result


class TestExportIntegration:
    """Test data export integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.console = get_console()
        self.exporter = DataExporter(self.console)
        
        self.test_data = {
            'agents': [
                {'id': 'agent_001', 'status': 'active'},
                {'id': 'agent_002', 'status': 'idle'}
            ],
            'summary': {
                'total_agents': 2,
                'active_agents': 1
            }
        }
    
    def test_export_all_formats(self):
        """Test exporting data in all supported formats"""
        formats = [ReportFormat.JSON, ReportFormat.CSV, ReportFormat.MARKDOWN, ReportFormat.TXT]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in formats:
                output_path = Path(temp_dir) / f"test_export_{fmt.value}"
                
                result_path = self.exporter.export_data(self.test_data, fmt, output_path)
                
                assert result_path.exists()
                assert result_path.suffix == f'.{fmt.value}'
                assert result_path.stat().st_size > 0
    
    def test_export_with_filtering(self):
        """Test export after filtering data"""
        # Create filter
        filter_engine = DataFilter()
        
        raw_data = [
            {'id': 'agent_001', 'status': 'active', 'confidence': 0.9},
            {'id': 'agent_002', 'status': 'idle', 'confidence': 0.7},
            {'id': 'agent_003', 'status': 'error', 'confidence': 0.3}
        ]
        
        # Apply filter
        query = SearchQuery(
            conditions=[FilterCondition("status", FilterOperator.NOT_EQUALS, "error")]
        )
        
        filtered_data = filter_engine.apply_filter(raw_data, query)
        
        # Export filtered data
        export_data = {
            'filtered_agents': filtered_data,
            'filter_query': {
                'conditions': len(query.conditions),
                'logic': query.logic
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "filtered_export"
            
            result_path = self.exporter.export_data(
                export_data, 
                ReportFormat.JSON, 
                output_path
            )
            
            assert result_path.exists()
            
            # Verify exported content
            with open(result_path, 'r') as f:
                exported = json.load(f)
            
            assert len(exported['filtered_agents']) == 2
            assert exported['filter_query']['conditions'] == 1


class TestColorSchemeIntegration:
    """Test color scheme integration across components"""
    
    def test_color_scheme_persistence(self):
        """Test color scheme persistence across sessions"""
        # Set a color scheme
        result = set_color_scheme("dark")
        assert result is True
        
        # Get console with the scheme
        console = get_console()
        assert console is not None
        
        # Verify theme is applied (basic check)
        assert hasattr(console, 'theme')
    
    def test_color_scheme_in_charts(self):
        """Test color schemes work with chart components"""
        from escai_framework.cli.utils.ascii_viz import ASCIIBarChart, ChartConfig
        
        schemes = ["default", "dark", "light"]
        data = [1, 2, 3, 4, 5]
        
        for scheme in schemes:
            config = ChartConfig(
                width=30,
                height=6,
                title=f"Test {scheme}",
                color_scheme=scheme
            )
            
            chart = ASCIIBarChart(config)
            result = chart.create(data)
            
            assert isinstance(result, str)
            assert f"Test {scheme}" in result


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""
    
    def test_analyze_filter_export_workflow(self):
        """Test complete analyze -> filter -> export workflow"""
        runner = CliRunner()
        
        # Test the search command (simulates data analysis)
        with patch('escai_framework.cli.commands.analyze.console') as mock_console:
            result = runner.invoke(cli, [
                'analyze', 'search', 
                '--query', 'active',
                '--field', 'status'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()
        
        # Test the export command (simulates data export)
        with patch('escai_framework.cli.commands.analyze.console') as mock_console:
            result = runner.invoke(cli, [
                'analyze', 'export',
                '--format', 'json',
                '--timeframe', '1h'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()
    
    def test_config_theme_analyze_workflow(self):
        """Test config theme -> analyze workflow"""
        runner = CliRunner()
        
        # Set theme
        with patch('escai_framework.cli.commands.config.console') as mock_console:
            result = runner.invoke(cli, [
                'config', 'theme',
                '--scheme', 'dark'
            ])
            
            assert result.exit_code == 0
        
        # Use analyze command with new theme
        with patch('escai_framework.cli.commands.analyze.console') as mock_console:
            result = runner.invoke(cli, [
                'analyze', 'visualize',
                '--chart-type', 'bar',
                '--metric', 'confidence'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()
    
    def test_monitor_analyze_workflow(self):
        """Test monitor -> analyze workflow"""
        runner = CliRunner()
        
        # Start monitoring
        with patch('escai_framework.cli.commands.monitor.console') as mock_console:
            result = runner.invoke(cli, [
                'monitor', 'start',
                '--agent-id', 'test_agent',
                '--framework', 'langchain'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()
        
        # Analyze patterns
        with patch('escai_framework.cli.commands.analyze.console') as mock_console:
            result = runner.invoke(cli, [
                'analyze', 'patterns',
                '--agent-id', 'test_agent'
            ])
            
            assert result.exit_code == 0
            mock_console.print.assert_called()


if __name__ == '__main__':
    pytest.main([__file__])