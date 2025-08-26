"""
Unit tests for ASCII visualization components
"""

import pytest
from datetime import timedelta
from escai_framework.cli.utils.ascii_viz import (
    ASCIIChart, ASCIIBarChart, ASCIILineChart, ASCIIHistogram,
    ASCIIScatterPlot, ASCIIHeatmap, ASCIISparkline, ASCIIProgressBar,
    ASCIITreeView, ChartConfig, ChartType,
    create_epistemic_state_chart, create_pattern_frequency_heatmap,
    create_causal_strength_scatter
)


class TestChartConfig:
    """Test chart configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ChartConfig()
        assert config.width == 60
        assert config.height == 15
        assert config.title == ""
        assert config.show_values is False
        assert config.show_grid is True
        assert config.unicode_chars is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ChartConfig(
            width=80,
            height=20,
            title="Test Chart",
            show_values=True,
            unicode_chars=False
        )
        assert config.width == 80
        assert config.height == 20
        assert config.title == "Test Chart"
        assert config.show_values is True
        assert config.unicode_chars is False


class TestASCIIChart:
    """Test base ASCII chart functionality"""
    
    def test_normalize_data(self):
        """Test data normalization"""
        chart = ASCIIChart()
        
        # Normal case
        data = [1, 2, 3, 4, 5]
        normalized = chart._normalize_data(data)
        assert normalized == [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Single value
        data = [5]
        normalized = chart._normalize_data(data)
        assert normalized == [0.5]
        
        # All same values
        data = [3, 3, 3, 3]
        normalized = chart._normalize_data(data)
        assert all(val == 0.5 for val in normalized)
        
        # Empty data
        data = []
        normalized = chart._normalize_data(data)
        assert normalized == []
    
    def test_get_intensity_char(self):
        """Test intensity character selection"""
        chart = ASCIIChart()
        
        # Test different intensity levels
        assert chart._get_intensity_char(1.0) == '█'
        assert chart._get_intensity_char(0.9) == '█'
        assert chart._get_intensity_char(0.7) == '▉'
        assert chart._get_intensity_char(0.5) == '▌'
        assert chart._get_intensity_char(0.2) == '▎'
        assert chart._get_intensity_char(0.0) == ' '
    
    def test_unicode_disabled(self):
        """Test chart with unicode disabled"""
        config = ChartConfig(unicode_chars=False)
        chart = ASCIIChart(config)
        
        assert chart.chars['full'] == '#'
        assert chart.chars['horizontal'] == '-'
        assert chart.chars['vertical'] == '|'


class TestASCIIBarChart:
    """Test ASCII bar chart"""
    
    def test_create_basic_bar_chart(self):
        """Test basic bar chart creation"""
        chart = ASCIIBarChart()
        data = [1, 3, 2, 4, 2]
        result = chart.create(data)
        
        assert isinstance(result, str)
        assert len(result.split('\n')) > 5  # Should have multiple lines
        assert '█' in result or '#' in result  # Should contain bar characters
    
    def test_create_with_labels(self):
        """Test bar chart with custom labels"""
        chart = ASCIIBarChart()
        data = [10, 20, 15]
        labels = ["A", "B", "C"]
        result = chart.create(data, labels)
        
        assert "A" in result
        assert "B" in result
        assert "C" in result
    
    def test_create_with_title(self):
        """Test bar chart with title"""
        config = ChartConfig(title="Test Chart")
        chart = ASCIIBarChart(config)
        data = [1, 2, 3]
        result = chart.create(data)
        
        assert "Test Chart" in result
    
    def test_empty_data(self):
        """Test bar chart with empty data"""
        chart = ASCIIBarChart()
        result = chart.create([])
        
        assert result == "No data available"
    
    def test_single_value(self):
        """Test bar chart with single value"""
        chart = ASCIIBarChart()
        result = chart.create([5])
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestASCIILineChart:
    """Test ASCII line chart"""
    
    def test_create_basic_line_chart(self):
        """Test basic line chart creation"""
        chart = ASCIILineChart()
        data = [1, 3, 2, 4, 2, 1]
        result = chart.create(data)
        
        assert isinstance(result, str)
        assert '●' in result  # Should contain point markers
        assert 'Min:' in result and 'Max:' in result
    
    def test_create_with_title(self):
        """Test line chart with title"""
        config = ChartConfig(title="Trend Analysis")
        chart = ASCIILineChart(config)
        data = [1, 2, 3, 2, 1]
        result = chart.create(data)
        
        assert "Trend Analysis" in result
    
    def test_create_with_grid(self):
        """Test line chart with grid enabled"""
        config = ChartConfig(show_grid=True)
        chart = ASCIILineChart(config)
        data = [1, 2, 3, 2, 1]
        result = chart.create(data)
        
        assert '·' in result  # Grid dots
    
    def test_empty_data(self):
        """Test line chart with empty data"""
        chart = ASCIILineChart()
        result = chart.create([])
        
        assert result == "No data available"


class TestASCIIHistogram:
    """Test ASCII histogram"""
    
    def test_create_basic_histogram(self):
        """Test basic histogram creation"""
        chart = ASCIIHistogram()
        data = [1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5]
        result = chart.create(data)
        
        assert isinstance(result, str)
        assert '█' in result or '#' in result
    
    def test_create_with_custom_bins(self):
        """Test histogram with custom bin count"""
        chart = ASCIIHistogram()
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = chart.create(data, bins=5)
        
        assert isinstance(result, str)
        # Should have 5 bins represented
    
    def test_empty_data(self):
        """Test histogram with empty data"""
        chart = ASCIIHistogram()
        result = chart.create([])
        
        assert result == "No data available"


class TestASCIIScatterPlot:
    """Test ASCII scatter plot"""
    
    def test_create_basic_scatter(self):
        """Test basic scatter plot creation"""
        chart = ASCIIScatterPlot()
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 1, 5, 3]
        result = chart.create(x_data, y_data)
        
        assert isinstance(result, str)
        assert '●' in result  # Should contain point markers
        assert 'X:' in result and 'Y:' in result
    
    def test_overlapping_points(self):
        """Test scatter plot with overlapping points"""
        chart = ASCIIScatterPlot()
        x_data = [1, 1, 2, 2]
        y_data = [1, 1, 2, 2]
        result = chart.create(x_data, y_data)
        
        assert '◉' in result  # Should show overlapping points
    
    def test_mismatched_data(self):
        """Test scatter plot with mismatched data lengths"""
        chart = ASCIIScatterPlot()
        x_data = [1, 2, 3]
        y_data = [1, 2]  # Different length
        result = chart.create(x_data, y_data)
        
        assert "Invalid or missing data" in result
    
    def test_empty_data(self):
        """Test scatter plot with empty data"""
        chart = ASCIIScatterPlot()
        result = chart.create([], [])
        
        assert "Invalid or missing data" in result


class TestASCIIHeatmap:
    """Test ASCII heatmap"""
    
    def test_create_basic_heatmap(self):
        """Test basic heatmap creation"""
        chart = ASCIIHeatmap()
        data = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        result = chart.create(data)
        
        assert isinstance(result, str)
        assert 'Legend:' in result
        assert 'Range:' in result
    
    def test_create_with_labels(self):
        """Test heatmap with custom labels"""
        chart = ASCIIHeatmap()
        data = [[1, 2], [3, 4]]
        row_labels = ["Row1", "Row2"]
        col_labels = ["Col1", "Col2"]
        result = chart.create(data, row_labels, col_labels)
        
        assert "Row1" in result
        assert "Row2" in result
        assert "Col1" in result
        assert "Col2" in result
    
    def test_empty_data(self):
        """Test heatmap with empty data"""
        chart = ASCIIHeatmap()
        result = chart.create([])
        
        assert "No data available" in result
    
    def test_single_cell(self):
        """Test heatmap with single cell"""
        chart = ASCIIHeatmap()
        data = [[5]]
        result = chart.create(data)
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestASCIISparkline:
    """Test ASCII sparkline"""
    
    def test_create_basic_sparkline(self):
        """Test basic sparkline creation"""
        sparkline = ASCIISparkline()
        data = [1, 3, 2, 5, 4, 2, 1]
        result = sparkline.create(data)
        
        assert isinstance(result, str)
        assert len(result) == 20  # Default width
        assert any(c in result for c in ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'])
    
    def test_create_custom_width(self):
        """Test sparkline with custom width"""
        sparkline = ASCIISparkline()
        data = [1, 2, 3, 4, 5]
        result = sparkline.create(data, width=10)
        
        assert len(result) == 10
    
    def test_empty_data(self):
        """Test sparkline with empty data"""
        sparkline = ASCIISparkline()
        result = sparkline.create([])
        
        assert result == '─' * 20  # Default width with dashes
    
    def test_unicode_disabled(self):
        """Test sparkline with unicode disabled"""
        sparkline = ASCIISparkline(unicode_chars=False)
        data = [1, 2, 3, 4, 5]
        result = sparkline.create(data)
        
        assert any(c in result for c in ['.', ':', '|', '#'])


class TestASCIIProgressBar:
    """Test ASCII progress bar"""
    
    def test_create_basic_progress(self):
        """Test basic progress bar creation"""
        progress_bar = ASCIIProgressBar()
        result = progress_bar.create(0.5)
        
        assert isinstance(result, str)
        assert '[' in result and ']' in result
        assert '50.0%' in result
        assert '█' in result or '#' in result
    
    def test_create_with_status(self):
        """Test progress bar with status message"""
        progress_bar = ASCIIProgressBar()
        result = progress_bar.create(0.75, status="Processing...")
        
        assert "Processing..." in result
        assert '75.0%' in result
    
    def test_create_with_eta(self):
        """Test progress bar with ETA"""
        progress_bar = ASCIIProgressBar()
        eta = timedelta(minutes=5, seconds=30)
        result = progress_bar.create(0.3, eta=eta)
        
        assert "ETA:" in result
        assert "5:30" in result
    
    def test_create_with_rate(self):
        """Test progress bar with rate"""
        progress_bar = ASCIIProgressBar()
        result = progress_bar.create(0.6, rate=15.5)
        
        assert "(15.5/s)" in result
    
    def test_progress_bounds(self):
        """Test progress bar with out-of-bounds values"""
        progress_bar = ASCIIProgressBar()
        
        # Test negative progress
        result = progress_bar.create(-0.1)
        assert '0.0%' in result
        
        # Test progress > 1
        result = progress_bar.create(1.5)
        assert '100.0%' in result
    
    def test_unicode_disabled(self):
        """Test progress bar with unicode disabled"""
        progress_bar = ASCIIProgressBar(unicode_chars=False)
        result = progress_bar.create(0.5)
        
        assert '#' in result
        assert '-' in result


class TestASCIITreeView:
    """Test ASCII tree view"""
    
    def test_create_basic_tree(self):
        """Test basic tree creation"""
        tree_view = ASCIITreeView()
        tree_data = {
            'name': 'Root',
            'children': [
                {'name': 'Child1', 'value': '10'},
                {'name': 'Child2', 'children': [
                    {'name': 'Grandchild1', 'value': '5'}
                ]}
            ]
        }
        result = tree_view.create(tree_data)
        
        assert isinstance(result, str)
        assert 'Root' in result
        assert 'Child1' in result
        assert 'Child2' in result
        assert 'Grandchild1' in result
        assert '├──' in result or '|--' in result
    
    def test_create_with_max_depth(self):
        """Test tree with depth limit"""
        tree_view = ASCIITreeView()
        tree_data = {
            'name': 'Root',
            'children': [
                {'name': 'Level1', 'children': [
                    {'name': 'Level2', 'children': [
                        {'name': 'Level3'}
                    ]}
                ]}
            ]
        }
        result = tree_view.create(tree_data, max_depth=2)
        
        assert 'Root' in result
        assert 'Level1' in result
        assert 'Level2' in result
        # Level3 should not appear due to depth limit
    
    def test_unicode_disabled(self):
        """Test tree with unicode disabled"""
        tree_view = ASCIITreeView(unicode_chars=False)
        tree_data = {
            'name': 'Root',
            'children': [{'name': 'Child'}]
        }
        result = tree_view.create(tree_data)
        
        assert '|--' in result or '`--' in result


class TestUtilityFunctions:
    """Test utility functions for common visualizations"""
    
    def test_create_epistemic_state_chart(self):
        """Test epistemic state chart creation"""
        epistemic_data = {
            'beliefs': [
                {'content': 'Belief 1', 'confidence': 0.8},
                {'content': 'Belief 2', 'confidence': 0.6}
            ],
            'uncertainty_history': [0.1, 0.2, 0.15, 0.3, 0.25]
        }
        result = create_epistemic_state_chart(epistemic_data)
        
        assert isinstance(result, str)
        assert 'Belief Confidences' in result
        assert 'Uncertainty Trend:' in result
    
    def test_create_pattern_frequency_heatmap(self):
        """Test pattern frequency heatmap creation"""
        pattern_data = [
            {'pattern_name': 'Pattern1', 'time_period': 'Morning', 'frequency': 10},
            {'pattern_name': 'Pattern1', 'time_period': 'Evening', 'frequency': 5},
            {'pattern_name': 'Pattern2', 'time_period': 'Morning', 'frequency': 8},
            {'pattern_name': 'Pattern2', 'time_period': 'Evening', 'frequency': 12}
        ]
        result = create_pattern_frequency_heatmap(pattern_data)
        
        assert isinstance(result, str)
        assert 'Pattern Frequency Heatmap' in result
        assert 'Pattern1' in result
        assert 'Pattern2' in result
    
    def test_create_causal_strength_scatter(self):
        """Test causal strength scatter plot creation"""
        causal_data = [
            {'strength': 0.8, 'confidence': 0.9},
            {'strength': 0.6, 'confidence': 0.7},
            {'strength': 0.4, 'confidence': 0.5}
        ]
        result = create_causal_strength_scatter(causal_data)
        
        assert isinstance(result, str)
        assert 'Causal Relationships: Strength vs Confidence' in result
    
    def test_empty_utility_functions(self):
        """Test utility functions with empty data"""
        # Empty epistemic data
        result = create_epistemic_state_chart({})
        assert isinstance(result, str)
        
        # Empty pattern data
        result = create_pattern_frequency_heatmap([])
        assert "No pattern data available" in result
        
        # Empty causal data
        result = create_causal_strength_scatter([])
        assert "No causal data available" in result


class TestChartIntegration:
    """Test integration between different chart types"""
    
    def test_consistent_styling(self):
        """Test that all charts use consistent styling"""
        config = ChartConfig(unicode_chars=False)
        
        bar_chart = ASCIIBarChart(config)
        line_chart = ASCIILineChart(config)
        histogram = ASCIIHistogram(config)
        
        data = [1, 2, 3, 4, 5]
        
        bar_result = bar_chart.create(data)
        line_result = line_chart.create(data)
        hist_result = histogram.create(data)
        
        # All should use ASCII characters when unicode is disabled
        assert '█' not in bar_result
        assert '█' not in line_result
        assert '█' not in hist_result
    
    def test_chart_dimensions(self):
        """Test that charts respect dimension settings"""
        config = ChartConfig(width=40, height=10)
        
        bar_chart = ASCIIBarChart(config)
        line_chart = ASCIILineChart(config)
        
        data = [1, 2, 3, 4, 5]
        
        bar_result = bar_chart.create(data)
        line_result = line_chart.create(data)
        
        # Check that results don't exceed specified dimensions
        bar_lines = bar_result.split('\n')
        line_lines = line_result.split('\n')
        
        # Most lines should be within width limits (allowing for labels)
        for line in bar_lines[:-2]:  # Exclude label lines
            if line.strip():  # Skip empty lines
                assert len(line) <= config.width + 20  # Some tolerance for labels
        
        for line in line_lines[:-2]:  # Exclude summary lines
            if line.strip():  # Skip empty lines
                assert len(line) <= config.width + 20  # Some tolerance for labels


if __name__ == '__main__':
    pytest.main([__file__])