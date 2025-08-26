"""
Advanced ASCII-based data visualization components for ESCAI CLI
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import statistics
from datetime import datetime, timedelta


class ChartType(Enum):
    """Chart type enumeration"""
    BAR = "bar"
    LINE = "line"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    SPARKLINE = "sparkline"
    TREE = "tree"


@dataclass
class ChartConfig:
    """Configuration for ASCII charts"""
    width: int = 60
    height: int = 15
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    show_values: bool = False
    show_grid: bool = True
    color_gradient: bool = True
    unicode_chars: bool = True
    color_scheme: str = "default"  # default, dark, light, high_contrast
    show_legend: bool = True
    border_style: str = "rounded"  # rounded, square, double, thick


class ASCIIChart:
    """Base class for ASCII chart generation"""
    
    def __init__(self, config: ChartConfig = None):
        self.config = config or ChartConfig()
        
        # Unicode characters for better visualization
        self.chars = self._get_border_chars()
        self.colors = self._get_color_scheme()
        
    def _get_border_chars(self) -> Dict[str, str]:
        """Get border characters based on style"""
        if not self.config.unicode_chars:
            return {
                'full': '#', 'three_quarters': '#', 'half': '#', 'quarter': '.',
                'light': '.', 'medium': ':', 'dark': '#',
                'horizontal': '-', 'vertical': '|',
                'corner_tl': '+', 'corner_tr': '+', 'corner_bl': '+', 'corner_br': '+',
                'cross': '+', 'tee_down': '+', 'tee_up': '+', 'tee_right': '+', 'tee_left': '+',
            }
        
        if self.config.border_style == "double":
            return {
                'full': '█', 'three_quarters': '▉', 'half': '▌', 'quarter': '▎',
                'light': '░', 'medium': '▒', 'dark': '▓',
                'horizontal': '═', 'vertical': '║',
                'corner_tl': '╔', 'corner_tr': '╗', 'corner_bl': '╚', 'corner_br': '╝',
                'cross': '╬', 'tee_down': '╦', 'tee_up': '╩', 'tee_right': '╠', 'tee_left': '╣',
            }
        elif self.config.border_style == "thick":
            return {
                'full': '█', 'three_quarters': '▉', 'half': '▌', 'quarter': '▎',
                'light': '░', 'medium': '▒', 'dark': '▓',
                'horizontal': '━', 'vertical': '┃',
                'corner_tl': '┏', 'corner_tr': '┓', 'corner_bl': '┗', 'corner_br': '┛',
                'cross': '╋', 'tee_down': '┳', 'tee_up': '┻', 'tee_right': '┣', 'tee_left': '┫',
            }
        else:  # rounded (default)
            return {
                'full': '█', 'three_quarters': '▉', 'half': '▌', 'quarter': '▎',
                'light': '░', 'medium': '▒', 'dark': '▓',
                'horizontal': '─', 'vertical': '│',
                'corner_tl': '╭', 'corner_tr': '╮', 'corner_bl': '╰', 'corner_br': '╯',
                'cross': '┼', 'tee_down': '┬', 'tee_up': '┴', 'tee_right': '├', 'tee_left': '┤',
            }
    
    def _get_color_scheme(self) -> Dict[str, str]:
        """Get color scheme based on configuration"""
        schemes = {
            "default": {
                "primary": "cyan", "secondary": "blue", "accent": "magenta",
                "success": "green", "warning": "yellow", "error": "red",
                "text": "white", "muted": "dim white"
            },
            "dark": {
                "primary": "bright_cyan", "secondary": "bright_blue", "accent": "bright_magenta",
                "success": "bright_green", "warning": "bright_yellow", "error": "bright_red",
                "text": "bright_white", "muted": "white"
            },
            "light": {
                "primary": "blue", "secondary": "cyan", "accent": "purple",
                "success": "green", "warning": "orange3", "error": "red3",
                "text": "black", "muted": "grey50"
            },
            "high_contrast": {
                "primary": "bright_white", "secondary": "bright_yellow", "accent": "bright_cyan",
                "success": "bright_green", "warning": "bright_yellow", "error": "bright_red",
                "text": "bright_white", "muted": "bright_black"
            }
        }
        return schemes.get(self.config.color_scheme, schemes["default"])
    
    def _normalize_data(self, data: List[float]) -> List[float]:
        """Normalize data to 0-1 range"""
        if not data:
            return []
        
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val
        
        if range_val == 0:
            return [0.5] * len(data)
        
        return [(val - min_val) / range_val for val in data]
    
    def _get_intensity_char(self, intensity: float) -> str:
        """Get character based on intensity (0-1)"""
        if intensity >= 0.875:
            return self.chars['full']
        elif intensity >= 0.625:
            return self.chars['three_quarters']
        elif intensity >= 0.375:
            return self.chars['half']
        elif intensity >= 0.125:
            return self.chars['quarter']
        else:
            return ' '
    
    def _create_border(self, content_width: int, content_height: int) -> List[str]:
        """Create border for chart"""
        lines = []
        
        # Top border
        top = (self.chars['corner_tl'] + 
               self.chars['horizontal'] * content_width + 
               self.chars['corner_tr'])
        lines.append(top)
        
        # Content area (will be filled by subclasses)
        for _ in range(content_height):
            lines.append(self.chars['vertical'] + ' ' * content_width + self.chars['vertical'])
        
        # Bottom border
        bottom = (self.chars['corner_bl'] + 
                 self.chars['horizontal'] * content_width + 
                 self.chars['corner_br'])
        lines.append(bottom)
        
        return lines


class ASCIIBarChart(ASCIIChart):
    """ASCII bar chart implementation"""
    
    def create(self, data: List[float], labels: List[str] = None) -> str:
        """Create ASCII bar chart"""
        if not data:
            return "No data available"
        
        labels = labels or [f"Item {i+1}" for i in range(len(data))]
        normalized = self._normalize_data(data)
        
        lines = []
        
        # Title
        if self.config.title:
            title_line = f" {self.config.title} ".center(self.config.width, self.chars['horizontal'])
            lines.append(title_line)
            lines.append("")
        
        # Calculate bar width and spacing
        available_width = self.config.width - 10  # Reserve space for values
        bar_width = max(1, available_width // len(data))
        
        # Create bars
        for row in range(self.config.height - 1, -1, -1):
            line = ""
            threshold = row / (self.config.height - 1)
            
            for i, (value, norm_val) in enumerate(zip(data, normalized)):
                if norm_val >= threshold:
                    line += self.chars['full'] * bar_width
                else:
                    line += ' ' * bar_width
                line += ' '  # Spacing between bars
            
            # Add value scale on the right
            if row == self.config.height - 1:
                max_val = max(data)
                line += f" {max_val:.2f}"
            elif row == 0:
                min_val = min(data)
                line += f" {min_val:.2f}"
            
            lines.append(line)
        
        # Add labels
        label_line = ""
        for i, label in enumerate(labels):
            truncated_label = label[:bar_width] if len(label) > bar_width else label
            label_line += truncated_label.center(bar_width) + ' '
        lines.append(label_line)
        
        return '\n'.join(lines)


class ASCIILineChart(ASCIIChart):
    """ASCII line chart implementation"""
    
    def create(self, data: List[float], x_labels: List[str] = None) -> str:
        """Create ASCII line chart"""
        if not data:
            return "No data available"
        
        normalized = self._normalize_data(data)
        lines = []
        
        # Title
        if self.config.title:
            title_line = f" {self.config.title} ".center(self.config.width, self.chars['horizontal'])
            lines.append(title_line)
            lines.append("")
        
        # Create chart area
        chart_lines = [' ' * self.config.width for _ in range(self.config.height)]
        
        # Plot points and lines
        for i in range(len(normalized)):
            x = int(i * (self.config.width - 1) / max(1, len(normalized) - 1))
            y = int((1 - normalized[i]) * (self.config.height - 1))
            
            # Plot point
            if 0 <= x < self.config.width and 0 <= y < self.config.height:
                chart_lines[y] = chart_lines[y][:x] + '●' + chart_lines[y][x+1:]
            
            # Draw line to next point
            if i < len(normalized) - 1:
                next_x = int((i + 1) * (self.config.width - 1) / max(1, len(normalized) - 1))
                next_y = int((1 - normalized[i + 1]) * (self.config.height - 1))
                
                # Simple line drawing
                steps = max(abs(next_x - x), abs(next_y - y))
                if steps > 0:
                    for step in range(1, steps):
                        line_x = x + int((next_x - x) * step / steps)
                        line_y = y + int((next_y - y) * step / steps)
                        
                        if (0 <= line_x < self.config.width and 
                            0 <= line_y < self.config.height and
                            chart_lines[line_y][line_x] == ' '):
                            chart_lines[line_y] = (chart_lines[line_y][:line_x] + 
                                                 '·' + chart_lines[line_y][line_x+1:])
        
        # Add grid if enabled
        if self.config.show_grid:
            for y in range(0, self.config.height, self.config.height // 4):
                if y < self.config.height:
                    for x in range(self.config.width):
                        if chart_lines[y][x] == ' ':
                            chart_lines[y] = chart_lines[y][:x] + '·' + chart_lines[y][x+1:]
        
        lines.extend(chart_lines)
        
        # Add value indicators
        max_val = max(data)
        min_val = min(data)
        lines.append(f"Min: {min_val:.2f}, Max: {max_val:.2f}")
        
        return '\n'.join(lines)


class ASCIIHistogram(ASCIIChart):
    """ASCII histogram implementation"""
    
    def create(self, data: List[float], bins: int = 10) -> str:
        """Create ASCII histogram"""
        if not data:
            return "No data available"
        
        # Calculate histogram bins
        min_val = min(data)
        max_val = max(data)
        bin_width = (max_val - min_val) / bins
        
        bin_counts = [0] * bins
        bin_labels = []
        
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            bin_labels.append(f"{bin_start:.1f}-{bin_end:.1f}")
            
            # Count values in this bin
            for val in data:
                if bin_start <= val < bin_end or (i == bins - 1 and val == bin_end):
                    bin_counts[i] += 1
        
        # Create bar chart of histogram
        bar_chart = ASCIIBarChart(self.config)
        return bar_chart.create(bin_counts, bin_labels)


class ASCIIScatterPlot(ASCIIChart):
    """ASCII scatter plot implementation"""
    
    def create(self, x_data: List[float], y_data: List[float]) -> str:
        """Create ASCII scatter plot"""
        if not x_data or not y_data or len(x_data) != len(y_data):
            return "Invalid or missing data for scatter plot"
        
        lines = []
        
        # Title
        if self.config.title:
            title_line = f" {self.config.title} ".center(self.config.width, self.chars['horizontal'])
            lines.append(title_line)
            lines.append("")
        
        # Normalize data
        x_normalized = self._normalize_data(x_data)
        y_normalized = self._normalize_data(y_data)
        
        # Create chart area
        chart_lines = [' ' * self.config.width for _ in range(self.config.height)]
        
        # Plot points
        for x_norm, y_norm in zip(x_normalized, y_normalized):
            x = int(x_norm * (self.config.width - 1))
            y = int((1 - y_norm) * (self.config.height - 1))
            
            if 0 <= x < self.config.width and 0 <= y < self.config.height:
                current_char = chart_lines[y][x]
                if current_char == ' ':
                    chart_lines[y] = chart_lines[y][:x] + '●' + chart_lines[y][x+1:]
                elif current_char == '●':
                    chart_lines[y] = chart_lines[y][:x] + '◉' + chart_lines[y][x+1:]  # Multiple points
        
        lines.extend(chart_lines)
        
        # Add axis labels
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        lines.append(f"X: {x_min:.2f} to {x_max:.2f}")
        lines.append(f"Y: {y_min:.2f} to {y_max:.2f}")
        
        return '\n'.join(lines)


class ASCIIHeatmap(ASCIIChart):
    """ASCII heatmap implementation"""
    
    def create(self, data: List[List[float]], row_labels: List[str] = None, 
               col_labels: List[str] = None) -> str:
        """Create ASCII heatmap"""
        if not data or not data[0]:
            return "No data available for heatmap"
        
        rows, cols = len(data), len(data[0])
        row_labels = row_labels or [f"R{i+1}" for i in range(rows)]
        col_labels = col_labels or [f"C{i+1}" for i in range(cols)]
        
        lines = []
        
        # Title
        if self.config.title:
            title_line = f" {self.config.title} ".center(self.config.width, self.chars['horizontal'])
            lines.append(title_line)
            lines.append("")
        
        # Flatten and normalize all data
        flat_data = [val for row in data for val in row]
        min_val, max_val = min(flat_data), max(flat_data)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Calculate cell dimensions
        max_label_width = max(len(label) for label in row_labels)
        cell_width = max(2, (self.config.width - max_label_width - 2) // cols)
        
        # Column headers
        header_line = ' ' * (max_label_width + 1)
        for col_label in col_labels:
            header_line += col_label[:cell_width].center(cell_width)
        lines.append(header_line)
        
        # Data rows
        for i, (row_data, row_label) in enumerate(zip(data, row_labels)):
            line = row_label.ljust(max_label_width) + ' '
            
            for val in row_data:
                # Normalize value and get intensity character
                normalized = (val - min_val) / range_val
                intensity_char = self._get_intensity_char(normalized)
                
                # Fill cell
                cell_content = intensity_char * cell_width
                line += cell_content
            
            lines.append(line)
        
        # Legend
        lines.append("")
        legend_line = f"Legend: {self.chars['light']} Low → {self.chars['full']} High "
        legend_line += f"(Range: {min_val:.2f} - {max_val:.2f})"
        lines.append(legend_line)
        
        return '\n'.join(lines)


class ASCIISparkline:
    """ASCII sparkline implementation for compact trend visualization"""
    
    def __init__(self, unicode_chars: bool = True):
        self.unicode_chars = unicode_chars
        self.spark_chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'] if unicode_chars else ['.', ':', '|', '#']
    
    def create(self, data: List[float], width: int = 20) -> str:
        """Create compact sparkline"""
        if not data:
            return '─' * width
        
        # Normalize data
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Sample data to fit width
        if len(data) > width:
            step = len(data) / width
            sampled_data = [data[int(i * step)] for i in range(width)]
        else:
            sampled_data = data + [data[-1]] * (width - len(data))
        
        # Create sparkline
        sparkline = ""
        for val in sampled_data:
            normalized = (val - min_val) / range_val
            char_index = min(len(self.spark_chars) - 1, int(normalized * len(self.spark_chars)))
            sparkline += self.spark_chars[char_index]
        
        return sparkline


class ASCIIProgressBar:
    """Advanced ASCII progress bar with detailed status"""
    
    def __init__(self, width: int = 40, unicode_chars: bool = True):
        self.width = width
        self.unicode_chars = unicode_chars
        self.chars = {
            'full': '█' if unicode_chars else '#',
            'empty': '░' if unicode_chars else '-',
            'partial': ['▏', '▎', '▍', '▌', '▋', '▊', '▉'] if unicode_chars else ['.']
        }
    
    def create(self, progress: float, status: str = "", eta: Optional[timedelta] = None,
               rate: Optional[float] = None) -> str:
        """Create detailed progress bar"""
        # Clamp progress to 0-1
        progress = max(0, min(1, progress))
        
        # Calculate filled portion
        filled_width = progress * self.width
        full_blocks = int(filled_width)
        partial_block = filled_width - full_blocks
        
        # Build progress bar
        bar = self.chars['full'] * full_blocks
        
        # Add partial block if needed
        if partial_block > 0 and full_blocks < self.width:
            if self.unicode_chars:
                partial_index = min(len(self.chars['partial']) - 1, 
                                  int(partial_block * len(self.chars['partial'])))
                bar += self.chars['partial'][partial_index]
                full_blocks += 1
            else:
                bar += self.chars['partial'][0]
                full_blocks += 1
        
        # Fill remaining with empty blocks
        bar += self.chars['empty'] * (self.width - full_blocks)
        
        # Add percentage
        percentage = f"{progress * 100:5.1f}%"
        
        # Build complete line
        result = f"[{bar}] {percentage}"
        
        if status:
            result += f" {status}"
        
        if eta:
            eta_str = str(eta).split('.')[0]  # Remove microseconds
            result += f" ETA: {eta_str}"
        
        if rate:
            result += f" ({rate:.1f}/s)"
        
        return result


class ASCIITreeView:
    """ASCII tree visualization for hierarchical data"""
    
    def __init__(self, unicode_chars: bool = True):
        self.unicode_chars = unicode_chars
        self.chars = {
            'branch': '├── ' if unicode_chars else '|-- ',
            'last_branch': '└── ' if unicode_chars else '`-- ',
            'vertical': '│   ' if unicode_chars else '|   ',
            'space': '    '
        }
    
    def create(self, tree_data: Dict[str, Any], max_depth: int = 10) -> str:
        """Create ASCII tree from hierarchical data"""
        lines = []
        self._render_node(tree_data, lines, "", True, 0, max_depth)
        return '\n'.join(lines)
    
    def _render_node(self, node: Dict[str, Any], lines: List[str], prefix: str, 
                    is_last: bool, depth: int, max_depth: int):
        """Recursively render tree nodes"""
        if depth > max_depth:
            return
        
        # Get node info
        name = node.get('name', 'Unknown')
        value = node.get('value', '')
        children = node.get('children', [])
        
        # Create node line
        if depth == 0:
            node_line = name
        else:
            connector = self.chars['last_branch'] if is_last else self.chars['branch']
            node_line = prefix + connector + name
        
        if value:
            node_line += f" ({value})"
        
        lines.append(node_line)
        
        # Render children
        if children and depth < max_depth:
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                
                if depth == 0:
                    child_prefix = ""
                else:
                    extension = self.chars['space'] if is_last else self.chars['vertical']
                    child_prefix = prefix + extension
                
                self._render_node(child, lines, child_prefix, is_last_child, depth + 1, max_depth)


# Utility functions for common visualizations

def create_epistemic_state_chart(epistemic_data: Dict[str, Any]) -> str:
    """Create visualization for epistemic state data"""
    lines = []
    
    # Beliefs confidence chart
    beliefs = epistemic_data.get('beliefs', [])
    if beliefs:
        confidences = [b.get('confidence', 0) for b in beliefs[:10]]  # Top 10
        labels = [b.get('content', '')[:15] + '...' if len(b.get('content', '')) > 15 
                 else b.get('content', '') for b in beliefs[:10]]
        
        config = ChartConfig(width=60, height=8, title="Belief Confidences")
        chart = ASCIIBarChart(config)
        lines.append(chart.create(confidences, labels))
        lines.append("")
    
    # Uncertainty sparkline
    uncertainty_history = epistemic_data.get('uncertainty_history', [])
    if uncertainty_history:
        sparkline = ASCIISparkline()
        spark = sparkline.create(uncertainty_history, 40)
        lines.append(f"Uncertainty Trend: {spark} ({uncertainty_history[-1]:.2f})")
        lines.append("")
    
    return '\n'.join(lines)


def create_pattern_frequency_heatmap(pattern_data: List[Dict[str, Any]]) -> str:
    """Create heatmap for behavioral pattern frequencies"""
    if not pattern_data:
        return "No pattern data available"
    
    # Extract pattern names and time periods
    patterns = list(set(p.get('pattern_name', 'Unknown') for p in pattern_data))
    time_periods = list(set(p.get('time_period', 'Unknown') for p in pattern_data))
    
    # Create frequency matrix
    freq_matrix = []
    for pattern in patterns:
        row = []
        for period in time_periods:
            # Find frequency for this pattern-period combination
            freq = 0
            for p in pattern_data:
                if (p.get('pattern_name') == pattern and 
                    p.get('time_period') == period):
                    freq = p.get('frequency', 0)
                    break
            row.append(freq)
        freq_matrix.append(row)
    
    config = ChartConfig(width=80, height=len(patterns) + 5, 
                        title="Pattern Frequency Heatmap")
    heatmap = ASCIIHeatmap(config)
    return heatmap.create(freq_matrix, patterns, time_periods)


def create_causal_strength_scatter(causal_data: List[Dict[str, Any]]) -> str:
    """Create scatter plot of causal relationship strength vs confidence"""
    if not causal_data:
        return "No causal data available"
    
    strengths = [c.get('strength', 0) for c in causal_data]
    confidences = [c.get('confidence', 0) for c in causal_data]
    
    config = ChartConfig(width=50, height=15, 
                        title="Causal Relationships: Strength vs Confidence")
    scatter = ASCIIScatterPlot(config)
    return scatter.create(strengths, confidences)