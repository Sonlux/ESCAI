"""
Advanced analysis and exploration tools for ESCAI CLI
"""

import re
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.text import Text

from .console import get_console
from .ascii_viz import ASCIIScatterPlot, ASCIIHeatmap, ASCIILineChart, ChartConfig


class QueryOperator(Enum):
    """Query operators for filtering"""
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"


class AggregateFunction(Enum):
    """Aggregate functions for data analysis"""
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STDDEV = "stddev"
    VARIANCE = "variance"
    PERCENTILE = "percentile"


@dataclass
class QueryCondition:
    """Represents a single query condition"""
    field: str
    operator: QueryOperator
    value: Any
    case_sensitive: bool = True
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate condition against data record"""
        field_value = self._get_nested_value(data, self.field)
        
        if field_value is None:
            return False
        
        # Convert to string for string operations if needed
        if self.operator in [QueryOperator.CONTAINS, QueryOperator.STARTS_WITH, 
                           QueryOperator.ENDS_WITH, QueryOperator.REGEX]:
            field_value = str(field_value)
            compare_value = str(self.value)
            
            if not self.case_sensitive:
                field_value = field_value.lower()
                compare_value = compare_value.lower()
        else:
            compare_value = self.value
        
        # Evaluate based on operator
        if self.operator == QueryOperator.EQUALS:
            return field_value == compare_value
        elif self.operator == QueryOperator.NOT_EQUALS:
            return field_value != compare_value
        elif self.operator == QueryOperator.GREATER_THAN:
            return field_value > compare_value
        elif self.operator == QueryOperator.LESS_THAN:
            return field_value < compare_value
        elif self.operator == QueryOperator.GREATER_EQUAL:
            return field_value >= compare_value
        elif self.operator == QueryOperator.LESS_EQUAL:
            return field_value <= compare_value
        elif self.operator == QueryOperator.CONTAINS:
            return compare_value in field_value
        elif self.operator == QueryOperator.STARTS_WITH:
            return field_value.startswith(compare_value)
        elif self.operator == QueryOperator.ENDS_WITH:
            return field_value.endswith(compare_value)
        elif self.operator == QueryOperator.REGEX:
            try:
                pattern = re.compile(compare_value, re.IGNORECASE if not self.case_sensitive else 0)
                return bool(pattern.search(field_value))
            except re.error:
                return False
        elif self.operator == QueryOperator.IN:
            return field_value in compare_value
        elif self.operator == QueryOperator.NOT_IN:
            return field_value not in compare_value
        elif self.operator == QueryOperator.BETWEEN:
            if isinstance(compare_value, (list, tuple)) and len(compare_value) == 2:
                return compare_value[0] <= field_value <= compare_value[1]
            return False
        
        return False
    
    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """Get nested field value using dot notation"""
        keys = field.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value


@dataclass
class QueryBuilder:
    """Interactive query builder for complex data filtering"""
    conditions: List[QueryCondition] = field(default_factory=list)
    logic_operator: str = "AND"  # AND or OR
    
    def add_condition(self, field: str, operator: QueryOperator, value: Any, case_sensitive: bool = True):
        """Add a condition to the query"""
        condition = QueryCondition(field, operator, value, case_sensitive)
        self.conditions.append(condition)
    
    def remove_condition(self, index: int):
        """Remove a condition by index"""
        if 0 <= index < len(self.conditions):
            self.conditions.pop(index)
    
    def clear_conditions(self):
        """Clear all conditions"""
        self.conditions.clear()
    
    def execute(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute query against data"""
        if not self.conditions:
            return data
        
        results = []
        for record in data:
            if self.logic_operator == "AND":
                if all(condition.evaluate(record) for condition in self.conditions):
                    results.append(record)
            else:  # OR
                if any(condition.evaluate(record) for condition in self.conditions):
                    results.append(record)
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary representation"""
        return {
            "conditions": [
                {
                    "field": c.field,
                    "operator": c.operator.value,
                    "value": c.value,
                    "case_sensitive": c.case_sensitive
                }
                for c in self.conditions
            ],
            "logic_operator": self.logic_operator
        }
    
    def from_dict(self, query_dict: Dict[str, Any]):
        """Load query from dictionary representation"""
        self.conditions.clear()
        self.logic_operator = query_dict.get("logic_operator", "AND")
        
        for cond_dict in query_dict.get("conditions", []):
            operator = QueryOperator(cond_dict["operator"])
            self.add_condition(
                cond_dict["field"],
                operator,
                cond_dict["value"],
                cond_dict.get("case_sensitive", True)
            )


class AdvancedSearch:
    """Advanced search with regex, fuzzy matching, and saved searches"""
    
    def __init__(self):
        self.saved_searches: Dict[str, Dict[str, Any]] = {}
        self.search_history: List[str] = []
        self.max_history = 50
    
    def regex_search(self, data: List[Dict[str, Any]], field: str, pattern: str, 
                    case_sensitive: bool = True) -> List[Dict[str, Any]]:
        """Perform regex search on specified field"""
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
            
            results = []
            for record in data:
                field_value = self._get_field_value(record, field)
                if field_value and regex.search(str(field_value)):
                    results.append(record)
            
            return results
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def fuzzy_search(self, data: List[Dict[str, Any]], field: str, query: str, 
                    threshold: float = 0.6) -> List[Tuple[Dict[str, Any], float]]:
        """Perform fuzzy search with similarity scoring"""
        results = []
        query_lower = query.lower()
        
        for record in data:
            field_value = self._get_field_value(record, field)
            if field_value:
                value_lower = str(field_value).lower()
                similarity = self._calculate_similarity(query_lower, value_lower)
                
                if similarity >= threshold:
                    results.append((record, similarity))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def multi_field_search(self, data: List[Dict[str, Any]], fields: List[str], 
                          query: str, match_all: bool = False) -> List[Dict[str, Any]]:
        """Search across multiple fields"""
        results = []
        query_lower = query.lower()
        
        for record in data:
            matches = []
            for field in fields:
                field_value = self._get_field_value(record, field)
                if field_value and query_lower in str(field_value).lower():
                    matches.append(True)
                else:
                    matches.append(False)
            
            if match_all and all(matches):
                results.append(record)
            elif not match_all and any(matches):
                results.append(record)
        
        return results
    
    def save_search(self, name: str, query: Dict[str, Any]):
        """Save a search query"""
        self.saved_searches[name] = {
            "query": query,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0
        }
    
    def load_search(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a saved search query"""
        if name in self.saved_searches:
            self.saved_searches[name]["usage_count"] += 1
            return self.saved_searches[name]["query"]
        return None
    
    def list_saved_searches(self) -> List[Dict[str, Any]]:
        """List all saved searches"""
        return [
            {
                "name": name,
                "created_at": search["created_at"],
                "usage_count": search["usage_count"]
            }
            for name, search in self.saved_searches.items()
        ]
    
    def add_to_history(self, query: str):
        """Add query to search history"""
        if query not in self.search_history:
            self.search_history.append(query)
            if len(self.search_history) > self.max_history:
                self.search_history.pop(0)
    
    def _get_field_value(self, record: Dict[str, Any], field: str) -> Any:
        """Get field value with dot notation support"""
        keys = field.split('.')
        value = record
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Levenshtein distance"""
        if not str1 or not str2:
            return 0.0
        
        # Simple similarity calculation
        if str1 == str2:
            return 1.0
        
        # Check for substring matches
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # Calculate character overlap
        set1, set2 = set(str1), set(str2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0


class StatisticalAnalysis:
    """Statistical analysis tools with hypothesis testing"""
    
    def __init__(self):
        self.console = get_console()
    
    def descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics"""
        if not data:
            return {}
        
        sorted_data = sorted(data)
        n = len(data)
        
        stats = {
            "count": n,
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "mode": statistics.mode(data) if n > 1 else data[0],
            "min": min(data),
            "max": max(data),
            "range": max(data) - min(data),
            "variance": statistics.variance(data) if n > 1 else 0,
            "std_dev": statistics.stdev(data) if n > 1 else 0,
            "q1": self._percentile(sorted_data, 25),
            "q3": self._percentile(sorted_data, 75),
        }
        
        stats["iqr"] = stats["q3"] - stats["q1"]
        stats["cv"] = stats["std_dev"] / stats["mean"] if stats["mean"] != 0 else 0
        
        return stats
    
    def correlation_analysis(self, x_data: List[float], y_data: List[float]) -> Dict[str, float]:
        """Calculate correlation between two datasets"""
        if len(x_data) != len(y_data) or len(x_data) < 2:
            return {"error": "Invalid data for correlation analysis"}
        
        # Pearson correlation coefficient
        n = len(x_data)
        sum_x = sum(x_data)
        sum_y = sum(y_data)
        sum_xy = sum(x * y for x, y in zip(x_data, y_data))
        sum_x2 = sum(x * x for x in x_data)
        sum_y2 = sum(y * y for y in y_data)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        if denominator == 0:
            correlation = 0.0
        else:
            correlation = float(numerator / denominator)
        
        # Determine correlation strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            strength = "very strong"
        elif abs_corr >= 0.6:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        return {
            "correlation": float(correlation),
            "strength": strength,
            "r_squared": float(correlation ** 2),
            "sample_size": n
        }
    
    def hypothesis_test(self, sample1: List[float], sample2: List[float], 
                       alpha: float = 0.05) -> Dict[str, Any]:
        """Perform two-sample t-test"""
        if len(sample1) < 2 or len(sample2) < 2:
            return {"error": "Insufficient data for hypothesis test"}
        
        # Calculate sample statistics
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
        var1 = statistics.variance(sample1) if n1 > 1 else 0
        var2 = statistics.variance(sample2) if n2 > 1 else 0
        
        # Pooled standard error
        pooled_se = math.sqrt(var1/n1 + var2/n2) if (var1/n1 + var2/n2) > 0 else 1
        
        # t-statistic
        t_stat = (mean1 - mean2) / pooled_se if pooled_se != 0 else 0
        
        # Degrees of freedom (approximation)
        df = n1 + n2 - 2
        
        # Critical value (approximation for common alpha levels)
        critical_values = {0.05: 1.96, 0.01: 2.58, 0.001: 3.29}
        critical_value = critical_values.get(alpha, 1.96)
        
        # p-value approximation
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        
        return {
            "t_statistic": float(t_stat),
            "degrees_of_freedom": df,
            "p_value": float(p_value),
            "critical_value": float(critical_value),
            "significant": abs(t_stat) > critical_value,
            "alpha": alpha,
            "mean_difference": float(mean1 - mean2),
            "effect_size": float((mean1 - mean2) / math.sqrt((var1 + var2) / 2)) if (var1 + var2) > 0 else 0.0
        }
    
    def trend_analysis(self, data: List[float], time_points: Optional[List[float]] = None) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        if len(data) < 3:
            return {"error": "Insufficient data for trend analysis"}
        
        if time_points is None:
            time_points = list(range(len(data)))
        
        # Linear regression for trend
        n = len(data)
        sum_x = sum(time_points)
        sum_y = sum(data)
        sum_xy = sum(x * y for x, y in zip(time_points, data))
        sum_x2 = sum(x * x for x in time_points)
        
        # Slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # R-squared
        y_mean = statistics.mean(data)
        ss_tot = sum((y - y_mean) ** 2 for y in data)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(time_points, data))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Trend direction
        if slope > 0.001:
            direction = "increasing"
        elif slope < -0.001:
            direction = "decreasing"
        else:
            direction = "stable"
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "direction": direction,
            "strength": "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "weak"
        }
    
    def _percentile(self, sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile of sorted data"""
        if not sorted_data:
            return 0
        
        k = (len(sorted_data) - 1) * percentile / 100
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_data[int(k)]
        
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal cumulative distribution function"""
        # Approximation using error function
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class DataCorrelationExplorer:
    """Interactive data correlation explorer"""
    
    def __init__(self):
        self.console = get_console()
        self.stats = StatisticalAnalysis()
    
    def explore_correlations(self, data: List[Dict[str, Any]], numeric_fields: List[str]) -> Dict[str, Any]:
        """Explore correlations between numeric fields"""
        if len(numeric_fields) < 2:
            return {"error": "Need at least 2 numeric fields for correlation analysis"}
        
        # Extract numeric data for each field
        field_data = {}
        for field in numeric_fields:
            values = []
            for record in data:
                value = self._get_numeric_value(record, field)
                if value is not None:
                    values.append(value)
            field_data[field] = values
        
        # Calculate correlation matrix
        correlation_matrix: Dict[str, Dict[str, float]] = {}
        for i, field1 in enumerate(numeric_fields):
            correlation_matrix[field1] = {}
            for j, field2 in enumerate(numeric_fields):
                if i == j:
                    correlation_matrix[field1][field2] = 1.0
                elif field2 in correlation_matrix and field1 in correlation_matrix[field2]:
                    # Use already calculated correlation
                    correlation_matrix[field1][field2] = correlation_matrix[field2][field1]
                else:
                    # Calculate correlation
                    data1 = field_data[field1]
                    data2 = field_data[field2]
                    
                    # Align data (only use records where both fields have values)
                    aligned_data1, aligned_data2 = [], []
                    for record in data:
                        val1 = self._get_numeric_value(record, field1)
                        val2 = self._get_numeric_value(record, field2)
                        if val1 is not None and val2 is not None:
                            aligned_data1.append(val1)
                            aligned_data2.append(val2)
                    
                    if len(aligned_data1) >= 2:
                        corr_result = self.stats.correlation_analysis(aligned_data1, aligned_data2)
                        correlation_matrix[field1][field2] = corr_result.get("correlation", 0)
                    else:
                        correlation_matrix[field1][field2] = 0
        
        return {
            "correlation_matrix": correlation_matrix,
            "field_stats": {field: self.stats.descriptive_stats(values) 
                          for field, values in field_data.items()},
            "sample_sizes": {field: len(values) for field, values in field_data.items()}
        }
    
    def create_correlation_heatmap(self, correlation_matrix: Dict[str, Dict[str, float]]) -> str:
        """Create ASCII heatmap of correlation matrix"""
        fields = list(correlation_matrix.keys())
        
        # Convert to 2D array
        matrix_data = []
        for field1 in fields:
            row = []
            for field2 in fields:
                row.append(correlation_matrix[field1][field2])
            matrix_data.append(row)
        
        # Create heatmap
        config = ChartConfig(width=80, height=len(fields) + 5, title="Correlation Matrix")
        heatmap = ASCIIHeatmap(config)
        return heatmap.create(matrix_data, fields, fields)
    
    def create_scatter_plot(self, data: List[Dict[str, Any]], x_field: str, y_field: str) -> str:
        """Create scatter plot for two fields"""
        x_values, y_values = [], []
        
        for record in data:
            x_val = self._get_numeric_value(record, x_field)
            y_val = self._get_numeric_value(record, y_field)
            if x_val is not None and y_val is not None:
                x_values.append(x_val)
                y_values.append(y_val)
        
        if len(x_values) < 2:
            return "Insufficient data for scatter plot"
        
        config = ChartConfig(width=60, height=15, title=f"{x_field} vs {y_field}")
        scatter = ASCIIScatterPlot(config)
        return scatter.create(x_values, y_values)
    
    def _get_numeric_value(self, record: Dict[str, Any], field: str) -> Optional[float]:
        """Extract numeric value from record field"""
        keys = field.split('.')
        value = record
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        # Try to convert to float
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


class TimeSeriesAnalyzer:
    """Time series analysis tools"""
    
    def __init__(self):
        self.console = get_console()
        self.stats = StatisticalAnalysis()
    
    def analyze_time_series(self, data: List[Dict[str, Any]], time_field: str, 
                          value_field: str, window_size: int = 5) -> Dict[str, Any]:
        """Comprehensive time series analysis"""
        # Extract time series data
        time_series = []
        for record in data:
            time_val = record.get(time_field)
            value_val = self._get_numeric_value(record, value_field)
            
            if time_val and value_val is not None:
                # Convert time to timestamp if needed
                if isinstance(time_val, str):
                    try:
                        time_val = datetime.fromisoformat(time_val.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                
                time_series.append((time_val, value_val))
        
        if len(time_series) < 3:
            return {"error": "Insufficient time series data"}
        
        # Sort by time
        time_series.sort(key=lambda x: x[0])
        
        # Extract values and time points
        times = [t for t, v in time_series]
        values = [v for t, v in time_series]
        time_numeric = [(t - times[0]).total_seconds() for t in times]
        
        # Basic statistics
        basic_stats = self.stats.descriptive_stats(values)
        
        # Trend analysis
        trend = self.stats.trend_analysis(values, time_numeric)
        
        # Moving averages
        moving_avg = self._calculate_moving_average(values, window_size)
        
        # Seasonality detection (simple)
        seasonality = self._detect_seasonality(values)
        
        # Volatility
        volatility = self._calculate_volatility(values)
        
        return {
            "basic_stats": basic_stats,
            "trend": trend,
            "moving_average": moving_avg,
            "seasonality": seasonality,
            "volatility": volatility,
            "data_points": len(time_series),
            "time_range": {
                "start": times[0].isoformat() if hasattr(times[0], 'isoformat') else str(times[0]),
                "end": times[-1].isoformat() if hasattr(times[-1], 'isoformat') else str(times[-1])
            }
        }
    
    def create_time_series_chart(self, data: List[Dict[str, Any]], time_field: str, 
                               value_field: str) -> str:
        """Create ASCII line chart for time series"""
        values = []
        for record in data:
            value = self._get_numeric_value(record, value_field)
            if value is not None:
                values.append(value)
        
        if not values:
            return "No numeric data available for chart"
        
        config = ChartConfig(width=70, height=15, title=f"{value_field} Over Time")
        chart = ASCIILineChart(config)
        return chart.create(values)
    
    def _calculate_moving_average(self, values: List[float], window_size: int) -> List[float]:
        """Calculate moving average"""
        if window_size >= len(values):
            return [statistics.mean(values)] * len(values)
        
        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            window = values[start_idx:end_idx]
            moving_avg.append(statistics.mean(window))
        
        return moving_avg
    
    def _detect_seasonality(self, values: List[float]) -> Dict[str, Any]:
        """Simple seasonality detection"""
        if len(values) < 12:
            return {"detected": False, "reason": "Insufficient data"}
        
        # Check for repeating patterns (very basic)
        # Look for correlation with lagged versions
        best_lag = 0
        best_correlation = 0
        
        for lag in range(2, min(len(values) // 2, 24)):
            if len(values) - lag < 2:
                continue
            
            original = values[:-lag]
            lagged = values[lag:]
            
            if len(original) == len(lagged) and len(original) >= 2:
                try:
                    corr_result = self.stats.correlation_analysis(original, lagged)
                    correlation = abs(float(corr_result.get("correlation", 0)))
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_lag = lag
                except:
                    continue
        
        return {
            "detected": best_correlation > 0.5,
            "period": best_lag if best_correlation > 0.5 else None,
            "strength": best_correlation,
            "confidence": "high" if best_correlation > 0.7 else "medium" if best_correlation > 0.5 else "low"
        }
    
    def _calculate_volatility(self, values: List[float]) -> Dict[str, float]:
        """Calculate volatility measures"""
        if len(values) < 2:
            return {"error": "Insufficient data"}
        
        # Calculate returns (percentage changes)
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)
        
        if not returns:
            return {"error": "Cannot calculate returns"}
        
        return {
            "return_volatility": float(statistics.stdev(returns)) if len(returns) > 1 else 0.0,
            "price_volatility": float(statistics.stdev(values)),
            "max_drawdown": self._calculate_max_drawdown(values),
            "average_return": float(statistics.mean(returns))
        }
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not values:
            return 0
        
        peak = values[0]
        max_drawdown = 0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown_ratio = (peak - value) / peak if peak != 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown_ratio)
        
        return max_drawdown
    
    def _get_numeric_value(self, record: Dict[str, Any], field: str) -> Optional[float]:
        """Extract numeric value from record field"""
        keys = field.split('.')
        value = record
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None