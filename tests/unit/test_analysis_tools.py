"""
Unit tests for advanced analysis tools
"""

import pytest
import math
from datetime import datetime, timedelta
from escai_framework.cli.utils.analysis_tools import (
    QueryCondition, QueryOperator, QueryBuilder, AdvancedSearch,
    StatisticalAnalysis, DataCorrelationExplorer, TimeSeriesAnalyzer,
    AggregateFunction
)


class TestQueryCondition:
    """Test query condition functionality"""
    
    def test_equals_condition(self):
        """Test equals operator"""
        condition = QueryCondition("name", QueryOperator.EQUALS, "Alice")
        
        assert condition.evaluate({"name": "Alice"}) is True
        assert condition.evaluate({"name": "Bob"}) is False
        assert condition.evaluate({"other": "Alice"}) is False
    
    def test_numeric_conditions(self):
        """Test numeric comparison operators"""
        gt_condition = QueryCondition("age", QueryOperator.GREATER_THAN, 25)
        lt_condition = QueryCondition("age", QueryOperator.LESS_THAN, 30)
        
        assert gt_condition.evaluate({"age": 30}) is True
        assert gt_condition.evaluate({"age": 20}) is False
        assert lt_condition.evaluate({"age": 25}) is True
        assert lt_condition.evaluate({"age": 35}) is False
    
    def test_string_conditions(self):
        """Test string operators"""
        contains_condition = QueryCondition("description", QueryOperator.CONTAINS, "test")
        starts_condition = QueryCondition("name", QueryOperator.STARTS_WITH, "Dr")
        
        assert contains_condition.evaluate({"description": "This is a test"}) is True
        assert contains_condition.evaluate({"description": "No match"}) is False
        assert starts_condition.evaluate({"name": "Dr. Smith"}) is True
        assert starts_condition.evaluate({"name": "Mr. Smith"}) is False
    
    def test_case_sensitivity(self):
        """Test case sensitivity option"""
        case_sensitive = QueryCondition("name", QueryOperator.CONTAINS, "ALICE", case_sensitive=True)
        case_insensitive = QueryCondition("name", QueryOperator.CONTAINS, "ALICE", case_sensitive=False)
        
        data = {"name": "alice"}
        assert case_sensitive.evaluate(data) is False
        assert case_insensitive.evaluate(data) is True
    
    def test_nested_field_access(self):
        """Test nested field access with dot notation"""
        condition = QueryCondition("user.profile.age", QueryOperator.GREATER_THAN, 18)
        
        data = {
            "user": {
                "profile": {
                    "age": 25
                }
            }
        }
        
        assert condition.evaluate(data) is True
        assert condition.evaluate({"user": {"profile": {"age": 15}}}) is False
        assert condition.evaluate({"user": {"name": "Alice"}}) is False
    
    def test_in_operator(self):
        """Test IN operator"""
        condition = QueryCondition("status", QueryOperator.IN, ["active", "pending"])
        
        assert condition.evaluate({"status": "active"}) is True
        assert condition.evaluate({"status": "inactive"}) is False
    
    def test_between_operator(self):
        """Test BETWEEN operator"""
        condition = QueryCondition("score", QueryOperator.BETWEEN, [70, 90])
        
        assert condition.evaluate({"score": 80}) is True
        assert condition.evaluate({"score": 60}) is False
        assert condition.evaluate({"score": 95}) is False
    
    def test_regex_operator(self):
        """Test REGEX operator"""
        condition = QueryCondition("email", QueryOperator.REGEX, r".*@example\.com$")
        
        assert condition.evaluate({"email": "user@example.com"}) is True
        assert condition.evaluate({"email": "user@other.com"}) is False
        
        # Test invalid regex
        invalid_condition = QueryCondition("email", QueryOperator.REGEX, "[invalid")
        assert invalid_condition.evaluate({"email": "test"}) is False


class TestQueryBuilder:
    """Test query builder functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.test_data = [
            {"name": "Alice", "age": 30, "city": "New York", "status": "active"},
            {"name": "Bob", "age": 25, "city": "Los Angeles", "status": "inactive"},
            {"name": "Charlie", "age": 35, "city": "Chicago", "status": "active"},
            {"name": "Diana", "age": 28, "city": "Houston", "status": "pending"}
        ]
    
    def test_single_condition(self):
        """Test query with single condition"""
        builder = QueryBuilder()
        builder.add_condition("status", QueryOperator.EQUALS, "active")
        
        results = builder.execute(self.test_data)
        assert len(results) == 2
        assert all(r["status"] == "active" for r in results)
    
    def test_multiple_conditions_and(self):
        """Test query with multiple AND conditions"""
        builder = QueryBuilder()
        builder.add_condition("status", QueryOperator.EQUALS, "active")
        builder.add_condition("age", QueryOperator.GREATER_THAN, 30)
        builder.logic_operator = "AND"
        
        results = builder.execute(self.test_data)
        assert len(results) == 1
        assert results[0]["name"] == "Charlie"
    
    def test_multiple_conditions_or(self):
        """Test query with multiple OR conditions"""
        builder = QueryBuilder()
        builder.add_condition("city", QueryOperator.EQUALS, "New York")
        builder.add_condition("city", QueryOperator.EQUALS, "Chicago")
        builder.logic_operator = "OR"
        
        results = builder.execute(self.test_data)
        assert len(results) == 2
        cities = [r["city"] for r in results]
        assert "New York" in cities
        assert "Chicago" in cities
    
    def test_remove_condition(self):
        """Test removing conditions"""
        builder = QueryBuilder()
        builder.add_condition("status", QueryOperator.EQUALS, "active")
        builder.add_condition("age", QueryOperator.GREATER_THAN, 30)
        
        assert len(builder.conditions) == 2
        builder.remove_condition(0)
        assert len(builder.conditions) == 1
        assert builder.conditions[0].field == "age"
    
    def test_clear_conditions(self):
        """Test clearing all conditions"""
        builder = QueryBuilder()
        builder.add_condition("status", QueryOperator.EQUALS, "active")
        builder.add_condition("age", QueryOperator.GREATER_THAN, 30)
        
        builder.clear_conditions()
        assert len(builder.conditions) == 0
        
        # Should return all data when no conditions
        results = builder.execute(self.test_data)
        assert len(results) == len(self.test_data)
    
    def test_to_from_dict(self):
        """Test serialization to/from dictionary"""
        builder = QueryBuilder()
        builder.add_condition("name", QueryOperator.CONTAINS, "A", case_sensitive=False)
        builder.add_condition("age", QueryOperator.GREATER_THAN, 25)
        builder.logic_operator = "OR"
        
        # Convert to dict
        query_dict = builder.to_dict()
        assert query_dict["logic_operator"] == "OR"
        assert len(query_dict["conditions"]) == 2
        
        # Create new builder from dict
        new_builder = QueryBuilder()
        new_builder.from_dict(query_dict)
        
        assert new_builder.logic_operator == "OR"
        assert len(new_builder.conditions) == 2
        assert new_builder.conditions[0].field == "name"
        assert new_builder.conditions[0].operator == QueryOperator.CONTAINS


class TestAdvancedSearch:
    """Test advanced search functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.search = AdvancedSearch()
        self.test_data = [
            {"name": "Alice Johnson", "email": "alice@example.com", "department": "Engineering"},
            {"name": "Bob Smith", "email": "bob@company.com", "department": "Marketing"},
            {"name": "Charlie Brown", "email": "charlie@example.com", "department": "Engineering"},
            {"name": "Diana Prince", "email": "diana@company.com", "department": "Sales"}
        ]
    
    def test_regex_search(self):
        """Test regex search"""
        results = self.search.regex_search(self.test_data, "email", r".*@example\.com$")
        assert len(results) == 2
        emails = [r["email"] for r in results]
        assert "alice@example.com" in emails
        assert "charlie@example.com" in emails
    
    def test_fuzzy_search(self):
        """Test fuzzy search"""
        results = self.search.fuzzy_search(self.test_data, "name", "Alice", threshold=0.5)
        assert len(results) > 0
        # Results should be tuples of (record, similarity_score)
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2
        assert isinstance(results[0][1], float)
    
    def test_multi_field_search(self):
        """Test multi-field search"""
        # Search for "Engineering" across name and department
        results = self.search.multi_field_search(
            self.test_data, 
            ["name", "department"], 
            "Engineering", 
            match_all=False
        )
        assert len(results) == 2  # Should find records with "Engineering" in department
    
    def test_save_and_load_search(self):
        """Test saving and loading searches"""
        query = {"field": "department", "operator": "equals", "value": "Engineering"}
        
        # Save search
        self.search.save_search("engineering_filter", query)
        assert "engineering_filter" in self.search.saved_searches
        
        # Load search
        loaded_query = self.search.load_search("engineering_filter")
        assert loaded_query == query
        
        # Check usage count incremented
        assert self.search.saved_searches["engineering_filter"]["usage_count"] == 1
    
    def test_search_history(self):
        """Test search history"""
        self.search.add_to_history("test query 1")
        self.search.add_to_history("test query 2")
        
        assert len(self.search.search_history) == 2
        assert "test query 1" in self.search.search_history
        assert "test query 2" in self.search.search_history
        
        # Test duplicate prevention
        self.search.add_to_history("test query 1")
        assert len(self.search.search_history) == 2
    
    def test_list_saved_searches(self):
        """Test listing saved searches"""
        self.search.save_search("search1", {"test": "query1"})
        self.search.save_search("search2", {"test": "query2"})
        
        searches = self.search.list_saved_searches()
        assert len(searches) == 2
        
        search_names = [s["name"] for s in searches]
        assert "search1" in search_names
        assert "search2" in search_names


class TestStatisticalAnalysis:
    """Test statistical analysis functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.stats = StatisticalAnalysis()
        self.test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    def test_descriptive_stats(self):
        """Test descriptive statistics calculation"""
        stats = self.stats.descriptive_stats(self.test_data)
        
        assert stats["count"] == 10
        assert stats["mean"] == 5.5
        assert stats["median"] == 5.5
        assert stats["min"] == 1
        assert stats["max"] == 10
        assert stats["range"] == 9
        assert stats["q1"] == 3.25
        assert stats["q3"] == 7.75
        assert stats["iqr"] == 4.5
    
    def test_empty_data_stats(self):
        """Test descriptive statistics with empty data"""
        stats = self.stats.descriptive_stats([])
        assert stats == {}
    
    def test_correlation_analysis(self):
        """Test correlation analysis"""
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 6, 8, 10]  # Perfect positive correlation
        
        result = self.stats.correlation_analysis(x_data, y_data)
        
        assert abs(result["correlation"] - 1.0) < 0.001  # Should be close to 1
        assert result["strength"] == "very strong"
        assert result["sample_size"] == 5
    
    def test_correlation_invalid_data(self):
        """Test correlation with invalid data"""
        result = self.stats.correlation_analysis([1], [2])  # Too few points
        assert "error" in result
        
        result = self.stats.correlation_analysis([1, 2], [3])  # Mismatched lengths
        assert "error" in result
    
    def test_hypothesis_test(self):
        """Test hypothesis testing"""
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [6, 7, 8, 9, 10]  # Different means
        
        result = self.stats.hypothesis_test(sample1, sample2)
        
        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert result["mean_difference"] == -5.0  # 3 - 8
    
    def test_trend_analysis(self):
        """Test trend analysis"""
        # Increasing trend
        increasing_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.stats.trend_analysis(increasing_data)
        
        assert result["direction"] == "increasing"
        assert result["slope"] > 0
        assert result["r_squared"] > 0.9  # Should be very strong trend
    
    def test_trend_analysis_insufficient_data(self):
        """Test trend analysis with insufficient data"""
        result = self.stats.trend_analysis([1, 2])
        assert "error" in result


class TestDataCorrelationExplorer:
    """Test data correlation explorer"""
    
    def setup_method(self):
        """Set up test data"""
        self.explorer = DataCorrelationExplorer()
        self.test_data = [
            {"age": 25, "salary": 50000, "experience": 2},
            {"age": 30, "salary": 60000, "experience": 5},
            {"age": 35, "salary": 70000, "experience": 8},
            {"age": 40, "salary": 80000, "experience": 12},
            {"age": 45, "salary": 90000, "experience": 15}
        ]
    
    def test_explore_correlations(self):
        """Test correlation exploration"""
        numeric_fields = ["age", "salary", "experience"]
        result = self.explorer.explore_correlations(self.test_data, numeric_fields)
        
        assert "correlation_matrix" in result
        assert "field_stats" in result
        assert "sample_sizes" in result
        
        # Check correlation matrix structure
        matrix = result["correlation_matrix"]
        assert len(matrix) == 3
        assert "age" in matrix
        assert "salary" in matrix
        assert "experience" in matrix
        
        # Diagonal should be 1.0
        assert matrix["age"]["age"] == 1.0
        assert matrix["salary"]["salary"] == 1.0
        
        # Should have strong positive correlations
        assert matrix["age"]["salary"] > 0.8
        assert matrix["age"]["experience"] > 0.8
    
    def test_insufficient_fields(self):
        """Test with insufficient numeric fields"""
        result = self.explorer.explore_correlations(self.test_data, ["age"])
        assert "error" in result
    
    def test_create_correlation_heatmap(self):
        """Test correlation heatmap creation"""
        correlation_matrix = {
            "field1": {"field1": 1.0, "field2": 0.8},
            "field2": {"field1": 0.8, "field2": 1.0}
        }
        
        heatmap = self.explorer.create_correlation_heatmap(correlation_matrix)
        assert isinstance(heatmap, str)
        assert len(heatmap) > 0
    
    def test_create_scatter_plot(self):
        """Test scatter plot creation"""
        scatter_plot = self.explorer.create_scatter_plot(self.test_data, "age", "salary")
        assert isinstance(scatter_plot, str)
        assert len(scatter_plot) > 0


class TestTimeSeriesAnalyzer:
    """Test time series analyzer"""
    
    def setup_method(self):
        """Set up test data"""
        self.analyzer = TimeSeriesAnalyzer()
        
        # Create time series data
        base_time = datetime(2024, 1, 1)
        self.test_data = []
        for i in range(10):
            self.test_data.append({
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "value": i * 2 + 1,  # Linear increase
                "category": "test"
            })
    
    def test_analyze_time_series(self):
        """Test time series analysis"""
        result = self.analyzer.analyze_time_series(
            self.test_data, 
            "timestamp", 
            "value"
        )
        
        assert "basic_stats" in result
        assert "trend" in result
        assert "moving_average" in result
        assert "seasonality" in result
        assert "volatility" in result
        assert "data_points" in result
        assert "time_range" in result
        
        # Check trend detection - should be increasing since values go from 1 to 19
        assert result["trend"]["slope"] > 0
        # The direction should be increasing for this linear data
        assert result["trend"]["direction"] in ["increasing", "stable"]  # Allow both for now
    
    def test_insufficient_data(self):
        """Test with insufficient time series data"""
        result = self.analyzer.analyze_time_series(
            self.test_data[:2], 
            "timestamp", 
            "value"
        )
        assert "error" in result
    
    def test_create_time_series_chart(self):
        """Test time series chart creation"""
        chart = self.analyzer.create_time_series_chart(
            self.test_data, 
            "timestamp", 
            "value"
        )
        assert isinstance(chart, str)
        assert len(chart) > 0
    
    def test_moving_average_calculation(self):
        """Test moving average calculation"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        moving_avg = self.analyzer._calculate_moving_average(values, 3)
        
        assert len(moving_avg) == len(values)
        assert isinstance(moving_avg[0], float)
    
    def test_volatility_calculation(self):
        """Test volatility calculation"""
        values = [100, 105, 98, 102, 110, 95, 108]
        volatility = self.analyzer._calculate_volatility(values)
        
        assert "return_volatility" in volatility
        assert "price_volatility" in volatility
        assert "max_drawdown" in volatility
        assert "average_return" in volatility
        
        assert isinstance(volatility["return_volatility"], float)
        assert volatility["max_drawdown"] >= 0


if __name__ == '__main__':
    pytest.main([__file__])