"""
Advanced data filtering and search capabilities for ESCAI CLI.

This module provides interactive filtering, search, and data exploration
tools for analyzing agent behavior and system metrics.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .console import get_console

console = get_console()


class FilterOperator(Enum):
    """Filter operators for data filtering"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "ge"
    LESS_EQUAL = "le"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"


@dataclass
class FilterCondition:
    """Represents a single filter condition"""
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = False


@dataclass
class SearchQuery:
    """Represents a search query with multiple conditions"""
    conditions: List[FilterCondition]
    logic: str = "AND"  # AND or OR
    limit: Optional[int] = None
    sort_by: Optional[str] = None
    sort_desc: bool = False


class DataFilter:
    """Advanced data filtering and search engine"""
    
    def __init__(self) -> None:
        self.saved_queries: Dict[str, SearchQuery] = {}
        self.filter_history: List[SearchQuery] = []
    
    def apply_filter(self, data: List[Dict[str, Any]], query: SearchQuery) -> List[Dict[str, Any]]:
        """Apply filter query to data"""
        if not query.conditions:
            return data
        
        filtered_data = []
        
        for item in data:
            if self._matches_query(item, query):
                filtered_data.append(item)
        
        # Apply sorting
        if query.sort_by and query.sort_by in (filtered_data[0].keys() if filtered_data else []):
            filtered_data.sort(
                key=lambda x: x.get(query.sort_by, 0),
                reverse=query.sort_desc
            )
        
        # Apply limit
        if query.limit:
            filtered_data = filtered_data[:query.limit]
        
        return filtered_data
    
    def _matches_query(self, item: Dict[str, Any], query: SearchQuery) -> bool:
        """Check if item matches the query conditions"""
        if not query.conditions:
            return True
        
        results = []
        
        for condition in query.conditions:
            result = self._matches_condition(item, condition)
            results.append(result)
        
        # Apply logic
        if query.logic.upper() == "OR":
            return any(results)
        else:  # AND
            return all(results)
    
    def _matches_condition(self, item: Dict[str, Any], condition: FilterCondition) -> bool:
        """Check if item matches a single condition"""
        field_value = self._get_nested_value(item, condition.field)
        
        if field_value is None:
            return False
        
        # Convert to string for string operations
        if isinstance(field_value, str) and not condition.case_sensitive:
            field_value = field_value.lower()
            if isinstance(condition.value, str):
                condition.value = condition.value.lower()
        
        # Apply operator
        if condition.operator == FilterOperator.EQUALS:
            return field_value == condition.value
        elif condition.operator == FilterOperator.NOT_EQUALS:
            return field_value != condition.value
        elif condition.operator == FilterOperator.GREATER_THAN:
            return field_value > condition.value
        elif condition.operator == FilterOperator.LESS_THAN:
            return field_value < condition.value
        elif condition.operator == FilterOperator.GREATER_EQUAL:
            return field_value >= condition.value
        elif condition.operator == FilterOperator.LESS_EQUAL:
            return field_value <= condition.value
        elif condition.operator == FilterOperator.CONTAINS:
            return str(condition.value) in str(field_value)
        elif condition.operator == FilterOperator.NOT_CONTAINS:
            return str(condition.value) not in str(field_value)
        elif condition.operator == FilterOperator.STARTS_WITH:
            return str(field_value).startswith(str(condition.value))
        elif condition.operator == FilterOperator.ENDS_WITH:
            return str(field_value).endswith(str(condition.value))
        elif condition.operator == FilterOperator.REGEX:
            try:
                pattern = re.compile(str(condition.value))
                return bool(pattern.search(str(field_value)))
            except re.error:
                return False
        elif condition.operator == FilterOperator.IN:
            return field_value in condition.value
        elif condition.operator == FilterOperator.NOT_IN:
            return field_value not in condition.value
        elif condition.operator == FilterOperator.BETWEEN:
            if isinstance(condition.value, (list, tuple)) and len(condition.value) == 2:
                return condition.value[0] <= field_value <= condition.value[1]
            return False
        
        return False
    
    def _get_nested_value(self, item: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = field_path.split('.')
        value = item
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def save_query(self, name: str, query: SearchQuery):
        """Save a query for later use"""
        self.saved_queries[name] = query
        console.print(f"[success]Query '{name}' saved successfully[/success]")
    
    def load_query(self, name: str) -> Optional[SearchQuery]:
        """Load a saved query"""
        return self.saved_queries.get(name)
    
    def list_saved_queries(self) -> List[str]:
        """List all saved query names"""
        return list(self.saved_queries.keys())
    
    def interactive_filter_builder(self, sample_data: Dict[str, Any]) -> SearchQuery:
        """Interactive filter builder with guided prompts"""
        console.print("\n[bold cyan]🔍 Interactive Filter Builder[/bold cyan]")
        console.print("Build complex filters step by step\n")
        
        # Show available fields
        fields = self._get_available_fields(sample_data)
        console.print("[bold]Available fields:[/bold]")
        for i, field in enumerate(fields, 1):
            console.print(f"  {i}. {field}")
        
        conditions: List[FilterCondition] = []
        
        while True:
            console.print(f"\n[bold]Current conditions: {len(conditions)}[/bold]")
            
            if conditions:
                console.print("Existing conditions:")
                for i, cond in enumerate(conditions, 1):
                    console.print(f"  {i}. {cond.field} {cond.operator.value} {cond.value}")
            
            action = Prompt.ask(
                "\nWhat would you like to do?",
                choices=["add", "remove", "done", "clear"],
                default="add"
            )
            
            if action == "add":
                condition = self._build_condition(fields, sample_data)
                if condition:
                    conditions.append(condition)
            
            elif action == "remove" and conditions:
                try:
                    index = int(Prompt.ask("Enter condition number to remove")) - 1
                    if 0 <= index < len(conditions):
                        removed = conditions.pop(index)
                        console.print(f"[success]Removed: {removed.field} {removed.operator.value} {removed.value}[/success]")
                except ValueError:
                    console.print("[error]Invalid number[/error]")
            
            elif action == "clear":
                conditions.clear()
                console.print("[success]All conditions cleared[/success]")
            
            elif action == "done":
                break
        
        if not conditions:
            console.print("[warning]No conditions specified - returning empty query[/warning]")
            return SearchQuery(conditions=[])
        
        # Configure query logic
        logic = "AND"
        if len(conditions) > 1:
            logic = Prompt.ask(
                "Combine conditions with",
                choices=["AND", "OR"],
                default="AND"
            )
        
        # Configure sorting
        sort_by = None
        sort_desc = False
        
        if Confirm.ask("Add sorting?", default=False):
            sort_by = Prompt.ask(
                "Sort by field",
                choices=fields + ["none"],
                default="none"
            )
            if sort_by != "none":
                sort_desc = Confirm.ask("Sort descending?", default=False)
            else:
                sort_by = None
        
        # Configure limit
        limit = None
        if Confirm.ask("Limit results?", default=False):
            try:
                limit = int(Prompt.ask("Maximum results", default="100"))
            except ValueError:
                limit = None
        
        query = SearchQuery(
            conditions=conditions,
            logic=logic,
            limit=limit,
            sort_by=sort_by,
            sort_desc=sort_desc
        )
        
        # Save query option
        if Confirm.ask("Save this query?", default=False):
            name = Prompt.ask("Query name")
            if name:
                self.save_query(name, query)
        
        return query
    
    def _build_condition(self, fields: List[str], sample_data: Dict[str, Any]) -> Optional[FilterCondition]:
        """Build a single filter condition interactively"""
        console.print("\n[bold]Building new condition...[/bold]")
        
        # Select field
        field = Prompt.ask(
            "Select field",
            choices=fields + ["cancel"],
            default="cancel"
        )
        
        if field == "cancel":
            return None
        
        # Get sample value for reference
        sample_value = self._get_nested_value(sample_data, field)
        if sample_value is not None:
            console.print(f"[muted]Sample value: {sample_value} ({type(sample_value).__name__})[/muted]")
        
        # Select operator
        operators = [op.value for op in FilterOperator]
        operator_str = Prompt.ask(
            "Select operator",
            choices=operators,
            default="eq"
        )
        
        operator = FilterOperator(operator_str)
        
        # Get value
        value = self._get_condition_value(operator, sample_value)
        if value is None:
            return None
        
        # Case sensitivity for string operations
        case_sensitive = False
        if isinstance(value, str) and operator in [
            FilterOperator.CONTAINS, FilterOperator.NOT_CONTAINS,
            FilterOperator.STARTS_WITH, FilterOperator.ENDS_WITH,
            FilterOperator.EQUALS, FilterOperator.NOT_EQUALS
        ]:
            case_sensitive = Confirm.ask("Case sensitive?", default=False)
        
        return FilterCondition(
            field=field,
            operator=operator,
            value=value,
            case_sensitive=case_sensitive
        )
    
    def _get_condition_value(self, operator: FilterOperator, sample_value: Any) -> Any:
        """Get value for condition based on operator"""
        if operator == FilterOperator.BETWEEN:
            console.print("Enter range values:")
            try:
                min_val = float(Prompt.ask("Minimum value"))
                max_val = float(Prompt.ask("Maximum value"))
                return [min_val, max_val]
            except ValueError:
                console.print("[error]Invalid numeric values[/error]")
                return None
        
        elif operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            values_str = Prompt.ask("Enter comma-separated values")
            return [v.strip() for v in values_str.split(',')]
        
        else:
            value_str = Prompt.ask("Enter value")
            
            # Try to convert to appropriate type based on sample
            if sample_value is not None:
                try:
                    if isinstance(sample_value, bool):
                        return value_str.lower() in ['true', '1', 'yes', 'on']
                    elif isinstance(sample_value, int):
                        return int(value_str)
                    elif isinstance(sample_value, float):
                        return float(value_str)
                    elif isinstance(sample_value, datetime):
                        # Try to parse datetime
                        return datetime.fromisoformat(value_str)
                except ValueError:
                    pass  # Fall back to string
            
            return value_str
    
    def _get_available_fields(self, data: Dict[str, Any], prefix: str = "") -> List[str]:
        """Get all available fields from sample data (including nested)"""
        fields = []
        
        for key, value in data.items():
            field_name = f"{prefix}.{key}" if prefix else key
            fields.append(field_name)
            
            # Add nested fields for dictionaries
            if isinstance(value, dict):
                nested_fields = self._get_available_fields(value, field_name)
                fields.extend(nested_fields)
        
        return sorted(fields)
    
    def quick_search(self, data: List[Dict[str, Any]], search_term: str, 
                    fields: List[str] = None) -> List[Dict[str, Any]]:
        """Quick text search across specified fields or all string fields"""
        if not search_term:
            return data
        
        results = []
        search_term = search_term.lower()
        
        for item in data:
            if self._item_contains_term(item, search_term, fields):
                results.append(item)
        
        return results
    
    def _item_contains_term(self, item: Dict[str, Any], term: str, 
                           fields: List[str] = None) -> bool:
        """Check if item contains search term in specified fields"""
        if fields:
            # Search only in specified fields
            for field in fields:
                value = self._get_nested_value(item, field)
                if value and term in str(value).lower():
                    return True
        else:
            # Search in all string fields
            return self._search_in_dict(item, term)
        
        return False
    
    def _search_in_dict(self, data: Dict[str, Any], term: str) -> bool:
        """Recursively search for term in dictionary values"""
        for value in data.values():
            if isinstance(value, dict):
                if self._search_in_dict(value, term):
                    return True
            elif isinstance(value, list):
                for list_item in value:
                    if isinstance(list_item, dict):
                        if self._search_in_dict(list_item, term):
                            return True
                    elif term in str(list_item).lower():
                        return True
            elif value and term in str(value).lower():
                return True
        
        return False
    
    def fuzzy_search(self, data: List[Dict[str, Any]], search_term: str,
                    field: str, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Fuzzy search using simple string similarity"""
        if not search_term:
            return data
        
        results = []
        search_term = search_term.lower()
        
        for item in data:
            value = self._get_nested_value(item, field)
            if value:
                similarity = self._string_similarity(search_term, str(value).lower())
                if similarity >= threshold:
                    results.append((item, similarity))
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results]
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple string similarity (Jaccard similarity)"""
        if not s1 or not s2:
            return 0.0
        
        # Convert to sets of characters
        set1 = set(s1)
        set2 = set(s2)
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def create_filter_summary(self, query: SearchQuery, result_count: int) -> Panel:
        """Create a summary panel for applied filters"""
        content = []
        
        if query.conditions:
            content.append(f"[bold]Conditions ({query.logic}):[/bold]")
            for i, cond in enumerate(query.conditions, 1):
                content.append(f"  {i}. {cond.field} {cond.operator.value} {cond.value}")
        
        if query.sort_by:
            direction = "DESC" if query.sort_desc else "ASC"
            content.append(f"\n[bold]Sort:[/bold] {query.sort_by} ({direction})")
        
        if query.limit:
            content.append(f"[bold]Limit:[/bold] {query.limit}")
        
        content.append(f"\n[bold]Results:[/bold] {result_count} items")
        
        return Panel(
            "\n".join(content) if content else "No filters applied",
            title="🔍 Filter Summary",
            border_style="blue"
        )


def create_data_filter() -> DataFilter:
    """Create a new data filter instance"""
    return DataFilter()


def interactive_data_explorer(data: List[Dict[str, Any]], title: str = "Data Explorer"):
    """Launch interactive data exploration interface"""
    console.print(f"\n[bold cyan]🔍 {title}[/bold cyan]")
    console.print("Interactive data exploration with filtering and search\n")
    
    if not data:
        console.print("[warning]No data available for exploration[/warning]")
        return
    
    filter_engine = DataFilter()
    current_data = data
    current_query = SearchQuery(conditions=[])
    
    while True:
        # Show current status
        console.print(f"\n[bold]Current dataset:[/bold] {len(current_data)} items")
        
        if current_query.conditions:
            summary = filter_engine.create_filter_summary(current_query, len(current_data))
            console.print(summary)
        
        # Show menu
        console.print("\n[bold]Available actions:[/bold]")
        console.print("  1. View data")
        console.print("  2. Quick search")
        console.print("  3. Advanced filter")
        console.print("  4. Clear filters")
        console.print("  5. Save query")
        console.print("  6. Load query")
        console.print("  7. Export results")
        console.print("  8. Exit")
        
        choice = Prompt.ask("Select action", choices=["1", "2", "3", "4", "5", "6", "7", "8"])
        
        if choice == "1":
            # View data
            _display_data_table(current_data[:20])  # Show first 20 items
            if len(current_data) > 20:
                console.print(f"[muted]Showing first 20 of {len(current_data)} items[/muted]")
        
        elif choice == "2":
            # Quick search
            search_term = Prompt.ask("Enter search term")
            if search_term:
                current_data = filter_engine.quick_search(data, search_term)
                console.print(f"[success]Found {len(current_data)} matching items[/success]")
        
        elif choice == "3":
            # Advanced filter
            sample_data = data[0] if data else {}
            query = filter_engine.interactive_filter_builder(sample_data)
            current_data = filter_engine.apply_filter(data, query)
            current_query = query
            console.print(f"[success]Filter applied - {len(current_data)} items match[/success]")
        
        elif choice == "4":
            # Clear filters
            current_data = data
            current_query = SearchQuery(conditions=[])
            console.print("[success]Filters cleared[/success]")
        
        elif choice == "5":
            # Save query
            if current_query.conditions:
                name = Prompt.ask("Query name")
                if name:
                    filter_engine.save_query(name, current_query)
            else:
                console.print("[warning]No query to save[/warning]")
        
        elif choice == "6":
            # Load query
            saved_queries = filter_engine.list_saved_queries()
            if saved_queries:
                query_name = Prompt.ask("Select query", choices=saved_queries + ["cancel"])
                if query_name != "cancel":
                    loaded_query = filter_engine.load_query(query_name)
                    if loaded_query:
                        current_data = filter_engine.apply_filter(data, loaded_query)
                        current_query = loaded_query
                        console.print(f"[success]Query '{query_name}' loaded[/success]")
            else:
                console.print("[warning]No saved queries available[/warning]")
        
        elif choice == "7":
            # Export results
            if current_data:
                from .reporting import DataExporter, ReportFormat
                from pathlib import Path
                
                format_choice = Prompt.ask(
                    "Export format",
                    choices=["json", "csv", "txt"],
                    default="json"
                )
                
                filename = Prompt.ask("Filename (without extension)", default="export")
                
                exporter = DataExporter(console)
                try:
                    output_path = exporter.export_data(
                        {"results": current_data, "query": current_query.__dict__},
                        ReportFormat(format_choice),
                        Path(filename)
                    )
                    console.print(f"[success]Data exported to {output_path}[/success]")
                except Exception as e:
                    console.print(f"[error]Export failed: {str(e)}[/error]")
            else:
                console.print("[warning]No data to export[/warning]")
        
        elif choice == "8":
            # Exit
            break


def _display_data_table(data: List[Dict[str, Any]]):
    """Display data in a formatted table"""
    if not data:
        console.print("[warning]No data to display[/warning]")
        return
    
    # Get all unique keys from all items
    all_keys: set[str] = set()
    for item in data:
        all_keys.update(item.keys())
    
    # Limit columns for readability
    keys = sorted(list(all_keys))[:8]  # Show max 8 columns
    
    table = Table(show_header=True, header_style="bold magenta")
    
    for key in keys:
        table.add_column(key, style="cyan", max_width=20)
    
    for item in data:
        row = []
        for key in keys:
            value = item.get(key, "")
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 18:
                str_value = str_value[:15] + "..."
            row.append(str_value)
        table.add_row(*row)
    
    console.print(table)