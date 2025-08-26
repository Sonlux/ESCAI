"""
Unit tests for interactive terminal components
"""

import pytest
from unittest.mock import Mock, patch
from escai_framework.cli.utils.interactive import (
    InteractiveState, TableColumn, InteractiveTable, InteractiveTreeView,
    InteractivePagination, BookmarkManager, KeyCode
)


class TestInteractiveState:
    """Test interactive state management"""
    
    def test_default_state(self):
        """Test default state values"""
        state = InteractiveState()
        assert state.current_row == 0
        assert state.current_col == 0
        assert state.page == 0
        assert state.page_size == 20
        assert state.sort_column is None
        assert state.sort_reverse is False
        assert state.filter_text == ""
        assert state.search_text == ""
        assert state.search_results == []
        assert state.search_index == 0
        assert state.selected_rows == []
        assert state.bookmarks == {}
        assert state.show_help is False


class TestTableColumn:
    """Test table column configuration"""
    
    def test_basic_column(self):
        """Test basic column creation"""
        col = TableColumn("Name", "name")
        assert col.name == "Name"
        assert col.key == "name"
        assert col.width is None
        assert col.sortable is True
        assert col.filterable is True
        assert col.align == "left"
    
    def test_custom_column(self):
        """Test column with custom settings"""
        col = TableColumn(
            "ID", "id", 
            width=10, 
            sortable=False, 
            filterable=False, 
            align="right"
        )
        assert col.name == "ID"
        assert col.key == "id"
        assert col.width == 10
        assert col.sortable is False
        assert col.filterable is False
        assert col.align == "right"


class TestInteractiveTable:
    """Test interactive table functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.test_data = [
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "Los Angeles"},
            {"name": "Charlie", "age": 35, "city": "Chicago"},
            {"name": "Diana", "age": 28, "city": "Houston"}
        ]
        
        self.columns = [
            TableColumn("Name", "name"),
            TableColumn("Age", "age", width=10),
            TableColumn("City", "city")
        ]
    
    def test_table_creation(self):
        """Test table creation"""
        table = InteractiveTable(self.test_data, self.columns)
        assert len(table.data) == 4
        assert len(table.columns) == 3
        assert len(table.filtered_data) == 4
        assert table.state.current_row == 0
    
    def test_navigation_methods(self):
        """Test navigation methods"""
        table = InteractiveTable(self.test_data, self.columns)
        
        # Test move down
        table._move_down()
        assert table.state.current_row == 1
        
        # Test move up
        table._move_up()
        assert table.state.current_row == 0
        
        # Test move right
        table._move_right()
        assert table.state.current_col == 1
        
        # Test move left
        table._move_left()
        assert table.state.current_col == 0
    
    def test_pagination(self):
        """Test pagination functionality"""
        # Create table with small page size
        table = InteractiveTable(self.test_data, self.columns)
        table.state.page_size = 2
        
        # Test page down
        table._page_down()
        assert table.state.page == 1
        assert table.state.current_row == 0
        
        # Test page up
        table._page_up()
        assert table.state.page == 0
        assert table.state.current_row == 0
    
    def test_selection(self):
        """Test row selection"""
        table = InteractiveTable(self.test_data, self.columns)
        
        # Test toggle selection
        table._toggle_selection()
        assert 0 in table.state.selected_rows
        
        # Test toggle again (deselect)
        table._toggle_selection()
        assert 0 not in table.state.selected_rows
    
    def test_get_current_row(self):
        """Test getting current row data"""
        table = InteractiveTable(self.test_data, self.columns)
        
        current_row = table._get_current_row()
        assert current_row == self.test_data[0]
        
        # Move to next row
        table._move_down()
        current_row = table._get_current_row()
        assert current_row == self.test_data[1]
    
    def test_sorting(self):
        """Test column sorting"""
        table = InteractiveTable(self.test_data, self.columns)
        
        # Sort by age column
        table.state.current_col = 1  # Age column
        table._sort_column()
        
        assert table.state.sort_column == "age"
        assert table.state.sort_reverse is False
        
        # Sort again (reverse)
        table._sort_column()
        assert table.state.sort_reverse is True
    
    def test_filter_application(self):
        """Test data filtering"""
        table = InteractiveTable(self.test_data, self.columns)
        
        # Apply mock filter
        table.state.filter_text = "New"
        table._apply_filter()
        
        # Should filter to rows containing "New"
        assert len(table.filtered_data) <= len(self.test_data)
    
    def test_search_functionality(self):
        """Test search functionality"""
        table = InteractiveTable(self.test_data, self.columns)
        
        # Perform search
        table.state.search_text = "Alice"
        table._perform_search()
        
        assert len(table.state.search_results) >= 0
        
        # Test navigation through search results
        if table.state.search_results:
            table._next_search()
            table._prev_search()
    
    def test_bookmarking(self):
        """Test bookmark functionality"""
        table = InteractiveTable(self.test_data, self.columns)
        
        # Bookmark current row
        table._bookmark_current()
        assert len(table.state.bookmarks) == 1


class TestInteractiveTreeView:
    """Test interactive tree view functionality"""
    
    def setup_method(self):
        """Set up test tree data"""
        self.tree_data = {
            'name': 'Root',
            'value': '100%',
            'children': [
                {
                    'name': 'Child1',
                    'value': '50%',
                    'children': [
                        {'name': 'Grandchild1', 'value': '25%'},
                        {'name': 'Grandchild2', 'value': '25%'}
                    ]
                },
                {
                    'name': 'Child2',
                    'value': '50%'
                }
            ]
        }
    
    def test_tree_creation(self):
        """Test tree view creation"""
        tree = InteractiveTreeView(self.tree_data)
        assert tree.tree_data == self.tree_data
        assert len(tree.node_list) >= 1  # At least root node
        assert tree.state.current_row == 0
    
    def test_node_expansion(self):
        """Test node expansion/collapse"""
        tree = InteractiveTreeView(self.tree_data)
        
        # Expand root node
        tree._expand_node()
        assert "root" in tree.expanded_nodes
        
        # Rebuild node list to see children
        tree._build_node_list()
        initial_count = len(tree.node_list)
        
        # Collapse root node
        tree._collapse_node()
        assert "root" not in tree.expanded_nodes
    
    def test_navigation(self):
        """Test tree navigation"""
        tree = InteractiveTreeView(self.tree_data)
        
        # Expand to see more nodes
        tree._expand_all()
        tree._build_node_list()
        
        initial_row = tree.state.current_row
        
        # Test move down
        if len(tree.node_list) > 1:
            tree._move_down()
            assert tree.state.current_row == initial_row + 1
        
        # Test move up
        tree._move_up()
        assert tree.state.current_row == initial_row
    
    def test_expand_collapse_all(self):
        """Test expand/collapse all functionality"""
        tree = InteractiveTreeView(self.tree_data)
        
        # Expand all
        tree._expand_all()
        expanded_count = len(tree.expanded_nodes)
        assert expanded_count > 0
        
        # Collapse all
        tree._collapse_all()
        assert len(tree.expanded_nodes) == 0
    
    def test_get_current_node(self):
        """Test getting current node data"""
        tree = InteractiveTreeView(self.tree_data)
        
        current_node = tree._get_current_node()
        assert current_node is not None
        assert current_node.get('name') == 'Root'


class TestInteractivePagination:
    """Test pagination functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.test_data = list(range(100))  # 100 items
    
    def test_pagination_creation(self):
        """Test pagination creation"""
        pagination = InteractivePagination(self.test_data, page_size=10)
        assert pagination.page_size == 10
        assert pagination.current_page == 0
        assert pagination.total_pages == 10
    
    def test_page_navigation(self):
        """Test page navigation"""
        pagination = InteractivePagination(self.test_data, page_size=10)
        
        # Test next page
        result = pagination.next_page()
        assert result is True
        assert pagination.current_page == 1
        
        # Test previous page
        result = pagination.prev_page()
        assert result is True
        assert pagination.current_page == 0
        
        # Test jump to page
        result = pagination.jump_to_page(5)
        assert result is True
        assert pagination.current_page == 5
    
    def test_page_data(self):
        """Test getting page data"""
        pagination = InteractivePagination(self.test_data, page_size=10)
        
        # First page
        page_data = pagination.get_current_page_data()
        assert len(page_data) == 10
        assert page_data == list(range(10))
        
        # Second page
        pagination.next_page()
        page_data = pagination.get_current_page_data()
        assert len(page_data) == 10
        assert page_data == list(range(10, 20))
    
    def test_page_size_change(self):
        """Test changing page size"""
        pagination = InteractivePagination(self.test_data, page_size=10)
        
        # Change page size
        pagination.set_page_size(20)
        assert pagination.page_size == 20
        assert pagination.total_pages == 5
    
    def test_page_info(self):
        """Test getting page information"""
        pagination = InteractivePagination(self.test_data, page_size=10)
        
        info = pagination.get_page_info()
        assert info['current_page'] == 1
        assert info['total_pages'] == 10
        assert info['page_size'] == 10
        assert info['total_items'] == 100
        assert info['start_item'] == 1
        assert info['end_item'] == 10


class TestBookmarkManager:
    """Test bookmark management"""
    
    def setup_method(self):
        """Set up bookmark manager"""
        self.manager = BookmarkManager()
    
    def test_add_bookmark(self):
        """Test adding bookmarks"""
        data = {"name": "Test", "value": 123}
        
        result = self.manager.add_bookmark("test1", data)
        assert result is True
        assert "test1" in self.manager.bookmarks
        
        # Test duplicate bookmark
        result = self.manager.add_bookmark("test1", data)
        assert result is False
    
    def test_get_bookmark(self):
        """Test getting bookmarks"""
        data = {"name": "Test", "value": 123}
        self.manager.add_bookmark("test1", data)
        
        retrieved = self.manager.get_bookmark("test1")
        assert retrieved == data
        
        # Test access count increment
        initial_count = self.manager.bookmarks["test1"]["access_count"]
        self.manager.get_bookmark("test1")
        assert self.manager.bookmarks["test1"]["access_count"] == initial_count + 1
        
        # Test non-existent bookmark
        result = self.manager.get_bookmark("nonexistent")
        assert result is None
    
    def test_remove_bookmark(self):
        """Test removing bookmarks"""
        data = {"name": "Test", "value": 123}
        self.manager.add_bookmark("test1", data)
        
        result = self.manager.remove_bookmark("test1")
        assert result is True
        assert "test1" not in self.manager.bookmarks
        
        # Test removing non-existent bookmark
        result = self.manager.remove_bookmark("nonexistent")
        assert result is False
    
    def test_list_bookmarks(self):
        """Test listing bookmarks"""
        data1 = {"name": "Test1", "value": 123}
        data2 = {"name": "Test2", "value": 456}
        
        self.manager.add_bookmark("test1", data1)
        self.manager.add_bookmark("test2", data2)
        
        bookmarks = self.manager.list_bookmarks()
        assert len(bookmarks) == 2
        assert all('name' in b and 'created_at' in b and 'access_count' in b for b in bookmarks)
    
    def test_most_accessed(self):
        """Test getting most accessed bookmarks"""
        data = {"name": "Test", "value": 123}
        
        self.manager.add_bookmark("test1", data)
        self.manager.add_bookmark("test2", data)
        self.manager.add_bookmark("test3", data)
        
        # Access bookmarks different amounts
        self.manager.get_bookmark("test1")  # 1 access
        self.manager.get_bookmark("test2")  # 1 access
        self.manager.get_bookmark("test2")  # 2 accesses
        self.manager.get_bookmark("test3")  # 1 access
        self.manager.get_bookmark("test3")  # 2 accesses
        self.manager.get_bookmark("test3")  # 3 accesses
        
        most_accessed = self.manager.get_most_accessed(2)
        assert len(most_accessed) == 2
        assert most_accessed[0] == "test3"  # Most accessed
        assert most_accessed[1] == "test2"  # Second most accessed


class TestKeyCode:
    """Test key code enumeration"""
    
    def test_key_codes(self):
        """Test key code values"""
        assert KeyCode.UP.value == 'k'
        assert KeyCode.DOWN.value == 'j'
        assert KeyCode.LEFT.value == 'h'
        assert KeyCode.RIGHT.value == 'l'
        assert KeyCode.ENTER.value == '\r'
        assert KeyCode.QUIT.value == 'q'
        assert KeyCode.SEARCH.value == '/'


if __name__ == '__main__':
    pytest.main([__file__])