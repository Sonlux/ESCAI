"""
Interactive terminal-based exploration interface for ESCAI CLI
"""

import sys
try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    # Windows doesn't have termios
    HAS_TERMIOS = False
from typing import List, Dict, Any, Optional, Callable, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.layout import Layout
from rich.live import Live

from .console import get_console


class KeyCode(Enum):
    """Key code constants for navigation"""
    UP = 'k'
    DOWN = 'j'
    LEFT = 'h'
    RIGHT = 'l'
    ENTER = '\r'
    ESCAPE = '\x1b'
    SPACE = ' '
    TAB = '\t'
    BACKSPACE = '\x7f'
    DELETE = '\x1b[3~'
    HOME = '\x1b[H'
    END = '\x1b[F'
    PAGE_UP = '\x1b[5~'
    PAGE_DOWN = '\x1b[6~'
    F1 = '\x1bOP'
    SEARCH = '/'
    NEXT_SEARCH = 'n'
    PREV_SEARCH = 'N'
    QUIT = 'q'


@dataclass
class TableColumn:
    """Table column configuration"""
    name: str
    key: str
    width: Optional[int] = None
    sortable: bool = True
    filterable: bool = True
    align: Literal['default', 'left', 'center', 'right', 'full'] = "left"


@dataclass
class InteractiveState:
    """State management for interactive interface"""
    current_row: int = 0
    current_col: int = 0
    page: int = 0
    page_size: int = 20
    sort_column: Optional[str] = None
    sort_reverse: bool = False
    filter_text: str = ""
    search_text: str = ""
    search_results: List[int] = field(default_factory=list)
    search_index: int = 0
    selected_rows: List[int] = field(default_factory=list)
    bookmarks: Dict[str, Any] = field(default_factory=dict)
    show_help: bool = False


class KeyboardHandler:
    """Handle keyboard input for interactive navigation"""
    
    def __init__(self):
        self.console = get_console()
        self.old_settings = None
    
    def __enter__(self):
        """Enter raw keyboard mode"""
        if HAS_TERMIOS and sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit raw keyboard mode"""
        if HAS_TERMIOS and self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self) -> str:
        """Get a single key press"""
        if not HAS_TERMIOS or not sys.stdin.isatty():
            # Fallback for Windows or non-TTY environments
            return input("Press key (or 'q' to quit): ").strip() or 'q'
        
        key = sys.stdin.read(1)
        
        # Handle escape sequences
        if key == '\x1b':
            key += sys.stdin.read(2)
            if key == '\x1b[5':
                key += sys.stdin.read(1)  # Page Up/Down
            elif key == '\x1b[3':
                key += sys.stdin.read(1)  # Delete
        
        return key


class InteractiveTable:
    """Interactive table browser with vim-like navigation"""
    
    def __init__(self, data: List[Dict[str, Any]], columns: List[TableColumn]):
        self.data = data
        self.columns = columns
        self.state = InteractiveState()
        self.console = get_console()
        self.filtered_data = data.copy()
        self.keyboard = KeyboardHandler()
    
    def run(self) -> Optional[Dict[str, Any]]:
        """Run interactive table browser"""
        with self.keyboard:
            with Live(self._render(), refresh_per_second=10, screen=True) as live:
                while True:
                    key = self.keyboard.get_key()
                    
                    if key == KeyCode.QUIT.value:
                        break
                    elif key == KeyCode.UP.value:
                        self._move_up()
                    elif key == KeyCode.DOWN.value:
                        self._move_down()
                    elif key == KeyCode.LEFT.value:
                        self._move_left()
                    elif key == KeyCode.RIGHT.value:
                        self._move_right()
                    elif key == KeyCode.ENTER.value:
                        return self._get_current_row()
                    elif key == KeyCode.SPACE.value:
                        self._toggle_selection()
                    elif key == KeyCode.SEARCH.value:
                        self._start_search()
                    elif key == KeyCode.NEXT_SEARCH.value:
                        self._next_search()
                    elif key == KeyCode.PREV_SEARCH.value:
                        self._prev_search()
                    elif key == KeyCode.PAGE_UP.value:
                        self._page_up()
                    elif key == KeyCode.PAGE_DOWN.value:
                        self._page_down()
                    elif key == KeyCode.HOME.value:
                        self._go_home()
                    elif key == KeyCode.END.value:
                        self._go_end()
                    elif key == KeyCode.F1.value:
                        self._toggle_help()
                    elif key == 's':
                        self._sort_column()
                    elif key == 'f':
                        self._filter_data()
                    elif key == 'b':
                        self._bookmark_current()
                    elif key == 'B':
                        self._show_bookmarks()
                    elif key == 'r':
                        self._refresh_data()
                    
                    live.update(self._render())
        
        return None
    
    def _render(self) -> Layout:
        """Render the interactive table interface"""
        layout = Layout()
        
        # Create main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="table", ratio=1),
            Layout(name="footer", size=5)
        )
        
        # Header with title and status
        header_text = f"Interactive Table Browser - {len(self.filtered_data)} items"
        if self.state.filter_text:
            header_text += f" (filtered: '{self.state.filter_text}')"
        if self.state.search_text:
            header_text += f" (search: '{self.state.search_text}')"
        
        layout["header"].update(Panel(header_text, style="bold blue"))
        
        # Main table
        layout["table"].update(self._render_table())
        
        # Footer with help and status
        layout["footer"].update(self._render_footer())
        
        return layout
    
    def _render_table(self) -> Table:
        """Render the data table"""
        table = Table(show_header=True, header_style="bold magenta")
        
        # Add columns
        for col in self.columns:
            style = "cyan" if col.key == self.state.sort_column else "white"
            table.add_column(
                col.name + (" ↓" if self.state.sort_column == col.key and not self.state.sort_reverse 
                          else " ↑" if self.state.sort_column == col.key else ""),
                style=style,
                width=col.width,
                justify=col.align
            )
        
        # Calculate visible rows
        start_idx = self.state.page * self.state.page_size
        end_idx = min(start_idx + self.state.page_size, len(self.filtered_data))
        
        # Add rows
        for i in range(start_idx, end_idx):
            row_data = self.filtered_data[i]
            row_values = []
            
            for col in self.columns:
                value = str(row_data.get(col.key, ""))
                if len(value) > 30:
                    value = value[:27] + "..."
                row_values.append(value)
            
            # Highlight current row and selected rows
            style = None
            if i == start_idx + self.state.current_row:
                style = "bold yellow on blue"
            elif i in self.state.selected_rows:
                style = "bold green"
            
            table.add_row(*row_values, style=style)
        
        return table
    
    def _render_footer(self) -> Panel:
        """Render footer with help and navigation info"""
        if self.state.show_help:
            help_text = (
                "[bold]Navigation:[/bold] hjkl (vim-style), Enter (select), Space (toggle)\n"
                "[bold]Search:[/bold] / (search), n (next), N (previous)\n"
                "[bold]Actions:[/bold] s (sort), f (filter), b (bookmark), B (show bookmarks)\n"
                "[bold]Other:[/bold] F1 (help), q (quit), r (refresh)"
            )
        else:
            current_page = self.state.page + 1
            total_pages = (len(self.filtered_data) - 1) // self.state.page_size + 1
            help_text = (
                f"Page {current_page}/{total_pages} | "
                f"Row {self.state.current_row + 1}/{min(self.state.page_size, len(self.filtered_data) - self.state.page * self.state.page_size)} | "
                f"Selected: {len(self.state.selected_rows)} | "
                f"Press F1 for help, q to quit"
            )
        
        return Panel(help_text, style="dim")
    
    def _move_up(self):
        """Move cursor up"""
        if self.state.current_row > 0:
            self.state.current_row -= 1
        elif self.state.page > 0:
            self.state.page -= 1
            self.state.current_row = min(self.state.page_size - 1, 
                                       len(self.filtered_data) - self.state.page * self.state.page_size - 1)
    
    def _move_down(self):
        """Move cursor down"""
        max_row = min(self.state.page_size - 1, 
                     len(self.filtered_data) - self.state.page * self.state.page_size - 1)
        
        if self.state.current_row < max_row:
            self.state.current_row += 1
        elif (self.state.page + 1) * self.state.page_size < len(self.filtered_data):
            self.state.page += 1
            self.state.current_row = 0
    
    def _move_left(self):
        """Move cursor left (column navigation)"""
        if self.state.current_col > 0:
            self.state.current_col -= 1
    
    def _move_right(self):
        """Move cursor right (column navigation)"""
        if self.state.current_col < len(self.columns) - 1:
            self.state.current_col += 1
    
    def _page_up(self):
        """Move up one page"""
        if self.state.page > 0:
            self.state.page -= 1
            self.state.current_row = 0
    
    def _page_down(self):
        """Move down one page"""
        max_page = (len(self.filtered_data) - 1) // self.state.page_size
        if self.state.page < max_page:
            self.state.page += 1
            self.state.current_row = 0
    
    def _go_home(self):
        """Go to first row"""
        self.state.page = 0
        self.state.current_row = 0
    
    def _go_end(self):
        """Go to last row"""
        self.state.page = (len(self.filtered_data) - 1) // self.state.page_size
        self.state.current_row = len(self.filtered_data) - self.state.page * self.state.page_size - 1
    
    def _toggle_selection(self):
        """Toggle selection of current row"""
        current_idx = self.state.page * self.state.page_size + self.state.current_row
        if current_idx in self.state.selected_rows:
            self.state.selected_rows.remove(current_idx)
        else:
            self.state.selected_rows.append(current_idx)
    
    def _get_current_row(self) -> Optional[Dict[str, Any]]:
        """Get currently selected row data"""
        current_idx = self.state.page * self.state.page_size + self.state.current_row
        if 0 <= current_idx < len(self.filtered_data):
            return self.filtered_data[current_idx]
        return None
    
    def _toggle_help(self):
        """Toggle help display"""
        self.state.show_help = not self.state.show_help
    
    def _start_search(self):
        """Start search mode"""
        # In a real implementation, this would open a search input
        # For now, we'll use a simple mock search
        self.state.search_text = "mock_search"
        self._perform_search()
    
    def _perform_search(self):
        """Perform search and update results"""
        if not self.state.search_text:
            self.state.search_results = []
            return
        
        self.state.search_results = []
        for i, row in enumerate(self.filtered_data):
            for col in self.columns:
                value = str(row.get(col.key, "")).lower()
                if self.state.search_text.lower() in value:
                    self.state.search_results.append(i)
                    break
        
        self.state.search_index = 0
        if self.state.search_results:
            self._go_to_search_result(0)
    
    def _next_search(self):
        """Go to next search result"""
        if self.state.search_results:
            self.state.search_index = (self.state.search_index + 1) % len(self.state.search_results)
            self._go_to_search_result(self.state.search_index)
    
    def _prev_search(self):
        """Go to previous search result"""
        if self.state.search_results:
            self.state.search_index = (self.state.search_index - 1) % len(self.state.search_results)
            self._go_to_search_result(self.state.search_index)
    
    def _go_to_search_result(self, index: int):
        """Navigate to specific search result"""
        if 0 <= index < len(self.state.search_results):
            result_idx = self.state.search_results[index]
            self.state.page = result_idx // self.state.page_size
            self.state.current_row = result_idx % self.state.page_size
    
    def _sort_column(self):
        """Sort by current column"""
        current_col = self.columns[self.state.current_col]
        if not current_col.sortable:
            return
        
        if self.state.sort_column == current_col.key:
            self.state.sort_reverse = not self.state.sort_reverse
        else:
            self.state.sort_column = current_col.key
            self.state.sort_reverse = False
        
        self.filtered_data.sort(
            key=lambda x: x.get(current_col.key, ""),
            reverse=self.state.sort_reverse
        )
    
    def _filter_data(self):
        """Filter data (mock implementation)"""
        # In a real implementation, this would open a filter input
        self.state.filter_text = "mock_filter"
        self._apply_filter()
    
    def _apply_filter(self):
        """Apply current filter"""
        if not self.state.filter_text:
            self.filtered_data = self.data.copy()
        else:
            self.filtered_data = [
                row for row in self.data
                if any(self.state.filter_text.lower() in str(row.get(col.key, "")).lower()
                      for col in self.columns if col.filterable)
            ]
        
        # Reset navigation
        self.state.page = 0
        self.state.current_row = 0
    
    def _bookmark_current(self):
        """Bookmark current row"""
        current_row = self._get_current_row()
        if current_row:
            bookmark_name = f"bookmark_{len(self.state.bookmarks) + 1}"
            self.state.bookmarks[bookmark_name] = current_row
    
    def _show_bookmarks(self):
        """Show bookmarks (mock implementation)"""
        # In a real implementation, this would show a bookmarks panel
        pass
    
    def _refresh_data(self):
        """Refresh data"""
        # In a real implementation, this would reload data from source
        pass


class InteractiveTreeView:
    """Interactive tree browser with expandable/collapsible nodes"""
    
    def __init__(self, tree_data: Dict[str, Any]):
        self.tree_data = tree_data
        self.state = InteractiveState()
        self.console = get_console()
        self.keyboard = KeyboardHandler()
        self.expanded_nodes: Set[str] = set()
        self.node_list: List[Tuple[Dict[str, Any], int, str]] = []  # Flattened list for navigation
        self._build_node_list()
    
    def run(self) -> Optional[Dict[str, Any]]:
        """Run interactive tree browser"""
        with self.keyboard:
            with Live(self._render(), refresh_per_second=10, screen=True) as live:
                while True:
                    key = self.keyboard.get_key()
                    
                    if key == KeyCode.QUIT.value:
                        break
                    elif key == KeyCode.UP.value:
                        self._move_up()
                    elif key == KeyCode.DOWN.value:
                        self._move_down()
                    elif key == KeyCode.LEFT.value:
                        self._collapse_node()
                    elif key == KeyCode.RIGHT.value:
                        self._expand_node()
                    elif key == KeyCode.ENTER.value:
                        self._toggle_node()
                    elif key == KeyCode.SPACE.value:
                        return self._get_current_node()
                    elif key == KeyCode.F1.value:
                        self._toggle_help()
                    elif key == 'e':
                        self._expand_all()
                    elif key == 'c':
                        self._collapse_all()
                    
                    self._build_node_list()
                    live.update(self._render())
        
        return None
    
    def _render(self) -> Layout:
        """Render the interactive tree interface"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="tree", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Header
        header_text = f"Interactive Tree Browser - {len(self.node_list)} nodes"
        layout["header"].update(Panel(header_text, style="bold blue"))
        
        # Tree
        layout["tree"].update(self._render_tree())
        
        # Footer
        if self.state.show_help:
            help_text = (
                "[bold]Navigation:[/bold] jk (up/down), hl (collapse/expand), Enter (toggle)\n"
                "[bold]Actions:[/bold] Space (select), e (expand all), c (collapse all)\n"
                "[bold]Other:[/bold] F1 (help), q (quit)"
            )
        else:
            help_text = f"Node {self.state.current_row + 1}/{len(self.node_list)} | Press F1 for help, q to quit"
        
        layout["footer"].update(Panel(help_text, style="dim"))
        
        return layout
    
    def _render_tree(self) -> Panel:
        """Render the tree structure"""
        lines = []
        
        for i, (node, depth, path) in enumerate(self.node_list):
            # Create indentation
            indent = "  " * depth
            
            # Node icon and name
            has_children = bool(node.get('children'))
            if has_children:
                if path in self.expanded_nodes:
                    icon = "▼ "
                else:
                    icon = "▶ "
            else:
                icon = "• "
            
            name = node.get('name', 'Unknown')
            value = node.get('value', '')
            if value:
                name += f" ({value})"
            
            # Highlight current node
            if i == self.state.current_row:
                line = f"{indent}[bold yellow on blue]{icon}{name}[/bold yellow on blue]"
            else:
                line = f"{indent}{icon}{name}"
            
            lines.append(line)
        
        content = "\n".join(lines) if lines else "No nodes to display"
        return Panel(content, title="Tree Structure", border_style="green")
    
    def _build_node_list(self):
        """Build flattened list of visible nodes for navigation"""
        self.node_list = []
        self._add_node_to_list(self.tree_data, 0, "root")
    
    def _add_node_to_list(self, node: Dict[str, Any], depth: int, path: str):
        """Recursively add nodes to the navigation list"""
        self.node_list.append((node, depth, path))
        
        # Add children if node is expanded
        if path in self.expanded_nodes:
            children = node.get('children', [])
            for i, child in enumerate(children):
                child_path = f"{path}.{i}"
                self._add_node_to_list(child, depth + 1, child_path)
    
    def _move_up(self):
        """Move cursor up"""
        if self.state.current_row > 0:
            self.state.current_row -= 1
    
    def _move_down(self):
        """Move cursor down"""
        if self.state.current_row < len(self.node_list) - 1:
            self.state.current_row += 1
    
    def _expand_node(self):
        """Expand current node"""
        if self.state.current_row < len(self.node_list):
            _, _, path = self.node_list[self.state.current_row]
            self.expanded_nodes.add(path)
    
    def _collapse_node(self):
        """Collapse current node"""
        if self.state.current_row < len(self.node_list):
            _, _, path = self.node_list[self.state.current_row]
            self.expanded_nodes.discard(path)
    
    def _toggle_node(self):
        """Toggle expansion of current node"""
        if self.state.current_row < len(self.node_list):
            _, _, path = self.node_list[self.state.current_row]
            if path in self.expanded_nodes:
                self.expanded_nodes.discard(path)
            else:
                self.expanded_nodes.add(path)
    
    def _get_current_node(self) -> Optional[Dict[str, Any]]:
        """Get currently selected node"""
        if self.state.current_row < len(self.node_list):
            node, _, _ = self.node_list[self.state.current_row]
            return node
        return None
    
    def _expand_all(self):
        """Expand all nodes"""
        self._expand_all_recursive(self.tree_data, "root")
    
    def _expand_all_recursive(self, node: Dict[str, Any], path: str):
        """Recursively expand all nodes"""
        self.expanded_nodes.add(path)
        children = node.get('children', [])
        for i, child in enumerate(children):
            child_path = f"{path}.{i}"
            self._expand_all_recursive(child, child_path)
    
    def _collapse_all(self):
        """Collapse all nodes"""
        self.expanded_nodes.clear()
    
    def _toggle_help(self):
        """Toggle help display"""
        self.state.show_help = not self.state.show_help


class InteractivePagination:
    """Interactive pagination with customizable page sizes"""
    
    def __init__(self, data: List[Any], page_size: int = 20):
        self.data = data
        self.page_size = page_size
        self.current_page = 0
        self.total_pages = (len(data) - 1) // page_size + 1 if data else 0
    
    def get_current_page_data(self) -> List[Any]:
        """Get data for current page"""
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.data))
        return self.data[start_idx:end_idx]
    
    def next_page(self) -> bool:
        """Go to next page"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            return True
        return False
    
    def prev_page(self) -> bool:
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            return True
        return False
    
    def jump_to_page(self, page: int) -> bool:
        """Jump to specific page"""
        if 0 <= page < self.total_pages:
            self.current_page = page
            return True
        return False
    
    def set_page_size(self, size: int):
        """Change page size"""
        if size > 0:
            self.page_size = size
            self.total_pages = (len(self.data) - 1) // size + 1 if self.data else 0
            # Adjust current page if necessary
            if self.current_page >= self.total_pages:
                self.current_page = max(0, self.total_pages - 1)
    
    def get_page_info(self) -> Dict[str, int]:
        """Get pagination information"""
        return {
            'current_page': self.current_page + 1,
            'total_pages': self.total_pages,
            'page_size': self.page_size,
            'total_items': len(self.data),
            'start_item': self.current_page * self.page_size + 1,
            'end_item': min((self.current_page + 1) * self.page_size, len(self.data))
        }


class BookmarkManager:
    """Manage bookmarks for frequently accessed items"""
    
    def __init__(self):
        self.bookmarks: Dict[str, Dict[str, Any]] = {}
    
    def add_bookmark(self, name: str, data: Dict[str, Any]) -> bool:
        """Add a bookmark"""
        if name not in self.bookmarks:
            self.bookmarks[name] = {
                'data': data,
                'created_at': datetime.now(),
                'access_count': 0
            }
            return True
        return False
    
    def get_bookmark(self, name: str) -> Optional[Dict[str, Any]]:
        """Get bookmark data"""
        if name in self.bookmarks:
            self.bookmarks[name]['access_count'] += 1
            return self.bookmarks[name]['data']
        return None
    
    def remove_bookmark(self, name: str) -> bool:
        """Remove a bookmark"""
        if name in self.bookmarks:
            del self.bookmarks[name]
            return True
        return False
    
    def list_bookmarks(self) -> List[Dict[str, Any]]:
        """List all bookmarks"""
        return [
            {
                'name': name,
                'created_at': info['created_at'],
                'access_count': info['access_count']
            }
            for name, info in self.bookmarks.items()
        ]
    
    def get_most_accessed(self, limit: int = 5) -> List[str]:
        """Get most accessed bookmarks"""
        sorted_bookmarks = sorted(
            self.bookmarks.items(),
            key=lambda x: x[1]['access_count'],
            reverse=True
        )
        return [name for name, _ in sorted_bookmarks[:limit]]


# Utility functions for creating interactive interfaces

def create_interactive_table(data: List[Dict[str, Any]], columns: List[TableColumn]) -> Optional[Dict[str, Any]]:
    """Create and run an interactive table"""
    if not data:
        console = get_console()
        console.print("[yellow]No data available for interactive table[/yellow]")
        return None
    
    table = InteractiveTable(data, columns)
    return table.run()


def create_interactive_tree(tree_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create and run an interactive tree browser"""
    tree = InteractiveTreeView(tree_data)
    return tree.run()


def create_multi_select_interface(items: List[str], title: str = "Select Items") -> List[str]:
    """Create a multi-select interface"""
    # Convert strings to dict format for table
    data = [{'item': item, 'index': i} for i, item in enumerate(items)]
    columns = [
        TableColumn('Item', 'item', sortable=False),
        TableColumn('Index', 'index', width=10, sortable=False)
    ]
    
    table = InteractiveTable(data, columns)
    table.run()
    
    # Return selected items
    selected_indices = table.state.selected_rows
    return [items[i] for i in selected_indices if i < len(items)]