"""
Interactive menu system for ESCAI CLI with enhanced navigation and user experience.
"""

import sys
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.live import Live

from .console import get_console
from .logo import display_logo
from ..session_storage import SessionStorage


class MenuAction(Enum):
    """Menu action types"""
    NAVIGATE = "navigate"
    EXECUTE = "execute"
    BACK = "back"
    EXIT = "exit"


@dataclass
class MenuItem:
    """Menu item configuration"""
    key: str
    title: str
    description: str
    action: MenuAction
    target: Optional[str] = None
    handler: Optional[Callable] = None
    enabled: bool = True
    icon: str = "•"


@dataclass
class Menu:
    """Menu configuration"""
    name: str
    title: str
    description: str
    items: List[MenuItem]
    parent: Optional[str] = None
    show_breadcrumb: bool = True


class InteractiveMenuSystem:
    """
    Enhanced interactive menu system for ESCAI CLI.
    
    Provides:
    - Hierarchical menu navigation with breadcrumbs
    - Context-sensitive help and documentation
    - Quick access to common operations
    - Session management integration
    - User preference persistence
    """
    
    def __init__(self):
        self.console = get_console()
        self.session_storage = SessionStorage()
        self.current_menu = "main"
        self.menu_stack: List[str] = []
        self.breadcrumb: List[str] = ["Main Menu"]
        
        # Initialize menu structure
        self.menus = self._create_menu_structure()
    
    def _create_menu_structure(self) -> Dict[str, Menu]:
        """Create the complete menu structure"""
        menus = {}
        
        # Main Menu
        menus["main"] = Menu(
            name="main",
            title="ESCAI Framework - Main Menu",
            description="Navigate through ESCAI functionality",
            items=[
                MenuItem("1", "Monitor Agents", "Start and manage real-time agent monitoring", 
                        MenuAction.NAVIGATE, "monitor", icon="📊"),
                MenuItem("2", "Analyze Data", "Explore patterns, causality, and predictions", 
                        MenuAction.NAVIGATE, "analyze", icon="🔍"),
                MenuItem("3", "Manage Sessions", "View, replay, and organize monitoring sessions", 
                        MenuAction.NAVIGATE, "sessions", icon="📁"),
                MenuItem("4", "Configuration", "Setup databases, frameworks, and preferences", 
                        MenuAction.NAVIGATE, "config", icon="⚙️"),
                MenuItem("5", "Publication Tools", "Generate academic papers and reports", 
                        MenuAction.NAVIGATE, "publication", icon="📄"),
                MenuItem("6", "Help & Documentation", "Access guides, examples, and support", 
                        MenuAction.NAVIGATE, "help", icon="📚"),
                MenuItem("7", "Exit", "Exit the interactive menu system", 
                        MenuAction.EXIT, icon="🚪")
            ]
        )
        
        # Monitor Menu
        menus["monitor"] = Menu(
            name="monitor",
            title="Agent Monitoring",
            description="Real-time monitoring and observation tools",
            items=[
                MenuItem("1", "Start Monitoring", "Begin monitoring a specific agent", 
                        MenuAction.EXECUTE, handler=self._start_monitoring, icon="▶️"),
                MenuItem("2", "View Active Sessions", "See currently running monitoring sessions", 
                        MenuAction.EXECUTE, handler=self._view_active_sessions, icon="👁️"),
                MenuItem("3", "Stop Monitoring", "Stop an active monitoring session", 
                        MenuAction.EXECUTE, handler=self._stop_monitoring, icon="⏹️"),
                MenuItem("4", "Live Dashboard", "View real-time monitoring dashboard", 
                        MenuAction.EXECUTE, handler=self._live_dashboard, icon="📈"),
                MenuItem("5", "Back to Main Menu", "Return to the main menu", 
                        MenuAction.BACK, icon="⬅️")
            ],
            parent="main"
        )
        
        # Analyze Menu
        menus["analyze"] = Menu(
            name="analyze",
            title="Data Analysis",
            description="Pattern analysis and causal relationship exploration",
            items=[
                MenuItem("1", "Pattern Analysis", "Analyze behavioral patterns in agent data", 
                        MenuAction.EXECUTE, handler=self._pattern_analysis, icon="🔄"),
                MenuItem("2", "Causal Relationships", "Explore cause-effect relationships", 
                        MenuAction.EXECUTE, handler=self._causal_analysis, icon="🔗"),
                MenuItem("3", "Statistical Analysis", "Perform statistical analysis on collected data", 
                        MenuAction.EXECUTE, handler=self._statistical_analysis, icon="📊"),
                MenuItem("4", "Prediction Models", "Generate and evaluate performance predictions", 
                        MenuAction.EXECUTE, handler=self._prediction_models, icon="🔮"),
                MenuItem("5", "Back to Main Menu", "Return to the main menu", 
                        MenuAction.BACK, icon="⬅️")
            ],
            parent="main"
        )
        
        # Sessions Menu
        menus["sessions"] = Menu(
            name="sessions",
            title="Session Management",
            description="Session history, replay, and organization tools",
            items=[
                MenuItem("1", "List Sessions", "View all monitoring sessions", 
                        MenuAction.EXECUTE, handler=self._list_sessions, icon="📋"),
                MenuItem("2", "Session Details", "View detailed information about a session", 
                        MenuAction.EXECUTE, handler=self._session_details, icon="🔍"),
                MenuItem("3", "Replay Session", "Replay commands from a previous session", 
                        MenuAction.EXECUTE, handler=self._replay_session, icon="🔄"),
                MenuItem("4", "Compare Sessions", "Compare multiple sessions side by side", 
                        MenuAction.EXECUTE, handler=self._compare_sessions, icon="⚖️"),
                MenuItem("5", "Search Sessions", "Search sessions by text query", 
                        MenuAction.EXECUTE, handler=self._search_sessions, icon="🔍"),
                MenuItem("6", "Session Statistics", "View session statistics and summaries", 
                        MenuAction.EXECUTE, handler=self._session_statistics, icon="📊"),
                MenuItem("7", "Back to Main Menu", "Return to the main menu", 
                        MenuAction.BACK, icon="⬅️")
            ],
            parent="main"
        )
        
        # Config Menu
        menus["config"] = Menu(
            name="config",
            title="Configuration",
            description="System configuration and preferences",
            items=[
                MenuItem("1", "Database Setup", "Configure database connections", 
                        MenuAction.EXECUTE, handler=self._database_setup, icon="🗄️"),
                MenuItem("2", "Framework Configuration", "Configure agent framework integrations", 
                        MenuAction.EXECUTE, handler=self._framework_config, icon="🔧"),
                MenuItem("3", "Output Preferences", "Configure output formatting and preferences", 
                        MenuAction.EXECUTE, handler=self._output_preferences, icon="🎨"),
                MenuItem("4", "View Current Config", "Display current configuration settings", 
                        MenuAction.EXECUTE, handler=self._view_config, icon="👁️"),
                MenuItem("5", "Export Configuration", "Export configuration for sharing", 
                        MenuAction.EXECUTE, handler=self._export_config, icon="📤"),
                MenuItem("6", "Back to Main Menu", "Return to the main menu", 
                        MenuAction.BACK, icon="⬅️")
            ],
            parent="main"
        )
        
        # Publication Menu
        menus["publication"] = Menu(
            name="publication",
            title="Publication Tools",
            description="Generate academic papers, reports, and citations",
            items=[
                MenuItem("1", "Generate Paper", "Create academic paper from analysis results", 
                        MenuAction.EXECUTE, handler=self._generate_paper, icon="📝"),
                MenuItem("2", "Statistical Report", "Generate comprehensive statistical report", 
                        MenuAction.EXECUTE, handler=self._generate_report, icon="📊"),
                MenuItem("3", "Manage Citations", "View and manage bibliography citations", 
                        MenuAction.EXECUTE, handler=self._manage_citations, icon="📚"),
                MenuItem("4", "LaTeX Templates", "View and manage LaTeX templates", 
                        MenuAction.EXECUTE, handler=self._latex_templates, icon="📄"),
                MenuItem("5", "Export Bibliography", "Generate bibliography in various formats", 
                        MenuAction.EXECUTE, handler=self._export_bibliography, icon="📤"),
                MenuItem("6", "Back to Main Menu", "Return to the main menu", 
                        MenuAction.BACK, icon="⬅️")
            ],
            parent="main"
        )
        
        # Help Menu
        menus["help"] = Menu(
            name="help",
            title="Help & Documentation",
            description="Documentation, guides, and support resources",
            items=[
                MenuItem("1", "Getting Started Guide", "Learn the basics of using ESCAI", 
                        MenuAction.EXECUTE, handler=self._getting_started, icon="🚀"),
                MenuItem("2", "Command Reference", "Detailed command documentation", 
                        MenuAction.EXECUTE, handler=self._command_reference, icon="📖"),
                MenuItem("3", "Examples & Tutorials", "Practical examples and step-by-step tutorials", 
                        MenuAction.EXECUTE, handler=self._examples_tutorials, icon="🎓"),
                MenuItem("4", "Framework Integration", "Guide to integrating with agent frameworks", 
                        MenuAction.EXECUTE, handler=self._framework_integration, icon="🔌"),
                MenuItem("5", "Troubleshooting", "Common issues and solutions", 
                        MenuAction.EXECUTE, handler=self._troubleshooting, icon="🔧"),
                MenuItem("6", "About ESCAI", "Information about the ESCAI Framework", 
                        MenuAction.EXECUTE, handler=self._about_escai, icon="ℹ️"),
                MenuItem("7", "Back to Main Menu", "Return to the main menu", 
                        MenuAction.BACK, icon="⬅️")
            ],
            parent="main"
        )
        
        return menus
    
    def run(self):
        """Run the interactive menu system"""
        display_logo()
        self.console.print("\n[bold cyan]Welcome to ESCAI Interactive Menu System![/bold cyan]")
        self.console.print("[dim]Navigate using numbers, type 'help' for assistance, or 'q' to quit[/dim]\n")
        
        try:
            while True:
                if not self._display_current_menu():
                    break
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Exiting interactive menu...[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Error in interactive menu: {str(e)}[/red]")
    
    def _display_current_menu(self) -> bool:
        """Display current menu and handle user input. Returns False to exit."""
        menu = self.menus[self.current_menu]
        
        # Show breadcrumb navigation
        if menu.show_breadcrumb and len(self.breadcrumb) > 1:
            breadcrumb_text = " > ".join(self.breadcrumb)
            self.console.print(f"[dim]📍 {breadcrumb_text}[/dim]\n")
        
        # Create menu panel
        menu_content = []
        menu_content.append(f"[dim]{menu.description}[/dim]\n")
        
        for item in menu.items:
            if item.enabled:
                menu_content.append(f"[accent]{item.key}.[/accent] {item.icon} [bold]{item.title}[/bold]")
                menu_content.append(f"   [dim]{item.description}[/dim]\n")
        
        menu_panel = Panel(
            "\n".join(menu_content),
            title=f"[bold]{menu.title}[/bold]",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print(menu_panel)
        
        # Get user input
        valid_choices = [item.key for item in menu.items if item.enabled]
        valid_choices.extend(["help", "h", "back", "b", "main", "m", "quit", "q", "exit"])
        
        try:
            choice = Prompt.ask(
                "\n[info]Select an option[/info]",
                choices=valid_choices,
                default="help",
                show_choices=False
            )
        except (EOFError, KeyboardInterrupt):
            return False
        
        return self._handle_menu_choice(choice)
    
    def _handle_menu_choice(self, choice: str) -> bool:
        """Handle menu choice. Returns False to exit."""
        # Handle special commands
        if choice in ["help", "h"]:
            self._show_contextual_help()
            return True
        elif choice in ["back", "b"]:
            return self._navigate_back()
        elif choice in ["main", "m"]:
            return self._navigate_to_main()
        elif choice in ["quit", "q", "exit"]:
            if Confirm.ask("[yellow]Are you sure you want to exit?[/yellow]", default=False):
                self.console.print("[success]Thank you for using ESCAI Framework![/success]")
                return False
            return True
        
        # Handle menu item selection
        menu = self.menus[self.current_menu]
        selected_item = None
        
        for item in menu.items:
            if item.key == choice and item.enabled:
                selected_item = item
                break
        
        if not selected_item:
            self.console.print(f"[error]Invalid selection: {choice}[/error]")
            return True
        
        # Execute the selected action
        try:
            if selected_item.action == MenuAction.NAVIGATE:
                return self._navigate_to_menu(selected_item.target, selected_item.title)
            elif selected_item.action == MenuAction.EXECUTE:
                if selected_item.handler:
                    selected_item.handler()
                return True
            elif selected_item.action == MenuAction.BACK:
                return self._navigate_back()
            elif selected_item.action == MenuAction.EXIT:
                if Confirm.ask("[yellow]Are you sure you want to exit?[/yellow]", default=False):
                    self.console.print("[success]Thank you for using ESCAI Framework![/success]")
                    return False
                return True
        except Exception as e:
            self.console.print(f"[error]Error executing action: {str(e)}[/error]")
            return True
        
        return True
    
    def _navigate_to_menu(self, menu_name: str, title: str) -> bool:
        """Navigate to a submenu"""
        if menu_name in self.menus:
            self.menu_stack.append(self.current_menu)
            self.current_menu = menu_name
            self.breadcrumb.append(title)
        return True
    
    def _navigate_back(self) -> bool:
        """Navigate back to previous menu"""
        if self.menu_stack:
            self.current_menu = self.menu_stack.pop()
            if self.breadcrumb:
                self.breadcrumb.pop()
        return True
    
    def _navigate_to_main(self) -> bool:
        """Navigate to main menu"""
        self.current_menu = "main"
        self.menu_stack.clear()
        self.breadcrumb = ["Main Menu"]
        return True
    
    def _show_contextual_help(self):
        """Show help relevant to current menu context"""
        menu = self.menus[self.current_menu]
        
        help_content = f"""
[bold]Interactive Menu Help[/bold]

[accent]Current Menu:[/accent] {menu.title}
{menu.description}

[accent]Navigation:[/accent]
• Use numbers (1-{len([i for i in menu.items if i.enabled])}) to select menu options
• Type 'help' or 'h' to show this help
• Type 'back' or 'b' to go back one level
• Type 'main' or 'm' to return to main menu
• Type 'quit', 'q', or 'exit' to quit

[accent]Available Options:[/accent]"""
        
        for item in menu.items:
            if item.enabled:
                help_content += f"\n• {item.key}. {item.title} - {item.description}"
        
        help_content += """

[accent]Tips:[/accent]
• Breadcrumb navigation shows your current location
• Most operations can be cancelled with Ctrl+C
• Use the session management menu to track your work
"""
        
        help_panel = Panel(
            help_content,
            title="[bold]Help[/bold]",
            border_style="yellow"
        )
        self.console.print(help_panel)
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    # Menu action handlers (placeholders for now - will be enhanced in subsequent tasks)
    def _start_monitoring(self):
        """Start monitoring workflow"""
        self.console.print("[info]Starting agent monitoring workflow...[/info]")
        
        # Get agent details
        agent_id = Prompt.ask("Enter agent ID")
        framework = Prompt.ask("Select framework", 
                              choices=["langchain", "autogen", "crewai", "openai"],
                              default="langchain")
        description = Prompt.ask("Enter description (optional)", default="")
        
        # This would integrate with actual monitoring commands
        self.console.print(f"[success]✅ Would start monitoring {agent_id} ({framework})[/success]")
        self.console.print("[dim]This will be implemented in subsequent tasks[/dim]")
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _view_active_sessions(self):
        """View active monitoring sessions"""
        sessions = self.session_storage.list_sessions(status="active", limit=10)
        
        if not sessions:
            self.console.print("[info]No active monitoring sessions found[/info]")
        else:
            table = Table(title="Active Sessions", show_header=True, header_style="bold green")
            table.add_column("Session ID", style="yellow")
            table.add_column("Agent ID", style="blue")
            table.add_column("Framework", style="green")
            table.add_column("Start Time", style="muted")
            
            for session in sessions:
                table.add_row(
                    session['session_id'][:12] + "...",
                    session.get('agent_id', 'N/A'),
                    session.get('framework', 'N/A'),
                    session.get('start_time', 'N/A')[:16]
                )
            
            self.console.print(table)
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _stop_monitoring(self):
        """Stop monitoring workflow"""
        active_sessions = self.session_storage.list_sessions(status="active", limit=20)
        
        if not active_sessions:
            self.console.print("[info]No active sessions to stop[/info]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
            return
        
        # Show active sessions
        self.console.print("[info]Active sessions:[/info]")
        for i, session in enumerate(active_sessions, 1):
            self.console.print(f"{i}. {session['session_id'][:12]}... - {session.get('agent_id', 'N/A')}")
        
        try:
            choice = IntPrompt.ask("Select session to stop (0 to cancel)", 
                                 default=0, 
                                 show_default=True)
            
            if choice > 0 and choice <= len(active_sessions):
                session = active_sessions[choice - 1]
                if Confirm.ask(f"Stop session {session['session_id'][:12]}...?", default=True):
                    self.session_storage.end_session(session['session_id'])
                    self.console.print(f"[success]✅ Session stopped[/success]")
            else:
                self.console.print("[info]Cancelled[/info]")
        except (ValueError, KeyboardInterrupt):
            self.console.print("[info]Cancelled[/info]")
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _live_dashboard(self):
        """Live monitoring dashboard"""
        self.console.print("[info]Live dashboard functionality - Implementation pending[/info]")
        self.console.print("[dim]This will show real-time monitoring data in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _pattern_analysis(self):
        """Pattern analysis workflow"""
        self.console.print("[info]Pattern analysis functionality - Implementation pending[/info]")
        self.console.print("[dim]This will analyze behavioral patterns in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _causal_analysis(self):
        """Causal analysis workflow"""
        self.console.print("[info]Causal analysis functionality - Implementation pending[/info]")
        self.console.print("[dim]This will explore causal relationships in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _statistical_analysis(self):
        """Statistical analysis workflow"""
        self.console.print("[info]Statistical analysis functionality - Implementation pending[/info]")
        self.console.print("[dim]This will perform statistical analysis in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _prediction_models(self):
        """Prediction models workflow"""
        self.console.print("[info]Prediction models functionality - Implementation pending[/info]")
        self.console.print("[dim]This will generate predictions in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _list_sessions(self):
        """List sessions workflow"""
        sessions = self.session_storage.list_sessions(limit=20)
        
        if not sessions:
            self.console.print("[info]No sessions found[/info]")
        else:
            table = Table(title="Recent Sessions", show_header=True, header_style="bold cyan")
            table.add_column("Session ID", style="yellow")
            table.add_column("Agent ID", style="blue")
            table.add_column("Framework", style="green")
            table.add_column("Status", justify="center")
            table.add_column("Start Time", style="muted")
            
            for session in sessions:
                status = session.get('status', 'unknown')
                status_icon = "🟢" if status == 'active' else "🔴" if status == 'completed' else "🟡"
                
                table.add_row(
                    session['session_id'][:12] + "...",
                    session.get('agent_id', 'N/A'),
                    session.get('framework', 'N/A'),
                    f"{status_icon} {status}",
                    session.get('start_time', 'N/A')[:16]
                )
            
            self.console.print(table)
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _session_details(self):
        """Session details workflow"""
        session_id = Prompt.ask("Enter session ID (or partial ID)")
        
        # Try to find session by partial ID
        sessions = self.session_storage.list_sessions(limit=100)
        matching_sessions = [s for s in sessions if s['session_id'].startswith(session_id)]
        
        if not matching_sessions:
            self.console.print(f"[error]No sessions found matching '{session_id}'[/error]")
        elif len(matching_sessions) == 1:
            session = matching_sessions[0]
            self._display_session_details(session)
        else:
            self.console.print(f"[warning]Multiple sessions match '{session_id}':[/warning]")
            for i, session in enumerate(matching_sessions[:5], 1):
                self.console.print(f"{i}. {session['session_id']} - {session.get('agent_id', 'N/A')}")
            
            try:
                choice = IntPrompt.ask("Select session (0 to cancel)", default=0)
                if choice > 0 and choice <= len(matching_sessions):
                    self._display_session_details(matching_sessions[choice - 1])
            except (ValueError, KeyboardInterrupt):
                pass
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _display_session_details(self, session: Dict[str, Any]):
        """Display detailed session information"""
        details_content = [
            f"[bold]Session ID:[/bold] {session['session_id']}",
            f"[bold]Agent ID:[/bold] {session.get('agent_id', 'N/A')}",
            f"[bold]Framework:[/bold] {session.get('framework', 'N/A')}",
            f"[bold]Status:[/bold] {session.get('status', 'N/A')}",
            f"[bold]Start Time:[/bold] {session.get('start_time', 'N/A')}",
        ]
        
        if session.get('end_time'):
            details_content.append(f"[bold]End Time:[/bold] {session.get('end_time')}")
        
        if session.get('description'):
            details_content.append(f"[bold]Description:[/bold] {session.get('description')}")
        
        if session.get('tags'):
            details_content.append(f"[bold]Tags:[/bold] {', '.join(session.get('tags', []))}")
        
        # Get command count
        commands = self.session_storage.get_command_history(session['session_id'])
        details_content.append(f"[bold]Commands Executed:[/bold] {len(commands)}")
        
        details_panel = Panel(
            "\n".join(details_content),
            title="Session Details",
            border_style="blue"
        )
        self.console.print(details_panel)
    
    def _replay_session(self):
        """Session replay workflow"""
        self.console.print("[info]Session replay functionality - Implementation pending[/info]")
        self.console.print("[dim]This will replay session commands in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _compare_sessions(self):
        """Session comparison workflow"""
        self.console.print("[info]Session comparison functionality - Implementation pending[/info]")
        self.console.print("[dim]This will compare multiple sessions in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _search_sessions(self):
        """Session search workflow"""
        query = Prompt.ask("Enter search query")
        sessions = self.session_storage.search_sessions(query)
        
        if not sessions:
            self.console.print(f"[info]No sessions found matching '{query}'[/info]")
        else:
            self.console.print(f"[info]Found {len(sessions)} sessions matching '{query}':[/info]")
            
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Session ID", style="yellow")
            table.add_column("Agent ID", style="blue")
            table.add_column("Framework", style="green")
            table.add_column("Description", style="white")
            
            for session in sessions[:10]:  # Show first 10
                table.add_row(
                    session['session_id'][:12] + "...",
                    session.get('agent_id', 'N/A'),
                    session.get('framework', 'N/A'),
                    session.get('description', 'N/A')[:30] + "..." if len(session.get('description', '')) > 30 else session.get('description', 'N/A')
                )
            
            self.console.print(table)
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _session_statistics(self):
        """Session statistics workflow"""
        stats = self.session_storage.get_session_statistics()
        
        stats_content = [
            f"[bold]Total Sessions:[/bold] {stats['total_sessions']}",
            f"[bold]Active Sessions:[/bold] {stats['active_sessions']}",
            f"[bold]Completed Sessions:[/bold] {stats['completed_sessions']}",
            f"[bold]Unique Agents:[/bold] {stats['unique_agents']}",
            f"[bold]Frameworks Used:[/bold] {stats['frameworks_used']}",
            f"[bold]Total Commands:[/bold] {stats['total_commands']}",
            f"[bold]Successful Commands:[/bold] {stats['successful_commands']}",
        ]
        
        if stats['total_commands'] > 0:
            success_rate = stats['successful_commands'] / stats['total_commands'] * 100
            stats_content.append(f"[bold]Success Rate:[/bold] {success_rate:.1f}%")
        
        if stats['avg_execution_time']:
            stats_content.append(f"[bold]Average Execution Time:[/bold] {stats['avg_execution_time']:.3f}s")
        
        stats_panel = Panel(
            "\n".join(stats_content),
            title="Session Statistics",
            border_style="green"
        )
        self.console.print(stats_panel)
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    # Configuration handlers
    def _database_setup(self):
        """Database setup workflow"""
        self.console.print("[info]Database setup functionality - Implementation pending[/info]")
        self.console.print("[dim]This will configure database connections in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _framework_config(self):
        """Framework configuration workflow"""
        self.console.print("[info]Framework configuration functionality - Implementation pending[/info]")
        self.console.print("[dim]This will configure agent frameworks in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _output_preferences(self):
        """Output preferences workflow"""
        self.console.print("[info]Output preferences functionality - Implementation pending[/info]")
        self.console.print("[dim]This will configure output formatting in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _view_config(self):
        """View configuration workflow"""
        self.console.print("[info]View configuration functionality - Implementation pending[/info]")
        self.console.print("[dim]This will display current configuration in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _export_config(self):
        """Export configuration workflow"""
        self.console.print("[info]Export configuration functionality - Implementation pending[/info]")
        self.console.print("[dim]This will export configuration in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    # Help handlers
    def _getting_started(self):
        """Getting started guide"""
        guide_content = """
[bold]Getting Started with ESCAI Framework[/bold]

[accent]1. Basic Monitoring:[/accent]
   • Use 'escai monitor start' to begin monitoring an agent
   • Specify the agent ID and framework (langchain, autogen, crewai, openai)
   • Monitor real-time epistemic state changes and behavioral patterns

[accent]2. Data Analysis:[/accent]
   • Use 'escai analyze patterns' to discover behavioral patterns
   • Use 'escai analyze causal' to explore cause-effect relationships
   • Generate statistical reports and predictions

[accent]3. Session Management:[/accent]
   • All monitoring creates sessions that are automatically saved
   • Use 'escai session list' to view your monitoring history
   • Replay sessions to reproduce experiments

[accent]4. Configuration:[/accent]
   • Configure database connections for data persistence
   • Set up framework-specific monitoring parameters
   • Customize output formats for your research needs

[accent]Next Steps:[/accent]
   • Try the interactive menu system with 'escai --interactive'
   • Read the command reference for detailed usage
   • Explore examples and tutorials for your specific use case
"""
        
        guide_panel = Panel(
            guide_content,
            title="Getting Started Guide",
            border_style="green"
        )
        self.console.print(guide_panel)
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _command_reference(self):
        """Command reference"""
        self.console.print("[info]Command reference functionality - Implementation pending[/info]")
        self.console.print("[dim]This will show detailed command documentation in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _examples_tutorials(self):
        """Examples and tutorials"""
        self.console.print("[info]Examples and tutorials functionality - Implementation pending[/info]")
        self.console.print("[dim]This will show practical examples in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _framework_integration(self):
        """Framework integration guide"""
        self.console.print("[info]Framework integration functionality - Implementation pending[/info]")
        self.console.print("[dim]This will show integration guides in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _troubleshooting(self):
        """Troubleshooting guide"""
        self.console.print("[info]Troubleshooting functionality - Implementation pending[/info]")
        self.console.print("[dim]This will show troubleshooting guides in subsequent tasks[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _about_escai(self):
        """About ESCAI"""
        about_content = """
[bold]ESCAI Framework[/bold]
[dim]Epistemic State and Causal Analysis Intelligence[/dim]

[accent]Version:[/accent] 1.0.0
[accent]Purpose:[/accent] Monitor autonomous agent cognition in real-time

[accent]Key Features:[/accent]
• Real-time epistemic state monitoring
• Behavioral pattern analysis
• Causal relationship discovery
• Performance prediction
• Multi-framework support (LangChain, AutoGen, CrewAI, OpenAI)
• Publication-ready outputs

[accent]Research Applications:[/accent]
• Agent behavior analysis
• Cognitive architecture evaluation
• Multi-agent system studies
• AI safety research
• Performance optimization

[accent]Support:[/accent]
• Documentation: Available through help menu
• Examples: Practical use cases and tutorials
• Community: Framework integration guides
"""
        
        about_panel = Panel(
            about_content,
            title="About ESCAI Framework",
            border_style="blue"
        )
        self.console.print(about_panel)
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    # Publication handlers
    def _generate_paper(self):
        """Generate academic paper workflow"""
        self.console.print("[bold cyan]Generate Academic Paper[/bold cyan]\n")
        
        # Get input file
        input_file = Prompt.ask("Enter path to analysis data file (JSON/CSV)")
        if not input_file:
            self.console.print("[error]Input file is required[/error]")
            return
        
        # Get output file
        output_file = Prompt.ask("Enter output file path", default="paper.tex")
        
        # Get template
        templates = ["ieee_conference", "acm", "springer_lncs", "elsevier", "generic"]
        template_choice = Prompt.ask(
            "Select template",
            choices=templates,
            default="ieee_conference"
        )
        
        # Get paper details
        title = Prompt.ask("Enter paper title", default="ESCAI Framework Analysis Results")
        authors = Prompt.ask("Enter authors (comma-separated)", default="Research Team")
        abstract = Prompt.ask("Enter abstract", default="Analysis results from ESCAI Framework monitoring.")
        
        # Execute command
        try:
            import subprocess
            cmd = [
                "escai", "publication", "generate",
                "--input", input_file,
                "--output", output_file,
                "--format", "latex",
                "--template", template_choice,
                "--title", title,
                "--authors", authors,
                "--abstract", abstract
            ]
            
            self.console.print("[info]Generating paper...[/info]")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.console.print(f"[success]Paper generated successfully: {output_file}[/success]")
            else:
                self.console.print(f"[error]Error generating paper: {result.stderr}[/error]")
        
        except Exception as e:
            self.console.print(f"[error]Error: {str(e)}[/error]")
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _generate_report(self):
        """Generate statistical report workflow"""
        self.console.print("[bold cyan]Generate Statistical Report[/bold cyan]\n")
        
        # Get input file
        input_file = Prompt.ask("Enter path to analysis data file (JSON/CSV)")
        if not input_file:
            self.console.print("[error]Input file is required[/error]")
            return
        
        # Get output file
        output_file = Prompt.ask("Enter output file path", default="statistical_report.tex")
        
        # Get format
        formats = ["latex", "markdown", "html"]
        format_choice = Prompt.ask(
            "Select output format",
            choices=formats,
            default="latex"
        )
        
        # Execute command
        try:
            import subprocess
            cmd = [
                "escai", "publication", "report",
                "--input", input_file,
                "--output", output_file,
                "--format", format_choice
            ]
            
            self.console.print("[info]Generating statistical report...[/info]")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.console.print(f"[success]Report generated successfully: {output_file}[/success]")
            else:
                self.console.print(f"[error]Error generating report: {result.stderr}[/error]")
        
        except Exception as e:
            self.console.print(f"[error]Error: {str(e)}[/error]")
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _manage_citations(self):
        """Manage citations workflow"""
        self.console.print("[bold cyan]Manage Citations[/bold cyan]\n")
        
        while True:
            action = Prompt.ask(
                "What would you like to do?",
                choices=["search", "list_methodologies", "back"],
                default="search"
            )
            
            if action == "back":
                break
            elif action == "search":
                query = Prompt.ask("Enter search query")
                if query:
                    try:
                        import subprocess
                        cmd = ["escai", "publication", "citations", "--search", query]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            self.console.print(result.stdout)
                        else:
                            self.console.print(f"[error]Error: {result.stderr}[/error]")
                    except Exception as e:
                        self.console.print(f"[error]Error: {str(e)}[/error]")
            
            elif action == "list_methodologies":
                try:
                    import subprocess
                    cmd = ["escai", "publication", "citations"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.console.print(result.stdout)
                    else:
                        self.console.print(f"[error]Error: {result.stderr}[/error]")
                except Exception as e:
                    self.console.print(f"[error]Error: {str(e)}[/error]")
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _latex_templates(self):
        """LaTeX templates workflow"""
        self.console.print("[bold cyan]LaTeX Templates[/bold cyan]\n")
        
        action = Prompt.ask(
            "What would you like to do?",
            choices=["list", "show_details", "generate_sample"],
            default="list"
        )
        
        try:
            import subprocess
            
            if action == "list":
                cmd = ["escai", "publication", "templates", "--list"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.console.print(result.stdout)
                else:
                    self.console.print(f"[error]Error: {result.stderr}[/error]")
            
            elif action == "show_details":
                template = Prompt.ask(
                    "Enter template name",
                    choices=["ieee_conference", "acm", "springer_lncs", "elsevier", "generic"],
                    default="ieee_conference"
                )
                
                cmd = ["escai", "publication", "templates", "--show", template]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.console.print(result.stdout)
                else:
                    self.console.print(f"[error]Error: {result.stderr}[/error]")
            
            elif action == "generate_sample":
                output_file = Prompt.ask("Enter output file path", default="sample_paper.tex")
                
                cmd = ["escai", "publication", "templates", "--output", output_file]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.console.print(f"[success]Sample document generated: {output_file}[/success]")
                else:
                    self.console.print(f"[error]Error: {result.stderr}[/error]")
        
        except Exception as e:
            self.console.print(f"[error]Error: {str(e)}[/error]")
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
    
    def _export_bibliography(self):
        """Export bibliography workflow"""
        self.console.print("[bold cyan]Export Bibliography[/bold cyan]\n")
        
        # Get methodologies
        methodologies = Prompt.ask(
            "Enter methodologies (comma-separated)",
            default="statistical_analysis,epistemic_extraction"
        )
        
        # Get format
        formats = ["bibtex", "apa", "ieee"]
        format_choice = Prompt.ask(
            "Select bibliography format",
            choices=formats,
            default="bibtex"
        )
        
        # Get output file
        output_file = Prompt.ask("Enter output file path", default=f"bibliography.{format_choice}")
        
        # Execute command
        try:
            import subprocess
            cmd = [
                "escai", "publication", "citations",
                "--format", format_choice,
                "--output", output_file
            ]
            
            # Add methodologies
            for methodology in methodologies.split(","):
                cmd.extend(["--methodology", methodology.strip()])
            
            self.console.print("[info]Generating bibliography...[/info]")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.console.print(f"[success]Bibliography generated: {output_file}[/success]")
            else:
                self.console.print(f"[error]Error generating bibliography: {result.stderr}[/error]")
        
        except Exception as e:
            self.console.print(f"[error]Error: {str(e)}[/error]")
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def launch_interactive_menu():
    """Launch the interactive menu system"""
    menu_system = InteractiveMenuSystem()
    menu_system.run()