"""
User experience and productivity features for ESCAI CLI
"""

import os
import json
import logging

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re

from rich.console import Console
from rich.theme import Theme
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

from .console import get_console


@dataclass
class CommandHistory:
    """Command history management"""
    commands: List[str] = field(default_factory=list)
    max_history: int = 1000
    favorites: List[str] = field(default_factory=list)
    
    def add_command(self, command: str):
        """Add command to history"""
        if command and command not in self.commands:
            self.commands.append(command)
            if len(self.commands) > self.max_history:
                self.commands.pop(0)
    
    def search_history(self, pattern: str) -> List[str]:
        """Search command history"""
        matches = []
        for cmd in self.commands:
            if pattern.lower() in cmd.lower():
                matches.append(cmd)
        return matches
    
    def add_favorite(self, command: str):
        """Add command to favorites"""
        if command not in self.favorites:
            self.favorites.append(command)
    
    def remove_favorite(self, command: str):
        """Remove command from favorites"""
        if command in self.favorites:
            self.favorites.remove(command)
    
    def get_recent(self, count: int = 10) -> List[str]:
        """Get recent commands"""
        return self.commands[-count:] if self.commands else []


@dataclass
class AutoCompleteRule:
    """Auto-completion rule"""
    pattern: str
    suggestions: List[str]
    context: Optional[str] = None
    dynamic: bool = False
    generator: Optional[Callable[[], List[str]]] = None


class AutoCompleter:
    """Intelligent auto-completion system"""
    
    def __init__(self) -> None:
        self.rules: List[AutoCompleteRule] = []
        self.command_history = CommandHistory()
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default auto-completion rules"""
        # Command suggestions
        self.add_rule("escai", [
            "monitor", "analyze", "config", "session"
        ])
        
        self.add_rule("escai monitor", [
            "start", "stop", "status", "epistemic", "dashboard", "logs", "live"
        ])
        
        self.add_rule("escai analyze", [
            "patterns", "causal", "predictions", "events", "visualize", 
            "epistemic", "heatmap", "causal-scatter", "tree", "tree-explorer",
            "interactive", "query", "stats", "correlate", "timeseries", "progress"
        ])
        
        self.add_rule("escai config", [
            "setup", "show", "set", "get", "reset"
        ])
        
        self.add_rule("escai session", [
            "list", "show", "stop", "export"
        ])
        
        # Framework options
        self.add_rule("--framework", [
            "langchain", "autogen", "crewai", "openai"
        ])
        
        # Chart type options
        self.add_rule("--chart-type", [
            "bar", "line", "histogram", "scatter", "heatmap"
        ])
        
        # Common field names
        self.add_rule("--field", [
            "response_time", "success_rate", "events", "cpu_usage", 
            "memory_usage", "confidence", "age", "status"
        ])
    
    def add_rule(self, pattern: str, suggestions: List[str], 
                 context: Optional[str] = None, dynamic: bool = False,
                 generator: Optional[Callable[[], List[str]]] = None):
        """Add auto-completion rule"""
        rule = AutoCompleteRule(pattern, suggestions, context, dynamic, generator)
        self.rules.append(rule)
    
    def get_suggestions(self, input_text: str, context: Optional[str] = None) -> List[str]:
        """Get auto-completion suggestions"""
        suggestions = set()
        
        # Check rules
        for rule in self.rules:
            if self._matches_pattern(input_text, rule.pattern):
                if rule.dynamic and rule.generator:
                    suggestions.update(rule.generator())
                else:
                    suggestions.update(rule.suggestions)
        
        # Add from command history
        history_matches = self.command_history.search_history(input_text)
        suggestions.update(history_matches[:5])  # Limit to 5 history matches
        
        # Filter and sort suggestions
        filtered = [s for s in suggestions if input_text.lower() in s.lower()]
        return sorted(filtered)[:10]  # Limit to 10 suggestions
    
    def _matches_pattern(self, input_text: str, pattern: str) -> bool:
        """Check if input matches pattern"""
        # Exact pattern match
        if pattern == input_text:
            return True
        
        # Check if input starts with pattern
        if input_text.startswith(pattern):
            return True
        
        # Check word-by-word matching
        words = input_text.split()
        pattern_words = pattern.split()
        
        if len(words) >= len(pattern_words):
            return words[:len(pattern_words)] == pattern_words
        
        # Partial word matching
        if len(pattern_words) == 1 and len(words) >= 1:
            return words[-1].startswith(pattern_words[0]) or pattern_words[0] in input_text
        
        return False


@dataclass
class ThemeConfig:
    """Theme configuration"""
    name: str
    colors: Dict[str, str]
    styles: Dict[str, str]
    
    def to_rich_theme(self) -> Theme:
        """Convert to Rich theme"""
        theme_dict = {}
        theme_dict.update(self.colors)
        theme_dict.update(self.styles)
        return Theme(theme_dict)


class ThemeManager:
    """Theme and color scheme management"""
    
    def __init__(self) -> None:
        self.themes: Dict[str, ThemeConfig] = {}
        self.current_theme = "default"
        self._setup_default_themes()
    
    def _setup_default_themes(self) -> None:
        """Setup default themes"""
        # Default theme
        self.themes["default"] = ThemeConfig(
            name="Default",
            colors={
                "primary": "cyan",
                "secondary": "blue",
                "success": "green",
                "warning": "yellow",
                "error": "red",
                "info": "blue",
                "muted": "dim white"
            },
            styles={
                "header": "bold cyan",
                "subheader": "bold blue",
                "highlight": "bold yellow",
                "accent": "magenta"
            }
        )
        
        # Dark theme
        self.themes["dark"] = ThemeConfig(
            name="Dark",
            colors={
                "primary": "bright_cyan",
                "secondary": "bright_blue",
                "success": "bright_green",
                "warning": "bright_yellow",
                "error": "bright_red",
                "info": "bright_blue",
                "muted": "grey50"
            },
            styles={
                "header": "bold bright_cyan",
                "subheader": "bold bright_blue",
                "highlight": "bold bright_yellow",
                "accent": "bright_magenta"
            }
        )
        
        # Light theme
        self.themes["light"] = ThemeConfig(
            name="Light",
            colors={
                "primary": "blue",
                "secondary": "navy_blue",
                "success": "dark_green",
                "warning": "dark_orange",
                "error": "dark_red",
                "info": "blue",
                "muted": "grey70"
            },
            styles={
                "header": "bold blue",
                "subheader": "bold navy_blue",
                "highlight": "bold dark_orange",
                "accent": "purple"
            }
        )
        
        # High contrast theme
        self.themes["high_contrast"] = ThemeConfig(
            name="High Contrast",
            colors={
                "primary": "white",
                "secondary": "bright_white",
                "success": "bright_green",
                "warning": "bright_yellow",
                "error": "bright_red",
                "info": "bright_cyan",
                "muted": "grey70"
            },
            styles={
                "header": "bold white",
                "subheader": "bold bright_white",
                "highlight": "bold bright_yellow on black",
                "accent": "bright_magenta"
            }
        )
    
    def set_theme(self, theme_name: str) -> bool:
        """Set current theme"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            return True
        return False
    
    def get_current_theme(self) -> ThemeConfig:
        """Get current theme configuration"""
        return self.themes[self.current_theme]
    
    def list_themes(self) -> List[str]:
        """List available themes"""
        return list(self.themes.keys())
    
    def add_custom_theme(self, name: str, colors: Dict[str, str], styles: Dict[str, str]):
        """Add custom theme"""
        self.themes[name] = ThemeConfig(name, colors, styles)


@dataclass
class ConfigProfile:
    """Configuration profile for different use cases"""
    name: str
    description: str
    settings: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    
    def apply_settings(self) -> Dict[str, Any]:
        """Apply profile settings"""
        return self.settings.copy()


class ProfileManager:
    """Configuration profile management"""
    
    def __init__(self) -> None:
        self.profiles: Dict[str, ConfigProfile] = {}
        self.current_profile = "default"
        self._setup_default_profiles()
    
    def _setup_default_profiles(self) -> None:
        """Setup default configuration profiles"""
        # Development profile
        self.profiles["development"] = ConfigProfile(
            name="Development",
            description="Settings optimized for development and testing",
            settings={
                "refresh_rate": 0.5,
                "log_level": "DEBUG",
                "show_debug_info": True,
                "auto_refresh": True,
                "page_size": 10,
                "theme": "dark",
                "enable_animations": True,
                "verbose_output": True
            }
        )
        
        # Production profile
        self.profiles["production"] = ConfigProfile(
            name="Production",
            description="Settings optimized for production monitoring",
            settings={
                "refresh_rate": 2.0,
                "log_level": "INFO",
                "show_debug_info": False,
                "auto_refresh": True,
                "page_size": 20,
                "theme": "default",
                "enable_animations": False,
                "verbose_output": False
            }
        )
        
        # Research profile
        self.profiles["research"] = ConfigProfile(
            name="Research",
            description="Settings optimized for research and analysis",
            settings={
                "refresh_rate": 1.0,
                "log_level": "INFO",
                "show_debug_info": True,
                "auto_refresh": False,
                "page_size": 50,
                "theme": "light",
                "enable_animations": False,
                "verbose_output": True,
                "export_format": "json",
                "include_metadata": True
            }
        )
        
        # Demo profile
        self.profiles["demo"] = ConfigProfile(
            name="Demo",
            description="Settings optimized for demonstrations",
            settings={
                "refresh_rate": 1.0,
                "log_level": "INFO",
                "show_debug_info": False,
                "auto_refresh": True,
                "page_size": 15,
                "theme": "high_contrast",
                "enable_animations": True,
                "verbose_output": False,
                "show_tips": True
            }
        )
    
    def create_profile(self, name: str, description: str, settings: Dict[str, Any]):
        """Create new configuration profile"""
        self.profiles[name] = ConfigProfile(name, description, settings)
    
    def set_profile(self, profile_name: str) -> bool:
        """Set current profile"""
        if profile_name in self.profiles:
            self.current_profile = profile_name
            return True
        return False
    
    def get_current_profile(self) -> ConfigProfile:
        """Get current profile"""
        return self.profiles[self.current_profile]
    
    def list_profiles(self) -> List[ConfigProfile]:
        """List all profiles"""
        return list(self.profiles.values())
    
    def delete_profile(self, profile_name: str) -> bool:
        """Delete a profile"""
        if profile_name in self.profiles and profile_name != "default":
            del self.profiles[profile_name]
            if self.current_profile == profile_name:
                self.current_profile = "default"
            return True
        return False


@dataclass
class MacroStep:
    """Single step in a macro"""
    command: str
    args: List[str]
    delay: float = 0.0
    description: str = ""


@dataclass
class Macro:
    """Command macro for automation"""
    name: str
    description: str
    steps: List[MacroStep]
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    
    def execute_step(self, step_index: int) -> Optional[MacroStep]:
        """Get step for execution"""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None
    
    def add_step(self, command: str, args: List[str], delay: float = 0.0, description: str = ""):
        """Add step to macro"""
        step = MacroStep(command, args, delay, description)
        self.steps.append(step)


class MacroSystem:
    """Macro recording and playback system"""
    
    def __init__(self) -> None:
        self.macros: Dict[str, Macro] = {}
        self.recording: bool = False
        self.current_recording: Optional[Macro] = None
        self.recorded_commands: List[MacroStep] = []
    
    def start_recording(self, name: str, description: str = "") -> bool:
        """Start recording a macro"""
        if not self.recording:
            self.recording = True
            self.current_recording = Macro(name, description, [])
            self.recorded_commands = []
            return True
        return False
    
    def stop_recording(self) -> Optional[Macro]:
        """Stop recording and save macro"""
        if self.recording and self.current_recording:
            self.current_recording.steps = self.recorded_commands.copy()
            self.macros[self.current_recording.name] = self.current_recording
            
            macro = self.current_recording
            self.recording = False
            self.current_recording = None
            self.recorded_commands = []
            return macro
        return None
    
    def record_command(self, command: str, args: List[str], description: str = "") -> None:
        """Record a command during macro recording"""
        if self.recording:
            step = MacroStep(command, args, 0.0, description)
            self.recorded_commands.append(step)
    
    def get_macro(self, name: str) -> Optional[Macro]:
        """Get macro by name"""
        return self.macros.get(name)
    
    def list_macros(self) -> List[Macro]:
        """List all macros"""
        return list(self.macros.values())
    
    def delete_macro(self, name: str) -> bool:
        """Delete a macro"""
        if name in self.macros:
            del self.macros[name]
            return True
        return False
    
    def execute_macro(self, name: str) -> Optional[List[MacroStep]]:
        """Get macro steps for execution"""
        if name in self.macros:
            macro = self.macros[name]
            macro.usage_count += 1
            return macro.steps
        return None


class WorkspaceManager:
    """Workspace management for organizing projects"""
    
    def __init__(self) -> None:
        self.workspaces: Dict[str, Dict[str, Any]] = {}
        self.current_workspace = "default"
        self._setup_default_workspace()
    
    def _setup_default_workspace(self) -> None:
        """Setup default workspace"""
        self.workspaces["default"] = {
            "name": "Default",
            "description": "Default workspace",
            "created_at": datetime.now().isoformat(),
            "agents": [],
            "sessions": [],
            "bookmarks": [],
            "settings": {},
            "last_accessed": datetime.now().isoformat()
        }
    
    def create_workspace(self, name: str, description: str = "") -> bool:
        """Create new workspace"""
        if name not in self.workspaces:
            self.workspaces[name] = {
                "name": name,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "agents": [],
                "sessions": [],
                "bookmarks": [],
                "settings": {},
                "last_accessed": datetime.now().isoformat()
            }
            return True
        return False
    
    def switch_workspace(self, name: str) -> bool:
        """Switch to workspace"""
        if name in self.workspaces:
            self.current_workspace = name
            self.workspaces[name]["last_accessed"] = datetime.now().isoformat()
            return True
        return False
    
    def get_current_workspace(self) -> Dict[str, Any]:
        """Get current workspace"""
        return self.workspaces[self.current_workspace]
    
    def list_workspaces(self) -> List[Dict[str, Any]]:
        """List all workspaces"""
        return list(self.workspaces.values())
    
    def delete_workspace(self, name: str) -> bool:
        """Delete workspace"""
        if name in self.workspaces and name != "default":
            del self.workspaces[name]
            if self.current_workspace == name:
                self.current_workspace = "default"
            return True
        return False
    
    def add_agent_to_workspace(self, agent_id: str):
        """Add agent to current workspace"""
        workspace = self.get_current_workspace()
        if agent_id not in workspace["agents"]:
            workspace["agents"].append(agent_id)
    
    def add_bookmark_to_workspace(self, bookmark: Dict[str, Any]):
        """Add bookmark to current workspace"""
        workspace = self.get_current_workspace()
        workspace["bookmarks"].append(bookmark)


class AccessibilityFeatures:
    """Accessibility features for screen readers and keyboard navigation"""
    
    def __init__(self) -> None:
        self.screen_reader_mode = False
        self.high_contrast_mode = False
        self.keyboard_only_mode = False
        self.announce_changes = True
        self.verbose_descriptions = False
    
    def enable_screen_reader_mode(self):
        """Enable screen reader compatibility"""
        self.screen_reader_mode = True
        self.verbose_descriptions = True
        self.announce_changes = True
    
    def enable_high_contrast_mode(self):
        """Enable high contrast mode"""
        self.high_contrast_mode = True
    
    def enable_keyboard_only_mode(self):
        """Enable keyboard-only navigation"""
        self.keyboard_only_mode = True
    
    def format_for_screen_reader(self, content: str) -> str:
        """Format content for screen readers"""
        if not self.screen_reader_mode:
            return content
        
        # Remove visual formatting
        content = re.sub(r'\[.*?\]', '', content)  # Remove Rich markup
        
        # Add descriptive text
        content = content.replace('█', 'filled block')
        content = content.replace('░', 'empty block')
        content = content.replace('●', 'data point')
        content = content.replace('▲', 'up arrow')
        content = content.replace('▼', 'down arrow')
        
        return content
    
    def announce_change(self, message: str) -> str:
        """Announce changes for screen readers"""
        if self.announce_changes and self.screen_reader_mode:
            return f"[Announcement] {message}"
        return message


# Utility functions for productivity features

def save_user_data(data: Dict[str, Any], filename: str):
    """Save user data to file"""
    config_dir = Path.home() / ".escai"
    config_dir.mkdir(exist_ok=True)
    
    filepath = config_dir / filename
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception:
        return False


def load_user_data(filename: str) -> Optional[Dict[str, Any]]:
    """Load user data from file"""
    config_dir = Path.home() / ".escai"
    filepath = config_dir / filename
    
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            logging.warning(f"Could not load user data from {filepath}.", exc_info=True)
            pass
    
    return None


def get_user_config_dir() -> Path:
    """Get user configuration directory"""
    config_dir = Path.home() / ".escai"
    config_dir.mkdir(exist_ok=True)
    return config_dir