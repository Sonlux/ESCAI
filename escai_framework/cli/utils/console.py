"""
Console utilities for rich formatting
"""

from rich.console import Console
from rich.theme import Theme

# Color schemes for different environments and preferences
COLOR_SCHEMES = {
    "default": Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "success": "green bold",
        "highlight": "magenta",
        "accent": "blue bold",
        "muted": "dim white",
        "primary": "cyan",
        "secondary": "blue",
        "chart_bar": "green",
        "chart_line": "blue",
        "progress": "green"
    }),
    
    "dark": Theme({
        "info": "bright_cyan",
        "warning": "bright_yellow",
        "error": "bright_red bold",
        "success": "bright_green bold",
        "highlight": "bright_magenta",
        "accent": "bright_blue bold",
        "muted": "white",
        "primary": "bright_cyan",
        "secondary": "bright_blue",
        "chart_bar": "bright_green",
        "chart_line": "bright_blue",
        "progress": "bright_green"
    }),
    
    "light": Theme({
        "info": "blue",
        "warning": "orange3",
        "error": "red3 bold",
        "success": "green bold",
        "highlight": "purple",
        "accent": "blue bold",
        "muted": "grey50",
        "primary": "blue",
        "secondary": "cyan",
        "chart_bar": "green",
        "chart_line": "blue",
        "progress": "green"
    }),
    
    "high_contrast": Theme({
        "info": "bright_white on blue",
        "warning": "black on bright_yellow",
        "error": "bright_white on red bold",
        "success": "black on bright_green bold",
        "highlight": "bright_white on magenta",
        "accent": "bright_white on blue bold",
        "muted": "bright_black",
        "primary": "bright_white",
        "secondary": "bright_yellow",
        "chart_bar": "bright_green",
        "chart_line": "bright_cyan",
        "progress": "bright_green"
    }),
    
    "monochrome": Theme({
        "info": "white",
        "warning": "bright_white",
        "error": "bright_white bold",
        "success": "white bold",
        "highlight": "bright_white",
        "accent": "white bold",
        "muted": "dim white",
        "primary": "white",
        "secondary": "bright_white",
        "chart_bar": "white",
        "chart_line": "bright_white",
        "progress": "white"
    })
}

_console = None
_current_theme = "default"

def get_console(theme: str | None = None) -> Console:
    """Get the global console instance with optional theme"""
    global _console, _current_theme
    
    if theme and theme != _current_theme:
        _current_theme = theme
        _console = None  # Force recreation with new theme
    
    if _console is None:
        selected_theme = COLOR_SCHEMES.get(_current_theme, COLOR_SCHEMES["default"])
        _console = Console(theme=selected_theme)
    
    return _console

def set_color_scheme(scheme: str) -> bool:
    """Set the global color scheme"""
    global _console, _current_theme
    
    if scheme in COLOR_SCHEMES:
        _current_theme = scheme
        _console = None  # Force recreation
        return True
    return False

def get_available_schemes() -> list:
    """Get list of available color schemes"""
    return list(COLOR_SCHEMES.keys())

def create_themed_console(scheme: str) -> Console:
    """Create a console with specific theme without affecting global instance"""
    if scheme in COLOR_SCHEMES:
        return Console(theme=COLOR_SCHEMES[scheme])
    return Console(theme=COLOR_SCHEMES["default"])

# Export the default theme as escai_theme for backward compatibility
escai_theme = COLOR_SCHEMES["default"]