"""
Console utilities for rich formatting
"""

from rich.console import Console
from rich.theme import Theme

# Custom theme for ESCAI CLI
escai_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold",
    "highlight": "magenta",
    "accent": "blue bold",
    "muted": "dim white"
})

_console = None

def get_console() -> Console:
    """Get the global console instance"""
    global _console
    if _console is None:
        _console = Console(theme=escai_theme)
    return _console