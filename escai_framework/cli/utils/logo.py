"""
ASCII art logo for ESCAI Framework
"""

from rich.console import Console
from rich.text import Text
from .console import get_console

def display_logo():
    """Display the colorful ESCAI ASCII art logo"""
    console = get_console()
    
    logo = """
    ███████╗███████╗ ██████╗ █████╗ ██╗
    ██╔════╝██╔════╝██╔════╝██╔══██╗██║
    █████╗  ███████╗██║     ███████║██║
    ██╔══╝  ╚════██║██║     ██╔══██║██║
    ███████╗███████║╚██████╗██║  ██║██║
    ╚══════╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝
    """
    
    # Create colorful logo
    logo_text = Text()
    lines = logo.strip().split('\n')
    
    colors = ["bright_blue", "cyan", "bright_cyan", "blue", "bright_blue", "cyan"]
    
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        logo_text.append(line + "\n", style=color)
    
    # Add subtitle
    subtitle = Text("Epistemic State and Causal Analysis Intelligence", style="bold magenta")
    tagline = Text("Monitor • Analyze • Predict • Explain", style="dim white")
    
    console.print()
    console.print(logo_text, justify="center")
    console.print(subtitle, justify="center")
    console.print(tagline, justify="center")
    console.print()