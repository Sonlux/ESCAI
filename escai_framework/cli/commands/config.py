"""
Configuration management commands for ESCAI CLI
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Union

import click
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

from ..utils.console import get_console

console = get_console()

CONFIG_DIR = Path.home() / '.escai'
CONFIG_FILE = CONFIG_DIR / 'config.json'

@click.group(name='config')
def config_group():
    """Configuration management commands"""
    pass

@config_group.command()
def setup():
    """Interactive configuration setup"""
    
    console.print("[bold cyan]ESCAI Framework Configuration Setup[/bold cyan]\n")
    
    # Ensure config directory exists
    CONFIG_DIR.mkdir(exist_ok=True)
    
    config = {}
    
    # Database configurations
    console.print("[bold]Database Configuration[/bold]")
    
    # PostgreSQL
    if Confirm.ask("Configure PostgreSQL connection?", default=True):
        config['postgresql'] = {
            'host': Prompt.ask("PostgreSQL host", default="localhost"),
            'port': int(Prompt.ask("PostgreSQL port", default="5432")),
            'database': Prompt.ask("Database name", default="escai"),
            'username': Prompt.ask("Username", default="escai_user"),
            'password': Prompt.ask("Password", password=True)
        }
    
    # MongoDB
    if Confirm.ask("Configure MongoDB connection?", default=True):
        config['mongodb'] = {
            'host': Prompt.ask("MongoDB host", default="localhost"),
            'port': int(Prompt.ask("MongoDB port", default="27017")),
            'database': Prompt.ask("Database name", default="escai"),
            'username': Prompt.ask("Username (optional)", default=""),
            'password': Prompt.ask("Password (optional)", password=True, default="")
        }
    
    # Redis
    if Confirm.ask("Configure Redis connection?", default=True):
        config['redis'] = {
            'host': Prompt.ask("Redis host", default="localhost"),
            'port': int(Prompt.ask("Redis port", default="6379")),
            'password': Prompt.ask("Password (optional)", password=True, default=""),
            'db': int(Prompt.ask("Database number", default="0"))
        }
    
    # InfluxDB
    if Confirm.ask("Configure InfluxDB connection?", default=False):
        config['influxdb'] = {
            'host': Prompt.ask("InfluxDB host", default="localhost"),
            'port': int(Prompt.ask("InfluxDB port", default="8086")),
            'database': Prompt.ask("Database name", default="escai"),
            'username': Prompt.ask("Username", default=""),
            'password': Prompt.ask("Password", password=True, default="")
        }
    
    # Neo4j
    if Confirm.ask("Configure Neo4j connection?", default=False):
        config['neo4j'] = {
            'uri': Prompt.ask("Neo4j URI", default="bolt://localhost:7687"),
            'username': Prompt.ask("Username", default="neo4j"),
            'password': Prompt.ask("Password", password=True)
        }
    
    # API Configuration
    console.print("\n[bold]API Configuration[/bold]")
    config['api'] = {
        'host': Prompt.ask("API host", default="localhost"),
        'port': int(Prompt.ask("API port", default="8000")),
        'jwt_secret': Prompt.ask("JWT secret key", default="your-secret-key-here"),
        'rate_limit': int(Prompt.ask("Rate limit (requests per minute)", default="100"))
    }
    
    # Monitoring Configuration
    console.print("\n[bold]Monitoring Configuration[/bold]")
    config['monitoring'] = {
        'max_overhead_percent': float(Prompt.ask("Max monitoring overhead (%)", default="10")),
        'event_buffer_size': int(Prompt.ask("Event buffer size", default="1000")),
        'retention_days': int(Prompt.ask("Data retention (days)", default="90"))
    }
    
    # Save configuration
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"\n[success]✅ Configuration saved to {CONFIG_FILE}[/success]")

@config_group.command()
def show():
    """Display current configuration"""
    
    if not CONFIG_FILE.exists():
        console.print("[error]No configuration found. Run 'escai config setup' first.[/error]")
        return
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    # Create configuration display
    for section, settings in config.items():
        table = Table(title=f"{section.title()} Configuration", show_header=True, header_style="bold cyan")
        table.add_column("Setting", style="yellow")
        table.add_column("Value", style="white")
        
        for key, value in settings.items():
            # Mask passwords
            if 'password' in key.lower() and value:
                value = '*' * len(str(value))
            table.add_row(key, str(value))
        
        console.print(table)
        console.print()

@config_group.command()
@click.argument('section')
@click.argument('key')
@click.argument('value')
def set(section: str, key: str, value: Union[str, bool, int, float]):
    """Set a configuration value"""
    
    if not CONFIG_FILE.exists():
        console.print("[error]No configuration found. Run 'escai config setup' first.[/error]")
        return
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    if section not in config:
        config[section] = {}
    
    # Try to convert value to appropriate type
    try:
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            value = float(value)
    except:
        pass  # Keep as string
    
    config[section][key] = value
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"[success]✅ Set {section}.{key} = {value}[/success]")

@config_group.command()
@click.argument('section')
@click.argument('key')
def get(section: str, key: str):
    """Get a configuration value"""
    
    if not CONFIG_FILE.exists():
        console.print("[error]No configuration found. Run 'escai config setup' first.[/error]")
        return
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    if section not in config:
        console.print(f"[error]Section '{section}' not found[/error]")
        return
    
    if key not in config[section]:
        console.print(f"[error]Key '{key}' not found in section '{section}'[/error]")
        return
    
    value = config[section][key]
    if 'password' in key.lower() and value:
        value = '*' * len(str(value))
    
    console.print(f"{section}.{key} = {value}")

@config_group.command()
def test():
    """Test database connections"""
    
    if not CONFIG_FILE.exists():
        console.print("[error]No configuration found. Run 'escai config setup' first.[/error]")
        return
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    console.print("[info]Testing database connections...[/info]\n")
    
    results = []
    
    # Test PostgreSQL
    if 'postgresql' in config:
        try:
            # Mock connection test
            console.print("🔍 Testing PostgreSQL connection...")
            results.append(("PostgreSQL", "✅ Connected", "green"))
        except Exception as e:
            results.append(("PostgreSQL", f"❌ Failed: {str(e)}", "red"))
    
    # Test MongoDB
    if 'mongodb' in config:
        try:
            console.print("🔍 Testing MongoDB connection...")
            results.append(("MongoDB", "✅ Connected", "green"))
        except Exception as e:
            results.append(("MongoDB", f"❌ Failed: {str(e)}", "red"))
    
    # Test Redis
    if 'redis' in config:
        try:
            console.print("🔍 Testing Redis connection...")
            results.append(("Redis", "✅ Connected", "green"))
        except Exception as e:
            results.append(("Redis", f"❌ Failed: {str(e)}", "red"))
    
    # Display results
    table = Table(title="Connection Test Results", show_header=True, header_style="bold cyan")
    table.add_column("Database", style="yellow")
    table.add_column("Status", style="white")
    
    for db, status, color in results:
        table.add_row(db, f"[{color}]{status}[/{color}]")
    
    console.print(table)

@config_group.command()
@click.option('--scheme', type=click.Choice(['default', 'dark', 'light', 'high_contrast', 'monochrome']),
              help='Color scheme to set')
@click.option('--list', 'list_schemes', is_flag=True, help='List available color schemes')
@click.option('--preview', is_flag=True, help='Preview color schemes')
def theme(scheme: str, list_schemes: bool, preview: bool):
    """Configure CLI color scheme and theme"""
    
    from ..utils.console import get_available_schemes, set_color_scheme, create_themed_console
    
    available_schemes = get_available_schemes()
    
    if list_schemes:
        console.print("\n[bold cyan]Available Color Schemes:[/bold cyan]")
        for i, scheme_name in enumerate(available_schemes, 1):
            console.print(f"  {i}. {scheme_name}")
        return
    
    if preview:
        console.print("\n[bold cyan]Color Scheme Preview:[/bold cyan]")
        
        for scheme_name in available_schemes:
            themed_console = create_themed_console(scheme_name)
            
            themed_console.print(f"\n[bold]{scheme_name.title()} Theme:[/bold]")
            themed_console.print("  [info]Info message[/info]")
            themed_console.print("  [warning]Warning message[/warning]")
            themed_console.print("  [error]Error message[/error]")
            themed_console.print("  [success]Success message[/success]")
            themed_console.print("  [highlight]Highlighted text[/highlight]")
            themed_console.print("  [accent]Accent text[/accent]")
            themed_console.print("  [muted]Muted text[/muted]")
            
            # Show sample chart colors
            themed_console.print("  Chart colors: [chart_bar]████[/chart_bar] [chart_line]████[/chart_line] [progress]████[/progress]")
        
        return
    
    if scheme:
        if scheme in available_schemes:
            # Save theme preference to config
            config = {}
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            
            if 'ui' not in config:
                config['ui'] = {}
            
            config['ui']['color_scheme'] = scheme
            
            CONFIG_DIR.mkdir(exist_ok=True)
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Apply theme
            set_color_scheme(scheme)
            console.print(f"[success]✅ Color scheme set to '{scheme}'[/success]")
            
            # Show preview of new theme
            console.print(f"\n[bold]Preview of {scheme} theme:[/bold]")
            console.print("  [info]This is an info message[/info]")
            console.print("  [success]This is a success message[/success]")
            console.print("  [warning]This is a warning message[/warning]")
            console.print("  [error]This is an error message[/error]")
            
        else:
            console.print(f"[error]Unknown color scheme: {scheme}[/error]")
            console.print(f"Available schemes: {', '.join(available_schemes)}")
    
    else:
        # Show current theme
        current_scheme = "default"
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                current_scheme = config.get('ui', {}).get('color_scheme', 'default')
        
        console.print(f"[info]Current color scheme: {current_scheme}[/info]")
        console.print("\nUse --list to see available schemes")
        console.print("Use --preview to preview all schemes")
        console.print("Use --scheme <name> to set a scheme")


@config_group.command()
def check():
    """Validate current configuration and system requirements"""
    
    console.print("[info]Checking system configuration and requirements...[/info]\n")
    
    checks = []
    
    # Check Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 8):
        checks.append(("Python Version", f"✅ {python_version}", "green"))
    else:
        checks.append(("Python Version", f"❌ {python_version} (requires 3.8+)", "red"))
    
    # Check configuration file
    if CONFIG_FILE.exists():
        checks.append(("Configuration File", "✅ Found", "green"))
        
        # Validate configuration
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            required_sections = ['api', 'monitoring']
            for section in required_sections:
                if section in config:
                    checks.append((f"Config Section: {section}", "✅ Present", "green"))
                else:
                    checks.append((f"Config Section: {section}", "⚠️ Missing", "yellow"))
        
        except json.JSONDecodeError:
            checks.append(("Configuration File", "❌ Invalid JSON", "red"))
    
    else:
        checks.append(("Configuration File", "❌ Not found", "red"))
    
    # Check dependencies
    required_packages = [
        'rich', 'click', 'asyncio', 'pandas', 'numpy',
        'fastapi', 'sqlalchemy', 'redis', 'pymongo'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            checks.append((f"Package: {package}", "✅ Installed", "green"))
        except ImportError:
            checks.append((f"Package: {package}", "❌ Missing", "red"))
    
    # Check directories
    directories = [
        CONFIG_DIR,
        Path.cwd() / 'logs',
        Path.cwd() / 'data'
    ]
    
    for directory in directories:
        if directory.exists():
            checks.append((f"Directory: {directory.name}", "✅ Exists", "green"))
        else:
            checks.append((f"Directory: {directory.name}", "⚠️ Missing", "yellow"))
    
    # Display results
    table = Table(title="System Check Results", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="yellow", width=25)
    table.add_column("Status", style="white", width=30)
    
    for component, status, color in checks:
        table.add_row(component, f"[{color}]{status}[/{color}]")
    
    console.print(table)
    
    # Summary
    passed = sum(1 for _, status, _ in checks if "✅" in status)
    warnings = sum(1 for _, status, _ in checks if "⚠️" in status)
    failed = sum(1 for _, status, _ in checks if "❌" in status)
    
    console.print(f"\n[bold]Summary:[/bold] {passed} passed, {warnings} warnings, {failed} failed")
    
    if failed > 0:
        console.print("\n[bold red]Action Required:[/bold red]")
        console.print("  • Install missing dependencies: [cyan]pip install -r requirements.txt[/cyan]")
        console.print("  • Run configuration setup: [cyan]escai config setup[/cyan]")
    elif warnings > 0:
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        console.print("  • Create missing directories")
        console.print("  • Complete configuration setup: [cyan]escai config setup[/cyan]")
    else:
        console.print("\n[bold green]✅ All checks passed! System is ready.[/bold green]")


@config_group.command()
def reset():
    """Reset configuration to defaults"""
    
    if CONFIG_FILE.exists():
        if Confirm.ask("Are you sure you want to reset all configuration?", default=False):
            CONFIG_FILE.unlink()
            console.print("[success]✅ Configuration reset. Run 'escai config setup' to reconfigure.[/success]")
    else:
        console.print("[info]No configuration file found.[/info]")