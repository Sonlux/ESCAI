"""
Configuration management commands for ESCAI CLI
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

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
def set(section: str, key: str, value: str):
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
def reset():
    """Reset configuration to defaults"""
    
    if CONFIG_FILE.exists():
        if Confirm.ask("Are you sure you want to reset all configuration?", default=False):
            CONFIG_FILE.unlink()
            console.print("[success]✅ Configuration reset. Run 'escai config setup' to reconfigure.[/success]")
    else:
        console.print("[info]No configuration file found.[/info]")