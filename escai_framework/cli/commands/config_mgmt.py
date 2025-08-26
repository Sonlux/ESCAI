"""
Configuration management CLI commands for the ESCAI framework.

This module provides CLI commands for managing configuration including
validation, encryption, versioning, and template generation.
"""

import json
import yaml
from typing import Optional, Dict, Any
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm

from ...config import (
    ConfigManager, ConfigValidator, ConfigEncryption, 
    ConfigVersioning, ConfigTemplates, Environment
)


console = Console()


@click.group(name='config')
def config_group():
    """Configuration management commands."""
    pass


@config_group.command()
@click.option('--environment', '-e', 
              type=click.Choice(['development', 'testing', 'staging', 'production']),
              default='development',
              help='Target environment')
@click.option('--output', '-o', 
              type=click.Path(),
              help='Output file path')
@click.option('--format', '-f',
              type=click.Choice(['yaml', 'json']),
              default='yaml',
              help='Output format')
def init(environment: str, output: Optional[str], format: str):
    """Initialize configuration for specified environment."""
    try:
        templates = ConfigTemplates()
        config_data = templates.generate_config_template(Environment(environment))
        
        if output:
            output_path = Path(output)
        else:
            output_path = Path(f"config/config.{environment}.{format}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(output_path, 'w') as f:
            if format == 'yaml':
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_data, f, indent=2)
        
        console.print(f"✅ Configuration initialized for {environment} environment")
        console.print(f"📁 Saved to: {output_path}")
        
        # Show next steps
        console.print("\n📋 Next steps:")
        console.print("1. Review and update the generated configuration")
        console.print("2. Set secure passwords and secrets")
        console.print("3. Validate configuration: escai config validate")
        
    except Exception as e:
        console.print(f"❌ Failed to initialize configuration: {e}")
        raise click.Abort()


@config_group.command()
@click.option('--config-file', '-c',
              type=click.Path(exists=True),
              help='Configuration file to validate')
@click.option('--environment', '-e',
              type=click.Choice(['development', 'testing', 'staging', 'production']),
              help='Environment to validate for')
@click.option('--detailed', '-d',
              is_flag=True,
              help='Show detailed validation report')
def validate(config_file: Optional[str], environment: Optional[str], detailed: bool):
    """Validate configuration file."""
    try:
        validator = ConfigValidator()
        
        if config_file:
            is_valid, errors = validator.validate_config_file(config_file)
            config_path = config_file
        else:
            # Auto-detect configuration file
            config_manager = ConfigManager(environment=environment)
            config_path = config_manager._get_environment_config_path()
            
            if not config_path.exists():
                console.print(f"❌ Configuration file not found: {config_path}")
                raise click.Abort()
            
            is_valid, errors = validator.validate_config_file(str(config_path))
        
        # Show validation results
        if is_valid:
            console.print("✅ Configuration is valid")
        else:
            console.print("❌ Configuration validation failed")
            
            error_table = Table(title="Validation Errors")
            error_table.add_column("Error", style="red")
            
            for error in errors:
                error_table.add_row(error)
            
            console.print(error_table)
        
        # Show detailed report if requested
        if detailed:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            report = validator.generate_validation_report(config_data)
            
            # Show warnings
            if report['warnings']:
                warning_table = Table(title="Warnings", title_style="yellow")
                warning_table.add_column("Warning", style="yellow")
                
                for warning in report['warnings']:
                    warning_table.add_row(warning)
                
                console.print(warning_table)
            
            # Show recommendations
            if report['recommendations']:
                rec_table = Table(title="Recommendations", title_style="blue")
                rec_table.add_column("Recommendation", style="blue")
                
                for rec in report['recommendations']:
                    rec_table.add_row(rec)
                
                console.print(rec_table)
        
    except Exception as e:
        console.print(f"❌ Validation failed: {e}")
        raise click.Abort()


@config_group.command()
@click.option('--config-file', '-c',
              type=click.Path(exists=True),
              required=True,
              help='Configuration file to encrypt')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output file for encrypted configuration')
@click.option('--key-file', '-k',
              type=click.Path(),
              help='Encryption key file')
@click.option('--generate-key', '-g',
              is_flag=True,
              help='Generate new encryption key')
def encrypt(config_file: str, output: Optional[str], key_file: Optional[str], generate_key: bool):
    """Encrypt sensitive configuration values."""
    try:
        # Initialize encryption
        if generate_key:
            master_key = ConfigEncryption.generate_master_key()
            console.print("🔑 Generated new encryption key")
            
            if key_file:
                with open(key_file, 'w') as f:
                    f.write(master_key)
                console.print(f"💾 Saved encryption key to: {key_file}")
            else:
                console.print(f"🔑 Master key: {master_key}")
                console.print("⚠️  Store this key securely!")
            
            encryption = ConfigEncryption(master_key=master_key)
        else:
            encryption = ConfigEncryption(key_file=key_file)
        
        # Load configuration
        with open(config_file, 'r') as f:
            if Path(config_file).suffix.lower() in ['.yml', '.yaml']:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        # Encrypt configuration
        encrypted_config = encryption.encrypt_config(config_data)
        
        # Save encrypted configuration
        if output:
            output_path = Path(output)
        else:
            config_path = Path(config_file)
            output_path = config_path.parent / f"{config_path.stem}.encrypted{config_path.suffix}"
        
        with open(output_path, 'w') as f:
            if output_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(encrypted_config, f, default_flow_style=False, indent=2)
            else:
                json.dump(encrypted_config, f, indent=2)
        
        console.print("🔒 Configuration encrypted successfully")
        console.print(f"📁 Saved to: {output_path}")
        
    except Exception as e:
        console.print(f"❌ Encryption failed: {e}")
        raise click.Abort()


@config_group.command()
@click.option('--config-file', '-c',
              type=click.Path(exists=True),
              required=True,
              help='Encrypted configuration file to decrypt')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output file for decrypted configuration')
@click.option('--key-file', '-k',
              type=click.Path(exists=True),
              help='Encryption key file')
def decrypt(config_file: str, output: Optional[str], key_file: Optional[str]):
    """Decrypt encrypted configuration values."""
    try:
        # Initialize encryption
        encryption = ConfigEncryption(key_file=key_file)
        
        # Load encrypted configuration
        with open(config_file, 'r') as f:
            if Path(config_file).suffix.lower() in ['.yml', '.yaml']:
                encrypted_config = yaml.safe_load(f)
            else:
                encrypted_config = json.load(f)
        
        # Decrypt configuration
        decrypted_config = encryption.decrypt_config(encrypted_config)
        
        # Save decrypted configuration
        if output:
            output_path = Path(output)
        else:
            config_path = Path(config_file)
            output_path = config_path.parent / f"{config_path.stem}.decrypted{config_path.suffix}"
        
        with open(output_path, 'w') as f:
            if output_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(decrypted_config, f, default_flow_style=False, indent=2)
            else:
                json.dump(decrypted_config, f, indent=2)
        
        console.print("🔓 Configuration decrypted successfully")
        console.print(f"📁 Saved to: {output_path}")
        
    except Exception as e:
        console.print(f"❌ Decryption failed: {e}")
        raise click.Abort()


@config_group.command()
@click.option('--config-dir', '-d',
              type=click.Path(),
              default='config',
              help='Configuration directory')
@click.option('--limit', '-l',
              type=int,
              default=10,
              help='Number of versions to show')
def history(config_dir: str, limit: int):
    """Show configuration version history."""
    try:
        versioning = ConfigVersioning(f"{config_dir}/versions")
        versions = versioning.get_version_history(limit)
        
        if not versions:
            console.print("📝 No configuration versions found")
            return
        
        # Create version history table
        table = Table(title="Configuration Version History")
        table.add_column("Version ID", style="cyan")
        table.add_column("Timestamp", style="green")
        table.add_column("Description", style="white")
        table.add_column("Tags", style="yellow")
        table.add_column("Size", style="blue")
        
        for version in versions:
            tags_str = ", ".join(version.get('tags', []))
            size_str = f"{version.get('size', 0):,} bytes"
            
            table.add_row(
                version['id'][:12],
                version['timestamp'][:19],
                version.get('description', 'No description'),
                tags_str or 'None',
                size_str
            )
        
        console.print(table)
        
        # Show statistics
        stats = versioning.get_statistics()
        console.print(f"\n📊 Total versions: {stats['total_versions']}")
        
    except Exception as e:
        console.print(f"❌ Failed to show history: {e}")
        raise click.Abort()


@config_group.command()
@click.option('--config-dir', '-d',
              type=click.Path(),
              default='config',
              help='Configuration directory')
@click.argument('version_id')
def rollback(config_dir: str, version_id: str):
    """Rollback configuration to a previous version."""
    try:
        config_manager = ConfigManager(config_dir=config_dir, enable_versioning=True)
        
        # Confirm rollback
        if not Confirm.ask(f"Are you sure you want to rollback to version {version_id}?"):
            console.print("❌ Rollback cancelled")
            return
        
        # Perform rollback
        config_manager.rollback_config(version_id)
        
        console.print(f"✅ Configuration rolled back to version: {version_id}")
        console.print("⚠️  Remember to restart services to apply changes")
        
    except Exception as e:
        console.print(f"❌ Rollback failed: {e}")
        raise click.Abort()


@config_group.command()
@click.option('--config-dir', '-d',
              type=click.Path(),
              default='config',
              help='Configuration directory')
@click.argument('version1_id')
@click.argument('version2_id')
def diff(config_dir: str, version1_id: str, version2_id: str):
    """Compare two configuration versions."""
    try:
        versioning = ConfigVersioning(f"{config_dir}/versions")
        comparison = versioning.compare_versions(version1_id, version2_id)
        
        # Show version info
        console.print(Panel(
            f"Comparing versions:\n"
            f"Version 1: {version1_id} ({comparison['version1']['timestamp']})\n"
            f"Version 2: {version2_id} ({comparison['version2']['timestamp']})",
            title="Version Comparison"
        ))
        
        # Show differences
        if not comparison['differences']:
            console.print("✅ No differences found between versions")
            return
        
        diff_table = Table(title=f"Differences ({comparison['total_changes']} changes)")
        diff_table.add_column("Key", style="cyan")
        diff_table.add_column("Type", style="yellow")
        diff_table.add_column("Old Value", style="red")
        diff_table.add_column("New Value", style="green")
        
        for diff in comparison['differences']:
            old_val = str(diff['old_value']) if diff['old_value'] is not None else 'None'
            new_val = str(diff['new_value']) if diff['new_value'] is not None else 'None'
            
            # Truncate long values
            if len(old_val) > 50:
                old_val = old_val[:47] + "..."
            if len(new_val) > 50:
                new_val = new_val[:47] + "..."
            
            diff_table.add_row(
                diff['key'],
                diff['type'],
                old_val,
                new_val
            )
        
        console.print(diff_table)
        
    except Exception as e:
        console.print(f"❌ Comparison failed: {e}")
        raise click.Abort()


@config_group.command()
@click.option('--environment', '-e',
              type=click.Choice(['development', 'testing', 'staging', 'production']),
              required=True,
              help='Target environment')
@click.option('--output-dir', '-o',
              type=click.Path(),
              default='deploy',
              help='Output directory for deployment files')
@click.option('--format', '-f',
              type=click.Choice(['docker-compose', 'kubernetes']),
              default='docker-compose',
              help='Deployment format')
def generate_deploy(environment: str, output_dir: str, format: str):
    """Generate deployment configuration files."""
    try:
        templates = ConfigTemplates()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format == 'docker-compose':
            # Generate Docker Compose file
            compose_content = templates.generate_docker_compose_template(Environment(environment))
            
            compose_file = output_path / f"docker-compose.{environment}.yml"
            with open(compose_file, 'w') as f:
                f.write(compose_content)
            
            console.print(f"✅ Generated Docker Compose configuration")
            console.print(f"📁 Saved to: {compose_file}")
            
        elif format == 'kubernetes':
            # Generate Kubernetes manifests
            manifests = templates.generate_kubernetes_template(Environment(environment))
            
            for filename, content in manifests.items():
                manifest_file = output_path / filename
                with open(manifest_file, 'w') as f:
                    f.write(content)
                
                console.print(f"📁 Generated: {manifest_file}")
            
            console.print(f"✅ Generated Kubernetes manifests for {environment}")
        
        # Show next steps
        console.print("\n📋 Next steps:")
        if format == 'docker-compose':
            console.print(f"1. Review {compose_file}")
            console.print("2. Update secrets and passwords")
            console.print(f"3. Deploy: docker-compose -f {compose_file} up -d")
        else:
            console.print(f"1. Review manifests in {output_path}")
            console.print("2. Create secrets: kubectl create secret ...")
            console.print(f"3. Deploy: kubectl apply -f {output_path}")
        
    except Exception as e:
        console.print(f"❌ Failed to generate deployment files: {e}")
        raise click.Abort()


@config_group.command()
@click.option('--config-file', '-c',
              type=click.Path(exists=True),
              help='Configuration file to show')
@click.option('--key', '-k',
              help='Specific configuration key to show (dot notation)')
@click.option('--format', '-f',
              type=click.Choice(['yaml', 'json', 'table']),
              default='yaml',
              help='Output format')
def show(config_file: Optional[str], key: Optional[str], format: str):
    """Show current configuration."""
    try:
        if config_file:
            config_path = Path(config_file)
        else:
            # Auto-detect configuration file
            config_manager = ConfigManager()
            config_path = config_manager._get_environment_config_path()
        
        if not config_path.exists():
            console.print(f"❌ Configuration file not found: {config_path}")
            raise click.Abort()
        
        # Load configuration
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        # Get specific key if requested
        if key:
            try:
                value = config_data
                for k in key.split('.'):
                    value = value[k]
                config_data = {key: value}
            except KeyError:
                console.print(f"❌ Configuration key not found: {key}")
                raise click.Abort()
        
        # Display configuration
        if format == 'yaml':
            syntax = Syntax(yaml.dump(config_data, default_flow_style=False, indent=2), 
                          "yaml", theme="monokai", line_numbers=True)
            console.print(syntax)
        elif format == 'json':
            syntax = Syntax(json.dumps(config_data, indent=2), 
                          "json", theme="monokai", line_numbers=True)
            console.print(syntax)
        elif format == 'table':
            # Show as table for flat configuration
            table = Table(title="Configuration")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="white")
            
            def add_rows(data, prefix=""):
                for k, v in data.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        add_rows(v, full_key)
                    else:
                        table.add_row(full_key, str(v))
            
            add_rows(config_data)
            console.print(table)
        
    except Exception as e:
        console.print(f"❌ Failed to show configuration: {e}")
        raise click.Abort()


@config_group.command()
@click.option('--config-file', '-c',
              type=click.Path(),
              help='Configuration file to update')
@click.argument('key')
@click.argument('value')
def set(config_file: Optional[str], key: str, value: str):
    """Set configuration value."""
    try:
        config_manager = ConfigManager()
        
        if config_file:
            config_manager.load_config(config_file)
        else:
            config_manager.load_config()
        
        # Parse value (try to detect type)
        parsed_value = value
        if value.lower() in ['true', 'false']:
            parsed_value = value.lower() == 'true'
        elif value.isdigit():
            parsed_value = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            parsed_value = float(value)
        
        # Set configuration value
        config_manager.set_config_value(key, parsed_value)
        
        # Save configuration
        config_manager.save_config(config_file)
        
        console.print(f"✅ Set {key} = {parsed_value}")
        console.print("💾 Configuration saved")
        
    except Exception as e:
        console.print(f"❌ Failed to set configuration: {e}")
        raise click.Abort()


# Add the config group to the main CLI
def register_config_commands(cli_group):
    """Register configuration management commands with the main CLI."""
    cli_group.add_command(config_group)