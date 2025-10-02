"""
CLI Validation Configuration System

Provides configurable validation rules and security policies for CLI commands.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from escai_framework.security.input_validator import ValidationLevel

logger = logging.getLogger(__name__)


class ValidationPolicy(Enum):
    """Validation policy levels"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    SECURITY_CRITICAL = "security_critical"


@dataclass
class ParameterValidationRule:
    """Configuration for parameter validation"""
    name: str
    type: str
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    choices: Optional[List[str]] = None
    allow_empty: bool = False
    sanitize: bool = True
    custom_validator: Optional[str] = None
    error_message: Optional[str] = None
    help_text: Optional[str] = None


@dataclass
class CommandValidationConfig:
    """Validation configuration for a specific command"""
    command_name: str
    description: str
    parameters: List[ParameterValidationRule]
    security_level: ValidationLevel = ValidationLevel.STANDARD
    allow_unknown_params: bool = False
    rate_limit: Optional[int] = None
    audit_log: bool = True


@dataclass
class ValidationConfiguration:
    """Complete validation configuration"""
    policy: ValidationPolicy
    default_security_level: ValidationLevel
    commands: Dict[str, CommandValidationConfig]
    global_rules: Dict[str, Any]
    security_settings: Dict[str, Any]


class ValidationConfigManager:
    """Manages validation configuration loading and application"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config: Optional[ValidationConfiguration] = None
        self._load_configuration()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path"""
        # Look for config in multiple locations
        possible_paths = [
            Path.cwd() / '.escai' / 'validation_config.yaml',
            Path.home() / '.escai' / 'validation_config.yaml',
            Path(__file__).parent / 'default_validation_config.yaml'
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Return default location for creation
        return possible_paths[0]
    
    def _load_configuration(self):
        """Load validation configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                self.config = self._parse_configuration(config_data)
                logger.info(f"Loaded validation configuration from {self.config_path}")
            else:
                self.config = self._create_default_configuration()
                logger.info("Using default validation configuration")
                
        except Exception as e:
            logger.error(f"Failed to load validation configuration: {e}")
            self.config = self._create_default_configuration()
    
    def _parse_configuration(self, config_data: Dict[str, Any]) -> ValidationConfiguration:
        """Parse configuration data into ValidationConfiguration object"""
        # Parse policy
        policy = ValidationPolicy(config_data.get('policy', 'production'))
        
        # Parse default security level
        default_security_level = ValidationLevel(config_data.get('default_security_level', 2))
        
        # Parse commands
        commands = {}
        for cmd_name, cmd_config in config_data.get('commands', {}).items():
            parameters = []
            for param_config in cmd_config.get('parameters', []):
                param_rule = ParameterValidationRule(**param_config)
                parameters.append(param_rule)
            
            command_config = CommandValidationConfig(
                command_name=cmd_name,
                description=cmd_config.get('description', ''),
                parameters=parameters,
                security_level=ValidationLevel(cmd_config.get('security_level', 2)),
                allow_unknown_params=cmd_config.get('allow_unknown_params', False),
                rate_limit=cmd_config.get('rate_limit'),
                audit_log=cmd_config.get('audit_log', True)
            )
            commands[cmd_name] = command_config
        
        # Parse global rules and security settings
        global_rules = config_data.get('global_rules', {})
        security_settings = config_data.get('security_settings', {})
        
        return ValidationConfiguration(
            policy=policy,
            default_security_level=default_security_level,
            commands=commands,
            global_rules=global_rules,
            security_settings=security_settings
        )
    
    def _create_default_configuration(self) -> ValidationConfiguration:
        """Create default validation configuration"""
        # Default parameter rules for common CLI commands
        monitor_params = [
            ParameterValidationRule(
                name='agent_id',
                type='string',
                required=True,
                pattern=r'^[a-zA-Z0-9_-]{1,50}$',
                error_message='Agent ID must be alphanumeric with underscores/hyphens, max 50 chars',
                help_text='Unique identifier for the agent to monitor'
            ),
            ParameterValidationRule(
                name='framework',
                type='string',
                required=True,
                choices=['langchain', 'autogen', 'crewai', 'openai'],
                error_message='Framework must be one of: langchain, autogen, crewai, openai',
                help_text='Agent framework to monitor'
            ),
            ParameterValidationRule(
                name='interval',
                type='integer',
                required=False,
                min_value=1,
                max_value=3600,
                error_message='Interval must be between 1 and 3600 seconds',
                help_text='Monitoring interval in seconds'
            ),
            ParameterValidationRule(
                name='output_format',
                type='string',
                required=False,
                choices=['json', 'csv', 'table', 'yaml'],
                error_message='Output format must be: json, csv, table, or yaml',
                help_text='Format for monitoring output'
            )
        ]
        
        analyze_params = [
            ParameterValidationRule(
                name='session_id',
                type='string',
                required=True,
                pattern=r'^[a-zA-Z0-9_-]{8,64}$',
                error_message='Session ID must be 8-64 alphanumeric characters with underscores/hyphens',
                help_text='Session identifier for analysis'
            ),
            ParameterValidationRule(
                name='confidence_threshold',
                type='float',
                required=False,
                min_value=0.0,
                max_value=1.0,
                error_message='Confidence threshold must be between 0.0 and 1.0',
                help_text='Minimum confidence level for analysis results'
            ),
            ParameterValidationRule(
                name='max_results',
                type='integer',
                required=False,
                min_value=1,
                max_value=10000,
                error_message='Max results must be between 1 and 10000',
                help_text='Maximum number of analysis results to return'
            )
        ]
        
        config_params = [
            ParameterValidationRule(
                name='config_file',
                type='file_path',
                required=False,
                error_message='Config file must be a valid, readable file path',
                help_text='Path to configuration file'
            ),
            ParameterValidationRule(
                name='profile_name',
                type='string',
                required=False,
                pattern=r'^[a-zA-Z0-9_-]{1,32}$',
                error_message='Profile name must be 1-32 alphanumeric characters with underscores/hyphens',
                help_text='Name of the configuration profile'
            )
        ]
        
        export_params = [
            ParameterValidationRule(
                name='output_dir',
                type='directory_path',
                required=True,
                error_message='Output directory must be a valid, writable directory path',
                help_text='Directory for exported files'
            ),
            ParameterValidationRule(
                name='format',
                type='string',
                required=False,
                choices=['json', 'csv', 'yaml', 'xml'],
                error_message='Export format must be: json, csv, yaml, or xml',
                help_text='Format for exported data'
            )
        ]
        
        # Create command configurations
        commands = {
            'monitor': CommandValidationConfig(
                command_name='monitor',
                description='Monitor agent execution in real-time',
                parameters=monitor_params,
                security_level=ValidationLevel.STANDARD,
                audit_log=True
            ),
            'analyze': CommandValidationConfig(
                command_name='analyze',
                description='Analyze agent behavior patterns',
                parameters=analyze_params,
                security_level=ValidationLevel.STANDARD,
                audit_log=True
            ),
            'config': CommandValidationConfig(
                command_name='config',
                description='Manage CLI configuration',
                parameters=config_params,
                security_level=ValidationLevel.STRICT,
                audit_log=True
            ),
            'export': CommandValidationConfig(
                command_name='export',
                description='Export analysis results',
                parameters=export_params,
                security_level=ValidationLevel.STANDARD,
                audit_log=True
            )
        }
        
        # Global rules
        global_rules = {
            'max_input_length': 10000,
            'require_sanitization': True,
            'log_validation_failures': True,
            'rate_limit_enabled': True,
            'default_rate_limit': 100  # requests per minute
        }
        
        # Security settings
        security_settings = {
            'block_path_traversal': True,
            'block_command_injection': True,
            'block_sql_injection': True,
            'block_xss': True,
            'require_https_urls': False,  # Allow HTTP in development
            'validate_file_extensions': True,
            'max_file_size': 10485760,  # 10MB
            'allowed_file_extensions': ['.yaml', '.yml', '.json', '.csv', '.txt', '.log']
        }
        
        return ValidationConfiguration(
            policy=ValidationPolicy.PRODUCTION,
            default_security_level=ValidationLevel.STANDARD,
            commands=commands,
            global_rules=global_rules,
            security_settings=security_settings
        )
    
    def get_command_config(self, command_name: str) -> Optional[CommandValidationConfig]:
        """Get validation configuration for a specific command"""
        if not self.config:
            return None
        
        return self.config.commands.get(command_name)
    
    def get_parameter_config(self, command_name: str, parameter_name: str) -> Optional[ParameterValidationRule]:
        """Get validation configuration for a specific parameter"""
        command_config = self.get_command_config(command_name)
        if not command_config:
            return None
        
        for param in command_config.parameters:
            if param.name == parameter_name:
                return param
        
        return None
    
    def get_security_level(self, command_name: str) -> ValidationLevel:
        """Get security level for a command"""
        command_config = self.get_command_config(command_name)
        if command_config:
            return command_config.security_level
        
        return self.config.default_security_level if self.config else ValidationLevel.STANDARD
    
    def is_parameter_required(self, command_name: str, parameter_name: str) -> bool:
        """Check if a parameter is required for a command"""
        param_config = self.get_parameter_config(command_name, parameter_name)
        return param_config.required if param_config else False
    
    def get_parameter_choices(self, command_name: str, parameter_name: str) -> Optional[List[str]]:
        """Get valid choices for a parameter"""
        param_config = self.get_parameter_config(command_name, parameter_name)
        return param_config.choices if param_config else None
    
    def get_parameter_pattern(self, command_name: str, parameter_name: str) -> Optional[str]:
        """Get validation pattern for a parameter"""
        param_config = self.get_parameter_config(command_name, parameter_name)
        return param_config.pattern if param_config else None
    
    def get_parameter_range(self, command_name: str, parameter_name: str) -> tuple[Optional[Union[int, float]], Optional[Union[int, float]]]:
        """Get value range for a parameter"""
        param_config = self.get_parameter_config(command_name, parameter_name)
        if param_config:
            return param_config.min_value, param_config.max_value
        return None, None
    
    def get_global_rule(self, rule_name: str) -> Any:
        """Get a global validation rule"""
        if not self.config:
            return None
        
        return self.config.global_rules.get(rule_name)
    
    def get_security_setting(self, setting_name: str) -> Any:
        """Get a security setting"""
        if not self.config:
            return None
        
        return self.config.security_settings.get(setting_name)
    
    def save_configuration(self, config_path: Optional[Path] = None):
        """Save current configuration to file"""
        if not self.config:
            return
        
        save_path = config_path or self.config_path
        
        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert configuration to dictionary
            config_dict = {
                'policy': self.config.policy.value,
                'default_security_level': self.config.default_security_level.value,
                'commands': {},
                'global_rules': self.config.global_rules,
                'security_settings': self.config.security_settings
            }
            
            # Convert command configurations
            for cmd_name, cmd_config in self.config.commands.items():
                commands_dict: Dict[str, Any] = config_dict['commands']  # type: ignore[assignment]
                commands_dict[cmd_name] = {
                    'description': cmd_config.description,
                    'security_level': cmd_config.security_level.value,
                    'allow_unknown_params': cmd_config.allow_unknown_params,
                    'rate_limit': cmd_config.rate_limit,
                    'audit_log': cmd_config.audit_log,
                    'parameters': [asdict(param) for param in cmd_config.parameters]
                }
            
            # Save as YAML
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved validation configuration to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save validation configuration: {e}")
            raise
    
    def add_command_config(self, command_config: CommandValidationConfig):
        """Add or update command configuration"""
        if not self.config:
            self.config = self._create_default_configuration()
        
        self.config.commands[command_config.command_name] = command_config
    
    def add_parameter_rule(self, command_name: str, parameter_rule: ParameterValidationRule):
        """Add parameter rule to a command"""
        command_config = self.get_command_config(command_name)
        if not command_config:
            # Create new command config
            command_config = CommandValidationConfig(
                command_name=command_name,
                description=f'Configuration for {command_name} command',
                parameters=[parameter_rule]
            )
            self.add_command_config(command_config)
        else:
            # Add to existing command
            # Remove existing parameter with same name
            command_config.parameters = [p for p in command_config.parameters if p.name != parameter_rule.name]
            command_config.parameters.append(parameter_rule)
    
    def update_security_level(self, command_name: str, security_level: ValidationLevel):
        """Update security level for a command"""
        command_config = self.get_command_config(command_name)
        if command_config:
            command_config.security_level = security_level
    
    def update_global_rule(self, rule_name: str, rule_value: Any):
        """Update a global validation rule"""
        if not self.config:
            self.config = self._create_default_configuration()
        
        self.config.global_rules[rule_name] = rule_value
    
    def update_security_setting(self, setting_name: str, setting_value: Any):
        """Update a security setting"""
        if not self.config:
            self.config = self._create_default_configuration()
        
        self.config.security_settings[setting_name] = setting_value
    
    def validate_configuration(self) -> List[str]:
        """Validate the current configuration and return any issues"""
        issues = []
        
        if not self.config:
            issues.append("No configuration loaded")
            return issues
        
        # Validate command configurations
        for cmd_name, cmd_config in self.config.commands.items():
            # Check for duplicate parameter names
            param_names = [p.name for p in cmd_config.parameters]
            if len(param_names) != len(set(param_names)):
                issues.append(f"Command '{cmd_name}' has duplicate parameter names")
            
            # Validate parameter configurations
            for param in cmd_config.parameters:
                if param.type not in ['string', 'integer', 'float', 'boolean', 'file_path', 'directory_path', 'url', 'email', 'json', 'yaml', 'regex']:
                    issues.append(f"Command '{cmd_name}', parameter '{param.name}': invalid type '{param.type}'")
                
                if param.choices and param.type != 'string':
                    issues.append(f"Command '{cmd_name}', parameter '{param.name}': choices only valid for string type")
                
                if param.pattern and param.type != 'string':
                    issues.append(f"Command '{cmd_name}', parameter '{param.name}': pattern only valid for string type")
                
                if param.min_value is not None and param.type not in ['integer', 'float']:
                    issues.append(f"Command '{cmd_name}', parameter '{param.name}': min_value only valid for numeric types")
        
        return issues
    
    def export_schema(self) -> Dict[str, Any]:
        """Export configuration schema for documentation"""
        if not self.config:
            return {}
        
        schema: Dict[str, Any] = {
            'policy': self.config.policy.value,
            'default_security_level': self.config.default_security_level.name,
            'commands': {}
        }
        
        for cmd_name, cmd_config in self.config.commands.items():
            schema_commands: Dict[str, Any] = schema['commands']  # type: ignore[assignment]
            schema_commands[cmd_name] = {
                'description': cmd_config.description,
                'security_level': cmd_config.security_level.name,
                'parameters': {}
            }
            
            for param in cmd_config.parameters:
                param_schema = {
                    'type': param.type,
                    'required': param.required,
                    'description': param.help_text or f'{param.name} parameter'
                }
                
                if param.choices:
                    param_schema['choices'] = param.choices
                if param.pattern:
                    param_schema['pattern'] = param.pattern
                if param.min_value is not None:
                    param_schema['min_value'] = param.min_value
                if param.max_value is not None:
                    param_schema['max_value'] = param.max_value
                
                cmd_params: Dict[str, Any] = schema['commands'][cmd_name]['parameters']  # type: ignore[index]
                cmd_params[param.name] = param_schema
        
        return schema


# Global configuration manager instance
_config_manager: Optional[ValidationConfigManager] = None


def get_validation_config() -> ValidationConfigManager:
    """Get global validation configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ValidationConfigManager()
    return _config_manager


def reload_validation_config(config_path: Optional[Path] = None):
    """Reload validation configuration"""
    global _config_manager
    _config_manager = ValidationConfigManager(config_path)


def create_default_config_file(config_path: Path):
    """Create a default configuration file"""
    manager = ValidationConfigManager()
    manager.save_configuration(config_path)
    return manager