"""
CLI Validation Integration System

Integrates validation with CLI commands and provides comprehensive validation
decorators and utilities for CLI command implementations.
"""

import functools
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path

import click

from escai_framework.cli.utils.validators import (
    CLIInputValidator,
    CLIValidationError,
    CLIValidationResult
)
from escai_framework.cli.utils.validation_config import (
    get_validation_config,
    ValidationConfigManager,
    ParameterValidationRule
)
from escai_framework.security.input_validator import ValidationLevel

logger = logging.getLogger(__name__)


class ValidationIntegration:
    """Main validation integration class for CLI commands"""
    
    def __init__(self, config_manager: Optional[ValidationConfigManager] = None):
        self.config_manager = config_manager or get_validation_config()
        self.validator = CLIInputValidator()
        self._validation_cache: Dict[str, Dict[str, CLIValidationResult]] = {}
    
    def validate_command_parameters(self, command_name: str, parameters: Dict[str, Any]) -> Dict[str, CLIValidationResult]:
        """Validate all parameters for a command using configuration"""
        results = {}
        
        # Get command configuration
        command_config = self.config_manager.get_command_config(command_name)
        if not command_config:
            logger.warning(f"No validation configuration found for command: {command_name}")
            # Fall back to basic validation
            return self._validate_without_config(command_name, parameters)
        
        # Update validator security level
        self.validator = CLIInputValidator(command_config.security_level)
        
        # Validate each configured parameter
        for param_rule in command_config.parameters:
            param_name = param_rule.name
            param_value = parameters.get(param_name)
            
            # Check if parameter is required
            if param_rule.required and (param_value is None or param_value == ''):
                results[param_name] = CLIValidationResult(
                    is_valid=False,
                    errors=[param_rule.error_message or f"Parameter '{param_name}' is required"],
                    warnings=[],
                    sanitized_value=None,
                    suggestions=[f"Provide a value for {param_name}"],
                    help_text=param_rule.help_text
                )
                continue
            
            # Skip validation if parameter is optional and not provided
            if not param_rule.required and (param_value is None or param_value == ''):
                continue
            
            # Perform validation based on parameter configuration
            result = self._validate_parameter_with_config(param_rule, param_value)
            results[param_name] = result
        
        # Check for unknown parameters if not allowed
        if not command_config.allow_unknown_params:
            configured_params = {param.name for param in command_config.parameters}
            unknown_params = set(parameters.keys()) - configured_params
            
            for unknown_param in unknown_params:
                results[unknown_param] = CLIValidationResult(
                    is_valid=False,
                    errors=[f"Unknown parameter: {unknown_param}"],
                    warnings=[],
                    sanitized_value=parameters[unknown_param],
                    suggestions=[f"Remove {unknown_param} or check command documentation"],
                    help_text=f"Use --help to see valid parameters for {command_name}"
                )
        
        return results
    
    def _validate_parameter_with_config(self, param_rule: ParameterValidationRule, value: Any) -> CLIValidationResult:
        """Validate a parameter using its configuration rule"""
        # Start with basic type validation
        result = self.validator.validate_command_parameter(param_rule.name, value, param_rule.type)
        
        if not result.is_valid:
            # Use configured error message if available
            if param_rule.error_message:
                result.errors = [param_rule.error_message]
            
            # Add configured help text
            if param_rule.help_text:
                result.help_text = param_rule.help_text
            
            return result
        
        # Apply additional validations from configuration
        errors = []
        warnings = []
        sanitized_value = result.sanitized_value
        
        # Length validation for strings
        if param_rule.type == 'string' and isinstance(sanitized_value, str):
            if param_rule.min_length is not None and len(sanitized_value) < param_rule.min_length:
                errors.append(f"{param_rule.name} must be at least {param_rule.min_length} characters")
            
            if param_rule.max_length is not None and len(sanitized_value) > param_rule.max_length:
                errors.append(f"{param_rule.name} must be at most {param_rule.max_length} characters")
        
        # Value range validation for numbers
        if param_rule.type in ['integer', 'float']:
            try:
                num_value = float(sanitized_value)
                
                if param_rule.min_value is not None and num_value < param_rule.min_value:
                    errors.append(f"{param_rule.name} must be at least {param_rule.min_value}")
                
                if param_rule.max_value is not None and num_value > param_rule.max_value:
                    errors.append(f"{param_rule.name} must be at most {param_rule.max_value}")
            
            except (ValueError, TypeError):
                errors.append(f"{param_rule.name} must be a valid number")
        
        # Pattern validation
        if param_rule.pattern and param_rule.type == 'string':
            import re
            if not re.match(param_rule.pattern, str(sanitized_value)):
                errors.append(param_rule.error_message or f"{param_rule.name} format is invalid")
        
        # Choice validation
        if param_rule.choices:
            if str(sanitized_value).lower() not in [choice.lower() for choice in param_rule.choices]:
                errors.append(param_rule.error_message or f"{param_rule.name} must be one of: {', '.join(param_rule.choices)}")
        
        # Empty value validation
        if not param_rule.allow_empty and isinstance(sanitized_value, str) and sanitized_value.strip() == '':
            errors.append(f"{param_rule.name} cannot be empty")
        
        # Create final result
        final_result = CLIValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings + result.warnings,
            sanitized_value=sanitized_value,
            suggestions=result.suggestions,
            help_text=param_rule.help_text or result.help_text
        )
        
        return final_result
    
    def _validate_without_config(self, command_name: str, parameters: Dict[str, Any]) -> Dict[str, CLIValidationResult]:
        """Fallback validation without configuration"""
        results = {}
        
        # Basic validation for common parameter types
        for param_name, param_value in parameters.items():
            # Guess parameter type based on name patterns
            param_type = self._guess_parameter_type(param_name)
            result = self.validator.validate_command_parameter(param_name, param_value, param_type)
            results[param_name] = result
        
        return results
    
    def _guess_parameter_type(self, param_name: str) -> str:
        """Guess parameter type based on name patterns"""
        name_lower = param_name.lower()
        
        # File/directory patterns
        if any(keyword in name_lower for keyword in ['file', 'path', 'config']):
            if any(keyword in name_lower for keyword in ['dir', 'directory', 'folder']):
                return 'directory_path'
            return 'file_path'
        
        # URL patterns
        if any(keyword in name_lower for keyword in ['url', 'endpoint', 'uri']):
            return 'url'
        
        # Email patterns
        if 'email' in name_lower:
            return 'email'
        
        # Numeric patterns
        if any(keyword in name_lower for keyword in ['port', 'timeout', 'interval', 'limit', 'count', 'max', 'min']):
            return 'integer'
        
        if any(keyword in name_lower for keyword in ['confidence', 'threshold', 'rate', 'ratio']):
            return 'float'
        
        # Boolean patterns
        if any(keyword in name_lower for keyword in ['enable', 'disable', 'flag', 'is_', 'has_', 'should_']):
            return 'boolean'
        
        # JSON/YAML patterns
        if any(keyword in name_lower for keyword in ['json', 'config_data']):
            return 'json'
        
        if 'yaml' in name_lower:
            return 'yaml'
        
        # Default to string
        return 'string'
    
    def create_click_validator(self, command_name: str) -> Callable:
        """Create a Click parameter validator for a command"""
        def validate_parameter(ctx, param, value):
            """Click parameter validation callback"""
            if value is None:
                return value
            
            param_name = param.name
            result = self.validate_command_parameters(command_name, {param_name: value})
            
            if param_name in result and not result[param_name].is_valid:
                error_msg = '; '.join(result[param_name].errors)
                raise click.BadParameter(error_msg)
            
            return result[param_name].sanitized_value if param_name in result else value
        
        return validate_parameter
    
    def create_command_decorator(self, command_name: str, validate_all: bool = True):
        """Create a decorator for comprehensive command validation"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    if validate_all:
                        # Validate all parameters at once
                        validation_results = self.validate_command_parameters(command_name, kwargs)
                        
                        # Check for validation errors
                        errors = []
                        for param_name, result in validation_results.items():
                            if not result.is_valid:
                                errors.extend([f"{param_name}: {error}" for error in result.errors])
                            else:
                                # Replace with sanitized value
                                kwargs[param_name] = result.sanitized_value
                        
                        if errors:
                            raise CLIValidationError(
                                f"Validation failed for command '{command_name}':\n" + '\n'.join(errors),
                                [result.suggestions for result in validation_results.values() if result.suggestions]
                            )
                    
                    # Log successful validation
                    if self.config_manager.get_command_config(command_name):
                        command_config = self.config_manager.get_command_config(command_name)
                        if command_config.audit_log:
                            logger.info(f"Command '{command_name}' validation passed for parameters: {list(kwargs.keys())}")
                    
                    return func(*args, **kwargs)
                
                except CLIValidationError:
                    raise  # Re-raise validation errors
                except Exception as e:
                    logger.error(f"Error in command '{command_name}': {e}")
                    raise
            
            return wrapper
        return decorator
    
    def validate_and_sanitize_input(self, command_name: str, param_name: str, value: Any) -> Any:
        """Validate and sanitize a single input value"""
        result = self.validate_command_parameters(command_name, {param_name: value})
        
        if param_name in result:
            if not result[param_name].is_valid:
                error_msg = '; '.join(result[param_name].errors)
                raise CLIValidationError(f"Invalid {param_name}: {error_msg}")
            
            return result[param_name].sanitized_value
        
        return value
    
    def get_parameter_help(self, command_name: str, param_name: str) -> Optional[str]:
        """Get help text for a parameter"""
        param_config = self.config_manager.get_parameter_config(command_name, param_name)
        return param_config.help_text if param_config else None
    
    def get_parameter_choices(self, command_name: str, param_name: str) -> Optional[List[str]]:
        """Get valid choices for a parameter"""
        return self.config_manager.get_parameter_choices(command_name, param_name)
    
    def is_parameter_required(self, command_name: str, param_name: str) -> bool:
        """Check if a parameter is required"""
        return self.config_manager.is_parameter_required(command_name, param_name)
    
    def generate_parameter_documentation(self, command_name: str) -> Dict[str, Dict[str, Any]]:
        """Generate documentation for command parameters"""
        command_config = self.config_manager.get_command_config(command_name)
        if not command_config:
            return {}
        
        docs = {}
        for param in command_config.parameters:
            param_doc = {
                'type': param.type,
                'required': param.required,
                'description': param.help_text or f'{param.name} parameter',
                'error_message': param.error_message
            }
            
            if param.choices:
                param_doc['choices'] = param.choices
            if param.pattern:
                param_doc['pattern'] = param.pattern
            if param.min_value is not None:
                param_doc['min_value'] = param.min_value
            if param.max_value is not None:
                param_doc['max_value'] = param.max_value
            if param.min_length is not None:
                param_doc['min_length'] = param.min_length
            if param.max_length is not None:
                param_doc['max_length'] = param.max_length
            
            docs[param.name] = param_doc
        
        return docs


# Global validation integration instance
_validation_integration: Optional[ValidationIntegration] = None


def get_validation_integration() -> ValidationIntegration:
    """Get global validation integration instance"""
    global _validation_integration
    if _validation_integration is None:
        _validation_integration = ValidationIntegration()
    return _validation_integration


def validate_command(command_name: str, validate_all: bool = True):
    """Decorator for command validation"""
    integration = get_validation_integration()
    return integration.create_command_decorator(command_name, validate_all)


def validate_parameter(command_name: str, param_name: str, value: Any) -> Any:
    """Validate and sanitize a single parameter"""
    integration = get_validation_integration()
    return integration.validate_and_sanitize_input(command_name, param_name, value)


def get_parameter_help_text(command_name: str, param_name: str) -> Optional[str]:
    """Get help text for a parameter"""
    integration = get_validation_integration()
    return integration.get_parameter_help(command_name, param_name)


def create_click_option_with_validation(command_name: str, param_name: str, **click_kwargs):
    """Create a Click option with integrated validation"""
    integration = get_validation_integration()
    
    # Get parameter configuration
    param_config = integration.config_manager.get_parameter_config(command_name, param_name)
    
    if param_config:
        # Update click kwargs with configuration
        if param_config.help_text and 'help' not in click_kwargs:
            click_kwargs['help'] = param_config.help_text
        
        if param_config.required and 'required' not in click_kwargs:
            click_kwargs['required'] = param_config.required
        
        if param_config.choices and 'type' not in click_kwargs:
            click_kwargs['type'] = click.Choice(param_config.choices)
        
        # Add validation callback
        if 'callback' not in click_kwargs:
            click_kwargs['callback'] = integration.create_click_validator(command_name)
    
    return click.option(f'--{param_name.replace("_", "-")}', **click_kwargs)


class ValidationErrorHandler:
    """Handles validation errors and provides user-friendly feedback"""
    
    @staticmethod
    def handle_validation_error(error: CLIValidationError, command_name: str = None):
        """Handle validation error and provide user feedback"""
        click.echo(click.style("Validation Error:", fg='red', bold=True))
        click.echo(f"  {error.message}")
        
        if error.suggestions:
            click.echo(click.style("\nSuggestions:", fg='yellow', bold=True))
            for suggestion in error.suggestions:
                if isinstance(suggestion, list):
                    for sub_suggestion in suggestion:
                        click.echo(f"  • {sub_suggestion}")
                else:
                    click.echo(f"  • {suggestion}")
        
        if command_name:
            click.echo(f"\nUse 'escai {command_name} --help' for more information.")
    
    @staticmethod
    def handle_validation_warnings(warnings: List[str]):
        """Handle validation warnings"""
        if warnings:
            click.echo(click.style("Warnings:", fg='yellow', bold=True))
            for warning in warnings:
                click.echo(f"  ⚠ {warning}")


def validation_error_handler(func):
    """Decorator to handle validation errors gracefully"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CLIValidationError as e:
            ValidationErrorHandler.handle_validation_error(e, func.__name__)
            raise click.Abort()
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            click.echo(click.style(f"Error: {str(e)}", fg='red'))
            raise click.Abort()
    
    return wrapper