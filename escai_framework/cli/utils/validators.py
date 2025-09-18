"""
CLI Input Validation and Sanitization System

Provides comprehensive input validation and sanitization specifically for CLI commands
with helpful error messages and security-focused validation.
"""

import re
import os
import json
import yaml
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import ipaddress

from escai_framework.security.input_validator import (
    InputValidator, 
    ValidationResult, 
    ValidationLevel,
    SecuritySanitizer
)

logger = logging.getLogger(__name__)


class CLIValidationError(Exception):
    """Custom exception for CLI validation errors"""
    def __init__(self, message: str, suggestions: List[str] = None):
        self.message = message
        self.suggestions = suggestions or []
        super().__init__(self.message)


@dataclass
class CLIValidationResult:
    """Extended validation result for CLI with suggestions"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Any
    suggestions: List[str]
    help_text: Optional[str] = None


class CLIInputValidator:
    """CLI-specific input validator with comprehensive validation rules"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.base_validator = InputValidator(validation_level)
        self.sanitizer = SecuritySanitizer(validation_level)
        
        # CLI-specific validation patterns
        self.patterns = {
            'agent_id': r'^[a-zA-Z0-9_-]{1,50}$',
            'session_id': r'^[a-zA-Z0-9_-]{8,64}$',
            'framework_name': r'^(langchain|autogen|crewai|openai)$',
            'output_format': r'^(json|csv|table|yaml)$',
            'log_level': r'^(debug|info|warning|error|critical)$',
            'time_range': r'^(\d+[smhd]|\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?|\d+)$',
            'config_key': r'^[a-zA-Z0-9._-]+$',
            'profile_name': r'^[a-zA-Z0-9_-]{1,32}$',
            'command_name': r'^[a-zA-Z0-9_-]+$'
        }
    
    def validate_command_parameter(self, param_name: str, value: Any, 
                                 param_type: str = 'string') -> CLIValidationResult:
        """Validate individual command parameter with CLI-specific rules"""
        errors = []
        warnings = []
        suggestions = []
        sanitized_value = value
        help_text = None
        
        try:
            # Handle None/empty values
            if value is None or (isinstance(value, str) and value.strip() == ''):
                return CLIValidationResult(
                    is_valid=False,
                    errors=[f"Parameter '{param_name}' cannot be empty"],
                    warnings=[],
                    sanitized_value=None,
                    suggestions=[f"Provide a valid value for {param_name}"],
                    help_text=f"Use --help to see valid options for {param_name}"
                )
            
            # Type-specific validation
            if param_type == 'string':
                result = self._validate_string_parameter(param_name, value)
            elif param_type == 'integer':
                result = self._validate_integer_parameter(param_name, value)
            elif param_type == 'float':
                result = self._validate_float_parameter(param_name, value)
            elif param_type == 'boolean':
                result = self._validate_boolean_parameter(param_name, value)
            elif param_type == 'file_path':
                result = self._validate_file_path_parameter(param_name, value)
            elif param_type == 'directory_path':
                result = self._validate_directory_path_parameter(param_name, value)
            elif param_type == 'url':
                result = self._validate_url_parameter(param_name, value)
            elif param_type == 'email':
                result = self._validate_email_parameter(param_name, value)
            elif param_type == 'json':
                result = self._validate_json_parameter(param_name, value)
            elif param_type == 'yaml':
                result = self._validate_yaml_parameter(param_name, value)
            elif param_type == 'regex':
                result = self._validate_regex_parameter(param_name, value)
            elif param_type == 'choice':
                result = self._validate_choice_parameter(param_name, value)
            else:
                result = self._validate_generic_parameter(param_name, value)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error for parameter {param_name}: {e}")
            return CLIValidationResult(
                is_valid=False,
                errors=[f"Validation failed for parameter '{param_name}': {str(e)}"],
                warnings=[],
                sanitized_value=value,
                suggestions=["Check parameter format and try again"],
                help_text="Use --help for parameter documentation"
            )
    
    def _validate_string_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate string parameters with CLI-specific rules"""
        errors = []
        warnings = []
        suggestions = []
        
        # Convert to string
        str_value = str(value)
        
        # Check original input against patterns first (before sanitization)
        original_validation_errors = []
        
        # Pattern-specific validation on original input
        if param_name in self.patterns:
            pattern = self.patterns[param_name]
            if not re.match(pattern, str_value):
                original_validation_errors.append(f"Parameter '{param_name}' has invalid format")
                suggestions.extend(self._get_pattern_suggestions(param_name))
        
        # Special validations for known parameters on original input
        if param_name == 'agent_id':
            if not re.match(self.patterns['agent_id'], str_value):
                original_validation_errors.append("Agent ID must contain only alphanumeric characters, underscores, and hyphens")
                suggestions.append("Example: 'my-agent-1' or 'agent_001'")
        
        elif param_name == 'framework_name':
            valid_frameworks = ['langchain', 'autogen', 'crewai', 'openai']
            if str_value.lower() not in valid_frameworks:
                original_validation_errors.append(f"Unknown framework: {str_value}")
                suggestions.append(f"Valid frameworks: {', '.join(valid_frameworks)}")
        
        elif param_name == 'output_format':
            valid_formats = ['json', 'csv', 'table', 'yaml']
            if str_value.lower() not in valid_formats:
                original_validation_errors.append(f"Unknown output format: {str_value}")
                suggestions.append(f"Valid formats: {', '.join(valid_formats)}")
        
        # If original validation failed, return error immediately
        if original_validation_errors:
            errors.extend(original_validation_errors)
        
        # Apply sanitization
        sanitized_value = self.sanitizer.comprehensive_sanitize(str_value)
        
        if sanitized_value != str_value:
            warnings.append(f"Input sanitized for security: removed potentially dangerous characters")
        
        # Length validation
        if len(sanitized_value) > 1000:
            errors.append(f"Parameter '{param_name}' is too long (max 1000 characters)")
            suggestions.append("Shorten the input or use a file reference")
        
        return CLIValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized_value,
            suggestions=suggestions
        )
    
    def _validate_integer_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate integer parameters"""
        errors = []
        warnings = []
        suggestions = []
        
        try:
            int_value = int(value)
            
            # Range validation based on parameter name
            if param_name in ['port', 'server_port']:
                if not (1 <= int_value <= 65535):
                    errors.append(f"Port number must be between 1 and 65535")
                    suggestions.append("Use a valid port number like 8080 or 3000")
            
            elif param_name in ['timeout', 'interval']:
                if int_value < 0:
                    errors.append(f"{param_name.title()} cannot be negative")
                    suggestions.append(f"Use a positive number for {param_name}")
                elif int_value > 3600:
                    warnings.append(f"Large {param_name} value may cause performance issues")
            
            elif param_name in ['limit', 'max_results']:
                if int_value < 1:
                    errors.append(f"{param_name.title()} must be at least 1")
                    suggestions.append("Use a positive number")
                elif int_value > 10000:
                    warnings.append("Large result sets may impact performance")
            
            return CLIValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=int_value,
                suggestions=suggestions
            )
            
        except (ValueError, TypeError):
            return CLIValidationResult(
                is_valid=False,
                errors=[f"Parameter '{param_name}' must be a valid integer"],
                warnings=[],
                sanitized_value=value,
                suggestions=["Provide a whole number like 42 or 100"]
            )
    
    def _validate_float_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate float parameters"""
        errors = []
        warnings = []
        suggestions = []
        
        try:
            float_value = float(value)
            
            # Range validation based on parameter name
            if param_name in ['confidence', 'threshold', 'probability']:
                if not (0.0 <= float_value <= 1.0):
                    errors.append(f"{param_name.title()} must be between 0.0 and 1.0")
                    suggestions.append("Use a decimal value like 0.5 or 0.95")
            
            elif param_name in ['rate', 'frequency']:
                if float_value < 0:
                    errors.append(f"{param_name.title()} cannot be negative")
                    suggestions.append(f"Use a positive number for {param_name}")
            
            return CLIValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=float_value,
                suggestions=suggestions
            )
            
        except (ValueError, TypeError):
            return CLIValidationResult(
                is_valid=False,
                errors=[f"Parameter '{param_name}' must be a valid number"],
                warnings=[],
                sanitized_value=value,
                suggestions=["Provide a decimal number like 3.14 or 0.5"]
            )
    
    def _validate_boolean_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate boolean parameters"""
        if isinstance(value, bool):
            return CLIValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                sanitized_value=value,
                suggestions=[]
            )
        
        # Convert string representations
        str_value = str(value).lower().strip()
        true_values = ['true', 'yes', 'y', '1', 'on', 'enable', 'enabled']
        false_values = ['false', 'no', 'n', '0', 'off', 'disable', 'disabled']
        
        if str_value in true_values:
            return CLIValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                sanitized_value=True,
                suggestions=[]
            )
        elif str_value in false_values:
            return CLIValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                sanitized_value=False,
                suggestions=[]
            )
        else:
            return CLIValidationResult(
                is_valid=False,
                errors=[f"Parameter '{param_name}' must be a boolean value"],
                warnings=[],
                sanitized_value=value,
                suggestions=["Use: true/false, yes/no, 1/0, or enable/disable"]
            )
    
    def _validate_file_path_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate file path parameters"""
        errors = []
        warnings = []
        suggestions = []
        
        str_value = str(value)
        
        # Sanitize path traversal attempts
        sanitized_path = self.sanitizer.sanitize_path_traversal(str_value)
        
        if sanitized_path != str_value:
            warnings.append("Path sanitized to prevent directory traversal")
        
        # On Windows, fix path separators that might be removed by sanitization
        if os.name == 'nt' and ':' in str_value and ':' not in sanitized_path:
            # Restore drive letter colon that was removed by sanitization
            if len(str_value) > 1 and str_value[1] == ':':
                sanitized_path = str_value[0] + ':' + sanitized_path[1:]
        
        try:
            path = Path(sanitized_path)
            
            # Check for absolute vs relative paths
            if path.is_absolute():
                # Validate absolute path security
                if not self._is_safe_absolute_path(path):
                    errors.append("Absolute path not allowed in restricted directories")
                    suggestions.append("Use relative paths or allowed directories")
            
            # Check if file exists (for input files)
            if param_name in ['input_file', 'config_file', 'data_file']:
                if not path.exists():
                    errors.append(f"File does not exist: {sanitized_path}")
                    suggestions.append("Check the file path and ensure the file exists")
                elif not path.is_file():
                    errors.append(f"Path is not a file: {sanitized_path}")
                    suggestions.append("Provide a path to a file, not a directory")
            
            # Check file extension for specific parameters
            if param_name == 'config_file':
                valid_extensions = ['.yaml', '.yml', '.json', '.toml']
                if path.suffix.lower() not in valid_extensions:
                    warnings.append(f"Unexpected config file extension: {path.suffix}")
                    suggestions.append(f"Consider using: {', '.join(valid_extensions)}")
            
            # Check permissions
            if path.exists():
                if param_name in ['input_file', 'config_file'] and not os.access(path, os.R_OK):
                    errors.append(f"No read permission for file: {sanitized_path}")
                    suggestions.append("Check file permissions")
                
                if param_name in ['output_file', 'log_file'] and not os.access(path.parent, os.W_OK):
                    errors.append(f"No write permission for directory: {path.parent}")
                    suggestions.append("Check directory permissions or choose different location")
            
            return CLIValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=str(sanitized_path),
                suggestions=suggestions
            )
            
        except Exception as e:
            return CLIValidationResult(
                is_valid=False,
                errors=[f"Invalid file path: {str(e)}"],
                warnings=[],
                sanitized_value=sanitized_path,
                suggestions=["Provide a valid file path"]
            )
    
    def _validate_directory_path_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate directory path parameters"""
        errors = []
        warnings = []
        suggestions = []
        
        str_value = str(value)
        
        # Sanitize path traversal attempts
        sanitized_path = self.sanitizer.sanitize_path_traversal(str_value)
        
        if sanitized_path != str_value:
            warnings.append("Path sanitized to prevent directory traversal")
        
        # On Windows, fix path separators that might be removed by sanitization
        if os.name == 'nt' and ':' in str_value and ':' not in sanitized_path:
            # Restore drive letter colon that was removed by sanitization
            if len(str_value) > 1 and str_value[1] == ':':
                sanitized_path = str_value[0] + ':' + sanitized_path[1:]
        
        try:
            path = Path(sanitized_path)
            
            # Check for absolute vs relative paths
            if path.is_absolute():
                if not self._is_safe_absolute_path(path):
                    errors.append("Absolute path not allowed in restricted directories")
                    suggestions.append("Use relative paths or allowed directories")
            
            # Check if directory exists
            if param_name in ['input_dir', 'data_dir']:
                if not path.exists():
                    errors.append(f"Directory does not exist: {sanitized_path}")
                    suggestions.append("Check the directory path and ensure it exists")
                elif not path.is_dir():
                    errors.append(f"Path is not a directory: {sanitized_path}")
                    suggestions.append("Provide a path to a directory, not a file")
            
            # Check permissions
            if path.exists():
                if param_name in ['input_dir', 'data_dir'] and not os.access(path, os.R_OK):
                    errors.append(f"No read permission for directory: {sanitized_path}")
                    suggestions.append("Check directory permissions")
                
                if param_name in ['output_dir', 'log_dir'] and not os.access(path, os.W_OK):
                    errors.append(f"No write permission for directory: {sanitized_path}")
                    suggestions.append("Check directory permissions or choose different location")
            
            return CLIValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=str(sanitized_path),
                suggestions=suggestions
            )
            
        except Exception as e:
            return CLIValidationResult(
                is_valid=False,
                errors=[f"Invalid directory path: {str(e)}"],
                warnings=[],
                sanitized_value=sanitized_path,
                suggestions=["Provide a valid directory path"]
            )
    
    def _validate_url_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate URL parameters"""
        errors = []
        warnings = []
        suggestions = []
        
        str_value = str(value).strip()
        
        # Basic URL format validation
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, str_value, re.IGNORECASE):
            errors.append(f"Invalid URL format for '{param_name}'")
            suggestions.append("URL must start with http:// or https://")
            return CLIValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                sanitized_value=str_value,
                suggestions=suggestions
            )
        
        try:
            parsed = urllib.parse.urlparse(str_value)
            
            # Validate scheme
            if parsed.scheme.lower() not in ['http', 'https']:
                errors.append("URL must use http or https protocol")
                suggestions.append("Use http:// or https:// prefix")
            
            # Validate hostname
            if not parsed.netloc:
                errors.append("URL must include a valid hostname")
                suggestions.append("Example: https://api.example.com")
            
            # Security checks
            if parsed.scheme == 'http' and param_name in ['api_url', 'webhook_url']:
                warnings.append("Using HTTP instead of HTTPS may be insecure")
                suggestions.append("Consider using HTTPS for better security")
            
            # Check for localhost/private IPs in production contexts
            if param_name in ['webhook_url', 'callback_url']:
                hostname = parsed.hostname
                if hostname in ['localhost', '127.0.0.1'] or hostname.startswith('192.168.'):
                    warnings.append("Using localhost or private IP may not be accessible externally")
            
            return CLIValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=str_value,
                suggestions=suggestions
            )
            
        except Exception as e:
            return CLIValidationResult(
                is_valid=False,
                errors=[f"Invalid URL: {str(e)}"],
                warnings=[],
                sanitized_value=str_value,
                suggestions=["Provide a valid URL like https://example.com"]
            )
    
    def _validate_email_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate email parameters"""
        str_value = str(value).strip()
        
        # Use base validator for email validation
        base_result = self.base_validator.validate_field('email', str_value)
        
        suggestions = []
        if not base_result.is_valid:
            suggestions.append("Example: user@example.com")
        
        return CLIValidationResult(
            is_valid=base_result.is_valid,
            errors=base_result.errors,
            warnings=base_result.warnings or [],
            sanitized_value=base_result.sanitized_data.get('email', str_value),
            suggestions=suggestions
        )
    
    def _validate_json_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate JSON parameters"""
        errors = []
        warnings = []
        suggestions = []
        
        str_value = str(value).strip()
        
        try:
            parsed_json = json.loads(str_value)
            
            # Security check for large JSON
            if len(str_value) > 10000:
                warnings.append("Large JSON payload may impact performance")
            
            return CLIValidationResult(
                is_valid=True,
                errors=[],
                warnings=warnings,
                sanitized_value=parsed_json,
                suggestions=[]
            )
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
            suggestions.extend([
                "Check JSON syntax (quotes, commas, brackets)",
                "Example: {\"key\": \"value\"}",
                "Use online JSON validator to check format"
            ])
            
            return CLIValidationResult(
                is_valid=False,
                errors=errors,
                warnings=[],
                sanitized_value=str_value,
                suggestions=suggestions
            )
    
    def _validate_yaml_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate YAML parameters"""
        errors = []
        warnings = []
        suggestions = []
        
        str_value = str(value).strip()
        
        try:
            parsed_yaml = yaml.safe_load(str_value)
            
            # Security check for large YAML
            if len(str_value) > 10000:
                warnings.append("Large YAML payload may impact performance")
            
            return CLIValidationResult(
                is_valid=True,
                errors=[],
                warnings=warnings,
                sanitized_value=parsed_yaml,
                suggestions=[]
            )
            
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML format: {str(e)}")
            suggestions.extend([
                "Check YAML syntax (indentation, colons, dashes)",
                "Example: key: value",
                "Use online YAML validator to check format"
            ])
            
            return CLIValidationResult(
                is_valid=False,
                errors=errors,
                warnings=[],
                sanitized_value=str_value,
                suggestions=suggestions
            )
    
    def _validate_regex_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Validate regex pattern parameters"""
        errors = []
        warnings = []
        suggestions = []
        
        str_value = str(value)
        
        try:
            # Test if regex compiles
            compiled_pattern = re.compile(str_value)
            
            # Check for potentially dangerous patterns
            if len(str_value) > 1000:
                warnings.append("Very long regex pattern may cause performance issues")
            
            # Check for catastrophic backtracking patterns
            dangerous_patterns = [r'\(.*\)\*', r'\(.*\)\+', r'.*\*.*\*']
            for pattern in dangerous_patterns:
                if re.search(pattern, str_value):
                    warnings.append("Regex pattern may cause performance issues (catastrophic backtracking)")
                    break
            
            return CLIValidationResult(
                is_valid=True,
                errors=[],
                warnings=warnings,
                sanitized_value=str_value,
                suggestions=[]
            )
            
        except re.error as e:
            errors.append(f"Invalid regex pattern: {str(e)}")
            suggestions.extend([
                "Check regex syntax",
                "Example: ^[a-zA-Z0-9]+$",
                "Use online regex tester to validate pattern"
            ])
            
            return CLIValidationResult(
                is_valid=False,
                errors=errors,
                warnings=[],
                sanitized_value=str_value,
                suggestions=suggestions
            )
    
    def _validate_choice_parameter(self, param_name: str, value: Any, 
                                 choices: List[str] = None) -> CLIValidationResult:
        """Validate choice parameters against allowed values"""
        errors = []
        warnings = []
        suggestions = []
        
        str_value = str(value).strip().lower()
        
        # Get choices based on parameter name if not provided
        if choices is None:
            choices = self._get_parameter_choices(param_name)
        
        if choices and str_value not in [choice.lower() for choice in choices]:
            errors.append(f"Invalid choice for '{param_name}': {value}")
            suggestions.append(f"Valid choices: {', '.join(choices)}")
            
            # Suggest closest match
            closest_match = self._find_closest_match(str_value, choices)
            if closest_match:
                suggestions.append(f"Did you mean: {closest_match}?")
        
        return CLIValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=value,
            suggestions=suggestions
        )
    
    def _validate_generic_parameter(self, param_name: str, value: Any) -> CLIValidationResult:
        """Generic parameter validation with basic sanitization"""
        warnings = []
        
        # Apply basic sanitization for string values
        if isinstance(value, str):
            sanitized_value = self.sanitizer.comprehensive_sanitize(value)
            if sanitized_value != value:
                warnings.append("Input sanitized for security")
        else:
            sanitized_value = value
        
        return CLIValidationResult(
            is_valid=True,
            errors=[],
            warnings=warnings,
            sanitized_value=sanitized_value,
            suggestions=[]
        )
    
    def _is_safe_absolute_path(self, path: Path) -> bool:
        """Check if absolute path is in allowed directories"""
        # Define allowed base directories
        allowed_bases = [
            Path.home(),
            Path.cwd(),
            Path('/tmp'),
            Path('/var/tmp')
        ]
        
        try:
            resolved_path = path.resolve()
            for base in allowed_bases:
                if resolved_path.is_relative_to(base.resolve()):
                    return True
        except (OSError, ValueError):
            pass
        
        return False
    
    def _get_pattern_suggestions(self, param_name: str) -> List[str]:
        """Get helpful suggestions for pattern validation failures"""
        suggestions_map = {
            'agent_id': [
                "Use only letters, numbers, underscores, and hyphens",
                "Example: 'my-agent' or 'agent_001'"
            ],
            'session_id': [
                "Session ID should be 8-64 characters",
                "Use alphanumeric characters, underscores, and hyphens"
            ],
            'framework_name': [
                "Valid frameworks: langchain, autogen, crewai, openai"
            ],
            'output_format': [
                "Valid formats: json, csv, table, yaml"
            ],
            'log_level': [
                "Valid levels: debug, info, warning, error, critical"
            ]
        }
        
        return suggestions_map.get(param_name, ["Check parameter format"])
    
    def _get_parameter_choices(self, param_name: str) -> List[str]:
        """Get valid choices for choice parameters"""
        choices_map = {
            'framework_name': ['langchain', 'autogen', 'crewai', 'openai'],
            'output_format': ['json', 'csv', 'table', 'yaml'],
            'log_level': ['debug', 'info', 'warning', 'error', 'critical'],
            'theme': ['dark', 'light', 'auto'],
            'sort_order': ['asc', 'desc'],
            'time_unit': ['seconds', 'minutes', 'hours', 'days']
        }
        
        return choices_map.get(param_name, [])
    
    def _find_closest_match(self, value: str, choices: List[str]) -> Optional[str]:
        """Find closest matching choice using simple string similarity"""
        if not choices:
            return None
        
        # Simple similarity based on common characters
        best_match = None
        best_score = 0
        
        for choice in choices:
            # Calculate similarity score
            common_chars = set(value.lower()) & set(choice.lower())
            score = len(common_chars) / max(len(value), len(choice))
            
            if score > best_score and score > 0.5:  # At least 50% similarity
                best_score = score
                best_match = choice
        
        return best_match
    
    def validate_command_args(self, command_name: str, args: Dict[str, Any], 
                            arg_specs: Dict[str, Dict[str, Any]]) -> Dict[str, CLIValidationResult]:
        """Validate all arguments for a command"""
        results = {}
        
        for arg_name, arg_value in args.items():
            arg_spec = arg_specs.get(arg_name, {})
            arg_type = arg_spec.get('type', 'string')
            
            result = self.validate_command_parameter(arg_name, arg_value, arg_type)
            results[arg_name] = result
        
        return results
    
    def create_validation_decorator(self, arg_specs: Dict[str, Dict[str, Any]]):
        """Create a decorator for command validation"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Validate kwargs against arg_specs
                validation_results = self.validate_command_args(
                    func.__name__, kwargs, arg_specs
                )
                
                # Check for validation errors
                errors = []
                for arg_name, result in validation_results.items():
                    if not result.is_valid:
                        errors.extend([f"{arg_name}: {error}" for error in result.errors])
                    
                    # Replace with sanitized value
                    if result.sanitized_value is not None:
                        kwargs[arg_name] = result.sanitized_value
                
                if errors:
                    raise CLIValidationError(
                        f"Validation failed for command {func.__name__}",
                        errors
                    )
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


# Convenience functions for common validations
def validate_agent_id(agent_id: str) -> CLIValidationResult:
    """Validate agent ID format"""
    validator = CLIInputValidator()
    return validator.validate_command_parameter('agent_id', agent_id, 'string')


def validate_file_path(file_path: str, must_exist: bool = True) -> CLIValidationResult:
    """Validate file path"""
    validator = CLIInputValidator()
    param_name = 'input_file' if must_exist else 'output_file'
    return validator.validate_command_parameter(param_name, file_path, 'file_path')


def validate_url(url: str) -> CLIValidationResult:
    """Validate URL format"""
    validator = CLIInputValidator()
    return validator.validate_command_parameter('url', url, 'url')


def validate_json_string(json_str: str) -> CLIValidationResult:
    """Validate JSON string"""
    validator = CLIInputValidator()
    return validator.validate_command_parameter('json_data', json_str, 'json')


def sanitize_user_input(user_input: str, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> str:
    """Sanitize user input for security"""
    sanitizer = SecuritySanitizer(validation_level)
    return sanitizer.comprehensive_sanitize(user_input)