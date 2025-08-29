"""
Input Validation and Sanitization System

Provides comprehensive input validation and sanitization for all API endpoints
with configurable rules and security-focused validation.
"""

import re
import html
import urllib.parse
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import bleach
from pydantic import BaseModel, ValidationError, validator
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    PERMISSIVE = 1
    STANDARD = 2
    STRICT = 3
    PARANOID = 4


@dataclass
class ValidationRule:
    """Individual validation rule"""
    field_name: str
    rule_type: str
    parameters: Dict[str, Any]
    error_message: str
    required: bool = True


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[str]
    sanitized_data: Dict[str, Any]
    warnings: List[str] = None


class SecuritySanitizer:
    """Handles security-focused sanitization"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level
        self.allowed_tags = self._get_allowed_tags()
        self.allowed_attributes = self._get_allowed_attributes()
    
    def _get_allowed_tags(self) -> List[str]:
        """Get allowed HTML tags based on security level"""
        if self.level == ValidationLevel.PERMISSIVE:
            return ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'a', 'span', 'div']
        elif self.level == ValidationLevel.STANDARD:
            return ['p', 'br', 'strong', 'em', 'u']
        elif self.level == ValidationLevel.STRICT:
            return ['br']
        else:  # PARANOID
            return []
    
    def _get_allowed_attributes(self) -> Dict[str, List[str]]:
        """Get allowed HTML attributes based on security level"""
        if self.level == ValidationLevel.PERMISSIVE:
            return {
                'a': ['href', 'title'],
                'span': ['class'],
                'div': ['class']
            }
        elif self.level == ValidationLevel.STANDARD:
            return {'a': ['href']}
        else:
            return {}
    
    def sanitize_html(self, text: str) -> str:
        """Sanitize HTML content"""
        if not text:
            return ""
        
        # Use bleach to clean HTML
        cleaned = bleach.clean(
            text,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
        
        return cleaned
    
    def sanitize_sql_injection(self, text: str) -> str:
        """Sanitize potential SQL injection attempts"""
        if not text:
            return ""
        
        # Common SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?)",
            r"(;|\|\||&&)"
        ]
        
        sanitized = text
        for pattern in sql_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def sanitize_xss(self, text: str) -> str:
        """Sanitize XSS attempts"""
        if not text:
            return ""
        
        # HTML encode dangerous characters
        sanitized = html.escape(text, quote=True)
        
        # Remove javascript: URLs
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove data: URLs
        sanitized = re.sub(r'data:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove vbscript: URLs
        sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def sanitize_path_traversal(self, path: str) -> str:
        """Sanitize path traversal attempts"""
        if not path:
            return ""
        
        # Remove path traversal patterns
        sanitized = path.replace('..', '').replace('\\', '').replace('//', '/')
        
        # URL decode to catch encoded attempts
        sanitized = urllib.parse.unquote(sanitized)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        return sanitized.strip('/')
    
    def sanitize_command_injection(self, text: str) -> str:
        """Sanitize command injection attempts"""
        if not text:
            return ""
        
        # Command injection patterns
        dangerous_chars = ['|', '&', ';', '$', '`', '>', '<', '(', ')', '{', '}', '[', ']']
        
        sanitized = text
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()
    
    def sanitize_ldap_injection(self, text: str) -> str:
        """Sanitize LDAP injection attempts"""
        if not text:
            return ""
        
        # LDAP special characters
        ldap_chars = ['(', ')', '*', '\\', '/', '\x00']
        
        sanitized = text
        for char in ldap_chars:
            sanitized = sanitized.replace(char, f'\\{char}')
        
        return sanitized
    
    def comprehensive_sanitize(self, text: str) -> str:
        """Apply all sanitization methods"""
        if not text:
            return ""
        
        sanitized = text
        sanitized = self.sanitize_sql_injection(sanitized)
        sanitized = self.sanitize_xss(sanitized)
        sanitized = self.sanitize_command_injection(sanitized)
        sanitized = self.sanitize_ldap_injection(sanitized)
        
        return sanitized


class InputValidator:
    """Main input validation system"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level
        self.sanitizer = SecuritySanitizer(level)
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default validation rules"""
        
        # Common field validation rules
        self.validation_rules = {
            'email': [
                ValidationRule(
                    'email',
                    'regex',
                    {'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
                    'Invalid email format'
                ),
                ValidationRule(
                    'email',
                    'length',
                    {'min': 5, 'max': 254},
                    'Email must be between 5 and 254 characters'
                )
            ],
            'username': [
                ValidationRule(
                    'username',
                    'regex',
                    {'pattern': r'^[a-zA-Z0-9_-]{3,30}$'},
                    'Username must be 3-30 characters, alphanumeric, underscore, or dash only'
                )
            ],
            'password': [
                ValidationRule(
                    'password',
                    'length',
                    {'min': 8, 'max': 128},
                    'Password must be between 8 and 128 characters'
                ),
                ValidationRule(
                    'password',
                    'complexity',
                    {
                        'require_uppercase': True,
                        'require_lowercase': True,
                        'require_digit': True,
                        'require_special': True
                    },
                    'Password must contain uppercase, lowercase, digit, and special character'
                )
            ],
            'agent_id': [
                ValidationRule(
                    'agent_id',
                    'regex',
                    {'pattern': r'^[a-zA-Z0-9_-]{1,50}$'},
                    'Agent ID must be alphanumeric, underscore, or dash, max 50 characters'
                )
            ],
            'session_id': [
                ValidationRule(
                    'session_id',
                    'regex',
                    {'pattern': r'^[a-zA-Z0-9_-]{10,100}$'},
                    'Invalid session ID format'
                )
            ],
            'resource_id': [
                ValidationRule(
                    'resource_id',
                    'regex',
                    {'pattern': r'^[a-zA-Z0-9_-]{1,100}$'},
                    'Resource ID must be alphanumeric, underscore, or dash, max 100 characters'
                )
            ]
        }
    
    def add_validation_rule(self, field_name: str, rule: ValidationRule):
        """Add custom validation rule"""
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        self.validation_rules[field_name].append(rule)
    
    def validate_field(self, field_name: str, value: Any) -> ValidationResult:
        """Validate individual field"""
        errors = []
        warnings = []
        sanitized_value = value
        
        # Get validation rules for field
        rules = self.validation_rules.get(field_name, [])
        
        for rule in rules:
            try:
                # Check if field is required
                if rule.required and (value is None or value == ""):
                    errors.append(f"{field_name} is required")
                    continue
                
                # Skip validation if field is optional and empty
                if not rule.required and (value is None or value == ""):
                    continue
                
                # Apply validation based on rule type
                if rule.rule_type == 'regex':
                    if not self._validate_regex(value, rule.parameters['pattern']):
                        errors.append(rule.error_message)
                
                elif rule.rule_type == 'length':
                    if not self._validate_length(value, rule.parameters):
                        errors.append(rule.error_message)
                
                elif rule.rule_type == 'range':
                    if not self._validate_range(value, rule.parameters):
                        errors.append(rule.error_message)
                
                elif rule.rule_type == 'complexity':
                    if not self._validate_complexity(value, rule.parameters):
                        errors.append(rule.error_message)
                
                elif rule.rule_type == 'custom':
                    validator_func = rule.parameters.get('validator')
                    if validator_func and not validator_func(value):
                        errors.append(rule.error_message)
                
            except Exception as e:
                logger.error(f"Validation error for {field_name}: {e}")
                errors.append(f"Validation failed for {field_name}")
        
        # Apply sanitization if string
        if isinstance(sanitized_value, str):
            original_value = sanitized_value
            sanitized_value = self.sanitizer.comprehensive_sanitize(sanitized_value)
            
            if original_value != sanitized_value:
                warnings.append(f"Input sanitized for {field_name}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data={field_name: sanitized_value},
            warnings=warnings
        )
    
    def validate_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate entire data dictionary"""
        all_errors = []
        all_warnings = []
        sanitized_data: Dict[str, Any] = {}
        
        for field_name, value in data.items():
            result = self.validate_field(field_name, value)
            
            if not result.is_valid:
                all_errors.extend(result.errors)
            
            if result.warnings:
                all_warnings.extend(result.warnings)
            
            sanitized_data.update(result.sanitized_data)
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            sanitized_data=sanitized_data,
            warnings=all_warnings
        )
    
    def _validate_regex(self, value: str, pattern: str) -> bool:
        """Validate using regex pattern"""
        try:
            return bool(re.match(pattern, str(value)))
        except re.error:
            return False
    
    def _validate_length(self, value: Any, params: Dict[str, int]) -> bool:
        """Validate length constraints"""
        length = len(str(value)) if value is not None else 0
        min_len = params.get('min', 0)
        max_len = params.get('max', float('inf'))
        
        return min_len <= length <= max_len
    
    def _validate_range(self, value: Any, params: Dict[str, Union[int, float]]) -> bool:
        """Validate numeric range"""
        try:
            num_value = float(value)
            min_val = params.get('min', float('-inf'))
            max_val = params.get('max', float('inf'))
            
            return min_val <= num_value <= max_val
        except (ValueError, TypeError):
            return False
    
    def _validate_complexity(self, value: str, params: Dict[str, bool]) -> bool:
        """Validate password complexity"""
        if not isinstance(value, str):
            return False
        
        checks = []
        
        if params.get('require_uppercase', False):
            checks.append(any(c.isupper() for c in value))
        
        if params.get('require_lowercase', False):
            checks.append(any(c.islower() for c in value))
        
        if params.get('require_digit', False):
            checks.append(any(c.isdigit() for c in value))
        
        if params.get('require_special', False):
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            checks.append(any(c in special_chars for c in value))
        
        return all(checks)
    
    def validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
        """Validate data against JSON schema"""
        try:
            # This would integrate with jsonschema library
            # For now, basic implementation
            errors = []
            sanitized_data: Dict[str, Any] = {}
            
            for field, field_schema in schema.get('properties', {}).items():
                value = data.get(field)
                
                # Check required fields
                if field in schema.get('required', []) and value is None:
                    errors.append(f"{field} is required")
                    continue
                
                # Type validation
                expected_type = field_schema.get('type')
                if value is not None and expected_type:
                    if not self._validate_type(value, expected_type):
                        errors.append(f"{field} must be of type {expected_type}")
                        continue
                
                # Apply sanitization
                if isinstance(value, str):
                    sanitized_data[field] = self.sanitizer.comprehensive_sanitize(value)
                else:
                    sanitized_data[field] = value
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                sanitized_data=sanitized_data
            )
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=["Schema validation failed"],
                sanitized_data={}
            )
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True
    
    def create_api_validator(self, endpoint_name: str) -> Callable:
        """Create validator decorator for API endpoints"""
        def validator_decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract request data (this would be framework-specific)
                request_data = kwargs.get('request_data', {})
                
                # Validate data
                result = self.validate_data(request_data)
                
                if not result.is_valid:
                    # Return validation error response
                    return {
                        'error': 'Validation failed',
                        'details': result.errors,
                        'status_code': 400
                    }
                
                # Replace request data with sanitized version
                kwargs['request_data'] = result.sanitized_data
                
                # Log warnings if any
                if result.warnings:
                    logger.warning(f"Input sanitization warnings for {endpoint_name}: {result.warnings}")
                
                return await func(*args, **kwargs)
            
            return wrapper
        return validator_decorator