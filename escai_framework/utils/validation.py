"""
Validation utilities for the ESCAI framework.

This module provides common validation functions and decorators
for data model validation across the framework.
"""

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from datetime import datetime
import re
import functools

T = TypeVar('T')


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format the error message with field and value information."""
        if self.field:
            return f"Validation error in field '{self.field}': {self.message}"
        return f"Validation error: {self.message}"


def validate_string(value: Any, field_name: str, min_length: int = 1, max_length: Optional[int] = None, 
                   pattern: Optional[str] = None, allow_empty: bool = False) -> str:
    """
    Validate string values with optional constraints.
    
    Args:
        value: The value to validate
        field_name: Name of the field being validated
        min_length: Minimum string length (default: 1)
        max_length: Maximum string length (optional)
        pattern: Regex pattern to match (optional)
        allow_empty: Whether to allow empty strings (default: False)
    
    Returns:
        The validated string value
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"Expected string, got {type(value).__name__}", field_name, value)
    
    if not allow_empty and not value.strip():
        raise ValidationError("String cannot be empty or whitespace only", field_name, value)
    
    # Only check min_length if not allowing empty strings or if the string is not empty
    if len(value) < min_length and not (allow_empty and len(value) == 0):
        raise ValidationError(f"String length must be at least {min_length}", field_name, value)
    
    if max_length is not None and len(value) > max_length:
        raise ValidationError(f"String length must not exceed {max_length}", field_name, value)
    
    if pattern is not None and not re.match(pattern, value):
        raise ValidationError(f"String does not match required pattern: {pattern}", field_name, value)
    
    return value


def validate_number(value: Any, field_name: str, min_value: Optional[float] = None, 
                   max_value: Optional[float] = None, allow_int: bool = True, 
                   allow_float: bool = True) -> Union[int, float]:
    """
    Validate numeric values with optional constraints.
    
    Args:
        value: The value to validate
        field_name: Name of the field being validated
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        allow_int: Whether to allow integer values (default: True)
        allow_float: Whether to allow float values (default: True)
    
    Returns:
        The validated numeric value
    
    Raises:
        ValidationError: If validation fails
    """
    if not allow_int and isinstance(value, int):
        raise ValidationError("Integer values not allowed", field_name, value)
    
    if not allow_float and isinstance(value, float):
        raise ValidationError("Float values not allowed", field_name, value)
    
    if not isinstance(value, (int, float)):
        raise ValidationError(f"Expected numeric value, got {type(value).__name__}", field_name, value)
    
    if min_value is not None and value < min_value:
        raise ValidationError(f"Value must be at least {min_value}", field_name, value)
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"Value must not exceed {max_value}", field_name, value)
    
    return value


def validate_probability(value: Any, field_name: str) -> float:
    """
    Validate probability values (0.0 to 1.0).
    
    Args:
        value: The value to validate
        field_name: Name of the field being validated
    
    Returns:
        The validated probability value
    
    Raises:
        ValidationError: If validation fails
    """
    return validate_number(value, field_name, min_value=0.0, max_value=1.0)


def validate_datetime(value: Any, field_name: str, allow_future: bool = True, 
                     allow_past: bool = True) -> datetime:
    """
    Validate datetime values with optional constraints.
    
    Args:
        value: The value to validate
        field_name: Name of the field being validated
        allow_future: Whether to allow future dates (default: True)
        allow_past: Whether to allow past dates (default: True)
    
    Returns:
        The validated datetime value
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, datetime):
        raise ValidationError(f"Expected datetime, got {type(value).__name__}", field_name, value)
    
    now = datetime.utcnow()
    
    if not allow_future and value > now:
        raise ValidationError("Future dates not allowed", field_name, value)
    
    if not allow_past and value < now:
        raise ValidationError("Past dates not allowed", field_name, value)
    
    return value


def validate_list(value: Any, field_name: str, min_length: int = 0, max_length: Optional[int] = None,
                 item_validator: Optional[Callable[[Any, str], Any]] = None) -> List[Any]:
    """
    Validate list values with optional constraints.
    
    Args:
        value: The value to validate
        field_name: Name of the field being validated
        min_length: Minimum list length (default: 0)
        max_length: Maximum list length (optional)
        item_validator: Function to validate each item (optional)
    
    Returns:
        The validated list value
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, list):
        raise ValidationError(f"Expected list, got {type(value).__name__}", field_name, value)
    
    if len(value) < min_length:
        raise ValidationError(f"List length must be at least {min_length}", field_name, value)
    
    if max_length is not None and len(value) > max_length:
        raise ValidationError(f"List length must not exceed {max_length}", field_name, value)
    
    if item_validator is not None:
        for i, item in enumerate(value):
            try:
                item_validator(item, f"{field_name}[{i}]")
            except ValidationError as e:
                raise ValidationError(f"Item {i} validation failed: {e.message}", field_name, item)
    
    return value


def validate_dict(value: Any, field_name: str, required_keys: Optional[List[str]] = None,
                 optional_keys: Optional[List[str]] = None, 
                 key_validator: Optional[Callable[[Any, str], Any]] = None,
                 value_validator: Optional[Callable[[Any, str], Any]] = None) -> Dict[str, Any]:
    """
    Validate dictionary values with optional constraints.
    
    Args:
        value: The value to validate
        field_name: Name of the field being validated
        required_keys: List of required keys (optional)
        optional_keys: List of optional keys (optional)
        key_validator: Function to validate each key (optional)
        value_validator: Function to validate each value (optional)
    
    Returns:
        The validated dictionary value
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, dict):
        raise ValidationError(f"Expected dict, got {type(value).__name__}", field_name, value)
    
    if required_keys:
        missing_keys = set(required_keys) - set(value.keys())
        if missing_keys:
            raise ValidationError(f"Missing required keys: {missing_keys}", field_name, value)
    
    if optional_keys is not None:
        allowed_keys = set(required_keys or []) | set(optional_keys)
        extra_keys = set(value.keys()) - allowed_keys
        if extra_keys:
            raise ValidationError(f"Unexpected keys: {extra_keys}", field_name, value)
    
    if key_validator is not None:
        for key in value.keys():
            try:
                key_validator(key, f"{field_name}.key")
            except ValidationError as e:
                raise ValidationError(f"Key '{key}' validation failed: {e.message}", field_name, key)
    
    if value_validator is not None:
        for key, val in value.items():
            try:
                value_validator(val, f"{field_name}[{key}]")
            except ValidationError as e:
                raise ValidationError(f"Value for key '{key}' validation failed: {e.message}", field_name, val)
    
    return value


def validate_enum(value: Any, field_name: str, enum_class: Type) -> Any:
    """
    Validate enum values.
    
    Args:
        value: The value to validate
        field_name: Name of the field being validated
        enum_class: The enum class to validate against
    
    Returns:
        The validated enum value
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, enum_class):
        valid_values = [e.value for e in enum_class]
        raise ValidationError(f"Expected {enum_class.__name__}, got {type(value).__name__}. "
                            f"Valid values: {valid_values}", field_name, value)
    
    return value


def validate_id(value: Any, field_name: str, pattern: str = r'^[a-zA-Z0-9_-]+$') -> str:
    """
    Validate ID strings with a default pattern.
    
    Args:
        value: The value to validate
        field_name: Name of the field being validated
        pattern: Regex pattern for valid IDs (default: alphanumeric, underscore, hyphen)
    
    Returns:
        The validated ID string
    
    Raises:
        ValidationError: If validation fails
    """
    return validate_string(value, field_name, min_length=1, max_length=255, pattern=pattern)


def validation_decorator(validator_func: Callable[[Any], bool]) -> Callable:
    """
    Decorator to add validation to methods.
    
    Args:
        validator_func: Function that returns True if validation passes
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not validator_func(self):
                raise ValidationError(f"Validation failed for {self.__class__.__name__}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def validate_model_instance(instance: Any, model_class: Type[T]) -> T:
    """
    Validate that an instance is of the expected model class and passes validation.
    
    Args:
        instance: The instance to validate
        model_class: The expected model class
    
    Returns:
        The validated instance
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(instance, model_class):
        raise ValidationError(f"Expected {model_class.__name__}, got {type(instance).__name__}")
    
    if hasattr(instance, 'validate') and not instance.validate():
        raise ValidationError(f"Instance validation failed for {model_class.__name__}")
    
    return instance


class ValidationContext:
    """Context manager for collecting validation errors."""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.errors:
            error_messages = [str(error) for error in self.errors]
            raise ValidationError(f"Multiple validation errors: {'; '.join(error_messages)}")
    
    def validate(self, validator_func: Callable, *args, **kwargs) -> Any:
        """Run a validator function and collect any errors."""
        try:
            return validator_func(*args, **kwargs)
        except ValidationError as e:
            self.errors.append(e)
            return None
    
    def has_errors(self) -> bool:
        """Check if any validation errors were collected."""
        return len(self.errors) > 0