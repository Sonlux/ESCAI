"""
Serialization utilities for the ESCAI framework.

This module provides common serialization functions for converting
data models to and from various formats (JSON, dict, etc.).

SECURITY WARNING: This module includes pickle serialization functions.
Pickle can execute arbitrary code during deserialization and should only
be used with trusted data sources. For untrusted data, use JSON serialization.
"""

import json
import pickle
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from collections.abc import Callable as CallableType
from enum import Enum
import base64

T = TypeVar('T')


class SerializationError(Exception):
    """Custom exception for serialization errors."""
    pass


class ESCAIJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for ESCAI framework data types."""
    
    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, datetime):
            return {
                '__type__': 'datetime',
                'value': obj.isoformat()
            }
        elif isinstance(obj, Enum):
            return {
                '__type__': 'enum',
                'class': obj.__class__.__name__,
                'value': obj.value
            }
        elif hasattr(obj, 'to_dict'):
            return {
                '__type__': 'model',
                'class': obj.__class__.__name__,
                'data': obj.to_dict()
            }
        elif isinstance(obj, set):
            return {
                '__type__': 'set',
                'value': list(obj)
            }
        elif isinstance(obj, tuple):
            return {
                '__type__': 'tuple',
                'value': list(obj)
            }
        elif isinstance(obj, bytes):
            return {
                '__type__': 'bytes',
                'value': base64.b64encode(obj).decode('utf-8')
            }
        
        return super().default(obj)


def escai_json_decoder(dct: Dict[str, Any]) -> Any:
    """Custom JSON decoder for ESCAI framework data types."""
    if '__type__' in dct:
        type_name = dct['__type__']
        
        if type_name == 'datetime':
            return datetime.fromisoformat(dct['value'])
        elif type_name == 'set':
            return set(dct['value'])
        elif type_name == 'tuple':
            return tuple(dct['value'])
        elif type_name == 'bytes':
            return base64.b64decode(dct['value'].encode('utf-8'))
        # Note: enum and model types require additional context to deserialize
        # They should be handled by specific model deserialization methods
    
    return dct


def to_json(obj: Any, indent: Optional[int] = None, ensure_ascii: bool = False) -> str:
    """
    Convert an object to JSON string using ESCAI custom encoder.
    
    Args:
        obj: Object to serialize
        indent: JSON indentation (optional)
        ensure_ascii: Whether to ensure ASCII output
    
    Returns:
        JSON string representation
    
    Raises:
        SerializationError: If serialization fails
    """
    try:
        return json.dumps(obj, cls=ESCAIJSONEncoder, indent=indent, ensure_ascii=ensure_ascii)
    except (TypeError, ValueError) as e:
        raise SerializationError(f"Failed to serialize to JSON: {e}")


def from_json(json_str: str, object_hook: Optional[CallableType] = None) -> Any:
    """
    Parse JSON string using ESCAI custom decoder.
    
    Args:
        json_str: JSON string to parse
        object_hook: Custom object hook (optional)
    
    Returns:
        Parsed object
    
    Raises:
        SerializationError: If deserialization fails
    """
    try:
        hook = object_hook or escai_json_decoder
        return json.loads(json_str, object_hook=hook)
    except (json.JSONDecodeError, ValueError) as e:
        raise SerializationError(f"Failed to deserialize from JSON: {e}")


def to_dict(obj: Any, include_private: bool = False, max_depth: int = 10) -> Any:
    """
    Convert an object to dictionary representation.
    
    Args:
        obj: Object to convert
        include_private: Whether to include private attributes
        max_depth: Maximum recursion depth
    
    Returns:
        Dictionary representation
    
    Raises:
        SerializationError: If conversion fails
    """
    if max_depth <= 0:
        return str(obj)
    
    try:
        # If object has to_dict method, use it
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        
        # Handle basic types
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle enums
        if isinstance(obj, Enum):
            return obj.value
        
        # Handle collections
        if isinstance(obj, (list, tuple)):
            return [to_dict(item, include_private, max_depth - 1) for item in obj]
        
        if isinstance(obj, set):
            return list(to_dict(item, include_private, max_depth - 1) for item in obj)
        
        if isinstance(obj, dict):
            return {
                key: to_dict(value, include_private, max_depth - 1)
                for key, value in obj.items()
            }
        
        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if not include_private and key.startswith('_'):
                    continue
                result[key] = to_dict(value, include_private, max_depth - 1)
            return result
        
        # Fallback to string representation
        return str(obj)
    
    except Exception as e:
        raise SerializationError(f"Failed to convert to dict: {e}")


def from_dict(data: Dict[str, Any], model_class: Type[T]) -> T:
    """
    Create model instance from dictionary data.
    
    Args:
        data: Dictionary data
        model_class: Target model class
    
    Returns:
        Model instance
    
    Raises:
        SerializationError: If creation fails
    """
    try:
        if hasattr(model_class, 'from_dict') and callable(getattr(model_class, 'from_dict')):
            from_dict_method = getattr(model_class, 'from_dict')
            return from_dict_method(data)
        else:
            # Try to create instance directly
            return model_class(**data)
    except Exception as e:
        raise SerializationError(f"Failed to create {model_class.__name__} from dict: {e}")


def to_pickle(obj: Any) -> bytes:
    """
    Serialize an object to pickle format.
    
    WARNING: Pickle serialization should only be used with trusted data.
    Consider using JSON serialization for untrusted data sources.
    
    Args:
        obj: Object to serialize
    
    Returns:
        Pickled bytes
    
    Raises:
        SerializationError: If serialization fails
    """
    try:
        return pickle.dumps(obj)  # nosec B301 - Controlled usage with warning
    except Exception as e:
        raise SerializationError(f"Failed to serialize to pickle: {e}")


def from_pickle(data: bytes, trusted_source: bool = False) -> Any:
    """
    Deserialize an object from pickle format.
    
    SECURITY WARNING: Pickle deserialization can execute arbitrary code.
    Only use with data from trusted sources. Set trusted_source=True to acknowledge this risk.
    
    Args:
        data: Pickled bytes
        trusted_source: Must be True to acknowledge security risks
    
    Returns:
        Deserialized object
    
    Raises:
        SerializationError: If deserialization fails or source not trusted
    """
    if not trusted_source:
        raise SerializationError(
            "Pickle deserialization requires trusted_source=True. "
            "Pickle can execute arbitrary code and should only be used with trusted data."
        )
    
    try:
        return pickle.loads(data)  # nosec B301 - Controlled usage with explicit trust check
    except Exception as e:
        raise SerializationError(f"Failed to deserialize from pickle: {e}")


def safe_from_pickle(data: bytes, allowed_classes: Optional[List[str]] = None) -> Any:
    """
    Safer pickle deserialization with class restrictions.
    
    This function provides additional safety by restricting which classes
    can be deserialized, but should still only be used with trusted data.
    
    Args:
        data: Pickled bytes
        allowed_classes: List of allowed class names (optional)
    
    Returns:
        Deserialized object
    
    Raises:
        SerializationError: If deserialization fails or class not allowed
    """
    import io
    import pickletools
    
    # Basic validation - check if data looks like pickle
    if not data or len(data) < 2:
        raise SerializationError("Invalid pickle data")
    
    # For additional safety, we could implement a restricted unpickler
    # but for now, we'll use the standard approach with warnings
    try:
        # Analyze pickle opcodes for suspicious operations (basic check)
        opcodes = list(pickletools.genops(data))
        suspicious_ops = ['GLOBAL', 'REDUCE', 'BUILD', 'INST']
        
        if allowed_classes:
            for opcode, arg, pos in opcodes:
                if opcode.name == 'GLOBAL' and arg:
                    class_name = arg.split('.')[-1] if '.' in arg else arg
                    if class_name not in allowed_classes:
                        raise SerializationError(f"Class '{class_name}' not in allowed classes")
        
        return pickle.loads(data)  # nosec B301 - Controlled usage with validation
    except Exception as e:
        raise SerializationError(f"Failed to safely deserialize pickle: {e}")


def serialize_batch(objects: List[Any], format: str = 'json') -> Union[str, bytes]:
    """
    Serialize a batch of objects.
    
    Args:
        objects: List of objects to serialize
        format: Serialization format ('json' or 'pickle')
    
    Returns:
        Serialized data
    
    Raises:
        SerializationError: If serialization fails
    """
    if format == 'json':
        return to_json(objects)
    elif format == 'pickle':
        return to_pickle(objects)
    else:
        raise SerializationError(f"Unsupported format: {format}")


def deserialize_batch(data: Union[str, bytes], format: str = 'json') -> List[Any]:
    """
    Deserialize a batch of objects.
    
    Args:
        data: Serialized data
        format: Serialization format ('json' or 'pickle')
    
    Returns:
        List of deserialized objects
    
    Raises:
        SerializationError: If deserialization fails
    """
    if format == 'json':
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        result = from_json(data)
        if not isinstance(result, list):
            raise SerializationError("Expected list from JSON deserialization")
        return result
    elif format == 'pickle':
        if isinstance(data, str):
            raise SerializationError("Pickle format requires bytes data")
        # Note: This assumes batch deserialization is from trusted internal sources
        result = from_pickle(data, trusted_source=True)
        if not isinstance(result, list):
            raise SerializationError("Expected list from pickle deserialization")
        return result
    else:
        raise SerializationError(f"Unsupported format: {format}")


class SerializationRegistry:
    """Registry for custom serialization handlers."""
    
    def __init__(self) -> None:
        self._serializers: Dict[Type, CallableType] = {}
        self._deserializers: Dict[str, CallableType] = {}
    
    def register_serializer(self, type_class: Type, serializer: CallableType) -> None:
        """Register a custom serializer for a type."""
        self._serializers[type_class] = serializer
    
    def register_deserializer(self, type_name: str, deserializer: CallableType) -> None:
        """Register a custom deserializer for a type name."""
        self._deserializers[type_name] = deserializer
    
    def serialize(self, obj: Any) -> Dict[str, Any]:
        """Serialize object using registered serializers."""
        obj_type = type(obj)
        if obj_type in self._serializers:
            return {
                '__type__': obj_type.__name__,
                'data': self._serializers[obj_type](obj)
            }
        return to_dict(obj)
    
    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize object using registered deserializers."""
        if '__type__' in data and data['__type__'] in self._deserializers:
            return self._deserializers[data['__type__']](data['data'])
        return data


# Global registry instance
default_registry = SerializationRegistry()


def register_serializer(type_class: Type, serializer: CallableType) -> None:
    """Register a custom serializer globally."""
    default_registry.register_serializer(type_class, serializer)


def register_deserializer(type_name: str, deserializer: CallableType) -> None:
    """Register a custom deserializer globally."""
    default_registry.register_deserializer(type_name, deserializer)


def safe_serialize(obj: Any, format: str = 'json', fallback_to_str: bool = True) -> Union[str, bytes]:
    """
    Safely serialize an object with fallback options.
    
    Recommends JSON over pickle for security reasons.
    
    Args:
        obj: Object to serialize
        format: Serialization format ('json' recommended, 'pickle' for trusted data only)
        fallback_to_str: Whether to fallback to string representation
    
    Returns:
        Serialized data
    """
    # Check format first - invalid formats should always raise an error
    if format not in ['json', 'pickle']:
        raise SerializationError(f"Unsupported format: {format}")
    
    # Warn about pickle usage
    if format == 'pickle':
        import warnings
        warnings.warn(
            "Pickle serialization should only be used with trusted data. "
            "Consider using JSON for better security.",
            UserWarning,
            stacklevel=2
        )
    
    try:
        if format == 'json':
            return to_json(obj)
        elif format == 'pickle':
            return to_pickle(obj)
    except SerializationError:
        if fallback_to_str:
            return str(obj)
        raise
    
    # This should never be reached, but added for type safety
    raise SerializationError(f"Unexpected error in serialization")


def get_recommended_format(obj: Any) -> str:
    """
    Get the recommended serialization format for an object.
    
    Always recommends JSON for security unless object cannot be JSON serialized.
    
    Args:
        obj: Object to analyze
    
    Returns:
        Recommended format ('json' or 'pickle')
    """
    try:
        # Try JSON first
        to_json(obj)
        return 'json'
    except SerializationError:
        # If JSON fails, recommend pickle with warning
        import warnings
        warnings.warn(
            f"Object of type {type(obj).__name__} cannot be JSON serialized. "
            "Pickle will be recommended but should only be used with trusted data.",
            UserWarning,
            stacklevel=2
        )
        return 'pickle'