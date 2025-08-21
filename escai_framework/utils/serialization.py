"""
Serialization utilities for the ESCAI framework.

This module provides common serialization functions for converting
data models to and from various formats (JSON, dict, etc.).
"""

import json
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
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


def from_json(json_str: str, object_hook: Optional[callable] = None) -> Any:
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


def to_dict(obj: Any, include_private: bool = False, max_depth: int = 10) -> Dict[str, Any]:
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
            return model_class.from_dict(data)
        else:
            # Try to create instance directly
            return model_class(**data)
    except Exception as e:
        raise SerializationError(f"Failed to create {model_class.__name__} from dict: {e}")


def to_pickle(obj: Any) -> bytes:
    """
    Serialize object to pickle bytes.
    
    Args:
        obj: Object to serialize
    
    Returns:
        Pickled bytes
    
    Raises:
        SerializationError: If serialization fails
    """
    try:
        return pickle.dumps(obj)
    except Exception as e:
        raise SerializationError(f"Failed to pickle object: {e}")


def from_pickle(data: bytes) -> Any:
    """
    Deserialize object from pickle bytes.
    
    Args:
        data: Pickled bytes
    
    Returns:
        Deserialized object
    
    Raises:
        SerializationError: If deserialization fails
    """
    try:
        return pickle.loads(data)
    except Exception as e:
        raise SerializationError(f"Failed to unpickle data: {e}")


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
        result = from_json(data)
        if not isinstance(result, list):
            raise SerializationError("Expected list from JSON deserialization")
        return result
    elif format == 'pickle':
        result = from_pickle(data)
        if not isinstance(result, list):
            raise SerializationError("Expected list from pickle deserialization")
        return result
    else:
        raise SerializationError(f"Unsupported format: {format}")


class SerializationRegistry:
    """Registry for custom serialization handlers."""
    
    def __init__(self):
        self._serializers: Dict[Type, callable] = {}
        self._deserializers: Dict[str, callable] = {}
    
    def register_serializer(self, type_class: Type, serializer: callable) -> None:
        """Register a custom serializer for a type."""
        self._serializers[type_class] = serializer
    
    def register_deserializer(self, type_name: str, deserializer: callable) -> None:
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


def register_serializer(type_class: Type, serializer: callable) -> None:
    """Register a custom serializer globally."""
    default_registry.register_serializer(type_class, serializer)


def register_deserializer(type_name: str, deserializer: callable) -> None:
    """Register a custom deserializer globally."""
    default_registry.register_deserializer(type_name, deserializer)


def safe_serialize(obj: Any, format: str = 'json', fallback_to_str: bool = True) -> Union[str, bytes]:
    """
    Safely serialize an object with fallback options.
    
    Args:
        obj: Object to serialize
        format: Serialization format
        fallback_to_str: Whether to fallback to string representation
    
    Returns:
        Serialized data
    """
    # Check format first - invalid formats should always raise an error
    if format not in ['json', 'pickle']:
        raise SerializationError(f"Unsupported format: {format}")
    
    try:
        if format == 'json':
            return to_json(obj)
        elif format == 'pickle':
            return to_pickle(obj)
    except SerializationError:
        if fallback_to_str:
            return str(obj) if format == 'json' else to_pickle(str(obj))
        raise