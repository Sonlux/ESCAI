"""
Unit tests for serialization utilities.
"""

import pytest
import json
import pickle
from datetime import datetime
from enum import Enum

from escai_framework.utils.serialization import (
    SerializationError, ESCAIJSONEncoder, escai_json_decoder,
    to_json, from_json, to_dict, from_dict, to_pickle, from_pickle,
    serialize_batch, deserialize_batch, SerializationRegistry,
    safe_serialize
)


class TestEnum(Enum):
    """Test enum for serialization tests."""
    VALUE1 = "value1"
    VALUE2 = "value2"


class TestModel:
    """Test model class for serialization tests."""
    
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value
    
    def to_dict(self):
        return {"name": self.name, "value": self.value}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["value"])


class TestESCAIJSONEncoder:
    """Test cases for ESCAIJSONEncoder."""
    
    def test_encode_datetime(self):
        """Test encoding datetime objects."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        encoder = ESCAIJSONEncoder()
        result = encoder.default(dt)
        
        assert result["__type__"] == "datetime"
        assert result["value"] == dt.isoformat()
    
    def test_encode_enum(self):
        """Test encoding enum objects."""
        enum_val = TestEnum.VALUE1
        encoder = ESCAIJSONEncoder()
        result = encoder.default(enum_val)
        
        assert result["__type__"] == "enum"
        assert result["class"] == "TestEnum"
        assert result["value"] == "value1"
    
    def test_encode_model_with_to_dict(self):
        """Test encoding objects with to_dict method."""
        model = TestModel("test", 42)
        encoder = ESCAIJSONEncoder()
        result = encoder.default(model)
        
        assert result["__type__"] == "model"
        assert result["class"] == "TestModel"
        assert result["data"] == {"name": "test", "value": 42}
    
    def test_encode_set(self):
        """Test encoding set objects."""
        test_set = {1, 2, 3}
        encoder = ESCAIJSONEncoder()
        result = encoder.default(test_set)
        
        assert result["__type__"] == "set"
        assert set(result["value"]) == test_set
    
    def test_encode_tuple(self):
        """Test encoding tuple objects."""
        test_tuple = (1, 2, 3)
        encoder = ESCAIJSONEncoder()
        result = encoder.default(test_tuple)
        
        assert result["__type__"] == "tuple"
        assert result["value"] == [1, 2, 3]
    
    def test_encode_bytes(self):
        """Test encoding bytes objects."""
        test_bytes = b"hello world"
        encoder = ESCAIJSONEncoder()
        result = encoder.default(test_bytes)
        
        assert result["__type__"] == "bytes"
        assert isinstance(result["value"], str)


class TestESCAIJSONDecoder:
    """Test cases for escai_json_decoder."""
    
    def test_decode_datetime(self):
        """Test decoding datetime objects."""
        dt_data = {
            "__type__": "datetime",
            "value": "2023-01-01T12:00:00"
        }
        result = escai_json_decoder(dt_data)
        
        assert isinstance(result, datetime)
        assert result == datetime(2023, 1, 1, 12, 0, 0)
    
    def test_decode_set(self):
        """Test decoding set objects."""
        set_data = {
            "__type__": "set",
            "value": [1, 2, 3]
        }
        result = escai_json_decoder(set_data)
        
        assert isinstance(result, set)
        assert result == {1, 2, 3}
    
    def test_decode_tuple(self):
        """Test decoding tuple objects."""
        tuple_data = {
            "__type__": "tuple",
            "value": [1, 2, 3]
        }
        result = escai_json_decoder(tuple_data)
        
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)
    
    def test_decode_bytes(self):
        """Test decoding bytes objects."""
        import base64
        test_bytes = b"hello world"
        bytes_data = {
            "__type__": "bytes",
            "value": base64.b64encode(test_bytes).decode('utf-8')
        }
        result = escai_json_decoder(bytes_data)
        
        assert isinstance(result, bytes)
        assert result == test_bytes
    
    def test_decode_regular_dict(self):
        """Test decoding regular dictionary."""
        regular_dict = {"key": "value", "number": 42}
        result = escai_json_decoder(regular_dict)
        
        assert result == regular_dict


class TestJSONSerialization:
    """Test cases for JSON serialization functions."""
    
    def test_to_json_basic(self):
        """Test basic JSON serialization."""
        data = {"name": "test", "value": 42}
        result = to_json(data)
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == data
    
    def test_to_json_with_datetime(self):
        """Test JSON serialization with datetime."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        data = {"timestamp": dt, "value": 42}
        result = to_json(data)
        
        assert isinstance(result, str)
        # Should not raise exception
    
    def test_to_json_with_indent(self):
        """Test JSON serialization with indentation."""
        data = {"key": "value"}
        result = to_json(data, indent=2)
        
        assert isinstance(result, str)
        assert "\n" in result  # Should have newlines from indentation
    
    def test_to_json_serialization_error(self):
        """Test JSON serialization error handling."""
        # Create an object that can't be serialized
        class UnserializableClass:
            def __init__(self):
                self.circular_ref = self
        
        with pytest.raises(SerializationError):
            to_json(UnserializableClass())
    
    def test_from_json_basic(self):
        """Test basic JSON deserialization."""
        json_str = '{"name": "test", "value": 42}'
        result = from_json(json_str)
        
        assert result == {"name": "test", "value": 42}
    
    def test_from_json_with_custom_decoder(self):
        """Test JSON deserialization with custom decoder."""
        dt_json = '{"__type__": "datetime", "value": "2023-01-01T12:00:00"}'
        result = from_json(dt_json)
        
        assert isinstance(result, datetime)
        assert result == datetime(2023, 1, 1, 12, 0, 0)
    
    def test_from_json_deserialization_error(self):
        """Test JSON deserialization error handling."""
        invalid_json = '{"invalid": json}'
        
        with pytest.raises(SerializationError):
            from_json(invalid_json)


class TestDictSerialization:
    """Test cases for dictionary serialization functions."""
    
    def test_to_dict_basic(self):
        """Test basic dictionary conversion."""
        data = {"key": "value", "number": 42}
        result = to_dict(data)
        
        assert result == data
    
    def test_to_dict_with_model(self):
        """Test dictionary conversion with model object."""
        model = TestModel("test", 42)
        result = to_dict(model)
        
        assert result == {"name": "test", "value": 42}
    
    def test_to_dict_with_datetime(self):
        """Test dictionary conversion with datetime."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        result = to_dict(dt)
        
        assert result == dt.isoformat()
    
    def test_to_dict_with_enum(self):
        """Test dictionary conversion with enum."""
        enum_val = TestEnum.VALUE1
        result = to_dict(enum_val)
        
        assert result == "value1"
    
    def test_to_dict_with_collections(self):
        """Test dictionary conversion with collections."""
        data = {
            "list": [1, 2, 3],
            "tuple": (4, 5, 6),
            "set": {7, 8, 9}
        }
        result = to_dict(data)
        
        assert result["list"] == [1, 2, 3]
        assert result["tuple"] == [4, 5, 6]
        assert set(result["set"]) == {7, 8, 9}
    
    def test_to_dict_with_nested_object(self):
        """Test dictionary conversion with nested objects."""
        class NestedClass:
            def __init__(self):
                self.value = 42
                self.name = "nested"
        
        nested = NestedClass()
        result = to_dict(nested)
        
        assert result == {"value": 42, "name": "nested"}
    
    def test_to_dict_max_depth(self):
        """Test dictionary conversion with max depth limit."""
        class DeepClass:
            def __init__(self, depth):
                if depth > 0:
                    self.nested = DeepClass(depth - 1)
                else:
                    self.value = "deep"
        
        deep_obj = DeepClass(5)
        result = to_dict(deep_obj, max_depth=2)
        
        # Should limit depth and convert to string at max depth
        assert isinstance(result, dict)
    
    def test_to_dict_include_private(self):
        """Test dictionary conversion including private attributes."""
        class TestClass:
            def __init__(self):
                self.public = "public"
                self._private = "private"
        
        obj = TestClass()
        
        # Without private attributes
        result_no_private = to_dict(obj, include_private=False)
        assert "_private" not in result_no_private
        assert "public" in result_no_private
        
        # With private attributes
        result_with_private = to_dict(obj, include_private=True)
        assert "_private" in result_with_private
        assert "public" in result_with_private
    
    def test_from_dict_with_model(self):
        """Test creating model from dictionary."""
        data = {"name": "test", "value": 42}
        result = from_dict(data, TestModel)
        
        assert isinstance(result, TestModel)
        assert result.name == "test"
        assert result.value == 42
    
    def test_from_dict_serialization_error(self):
        """Test from_dict error handling."""
        class BadModel:
            def __init__(self, required_param):
                self.required_param = required_param
        
        with pytest.raises(SerializationError):
            from_dict({}, BadModel)  # Missing required parameter


class TestPickleSerialization:
    """Test cases for pickle serialization functions."""
    
    def test_to_pickle_basic(self):
        """Test basic pickle serialization."""
        data = {"key": "value", "number": 42}
        result = to_pickle(data)
        
        assert isinstance(result, bytes)
        unpickled = pickle.loads(result)
        assert unpickled == data
    
    def test_to_pickle_with_objects(self):
        """Test pickle serialization with complex objects."""
        model = TestModel("test", 42)
        result = to_pickle(model)
        
        assert isinstance(result, bytes)
        unpickled = pickle.loads(result)
        assert unpickled.name == "test"
        assert unpickled.value == 42
    
    def test_from_pickle_basic(self):
        """Test basic pickle deserialization."""
        data = {"key": "value", "number": 42}
        pickled = pickle.dumps(data)
        result = from_pickle(pickled, trusted_source=True)
        
        assert result == data
    
    def test_pickle_serialization_error(self):
        """Test pickle serialization error handling."""
        # Create an unpicklable object
        import threading
        lock = threading.Lock()
        
        with pytest.raises(SerializationError):
            to_pickle(lock)
    
    def test_pickle_deserialization_error(self):
        """Test pickle deserialization error handling."""
        invalid_pickle = b"invalid pickle data"
        
        with pytest.raises(SerializationError):
            from_pickle(invalid_pickle, trusted_source=True)
    
    def test_pickle_security_check(self):
        """Test that pickle deserialization requires trusted_source=True."""
        data = {"key": "value"}
        pickled = pickle.dumps(data)
        
        # Should raise error without trusted_source=True
        with pytest.raises(SerializationError, match="trusted_source=True"):
            from_pickle(pickled)
        
        # Should work with trusted_source=True
        result = from_pickle(pickled, trusted_source=True)
        assert result == data


class TestBatchSerialization:
    """Test cases for batch serialization functions."""
    
    def test_serialize_batch_json(self):
        """Test batch serialization with JSON format."""
        objects = [{"key1": "value1"}, {"key2": "value2"}]
        result = serialize_batch(objects, format='json')
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == objects
    
    def test_serialize_batch_pickle(self):
        """Test batch serialization with pickle format."""
        objects = [{"key1": "value1"}, {"key2": "value2"}]
        result = serialize_batch(objects, format='pickle')
        
        assert isinstance(result, bytes)
        unpickled = pickle.loads(result)
        assert unpickled == objects
    
    def test_serialize_batch_invalid_format(self):
        """Test batch serialization with invalid format."""
        objects = [{"key": "value"}]
        
        with pytest.raises(SerializationError):
            serialize_batch(objects, format='invalid')
    
    def test_deserialize_batch_json(self):
        """Test batch deserialization with JSON format."""
        objects = [{"key1": "value1"}, {"key2": "value2"}]
        json_str = json.dumps(objects)
        result = deserialize_batch(json_str, format='json')
        
        assert result == objects
    
    def test_deserialize_batch_pickle(self):
        """Test batch deserialization with pickle format."""
        objects = [{"key1": "value1"}, {"key2": "value2"}]
        pickled = pickle.dumps(objects)
        result = deserialize_batch(pickled, format='pickle')
        
        assert result == objects
    
    def test_deserialize_batch_invalid_format(self):
        """Test batch deserialization with invalid format."""
        with pytest.raises(SerializationError):
            deserialize_batch("data", format='invalid')
    
    def test_deserialize_batch_non_list(self):
        """Test batch deserialization with non-list data."""
        json_str = '{"key": "value"}'  # Not a list
        
        with pytest.raises(SerializationError):
            deserialize_batch(json_str, format='json')


class TestSerializationRegistry:
    """Test cases for SerializationRegistry."""
    
    def test_register_serializer(self):
        """Test registering custom serializer."""
        registry = SerializationRegistry()
        
        def custom_serializer(obj):
            return {"custom": True, "value": str(obj)}
        
        registry.register_serializer(TestModel, custom_serializer)
        
        model = TestModel("test", 42)
        result = registry.serialize(model)
        
        assert result["__type__"] == "TestModel"
        assert result["data"]["custom"] is True
    
    def test_register_deserializer(self):
        """Test registering custom deserializer."""
        registry = SerializationRegistry()
        
        def custom_deserializer(data):
            return f"Custom: {data['value']}"
        
        registry.register_deserializer("CustomType", custom_deserializer)
        
        data = {"__type__": "CustomType", "data": {"value": "test"}}
        result = registry.deserialize(data)
        
        assert result == "Custom: test"
    
    def test_serialize_unregistered_type(self):
        """Test serializing unregistered type."""
        registry = SerializationRegistry()
        
        data = {"key": "value"}
        result = registry.serialize(data)
        
        assert result == data  # Should fall back to to_dict
    
    def test_deserialize_unregistered_type(self):
        """Test deserializing unregistered type."""
        registry = SerializationRegistry()
        
        data = {"key": "value"}
        result = registry.deserialize(data)
        
        assert result == data  # Should return as-is


class TestSafeSerialization:
    """Test cases for safe serialization."""
    
    def test_safe_serialize_json_success(self):
        """Test successful safe serialization with JSON."""
        data = {"key": "value"}
        result = safe_serialize(data, format='json')
        
        assert isinstance(result, str)
        assert json.loads(result) == data
    
    def test_safe_serialize_pickle_success(self):
        """Test successful safe serialization with pickle."""
        data = {"key": "value"}
        result = safe_serialize(data, format='pickle')
        
        assert isinstance(result, bytes)
        assert pickle.loads(result) == data
    
    def test_safe_serialize_fallback_to_str(self):
        """Test safe serialization fallback to string."""
        # Create an object that can't be serialized normally
        class UnserializableClass:
            def __str__(self):
                return "unserializable_object"
        
        obj = UnserializableClass()
        result = safe_serialize(obj, format='json', fallback_to_str=True)
        
        assert result == "unserializable_object"
    
    def test_safe_serialize_no_fallback(self):
        """Test safe serialization without fallback."""
        class UnserializableClass:
            def __init__(self):
                self.circular_ref = self
        
        obj = UnserializableClass()
        
        with pytest.raises(SerializationError):
            safe_serialize(obj, format='json', fallback_to_str=False)
    
    def test_safe_serialize_invalid_format(self):
        """Test safe serialization with invalid format."""
        data = {"key": "value"}
        
        with pytest.raises(SerializationError):
            safe_serialize(data, format='invalid')


if __name__ == "__main__":
    pytest.main([__file__])