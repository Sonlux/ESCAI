"""
Unit tests for validation utilities.
"""

import pytest
from datetime import datetime, timedelta
from enum import Enum

from escai_framework.utils.validation import (
    ValidationError, validate_string, validate_number, validate_probability,
    validate_datetime, validate_list, validate_dict, validate_enum,
    validate_id, ValidationContext, validate_model_instance
)


class TestEnum(Enum):
    """Test enum for validation tests."""
    VALUE1 = "value1"
    VALUE2 = "value2"


class TestModel:
    """Test model class for validation tests."""
    
    def __init__(self, value: str):
        self.value = value
    
    def validate(self) -> bool:
        return isinstance(self.value, str) and len(self.value) > 0


class TestValidationError:
    """Test cases for ValidationError."""
    
    def test_validation_error_creation(self):
        """Test creating ValidationError."""
        error = ValidationError("Test error", "test_field", "test_value")
        assert error.message == "Test error"
        assert error.field == "test_field"
        assert error.value == "test_value"
    
    def test_validation_error_format_message(self):
        """Test error message formatting."""
        error = ValidationError("Test error", "test_field", "test_value")
        formatted = error.format_message()
        assert "test_field" in formatted
        assert "Test error" in formatted
        
        error_no_field = ValidationError("Test error")
        formatted_no_field = error_no_field.format_message()
        assert formatted_no_field == "Validation error: Test error"


class TestStringValidation:
    """Test cases for string validation."""
    
    def test_validate_string_valid(self):
        """Test valid string validation."""
        result = validate_string("valid string", "test_field")
        assert result == "valid string"
    
    def test_validate_string_invalid_type(self):
        """Test string validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_string(123, "test_field")
        assert "Expected string" in str(exc_info.value)
    
    def test_validate_string_empty(self):
        """Test string validation with empty string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_string("", "test_field")
        assert "cannot be empty" in str(exc_info.value)
        
        # Test with allow_empty=True
        result = validate_string("", "test_field", allow_empty=True)
        assert result == ""
    
    def test_validate_string_length_constraints(self):
        """Test string validation with length constraints."""
        # Test minimum length
        with pytest.raises(ValidationError) as exc_info:
            validate_string("ab", "test_field", min_length=3)
        assert "at least 3" in str(exc_info.value)
        
        # Test maximum length
        with pytest.raises(ValidationError) as exc_info:
            validate_string("abcdef", "test_field", max_length=5)
        assert "not exceed 5" in str(exc_info.value)
        
        # Test valid length
        result = validate_string("abc", "test_field", min_length=2, max_length=5)
        assert result == "abc"
    
    def test_validate_string_pattern(self):
        """Test string validation with pattern matching."""
        # Test valid pattern
        result = validate_string("abc123", "test_field", pattern=r'^[a-z]+\d+$')
        assert result == "abc123"
        
        # Test invalid pattern
        with pytest.raises(ValidationError) as exc_info:
            validate_string("ABC123", "test_field", pattern=r'^[a-z]+\d+$')
        assert "does not match required pattern" in str(exc_info.value)


class TestNumberValidation:
    """Test cases for number validation."""
    
    def test_validate_number_valid(self):
        """Test valid number validation."""
        result_int = validate_number(42, "test_field")
        assert result_int == 42
        
        result_float = validate_number(3.14, "test_field")
        assert result_float == 3.14
    
    def test_validate_number_invalid_type(self):
        """Test number validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_number("not a number", "test_field")
        assert "Expected numeric value" in str(exc_info.value)
    
    def test_validate_number_type_restrictions(self):
        """Test number validation with type restrictions."""
        # Test disallow int
        with pytest.raises(ValidationError) as exc_info:
            validate_number(42, "test_field", allow_int=False)
        assert "Integer values not allowed" in str(exc_info.value)
        
        # Test disallow float
        with pytest.raises(ValidationError) as exc_info:
            validate_number(3.14, "test_field", allow_float=False)
        assert "Float values not allowed" in str(exc_info.value)
    
    def test_validate_number_range_constraints(self):
        """Test number validation with range constraints."""
        # Test minimum value
        with pytest.raises(ValidationError) as exc_info:
            validate_number(5, "test_field", min_value=10)
        assert "at least 10" in str(exc_info.value)
        
        # Test maximum value
        with pytest.raises(ValidationError) as exc_info:
            validate_number(15, "test_field", max_value=10)
        assert "not exceed 10" in str(exc_info.value)
        
        # Test valid range
        result = validate_number(7, "test_field", min_value=5, max_value=10)
        assert result == 7


class TestProbabilityValidation:
    """Test cases for probability validation."""
    
    def test_validate_probability_valid(self):
        """Test valid probability validation."""
        result = validate_probability(0.5, "test_field")
        assert result == 0.5
        
        # Test boundary values
        assert validate_probability(0.0, "test_field") == 0.0
        assert validate_probability(1.0, "test_field") == 1.0
    
    def test_validate_probability_invalid(self):
        """Test invalid probability validation."""
        with pytest.raises(ValidationError) as exc_info:
            validate_probability(-0.1, "test_field")
        assert "at least 0.0" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_probability(1.1, "test_field")
        assert "not exceed 1.0" in str(exc_info.value)


class TestDatetimeValidation:
    """Test cases for datetime validation."""
    
    def test_validate_datetime_valid(self):
        """Test valid datetime validation."""
        now = datetime.utcnow()
        result = validate_datetime(now, "test_field")
        assert result == now
    
    def test_validate_datetime_invalid_type(self):
        """Test datetime validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_datetime("not a datetime", "test_field")
        assert "Expected datetime" in str(exc_info.value)
    
    def test_validate_datetime_time_constraints(self):
        """Test datetime validation with time constraints."""
        past = datetime.utcnow() - timedelta(hours=1)
        future = datetime.utcnow() + timedelta(hours=1)
        
        # Test disallow future
        with pytest.raises(ValidationError) as exc_info:
            validate_datetime(future, "test_field", allow_future=False)
        assert "Future dates not allowed" in str(exc_info.value)
        
        # Test disallow past
        with pytest.raises(ValidationError) as exc_info:
            validate_datetime(past, "test_field", allow_past=False)
        assert "Past dates not allowed" in str(exc_info.value)


class TestListValidation:
    """Test cases for list validation."""
    
    def test_validate_list_valid(self):
        """Test valid list validation."""
        result = validate_list([1, 2, 3], "test_field")
        assert result == [1, 2, 3]
    
    def test_validate_list_invalid_type(self):
        """Test list validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_list("not a list", "test_field")
        assert "Expected list" in str(exc_info.value)
    
    def test_validate_list_length_constraints(self):
        """Test list validation with length constraints."""
        # Test minimum length
        with pytest.raises(ValidationError) as exc_info:
            validate_list([1], "test_field", min_length=2)
        assert "at least 2" in str(exc_info.value)
        
        # Test maximum length
        with pytest.raises(ValidationError) as exc_info:
            validate_list([1, 2, 3], "test_field", max_length=2)
        assert "not exceed 2" in str(exc_info.value)
    
    def test_validate_list_item_validation(self):
        """Test list validation with item validator."""
        def string_validator(item, field):
            if not isinstance(item, str):
                raise ValidationError("Expected string item")
            return item
        
        # Test valid items
        result = validate_list(["a", "b", "c"], "test_field", item_validator=string_validator)
        assert result == ["a", "b", "c"]
        
        # Test invalid item
        with pytest.raises(ValidationError) as exc_info:
            validate_list(["a", 123, "c"], "test_field", item_validator=string_validator)
        assert "Item 1 validation failed" in str(exc_info.value)


class TestDictValidation:
    """Test cases for dict validation."""
    
    def test_validate_dict_valid(self):
        """Test valid dict validation."""
        result = validate_dict({"key": "value"}, "test_field")
        assert result == {"key": "value"}
    
    def test_validate_dict_invalid_type(self):
        """Test dict validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_dict("not a dict", "test_field")
        assert "Expected dict" in str(exc_info.value)
    
    def test_validate_dict_required_keys(self):
        """Test dict validation with required keys."""
        # Test missing required key
        with pytest.raises(ValidationError) as exc_info:
            validate_dict({"key1": "value1"}, "test_field", required_keys=["key1", "key2"])
        assert "Missing required keys" in str(exc_info.value)
        
        # Test valid required keys
        result = validate_dict(
            {"key1": "value1", "key2": "value2"}, 
            "test_field", 
            required_keys=["key1", "key2"]
        )
        assert result == {"key1": "value1", "key2": "value2"}
    
    def test_validate_dict_optional_keys(self):
        """Test dict validation with optional keys."""
        # Test unexpected key
        with pytest.raises(ValidationError) as exc_info:
            validate_dict(
                {"key1": "value1", "unexpected": "value"}, 
                "test_field", 
                required_keys=["key1"],
                optional_keys=[]
            )
        assert "Unexpected keys" in str(exc_info.value)


class TestEnumValidation:
    """Test cases for enum validation."""
    
    def test_validate_enum_valid(self):
        """Test valid enum validation."""
        result = validate_enum(TestEnum.VALUE1, "test_field", TestEnum)
        assert result == TestEnum.VALUE1
    
    def test_validate_enum_invalid(self):
        """Test invalid enum validation."""
        with pytest.raises(ValidationError) as exc_info:
            validate_enum("invalid_value", "test_field", TestEnum)
        assert "Expected TestEnum" in str(exc_info.value)
        assert "Valid values" in str(exc_info.value)


class TestIdValidation:
    """Test cases for ID validation."""
    
    def test_validate_id_valid(self):
        """Test valid ID validation."""
        result = validate_id("valid_id_123", "test_field")
        assert result == "valid_id_123"
    
    def test_validate_id_invalid(self):
        """Test invalid ID validation."""
        with pytest.raises(ValidationError) as exc_info:
            validate_id("invalid id with spaces", "test_field")
        assert "does not match required pattern" in str(exc_info.value)


class TestValidationContext:
    """Test cases for ValidationContext."""
    
    def test_validation_context_no_errors(self):
        """Test validation context with no errors."""
        with ValidationContext() as ctx:
            ctx.validate(validate_string, "valid", "field")
            ctx.validate(validate_number, 42, "field")
        # Should not raise exception
    
    def test_validation_context_with_errors(self):
        """Test validation context with errors."""
        with pytest.raises(ValidationError) as exc_info:
            with ValidationContext() as ctx:
                ctx.validate(validate_string, "", "field1")  # Invalid
                ctx.validate(validate_number, "not_number", "field2")  # Invalid
        
        assert "Multiple validation errors" in str(exc_info.value)
    
    def test_validation_context_has_errors(self):
        """Test checking for errors in validation context."""
        ctx = ValidationContext()
        assert ctx.has_errors() is False
        
        ctx.validate(validate_string, "", "field")  # Invalid
        assert ctx.has_errors() is True


class TestModelInstanceValidation:
    """Test cases for model instance validation."""
    
    def test_validate_model_instance_valid(self):
        """Test valid model instance validation."""
        model = TestModel("valid_value")
        result = validate_model_instance(model, TestModel)
        assert result == model
    
    def test_validate_model_instance_wrong_type(self):
        """Test model instance validation with wrong type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_instance("not_a_model", TestModel)
        assert "Expected TestModel" in str(exc_info.value)
    
    def test_validate_model_instance_invalid_validation(self):
        """Test model instance validation with failed validation."""
        invalid_model = TestModel("")  # Empty value should fail validation
        with pytest.raises(ValidationError) as exc_info:
            validate_model_instance(invalid_model, TestModel)
        assert "Instance validation failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])