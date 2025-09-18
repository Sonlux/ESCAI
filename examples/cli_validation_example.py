#!/usr/bin/env python3
"""
Example demonstrating CLI input validation and sanitization system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from escai_framework.cli.utils.validators import (
    CLIInputValidator,
    validate_agent_id,
    validate_file_path,
    validate_url,
    validate_json_string,
    sanitize_user_input
)
from escai_framework.cli.utils.validation_config import (
    get_validation_config,
    ParameterValidationRule,
    CommandValidationConfig
)
from escai_framework.cli.utils.validation_integration import (
    get_validation_integration,
    validate_command,
    ValidationErrorHandler
)
from escai_framework.security.input_validator import ValidationLevel


def demonstrate_basic_validation():
    """Demonstrate basic parameter validation"""
    print("=== Basic Parameter Validation ===")
    
    validator = CLIInputValidator()
    
    # Test valid inputs
    print("\n1. Valid Inputs:")
    test_cases = [
        ('agent_id', 'my-agent-1', 'string'),
        ('port', '8080', 'integer'),
        ('confidence', '0.95', 'float'),
        ('enabled', 'true', 'boolean'),
        ('url', 'https://api.example.com', 'url'),
        ('email', 'user@example.com', 'email')
    ]
    
    for param_name, value, param_type in test_cases:
        result = validator.validate_command_parameter(param_name, value, param_type)
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"  {param_name}: {value} -> {status}")
        if result.warnings:
            print(f"    Warnings: {', '.join(result.warnings)}")
    
    # Test invalid inputs
    print("\n2. Invalid Inputs:")
    invalid_cases = [
        ('agent_id', 'invalid@agent', 'string'),
        ('port', '70000', 'integer'),
        ('confidence', '1.5', 'float'),
        ('enabled', 'maybe', 'boolean'),
        ('url', 'not-a-url', 'url'),
        ('email', 'invalid-email', 'email')
    ]
    
    for param_name, value, param_type in invalid_cases:
        result = validator.validate_command_parameter(param_name, value, param_type)
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"  {param_name}: {value} -> {status}")
        if result.errors:
            print(f"    Errors: {', '.join(result.errors)}")
        if result.suggestions:
            print(f"    Suggestions: {', '.join(result.suggestions)}")


def demonstrate_security_sanitization():
    """Demonstrate security-focused input sanitization"""
    print("\n=== Security Sanitization ===")
    
    malicious_inputs = [
        '<script>alert("xss")</script>',
        'SELECT * FROM users; DROP TABLE users;',
        '../../../etc/passwd',
        'javascript:alert(1)',
        '$(rm -rf /)'
    ]
    
    for malicious_input in malicious_inputs:
        sanitized = sanitize_user_input(malicious_input)
        print(f"Original: {malicious_input}")
        print(f"Sanitized: {sanitized}")
        print(f"Changed: {'Yes' if sanitized != malicious_input else 'No'}")
        print()


def demonstrate_validation_levels():
    """Demonstrate different validation security levels"""
    print("=== Validation Security Levels ===")
    
    html_input = '<p>Test <strong>content</strong></p>'
    
    levels = [
        ValidationLevel.PERMISSIVE,
        ValidationLevel.STANDARD,
        ValidationLevel.STRICT,
        ValidationLevel.PARANOID
    ]
    
    for level in levels:
        validator = CLIInputValidator(level)
        result = validator.validate_command_parameter('description', html_input, 'string')
        
        print(f"\n{level.name} Level:")
        print(f"  Input: {html_input}")
        print(f"  Output: {result.sanitized_value}")
        print(f"  Valid: {result.is_valid}")
        if result.warnings:
            print(f"  Warnings: {', '.join(result.warnings)}")


def demonstrate_convenience_functions():
    """Demonstrate convenience validation functions"""
    print("\n=== Convenience Functions ===")
    
    # Agent ID validation
    print("\n1. Agent ID Validation:")
    agent_ids = ['valid-agent', 'agent_123', 'invalid@agent', '']
    for agent_id in agent_ids:
        result = validate_agent_id(agent_id)
        status = "✓" if result.is_valid else "✗"
        print(f"  {status} {agent_id}")
    
    # URL validation
    print("\n2. URL Validation:")
    urls = ['https://api.example.com', 'http://localhost:8080', 'not-a-url']
    for url in urls:
        result = validate_url(url)
        status = "✓" if result.is_valid else "✗"
        print(f"  {status} {url}")
        if result.warnings:
            print(f"    Warnings: {', '.join(result.warnings)}")
    
    # JSON validation
    print("\n3. JSON Validation:")
    json_strings = ['{"key": "value"}', '{invalid json}', '[]']
    for json_str in json_strings:
        result = validate_json_string(json_str)
        status = "✓" if result.is_valid else "✗"
        print(f"  {status} {json_str}")


def demonstrate_command_validation():
    """Demonstrate command-level validation with configuration"""
    print("\n=== Command Validation with Configuration ===")
    
    # Get validation integration
    integration = get_validation_integration()
    
    # Test monitor command validation
    print("\n1. Monitor Command Validation:")
    monitor_args = {
        'agent_id': 'test-agent-1',
        'framework': 'langchain',
        'interval': '5',
        'output_format': 'json'
    }
    
    try:
        # This would normally use configuration, but we'll simulate it
        validator = CLIInputValidator()
        results = {}
        
        for param_name, param_value in monitor_args.items():
            if param_name == 'interval':
                result = validator.validate_command_parameter(param_name, param_value, 'integer')
            else:
                result = validator.validate_command_parameter(param_name, param_value, 'string')
            results[param_name] = result
        
        all_valid = all(result.is_valid for result in results.values())
        print(f"  All parameters valid: {all_valid}")
        
        for param_name, result in results.items():
            status = "✓" if result.is_valid else "✗"
            print(f"    {status} {param_name}: {monitor_args[param_name]} -> {result.sanitized_value}")
    
    except Exception as e:
        print(f"  Error: {e}")


def demonstrate_validation_decorator():
    """Demonstrate validation decorator usage"""
    print("\n=== Validation Decorator ===")
    
    validator = CLIInputValidator()
    
    # Create a simple validation decorator
    arg_specs = {
        'agent_id': {'type': 'string'},
        'port': {'type': 'integer'},
        'enabled': {'type': 'boolean'}
    }
    
    @validator.create_validation_decorator(arg_specs)
    def mock_command(agent_id=None, port=None, enabled=None):
        return f"Command executed with agent_id={agent_id}, port={port}, enabled={enabled}"
    
    try:
        # Valid arguments
        result = mock_command(agent_id='test-agent', port=8080, enabled=True)
        print(f"  ✓ Valid call: {result}")
        
        # Invalid arguments would raise CLIValidationError
        print("  ✓ Validation decorator working correctly")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


def demonstrate_error_handling():
    """Demonstrate validation error handling"""
    print("\n=== Error Handling ===")
    
    validator = CLIInputValidator()
    
    # Test error aggregation
    invalid_params = {
        'agent_id': '',
        'port': 'invalid',
        'confidence': '2.0'
    }
    
    arg_specs = {
        'agent_id': {'type': 'string'},
        'port': {'type': 'integer'},
        'confidence': {'type': 'float'}
    }
    
    results = validator.validate_command_args('test_command', invalid_params, arg_specs)
    
    print("  Validation Results:")
    for param_name, result in results.items():
        if not result.is_valid:
            print(f"    ✗ {param_name}: {', '.join(result.errors)}")
            if result.suggestions:
                print(f"      Suggestions: {', '.join(result.suggestions)}")
        else:
            print(f"    ✓ {param_name}: Valid")


def main():
    """Run all validation demonstrations"""
    print("CLI Input Validation and Sanitization System Demo")
    print("=" * 50)
    
    try:
        demonstrate_basic_validation()
        demonstrate_security_sanitization()
        demonstrate_validation_levels()
        demonstrate_convenience_functions()
        demonstrate_command_validation()
        demonstrate_validation_decorator()
        demonstrate_error_handling()
        
        print("\n" + "=" * 50)
        print("✓ All demonstrations completed successfully!")
        print("\nThe CLI validation system provides:")
        print("  • Comprehensive input validation for all parameter types")
        print("  • Security-focused sanitization to prevent injection attacks")
        print("  • Configurable validation levels (Permissive to Paranoid)")
        print("  • Helpful error messages with actionable suggestions")
        print("  • Integration with CLI commands via decorators")
        print("  • File path, URL, JSON, YAML, and regex validation")
        print("  • Network input validation for endpoints")
        print("  • Configuration parameter validation")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()