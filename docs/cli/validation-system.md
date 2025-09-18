# CLI Input Validation and Sanitization System

The ESCAI CLI includes a comprehensive input validation and sanitization system designed to ensure security, data integrity, and user-friendly error handling for all command parameters.

## Overview

The validation system provides:

- **Comprehensive Input Validation**: Validates all parameter types with specific rules
- **Security-Focused Sanitization**: Prevents injection attacks and malicious input
- **Configurable Validation Levels**: From permissive to paranoid security levels
- **Helpful Error Messages**: Clear, actionable error messages with suggestions
- **CLI Integration**: Seamless integration with CLI commands via decorators
- **Multiple Parameter Types**: Support for strings, numbers, booleans, files, URLs, JSON, etc.

## Architecture

### Core Components

```
escai_framework/cli/utils/
├── validators.py              # Main validation logic
├── validation_config.py       # Configuration management
└── validation_integration.py  # CLI integration utilities
```

### Key Classes

- **`CLIInputValidator`**: Main validator with comprehensive parameter validation
- **`ValidationConfigManager`**: Manages validation rules and security policies
- **`ValidationIntegration`**: Integrates validation with CLI commands

## Parameter Types

### Supported Types

| Type             | Description                                 | Example                     |
| ---------------- | ------------------------------------------- | --------------------------- |
| `string`         | Text validation with pattern matching       | `"my-agent-1"`              |
| `integer`        | Whole number validation with range checks   | `8080`                      |
| `float`          | Decimal number validation with range checks | `0.95`                      |
| `boolean`        | Boolean value with flexible input formats   | `true`, `yes`, `1`          |
| `file_path`      | File path validation with existence checks  | `"/path/to/file.txt"`       |
| `directory_path` | Directory path validation                   | `"/path/to/dir"`            |
| `url`            | URL format and security validation          | `"https://api.example.com"` |
| `email`          | Email address format validation             | `"user@example.com"`        |
| `json`           | JSON string parsing and validation          | `'{"key": "value"}'`        |
| `yaml`           | YAML string parsing and validation          | `"key: value"`              |
| `regex`          | Regular expression pattern validation       | `"^[a-zA-Z0-9]+$"`          |

## Security Features

### Sanitization Levels

The system supports four security levels:

1. **PERMISSIVE**: Allows most content with basic sanitization
2. **STANDARD**: Balanced security with reasonable restrictions
3. **STRICT**: High security with aggressive sanitization
4. **PARANOID**: Maximum security, removes most special characters

### Security Protections

- **XSS Prevention**: HTML encoding and tag removal
- **SQL Injection**: Pattern detection and sanitization
- **Command Injection**: Dangerous character removal
- **Path Traversal**: Directory traversal prevention
- **LDAP Injection**: Special character escaping

## Usage Examples

### Basic Validation

```python
from escai_framework.cli.utils.validators import CLIInputValidator

validator = CLIInputValidator()

# Validate a parameter
result = validator.validate_command_parameter('agent_id', 'my-agent-1', 'string')

if result.is_valid:
    print(f"Valid: {result.sanitized_value}")
else:
    print(f"Errors: {result.errors}")
    print(f"Suggestions: {result.suggestions}")
```

### Command Validation Decorator

```python
from escai_framework.cli.utils.validation_integration import validate_command

@validate_command('monitor')
def monitor_command(agent_id=None, framework=None, interval=None):
    """Monitor command with automatic validation"""
    # Parameters are automatically validated and sanitized
    return f"Monitoring {agent_id} with {framework}"
```

### Convenience Functions

```python
from escai_framework.cli.utils.validators import (
    validate_agent_id,
    validate_file_path,
    validate_url,
    validate_json_string
)

# Quick validations
agent_result = validate_agent_id('my-agent')
file_result = validate_file_path('/path/to/file.txt')
url_result = validate_url('https://api.example.com')
json_result = validate_json_string('{"key": "value"}')
```

## Configuration

### Command Configuration

Commands can be configured with specific validation rules:

```yaml
# .escai/validation_config.yaml
commands:
  monitor:
    description: "Monitor agent execution"
    security_level: 2 # STANDARD
    parameters:
      - name: agent_id
        type: string
        required: true
        pattern: "^[a-zA-Z0-9_-]{1,50}$"
        error_message: "Agent ID must be alphanumeric with underscores/hyphens"
        help_text: "Unique identifier for the agent"

      - name: interval
        type: integer
        required: false
        min_value: 1
        max_value: 3600
        error_message: "Interval must be between 1 and 3600 seconds"
```

### Global Security Settings

```yaml
security_settings:
  block_path_traversal: true
  block_command_injection: true
  block_sql_injection: true
  block_xss: true
  require_https_urls: false
  max_file_size: 10485760 # 10MB
```

## Error Handling

### Validation Results

All validation operations return a `CLIValidationResult` object:

```python
@dataclass
class CLIValidationResult:
    is_valid: bool              # Whether validation passed
    errors: List[str]           # List of error messages
    warnings: List[str]         # List of warnings
    sanitized_value: Any        # Sanitized/converted value
    suggestions: List[str]      # Helpful suggestions
    help_text: Optional[str]    # Parameter help text
```

### Error Messages

The system provides helpful, actionable error messages:

```
✗ agent_id: Parameter 'agent_id' has invalid format
  Suggestions:
    • Use only letters, numbers, underscores, and hyphens
    • Example: 'my-agent-1' or 'agent_001'

✗ port: Port number must be between 1 and 65535
  Suggestions:
    • Use a valid port number like 8080 or 3000

✗ confidence: Confidence must be between 0.0 and 1.0
  Suggestions:
    • Use a decimal value like 0.5 or 0.95
```

## Integration with CLI Commands

### Click Integration

```python
import click
from escai_framework.cli.utils.validation_integration import (
    create_click_option_with_validation,
    validation_error_handler
)

@click.command()
@create_click_option_with_validation('monitor', 'agent_id')
@create_click_option_with_validation('monitor', 'framework')
@validation_error_handler
def monitor(agent_id, framework):
    """Monitor command with integrated validation"""
    # Parameters are automatically validated
    pass
```

### Manual Validation

```python
from escai_framework.cli.utils.validation_integration import validate_parameter

def my_command(agent_id, port):
    # Manually validate parameters
    agent_id = validate_parameter('my_command', 'agent_id', agent_id)
    port = validate_parameter('my_command', 'port', port)

    # Use validated parameters
    return f"Command with {agent_id} on port {port}"
```

## Testing

### Unit Tests

The validation system includes comprehensive unit tests:

```bash
# Run validation tests
python -m pytest tests/unit/test_cli_validators.py -v

# Run integration tests
python -m pytest tests/integration/test_cli_validation_integration.py -v
```

### Test Coverage

- Parameter type validation
- Security sanitization
- Error message quality
- Configuration management
- CLI integration
- Performance testing

## Performance

### Benchmarks

- **Single Parameter**: < 1ms validation time
- **Batch Validation**: < 1s for 1000 parameters
- **Memory Usage**: < 10MB for typical operations
- **Startup Time**: < 100ms initialization

### Optimization Features

- Validation result caching
- Lazy pattern compilation
- Efficient sanitization algorithms
- Minimal memory footprint

## Security Considerations

### Input Sanitization

All user inputs are sanitized to prevent:

- Cross-site scripting (XSS)
- SQL injection attacks
- Command injection
- Path traversal attacks
- LDAP injection

### File System Security

- Path validation and sanitization
- Permission checking
- Safe directory restrictions
- File extension validation

### Network Security

- URL format validation
- Protocol restrictions (HTTP/HTTPS)
- Hostname validation
- Port range validation

## Best Practices

### For CLI Command Developers

1. **Use Type-Specific Validation**: Choose appropriate parameter types
2. **Provide Clear Error Messages**: Include helpful suggestions
3. **Configure Security Levels**: Match security needs to use case
4. **Test Edge Cases**: Validate with malicious and edge case inputs
5. **Document Parameters**: Provide clear help text and examples

### For Users

1. **Follow Format Guidelines**: Use suggested formats for parameters
2. **Check Error Messages**: Read suggestions for fixing validation errors
3. **Use Absolute Paths**: Prefer absolute paths for file parameters
4. **Validate JSON/YAML**: Use online validators for complex data structures

## Troubleshooting

### Common Issues

**Q: Parameter validation fails with "invalid format"**
A: Check the parameter pattern requirements and use suggested formats.

**Q: File path validation fails on Windows**
A: Use forward slashes or double backslashes in paths.

**Q: URL validation rejects localhost URLs**
A: This is expected for security; use IP addresses or configure security settings.

**Q: JSON validation fails with valid JSON**
A: Check for special characters that may be sanitized; use simpler JSON structures.

### Debug Mode

Enable debug logging to see detailed validation information:

```python
import logging
logging.getLogger('escai_framework.cli.utils.validators').setLevel(logging.DEBUG)
```

## API Reference

### Main Classes

#### CLIInputValidator

```python
class CLIInputValidator:
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD)
    def validate_command_parameter(self, param_name: str, value: Any, param_type: str) -> CLIValidationResult
    def validate_command_args(self, command_name: str, args: Dict[str, Any], arg_specs: Dict[str, Dict[str, Any]]) -> Dict[str, CLIValidationResult]
    def create_validation_decorator(self, arg_specs: Dict[str, Dict[str, Any]]) -> Callable
```

#### ValidationConfigManager

```python
class ValidationConfigManager:
    def __init__(self, config_path: Optional[Path] = None)
    def get_command_config(self, command_name: str) -> Optional[CommandValidationConfig]
    def get_parameter_config(self, command_name: str, parameter_name: str) -> Optional[ParameterValidationRule]
    def save_configuration(self, config_path: Optional[Path] = None)
```

### Convenience Functions

```python
def validate_agent_id(agent_id: str) -> CLIValidationResult
def validate_file_path(file_path: str, must_exist: bool = True) -> CLIValidationResult
def validate_url(url: str) -> CLIValidationResult
def validate_json_string(json_str: str) -> CLIValidationResult
def sanitize_user_input(user_input: str, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> str
```

## Examples

See `examples/cli_validation_example.py` for comprehensive usage examples demonstrating all features of the validation system.

## Contributing

When adding new validation features:

1. Add unit tests for new validation types
2. Update configuration schema if needed
3. Document new parameter types and options
4. Test security implications thoroughly
5. Update this documentation
