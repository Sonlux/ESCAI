# Contributing to ESCAI Framework

We welcome contributions to the ESCAI Framework! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ESCAI.git
cd ESCAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Verify Installation

```bash
# Run basic functionality test
python test_basic_functionality.py

# Run unit tests (if pytest is available)
pytest tests/unit/
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-new-instrumentor`
- `bugfix/fix-serialization-issue`
- `docs/update-api-documentation`
- `refactor/improve-validation-performance`

### Commit Messages

Write clear, descriptive commit messages:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when applicable

Example:

```
Add support for CrewAI instrumentation

- Implement CrewAIInstrumentor class
- Add unit tests for CrewAI integration
- Update documentation with usage examples

Fixes #123
```

## Testing

### Running Tests

```bash
# Run basic functionality tests
python test_basic_functionality.py

# Run unit tests (if pytest is available)
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_epistemic_state.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Follow existing test patterns and naming conventions
- Include both positive and negative test cases
- Test edge cases and error conditions
- Ensure tests are independent and can run in any order

### Test Structure

```python
class TestNewFeature:
    """Test cases for NewFeature."""

    def test_feature_creation(self):
        """Test creating a valid NewFeature."""
        # Test implementation

    def test_feature_validation_valid(self):
        """Test validation of valid NewFeature."""
        # Test implementation

    def test_feature_validation_invalid(self):
        """Test validation with invalid data."""
        # Test implementation

    def test_feature_serialization(self):
        """Test NewFeature serialization."""
        # Test implementation
```

## Code Style

### Python Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all classes and public methods
- Use meaningful variable and function names
- Keep functions focused and reasonably sized

### Code Formatting

We use the following tools for code formatting:

```bash
# Format code with black
black escai_framework tests

# Sort imports with isort
isort escai_framework tests

# Lint code with flake8
flake8 escai_framework tests

# Type checking with mypy
mypy escai_framework
```

### Documentation Style

- Use Google-style docstrings
- Include parameter types and descriptions
- Include return type and description
- Include usage examples for complex functions
- Document exceptions that may be raised

Example:

```python
def validate_string(value: Any, field_name: str, min_length: int = 1) -> str:
    """
    Validate string values with optional constraints.

    Args:
        value: The value to validate
        field_name: Name of the field being validated
        min_length: Minimum string length (default: 1)

    Returns:
        The validated string value

    Raises:
        ValidationError: If validation fails

    Example:
        >>> validate_string("hello", "greeting", min_length=3)
        'hello'
    """
```

## Submitting Changes

### Pull Request Process

1. **Update Documentation**: Ensure any new features are documented
2. **Add Tests**: Include tests for new functionality
3. **Update CHANGELOG**: Add entry describing your changes
4. **Check CI**: Ensure all automated checks pass
5. **Request Review**: Request review from maintainers

### Pull Request Template

When creating a pull request, include:

- **Description**: Clear description of what the PR does
- **Motivation**: Why is this change needed?
- **Testing**: How was this tested?
- **Checklist**: Use the PR checklist template

### Review Process

- All PRs require at least one review from a maintainer
- Address all review comments before merging
- Maintain a clean commit history (squash if necessary)
- Ensure CI checks pass

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- **Environment**: Python version, OS, package versions
- **Steps to Reproduce**: Minimal example that reproduces the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Error Messages**: Full error messages and stack traces

### Feature Requests

When requesting features, include:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: What alternatives have you considered?
- **Additional Context**: Any other relevant information

### Issue Labels

We use labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## Development Guidelines

### Architecture Principles

- **Modularity**: Keep components loosely coupled
- **Extensibility**: Design for easy extension and customization
- **Performance**: Consider performance implications of changes
- **Reliability**: Prioritize correctness and robustness
- **Usability**: Make the API intuitive and well-documented

### Adding New Features

1. **Design**: Consider the design and how it fits with existing architecture
2. **Interface**: Design clean, consistent APIs
3. **Implementation**: Implement with proper error handling and validation
4. **Testing**: Add comprehensive tests
5. **Documentation**: Document the feature thoroughly
6. **Examples**: Provide usage examples

### Performance Considerations

- Profile performance-critical code
- Use appropriate data structures and algorithms
- Consider memory usage for large datasets
- Benchmark changes that might affect performance
- Document performance characteristics

## Getting Help

- **Documentation**: Check the README and documentation first
- **Issues**: Search existing issues for similar problems
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Contact**: Reach out to maintainers for guidance

## Recognition

Contributors will be recognized in:

- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes for major contributions

Thank you for contributing to the ESCAI Framework!
