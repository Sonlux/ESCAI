# ESCAI Framework Testing Guide

This guide explains how to run tests for the ESCAI framework both locally and in CI/CD environments.

## Quick Start

### Prerequisites

1. **Python 3.11+** installed
2. **Git** for cloning the repository
3. **Docker** (optional, for database services)

### Local Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourorg/ESCAI.git
   cd ESCAI
   ```

2. **Install the package in development mode:**

   ```bash
   # Install core dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

   # Install ESCAI in editable mode with test dependencies
   pip install -e ".[test]"
   ```

3. **Verify installation:**
   ```bash
   python scripts/validate_imports.py
   ```

## Running Tests

### Using the Test Runner Script (Recommended)

```bash
# Run all tests with coverage
python scripts/run_tests.py --coverage --verbose

# Run only unit tests
python scripts/run_tests.py --unit --verbose

# Run only integration tests
python scripts/run_tests.py --integration --verbose

# Run tests quickly (skip slow tests)
python scripts/run_tests.py --fast --verbose
```

### Using pytest directly

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_epistemic_state.py -v

# Run with coverage
pytest tests/unit/ -v --cov=escai_framework --cov-report=html

# Run integration tests (requires external services)
pytest tests/integration/ -v
```

## Test Categories

### Unit Tests (`tests/unit/`)

- **Fast execution** (< 1 second per test)
- **No external dependencies** (databases, APIs, etc.)
- **Mock external services**
- **Test individual components in isolation**

Examples:

- `test_epistemic_state.py` - Model validation and serialization
- `test_behavioral_pattern.py` - Pattern analysis logic
- `test_causal_relationship.py` - Causal inference algorithms

### Integration Tests (`tests/integration/`)

- **Test component interactions**
- **May require external services** (PostgreSQL, Redis, MongoDB)
- **Slower execution** (1-10 seconds per test)
- **End-to-end workflows**

Examples:

- `test_database_integration.py` - Database operations
- `test_api_endpoints.py` - API functionality
- `test_framework_instrumentors.py` - Agent framework integrations

### Performance Tests (`tests/performance/`)

- **Benchmark critical operations**
- **Monitor resource usage**
- **Validate performance requirements**

### Load Tests (`tests/load/`)

- **Test system under load**
- **Concurrent user simulation**
- **Scalability validation**

## CI/CD Integration

### GitHub Actions

The project uses GitHub Actions for automated testing. The workflow:

1. **Sets up Python 3.11**
2. **Installs dependencies and ESCAI package**
3. **Validates imports**
4. **Runs linting, type checking, and security scans**
5. **Executes unit and integration tests**
6. **Generates coverage reports**

### Key CI/CD Features

- ✅ **Automatic package installation** with `pip install -e ".[test]"`
- ✅ **Import validation** before running tests
- ✅ **Comprehensive test coverage** reporting
- ✅ **Multiple Python version support** (focused on 3.11)
- ✅ **Database services** (PostgreSQL, Redis, MongoDB) for integration tests

## Troubleshooting

### Import Errors

If you see `ImportError: Could not import ESCAI models`, ensure:

1. **Package is installed in development mode:**

   ```bash
   pip install -e .
   ```

2. **You're in the correct directory:**

   ```bash
   cd /path/to/ESCAI
   python -c "import escai_framework; print('✅ Success')"
   ```

3. **Python path is correct:**
   ```bash
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

### Test Failures

1. **Check test dependencies:**

   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Validate imports first:**

   ```bash
   python scripts/validate_imports.py
   ```

3. **Run tests with verbose output:**
   ```bash
   pytest tests/unit/ -v --tb=long
   ```

### Database Connection Issues

For integration tests requiring databases:

1. **Start services with Docker:**

   ```bash
   docker-compose up -d postgres redis mongodb
   ```

2. **Set environment variables:**
   ```bash
   export DATABASE_URL="postgresql://root:postgres@localhost:5432/test_db"
   export REDIS_URL="redis://localhost:6379/0"
   export MONGODB_URL="mongodb://admin:password@localhost:27017/test_db?authSource=admin"
   ```

## Test Configuration

### pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --strict-markers --strict-config"
asyncio_mode = "auto"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "performance: Performance tests",
    "load: Load tests",
    "slow: Slow running tests",
]
```

### Test Markers

Use markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_fast_unit_test():
    pass

@pytest.mark.integration
def test_database_integration():
    pass

@pytest.mark.slow
def test_expensive_operation():
    pass
```

Run specific markers:

```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

## Coverage Requirements

- **Minimum coverage:** 80%
- **Critical components:** 90%+
- **New code:** 100% coverage required

Generate coverage reports:

```bash
pytest --cov=escai_framework --cov-report=html
open htmlcov/index.html  # View in browser
```

## Best Practices

### Writing Tests

1. **Use descriptive test names:**

   ```python
   def test_epistemic_state_validates_confidence_range():
       pass
   ```

2. **Follow AAA pattern:**

   ```python
   def test_example():
       # Arrange
       data = create_test_data()

       # Act
       result = process_data(data)

       # Assert
       assert result.is_valid()
   ```

3. **Use fixtures for common setup:**

   ```python
   @pytest.fixture
   def sample_agent():
       return Agent(id="test_agent", name="Test Agent")
   ```

4. **Mock external dependencies:**
   ```python
   @patch('escai_framework.external_service')
   def test_with_mock(mock_service):
       mock_service.return_value = "mocked_response"
       # Test logic here
   ```

### Performance Considerations

- **Keep unit tests fast** (< 1 second each)
- **Use `@pytest.mark.slow`** for expensive tests
- **Mock I/O operations** in unit tests
- **Use test databases** for integration tests

## Continuous Integration

### Local Pre-commit Hooks

Install pre-commit hooks to run tests before commits:

```bash
pip install pre-commit
pre-commit install
```

This will automatically run:

- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)
- Security scanning (bandit)
- Basic tests

### GitHub Actions Workflow

The CI/CD pipeline runs on:

- **Push to main/develop branches**
- **Pull requests**
- **Manual workflow dispatch**

Key steps:

1. Python setup and dependency installation
2. ESCAI package installation in development mode
3. Import validation
4. Code quality checks (linting, type checking, security)
5. Unit and integration tests
6. Coverage reporting
7. Docker image building (on push)
8. Deployment to staging/production (on tags)

## Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run the validation script:** `python scripts/validate_imports.py`
3. **Check GitHub Actions logs** for CI/CD issues
4. **Create an issue** with detailed error information

## Contributing

When adding new tests:

1. **Follow the existing structure** (`tests/unit/`, `tests/integration/`)
2. **Add appropriate markers** (`@pytest.mark.unit`, etc.)
3. **Include docstrings** explaining what the test validates
4. **Ensure tests are deterministic** (no random failures)
5. **Update this guide** if adding new test categories or requirements
