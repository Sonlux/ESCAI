# ESCAI Framework Testing Suite

This directory contains the comprehensive testing suite for the ESCAI Framework, designed to ensure high quality, performance, and reliability of all components.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for component interactions
├── performance/             # Performance and monitoring overhead tests
├── load/                    # Load and stress tests
├── accuracy/                # ML model and prediction accuracy tests
├── e2e/                     # End-to-end workflow tests
├── utils/                   # Test utilities and data generators
├── conftest.py             # Pytest configuration and fixtures
├── run_comprehensive_tests.py  # Main test runner
└── README.md               # This file
```

## Test Types

### Unit Tests (`tests/unit/`)

- Test individual functions and classes in isolation
- Mock external dependencies
- Fast execution (< 1 second per test)
- Target: >95% code coverage

### Integration Tests (`tests/integration/`)

- Test component interactions
- Use real databases and services where possible
- Test framework instrumentors with actual frameworks
- Validate data flow between components

### Performance Tests (`tests/performance/`)

- Measure monitoring overhead (must be <10%)
- Test system scalability
- Benchmark processing speeds
- Memory usage validation

### Load Tests (`tests/load/`)

- Concurrent agent monitoring scenarios
- API endpoint stress testing
- System behavior under high load
- Resource exhaustion testing

### Accuracy Tests (`tests/accuracy/`)

- ML model prediction accuracy (>85% required)
- Pattern recognition validation
- Causal inference accuracy
- Cross-validation testing

### End-to-End Tests (`tests/e2e/`)

- Complete workflow testing
- Realistic agent scenarios
- Multi-component integration
- User journey validation

## Running Tests

### Quick Test Commands

```bash
# Run all unit tests
python run_tests.py unit

# Run integration tests
python run_tests.py integration

# Run performance tests
python run_tests.py performance

# Run all tests with coverage
python run_tests.py coverage

# Run comprehensive test suite
python run_tests.py all
```

### Comprehensive Test Suite

```bash
# Run full test suite with all validations
python tests/run_comprehensive_tests.py

# Skip specific test types
python tests/run_comprehensive_tests.py --skip-performance --skip-load

# Set custom coverage threshold
python tests/run_comprehensive_tests.py --coverage-threshold 90
```

### Individual Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests with database setup
pytest tests/integration/ -v -s

# Performance tests with detailed output
pytest tests/performance/ -v -s --durations=10

# Load tests (may take several minutes)
pytest tests/load/ -v -s --timeout=300

# Accuracy validation tests
pytest tests/accuracy/ -v -s

# End-to-end workflow tests
pytest tests/e2e/ -v -s
```

## Test Configuration

### Pytest Configuration

The test suite uses pytest with the following key configurations:

- Async test support via `pytest-asyncio`
- Coverage measurement via `pytest-cov`
- Timeout handling via `pytest-timeout`
- Mocking support via `pytest-mock`

### Test Fixtures

Common fixtures are defined in `conftest.py`:

- Sample data objects (epistemic states, patterns, etc.)
- Mock database connections
- Performance monitoring utilities
- Test data generators

### Environment Variables

Set these environment variables for testing:

```bash
export ESCAI_TEST_MODE=true
export ESCAI_LOG_LEVEL=DEBUG
export ESCAI_DB_URL=sqlite:///test.db
```

## Test Data Generation

The test suite includes automated test data generation utilities:

```python
from tests.utils.test_data_generator import TestDataGenerator

# Generate realistic test data
generator = TestDataGenerator(seed=42)
test_data = generator.generate_test_scenario_data("data_analysis")
```

### Available Test Scenarios

- `data_analysis`: Data processing and analysis workflows
- `web_scraping`: Web scraping and data extraction
- `api_integration`: API integration and synchronization
- `machine_learning`: ML model training and deployment

## Coverage Requirements

The test suite enforces strict coverage requirements:

- **Overall Coverage**: >95%
- **Critical Functions**: 100% coverage
- **Component Coverage**: >90% per component
- **Integration Coverage**: >85%

### Coverage Analysis

```bash
# Generate coverage report
python tests/utils/coverage_analyzer.py

# View coverage in browser
pytest --cov=escai_framework --cov-report=html
open htmlcov/index.html
```

## Performance Requirements

Tests validate that the system meets performance requirements:

- **Monitoring Overhead**: <10% impact on agent execution
- **API Response Time**: <500ms for standard endpoints
- **Event Processing**: >1000 events/second
- **Concurrent Agents**: Support for 100+ simultaneous agents
- **Memory Usage**: Stable memory consumption under load

## Writing New Tests

### Unit Test Example

```python
import pytest
from escai_framework.core.epistemic_extractor import EpistemicExtractor

class TestEpistemicExtractor:

    @pytest.mark.asyncio
    async def test_extract_beliefs(self, sample_agent_logs):
        extractor = EpistemicExtractor()
        beliefs = await extractor.extract_beliefs(sample_agent_logs)

        assert len(beliefs) > 0
        assert all(belief.confidence > 0 for belief in beliefs)
```

### Integration Test Example

```python
import pytest
from escai_framework.instrumentation.langchain_instrumentor import LangChainInstrumentor

class TestLangChainIntegration:

    @pytest.mark.asyncio
    async def test_full_monitoring_workflow(self, sample_agent_id):
        instrumentor = LangChainInstrumentor()

        session_id = await instrumentor.start_monitoring(sample_agent_id, {})
        # ... test monitoring workflow
        await instrumentor.stop_monitoring(session_id)
```

### Performance Test Example

```python
import pytest
import time

class TestPerformance:

    @pytest.mark.performance
    async def test_monitoring_overhead(self, performance_config):
        # Measure baseline performance
        start_time = time.perf_counter()
        # ... execute without monitoring
        baseline_time = time.perf_counter() - start_time

        # Measure with monitoring
        start_time = time.perf_counter()
        # ... execute with monitoring
        monitored_time = time.perf_counter() - start_time

        overhead = (monitored_time - baseline_time) / baseline_time
        assert overhead <= performance_config["max_monitoring_overhead"]
```

## Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run comprehensive tests
  run: |
    python tests/run_comprehensive_tests.py

- name: Upload coverage reports
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**

   - Ensure test databases are properly configured
   - Check connection strings and credentials

2. **Timeout Errors in Load Tests**

   - Increase timeout values for slower systems
   - Reduce concurrent load for resource-constrained environments

3. **Coverage Below Threshold**

   - Run coverage analyzer to identify gaps
   - Add tests for uncovered functions and branches

4. **Performance Test Failures**
   - Check system resources during test execution
   - Adjust performance thresholds for different hardware

### Debug Mode

Run tests with debug output:

```bash
pytest -v -s --log-cli-level=DEBUG tests/
```

### Test Isolation

Run tests in isolation to debug issues:

```bash
pytest tests/unit/test_specific_module.py::TestClass::test_method -v -s
```

## Contributing

When adding new features:

1. Write unit tests first (TDD approach)
2. Add integration tests for component interactions
3. Include performance tests for critical paths
4. Update accuracy tests for ML components
5. Add end-to-end tests for new workflows

Ensure all tests pass and coverage requirements are met before submitting pull requests.
