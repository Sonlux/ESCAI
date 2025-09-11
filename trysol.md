# Production-Ready GitHub Actions Workflow for ESCAI Framework

## 1. Project Structure Fix

First, ensure your project structure is correct:

```
ESCAI/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── escai_framework/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── epistemic_state.py
│   │   ├── belief_state.py
│   │   ├── knowledge_state.py
│   │   └── goal_state.py
│   └── ...
├── scripts/
│   └── validate_imports.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── unit/
│       └── test_*.py
├── setup.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## 2. Create/Update setup.py

```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="escai-framework",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ESCAI Framework for epistemic state management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ESCAI",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "docs"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-html>=3.1.0",
            "pytest-xdist>=3.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "bandit>=1.7.0",
        ]
    },
)
```

## 3. Create pyproject.toml (Alternative to setup.py)

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "escai-framework"
version = "0.1.0"
description = "ESCAI Framework for epistemic state management"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    # Add your production dependencies here
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    # ... other dependencies
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-html>=3.1.0",
    "pytest-xdist>=3.0.0",
    "flake8>=6.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
    "bandit>=1.7.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --tb=short --strict-markers"

[tool.coverage.run]
source = ["escai_framework"]
omit = ["*/tests/*", "*/test_*.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

## 4. Update validate_imports.py

```python
# scripts/validate_imports.py
#!/usr/bin/env python3
"""Validate that all ESCAI framework modules can be imported correctly."""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def validate_imports():
    """Test importing all main modules."""
    errors = []

    try:
        # Test main package import
        import escai_framework
        print("✓ ESCAI Framework package imported successfully")

        # Test models import
        from escai_framework.models import (
            EpistemicState,
            BeliefState,
            KnowledgeState,
            GoalState
        )
        print("✓ All model classes imported successfully")

        # Test individual module imports
        modules_to_test = [
            "escai_framework.models.epistemic_state",
            "escai_framework.models.belief_state",
            "escai_framework.models.knowledge_state",
            "escai_framework.models.goal_state",
        ]

        for module_name in modules_to_test:
            try:
                __import__(module_name)
                print(f"✓ {module_name} imported successfully")
            except ImportError as e:
                errors.append(f"✗ Failed to import {module_name}: {e}")

    except ImportError as e:
        errors.append(f"✗ Main import failed: {e}")

    if errors:
        print("\nImport validation failed:")
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print("\n✅ All imports validated successfully!")
        sys.exit(0)

if __name__ == "__main__":
    validate_imports()
```

## 5. Production-Ready GitHub Actions Workflow

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
    tags:
      - 'v*'
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run at 2 AM UTC every Monday
    - cron: '0 2 * * 1'

env:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "18"
  CACHE_VERSION: "v1"

jobs:
  # Job 1: Code Quality Checks
  quality-checks:
    name: Code Quality Checks
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better analysis

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ env.CACHE_VERSION }}-${{ hashFiles('**/requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-${{ env.CACHE_VERSION }}-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Install ESCAI package in development mode
        run: |
          pip install -e .
          echo "PYTHONPATH=$PYTHONPATH:$PWD" >> $GITHUB_ENV

      - name: Verify ESCAI installation
        run: |
          python -c "import escai_framework; print(f'ESCAI Framework version: {escai_framework.__version__ if hasattr(escai_framework, \"__version__\") else \"dev\"}')"
          python scripts/validate_imports.py

      - name: Lint with flake8
        run: |
          flake8 escai_framework/ --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 escai_framework/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

      - name: Format check with black
        run: |
          black --check escai_framework/ tests/

      - name: Type check with mypy
        run: |
          mypy escai_framework/ --ignore-missing-imports

      - name: Security check with bandit
        run: |
          bandit -r escai_framework/ -f json -o bandit-report.json
          bandit -r escai_framework/ -f screen

  # Job 2: Unit Tests
  unit-tests:
    name: Unit Tests (Python ${{ matrix.python-version }})
    runs-on: ${{ matrix.os }}
    needs: quality-checks

    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
        exclude:
          # Exclude some combinations to save CI time
          - os: windows-latest
            python-version: "3.10"
          - os: macos-latest
            python-version: "3.10"

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ env.CACHE_VERSION }}-${{ hashFiles('**/requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ env.CACHE_VERSION }}-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Install ESCAI package
        run: |
          pip install -e .

      - name: Run unit tests with coverage
        run: |
          pytest tests/unit/ -v \
            --cov=escai_framework \
            --cov-report=xml \
            --cov-report=html \
            --cov-report=term-missing \
            --junit-xml=junit.xml \
            --html=pytest-report.html \
            --self-contained-html

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results-${{ matrix.os }}-${{ matrix.python-version }}
          path: |
            junit.xml
            pytest-report.html
            htmlcov/

      - name: Upload coverage to Codecov
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: false

  # Job 3: Integration Tests
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-tests

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-integration-${{ env.CACHE_VERSION }}-${{ hashFiles('**/requirements*.txt') }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
          pip install -e .

      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://testuser:testpass@localhost:5432/testdb
          REDIS_URL: redis://localhost:6379/0
        run: |
          pytest tests/integration/ -v --tb=short

  # Job 4: Build and Publish
  build-publish:
    name: Build and Publish
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: |
          python -m build

      - name: Check package
        run: |
          twine check dist/*

      - name: Publish to Test PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}
        run: |
          twine upload --repository testpypi dist/*

      - name: Publish to PyPI
        if: success()
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          twine upload dist/*

  # Job 5: Docker Build
  docker-build:
    name: Docker Build and Push
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v'))

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ secrets.DOCKER_USERNAME }}/escai-framework
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # Job 6: Deploy (Example for staging/production)
  deploy:
    name: Deploy to ${{ matrix.environment }}
    runs-on: ubuntu-latest
    needs: [docker-build]
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v'))

    strategy:
      matrix:
        environment: [staging, production]
        exclude:
          - environment: production
            # Only deploy to production on version tags
            ${{ !startsWith(github.ref, 'refs/tags/v') }}

    environment:
      name: ${{ matrix.environment }}
      url: ${{ steps.deploy.outputs.url }}

    steps:
      - name: Deploy to ${{ matrix.environment }}
        id: deploy
        run: |
          echo "Deploying to ${{ matrix.environment }}..."
          # Add your deployment script here
          # For example: kubectl apply, helm upgrade, terraform apply, etc.
          echo "url=https://${{ matrix.environment }}.escai.example.com" >> $GITHUB_OUTPUT

# Status check job that other jobs can depend on
  status-check:
    name: CI Status Check
    runs-on: ubuntu-latest
    needs: [quality-checks, unit-tests, integration-tests]
    if: always()

    steps:
      - name: Check status
        run: |
          if [[ "${{ needs.quality-checks.result }}" == "failure" || \
                "${{ needs.unit-tests.result }}" == "failure" || \
                "${{ needs.integration-tests.result }}" == "failure" ]]; then
            echo "CI pipeline failed"
            exit 1
          else
            echo "CI pipeline succeeded"
          fi
```

## 6. Create requirements files

```txt
# requirements.txt (production dependencies)
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
pydantic>=2.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
sqlalchemy>=2.0.0
alembic>=1.11.0
redis>=4.6.0
celery>=5.3.0
```

```txt
# requirements-dev.txt (development dependencies)
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-html>=3.2.0
pytest-xdist>=3.3.0
pytest-asyncio>=0.21.0
pytest-mock>=3.11.0
flake8>=6.1.0
black>=23.7.0
mypy>=1.5.0
bandit>=1.7.5
pre-commit>=3.3.0
ipython>=8.14.0
jupyter>=1.0.0
```

## 7. Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Run the application
CMD ["python", "-m", "escai_framework"]
```

## 8. Fix **init**.py files

```python
# escai_framework/__init__.py
"""ESCAI Framework - Epistemic State Cognitive AI Framework."""

__version__ = "0.1.0"
__author__ = "Your Name"

from escai_framework.models import (
    EpistemicState,
    BeliefState,
    KnowledgeState,
    GoalState,
)

__all__ = [
    "EpistemicState",
    "BeliefState",
    "KnowledgeState",
    "GoalState",
]
```

```python
# escai_framework/models/__init__.py
"""ESCAI Framework Models."""

from escai_framework.models.epistemic_state import EpistemicState
from escai_framework.models.belief_state import BeliefState
from escai_framework.models.knowledge_state import KnowledgeState
from escai_framework.models.goal_state import GoalState

__all__ = [
    "EpistemicState",
    "BeliefState",
    "KnowledgeState",
    "GoalState",
]
```

## 9. Create conftest.py for tests

```python
# tests/conftest.py
"""Pytest configuration and fixtures."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_epistemic_state():
    """Create a sample epistemic state for testing."""
    from escai_framework.models import EpistemicState
    return EpistemicState()

# Add more fixtures as needed
```

## 10. GitHub Repository Settings

1. **Branch Protection Rules** (Settings → Branches):

   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Include administrators
   - Required status checks: `quality-checks`, `unit-tests`, `integration-tests`

2. **Secrets** (Settings → Secrets and variables → Actions):

   - `PYPI_API_TOKEN`: Your PyPI API token
   - `TEST_PYPI_API_TOKEN`: Your Test PyPI API token
   - `DOCKER_USERNAME`: Docker Hub username
   - `DOCKER_PASSWORD`: Docker Hub password
   - `CODECOV_TOKEN`: Codecov token (optional)

3. **Environments** (Settings → Environments):
   - Create `staging` and `production` environments
   - Add protection rules for production
   - Add required reviewers for production deployments

## Key Fixes Applied:

1. **Module Resolution**: Installing the package with `pip install -e .` ensures Python can find your modules
2. **Python Path**: Explicitly setting PYTHONPATH in the workflow
3. **Proper Package Structure**: Using `__init__.py` files with proper imports
4. **Validation Script**: Updated to handle imports correctly
5. **Caching**: Implemented dependency caching for faster CI runs
6. **Matrix Testing**: Testing across multiple OS and Python versions
7. **Proper Error Handling**: Each step has proper error handling and reporting
8. **Artifacts**: Storing test results and coverage reports
9. **Security Scanning**: Added bandit for security checks
10. **Docker Support**: Added containerization for deployment

This workflow is production-ready and includes:

- Code quality checks
- Multi-platform testing
- Security scanning
- Coverage reporting
- Automated deployment
- Docker containerization
- Proper error handling
- Caching for performance

The key fix for your immediate issue is ensuring the package is installed before running tests using `pip install -e .` which makes your package discoverable by Python.The main issue you're facing is that Python cannot find the escai_framework.models module. The key fixes are:

Install your package in editable mode before running tests:

bash pip install -e .

Ensure proper package structure with **init**.py files in all directories
Set PYTHONPATH in your GitHub Actions workflow

The artifact above provides a complete, production-ready CI/CD pipeline that addresses your issue and includes:

Proper module installation
Multi-OS and multi-Python version testing
Security scanning with bandit
Code quality checks (flake8, black, mypy)
Coverage reporting
Docker containerization
Automated deployment pipelines
Proper caching for performance

The immediate fix for your error is in the "Install ESCAI package in development mode" step which runs pip install -e . to make your package discoverable. This resolves the "No module named 'escai_framework.models'" error.
Would you like me to explain any specific part of the workflow or help you adapt it to your specific needs?
