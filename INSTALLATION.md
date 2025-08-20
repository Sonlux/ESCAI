# ESCAI Framework Installation Guide

This guide provides detailed instructions for installing and setting up the ESCAI Framework.

## Quick Start

### Basic Installation

For basic functionality with core data models:

```bash
pip install git+https://github.com/Sonlux/ESCAI.git
```

### Development Installation

For development with all tools:

```bash
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI
pip install -e ".[dev]"
```

## Installation Options

### 1. Minimal Installation (Recommended for Getting Started)

```bash
# Install with minimal dependencies
pip install git+https://github.com/Sonlux/ESCAI.git

# Or from source
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI
pip install -e .
```

**Includes:**

- Core data models (EpistemicState, BehavioralPattern, CausalRelationship, PredictionResult)
- Validation and serialization utilities
- Basic examples and documentation

**Dependencies:**

- pandas>=1.5.0
- numpy>=1.21.0
- python-dateutil>=2.8.0
- pyyaml>=6.0

### 2. Full Installation (For Production Use)

```bash
# Install with all optional dependencies
pip install "git+https://github.com/Sonlux/ESCAI.git[full]"

# Or from source
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI
pip install -e ".[full]"
```

**Additional features:**

- Machine learning models (scikit-learn, xgboost)
- NLP capabilities (transformers, nltk)
- API framework (FastAPI, WebSockets)
- Database integrations (PostgreSQL, MongoDB, Redis, Neo4j, InfluxDB)
- Visualization tools (Plotly, Streamlit, Matplotlib)
- Network analysis (NetworkX)

### 3. Development Installation

```bash
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI
pip install -e ".[dev]"
```

**Includes development tools:**

- pytest for testing
- black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

## System Requirements

### Python Version

- Python 3.8 or higher
- Recommended: Python 3.9+

### Operating Systems

- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+, CentOS 7+, etc.)

### Hardware Requirements

- **Minimal**: 2GB RAM, 1GB disk space
- **Recommended**: 8GB RAM, 5GB disk space
- **Full installation**: 16GB RAM, 10GB disk space

## Verification

### Test Basic Installation

```bash
# Run basic functionality test
python -c "
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, BeliefType
from datetime import datetime
belief = BeliefState('test', BeliefType.FACTUAL, 0.9)
state = EpistemicState('agent1', datetime.utcnow(), [belief])
print('✓ Installation successful:', state.validate())
"
```

### Run Examples

```bash
# Run comprehensive example
python examples/basic_usage.py

# Run test suite (if available)
python test_basic_functionality.py
```

### Run Unit Tests (Development Installation)

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_epistemic_state.py

# Run with verbose output
pytest -v
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
ModuleNotFoundError: No module named 'escai_framework'
```

**Solution:**

- Ensure you've installed the package: `pip install -e .`
- Check your Python environment is correct
- Verify the package is in your Python path

#### 2. Dependency Conflicts

```bash
ERROR: Could not find a version that satisfies the requirement...
```

**Solutions:**

- Use minimal installation: `pip install -e .`
- Create a fresh virtual environment
- Update pip: `pip install --upgrade pip`

#### 3. Permission Errors (Windows)

```bash
PermissionError: [WinError 5] Access is denied
```

**Solutions:**

- Run command prompt as administrator
- Use `--user` flag: `pip install --user -e .`
- Check antivirus software isn't blocking installation

#### 4. Virtual Environment Issues

```bash
Command not found or wrong Python version
```

**Solutions:**

```bash
# Create new virtual environment
python -m venv escai_env

# Activate (Windows)
escai_env\Scripts\activate

# Activate (macOS/Linux)
source escai_env/bin/activate

# Install in virtual environment
pip install -e .
```

### Getting Help

If you encounter issues:

1. **Check the documentation**: README.md and examples/
2. **Search existing issues**: [GitHub Issues](https://github.com/Sonlux/ESCAI/issues)
3. **Create a new issue**: Include error messages, system info, and steps to reproduce
4. **Join discussions**: [GitHub Discussions](https://github.com/Sonlux/ESCAI/discussions)

## Virtual Environment Setup

### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv escai_env

# Activate virtual environment
# Windows:
escai_env\Scripts\activate
# macOS/Linux:
source escai_env/bin/activate

# Install ESCAI Framework
pip install -e .

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n escai python=3.9

# Activate environment
conda activate escai

# Install ESCAI Framework
pip install -e .

# Deactivate when done
conda deactivate
```

## Docker Installation (Advanced)

For containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .
RUN pip install -e .

# Run examples
CMD ["python", "examples/basic_usage.py"]
```

```bash
# Build and run
docker build -t escai-framework .
docker run escai-framework
```

## Performance Optimization

### For Large-Scale Deployments

1. **Use minimal installation** for core functionality
2. **Install only needed extras** (e.g., `[api]` for API features)
3. **Consider using conda** for better dependency management
4. **Use virtual environments** to avoid conflicts
5. **Monitor memory usage** with full installation

### Memory Usage

- **Minimal**: ~100MB
- **Full**: ~2-5GB (depending on ML models loaded)
- **Development**: ~500MB

## Next Steps

After successful installation:

1. **Run examples**: `python examples/basic_usage.py`
2. **Read documentation**: Check README.md and docstrings
3. **Explore the API**: Import and explore the modules
4. **Join the community**: GitHub Discussions and Issues
5. **Contribute**: See CONTRIBUTING.md for guidelines

## Support

- **Documentation**: README.md, examples/, and inline docstrings
- **Issues**: [GitHub Issues](https://github.com/Sonlux/ESCAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sonlux/ESCAI/discussions)
- **Email**: Create an issue for support requests
