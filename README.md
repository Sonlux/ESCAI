# ESCAI Framework 🧠

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![CI/CD](https://github.com/Sonlux/ESCAI/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Sonlux/ESCAI/actions/workflows/ci-cd.yml)
[![MyPy](https://img.shields.io/badge/mypy-passing-brightgreen.svg)](https://github.com/Sonlux/ESCAI)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://test.pypi.org/project/escai-framework/)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-blue.svg)](docs/)
[![CLI Commands](https://img.shields.io/badge/CLI%20commands-40%2B-green.svg)](docs/cli/)
[![Tests](https://img.shields.io/badge/tests-1429%2B-green.svg)](tests/)

**ESCAI** (Epistemic State and Causal Analysis Intelligence) is a **production-ready, CLI-first observability system** for monitoring autonomous AI agents in real-time. Built entirely around a powerful command-line interface, ESCAI provides researchers and developers with terminal-based tools to track agent cognition, analyze behavioral patterns, discover causal relationships, and predict performance—all without requiring a web interface.

> **🎯 A comprehensive CLI toolkit for AI researchers, ML engineers, and autonomous agent developers who prefer terminal-based workflows**

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [💻 CLI Usage](#-cli-usage)
- [🔧 API Integration](#-api-integration)
- [⚙️ Configuration](#️-configuration)
- [🧪 Testing](#-testing)
- [📚 Documentation](#-documentation)
- [🚀 Deployment](#-deployment)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🌟 Highlights & Achievements

### **Production-Ready Status**

- ✅ **Comprehensive CI/CD Pipeline**: Automated testing, building, and publishing to TestPyPI
- ✅ **Type-Safe Codebase**: 100% mypy compliant (resolved 154+ type errors)
- ✅ **Extensive Test Suite**: 1,429+ tests across unit, integration, E2E, performance, and UX categories
- ✅ **Security Hardening**: Pre-commit hooks, secret detection, OIDC trusted publishing
- ✅ **Zero Web Dependencies**: Pure CLI architecture with no frontend requirements

### **CLI Excellence**

- 📦 **40+ Commands**: Fully documented with comprehensive help system
- 🎨 **Rich Terminal UI**: ASCII art, colored output, tables, and progress indicators
- 🔄 **Interactive Mode**: Menu-driven interface for guided usage
- 📚 **Contextual Help**: Built-in tutorials, examples, and troubleshooting guides
- ⚡ **Performance Optimized**: 30-second test timeouts, intelligent caching

### **Research & Analysis Capabilities**

- 📊 **Pattern Mining**: Statistical behavioral pattern analysis
- 🔗 **Causal Discovery**: Graph-based causal relationship inference
- 🔮 **ML Predictions**: Performance forecasting with machine learning
- 📈 **Export Formats**: JSON, CSV, Markdown for research workflows
- 📄 **Academic Support**: IEEE-style research paper generation

---

## 🎯 Overview

### **A Pure CLI Experience - No Web Interface Required**

ESCAI Framework is built entirely around the command-line interface, making it perfect for researchers and developers who live in the terminal. Every feature—from real-time monitoring to complex analytics—is accessible through 40+ carefully designed CLI commands with rich terminal output, ASCII visualizations, and comprehensive help systems.

### **What ESCAI Provides**

- **🔍 Real-time Agent Monitoring**: Track agent execution, decisions, and epistemic state changes via terminal
- **🧠 Cognitive Analysis**: Extract and analyze agent beliefs, knowledge, goals, and uncertainty
- **📊 Behavioral Pattern Mining**: Discover recurring patterns in agent decision-making
- **🔗 Causal Relationship Discovery**: Understand cause-effect relationships between agent actions
- **📈 Performance Prediction**: Forecast task outcomes using ML models
- **🔌 Multi-Framework Support**: LangChain, AutoGen, CrewAI, and OpenAI Assistants
- **💾 Flexible Storage**: PostgreSQL, MongoDB, Redis, InfluxDB, and Neo4j
- **📦 Export Capabilities**: JSON, CSV, and Markdown output formats

### **Perfect For**

- **🔬 AI Researchers**: Study agent cognition and decision-making in controlled experiments
- **👨‍💻 ML Engineers**: Debug and optimize autonomous agents in production
- **🏢 Enterprise Teams**: Monitor multi-agent systems with comprehensive logging
- **🎓 Academics**: Analyze agent behavior for research papers and publications
- **⚡ CLI Enthusiasts**: Full-featured tooling for terminal-centric workflows

---

## ✨ Key Features

### 💻 **Powerful CLI Interface** (No Web Required!)

- **40+ Commands**: Organized into logical groups (`monitor`, `analyze`, `config`, `session`)
- **Rich Terminal Output**: Beautiful ASCII tables, charts, and colored formatting
- **Interactive Mode**: Guided menu system for discoverability
- **Comprehensive Help**: Built-in documentation with examples and tutorials
- **Tab Completion**: Shell completion support for faster workflows
- **Export Options**: JSON, CSV, and Markdown formats for all data

### 🔍 **Real-time Agent Monitoring**

- **Multi-Agent Support**: Monitor multiple agents simultaneously
- **Framework Agnostic**: Works with LangChain, AutoGen, CrewAI, and OpenAI
- **Epistemic State Tracking**: Capture beliefs, knowledge, goals, and uncertainty
- **Low Overhead**: < 5% performance impact with intelligent buffering
- **Session Management**: Save, resume, and replay monitoring sessions

### 🧠 **Cognitive Analysis**

- **Belief Extraction**: Track agent beliefs and how they evolve
- **Knowledge Graphs**: Visualize agent knowledge structures (via Neo4j)
- **Goal Tracking**: Monitor goal states and achievement progress
- **Uncertainty Analysis**: Measure confidence and uncertainty levels
- **Decision Pattern Recognition**: Identify recurring decision-making patterns

### 📊 **Advanced Analytics**

- **Pattern Mining**: Discover behavioral patterns with statistical significance
- **Causal Inference**: Graph-based causal relationship discovery
- **Performance Prediction**: ML-powered outcome forecasting
- **Failure Analysis**: Root cause identification for agent failures
- **Comparative Analysis**: Compare sessions and identify regressions

### 🌐 **Multi-Framework Instrumentation**

- **LangChain**: Chain execution, tool usage, and callback monitoring
- **AutoGen**: Multi-agent conversations and role-based tracking
- **CrewAI**: Task delegation and crew coordination analysis
- **OpenAI Assistants**: Function calls, tool usage, and thread monitoring

### 🗄️ **Enterprise-Grade Storage**

- **PostgreSQL**: Structured data, relationships, and SQLAlchemy models
- **MongoDB**: Document storage for complex nested data
- **Redis**: Real-time caching and session management
- **InfluxDB**: Time-series metrics and performance monitoring
- **Neo4j**: Graph-based causal networks and knowledge graphs

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ESCAI Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  CLI Interface         │  API Endpoints     │  Data Processing   │
│  ┌─────────────────┐   │  ┌─────────────┐  │  ┌─────────────┐   │
│  │ Monitor         │   │  │ REST API    │  │  │ Real-time   │   │
│  │ Analyze         │   │  │ WebSocket   │  │  │ Analytics   │   │
│  │ Config          │   │  │ GraphQL     │  │  │ Monitoring  │   │
│  │ Session         │   │  │ Streaming   │  │  │ Processing  │   │
│  └─────────────────┘   │  └─────────────┘  │  └─────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Core Processing Layer                       │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐│
│  │ Epistemic       │   │ Pattern         │   │ Causal          ││
│  │ Extractor       │   │ Analyzer        │   │ Engine          ││
│  └─────────────────┘   └─────────────────┘   └─────────────────┘│
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐│
│  │ Performance     │   │ Explanation     │   │ Analytics       ││
│  │ Predictor       │   │ Engine          │   │ Pipeline        ││
│  └─────────────────┘   └─────────────────┘   └─────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                   Framework Instrumentation                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │ LangChain   │ │ AutoGen     │ │ CrewAI      │ │ OpenAI      ││
│  │ Instrumentor│ │ Instrumentor│ │ Instrumentor│ │ Instrumentor││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                      Storage Layer                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │ PostgreSQL  │ │ MongoDB     │ │ Redis       │ │ InfluxDB    ││
│  │ (Structured)│ │ (Documents) │ │ (Cache)     │ │ (Metrics)   ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
│                              ┌─────────────┐                    │
│                              │ Neo4j       │                    │
│                              │ (Graphs)    │                    │
│                              └─────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 📁 **Framework Structure**

```
escai_framework/
├── core/                    # Core processing engines
│   ├── epistemic_extractor.py    # Extract agent beliefs, knowledge, goals
│   ├── pattern_analyzer.py       # Behavioral pattern mining
│   ├── causal_engine.py          # Causal relationship discovery
│   ├── performance_predictor.py  # Performance prediction models
│   └── explanation_engine.py     # Human-readable explanations
├── instrumentation/         # Framework integrations
│   ├── base_instrumentor.py      # Base instrumentation interface
│   ├── langchain_instrumentor.py # LangChain integration
│   ├── autogen_instrumentor.py   # AutoGen integration
│   ├── crewai_instrumentor.py    # CrewAI integration
│   └── openai_instrumentor.py    # OpenAI Assistants integration
├── cli/                     # Command-line interface
│   ├── main.py                   # CLI entry point
│   ├── commands/                 # Command implementations
│   ├── utils/                    # CLI utilities and formatters
│   └── integration/              # Framework connectors
├── api/                     # REST API and WebSocket endpoints
│   ├── main.py                   # FastAPI application
│   ├── monitoring.py             # Monitoring endpoints
│   ├── analysis.py               # Analysis endpoints
│   └── websocket.py              # Real-time WebSocket
├── storage/                 # Database management
│   ├── database.py               # Database connections
│   ├── models.py                 # SQLAlchemy models
│   ├── repositories/             # Data access layer
│   └── migrations/               # Database migrations
├── analytics/               # Machine learning and analytics
│   ├── pattern_mining.py         # Pattern discovery algorithms
│   ├── prediction_models.py      # ML prediction models
│   └── failure_analysis.py       # Failure analysis tools
├── security/                # Security and authentication
│   ├── auth_manager.py           # Authentication management
│   ├── rbac.py                   # Role-based access control
│   └── audit_logger.py           # Security audit logging
├── config/                  # Configuration management
│   ├── config_manager.py         # Configuration handling
│   ├── config_validator.py       # Configuration validation
│   └── config_encryption.py      # Secure configuration storage
└── utils/                   # Shared utilities
    ├── exceptions.py             # Custom exceptions
    ├── validation.py             # Data validation
    └── serialization.py          # Data serialization
```

---

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone the repository
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install ESCAI Framework
pip install -e .

# Install with all dependencies for research
pip install -e ".[full]"
```

### 2. **Verify Installation**

```bash
escai --version
```

_Expected output: Shows ESCAI Framework version and basic system info_

### 3. **View CLI Help System**

```bash
escai help
```

_Expected output: Shows the main help interface with command categories and quick reference_

### 4. **Interactive Setup**

```bash
escai config setup
```

_Expected output: Interactive setup wizard for database and framework configuration_

### 5. **Check System Status**

```bash
escai config check
```

_Expected output: System check results showing component status and framework availability_

### 6. **Launch Interactive Mode**

```bash
escai --interactive
```

_Expected output: Interactive menu system with numbered options for navigation_

---

## 📦 Installation

### Prerequisites

- **Python**: 3.11+ (3.12+ recommended)
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free disk space
- **OS**: Windows 10+, macOS 10.15+, or Linux

### Installation Options

#### **Option 1: From TestPyPI** (Latest Development Build)

```bash
pip install -i https://test.pypi.org/simple/ escai-framework
```

#### **Option 2: From Source** (Recommended for Development)

```bash
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI
pip install -e .
```

#### **Option 3: Full Research Installation** (Includes Optional Dependencies)

```bash
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI
pip install -e ".[full]"
# Includes: jupyter, plotly, advanced ML libraries, visualization tools
```

#### **Option 4: Docker Installation** (Isolated Environment)

```bash
docker-compose up
# Includes: API server, PostgreSQL, MongoDB, Redis, InfluxDB, Neo4j, Grafana
```

### Verify Installation

```bash
escai --version
```

_Shows: ESCAI Framework version, Python version, and system information_

```bash
escai config check
```

_Shows: System check results, component status, and framework availability_

---

## 💻 CLI Usage

The ESCAI CLI provides comprehensive commands for monitoring, analysis, and configuration with an intuitive interface.

<img width="2528" height="844" alt="image" src="https://github.com/user-attachments/assets/c601e5df-c66d-45cb-8243-b334efd45969" />

### **Core Commands**

#### **Monitor Commands**

```bash
# Start monitoring an agent
escai monitor start --agent-id demo-agent --framework langchain

# View monitoring status
escai monitor status

# Stop monitoring
escai monitor stop --agent-id demo-agent

# List all monitoring sessions
escai monitor list
```

#### **Analysis Commands**

```bash
# Analyze behavioral patterns
escai analyze patterns --agent-id demo-agent --timeframe 1h

# Perform causal analysis
escai analyze causal --agent-id demo-agent

# Generate performance predictions
escai analyze predict --agent-id demo-agent

# Export analysis results
escai analyze export --format json --output results.json
```

#### **Configuration Commands**

```bash
# Show current configuration
escai config show

# Interactive setup wizard
escai config setup

# Test framework compatibility
escai config test --framework langchain

# Validate configuration
escai config validate
```

#### **Session Management**

```bash
# List all sessions
escai session list

# View session details
escai session show --session-id abc123

# Export session data
escai session export --session-id abc123 --format csv
```

### **Advanced Features**

#### **Interactive Mode**

```bash
escai --interactive
```

Launches a menu-driven interface for guided usage.

#### **Debug Mode**

```bash
escai --debug config check
```

Provides verbose output for troubleshooting.

#### **Help System**

```bash
# General help
escai help

# Command-specific help
escai help monitor

# Topic-based help
escai help getting_started

# Search help content
escai help search monitoring
```

<img width="1651" height="851" alt="image" src="https://github.com/user-attachments/assets/9c352308-5c1d-40aa-86cd-4795aa18fd8f" />

---

## 🔧 API Integration

The ESCAI framework provides REST API and WebSocket endpoints for programmatic access to monitoring and analysis capabilities.

### **API Server**

#### **Start API Server**

```bash
python -m escai_framework.api.main
```

#### **Health Check**

```bash
curl http://localhost:8000/health
```

#### **API Documentation**

Interactive API documentation is available at `http://localhost:8000/docs` when the server is running.

### **Core API Endpoints**

#### **Monitoring Endpoints**

```bash
# Start monitoring session
curl -X POST http://localhost:8000/api/v1/monitor/start \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "demo", "framework": "langchain"}'

# Get agent status
curl http://localhost:8000/api/v1/agents/demo/status

# Stop monitoring
curl -X POST http://localhost:8000/api/v1/monitor/demo/stop

# List sessions
curl http://localhost:8000/api/v1/sessions
```

#### **Analysis Endpoints**

```bash
# Get behavioral patterns
curl http://localhost:8000/api/v1/patterns/analyze?agent_id=demo

# Causal analysis
curl -X POST http://localhost:8000/api/v1/causal/analyze \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "demo"}'

# Performance prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "demo", "horizon": "1h"}'
```

### **WebSocket Integration**

```bash
# Test WebSocket connection
python examples/websocket_example.py
```

Real-time data streaming for live monitoring and analysis updates.

### **Python SDK**

```python
from escai_framework import ESCAIClient

# Initialize client
client = ESCAIClient(api_url="http://localhost:8000")

# Start monitoring
session = client.monitor.start(
    agent_id="my-agent",
    framework="langchain"
)

# Get real-time status
status = client.monitor.status(session.id)

# Analyze patterns
patterns = client.analyze.patterns(
    agent_id="my-agent",
    timeframe="1h"
)

# Stop monitoring
client.monitor.stop(session.id)
```

---

## ⚙️ Configuration

### **Configuration File Structure**

```yaml
# config/config.yaml
postgresql:
  host: localhost
  port: 5432
  database: escai
  username: escai_user
  password: secure_password

mongodb:
  host: localhost
  port: 27017
  database: escai_docs

redis:
  host: localhost
  port: 6379
  db: 0

api:
  host: 0.0.0.0
  port: 8000
  jwt_secret: your-secret-key

monitoring:
  max_overhead_percent: 5
  buffer_size: 1000
  retention_days: 30

frameworks:
  langchain:
    capture_chains: true
    capture_tools: true
    max_depth: 10

  autogen:
    capture_conversations: true
    track_roles: true

  crewai:
    capture_tasks: true
    track_delegation: true

  openai:
    capture_function_calls: true
    track_tool_usage: true
```

### **Environment Variables**

```bash
export ESCAI_CONFIG_PATH=/path/to/config.yaml
export ESCAI_LOG_LEVEL=INFO
export ESCAI_DATABASE_URL=postgresql://user:pass@localhost/escai
export ESCAI_REDIS_URL=redis://localhost:6379/0
```

### **Configuration Management**

```bash
# Generate default configuration
escai config init

# Validate configuration
escai config validate

# Show current configuration
escai config show

# Test database connections
escai config test --databases

# Test framework integrations
escai config test --frameworks
```

---

## 🧪 Testing

ESCAI includes a comprehensive testing suite with 175+ tests across multiple categories.

### **Test Categories**

- **Unit Tests**: Individual component testing
- **Integration Tests**: Framework integration testing
- **End-to-End Tests**: Complete workflow testing
- **Performance Tests**: Large dataset handling
- **User Experience Tests**: CLI usability testing
- **Documentation Tests**: Help content accuracy
- **Accessibility Tests**: Screen reader compatibility

### **Running Tests**

ESCAI includes **1,429+ tests** with comprehensive coverage across all categories:

#### **Complete Test Suite** (All 1,429 Tests)

```bash
python tests/cli_test_runner.py --all
```

**Test Categories:**

- **Unit Tests** (~1,198 tests): Individual component testing
- **Integration Tests** (~231 tests): Framework integration testing
- **E2E Tests**: Complete workflow testing
- **Performance Tests**: Large dataset handling
- **UX Tests**: CLI usability and experience
- **Documentation Tests**: Help content accuracy

#### **CI/CD Pipeline Tests** (GitHub Actions)

```bash
# Runs automatically on every commit
# Tests complete in ~15 minutes (with 30-second per-test timeout)
# View at: https://github.com/Sonlux/ESCAI/actions
```

#### **Specific Test Categories**

```bash
# Unit tests (1,198 tests - fast)
python tests/cli_test_runner.py --category unit

# Integration tests (231 tests - requires databases)
python tests/cli_test_runner.py --category integration

# Performance tests (measure large dataset handling)
python tests/cli_test_runner.py --category performance

# End-to-end tests (complete workflows)
python tests/cli_test_runner.py --category e2e
```

#### **Test with Coverage**

```bash
# Generate coverage report
python -m pytest --cov=escai_framework --cov-report=html tests/

# View coverage in browser
open htmlcov/index.html
```

#### **Framework-Specific Tests**

```bash
# Test LangChain integration
python -m pytest tests/integration/test_langchain_instrumentor.py

# Test AutoGen integration
python -m pytest tests/integration/test_autogen_instrumentor.py

# Test CrewAI integration
python -m pytest tests/integration/test_crewai_instrumentor.py

# Test OpenAI integration
python -m pytest tests/integration/test_openai_instrumentor.py
```

### **Test Configuration**

```bash
# Validate test environment
python tests/cli_test_runner.py --validate

# Generate test coverage report
python -m pytest --cov=escai_framework tests/

# Run tests with verbose output
python tests/cli_test_runner.py --verbose
```

---

## 📚 Documentation

### **Documentation Structure**

```
docs/
├── api/                     # API documentation
│   ├── README.md           # API overview
│   └── openapi.yaml        # OpenAPI specification
├── cli/                     # CLI documentation
│   ├── commands.md         # Command reference
│   ├── examples.md         # Usage examples
│   └── validation-system.md # Input validation
├── integration/             # Framework integration guides
│   ├── langchain.md        # LangChain integration
│   ├── autogen.md          # AutoGen integration
│   └── framework-robustness.md # Robust integration
├── deployment/              # Deployment guides
│   ├── quick-start.md      # Quick deployment
│   └── production.md       # Production deployment
├── security/                # Security documentation
│   ├── README.md           # Security overview
│   └── best-practices.md   # Security best practices
├── architecture/            # Architecture documentation
│   ├── README.md           # System architecture
│   └── database-design.md  # Database design
├── performance/             # Performance documentation
│   ├── README.md           # Performance overview
│   └── optimization.md     # Optimization guide
└── troubleshooting/         # Troubleshooting guides
    ├── README.md           # Common issues
    └── framework-integration.md # Framework issues
```

### **Key Documentation**

- **[API Reference](docs/api/README.md)**: Complete API documentation
- **[CLI Guide](docs/cli/comprehensive-guide.md)**: Comprehensive CLI reference
- **[Integration Guide](docs/integration/README.md)**: Framework integration
- **[Deployment Guide](docs/deployment/quick-start.md)**: Production deployment
- **[Architecture Guide](docs/architecture/README.md)**: System architecture
- **[Security Guide](docs/security/README.md)**: Security best practices

---

## 🚀 Deployment

### **Development Deployment**

#### **Local Development**

```bash
# Start API server
python -m escai_framework.api.main

# Start with custom configuration
ESCAI_CONFIG_PATH=config/dev.yaml python -m escai_framework.api.main
```

#### **Docker Development**

```bash
# Build development image
docker build -t escai-framework:dev .

# Run development container
docker run -p 8000:8000 -v $(pwd):/app escai-framework:dev
```

### **Production Deployment**

#### **Docker Deployment**

```bash
# Build production image
docker build -t escai-framework:latest .

# Run production container
docker run -d \
  --name escai \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/escai \
  -e REDIS_URL=redis://redis:6379/0 \
  escai-framework:latest
```

#### **Docker Compose**

```yaml
# docker-compose.yml
version: "3.8"
services:
  escai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/escai
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:14
    environment:
      POSTGRES_DB: escai
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
```

#### **Kubernetes Deployment**

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=escai

# Access API
kubectl port-forward svc/escai-api 8000:8000
```

#### **Cloud Deployment**

- **AWS**: Use ECS, EKS, or Lambda
- **Google Cloud**: Use Cloud Run, GKE, or App Engine
- **Azure**: Use Container Instances, AKS, or App Service
- **Heroku**: Use container deployment

### **Production Configuration**

```yaml
# Production settings
api:
  host: 0.0.0.0
  port: 8000
  workers: 4

monitoring:
  max_overhead_percent: 2
  buffer_size: 10000

security:
  enable_auth: true
  jwt_secret: ${JWT_SECRET}
  rate_limit: 1000

logging:
  level: INFO
  format: json
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how to get started:

### **Development Setup**

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/ESCAI.git
cd ESCAI

# Create development environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### **Development Workflow**

1. **Create Feature Branch**: `git checkout -b feature/your-feature`
2. **Make Changes**: Implement your feature or fix
3. **Run Tests**: `python tests/cli_test_runner.py --all`
4. **Commit Changes**: `git commit -m "feat: add your feature"`
5. **Push Branch**: `git push origin feature/your-feature`
6. **Create Pull Request**: Submit PR for review

### **Code Standards**

- **Python**: Follow PEP 8 style guidelines
- **Type Hints**: Use type annotations for all functions
- **Documentation**: Add docstrings for all public methods
- **Testing**: Write tests for new functionality
- **Commit Messages**: Use conventional commit format

### **Areas for Contribution**

- **Framework Integrations**: Add support for new AI frameworks
- **Analytics Algorithms**: Improve pattern mining and causal inference
- **CLI Enhancements**: Add new commands and features
- **Documentation**: Improve guides and examples
- **Performance**: Optimize monitoring overhead
- **Security**: Enhance authentication and authorization

---

## ⌨️ CLI Command Quick Reference

ESCAI provides **40+ commands** organized into logical groups:

### **Core Command Groups**

| Group         | Commands    | Purpose                                            |
| ------------- | ----------- | -------------------------------------------------- |
| `monitor`     | 8 commands  | Start, stop, and manage real-time agent monitoring |
| `analyze`     | 13 commands | Analyze patterns, causality, and predictions       |
| `config`      | 8 commands  | Configure framework, databases, and settings       |
| `session`     | 9 commands  | Manage, replay, and export monitoring sessions     |
| `publication` | 5 commands  | Generate research reports and papers               |
| `help`        | 5 commands  | Comprehensive help system and tutorials            |

### **Most Used Commands**

```bash
# Quick Start
escai --version                    # Show version and system info
escai help                         # Main help interface
escai --interactive                # Launch interactive mode

# Monitoring
escai monitor start --framework langchain --agent-id my-agent
escai monitor status               # Check monitoring status
escai monitor stop --all           # Stop all sessions

# Analysis
escai analyze patterns --agent-id my-agent --timeframe 1h
escai analyze causal --min-strength 0.7
escai analyze predict --agent-id my-agent

# Configuration
escai config setup                 # Interactive setup wizard
escai config check                 # Validate system configuration
escai config test --framework langchain

# Sessions
escai session list --active        # List active sessions
escai session export --format json # Export session data
```

**Full Command Reference**: See [docs/cli/comprehensive-guide.md](docs/cli/comprehensive-guide.md)

---

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### **MIT License Summary**

- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❌ No liability
- ❌ No warranty

---

## 🙏 Acknowledgments

### **Technology Stack**

- **CLI Framework**: Click + Rich for beautiful terminal interfaces
- **AI Frameworks**: LangChain, AutoGen, CrewAI, and OpenAI teams
- **Databases**: PostgreSQL, MongoDB, Redis, InfluxDB, and Neo4j communities
- **Testing**: Pytest, MyPy, and comprehensive test tooling

### **Special Thanks**

- **Research Community**: For foundational work in agent cognition and observability
- **Open Source Contributors**: For framework integrations and improvements
- **GitHub Actions**: For OIDC trusted publishing and robust CI/CD

---

## 📞 Support & Community

### **Get Help**

- 📚 **Documentation**: [docs/](docs/) - Comprehensive guides and tutorials
- 🐛 **Issues**: [GitHub Issues](https://github.com/Sonlux/ESCAI/issues) - Bug reports and feature requests
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Sonlux/ESCAI/discussions) - Community Q&A
- ✉️ **Email**: <lakshanammineni@gmail.com> - Direct support

### **Contributing**

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup instructions
- Code standards and conventions
- Testing requirements
- Pull request process

### **Stay Updated**

- ⭐ Star this repo on GitHub
- 👀 Watch for releases
- 📦 Follow on TestPyPI

---

<div align="center">

**Built with ❤️ for the AI research and development community**

_ESCAI Framework - Making Agent Cognition Observable, One Command at a Time_

[![GitHub](https://img.shields.io/badge/GitHub-ESCAI-blue?logo=github)](https://github.com/Sonlux/ESCAI)
[![TestPyPI](https://img.shields.io/badge/TestPyPI-escai--framework-blue)](https://test.pypi.org/project/escai-framework/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>
