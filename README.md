# ESCAI Framework 🧠

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/Sonlux/ESCAI)
[![Coverage](https://img.shields.io/badge/coverage-87%25-green.svg)](https://github.com/Sonlux/ESCAI)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/Sonlux/ESCAI/releases)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-blue.svg)](docs/)
[![CLI Tests](https://img.shields.io/badge/CLI%20tests-175%2B-green.svg)](tests/)

**ESCAI** (Epistemic State and Causal Analysis Intelligence) is a comprehensive observability system for monitoring autonomous agent cognition in real-time. It provides deep insights into how AI agents think, decide, and behave during task execution, enabling researchers and developers to understand agent behavior patterns, causal relationships, and performance characteristics.

> **🎯 Perfect for AI researchers, developers, and organizations working with autonomous agents**

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

## 🎯 Overview

ESCAI Framework is designed for researchers, developers, and organizations working with autonomous AI agents. It provides:

- **Real-time Monitoring**: Track agent execution, decisions, and state changes
- **Cognitive Analysis**: Extract and analyze epistemic states (beliefs, knowledge, goals)
- **Behavioral Insights**: Identify patterns in agent behavior and decision-making
- **Causal Understanding**: Discover cause-effect relationships in agent actions
- **Performance Prediction**: Forecast task outcomes and identify potential failures
- **Multi-Framework Support**: Works with LangChain, AutoGen, CrewAI, and OpenAI Assistants

### Use Cases

- **AI Research**: Study agent cognition and decision-making processes
- **Production Monitoring**: Monitor AI agents in production environments
- **Performance Optimization**: Identify bottlenecks and optimization opportunities
- **Failure Analysis**: Understand why agents fail and how to prevent it
- **Compliance & Auditing**: Track agent decisions for regulatory compliance

---

## ✨ Key Features

### 🔍 **Real-time Agent Monitoring**

- Monitor multiple agents simultaneously across different frameworks
- Track epistemic states (beliefs, knowledge, goals, uncertainty)
- Capture behavioral patterns and decision sequences
- Real-time performance metrics and health monitoring

### 🧠 **Cognitive Analysis**

- Extract and analyze agent beliefs, knowledge, and goals
- Track uncertainty levels and confidence scores
- Monitor reasoning depth and context awareness
- Identify cognitive biases and decision patterns

### 📊 **Advanced Analytics**

- Behavioral pattern mining with statistical significance testing
- Causal relationship discovery using advanced inference algorithms
- Performance prediction with machine learning models
- Failure analysis and root cause identification

### 🎨 **Interactive CLI Interface**

- Comprehensive command-line interface with 40+ commands
- ASCII charts and tables for terminal-based analysis
- Export capabilities (JSON, CSV, Markdown)
- Real-time monitoring and analysis tools

### 🌐 **Multi-Framework Support**

- **LangChain**: Chain execution monitoring and analysis
- **AutoGen**: Multi-agent conversation tracking
- **CrewAI**: Task delegation and crew coordination
- **OpenAI Assistants**: Tool usage and function calling

### 🗄️ **Flexible Storage**

- **PostgreSQL**: Structured data and relationships
- **MongoDB**: Document-based storage for complex data
- **Redis**: Caching and real-time data
- **InfluxDB**: Time-series metrics and performance data
- **Neo4j**: Graph-based causal relationships

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

#### **Option 1: Standard Installation**

```bash
pip install escai-framework
```

#### **Option 2: Development Installation**

```bash
git clone https://github.com/Sonlux/ESCAI.git
cd ESCAI
pip install -e .
```

#### **Option 3: Full Research Installation**

```bash
pip install -e ".[full]"
# Includes: jupyter, plotly, advanced ML libraries
```

#### **Option 4: Docker Installation**

```bash
docker pull escai/framework:latest
docker run -p 8000:8000 escai/framework:latest
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

![alt text](image-23.png)

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

![alt text](image-24.png)

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

#### **Complete Test Suite**

```bash
python tests/cli_test_runner.py --all
```

#### **CLI Validation**

```bash
python scripts/validate_cli_integration_windows.py
```

#### **Specific Test Categories**

```bash
# Unit tests
python tests/cli_test_runner.py --category unit

# Integration tests
python tests/cli_test_runner.py --category integration

# Performance tests
python tests/cli_test_runner.py --category performance

# End-to-end tests
python tests/cli_test_runner.py --category e2e
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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### **MIT License Summary**

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

---

## 🙏 Acknowledgments

- **Research Community**: For foundational work in agent cognition
- **Open Source Contributors**: For framework integrations and improvements
- **AI Framework Teams**: LangChain, AutoGen, CrewAI, and OpenAI teams
- **Database Communities**: PostgreSQL, MongoDB, Redis, InfluxDB, and Neo4j

---

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/Sonlux/ESCAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sonlux/ESCAI/discussions)
- **Email**: support@escai-framework.org

---

**Built with ❤️ for the AI research and development community**
