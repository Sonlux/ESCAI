# ESCAI Framework 🧠

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/Sonlux/ESCAI)
[![Coverage](https://img.shields.io/badge/coverage-85%25-yellow.svg)](https://github.com/Sonlux/ESCAI)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Sonlux/ESCAI/releases)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-blue.svg)](docs/)
[![CLI Tests](https://img.shields.io/badge/CLI%20tests-175%2B-green.svg)](tests/)

**ESCAI** (Epistemic State and Causal Analysis Intelligence) is a comprehensive observability system for monitoring autonomous agent cognition in real-time. It provides deep insights into how AI agents think, decide, and behave during task execution, enabling researchers and developers to understand agent behavior patterns, causal relationships, and performance characteristics.

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [💻 CLI Usage](#-cli-usage)
- [🌐 Web Dashboard](#-web-dashboard)
- [🔧 Configuration](#-configuration)
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

### 🎨 **Interactive CLI & Dashboard**

- Comprehensive command-line interface with 40+ commands
- Real-time web dashboard with interactive visualizations
- ASCII charts and tables for terminal-based analysis
- Export capabilities (JSON, CSV, Markdown)

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
│  CLI Interface          │  Web Dashboard    │  API Endpoints    │
│  ┌─────────────────┐   │  ┌─────────────┐  │  ┌─────────────┐  │
│  │ Monitor         │   │  │ Real-time   │  │  │ REST API    │  │
│  │ Analyze         │   │  │ Visualizations│ │  │ WebSocket   │  │
│  │ Config          │   │  │ Interactive │  │  │ GraphQL     │  │
│  │ Session         │   │  │ Controls    │  │  │ Streaming   │  │
│  └─────────────────┘   │  └─────────────┘  │  └─────────────┘  │
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

### 2. **Basic Configuration**

```bash
# Run interactive setup
escai config setup

# Or configure manually
escai config set postgresql host localhost
escai config set postgresql port 5432
escai config set postgresql database escai
```

### 3. **Start Monitoring**

```bash
# Start monitoring a LangChain agent
escai monitor start --agent-id my-agent --framework langchain

# View real-time status
escai monitor status

# Analyze patterns
escai analyze patterns --agent-id my-agent --interactive
```

### 4. **Launch Dashboard**

```bash
# Start the web dashboard
escai monitor dashboard

# Open browser to http://localhost:8000
```

---

## 📦 Installation

### Prerequisites

- **Python**: 3.8+ (3.10+ recommended)
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
# Includes: jupyter, plotly, streamlit, advanced ML libraries
```

#### **Option 4: Docker Installation**

```bash
docker pull escai/framework:latest
docker run -p 8000:8000 escai/framework:latest
```

### Verify Installation

```bash
escai --version
escai config check
```

---

## 💻 CLI Usage

The ESCAI CLI provides comprehensive commands for monitoring, analysis, and configuration.

### **Monitor Commands**

```bash
# Start monitoring
escai monitor start --agent-id demo-agent --framework langchain

# View agent status
escai monitor status

# Real-time epistemic state monitoring
escai monitor epistemic --agent-id demo-agent --refresh 2

# Launch interactive dashboard
escai monitor dashboard

# Stream live logs
escai monitor logs --filter error --highlight "timeout"

# Stop monitoring
escai monitor stop --agent-id demo-agent
```

### **Analysis Commands**

```bash
# Analyze behavioral patterns
escai analyze patterns --agent-id demo-agent --timeframe 1h

# Interactive pattern exploration
escai analyze patterns --interactive

# Causal relationship analysis
escai analyze causal --min-strength 0.7

# Performance predictions
escai analyze predictions --agent-id demo-agent --horizon 30m

# Generate visualizations
escai analyze visualize --chart-type heatmap --metric confidence

# Export analysis results
escai analyze export --format json --timeframe 24h
```

### **Configuration Commands**

```bash
# Interactive setup wizard
escai config setup

# View current configuration
escai config show

# Set configuration values
escai config set api host localhost
escai config set monitoring max_overhead_percent 5

# Test database connections
escai config test

# Manage themes
escai config theme --scheme dark
```

### **Session Management**

```bash
# List monitoring sessions
escai session list

# View session details
escai session details session-123

# Export session data
escai session export session-123 --format csv

# Clean up old sessions
escai session cleanup --older-than 7d
```

### **Advanced Features**

```bash
# Search and filter data
escai analyze search --query "error" --field event_type

# Compare multiple agents
escai analyze health --compare-agents

# Generate research reports
escai analyze timeline --include-statistics --format-for-publication

# Batch operations
escai monitor start --config batch-config.yaml
```

---

## 🌐 Web Dashboard

The ESCAI web dashboard provides real-time visualization and interactive analysis capabilities.

### **Features**

- **Real-time Monitoring**: Live agent status and metrics
- **Interactive Charts**: Behavioral patterns, causal graphs, performance trends
- **Epistemic State Viewer**: Real-time belief, knowledge, and goal tracking
- **Alert Management**: Configure alerts for anomalies and failures
- **Export Tools**: Download data and visualizations

### **Access**

```bash
# Start dashboard
escai monitor dashboard

# Custom configuration
escai monitor dashboard --host 0.0.0.0 --port 8080

# Open browser to http://localhost:8000
```

### **Dashboard Sections**

- **Overview**: System status and key metrics
- **Agents**: Individual agent monitoring and analysis
- **Patterns**: Behavioral pattern visualization
- **Causal**: Causal relationship graphs
- **Performance**: Performance metrics and predictions
- **Sessions**: Session management and history

---

## 🔧 Configuration

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

ui:
  theme: dark
  color_scheme: professional
```

### **Environment Variables**

```bash
export ESCAI_CONFIG_PATH=/path/to/config.yaml
export ESCAI_LOG_LEVEL=INFO
export ESCAI_DATABASE_URL=postgresql://user:pass@localhost/escai
export ESCAI_REDIS_URL=redis://localhost:6379/0
```

### **Framework-Specific Configuration**

```yaml
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

```bash
# Run all tests
python tests/cli_test_runner.py --all

# Run specific category
python tests/cli_test_runner.py --category unit

# Run with coverage
python -m pytest --cov=escai_framework tests/

# Performance tests
python tests/cli_test_runner.py --category performance

# Validate test environment
python tests/cli_test_runner.py --validate
```

### **Test Configuration**

```bash
# Run tests with verbose output
python tests/cli_test_runner.py --category integration --verbose

# Run specific test file
python -m pytest tests/unit/test_cli_commands.py -v

# Run tests in parallel
python -m pytest -n auto tests/
```

### **Continuous Integration**

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -e ".[test]"
      - run: python tests/cli_test_runner.py --all
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
│   ├── production.md       # Production deployment
│   └── kubernetes.md       # Kubernetes deployment
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
- **[CLI Guide](docs/cli/commands.md)**: Comprehensive CLI reference
- **[Integration Guide](docs/integration/README.md)**: Framework integration
- **[Deployment Guide](docs/deployment/quick-start.md)**: Production deployment
- **[Architecture Guide](docs/architecture/README.md)**: System architecture
- **[Security Guide](docs/security/README.md)**: Security best practices

---

## 🚀 Deployment

### **Development Deployment**

```bash
# Start development server
escai monitor dashboard --dev

# With hot reload
escai monitor dashboard --dev --reload
```

### **Production Deployment**

#### **Docker Deployment**

```bash
# Build image
docker build -t escai-framework .

# Run container
docker run -d \
  --name escai \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/escai \
  escai-framework
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

# Access dashboard
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
4. **Run Linting**: `pre-commit run --all-files`
5. **Commit Changes**: `git commit -m "feat: your feature description"`
6. **Push Branch**: `git push origin feature/your-feature`
7. **Create Pull Request**: Submit PR with description

### **Contribution Guidelines**

- **Code Style**: Follow PEP 8 and use Black formatter
- **Testing**: Add tests for new features (aim for >80% coverage)
- **Documentation**: Update documentation for new features
- **Commit Messages**: Use conventional commit format
- **Pull Requests**: Provide clear description and link issues

### **Areas for Contribution**

- **New Framework Integrations**: Add support for new AI frameworks
- **Visualization Improvements**: Enhance charts and dashboards
- **Performance Optimizations**: Improve system performance
- **Documentation**: Improve guides and examples
- **Testing**: Add more comprehensive tests
- **Security**: Enhance security features

### **Community**

- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs and request features
- **Discord**: Join our community chat (coming soon)
- **Blog**: Read about ESCAI developments

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 ESCAI Framework Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Contributors**: Thanks to all contributors who have helped build ESCAI
- **Open Source Community**: Built on top of amazing open source projects
- **Research Community**: Inspired by cutting-edge research in agent cognition
- **Framework Authors**: LangChain, AutoGen, CrewAI, and OpenAI teams
- **Database Teams**: PostgreSQL, MongoDB, Redis, InfluxDB, and Neo4j teams

### **Built With**

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Streamlit](https://streamlit.io/) - Dashboard framework

---

## 📊 Project Stats

- **Lines of Code**: 50,000+
- **Test Coverage**: 85%+
- **CLI Commands**: 40+
- **Test Cases**: 175+
- **Documentation Pages**: 25+
- **Supported Frameworks**: 4
- **Database Integrations**: 5

---

## 🔗 Links

- **Repository**: [https://github.com/Sonlux/ESCAI](https://github.com/Sonlux/ESCAI)
- **Documentation**: [https://escai-framework.readthedocs.io](https://escai-framework.readthedocs.io)
- **PyPI Package**: [https://pypi.org/project/escai-framework](https://pypi.org/project/escai-framework)
- **Docker Hub**: [https://hub.docker.com/r/escai/framework](https://hub.docker.com/r/escai/framework)
- **Issues**: [https://github.com/Sonlux/ESCAI/issues](https://github.com/Sonlux/ESCAI/issues)
- **Discussions**: [https://github.com/Sonlux/ESCAI/discussions](https://github.com/Sonlux/ESCAI/discussions)

---

**Made with ❤️ by the ESCAI Framework team**

_Empowering AI research through comprehensive agent observability_
