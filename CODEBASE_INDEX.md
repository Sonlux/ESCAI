# ESCAI Framework - Codebase Index 📁

This document provides a comprehensive index of the ESCAI Framework codebase, organized by functionality and purpose.

---

## 📋 Table of Contents

- [🏗️ Core Architecture](#️-core-architecture)
- [🔧 Framework Components](#-framework-components)
- [💻 CLI System](#-cli-system)
- [🌐 API & Web Interface](#-api--web-interface)
- [🗄️ Storage & Data](#️-storage--data)
- [📊 Analytics & ML](#-analytics--ml)
- [🔒 Security & Auth](#-security--auth)
- [⚙️ Configuration](#️-configuration)
- [🧪 Testing Suite](#-testing-suite)
- [📚 Documentation](#-documentation)
- [🚀 Deployment](#-deployment)
- [🛠️ Development Tools](#️-development-tools)
- [📦 Examples & Demos](#-examples--demos)

---

## 🏗️ Core Architecture

### **Core Processing Engines**

```
escai_framework/core/
├── epistemic_extractor.py      # Extract agent beliefs, knowledge, goals
├── pattern_analyzer.py         # Behavioral pattern mining and analysis
├── causal_engine.py           # Causal relationship discovery
├── performance_predictor.py    # Performance prediction models
└── explanation_engine.py       # Human-readable explanations
```

**Key Features:**

- Real-time epistemic state extraction
- Statistical pattern analysis with significance testing
- Causal inference using advanced algorithms
- ML-based performance prediction
- Natural language explanation generation

### **Data Models**

```
escai_framework/models/
├── epistemic_state.py         # Agent cognitive state representation
├── behavioral_pattern.py      # Behavioral pattern data structures
├── causal_relationship.py     # Causal relationship models
└── prediction_result.py       # Prediction outcome models
```

### **Shared Utilities**

```
escai_framework/utils/
├── exceptions.py              # Custom exception classes
├── validation.py              # Data validation utilities
├── serialization.py           # Data serialization helpers
├── retry.py                   # Retry mechanisms
├── circuit_breaker.py         # Circuit breaker pattern
├── fallback.py               # Fallback strategies
├── load_shedding.py          # Load shedding mechanisms
└── error_tracking.py         # Error tracking and reporting
```

---

## 🔧 Framework Components

### **Instrumentation Layer**

```
escai_framework/instrumentation/
├── base_instrumentor.py       # Base instrumentation interface
├── langchain_instrumentor.py  # LangChain framework integration
├── autogen_instrumentor.py    # AutoGen framework integration
├── crewai_instrumentor.py     # CrewAI framework integration
├── openai_instrumentor.py     # OpenAI Assistants integration
├── robust_instrumentor.py     # Robust instrumentation wrapper
├── adaptive_sampling.py       # Adaptive sampling strategies
├── framework_compatibility.py # Framework compatibility layer
├── event_stream.py           # Event streaming infrastructure
├── log_processor.py          # Log processing utilities
└── events.py                 # Event data structures
```

**Supported Frameworks:**

- **LangChain**: Chain execution, tool usage, memory tracking
- **AutoGen**: Multi-agent conversations, role tracking
- **CrewAI**: Task delegation, crew coordination
- **OpenAI**: Function calling, tool usage, assistant interactions

### **Framework Integration Tests**

```
tests/integration/
├── test_langchain_instrumentor.py    # LangChain integration tests
├── test_autogen_instrumentor.py      # AutoGen integration tests
├── test_crewai_instrumentor.py       # CrewAI integration tests
├── test_openai_instrumentor.py       # OpenAI integration tests
└── test_framework_robustness.py      # Robustness testing
```

---

## 💻 CLI System

### **CLI Core**

```
escai_framework/cli/
├── main.py                    # CLI entry point and main interface
└── session_storage.py         # Session management and storage
```

### **Command Groups**

```
escai_framework/cli/commands/
├── monitor.py                 # Monitoring commands (start, stop, status)
├── analyze.py                 # Analysis commands (patterns, causal, predictions)
├── config.py                  # Configuration commands (setup, show, set, get)
├── config_mgmt.py            # Advanced configuration management
└── session.py                # Session management commands
```

**Command Categories:**

- **Monitor**: 8 commands for real-time monitoring
- **Analyze**: 13 commands for data analysis
- **Config**: 8 commands for configuration management
- **Session**: 9 commands for session handling

### **CLI Utilities**

```
escai_framework/cli/utils/
├── console.py                 # Rich console configuration
├── formatters.py             # Output formatting utilities
├── interactive_menu.py       # Interactive menu system
├── interactive.py            # Interactive command helpers
├── ascii_viz.py              # ASCII visualization tools
├── data_filters.py           # Data filtering and search
├── reporting.py              # Report generation
├── productivity.py           # Productivity enhancements
├── analysis_tools.py         # Analysis utilities
├── live_monitor.py           # Live monitoring tools
├── error_handling.py         # CLI error handling
├── error_decorators.py       # Error handling decorators
├── validators.py             # Input validation
├── validation_config.py      # Validation configuration
└── validation_integration.py # Validation integration
```

### **CLI Integration**

```
escai_framework/cli/integration/
└── framework_connector.py    # Framework connection management
```

### **CLI Documentation System**

```
escai_framework/cli/documentation/
├── command_docs.py           # Command documentation generator
├── examples.py               # Example generator
├── research_guides.py        # Research guide generator
├── prerequisite_checker.py   # Prerequisite validation
└── doc_integration.py        # Documentation integration
```

### **CLI Services**

```
escai_framework/cli/services/
└── api_client.py             # API client for CLI commands
```

---

## 🌐 API & Web Interface

### **FastAPI Application**

```
escai_framework/api/
├── main.py                   # FastAPI application entry point
├── monitoring.py             # Monitoring endpoints
├── analysis.py               # Analysis endpoints
├── auth.py                   # Authentication utilities
├── auth_endpoints.py         # Authentication endpoints
├── middleware.py             # Custom middleware
└── websocket.py              # WebSocket endpoints for real-time data
```

**API Features:**

- RESTful endpoints for all core functionality
- WebSocket support for real-time updates
- Authentication and authorization
- Rate limiting and security middleware
- OpenAPI/Swagger documentation

### **API Documentation**

```
docs/api/
├── README.md                 # API overview and getting started
└── openapi.yaml              # OpenAPI specification
```

---

## 🗄️ Storage & Data

### **Database Management**

```
escai_framework/storage/
├── database.py               # Database connection management
├── models.py                 # SQLAlchemy models
└── migrations/               # Database migrations
    ├── env.py               # Alembic environment
    └── versions/            # Migration versions
        └── 001_initial_schema.py
```

### **Repository Pattern**

```
escai_framework/storage/repositories/
├── base_repository.py        # Base repository interface
├── agent_repository.py       # Agent data repository
├── epistemic_state_repository.py      # Epistemic state data
├── behavioral_pattern_repository.py   # Pattern data
├── causal_relationship_repository.py  # Causal data
├── prediction_repository.py           # Prediction data
├── monitoring_session_repository.py   # Session data
├── raw_log_repository.py              # Raw log storage
├── processed_event_repository.py      # Processed events
├── explanation_repository.py          # Explanations
├── analytics_result_repository.py     # Analytics results
└── configuration_repository.py        # Configuration data
```

### **Database-Specific Implementations**

#### **MongoDB Integration**

```
escai_framework/storage/
├── mongo_manager.py          # MongoDB connection and operations
├── mongo_models.py           # MongoDB document models
└── repositories/
    └── mongo_base_repository.py # MongoDB repository base
```

#### **Redis Integration**

```
escai_framework/storage/
└── redis_manager.py          # Redis connection and caching
```

#### **InfluxDB Integration**

```
escai_framework/storage/
├── influx_manager.py         # InfluxDB time-series management
├── influx_models.py          # InfluxDB data models
└── influx_dashboard.py       # InfluxDB dashboard integration
```

#### **Neo4j Integration**

```
escai_framework/storage/
├── neo4j_manager.py          # Neo4j graph database management
├── neo4j_models.py           # Neo4j node and relationship models
└── neo4j_analytics.py        # Graph analytics utilities
```

---

## 📊 Analytics & ML

### **Analytics Pipeline**

```
escai_framework/analytics/
├── pattern_mining.py         # Behavioral pattern mining algorithms
├── prediction_models.py      # Machine learning prediction models
├── failure_analysis.py       # Failure analysis and root cause detection
├── statistical_analysis.py   # Statistical analysis tools
└── model_evaluation.py       # Model performance evaluation
```

**Analytics Capabilities:**

- Sequential pattern mining with PrefixSpan
- Statistical significance testing
- Machine learning model training and evaluation
- Failure prediction and analysis
- Performance trend analysis

---

## 🔒 Security & Auth

### **Security Framework**

```
escai_framework/security/
├── auth_manager.py           # Authentication management
├── rbac.py                   # Role-based access control
├── audit_logger.py           # Security audit logging
├── input_validator.py        # Input validation and sanitization
├── pii_detector.py           # PII detection and masking
├── config.py                 # Security configuration
└── tls_manager.py            # TLS/SSL management
```

**Security Features:**

- JWT-based authentication
- Role-based access control (RBAC)
- Input validation and sanitization
- PII detection and masking
- Comprehensive audit logging
- TLS/SSL encryption

---

## ⚙️ Configuration

### **Configuration Management**

```
escai_framework/config/
├── config_manager.py         # Configuration management
├── config_validator.py       # Configuration validation
├── config_encryption.py      # Secure configuration storage
├── config_schema.py          # Configuration schema definitions
├── config_templates.py       # Configuration templates
└── config_versioning.py      # Configuration versioning
```

### **Configuration Files**

```
config/
└── config.example.yaml       # Example configuration file
```

---

## 🧪 Testing Suite

### **Test Infrastructure**

```
tests/
├── cli_test_runner.py        # Comprehensive test runner
├── conftest_cli.py           # CLI-specific pytest configuration
├── conftest.py               # General pytest configuration
├── run_comprehensive_tests.py # Comprehensive test execution
└── CLI_TEST_SUITE_SUMMARY.md # Test suite documentation
```

### **Test Categories**

#### **Unit Tests (50+ tests)**

```
tests/unit/
├── test_cli_commands.py      # CLI command implementations
├── test_cli_console.py       # CLI console utilities
├── test_cli_formatters.py    # CLI output formatting
├── test_cli_documentation.py # CLI documentation system
├── test_cli_error_handling.py # CLI error handling
├── test_cli_validators.py    # CLI input validation
├── test_cli_visualization.py # CLI visualization tools
├── test_epistemic_extractor.py # Epistemic state extraction
├── test_pattern_analyzer.py  # Pattern analysis
├── test_causal_engine.py     # Causal analysis
├── test_performance_predictor.py # Performance prediction
├── test_explanation_engine.py # Explanation generation
└── [30+ more unit test files]
```

#### **Integration Tests (30+ tests)**

```
tests/integration/
├── test_cli_framework_integration.py # Framework integration
├── test_cli_integration.py          # General CLI integration
├── test_cli_enhancements.py         # CLI enhancement features
├── test_cli_error_handling_integration.py # Error handling
├── test_cli_validation_integration.py # Validation integration
├── test_framework_integration.py     # Framework integration
├── test_framework_robustness.py      # Framework robustness
├── test_langchain_instrumentor.py    # LangChain integration
├── test_autogen_instrumentor.py      # AutoGen integration
├── test_crewai_instrumentor.py       # CrewAI integration
├── test_openai_instrumentor.py       # OpenAI integration
└── [20+ more integration test files]
```

#### **End-to-End Tests (15+ tests)**

```
tests/e2e/
├── test_cli_workflows.py     # Complete CLI workflows
├── test_basic_workflow.py    # Basic system workflows
└── test_complete_workflows.py # Complex workflow scenarios
```

#### **Performance Tests (15+ tests)**

```
tests/performance/
├── test_cli_performance.py   # CLI performance with large datasets
├── test_basic_performance.py # Basic performance benchmarks
└── test_monitoring_overhead.py # Monitoring overhead testing
```

#### **User Experience Tests (25+ tests)**

```
tests/ux/
└── test_cli_user_experience.py # CLI usability and UX testing
```

#### **Documentation Tests (20+ tests)**

```
tests/documentation/
└── test_cli_documentation_quality.py # Documentation accuracy
```

#### **Accessibility Tests (20+ tests)**

```
tests/accessibility/
└── test_cli_accessibility.py # Screen reader compatibility
```

#### **Specialized Tests**

```
tests/accuracy/
├── test_basic_accuracy.py    # Basic accuracy testing
└── test_ml_model_accuracy.py # ML model accuracy

tests/load/
├── test_basic_load.py        # Basic load testing
└── test_concurrent_monitoring.py # Concurrent load testing

tests/utils/
├── coverage_analyzer.py     # Test coverage analysis
└── test_data_generator.py   # Test data generation
```

### **Test Configuration**

```
pytest_cli.ini               # CLI-specific pytest configuration
pyproject.toml               # Project configuration with test settings
```

---

## 📚 Documentation

### **Core Documentation**

```
docs/
├── README.md                 # Documentation overview
├── api/                      # API documentation
│   ├── README.md            # API overview
│   └── openapi.yaml         # OpenAPI specification
├── cli/                      # CLI documentation
│   ├── commands.md          # Command reference
│   ├── examples.md          # Usage examples
│   ├── documentation-system.md # Documentation system
│   └── validation-system.md # Validation system
├── integration/              # Integration guides
│   ├── README.md            # Integration overview
│   ├── langchain.md         # LangChain integration
│   ├── autogen.md           # AutoGen integration
│   └── framework-robustness.md # Framework robustness
├── deployment/               # Deployment documentation
│   ├── README.md            # Deployment overview
│   └── quick-start.md       # Quick start guide
├── security/                 # Security documentation
│   ├── README.md            # Security overview
│   └── best-practices.md    # Security best practices
├── architecture/             # Architecture documentation
│   └── README.md            # System architecture
├── performance/              # Performance documentation
│   └── README.md            # Performance overview
├── configuration/            # Configuration documentation
│   └── README.md            # Configuration guide
├── troubleshooting/          # Troubleshooting guides
│   ├── README.md            # Common issues
│   └── framework-integration.md # Framework troubleshooting
└── examples/                 # Example documentation
    ├── README.md            # Examples overview
    └── business/            # Business use cases
        └── customer_service_bot.py
```

### **Project Documentation**

```
├── README.md                 # Main project README
├── CODEBASE_INDEX.md        # This file - comprehensive codebase index
├── CHANGELOG.md             # Version history and changes
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT license
└── pyproject.toml           # Project configuration
```

---

## 🚀 Deployment

### **Docker Configuration**

```
├── Dockerfile               # Docker image configuration
├── docker-compose.yml       # Multi-container setup
└── docker/                  # Docker-related files
    └── mongo/
        └── init.js          # MongoDB initialization
```

### **Kubernetes Deployment**

```
k8s/
├── escai-api.yaml           # API service deployment
└── postgres.yaml            # PostgreSQL deployment
```

### **Helm Charts**

```
helm/escai/
├── values.yaml              # Default Helm values
├── values-production.yaml   # Production values
└── templates/
    ├── deployment.yaml      # Kubernetes deployment template
    └── _helpers.tpl         # Helm template helpers
```

### **Deployment Scripts**

```
scripts/
├── validate-deployment.py   # Deployment validation
└── test-deployment.py       # Deployment testing
```

---

## 🛠️ Development Tools

### **Development Scripts**

```
scripts/
├── run_tests.py             # Test execution script
├── validate_imports.py      # Import validation
└── [deployment scripts]
```

### **Development Configuration**

```
├── .gitignore               # Git ignore patterns
├── .gitignore.backup        # Backup of git ignore
├── alembic.ini              # Database migration configuration
├── setup.py                 # Package setup configuration
└── dev/
    └── README.md            # Development documentation
```

### **Kiro IDE Configuration**

```
.kiro/
├── steering/                # AI assistant steering rules
│   ├── tech.md             # Technology guidelines
│   └── structure.md        # Project structure guidelines
└── specs/                   # Project specifications
    ├── escai-framework/     # Main framework specs
    ├── interactive-cli-system/ # CLI system specs
    ├── security-fixes/      # Security specifications
    ├── codebase-cleanup/    # Cleanup specifications
    ├── ci-cd-workflow-cleanup/ # CI/CD specifications
    └── mypy-fixes/          # Type checking specifications
```

---

## 📦 Examples & Demos

### **Core Examples**

```
examples/
├── README.md                # Examples overview
├── basic_usage.py           # Basic framework usage
├── websocket_example.py     # WebSocket integration
├── error_handling_example.py # Error handling patterns
├── config_management_demo.py # Configuration management
└── security_example.py      # Security implementation
```

### **CLI Examples**

```
examples/
├── cli_validation_example.py      # CLI validation usage
├── cli_documentation_example.py   # CLI documentation
├── cli_error_handling_example.py  # CLI error handling
└── framework_integration_example.py # Framework integration
```

### **Storage Examples**

```
examples/
├── postgresql_storage_example.py  # PostgreSQL usage
├── mongodb_storage_example.py     # MongoDB usage
├── redis_storage_example.py       # Redis usage
├── influxdb_storage_example.py    # InfluxDB usage
└── neo4j_storage_example.py       # Neo4j usage
```

### **Core Component Examples**

```
examples/
├── explanation_engine_example.py  # Explanation generation
├── performance_prediction_example.py # Performance prediction
└── causal_analysis_example.py     # Causal analysis
```

---

## 📊 Codebase Statistics

### **File Count by Category**

- **Core Framework**: 45+ files
- **CLI System**: 25+ files
- **API & Web**: 10+ files
- **Storage & Data**: 30+ files
- **Testing**: 50+ files
- **Documentation**: 25+ files
- **Examples**: 15+ files
- **Configuration**: 20+ files

### **Lines of Code**

- **Total**: ~50,000+ lines
- **Python Code**: ~40,000+ lines
- **Documentation**: ~8,000+ lines
- **Configuration**: ~2,000+ lines

### **Test Coverage**

- **Unit Tests**: 50+ test files
- **Integration Tests**: 30+ test files
- **End-to-End Tests**: 15+ test files
- **Total Test Cases**: 175+ tests
- **Coverage**: 85%+

### **Framework Support**

- **LangChain**: Full integration with chains, tools, memory
- **AutoGen**: Multi-agent conversation tracking
- **CrewAI**: Task delegation and crew coordination
- **OpenAI**: Function calling and tool usage

### **Database Support**

- **PostgreSQL**: Structured data and relationships
- **MongoDB**: Document storage for complex data
- **Redis**: Caching and real-time data
- **InfluxDB**: Time-series metrics
- **Neo4j**: Graph-based causal relationships

---

## 🔍 Quick Navigation

### **Most Important Files**

1. **[README.md](README.md)** - Project overview and getting started
2. **[escai_framework/cli/main.py](escai_framework/cli/main.py)** - CLI entry point
3. **[escai_framework/core/](escai_framework/core/)** - Core processing engines
4. **[tests/cli_test_runner.py](tests/cli_test_runner.py)** - Comprehensive test runner
5. **[docs/](docs/)** - Complete documentation

### **Key Entry Points**

- **CLI**: `escai_framework/cli/main.py`
- **API**: `escai_framework/api/main.py`
- **Core**: `escai_framework/core/`
- **Tests**: `tests/cli_test_runner.py`
- **Config**: `escai_framework/config/config_manager.py`

### **Development Workflow**

1. **Setup**: Follow [README.md](README.md) installation guide
2. **Development**: Use files in `escai_framework/`
3. **Testing**: Run `python tests/cli_test_runner.py --all`
4. **Documentation**: Update files in `docs/`
5. **Examples**: Add to `examples/`

---

**This index is automatically maintained and reflects the current state of the ESCAI Framework codebase.**

_Last updated: 2024-12-19_
