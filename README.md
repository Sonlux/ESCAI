# ESCAI Framework

**Epistemic State and Causal Analysis Intelligence**

A comprehensive CLI framework for monitoring autonomous agent cognition in real-time, designed for research in epistemic state monitoring and causal inference. ESCAI provides deep insights into how AI agents think, decide, and behave during task execution.

## 🎓 Research Context

This framework supports research in **"Epistemic State monitoring and causal inference in autonomous agent cognition"** by providing:

- **Real-time Epistemic State Extraction**: Monitor agent beliefs, knowledge, and goals as they evolve
- **Causal Relationship Discovery**: Analyze cause-effect relationships in agent decision-making
- **Behavioral Pattern Analysis**: Identify and analyze patterns in agent execution strategies
- **Performance Prediction**: Forecast task outcomes and identify potential failure modes
- **Multi-Framework Support**: Compatible with LangChain, AutoGen, CrewAI, and OpenAI Assistants

## 🚀 Installation

### Quick Install

```bash
# Install the ESCAI framework
pip install -e .

# Install with full dependencies for research
pip install -e ".[full]"

# Verify installation
escai --version
```

### System Requirements

- **Python**: 3.10 or higher
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)
- **Storage**: 1GB free space for installation and data
- **Monitoring Overhead**: < 5% impact on agent execution (configurable)

## 🖥️ CLI Commands

The ESCAI CLI provides a comprehensive interface for epistemic state monitoring and causal analysis research.

### Core Command Groups

#### 🔍 Monitoring Commands

```bash
# Start monitoring an agent
escai monitor start --agent-id <agent-id> --framework <framework>

# View real-time agent status
escai monitor status --refresh 2

# Monitor epistemic states in real-time
escai monitor epistemic --agent-id <agent-id> --refresh 3

# Launch comprehensive monitoring dashboard
escai monitor dashboard

# Stream live agent logs with filtering
escai monitor logs --filter "error" --highlight "timeout"

# Stop monitoring sessions
escai monitor stop --session-id <session-id>
```

**Supported Frameworks:**

- `langchain` - LangChain agents and chains
- `autogen` - Multi-agent conversations
- `crewai` - Crew workflow monitoring
- `openai` - OpenAI Assistants

#### 📊 Analysis Commands

```bash
# Analyze behavioral patterns
escai analyze patterns --agent-id <agent-id> --timeframe 24h --min-frequency 5

# Interactive pattern exploration
escai analyze patterns --interactive

# Explore causal relationships
escai analyze causal --min-strength 0.7 --interactive

# Generate performance predictions
escai analyze predictions --agent-id <agent-id> --horizon 1h

# View recent agent events
escai analyze events --agent-id <agent-id> --limit 20

# Create ASCII visualizations
escai analyze visualize --chart-type heatmap --metric confidence
escai analyze visualize --chart-type scatter --metric performance

# Interactive data exploration
escai analyze interactive --agent-id <agent-id>

# Statistical analysis
escai analyze stats --field confidence --agent-id <agent-id>

# Time series analysis
escai analyze timeseries --metric performance --timeframe 7d
```

#### ⚙️ Configuration Commands

```bash
# Interactive configuration setup
escai config setup

# Show current configuration
escai config show

# Set specific configuration values
escai config set database host localhost
escai config set api port 8080

# Test database connections
escai config test

# System health check
escai config check
```

#### 📋 Session Management

```bash
# List all monitoring sessions
escai session list

# Show detailed session information
escai session show <session-id>

# Stop active sessions
escai session stop <session-id>

# Clean up old sessions
escai session cleanup --older-than 7d

# Export session data
escai session export <session-id> --format json --output session_data.json
```

## 🧪 Research Workflow

### Quick Research Session

```bash
# 1. Start monitoring
escai monitor start --agent-id research-agent --framework langchain

# 2. Real-time epistemic state monitoring
escai monitor epistemic --agent-id research-agent --refresh 2

# 3. Analyze behavioral patterns
escai analyze patterns --agent-id research-agent --timeframe 1h --interactive

# 4. Discover causal relationships
escai analyze causal --min-strength 0.7 --interactive

# 5. Export research data
escai analyze export --type all --format json --output research_data.json
```

### Advanced Features

- **Real-time Updates**: Live monitoring with automatic refresh
- **Interactive Tables**: Navigate and explore data with keyboard controls
- **ASCII Visualizations**: Charts, graphs, and progress bars in the terminal
- **Color Themes**: Multiple color schemes for different environments
- **Progress Indicators**: Real-time progress bars with ETA and rate information

## 📊 Data Export and Analysis

### Export Capabilities

```bash
# Export epistemic states
escai analyze export --type epistemic --agent-id research-agent --format json

# Export causal relationships
escai analyze export --type causal --format csv --output causal_data.csv

# Export behavioral patterns
escai analyze export --type patterns --timeframe 7d --format json

# Export complete session data
escai session export session_123 --format json --output complete_session.json
```

### Report Generation

```bash
# Generate comprehensive analysis report
escai analyze report --agent-id research-agent --timeframe 24h --format pdf

# Create custom report interactively
escai analyze custom-report

# Schedule automated reports
escai analyze schedule-report --frequency daily --format json
```

## 🔧 Configuration

### Database Setup

The framework supports multiple databases for different data types:

```bash
# Configure PostgreSQL (structured data)
escai config set postgresql host localhost
escai config set postgresql port 5432
escai config set postgresql database escai_research

# Configure MongoDB (unstructured data)
escai config set mongodb host localhost
escai config set mongodb port 27017

# Configure Redis (caching and real-time data)
escai config set redis host localhost
escai config set redis port 6379

# Test all connections
escai config test
```

### Research Configuration

```bash
# Set up for research use
escai config set research mode enabled
escai config set monitoring overhead_limit 0.05  # 5% max overhead
escai config set analysis confidence_threshold 0.7
escai config set causal min_strength 0.6
```

## 📖 Getting Help

```bash
# General help
escai --help

# Command-specific help
escai monitor --help
escai analyze --help
escai config --help

# Subcommand help
escai monitor start --help
escai analyze patterns --help
```

## 🏗️ Framework Architecture

### Core Components

- **📊 Data Models**: Formal representations of epistemic states, behavioral patterns, causal relationships, and predictions
- **🔧 Instrumentation Layer**: Non-intrusive adapters for LangChain, AutoGen, CrewAI, and OpenAI Assistants
- **⚙️ Processing Engines**: Advanced algorithms for causal inference, pattern mining, and performance prediction
- **🧠 Analytics Engine**: Statistical analysis, machine learning models, and hypothesis testing
- **🗄️ Multi-Database Storage**: PostgreSQL, MongoDB, Redis, InfluxDB, and Neo4j support
- **🖥️ CLI Research Interface**: Comprehensive command-line tools designed for academic research

### Research-Focused Design

1. **Minimal Overhead**: <5% performance impact on monitored agents
2. **Real-time Processing**: Sub-second latency for epistemic state extraction
3. **Statistical Rigor**: Built-in statistical validation and significance testing
4. **Reproducibility**: Complete audit trails and deterministic analysis
5. **Extensibility**: Plugin architecture for custom analysis methods

## 📚 Research Applications

### Academic Use Cases

1. **Epistemic State Evolution Studies**: Track how agent beliefs change over time
2. **Causal Inference Research**: Discover causal relationships in agent cognition
3. **Behavioral Pattern Analysis**: Identify recurring patterns in agent behavior
4. **Performance Prediction Models**: Develop models to predict agent success
5. **Multi-Agent Interaction Studies**: Analyze interactions between multiple agents

### Methodological Contributions

1. **Real-time Epistemic State Extraction**: Novel algorithms for extracting beliefs, knowledge, and goals
2. **Causal Discovery in Agent Systems**: Adapted causal inference methods for agent cognition
3. **Behavioral Pattern Analysis**: Sequential pattern mining techniques for agent behavior
4. **Performance Prediction**: Machine learning models using epistemic state features
5. **Multi-Framework Instrumentation**: Unified monitoring across diverse agent frameworks

## 🧪 Testing and Validation

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/          # Unit tests
python -m pytest tests/integration/   # Integration tests

# Test CLI functionality
python -m pytest tests/unit/test_cli_*.py

# Test with coverage
python -m pytest --cov=escai_framework tests/
```

## 📄 Academic Citation

If you use the ESCAI framework in your research, please cite:

```bibtex
@software{escai_framework_2024,
  title={ESCAI Framework: Epistemic State and Causal Analysis Intelligence},
  author={ESCAI Research Team},
  year={2024},
  url={https://github.com/your-repo/ESCAI},
  version={1.0.0},
  note={Framework for epistemic state monitoring and causal inference in autonomous agent cognition},
  keywords={epistemic states, causal inference, autonomous agents, cognitive monitoring}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code standards, testing requirements, and the pull request process.

---

**Ready to start your epistemic state monitoring research?**

```bash
pip install -e ".[full]"
escai config setup
escai monitor start --agent-id your-research-agent --framework langchain
```
