# ESCAI Framework

**Epistemic State and Causal Analysis Intelligence**

A comprehensive framework for monitoring autonomous agent cognition in real-time, designed for research in epistemic state monitoring and causal inference. ESCAI provides deep insights into how AI agents think, decide, and behave during task execution through advanced CLI tools and real-time analysis capabilities.

## 🎓 Research Context

This framework supports research in **"Epistemic State monitoring and causal inference in autonomous agent cognition"** by providing:

- **Real-time Epistemic State Extraction**: Monitor agent beliefs, knowledge, and goals as they evolve
- **Causal Relationship Discovery**: Analyze cause-effect relationships in agent decision-making
- **Behavioral Pattern Analysis**: Identify and analyze patterns in agent execution strategies
- **Performance Prediction**: Forecast task outcomes and identify potential failure modes
- **Multi-Framework Support**: Compatible with LangChain, AutoGen, CrewAI, and OpenAI Assistants

## 🚀 Quick Start

### Installation

```bash
# Install the ESCAI framework
pip install -e .

# Install with full dependencies for research
pip install -e ".[full]"

# Verify installation
python -m escai_framework.cli.main --version
```

### Basic Usage

```bash
# Start monitoring an agent
python -m escai_framework.cli.main monitor start --agent-id research-agent --framework langchain

# View real-time status
python -m escai_framework.cli.main monitor status

# Analyze behavioral patterns
python -m escai_framework.cli.main analyze patterns --agent-id research-agent --timeframe 24h

# Explore causal relationships
python -m escai_framework.cli.main analyze causal --interactive
```

## 🖥️ CLI Interface

The ESCAI CLI provides a comprehensive interface for epistemic state monitoring and causal analysis research. All functionality is accessible through the command line interface.

### Core Command Groups

#### 🔍 Monitoring Commands (`monitor`)

Real-time monitoring of agent cognition and epistemic states:

```bash
# Start monitoring an agent
python -m escai_framework.cli.main monitor start --agent-id <agent-id> --framework <framework>

# View real-time agent status
python -m escai_framework.cli.main monitor status --refresh 2

# Monitor epistemic states in real-time
python -m escai_framework.cli.main monitor epistemic --agent-id <agent-id> --refresh 3

# Launch comprehensive monitoring dashboard
python -m escai_framework.cli.main monitor dashboard

# Stream live agent logs with filtering
python -m escai_framework.cli.main monitor logs --filter "error" --highlight "timeout"

# Stop monitoring sessions
python -m escai_framework.cli.main monitor stop --session-id <session-id>
```

**Supported Frameworks:**

- `langchain` - LangChain agents and chains
- `autogen` - Multi-agent conversations
- `crewai` - Crew workflow monitoring
- `openai` - OpenAI Assistants

#### 📊 Analysis Commands (`analyze`)

Advanced analysis of epistemic states and causal relationships:

```bash
# Analyze behavioral patterns
python -m escai_framework.cli.main analyze patterns --agent-id <agent-id> --timeframe 24h --min-frequency 5

# Interactive pattern exploration
python -m escai_framework.cli.main analyze patterns --interactive

# Explore causal relationships
python -m escai_framework.cli.main analyze causal --min-strength 0.7 --interactive

# Generate performance predictions
python -m escai_framework.cli.main analyze predictions --agent-id <agent-id> --horizon 1h

# View recent agent events
python -m escai_framework.cli.main analyze events --agent-id <agent-id> --limit 20

# Create ASCII visualizations
python -m escai_framework.cli.main analyze visualize --chart-type heatmap --metric confidence
python -m escai_framework.cli.main analyze visualize --chart-type scatter --metric performance

# Interactive data exploration
python -m escai_framework.cli.main analyze interactive --agent-id <agent-id>

# Statistical analysis
python -m escai_framework.cli.main analyze stats --field confidence --agent-id <agent-id>

# Time series analysis
python -m escai_framework.cli.main analyze timeseries --metric performance --timeframe 7d
```

#### ⚙️ Configuration Commands (`config`)

System configuration and database setup:

```bash
# Interactive configuration setup
python -m escai_framework.cli.main config setup

# Show current configuration
python -m escai_framework.cli.main config show

# Set specific configuration values
python -m escai_framework.cli.main config set database host localhost
python -m escai_framework.cli.main config set api port 8080

# Test database connections
python -m escai_framework.cli.main config test

# System health check
python -m escai_framework.cli.main config check
```

#### 📋 Session Management (`session`)

Manage monitoring sessions:

```bash
# List all monitoring sessions
python -m escai_framework.cli.main session list

# Show detailed session information
python -m escai_framework.cli.main session show <session-id>

# Stop active sessions
python -m escai_framework.cli.main session stop <session-id>

# Clean up old sessions
python -m escai_framework.cli.main session cleanup --older-than 7d

# Export session data
python -m escai_framework.cli.main session export <session-id> --format json --output session_data.json
```

### Advanced CLI Features

#### 🎨 Rich Terminal Interface

- **Real-time Updates**: Live monitoring with automatic refresh
- **Interactive Tables**: Navigate and explore data with keyboard controls
- **ASCII Visualizations**: Charts, graphs, and progress bars in the terminal
- **Color Themes**: Multiple color schemes for different environments
- **Progress Indicators**: Real-time progress bars with ETA and rate information

#### 📈 Visualization Capabilities

```bash
# Various chart types
python -m escai_framework.cli.main analyze visualize --chart-type bar --metric confidence
python -m escai_framework.cli.main analyze visualize --chart-type line --metric performance
python -m escai_framework.cli.main analyze visualize --chart-type histogram --metric response_time
python -m escai_framework.cli.main analyze visualize --chart-type scatter --metric accuracy

# Specialized research visualizations
python -m escai_framework.cli.main analyze epistemic --agent-id <agent-id>  # Epistemic state evolution
python -m escai_framework.cli.main analyze heatmap --timeframe 24h          # Pattern frequency heatmap
python -m escai_framework.cli.main analyze tree --max-depth 5               # Causal relationship tree
python -m escai_framework.cli.main analyze timeline --agent-id <agent-id>   # Timeline visualization
```

#### 🔄 Interactive Features

- **Interactive Pattern Explorer**: Navigate through behavioral patterns with detailed analysis
- **Causal Relationship Explorer**: Explore cause-effect relationships interactively
- **Live Dashboard**: Real-time monitoring dashboard with multiple metrics
- **Advanced Search**: Complex filtering and search capabilities
- **Report Generation**: Automated and custom report creation

## 🧪 Research Examples

### Quick Research Workflow

```bash
# 1. Start monitoring
python -m escai_framework.cli.main monitor start --agent-id research-agent --framework langchain

# 2. Real-time epistemic state monitoring
python -m escai_framework.cli.main monitor epistemic --agent-id research-agent --refresh 2

# 3. Analyze behavioral patterns
python -m escai_framework.cli.main analyze patterns --agent-id research-agent --timeframe 1h --interactive

# 4. Discover causal relationships
python -m escai_framework.cli.main analyze causal --min-strength 0.7 --interactive

# 5. Export research data
python -m escai_framework.cli.main analyze export --type all --format json --output research_data.json
```

### Epistemic State Monitoring

Monitor how an agent's beliefs and knowledge evolve during task execution:

```bash
# Start monitoring epistemic states
python -m escai_framework.cli.main monitor start --agent-id research-agent --framework langchain

# View real-time epistemic state changes
python -m escai_framework.cli.main monitor epistemic --agent-id research-agent --refresh 1

# Analyze epistemic state evolution over time
python -m escai_framework.cli.main analyze epistemic --agent-id research-agent --timeframe 1h
```

### Causal Relationship Discovery

Discover causal relationships in agent decision-making:

```bash
# Analyze causal relationships with interactive exploration
python -m escai_framework.cli.main analyze causal --interactive --min-strength 0.6

# Visualize causal network
python -m escai_framework.cli.main analyze causal-network --agent-id research-agent

# Export causal relationships for further analysis
python -m escai_framework.cli.main analyze export --type causal --format json --output causal_data.json
```

### Behavioral Pattern Analysis

Identify patterns in agent behavior and decision-making:

```bash
# Comprehensive pattern analysis
python -m escai_framework.cli.main analyze pattern-analysis --agent-id research-agent --timeframe 24h

# Interactive pattern exploration
python -m escai_framework.cli.main analyze patterns --interactive --min-frequency 3

# Generate pattern frequency heatmap
python -m escai_framework.cli.main analyze heatmap --metric pattern_frequency --timeframe 7d
```

### Performance Prediction

Predict agent performance and identify potential failures:

```bash
# Generate performance predictions
python -m escai_framework.cli.main analyze predictions --agent-id research-agent --horizon 2h

# View prediction trends
python -m escai_framework.cli.main analyze prediction-trends --timeframe 7d

# Analyze prediction accuracy
python -m escai_framework.cli.main analyze stats --field prediction_accuracy --timeframe 30d
```

### 📚 Comprehensive Research Examples

For detailed research workflows, advanced analysis techniques, and complete experimental setups, see [CLI_RESEARCH_EXAMPLES.md](CLI_RESEARCH_EXAMPLES.md) which includes:

- **Complete Research Sessions**: End-to-end workflows for epistemic state studies
- **Advanced Causal Analysis**: Temporal causality, intervention analysis, and Granger causality
- **Multi-Agent Studies**: Comparative analysis and longitudinal studies
- **Statistical Validation**: Significance testing and model validation
- **Data Export**: Research-ready data formats and automated reporting

## 🏗️ Technical Architecture

### System Overview

The ESCAI framework implements a layered architecture designed for minimal overhead monitoring and real-time analysis of autonomous agent cognition:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Research Interface                    │
├─────────────────────────────────────────────────────────────┤
│  Analysis Engine  │  Visualization  │  Export & Reporting   │
├─────────────────────────────────────────────────────────────┤
│ Causal Inference │ Pattern Mining  │ Performance Prediction │
├─────────────────────────────────────────────────────────────┤
│           Epistemic State Extraction Engine                 │
├─────────────────────────────────────────────────────────────┤
│  LangChain  │  AutoGen  │  CrewAI  │  OpenAI Assistants    │
├─────────────────────────────────────────────────────────────┤
│                Multi-Database Storage Layer                 │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **📊 Data Models**: Formal representations of epistemic states, behavioral patterns, causal relationships, and predictions with statistical validation
- **🔧 Instrumentation Layer**: Non-intrusive adapters for LangChain, AutoGen, CrewAI, and OpenAI Assistants with <5% performance overhead
- **⚙️ Processing Engines**: Advanced algorithms for causal inference (Granger causality, temporal analysis), pattern mining (sequential patterns, clustering), and performance prediction (ML models, risk assessment)
- **🧠 Analytics Engine**: Statistical analysis, machine learning models, and hypothesis testing for research validation
- **🗄️ Multi-Database Storage**: Optimized storage architecture using PostgreSQL (structured data), MongoDB (unstructured), Redis (real-time), InfluxDB (time-series), and Neo4j (graph relationships)
- **🖥️ CLI Research Interface**: Comprehensive command-line tools designed for academic research workflows

### Research-Focused Design Principles

1. **Minimal Overhead**: <5% performance impact on monitored agents
2. **Real-time Processing**: Sub-second latency for epistemic state extraction
3. **Statistical Rigor**: Built-in statistical validation and significance testing
4. **Reproducibility**: Complete audit trails and deterministic analysis
5. **Extensibility**: Plugin architecture for custom analysis methods

### Formal Data Models

#### Epistemic State Representation

The framework uses a formal model for representing agent epistemic states:

```python
class EpistemicState:
    """
    Formal representation of an agent's epistemic state at time t.

    Based on the BDI (Belief-Desire-Intention) model extended with
    uncertainty quantification and evidence tracking.
    """
    agent_id: str
    timestamp: datetime
    belief_states: List[BeliefState]
    knowledge_base: KnowledgeBase
    goals: List[Goal]
    confidence_level: float  # [0,1] overall confidence
    uncertainty_score: float  # [0,1] epistemic uncertainty

    # Research extensions
    evidence_quality: float
    belief_coherence: float
    goal_alignment: float

# Example epistemic state for research
{
    "agent_id": "research-agent-001",
    "timestamp": "2024-01-15T10:30:00.123Z",
    "belief_states": [
        {
            "content": "Task requires image classification with 95% accuracy",
            "confidence": 0.92,
            "evidence": ["user_specification", "historical_performance"],
            "evidence_quality": 0.88,
            "formation_time": "2024-01-15T10:29:45.000Z"
        }
    ],
    "knowledge_base": {
        "facts": ["CNN architectures achieve >90% on ImageNet"],
        "rules": ["IF accuracy_requirement > 0.9 THEN use_pretrained_model"],
        "uncertainty": 0.12
    },
    "goals": ["achieve_target_accuracy", "minimize_inference_time"],
    "confidence_level": 0.89,
    "uncertainty_score": 0.11,
    "belief_coherence": 0.94,
    "goal_alignment": 0.87
}
```

#### Causal Relationship Model

Causal relationships are represented with statistical validation:

```python
class CausalRelationship:
    """
    Statistically validated causal relationship between agent events.

    Uses Granger causality testing and temporal analysis for validation.
    """
    cause_event: str
    effect_event: str
    causal_strength: float  # [0,1] strength of causal relationship
    temporal_lag: float     # seconds between cause and effect
    statistical_significance: float  # p-value from statistical test
    evidence_count: int     # number of supporting observations
    confidence_interval: Tuple[float, float]  # 95% CI for strength

    # Research metadata
    discovery_method: str   # "granger", "temporal", "intervention"
    validation_status: str  # "validated", "tentative", "rejected"

# Example causal relationship for research
{
    "cause_event": "high_confidence_belief_formation",
    "effect_event": "rapid_decision_execution",
    "causal_strength": 0.78,
    "temporal_lag": 0.45,  # 450ms average lag
    "statistical_significance": 0.003,  # p < 0.01
    "evidence_count": 23,
    "confidence_interval": [0.65, 0.91],
    "discovery_method": "granger_causality",
    "validation_status": "validated",
    "effect_size": "large",  # Cohen's d = 1.2
    "robustness_score": 0.85
}
```

#### Behavioral Pattern Schema

```python
class BehavioralPattern:
    """
    Recurring pattern in agent behavior with statistical significance.
    """
    pattern_id: str
    pattern_type: str  # "decision", "execution", "error_recovery"
    sequence: List[str]  # ordered sequence of events
    frequency: int
    support: float  # statistical support [0,1]
    confidence: float  # pattern confidence [0,1]
    temporal_characteristics: Dict

    # Research metrics
    statistical_significance: float
    effect_size: float
    generalizability_score: float
```

## 🔧 Configuration

### Database Setup

The framework supports multiple databases for different data types:

```bash
# Configure PostgreSQL (structured data)
python -m escai_framework.cli.main config set postgresql host localhost
python -m escai_framework.cli.main config set postgresql port 5432
python -m escai_framework.cli.main config set postgresql database escai_research

# Configure MongoDB (unstructured data)
python -m escai_framework.cli.main config set mongodb host localhost
python -m escai_framework.cli.main config set mongodb port 27017

# Configure Redis (caching and real-time data)
python -m escai_framework.cli.main config set redis host localhost
python -m escai_framework.cli.main config set redis port 6379

# Test all connections
python -m escai_framework.cli.main config test
```

### Research Configuration

```bash
# Set up for research use
python -m escai_framework.cli.main config set research mode enabled
python -m escai_framework.cli.main config set monitoring overhead_limit 0.05  # 5% max overhead
python -m escai_framework.cli.main config set analysis confidence_threshold 0.7
python -m escai_framework.cli.main config set causal min_strength 0.6
```

## 📊 Data Export and Analysis

### Export Capabilities

```bash
# Export epistemic states
python -m escai_framework.cli.main analyze export --type epistemic --agent-id research-agent --format json

# Export causal relationships
python -m escai_framework.cli.main analyze export --type causal --format csv --output causal_data.csv

# Export behavioral patterns
python -m escai_framework.cli.main analyze export --type patterns --timeframe 7d --format json

# Export complete session data
python -m escai_framework.cli.main session export session_123 --format json --output complete_session.json
```

### Report Generation

```bash
# Generate comprehensive analysis report
python -m escai_framework.cli.main analyze report --agent-id research-agent --timeframe 24h --format pdf

# Create custom report interactively
python -m escai_framework.cli.main analyze custom-report

# Schedule automated reports
python -m escai_framework.cli.main analyze schedule-report --frequency daily --format json
```

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

### Validation Examples

```bash
# Validate epistemic state extraction
python examples/basic_usage.py

# Test framework integration
python examples/langchain_example.py
python examples/autogen_example.py
```

## 📚 Research Applications

### Academic Use Cases

1. **Epistemic State Evolution Studies**: Track how agent beliefs change over time
2. **Causal Inference Research**: Discover causal relationships in agent cognition
3. **Behavioral Pattern Analysis**: Identify recurring patterns in agent behavior
4. **Performance Prediction Models**: Develop models to predict agent success
5. **Multi-Agent Interaction Studies**: Analyze interactions between multiple agents

### Research Workflow

```bash
# 1. Start monitoring
python -m escai_framework.cli.main monitor start --agent-id study-agent --framework langchain

# 2. Collect data during agent execution
python -m escai_framework.cli.main monitor status --refresh 5

# 3. Analyze collected data
python -m escai_framework.cli.main analyze patterns --agent-id study-agent --interactive

# 4. Discover causal relationships
python -m escai_framework.cli.main analyze causal --min-strength 0.7 --interactive

# 5. Export results for publication
python -m escai_framework.cli.main analyze export --type all --format json --output research_data.json
```

## 🎯 Performance and Monitoring

### System Requirements

- **Python**: 3.10 or higher
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)
- **Storage**: 1GB free space for installation and data
- **Monitoring Overhead**: < 5% impact on agent execution (configurable)

### Performance Monitoring

```bash
# Monitor system performance
python -m escai_framework.cli.main analyze health --refresh 5

# Check monitoring overhead
python -m escai_framework.cli.main config check --verbose

# Performance statistics
python -m escai_framework.cli.main analyze stats --field monitoring_overhead --timeframe 24h
```

## 📖 Getting Help

```bash
# General help
python -m escai_framework.cli.main --help

# Command-specific help
python -m escai_framework.cli.main monitor --help
python -m escai_framework.cli.main analyze --help
python -m escai_framework.cli.main config --help

# Subcommand help
python -m escai_framework.cli.main monitor start --help
python -m escai_framework.cli.main analyze patterns --help
```

## 📄 Academic Citation

If you use the ESCAI framework in your research, please cite:

```bibtex
@software{escai_framework_2024,
  title={ESCAI Framework: Epistemic State and Causal Analysis Intelligence for Autonomous Agent Cognition},
  author={ESCAI Research Team},
  year={2024},
  url={https://github.com/your-repo/ESCAI},
  version={1.0.0},
  note={A comprehensive framework for real-time epistemic state monitoring and causal inference in autonomous agent systems},
  keywords={epistemic states, causal inference, autonomous agents, cognitive monitoring, AI observability}
}
```

### Research Paper Context

This framework directly supports research in:

**"Epistemic State monitoring and causal inference in autonomous agent cognition"**

**Abstract**: The ESCAI framework provides a comprehensive solution for monitoring and analyzing the cognitive processes of autonomous agents in real-time. By tracking epistemic states (beliefs, knowledge, goals) and discovering causal relationships in agent behavior, researchers can gain unprecedented insights into how AI agents think, decide, and adapt during task execution.

**Key Research Contributions**:

- Real-time epistemic state extraction and monitoring methodology
- Causal inference algorithms for discovering cause-effect relationships in agent cognition
- Behavioral pattern analysis techniques for identifying recurring decision-making strategies
- Performance prediction models based on epistemic state evolution
- Multi-framework instrumentation supporting diverse agent architectures

**Methodology**: The framework employs a multi-layered approach combining:

1. **Instrumentation Layer**: Non-intrusive monitoring of agent frameworks
2. **Epistemic Extraction**: Real-time extraction of beliefs, knowledge, and goals
3. **Causal Analysis**: Temporal analysis and Granger causality testing
4. **Pattern Mining**: Sequential pattern discovery and statistical analysis
5. **Predictive Modeling**: Machine learning models for performance forecasting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Code standards and formatting
- Testing requirements
- Pull request process
- Issue reporting

## 🔗 Related Work

This framework supports research in:

- Autonomous agent cognition
- Epistemic state modeling
- Causal inference in AI systems
- Multi-agent system analysis
- AI explainability and interpretability

## 🔬 Research Applications and Related Work

### Primary Research Applications

**Epistemic State Monitoring Research**:

- Real-time tracking of agent belief formation and evolution
- Quantitative analysis of epistemic uncertainty in autonomous systems
- Comparative studies of epistemic state dynamics across different agent architectures

**Causal Inference in AI Systems**:

- Discovery of causal relationships in agent decision-making processes
- Temporal causality analysis using Granger causality testing
- Intervention analysis and counterfactual reasoning in agent behavior

**Cognitive Architecture Analysis**:

- Behavioral pattern mining in autonomous agent execution
- Performance prediction based on epistemic state evolution
- Multi-agent interaction and emergent behavior analysis

### Related Research Areas

This framework contributes to and builds upon research in:

- **Autonomous Agent Cognition**: BDI models, cognitive architectures, agent reasoning
- **Epistemic Logic and Uncertainty**: Belief revision, epistemic uncertainty quantification
- **Causal Inference**: Granger causality, temporal analysis, intervention analysis
- **AI Observability**: Monitoring, debugging, and explaining AI system behavior
- **Multi-Agent Systems**: Interaction analysis, emergent behavior, collective intelligence
- **AI Safety and Alignment**: Understanding and predicting agent behavior for safety assurance

### Methodological Contributions

1. **Real-time Epistemic State Extraction**: Novel algorithms for extracting beliefs, knowledge, and goals from agent execution traces
2. **Causal Discovery in Agent Systems**: Adapted causal inference methods for the unique characteristics of agent cognition
3. **Behavioral Pattern Analysis**: Sequential pattern mining techniques optimized for agent behavior analysis
4. **Performance Prediction**: Machine learning models that use epistemic state features for predicting agent performance
5. **Multi-Framework Instrumentation**: Unified monitoring approach across diverse agent frameworks

### Validation and Reproducibility

The framework includes comprehensive validation mechanisms:

- **Statistical Validation**: Built-in significance testing and confidence intervals
- **Reproducibility Tools**: Complete audit trails and deterministic analysis
- **Cross-Validation**: Methods for validating discovered patterns and causal relationships
- **Benchmark Datasets**: Standardized test cases for comparing analysis methods

---

**Ready to start your epistemic state monitoring research?**

```bash
pip install -e ".[full]"
python -m escai_framework.cli.main config setup
python -m escai_framework.cli.main monitor start --agent-id your-research-agent --framework langchain
```
