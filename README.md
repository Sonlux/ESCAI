# ESCAI Framework

**Epistemic State and Causal Analysis Intelligence Framework**

A comprehensive observability system for monitoring autonomous agent cognition in real-time. The ESCAI framework provides deep insights into how AI agents think, decide, and behave during task execution by tracking epistemic states, analyzing behavioral patterns, discovering causal relationships, and predicting performance outcomes.

## Features

- **Real-time Epistemic State Monitoring**: Track agent beliefs, knowledge, and goals as they evolve
- **Multi-Framework Support**: Compatible with LangChain, AutoGen, CrewAI, and OpenAI Assistants
- **Behavioral Pattern Analysis**: Identify and analyze patterns in agent decision-making
- **Causal Relationship Discovery**: Understand cause-effect relationships in agent behavior
- **Performance Prediction**: Forecast task outcomes and identify potential failures early
- **Human-Readable Explanations**: Generate natural language explanations of agent behavior

## Installation

### From Source

```bash
git clone https://github.com/escai-framework/escai.git
cd escai
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/escai-framework/escai.git
cd escai
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, BeliefType
from datetime import datetime

# Create a belief state
belief = BeliefState(
    content="The user wants to classify images",
    belief_type=BeliefType.FACTUAL,
    confidence=0.9,
    evidence=["user input", "task context"]
)

# Create an epistemic state
epistemic_state = EpistemicState(
    agent_id="image_classifier_agent",
    timestamp=datetime.utcnow(),
    belief_states=[belief],
    confidence_level=0.85,
    uncertainty_score=0.15
)

# Validate and serialize
if epistemic_state.validate():
    json_data = epistemic_state.to_json()
    print("Epistemic state captured successfully")
```

### Behavioral Pattern Analysis

```python
from escai_framework.models.behavioral_pattern import (
    BehavioralPattern, ExecutionSequence, ExecutionStep,
    PatternType, ExecutionStatus
)

# Create execution steps
step1 = ExecutionStep(
    step_id="step_001",
    action="load_model",
    timestamp=datetime.utcnow(),
    duration_ms=1500,
    status=ExecutionStatus.SUCCESS
)

step2 = ExecutionStep(
    step_id="step_002",
    action="process_image",
    timestamp=datetime.utcnow(),
    duration_ms=800,
    status=ExecutionStatus.SUCCESS
)

# Create execution sequence
sequence = ExecutionSequence(
    sequence_id="seq_001",
    agent_id="image_classifier",
    task_description="Image classification task",
    steps=[step1, step2]
)

# Create behavioral pattern
pattern = BehavioralPattern(
    pattern_id="classification_pattern",
    pattern_name="Standard Classification",
    pattern_type=PatternType.SEQUENTIAL,
    description="Standard image classification workflow",
    execution_sequences=[sequence]
)

pattern.calculate_statistics()
print(f"Pattern success rate: {pattern.success_rate}")
```

### Causal Relationship Analysis

```python
from escai_framework.models.causal_relationship import (
    CausalRelationship, CausalEvent, CausalEvidence,
    CausalType, EvidenceType
)

# Create cause and effect events
cause_event = CausalEvent(
    event_id="cause_001",
    event_type="parameter_change",
    description="Learning rate was reduced",
    timestamp=datetime.utcnow(),
    agent_id="training_agent"
)

effect_event = CausalEvent(
    event_id="effect_001",
    event_type="performance_improvement",
    description="Model accuracy increased",
    timestamp=datetime.utcnow(),
    agent_id="training_agent"
)

# Create evidence
evidence = CausalEvidence(
    evidence_type=EvidenceType.STATISTICAL,
    description="Strong correlation observed",
    strength=0.85,
    confidence=0.92,
    source="correlation_analysis"
)

# Create causal relationship
relationship = CausalRelationship(
    relationship_id="rel_001",
    cause_event=cause_event,
    effect_event=effect_event,
    causal_type=CausalType.DIRECT,
    strength=0.8,
    confidence=0.9,
    delay_ms=2000,
    evidence=[evidence]
)

print(f"Causal strength: {relationship.strength}")
```

### Performance Prediction

```python
from escai_framework.models.prediction_result import (
    PredictionResult, RiskFactor, Intervention,
    PredictionType, RiskLevel, InterventionType
)

# Create risk factor
risk_factor = RiskFactor(
    factor_id="risk_001",
    name="High Task Complexity",
    description="Task complexity exceeds normal parameters",
    impact_score=0.7,
    probability=0.6,
    category="task_complexity"
)

# Create intervention
intervention = Intervention(
    intervention_id="int_001",
    intervention_type=InterventionType.PARAMETER_ADJUSTMENT,
    name="Reduce Batch Size",
    description="Reduce processing batch size to improve stability",
    expected_impact=0.6,
    implementation_cost=0.2,
    urgency=RiskLevel.MEDIUM
)

# Create prediction result
prediction = PredictionResult(
    prediction_id="pred_001",
    agent_id="processing_agent",
    prediction_type=PredictionType.SUCCESS_PROBABILITY,
    predicted_value=0.75,
    confidence_score=0.88,
    risk_factors=[risk_factor],
    recommended_interventions=[intervention]
)

print(f"Predicted success probability: {prediction.predicted_value}")
print(f"Overall risk score: {prediction.calculate_overall_risk_score()}")
```

## Architecture

The ESCAI framework follows a modular architecture with the following key components:

- **Models**: Core data structures for epistemic states, behavioral patterns, causal relationships, and predictions
- **Instrumentation**: Framework-specific adapters for monitoring different agent systems
- **Core Processing**: Engines for extracting insights and performing analysis
- **Analytics**: Machine learning models and statistical analysis tools
- **API**: REST and WebSocket interfaces for real-time access
- **Storage**: Multi-database architecture for different data types
- **Visualization**: Dashboard and reporting components

## Data Models

### EpistemicState

Represents an agent's complete epistemic state including beliefs, knowledge, goals, and confidence levels.

### BehavioralPattern

Captures recurring patterns in agent execution sequences with statistical analysis.

### CausalRelationship

Models cause-effect relationships between events with evidence and confidence measures.

### PredictionResult

Contains performance predictions with risk factors and recommended interventions.

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=escai_framework --cov-report=html

# Run specific test file
pytest tests/unit/test_epistemic_state.py

# Run with verbose output
pytest -v
```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/escai-framework/escai.git
cd escai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

Run quality checks:

```bash
# Format code
black escai_framework tests

# Sort imports
isort escai_framework tests

# Lint code
flake8 escai_framework tests

# Type checking
mypy escai_framework

# Run all tests
pytest
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use the ESCAI framework in your research, please cite:

```bibtex
@software{escai_framework,
  title={ESCAI Framework: Epistemic State and Causal Analysis Intelligence},
  author={ESCAI Framework Team},
  year={2024},
  url={https://github.com/escai-framework/escai}
}
```

## Support

- **Documentation**: [https://escai-framework.readthedocs.io](https://escai-framework.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/escai-framework/escai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/escai-framework/escai/discussions)

## Roadmap

- [ ] Complete core processing engines
- [ ] Framework-specific instrumentors
- [ ] REST API implementation
- [ ] WebSocket real-time interface
- [ ] Machine learning models
- [ ] Visualization dashboard
- [ ] Production deployment tools
- [ ] Performance optimization
- [ ] Extended documentation
- [ ] Community examples
