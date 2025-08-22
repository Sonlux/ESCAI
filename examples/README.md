# ESCAI Framework Examples

This directory contains example scripts demonstrating how to use the ESCAI Framework for monitoring and analyzing autonomous agent behavior.

## Available Examples

### `basic_usage.py`

Comprehensive example showing how to use all core data models:

- **Epistemic State Monitoring**: Track agent beliefs, knowledge, and goals
- **Behavioral Pattern Analysis**: Analyze execution sequences and identify patterns
- **Causal Relationship Discovery**: Model cause-effect relationships with evidence
- **Performance Prediction**: Forecast outcomes with risk analysis and interventions

Run the example:

```bash
python examples/basic_usage.py
```

### `causal_analysis_example.py`

Advanced example demonstrating the causal inference capabilities:

- **Temporal Causality Detection**: Discover cause-effect relationships from event sequences
- **Granger Causality Testing**: Statistical testing for time series causality
- **Intervention Analysis**: Estimate effects of hypothetical interventions
- **Causal Graph Construction**: Build and analyze causal relationship networks

Run the example:

```bash
python examples/causal_analysis_example.py
```

## Example Output

The basic usage example will demonstrate:

1. **Epistemic State**: Creating and monitoring an image classification agent's beliefs about task requirements, knowledge about the dataset, and goals for achieving target accuracy.

2. **Behavioral Pattern**: Analyzing a standard training workflow with steps for data loading, preprocessing, model initialization, and training.

3. **Causal Relationship**: Discovering the relationship between learning rate reduction and accuracy improvement with statistical evidence.

4. **Performance Prediction**: Predicting success probability with risk factors and recommended interventions.

## Running Examples

### Prerequisites

Make sure you have the ESCAI Framework installed:

```bash
# Install from source
pip install -e .

# Or install development version
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Run the comprehensive example
python examples/basic_usage.py

# Run the causal analysis example
python examples/causal_analysis_example.py

# Run with Python module syntax
python -m examples.basic_usage
python -m examples.causal_analysis_example
```

## Example Use Cases

These examples demonstrate common patterns for:

- **AI Research**: Monitoring experimental agent behavior
- **Production Systems**: Real-time agent performance tracking
- **Debugging**: Understanding why agents make certain decisions
- **Optimization**: Identifying improvement opportunities through causal analysis
- **Risk Management**: Predicting and mitigating potential failures

## Next Steps

After running these examples, you can:

1. **Explore the API**: Check the full API documentation
2. **Integrate with Your Agents**: Use the instrumentation layer for your specific agent framework
3. **Build Custom Analytics**: Extend the framework with your own analysis methods
4. **Create Dashboards**: Use the visualization components for real-time monitoring

## Contributing Examples

We welcome contributions of new examples! Please:

1. Follow the existing code style and structure
2. Include comprehensive docstrings and comments
3. Add a description to this README
4. Test your example thoroughly
5. Submit a pull request

Example categories we'd love to see:

- Framework-specific integrations (LangChain, AutoGen, CrewAI)
- Real-world use cases and applications
- Advanced analytics and visualization
- Performance optimization techniques
- Custom instrumentation patterns
