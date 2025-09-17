# CLI Documentation Generation System

## Overview

The CLI Documentation Generation System provides comprehensive, dynamic documentation for all ESCAI CLI commands. It generates publication-ready help content, practical examples, research guidance, and prerequisite validation.

## Components

### 1. Command Documentation Generator (`command_docs.py`)

- **How to Use** guides with syntax examples
- **When to Use** scenarios with specific use cases
- **Why to Use** explanations with research benefits
- Complete command documentation with metadata

### 2. Example Generator (`examples.py`)

- Basic usage examples with sample outputs
- Advanced examples for complex scenarios
- Research-focused examples for academic use
- Complete workflow examples for end-to-end processes

### 3. Research Guide Generator (`research_guides.py`)

- Domain-specific research guidance
- Methodology section generation for papers
- Experimental design guides
- Publication checklists and standards

### 4. Prerequisite Checker (`prerequisite_checker.py`)

- System requirement validation
- Python version and package checking
- Environment variable verification
- Comprehensive system diagnostics

### 5. Documentation Integrator (`doc_integration.py`)

- Unified interface for all documentation components
- Command readiness checking
- Publication guide generation
- Comprehensive documentation packages

## Key Features

### Dynamic Content Generation

- Context-aware help text based on command and user needs
- Real-time prerequisite checking with detailed diagnostics
- Adaptive examples based on user experience level

### Research-Grade Documentation

- Publication-ready methodology sections
- Statistical reporting standards
- Reproducibility guidelines
- Citation information and templates

### Comprehensive Coverage

- All CLI commands documented with multiple detail levels
- Prerequisites validated with installation guidance
- Troubleshooting guides with common solutions
- Integration examples for complete workflows

## Usage Examples

### Basic Help Generation

```python
from escai_framework.cli.documentation import DocumentationIntegrator

integrator = DocumentationIntegrator()

# Generate how-to-use guide
help_text = integrator.generate_help_text("monitor", "basic")

# Check command readiness
readiness = integrator.check_command_readiness("monitor")

# Get comprehensive documentation
docs = integrator.get_comprehensive_documentation("monitor")
```

### Research Documentation

```python
# Generate publication guide
pub_guide = integrator.generate_publication_guide(["monitor", "analyze"])

# Get research guide for specific domain
from escai_framework.cli.documentation import ResearchDomain
research_guide = integrator.research_generator.generate_research_guide(
    ResearchDomain.BEHAVIORAL_ANALYSIS
)
```

### Prerequisite Validation

```python
# Check prerequisites for a command
prereq_result = integrator.prereq_checker.check_command_prerequisites("analyze")

# Get system diagnostics
diagnostics = integrator.prereq_checker.get_system_diagnostics()

# Generate prerequisite documentation
prereq_docs = integrator.prereq_checker.generate_prerequisite_documentation("monitor")
```

## Testing

The system includes comprehensive unit tests covering:

- Documentation generation for valid and invalid commands
- Example generation with sample outputs
- Research guide generation for all domains
- Prerequisite checking with mocked system conditions
- Integration testing for complete workflows

Run tests with:

```bash
python -m pytest tests/unit/test_cli_documentation.py -v
```

## Integration

The documentation system integrates with:

- CLI command implementations for context-aware help
- Session management for reproducibility documentation
- Error handling for troubleshooting guides
- Configuration management for setup documentation

## Benefits

### For Researchers

- Publication-ready methodology sections
- Statistical reporting standards
- Reproducibility guidelines
- Research domain-specific guidance

### For Developers

- Comprehensive API documentation
- Integration examples
- Troubleshooting guides
- System requirement validation

### For Users

- Clear usage instructions
- Practical examples with outputs
- Step-by-step setup guides
- Context-sensitive help

## Future Enhancements

- Interactive documentation with guided tutorials
- Video example generation
- Multi-language documentation support
- Community-contributed examples
- AI-powered documentation updates
