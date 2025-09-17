"""
Documentation integration module for CLI commands.

This module integrates all documentation components and provides
a unified interface for accessing comprehensive command documentation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import textwrap

from .command_docs import DocumentationGenerator, CommandDocumentation
from .examples import ExampleGenerator, CommandExample
from .research_guides import ResearchGuideGenerator, ResearchDomain, ResearchGuide
from .prerequisite_checker import PrerequisiteChecker, PrerequisiteCheckResult


@dataclass
class ComprehensiveDocumentation:
    """Complete documentation package for a command."""
    command: str
    basic_docs: CommandDocumentation
    examples: List[CommandExample]
    research_guide: Optional[ResearchGuide]
    prerequisites: PrerequisiteCheckResult
    quick_start: str
    troubleshooting: str


class DocumentationIntegrator:
    """Integrates all documentation components for unified access."""
    
    def __init__(self):
        self.doc_generator = DocumentationGenerator()
        self.example_generator = ExampleGenerator()
        self.research_generator = ResearchGuideGenerator()
        self.prereq_checker = PrerequisiteChecker()
    
    def get_comprehensive_documentation(self, command: str) -> ComprehensiveDocumentation:
        """
        Get complete documentation for a command.
        
        Args:
            command: The command name to get documentation for
            
        Returns:
            Complete ComprehensiveDocumentation object
        """
        # Get basic documentation
        basic_docs = self.doc_generator.get_complete_documentation(command)
        
        # Get examples
        basic_examples = self.example_generator.generate_basic_examples(command)
        advanced_examples = self.example_generator.generate_advanced_examples(command)
        research_examples = self.example_generator.generate_research_examples(command)
        all_examples = basic_examples + advanced_examples + research_examples
        
        # Get research guide if applicable
        research_guide = None
        if command in ["monitor", "analyze"]:
            domain = ResearchDomain.BEHAVIORAL_ANALYSIS  # Default domain
            research_guide = self.research_generator.generate_research_guide(domain)
        
        # Check prerequisites
        prerequisites = self.prereq_checker.check_command_prerequisites(command)
        
        # Generate quick start guide
        quick_start = self._generate_quick_start(command, prerequisites)
        
        # Generate troubleshooting guide
        troubleshooting = self._generate_troubleshooting(command)
        
        return ComprehensiveDocumentation(
            command=command,
            basic_docs=basic_docs,
            examples=all_examples,
            research_guide=research_guide,
            prerequisites=prerequisites,
            quick_start=quick_start,
            troubleshooting=troubleshooting
        )
    
    def generate_help_text(self, command: str, help_type: str = "basic") -> str:
        """
        Generate formatted help text for a command.
        
        Args:
            command: The command name
            help_type: Type of help ("basic", "examples", "research", "prerequisites")
            
        Returns:
            Formatted help text
        """
        if help_type == "basic":
            return self.doc_generator.generate_how_to_use(command)
        elif help_type == "examples":
            examples = self.example_generator.generate_basic_examples(command)
            return self._format_examples(examples)
        elif help_type == "research":
            return self.research_generator.generate_methodology_section([command])
        elif help_type == "prerequisites":
            return self.prereq_checker.generate_prerequisite_documentation(command)
        else:
            return f"Help type '{help_type}' not recognized"
    
    def check_command_readiness(self, command: str) -> Dict[str, Any]:
        """
        Check if a command is ready to use.
        
        Args:
            command: The command name to check
            
        Returns:
            Dictionary with readiness status and details
        """
        prereq_result = self.prereq_checker.check_command_prerequisites(command)
        
        return {
            "command": command,
            "ready": prereq_result.required_satisfied,
            "all_prerequisites_met": prereq_result.all_satisfied,
            "errors": prereq_result.errors,
            "warnings": prereq_result.warnings,
            "recommendations": prereq_result.recommendations,
            "next_steps": self._get_next_steps(prereq_result)
        }
    
    def generate_publication_guide(self, commands_used: List[str]) -> str:
        """
        Generate publication guidance for research using specific commands.
        
        Args:
            commands_used: List of commands used in the research
            
        Returns:
            Formatted publication guide
        """
        methodology = self.research_generator.generate_methodology_section(commands_used)
        checklist = self.research_generator.generate_publication_checklist()
        
        guide = f"""
# Publication Guide for ESCAI Research

{methodology}

## Publication Checklist

"""
        
        for item in checklist:
            guide += f"{item}\n"
        
        guide += """

## Citation Information

When using ESCAI in your research, please cite:

```bibtex
@software{escai_framework,
  title={ESCAI: Epistemic State and Causal Analysis Intelligence Framework},
  author={ESCAI Development Team},
  year={2024},
  url={https://github.com/escai-framework/escai}
}
```

## Methodology Template

Use this template for your methodology section:

```latex
\\subsection{Agent Monitoring and Analysis}

Agent behavior data was collected using the ESCAI (Epistemic State and Causal Analysis Intelligence) framework \\cite{escai_framework}, which provides real-time monitoring of autonomous agent cognition with minimal performance overhead (<10\\%). 

[Add specific details about your experimental setup, commands used, and analysis parameters]

All statistical analyses were performed using ESCAI's built-in analysis tools, which implement state-of-the-art statistical methods with appropriate significance testing and confidence interval calculation. Complete experimental reproducibility is ensured through ESCAI's session management system.
```
"""
        
        return textwrap.dedent(guide).strip()
    
    def _generate_quick_start(self, command: str, prereq_result: PrerequisiteCheckResult) -> str:
        """Generate quick start guide for a command."""
        if not prereq_result.required_satisfied:
            return f"""
# Quick Start: {command}

⚠️  **Prerequisites Not Met**

Before using this command, you need to satisfy the following requirements:

{chr(10).join(f"- {error}" for error in prereq_result.errors)}

## Setup Steps

{chr(10).join(f"{i+1}. {rec}" for i, rec in enumerate(prereq_result.recommendations))}

Once prerequisites are satisfied, run:
```bash
escai {command} --help
```
"""
        
        basic_example = self.example_generator.generate_basic_examples(command)
        example_cmd = basic_example[0].command if basic_example else f"escai {command}"
        
        return f"""
# Quick Start: {command}

✅ **Prerequisites Satisfied**

## Basic Usage

```bash
{example_cmd}
```

## Interactive Mode

```bash
escai --interactive
# Navigate to the {command} menu
```

## Next Steps

1. Try the basic example above
2. Explore advanced options with `escai {command} --help`
3. Check out research examples with `escai help {command} --research`
"""
    
    def _generate_troubleshooting(self, command: str) -> str:
        """Generate troubleshooting guide for a command."""
        return f"""
# Troubleshooting: {command}

## Common Issues

### Command Not Found
**Problem**: `escai: command not found`
**Solution**: 
- Ensure ESCAI is installed: `pip install escai-framework`
- Check your PATH includes Python scripts directory

### Permission Denied
**Problem**: Permission errors when running command
**Solution**:
- Check file permissions for data files
- Ensure write access to output directories
- Run with appropriate user privileges

### Import Errors
**Problem**: `ModuleNotFoundError` or import failures
**Solution**:
- Verify all dependencies: `pip check`
- Reinstall ESCAI: `pip install --upgrade --force-reinstall escai-framework`
- Check Python version compatibility

### Framework Integration Issues
**Problem**: Agent framework not detected or monitored
**Solution**:
- Ensure target framework is installed
- Verify framework instrumentation is properly configured
- Check framework version compatibility

### Performance Issues
**Problem**: Command runs slowly or uses too much memory
**Solution**:
- Reduce data volume with filtering options
- Use sampling for large datasets
- Check available system resources

## Getting Detailed Help

### Check Prerequisites
```bash
escai check-prerequisites {command}
```

### System Diagnostics
```bash
escai diagnostics
```

### Verbose Output
```bash
escai {command} --verbose --debug
```

### Log Analysis
```bash
escai logs --command {command} --level error
```

## Contact Support

If issues persist:
1. Gather system information: `escai diagnostics`
2. Check logs for detailed error messages
3. Create issue with reproduction steps
4. Include system information and error logs
"""
    
    def _format_examples(self, examples: List[CommandExample]) -> str:
        """Format examples for display."""
        if not examples:
            return "No examples available"
        
        formatted = "# Command Examples\n\n"
        
        for i, example in enumerate(examples, 1):
            formatted += f"""
## Example {i}: {example.title}

{example.description}

### Command
```bash
{example.command}
```

### Expected Output
```
{example.sample_output}
```

### Explanation
{example.explanation}

### Use Case
{example.use_case}

---

"""
        
        return formatted.strip()
    
    def _get_next_steps(self, prereq_result: PrerequisiteCheckResult) -> List[str]:
        """Get next steps based on prerequisite check results."""
        if prereq_result.required_satisfied:
            return [
                f"Run 'escai {prereq_result.command} --help' for detailed usage",
                f"Try basic examples with 'escai help {prereq_result.command} --examples'",
                f"Explore research features with 'escai help {prereq_result.command} --research'"
            ]
        else:
            return [
                "Install missing required prerequisites",
                f"Run 'escai check-prerequisites {prereq_result.command}' to verify installation",
                f"Try the command again once prerequisites are satisfied"
            ]