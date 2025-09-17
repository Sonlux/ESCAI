"""
Dynamic help content generator for CLI commands.

This module generates comprehensive documentation including "How to Use",
"When to Use", and "Why to Use" guides for all CLI commands.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import textwrap


class DocumentationType(Enum):
    """Types of documentation that can be generated."""
    HOW_TO_USE = "how_to_use"
    WHEN_TO_USE = "when_to_use"
    WHY_TO_USE = "why_to_use"
    SYNTAX_EXAMPLES = "syntax_examples"
    RESEARCH_BENEFITS = "research_benefits"


@dataclass
class CommandDocumentation:
    """Comprehensive documentation for a CLI command."""
    command_name: str
    description: str
    how_to_use: str
    when_to_use: str
    why_to_use: str
    syntax_examples: List[str]
    research_benefits: List[str]
    prerequisites: List[str]
    related_commands: List[str]


class DocumentationGenerator:
    """Generates comprehensive documentation for CLI commands."""
    
    def __init__(self):
        self._command_registry = self._initialize_command_registry()
        self._research_context = self._initialize_research_context()
    
    def generate_how_to_use(self, command: str) -> str:
        """
        Generate 'How to Use' guide with syntax examples.
        
        Args:
            command: The command name to generate documentation for
            
        Returns:
            Formatted how-to-use guide with syntax examples
        """
        if command not in self._command_registry:
            return f"Documentation not available for command: {command}"
        
        cmd_info = self._command_registry[command]
        
        guide = f"""
# How to Use: {command}

## Overview
{cmd_info['description']}

## Basic Syntax
```bash
escai {command} [OPTIONS] [ARGUMENTS]
```

## Interactive Mode
```bash
escai --interactive
# Navigate to: {cmd_info['menu_path']}
```

## Command Options
"""
        
        for option in cmd_info['options']:
            guide += f"- `{option['flag']}`: {option['description']}\n"
        
        guide += "\n## Step-by-Step Usage\n"
        for i, step in enumerate(cmd_info['usage_steps'], 1):
            guide += f"{i}. {step}\n"
        
        return textwrap.dedent(guide).strip()
    
    def generate_when_to_use(self, command: str) -> str:
        """
        Generate 'When to Use' scenarios with specific use cases.
        
        Args:
            command: The command name to generate scenarios for
            
        Returns:
            Formatted when-to-use scenarios
        """
        if command not in self._command_registry:
            return f"Scenarios not available for command: {command}"
        
        cmd_info = self._command_registry[command]
        
        scenarios = f"""
# When to Use: {command}

## Primary Use Cases

"""
        
        for scenario in cmd_info['use_cases']:
            scenarios += f"""
### {scenario['title']}
**Situation**: {scenario['situation']}
**Goal**: {scenario['goal']}
**Expected Outcome**: {scenario['outcome']}

"""
        
        scenarios += """
## Research Scenarios

"""
        
        for research_case in cmd_info['research_scenarios']:
            scenarios += f"""
### {research_case['context']}
{research_case['description']}

**Research Question**: {research_case['question']}
**Methodology**: {research_case['methodology']}

"""
        
        return textwrap.dedent(scenarios).strip()
    
    def generate_why_to_use(self, command: str) -> str:
        """
        Generate 'Why to Use' explanations with research benefits.
        
        Args:
            command: The command name to generate explanations for
            
        Returns:
            Formatted why-to-use explanations with research benefits
        """
        if command not in self._command_registry:
            return f"Benefits not available for command: {command}"
        
        cmd_info = self._command_registry[command]
        
        benefits = f"""
# Why to Use: {command}

## Research Benefits

"""
        
        for benefit in cmd_info['research_benefits']:
            benefits += f"""
### {benefit['category']}
{benefit['description']}

**Academic Value**: {benefit['academic_value']}
**Publication Impact**: {benefit['publication_impact']}

"""
        
        benefits += """
## Technical Advantages

"""
        
        for advantage in cmd_info['technical_advantages']:
            benefits += f"- **{advantage['feature']}**: {advantage['benefit']}\n"
        
        benefits += """

## Comparison with Alternatives

"""
        
        for comparison in cmd_info['comparisons']:
            benefits += f"""
### vs. {comparison['alternative']}
{comparison['advantage']}

"""
        
        return textwrap.dedent(benefits).strip()
    
    def generate_syntax_examples(self, command: str) -> List[str]:
        """
        Generate practical syntax examples with explanations.
        
        Args:
            command: The command name to generate examples for
            
        Returns:
            List of formatted syntax examples
        """
        if command not in self._command_registry:
            return [f"Examples not available for command: {command}"]
        
        cmd_info = self._command_registry[command]
        examples = []
        
        for example in cmd_info['examples']:
            formatted_example = f"""
# {example['title']}

## Command
```bash
{example['command']}
```

## Description
{example['description']}

## Expected Output
```
{example['output']}
```

## Use Case
{example['use_case']}
"""
            examples.append(textwrap.dedent(formatted_example).strip())
        
        return examples
    
    def get_complete_documentation(self, command: str) -> CommandDocumentation:
        """
        Generate complete documentation for a command.
        
        Args:
            command: The command name to generate documentation for
            
        Returns:
            Complete CommandDocumentation object
        """
        if command not in self._command_registry:
            return CommandDocumentation(
                command_name=command,
                description="Command not found",
                how_to_use="Documentation not available",
                when_to_use="Scenarios not available", 
                why_to_use="Benefits not available",
                syntax_examples=[],
                research_benefits=[],
                prerequisites=[],
                related_commands=[]
            )
        
        cmd_info = self._command_registry[command]
        
        return CommandDocumentation(
            command_name=command,
            description=cmd_info['description'],
            how_to_use=self.generate_how_to_use(command),
            when_to_use=self.generate_when_to_use(command),
            why_to_use=self.generate_why_to_use(command),
            syntax_examples=self.generate_syntax_examples(command),
            research_benefits=[b['description'] for b in cmd_info['research_benefits']],
            prerequisites=cmd_info['prerequisites'],
            related_commands=cmd_info['related_commands']
        )
    
    def _initialize_command_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the command registry with comprehensive command information."""
        return {
            "monitor": {
                "description": "Real-time monitoring of agent execution with epistemic state tracking",
                "menu_path": "Main Menu > Monitor Agents > Start Monitoring",
                "options": [
                    {"flag": "--framework", "description": "Specify agent framework (langchain, autogen, crewai, openai)"},
                    {"flag": "--agents", "description": "Comma-separated list of agent IDs to monitor"},
                    {"flag": "--output", "description": "Output format (json, csv, table)"},
                    {"flag": "--live", "description": "Enable live monitoring with real-time updates"},
                    {"flag": "--epistemic", "description": "Focus on epistemic state changes"},
                    {"flag": "--filter", "description": "Filter events by type or pattern"}
                ],
                "usage_steps": [
                    "Choose your agent framework (LangChain, AutoGen, CrewAI, or OpenAI)",
                    "Specify which agents to monitor (or use 'all' for all agents)",
                    "Select output format based on your analysis needs",
                    "Start monitoring and observe real-time agent behavior",
                    "Use filters to focus on specific types of events",
                    "Stop monitoring when your observation period is complete"
                ],
                "use_cases": [
                    {
                        "title": "Multi-Agent Conversation Analysis",
                        "situation": "You have multiple agents collaborating on a complex task",
                        "goal": "Understand how agents coordinate and share information",
                        "outcome": "Real-time view of agent interactions and decision patterns"
                    },
                    {
                        "title": "Epistemic State Evolution Tracking",
                        "situation": "You want to study how an agent's beliefs change over time",
                        "goal": "Track knowledge acquisition and belief updates",
                        "outcome": "Detailed timeline of epistemic state changes"
                    },
                    {
                        "title": "Performance Bottleneck Identification",
                        "situation": "Your agent system is running slower than expected",
                        "goal": "Identify where agents spend most of their time",
                        "outcome": "Performance metrics and bottleneck identification"
                    }
                ],
                "research_scenarios": [
                    {
                        "context": "Cognitive Architecture Research",
                        "description": "Studying how different agent architectures handle complex reasoning tasks",
                        "question": "How do different cognitive architectures affect reasoning patterns?",
                        "methodology": "Monitor agents with different architectures performing the same task"
                    },
                    {
                        "context": "Multi-Agent Coordination Studies",
                        "description": "Analyzing coordination patterns in multi-agent systems",
                        "question": "What communication patterns emerge in collaborative problem-solving?",
                        "methodology": "Monitor all agents in a collaborative scenario with interaction tracking"
                    }
                ],
                "research_benefits": [
                    {
                        "category": "Behavioral Analysis",
                        "description": "Provides unprecedented insight into agent decision-making processes",
                        "academic_value": "Enables publication of novel findings about agent cognition",
                        "publication_impact": "High-impact papers in AI conferences and journals"
                    },
                    {
                        "category": "Reproducible Research",
                        "description": "Complete monitoring logs enable exact reproduction of experiments",
                        "academic_value": "Meets highest standards for reproducible AI research",
                        "publication_impact": "Increases citation potential and research credibility"
                    }
                ],
                "technical_advantages": [
                    {"feature": "Real-time Processing", "benefit": "No delay between agent actions and analysis"},
                    {"feature": "Multi-framework Support", "benefit": "Works with any agent framework you're using"},
                    {"feature": "Minimal Overhead", "benefit": "Less than 10% impact on agent performance"},
                    {"feature": "Comprehensive Logging", "benefit": "Captures every aspect of agent behavior"}
                ],
                "comparisons": [
                    {
                        "alternative": "Manual Logging",
                        "advantage": "Automatic capture of all events without code modification"
                    },
                    {
                        "alternative": "Basic Debugging",
                        "advantage": "Structured analysis instead of scattered print statements"
                    }
                ],
                "examples": [
                    {
                        "title": "Monitor LangChain Agent with Live Updates",
                        "command": "escai monitor --framework langchain --agents agent_1 --live --epistemic",
                        "description": "Start real-time monitoring of a LangChain agent with focus on epistemic state changes",
                        "output": "Live dashboard showing agent thoughts, decisions, and knowledge updates",
                        "use_case": "Research into how LangChain agents build and update their knowledge"
                    },
                    {
                        "title": "Monitor Multiple AutoGen Agents",
                        "command": "escai monitor --framework autogen --agents all --output json --filter conversation",
                        "description": "Monitor all AutoGen agents and export conversation data as JSON",
                        "output": "JSON file with complete conversation history and agent interactions",
                        "use_case": "Analysis of multi-agent conversation patterns for publication"
                    }
                ],
                "prerequisites": [
                    "ESCAI framework installed and configured",
                    "Target agent framework (LangChain, AutoGen, etc.) installed",
                    "Agent code instrumented with ESCAI monitoring",
                    "Appropriate permissions for monitoring target processes"
                ],
                "related_commands": ["analyze", "session", "export"]
            },
            "analyze": {
                "description": "Advanced statistical analysis of agent behavior patterns and performance",
                "menu_path": "Main Menu > Analyze Data > Pattern Analysis",
                "options": [
                    {"flag": "--data", "description": "Path to monitoring data file or session ID"},
                    {"flag": "--type", "description": "Analysis type (patterns, causal, predictions, correlations)"},
                    {"flag": "--agents", "description": "Specific agents to analyze"},
                    {"flag": "--timeframe", "description": "Time range for analysis (e.g., '1h', '1d', '1w')"},
                    {"flag": "--confidence", "description": "Minimum confidence level for results (0.0-1.0)"},
                    {"flag": "--visualize", "description": "Generate visualizations (charts, heatmaps, graphs)"}
                ],
                "usage_steps": [
                    "Load monitoring data from a session or file",
                    "Choose the type of analysis you want to perform",
                    "Set confidence thresholds for statistical significance",
                    "Run the analysis and review results",
                    "Generate visualizations if needed",
                    "Export results in your preferred format"
                ],
                "use_cases": [
                    {
                        "title": "Behavioral Pattern Discovery",
                        "situation": "You have extensive monitoring data and want to find patterns",
                        "goal": "Identify recurring behavioral patterns in agent execution",
                        "outcome": "Statistical analysis of common patterns with significance testing"
                    },
                    {
                        "title": "Causal Relationship Analysis",
                        "situation": "You suspect certain events cause specific agent behaviors",
                        "goal": "Establish causal relationships between events and outcomes",
                        "outcome": "Causal inference results with confidence intervals"
                    },
                    {
                        "title": "Performance Prediction Modeling",
                        "situation": "You want to predict agent performance based on early indicators",
                        "goal": "Build predictive models for agent success/failure",
                        "outcome": "Trained models with accuracy metrics and prediction capabilities"
                    }
                ],
                "research_scenarios": [
                    {
                        "context": "Agent Learning Studies",
                        "description": "Analyzing how agents improve performance over time",
                        "question": "What patterns indicate successful learning in autonomous agents?",
                        "methodology": "Longitudinal analysis of agent performance with pattern mining"
                    },
                    {
                        "context": "Failure Mode Analysis",
                        "description": "Understanding why and how agents fail at tasks",
                        "question": "What early indicators predict agent task failure?",
                        "methodology": "Causal analysis of failure events with predictive modeling"
                    }
                ],
                "research_benefits": [
                    {
                        "category": "Statistical Rigor",
                        "description": "Provides statistically sound analysis with proper significance testing",
                        "academic_value": "Meets peer review standards for quantitative AI research",
                        "publication_impact": "Enables publication in top-tier venues requiring statistical validation"
                    },
                    {
                        "category": "Novel Insights",
                        "description": "Discovers patterns and relationships not visible through manual analysis",
                        "academic_value": "Generates novel research findings and hypotheses",
                        "publication_impact": "Potential for high-impact discoveries in agent behavior"
                    }
                ],
                "technical_advantages": [
                    {"feature": "Advanced Statistics", "benefit": "Uses state-of-the-art statistical methods"},
                    {"feature": "Causal Inference", "benefit": "Goes beyond correlation to establish causation"},
                    {"feature": "Predictive Modeling", "benefit": "Builds models for future performance prediction"},
                    {"feature": "Visualization", "benefit": "Creates publication-ready charts and graphs"}
                ],
                "comparisons": [
                    {
                        "alternative": "Manual Data Analysis",
                        "advantage": "Automated analysis with statistical rigor and significance testing"
                    },
                    {
                        "alternative": "Basic Analytics Tools",
                        "advantage": "Specialized for agent behavior with domain-specific insights"
                    }
                ],
                "examples": [
                    {
                        "title": "Pattern Analysis with Visualization",
                        "command": "escai analyze --data session_123 --type patterns --confidence 0.95 --visualize",
                        "description": "Analyze behavioral patterns from session 123 with 95% confidence and generate visualizations",
                        "output": "Statistical report with pattern significance and accompanying charts",
                        "use_case": "Publication-ready analysis of agent behavioral patterns"
                    },
                    {
                        "title": "Causal Analysis for Specific Agents",
                        "command": "escai analyze --data monitoring_data.json --type causal --agents agent_1,agent_2",
                        "description": "Perform causal analysis on specific agents from monitoring data",
                        "output": "Causal relationship graph with confidence intervals",
                        "use_case": "Understanding cause-effect relationships in multi-agent interactions"
                    }
                ],
                "prerequisites": [
                    "Monitoring data from previous sessions",
                    "Sufficient data volume for statistical analysis (recommended: 100+ events)",
                    "Understanding of statistical concepts for result interpretation",
                    "Python scientific computing libraries (automatically installed)"
                ],
                "related_commands": ["monitor", "export", "visualize"]
            }
        }
    
    def _initialize_research_context(self) -> Dict[str, Any]:
        """Initialize research context information for documentation generation."""
        return {
            "academic_standards": {
                "reproducibility": "All commands support full reproducibility with session management",
                "statistical_rigor": "Statistical analysis includes significance testing and confidence intervals",
                "documentation": "Comprehensive documentation suitable for methods sections"
            },
            "publication_support": {
                "formats": ["LaTeX tables", "Publication-ready figures", "Statistical reports"],
                "citations": "Automatic methodology citations for academic papers",
                "standards": "Follows best practices for AI research publication"
            }
        }