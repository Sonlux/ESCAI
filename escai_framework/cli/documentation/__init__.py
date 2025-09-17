"""
CLI Documentation Generation System

This module provides comprehensive documentation generation for all CLI commands,
including dynamic help content, usage examples, and research guidance.
"""

from .command_docs import CommandDocumentation, DocumentationGenerator
from .examples import ExampleGenerator, CommandExample
from .research_guides import ResearchGuideGenerator, ResearchDomain, ResearchGuide
from .prerequisite_checker import PrerequisiteChecker, PrerequisiteCheckResult
from .doc_integration import DocumentationIntegrator, ComprehensiveDocumentation

__all__ = [
    'CommandDocumentation',
    'DocumentationGenerator',
    'ExampleGenerator',
    'CommandExample',
    'ResearchGuideGenerator',
    'ResearchDomain',
    'ResearchGuide',
    'PrerequisiteChecker',
    'PrerequisiteCheckResult',
    'DocumentationIntegrator',
    'ComprehensiveDocumentation'
]