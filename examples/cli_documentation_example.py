#!/usr/bin/env python3
"""
Example demonstrating the CLI documentation generation system.

This example shows how to use the comprehensive documentation system
to generate help content, examples, and research guidance for CLI commands.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from escai_framework.cli.documentation import (
    DocumentationIntegrator,
    ResearchDomain
)


def main():
    """Demonstrate the documentation generation system."""
    print("=" * 80)
    print("ESCAI CLI Documentation Generation System Demo")
    print("=" * 80)
    
    # Initialize the documentation integrator
    integrator = DocumentationIntegrator()
    
    # Example 1: Generate comprehensive documentation for monitor command
    print("\n1. COMPREHENSIVE DOCUMENTATION FOR 'monitor' COMMAND")
    print("-" * 60)
    
    docs = integrator.get_comprehensive_documentation("monitor")
    print(f"Command: {docs.command}")
    print(f"Description: {docs.basic_docs.description}")
    print(f"Number of examples: {len(docs.examples)}")
    print(f"Prerequisites satisfied: {docs.prerequisites.required_satisfied}")
    print(f"Research guide available: {docs.research_guide is not None}")
    
    # Example 2: Generate different types of help text
    print("\n2. DIFFERENT TYPES OF HELP TEXT")
    print("-" * 60)
    
    help_types = ["basic", "examples", "research", "prerequisites"]
    for help_type in help_types:
        print(f"\n{help_type.upper()} HELP:")
        help_text = integrator.generate_help_text("monitor", help_type)
        # Show first 200 characters
        print(help_text[:200] + "..." if len(help_text) > 200 else help_text)
    
    # Example 3: Check command readiness
    print("\n3. COMMAND READINESS CHECK")
    print("-" * 60)
    
    readiness = integrator.check_command_readiness("monitor")
    print(f"Command ready: {readiness['ready']}")
    print(f"All prerequisites met: {readiness['all_prerequisites_met']}")
    print(f"Number of errors: {len(readiness['errors'])}")
    print(f"Number of warnings: {len(readiness['warnings'])}")
    print(f"Next steps: {len(readiness['next_steps'])}")
    
    if readiness['errors']:
        print("\nErrors:")
        for error in readiness['errors'][:2]:  # Show first 2 errors
            print(f"  - {error}")
    
    if readiness['warnings']:
        print("\nWarnings:")
        for warning in readiness['warnings'][:2]:  # Show first 2 warnings
            print(f"  - {warning}")
    
    # Example 4: Generate publication guide
    print("\n4. PUBLICATION GUIDE GENERATION")
    print("-" * 60)
    
    pub_guide = integrator.generate_publication_guide(["monitor", "analyze"])
    print("Publication guide generated successfully!")
    print(f"Guide length: {len(pub_guide)} characters")
    print("\nFirst 300 characters:")
    print(pub_guide[:300] + "...")
    
    # Example 5: Research guide for specific domain
    print("\n5. RESEARCH GUIDE FOR BEHAVIORAL ANALYSIS")
    print("-" * 60)
    
    research_guide = integrator.research_generator.generate_research_guide(
        ResearchDomain.BEHAVIORAL_ANALYSIS
    )
    print(f"Domain: {research_guide.domain.value}")
    print(f"Title: {research_guide.title}")
    print(f"Research questions: {len(research_guide.research_questions)}")
    print(f"Recommended commands: {len(research_guide.recommended_commands)}")
    print(f"Publication tips: {len(research_guide.publication_tips)}")
    
    print("\nFirst research question:")
    if research_guide.research_questions:
        print(f"  {research_guide.research_questions[0]}")
    
    # Example 6: Prerequisite checking details
    print("\n6. DETAILED PREREQUISITE CHECKING")
    print("-" * 60)
    
    prereq_result = integrator.prereq_checker.check_command_prerequisites("analyze")
    print(f"Command: {prereq_result.command}")
    print(f"Total prerequisites: {len(prereq_result.prerequisites)}")
    
    for prereq in prereq_result.prerequisites[:3]:  # Show first 3 prerequisites
        print(f"\n  {prereq.name}:")
        print(f"    Type: {prereq.type.value}")
        print(f"    Importance: {prereq.importance}")
        print(f"    Status: {prereq.status.value if prereq.status else 'Not checked'}")
        if prereq.details:
            print(f"    Details: {prereq.details}")
    
    # Example 7: System diagnostics
    print("\n7. SYSTEM DIAGNOSTICS")
    print("-" * 60)
    
    diagnostics = integrator.prereq_checker.get_system_diagnostics()
    print(f"Platform: {diagnostics['system_info']['platform']}")
    print(f"Python version: {diagnostics['system_info']['python_version']}")
    print(f"Architecture: {diagnostics['environment']['architecture']}")
    print(f"ESCAI version: {diagnostics['escai_info']['version']}")
    
    print("\n" + "=" * 80)
    print("Documentation generation demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()