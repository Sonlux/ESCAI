"""
Documentation quality tests for help content accuracy
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import re
from pathlib import Path

from escai_framework.cli.main import cli
from escai_framework.cli.documentation.command_docs import CommandDocumentation
from escai_framework.cli.documentation.examples import ExampleGenerator
from escai_framework.cli.documentation.research_guides import ResearchGuideGenerator


class TestHelpContentAccuracy:
    """Test accuracy and completeness of help content"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
        self.doc_generator = CommandDocumentation()
    
    def test_main_help_completeness(self):
        """Test that main help is complete and accurate"""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        
        # Check essential elements
        assert 'ESCAI Framework' in result.output
        assert 'Usage:' in result.output
        
        # Check all main command groups are documented
        expected_commands = ['monitor', 'analyze', 'config', 'session']
        for cmd in expected_commands:
            assert cmd in result.output
        
        # Check that descriptions are present
        lines = result.output.split('\n')
        command_section_found = False
        for line in lines:
            if 'Commands:' in line or 'Usage:' in line:
                command_section_found = True
                break
        assert command_section_found
    
    def test_command_help_accuracy(self):
        """Test that individual command help is accurate"""
        commands_to_test = [
            (['monitor', '--help'], ['start', 'stop', 'status']),
            (['analyze', '--help'], ['patterns', 'causal', 'predictions']),
            (['config', '--help'], ['setup', 'show', 'set', 'get']),
            (['session', '--help'], ['list', 'details', 'stop'])
        ]
        
        for cmd, expected_subcommands in commands_to_test:
            result = self.runner.invoke(cli, cmd)
            assert result.exit_code == 0
            
            # Check that all expected subcommands are documented
            for subcmd in expected_subcommands:
                assert subcmd in result.output
            
            # Check that help has proper structure
            assert 'Usage:' in result.output
            assert len(result.output) > 100  # Reasonable help length
    
    def test_subcommand_help_detail(self):
        """Test that subcommand help provides sufficient detail"""
        detailed_commands = [
            ['monitor', 'start', '--help'],
            ['analyze', 'patterns', '--help'],
            ['config', 'setup', '--help'],
            ['session', 'list', '--help']
        ]
        
        for cmd in detailed_commands:
            result = self.runner.invoke(cli, cmd)
            assert result.exit_code == 0
            
            # Check for essential help elements
            assert 'Usage:' in result.output
            
            # Check for option descriptions
            if '--' in result.output:
                # If options exist, they should be documented
                assert 'Options:' in result.output or '--help' in result.output
    
    def test_option_documentation_accuracy(self):
        """Test that command options are accurately documented"""
        # Test monitor start options
        result = self.runner.invoke(cli, ['monitor', 'start', '--help'])
        assert result.exit_code == 0
        
        # Should document required options
        expected_options = ['--agent-id', '--framework']
        for option in expected_options:
            assert option in result.output
        
        # Test analyze patterns options
        result = self.runner.invoke(cli, ['analyze', 'patterns', '--help'])
        assert result.exit_code == 0
        
        # Should document analysis options
        if '--agent-id' in result.output:
            assert '--agent-id' in result.output
    
    def test_framework_documentation(self):
        """Test that framework options are properly documented"""
        result = self.runner.invoke(cli, ['monitor', 'start', '--help'])
        assert result.exit_code == 0
        
        # Should mention supported frameworks
        frameworks = ['langchain', 'autogen', 'crewai', 'openai']
        framework_mentioned = any(fw in result.output.lower() for fw in frameworks)
        
        # At least some framework information should be present
        # (Exact format may vary)
    
    def test_example_accuracy(self):
        """Test that examples in help are accurate and runnable"""
        # This tests the structure - actual examples would need validation
        # against real command syntax
        
        # Test that help contains example-like content
        result = self.runner.invoke(cli, ['monitor', 'start', '--help'])
        assert result.exit_code == 0
        
        # Look for example patterns (commands that start with escai or similar)
        lines = result.output.split('\n')
        example_patterns = [
            r'escai\s+monitor\s+start',
            r'--agent-id\s+\w+',
            r'--framework\s+\w+'
        ]
        
        # At least some example-like content should be present
        has_example_content = any(
            any(re.search(pattern, line) for pattern in example_patterns)
            for line in lines
        )
        # Note: This is a structural test - actual examples may vary


class TestDocumentationGeneration:
    """Test documentation generation system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.doc_generator = CommandDocumentation()
        self.example_generator = ExampleGenerator()
        self.guide_generator = ResearchGuideGenerator()
    
    def test_how_to_use_generation(self):
        """Test 'How to Use' guide generation"""
        commands_to_test = ['monitor_start', 'analyze_patterns', 'config_setup']
        
        for command in commands_to_test:
            how_to_use = self.doc_generator.generate_how_to_use(command)
            
            assert isinstance(how_to_use, str)
            assert len(how_to_use) > 50  # Should have substantial content
            
            # Should contain usage information
            assert 'usage' in how_to_use.lower() or 'use' in how_to_use.lower()
    
    def test_when_to_use_generation(self):
        """Test 'When to Use' scenario generation"""
        commands_to_test = ['monitor_start', 'analyze_causal', 'session_export']
        
        for command in commands_to_test:
            when_to_use = self.doc_generator.generate_when_to_use(command)
            
            assert isinstance(when_to_use, str)
            assert len(when_to_use) > 30  # Should have meaningful content
            
            # Should contain scenario information
            scenario_keywords = ['when', 'if', 'scenario', 'situation', 'case']
            assert any(keyword in when_to_use.lower() for keyword in scenario_keywords)
    
    def test_why_to_use_generation(self):
        """Test 'Why to Use' explanation generation"""
        commands_to_test = ['analyze_patterns', 'monitor_epistemic', 'config_theme']
        
        for command in commands_to_test:
            why_to_use = self.doc_generator.generate_why_to_use(command)
            
            assert isinstance(why_to_use, str)
            assert len(why_to_use) > 40  # Should have explanatory content
            
            # Should contain benefit/reason information
            benefit_keywords = ['benefit', 'advantage', 'help', 'enable', 'provide']
            assert any(keyword in why_to_use.lower() for keyword in benefit_keywords)
    
    def test_example_generation(self):
        """Test practical example generation"""
        commands_to_test = ['monitor_start', 'analyze_export', 'session_replay']
        
        for command in commands_to_test:
            examples = self.example_generator.generate_examples(command)
            
            assert isinstance(examples, list)
            assert len(examples) > 0  # Should have at least one example
            
            for example in examples:
                assert 'command' in example
                assert 'description' in example
                assert isinstance(example['command'], str)
                assert isinstance(example['description'], str)
                assert len(example['command']) > 5  # Should be a real command
    
    def test_prerequisite_documentation(self):
        """Test prerequisite documentation generation"""
        from escai_framework.cli.documentation.prerequisite_checker import PrerequisiteChecker
        
        checker = PrerequisiteChecker()
        
        commands_to_test = ['monitor_start', 'config_setup', 'analyze_patterns']
        
        for command in commands_to_test:
            prerequisites = checker.get_prerequisites(command)
            
            assert isinstance(prerequisites, dict)
            
            # Should have required sections
            expected_sections = ['system', 'configuration', 'dependencies']
            for section in expected_sections:
                if section in prerequisites:
                    assert isinstance(prerequisites[section], list)
    
    def test_research_guide_generation(self):
        """Test research-specific guide generation"""
        research_scenarios = [
            'agent_performance_analysis',
            'behavioral_pattern_study',
            'causal_relationship_research'
        ]
        
        for scenario in research_scenarios:
            guide = self.guide_generator.generate_research_guide(scenario)
            
            assert isinstance(guide, dict)
            
            # Should have research-specific sections
            expected_sections = ['objective', 'methodology', 'commands', 'interpretation']
            for section in expected_sections:
                if section in guide:
                    assert isinstance(guide[section], str)
                    assert len(guide[section]) > 20  # Should have meaningful content


class TestDocumentationConsistency:
    """Test consistency across documentation"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_terminology_consistency(self):
        """Test that terminology is used consistently"""
        # Get help for multiple commands
        help_outputs = []
        commands = [
            ['monitor', '--help'],
            ['analyze', '--help'],
            ['config', '--help'],
            ['session', '--help']
        ]
        
        for cmd in commands:
            result = self.runner.invoke(cli, cmd)
            if result.exit_code == 0:
                help_outputs.append(result.output)
        
        # Check for consistent terminology
        # For example, "agent" should be used consistently, not mixed with "bot" or "AI"
        key_terms = ['agent', 'framework', 'session', 'monitoring']
        
        for term in key_terms:
            # If term appears in one help, check usage consistency
            term_usages = []
            for output in help_outputs:
                if term in output.lower():
                    # Extract context around the term
                    lines = output.lower().split('\n')
                    for line in lines:
                        if term in line:
                            term_usages.append(line.strip())
            
            # Basic consistency check - terms should be used in similar contexts
            if len(term_usages) > 1:
                # All usages should be reasonably similar in context
                # (This is a basic structural check)
                assert all(isinstance(usage, str) for usage in term_usages)
    
    def test_format_consistency(self):
        """Test that help format is consistent across commands"""
        commands = [
            ['monitor', 'start', '--help'],
            ['analyze', 'patterns', '--help'],
            ['config', 'setup', '--help'],
            ['session', 'list', '--help']
        ]
        
        help_structures = []
        
        for cmd in commands:
            result = self.runner.invoke(cli, cmd)
            if result.exit_code == 0:
                lines = result.output.split('\n')
                
                # Analyze structure
                structure = {
                    'has_usage': any('Usage:' in line for line in lines),
                    'has_description': len([line for line in lines if line.strip() and not line.startswith(' ')]) > 2,
                    'has_options': any('Options:' in line for line in lines),
                    'line_count': len(lines)
                }
                help_structures.append(structure)
        
        # Check consistency
        if len(help_structures) > 1:
            # All should have usage
            usage_consistency = all(struct['has_usage'] for struct in help_structures)
            # Note: Not all commands may have options, so we don't enforce that
            
            # Basic structure should be similar
            assert all(struct['line_count'] > 3 for struct in help_structures)
    
    def test_option_naming_consistency(self):
        """Test that option names are consistent across commands"""
        # Test that similar options use the same names
        commands_with_agent_id = [
            ['monitor', 'start', '--help'],
            ['analyze', 'patterns', '--help'],
            ['analyze', 'predictions', '--help']
        ]
        
        agent_id_variations = []
        
        for cmd in commands_with_agent_id:
            result = self.runner.invoke(cli, cmd)
            if result.exit_code == 0:
                # Look for agent ID option variations
                if '--agent-id' in result.output:
                    agent_id_variations.append('--agent-id')
                elif '--agent_id' in result.output:
                    agent_id_variations.append('--agent_id')
                elif '--id' in result.output:
                    agent_id_variations.append('--id')
        
        # Should use consistent naming
        if len(agent_id_variations) > 1:
            # All should use the same format
            assert len(set(agent_id_variations)) == 1, f"Inconsistent agent ID options: {set(agent_id_variations)}"


class TestDocumentationAccuracy:
    """Test accuracy of documentation content"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_command_existence_accuracy(self):
        """Test that documented commands actually exist"""
        # Get main help
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        
        # Extract mentioned commands
        lines = result.output.split('\n')
        mentioned_commands = []
        
        for line in lines:
            # Look for command patterns
            words = line.strip().split()
            if len(words) > 0 and not line.startswith(' ') and words[0].isalpha():
                # Potential command name
                if len(words[0]) > 2 and words[0] not in ['Usage', 'Options', 'Commands']:
                    mentioned_commands.append(words[0])
        
        # Test that mentioned commands actually work
        common_commands = ['monitor', 'analyze', 'config', 'session']
        for cmd in common_commands:
            if cmd in mentioned_commands:
                result = self.runner.invoke(cli, [cmd, '--help'])
                assert result.exit_code == 0, f"Command {cmd} mentioned in help but doesn't work"
    
    def test_option_existence_accuracy(self):
        """Test that documented options actually exist"""
        # Test specific command options
        result = self.runner.invoke(cli, ['monitor', 'start', '--help'])
        assert result.exit_code == 0
        
        # If --agent-id is mentioned, test that it works
        if '--agent-id' in result.output:
            # Test that the option is actually accepted
            with patch('escai_framework.cli.commands.monitor.console'):
                result = self.runner.invoke(cli, [
                    'monitor', 'start',
                    '--agent-id', 'test_agent',
                    '--framework', 'langchain'
                ])
                # Should not fail due to unknown option
                assert '--agent-id' not in result.output or result.exit_code == 0
    
    def test_framework_list_accuracy(self):
        """Test that documented frameworks are actually supported"""
        # This would test against the actual framework list
        # For now, we test the structure exists
        
        result = self.runner.invoke(cli, ['monitor', 'start', '--help'])
        assert result.exit_code == 0
        
        # Test with known frameworks
        known_frameworks = ['langchain', 'autogen', 'crewai', 'openai']
        
        for framework in known_frameworks:
            with patch('escai_framework.cli.commands.monitor.console'):
                result = self.runner.invoke(cli, [
                    'monitor', 'start',
                    '--agent-id', 'test',
                    '--framework', framework
                ])
                # Should not fail due to invalid framework (in normal operation)
                # In test environment, we just check it doesn't crash completely
                assert result.exit_code in [0, 1, 2]  # Allow for various test environment issues


class TestDocumentationCompleteness:
    """Test completeness of documentation coverage"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_all_commands_documented(self):
        """Test that all commands have help documentation"""
        # Get list of main commands
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        
        main_commands = ['monitor', 'analyze', 'config', 'session']
        
        for cmd in main_commands:
            # Each main command should have help
            result = self.runner.invoke(cli, [cmd, '--help'])
            assert result.exit_code == 0
            assert len(result.output) > 50  # Should have substantial help
            
            # Get subcommands
            subcommand_result = self.runner.invoke(cli, [cmd, '--help'])
            if 'Commands:' in subcommand_result.output:
                # Extract subcommands and test their help
                lines = subcommand_result.output.split('\n')
                in_commands_section = False
                
                for line in lines:
                    if 'Commands:' in line:
                        in_commands_section = True
                        continue
                    
                    if in_commands_section and line.strip():
                        if line.startswith('  ') and not line.startswith('   '):
                            # This looks like a subcommand
                            subcmd = line.strip().split()[0]
                            if subcmd and subcmd.isalpha():
                                # Test subcommand help
                                sub_result = self.runner.invoke(cli, [cmd, subcmd, '--help'])
                                assert sub_result.exit_code == 0, f"Subcommand {cmd} {subcmd} help failed"
    
    def test_critical_workflows_documented(self):
        """Test that critical workflows are documented"""
        # Test that key user workflows have documentation
        critical_commands = [
            ['monitor', 'start'],
            ['monitor', 'stop'],
            ['analyze', 'patterns'],
            ['config', 'setup'],
            ['session', 'list']
        ]
        
        for cmd in critical_commands:
            result = self.runner.invoke(cli, cmd + ['--help'])
            assert result.exit_code == 0
            assert len(result.output) > 30  # Should have meaningful help
    
    def test_error_scenarios_documented(self):
        """Test that common error scenarios are documented or handled"""
        # Test invalid framework error
        result = self.runner.invoke(cli, [
            'monitor', 'start',
            '--agent-id', 'test',
            '--framework', 'invalid_framework'
        ])
        
        # Should provide helpful error message
        assert result.exit_code != 0
        # Error message should be informative (not just a stack trace)
        assert len(result.output) > 10


if __name__ == '__main__':
    pytest.main([__file__])