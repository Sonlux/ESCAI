#!/usr/bin/env python3
"""
Comprehensive ESCAI CLI Usage Examples

This script demonstrates various ways to use the ESCAI CLI for monitoring
and analyzing autonomous agents across different frameworks.
"""

import asyncio
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Example configurations for different scenarios
EXAMPLE_CONFIGS = {
    "basic_monitoring": {
        "capture_epistemic_states": True,
        "capture_behavioral_patterns": True,
        "capture_performance_metrics": True,
        "max_events_per_second": 50,
        "buffer_size": 10000
    },
    "research_monitoring": {
        "capture_epistemic_states": True,
        "capture_behavioral_patterns": True,
        "capture_performance_metrics": True,
        "capture_causal_relationships": True,
        "max_events_per_second": 100,
        "buffer_size": 50000,
        "detailed_logging": True
    },
    "production_monitoring": {
        "capture_epistemic_states": False,
        "capture_behavioral_patterns": True,
        "capture_performance_metrics": True,
        "max_events_per_second": 20,
        "buffer_size": 5000,
        "lightweight_mode": True
    }
}


class ESCAICLIExamples:
    """
    Comprehensive examples for ESCAI CLI usage.
    
    This class provides examples for various use cases including:
    - Basic agent monitoring
    - Multi-framework integration
    - Research workflows
    - Production monitoring
    - Analysis and reporting
    """
    
    def __init__(self):
        """Initialize CLI examples."""
        self.session_ids: List[str] = []
        self.example_data_dir = Path("example_outputs")
        self.example_data_dir.mkdir(exist_ok=True)
    
    def run_cli_command(self, command: List[str], capture_output: bool = True) -> Dict[str, Any]:
        """
        Run a CLI command and return the result.
        
        Args:
            command: Command to run as list of strings
            capture_output: Whether to capture command output
            
        Returns:
            Dictionary with command result
        """
        try:
            print(f"Running: {' '.join(command)}")
            
            if capture_output:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            else:
                result = subprocess.run(command, timeout=30)
                return {
                    "success": result.returncode == 0,
                    "returncode": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def example_basic_setup(self) -> None:
        """Example: Basic ESCAI CLI setup."""
        print("\n" + "="*60)
        print("EXAMPLE 1: Basic ESCAI CLI Setup")
        print("="*60)
        
        # Check version
        print("\n1. Checking ESCAI version...")
        result = self.run_cli_command(["escai", "--version"])
        if result["success"]:
            print(f"✓ ESCAI version: {result['stdout'].strip()}")
        else:
            print(f"✗ Error: {result.get('error', 'Unknown error')}")
        
        # Show help
        print("\n2. Showing quick help...")
        result = self.run_cli_command(["escai", "help"])
        if result["success"]:
            print("✓ Help system available")
            print("First few lines of help:")
            lines = result["stdout"].split('\n')[:5]
            for line in lines:
                print(f"  {line}")
        
        # Test configuration
        print("\n3. Testing configuration...")
        result = self.run_cli_command(["escai", "config", "check"])
        if result["success"]:
            print("✓ Configuration check passed")
        else:
            print("⚠ Configuration needs setup")
            print("Run: escai config setup")
    
    def example_framework_detection(self) -> None:
        """Example: Detect available frameworks."""
        print("\n" + "="*60)
        print("EXAMPLE 2: Framework Detection")
        print("="*60)
        
        frameworks = ["langchain", "autogen", "crewai", "openai"]
        
        for framework in frameworks:
            print(f"\n1. Testing {framework} availability...")
            result = self.run_cli_command([
                "escai", "config", "test", "--framework", framework
            ])
            
            if result["success"]:
                print(f"✓ {framework} is available")
            else:
                print(f"✗ {framework} is not available")
                print(f"  Install with: pip install {framework}")
    
    def example_basic_monitoring(self) -> None:
        """Example: Basic agent monitoring workflow."""
        print("\n" + "="*60)
        print("EXAMPLE 3: Basic Agent Monitoring")
        print("="*60)
        
        agent_id = "example_basic_agent"
        framework = "langchain"  # Change based on availability
        
        print(f"\n1. Starting monitoring for {framework} agent...")
        
        # Start monitoring
        start_cmd = [
            "escai", "monitor", "start",
            "--framework", framework,
            "--agent-id", agent_id,
            "--capture-epistemic",
            "--capture-behavioral"
        ]
        
        result = self.run_cli_command(start_cmd)
        if result["success"]:
            print(f"✓ Monitoring started for agent: {agent_id}")
            
            # Extract session ID from output (this would be implementation-specific)
            # For demo purposes, we'll simulate
            session_id = f"session_{int(time.time())}"
            self.session_ids.append(session_id)
            
            # Check status
            print("\n2. Checking monitoring status...")
            status_result = self.run_cli_command(["escai", "monitor", "status"])
            if status_result["success"]:
                print("✓ Status check successful")
            
            # Simulate some monitoring time
            print("\n3. Simulating agent activity (5 seconds)...")
            time.sleep(5)
            
            # Stop monitoring
            print("\n4. Stopping monitoring...")
            stop_result = self.run_cli_command([
                "escai", "monitor", "stop", "--agent-id", agent_id
            ])
            if stop_result["success"]:
                print("✓ Monitoring stopped successfully")
            
        else:
            print(f"✗ Failed to start monitoring: {result.get('stderr', 'Unknown error')}")
    
    def example_pattern_analysis(self) -> None:
        """Example: Behavioral pattern analysis."""
        print("\n" + "="*60)
        print("EXAMPLE 4: Pattern Analysis")
        print("="*60)
        
        agent_id = "example_pattern_agent"
        
        print("\n1. Analyzing behavioral patterns...")
        
        # Analyze patterns
        analyze_cmd = [
            "escai", "analyze", "patterns",
            "--agent-id", agent_id,
            "--timeframe", "1h",
            "--confidence-threshold", "0.7"
        ]
        
        result = self.run_cli_command(analyze_cmd)
        if result["success"]:
            print("✓ Pattern analysis completed")
            
            # Save results
            output_file = self.example_data_dir / "patterns.json"
            export_cmd = [
                "escai", "analyze", "export",
                "--format", "json",
                "--output", str(output_file)
            ]
            
            export_result = self.run_cli_command(export_cmd)
            if export_result["success"]:
                print(f"✓ Results exported to: {output_file}")
        else:
            print("⚠ No pattern data available (need active monitoring)")
    
    def example_causal_analysis(self) -> None:
        """Example: Causal relationship analysis."""
        print("\n" + "="*60)
        print("EXAMPLE 5: Causal Analysis")
        print("="*60)
        
        agent_id = "example_causal_agent"
        
        print("\n1. Analyzing causal relationships...")
        
        # Causal analysis
        causal_cmd = [
            "escai", "analyze", "causal",
            "--agent-id", agent_id,
            "--confidence-threshold", "0.8"
        ]
        
        result = self.run_cli_command(causal_cmd)
        if result["success"]:
            print("✓ Causal analysis completed")
            
            # Visualize results
            print("\n2. Generating causal network visualization...")
            viz_cmd = [
                "escai", "analyze", "visualize",
                "--type", "causal",
                "--format", "network",
                "--agent-id", agent_id
            ]
            
            viz_result = self.run_cli_command(viz_cmd)
            if viz_result["success"]:
                print("✓ Visualization generated")
        else:
            print("⚠ No causal data available (need monitoring data)")
    
    def example_session_management(self) -> None:
        """Example: Session management workflow."""
        print("\n" + "="*60)
        print("EXAMPLE 6: Session Management")
        print("="*60)
        
        print("\n1. Listing active sessions...")
        list_cmd = ["escai", "session", "list", "--active"]
        result = self.run_cli_command(list_cmd)
        if result["success"]:
            print("✓ Session list retrieved")
        
        print("\n2. Listing all sessions...")
        list_all_cmd = ["escai", "session", "list"]
        result = self.run_cli_command(list_all_cmd)
        if result["success"]:
            print("✓ All sessions listed")
        
        # If we have session IDs from previous examples
        if self.session_ids:
            session_id = self.session_ids[0]
            
            print(f"\n3. Getting session details for: {session_id}")
            details_cmd = ["escai", "session", "details", "--session-id", session_id]
            result = self.run_cli_command(details_cmd)
            if result["success"]:
                print("✓ Session details retrieved")
            
            print(f"\n4. Exporting session data...")
            export_file = self.example_data_dir / f"session_{session_id}.csv"
            export_cmd = [
                "escai", "session", "export",
                "--session-id", session_id,
                "--format", "csv",
                "--output", str(export_file)
            ]
            result = self.run_cli_command(export_cmd)
            if result["success"]:
                print(f"✓ Session exported to: {export_file}")
    
    def example_publication_output(self) -> None:
        """Example: Generate publication-ready output."""
        print("\n" + "="*60)
        print("EXAMPLE 7: Publication Output")
        print("="*60)
        
        agent_id = "example_research_agent"
        
        print("\n1. Generating statistical report...")
        stats_cmd = [
            "escai", "publication", "generate",
            "--type", "statistical",
            "--agent-id", agent_id,
            "--format", "pdf"
        ]
        
        result = self.run_cli_command(stats_cmd)
        if result["success"]:
            print("✓ Statistical report generated")
        else:
            print("⚠ No data available for publication")
        
        print("\n2. Generating academic paper format...")
        academic_cmd = [
            "escai", "publication", "generate",
            "--type", "academic",
            "--format", "latex",
            "--include-citations",
            "--include-methodology"
        ]
        
        result = self.run_cli_command(academic_cmd)
        if result["success"]:
            print("✓ Academic format generated")
        else:
            print("⚠ Academic format requires monitoring data")
    
    def example_interactive_mode(self) -> None:
        """Example: Interactive mode demonstration."""
        print("\n" + "="*60)
        print("EXAMPLE 8: Interactive Mode")
        print("="*60)
        
        print("\nInteractive mode provides a menu-driven interface.")
        print("To try interactive mode, run:")
        print("  escai --interactive")
        print("\nThis will launch a menu system where you can:")
        print("  • Navigate using numbered options")
        print("  • Use 'b' to go back, 'q' to quit")
        print("  • Use 'h' for help at any menu level")
        print("  • Access all CLI functionality through menus")
    
    def example_configuration_management(self) -> None:
        """Example: Configuration management."""
        print("\n" + "="*60)
        print("EXAMPLE 9: Configuration Management")
        print("="*60)
        
        print("\n1. Showing current configuration...")
        show_cmd = ["escai", "config", "show"]
        result = self.run_cli_command(show_cmd)
        if result["success"]:
            print("✓ Configuration displayed")
        
        print("\n2. Testing database connection...")
        test_cmd = ["escai", "config", "test"]
        result = self.run_cli_command(test_cmd)
        if result["success"]:
            print("✓ Database connection test passed")
        else:
            print("⚠ Database connection test failed")
        
        print("\n3. Checking system status...")
        check_cmd = ["escai", "config", "check"]
        result = self.run_cli_command(check_cmd)
        if result["success"]:
            print("✓ System check passed")
    
    def example_advanced_workflows(self) -> None:
        """Example: Advanced workflow combinations."""
        print("\n" + "="*60)
        print("EXAMPLE 10: Advanced Workflows")
        print("="*60)
        
        print("\nAdvanced workflows combine multiple CLI features:")
        
        print("\n1. Research Workflow:")
        print("   escai monitor start --framework langchain --agent-id research_agent")
        print("   # ... run experiments ...")
        print("   escai analyze patterns --agent-id research_agent")
        print("   escai analyze causal --agent-id research_agent")
        print("   escai publication generate --type academic --format latex")
        
        print("\n2. Comparative Analysis:")
        print("   escai monitor start --framework langchain --agent-id agent1")
        print("   escai monitor start --framework autogen --agent-id agent2")
        print("   # ... run both agents ...")
        print("   escai analyze compare --agents agent1,agent2")
        print("   escai publication generate --type comparative")
        
        print("\n3. Production Monitoring:")
        print("   escai monitor start --framework openai --agent-id prod_agent \\")
        print("     --capture-performance --lightweight-mode")
        print("   escai monitor live --agent-id prod_agent")
        print("   escai analyze stats --agent-id prod_agent --timeframe 24h")
    
    def example_troubleshooting(self) -> None:
        """Example: Troubleshooting and debugging."""
        print("\n" + "="*60)
        print("EXAMPLE 11: Troubleshooting")
        print("="*60)
        
        print("\n1. Debug mode for troubleshooting...")
        print("   Use --debug flag for verbose output:")
        print("   escai --debug monitor start --framework langchain --agent-id debug_agent")
        
        print("\n2. Checking logs...")
        logs_cmd = ["escai", "logs", "show", "--level", "error", "--recent"]
        result = self.run_cli_command(logs_cmd)
        if result["success"]:
            print("✓ Error logs retrieved")
        
        print("\n3. System diagnostics...")
        print("   escai config check          # Check system status")
        print("   escai config test --all     # Test all frameworks")
        print("   escai logs system           # System resource usage")
        
        print("\n4. Performance optimization...")
        print("   escai session cleanup       # Clean old sessions")
        print("   escai config optimize       # Optimize database")
    
    def run_all_examples(self) -> None:
        """Run all CLI examples."""
        print("ESCAI CLI Comprehensive Examples")
        print("=" * 80)
        print("This script demonstrates various ESCAI CLI features and workflows.")
        print("Note: Some examples may show warnings if no monitoring data exists.")
        print("=" * 80)
        
        try:
            # Run examples in order
            self.example_basic_setup()
            self.example_framework_detection()
            self.example_basic_monitoring()
            self.example_pattern_analysis()
            self.example_causal_analysis()
            self.example_session_management()
            self.example_publication_output()
            self.example_interactive_mode()
            self.example_configuration_management()
            self.example_advanced_workflows()
            self.example_troubleshooting()
            
            print("\n" + "="*80)
            print("EXAMPLES COMPLETED")
            print("="*80)
            print(f"Example outputs saved to: {self.example_data_dir}")
            print("\nNext steps:")
            print("1. Try the interactive mode: escai --interactive")
            print("2. Set up monitoring for your agents")
            print("3. Explore the help system: escai help")
            print("4. Read the comprehensive guide in docs/cli/")
            
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
        except Exception as e:
            print(f"\n\nError running examples: {e}")


def main():
    """Main function to run CLI examples."""
    examples = ESCAICLIExamples()
    examples.run_all_examples()


if __name__ == "__main__":
    main()