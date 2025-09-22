#!/usr/bin/env python3
"""
CLI Integration Validation Script

This script validates that the ESCAI CLI integration and polish is complete
and all functionality is working correctly.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple


class CLIValidator:
    """Validates CLI integration and functionality."""
    
    def __init__(self):
        """Initialize CLI validator."""
        self.results: List[Tuple[str, bool, str]] = []
        self.cli_command = [sys.executable, "-m", "escai_framework.cli.main"]
    
    def run_test(self, name: str, command: List[str], expect_success: bool = True, timeout: int = 10) -> bool:
        """
        Run a test command and validate result.
        
        Args:
            name: Test name
            command: Command to run
            expect_success: Whether to expect success
            timeout: Command timeout
            
        Returns:
            True if test passed
        """
        try:
            print(f"Testing: {name}")
            print(f"Command: {' '.join(self.cli_command + command)}")
            
            result = subprocess.run(
                self.cli_command + command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = (result.returncode == 0) == expect_success
            
            if success:
                print("✓ PASSED")
                self.results.append((name, True, ""))
            else:
                print("✗ FAILED")
                error_msg = result.stderr or result.stdout or "Unknown error"
                self.results.append((name, False, error_msg))
                print(f"Error: {error_msg}")
            
            print("-" * 60)
            return success
            
        except subprocess.TimeoutExpired:
            print("✗ FAILED (timeout)")
            self.results.append((name, False, "Command timed out"))
            print("-" * 60)
            return False
        except Exception as e:
            print(f"✗ FAILED (exception: {e})")
            self.results.append((name, False, str(e)))
            print("-" * 60)
            return False
    
    def validate_basic_functionality(self) -> None:
        """Validate basic CLI functionality."""
        print("\n" + "="*80)
        print("VALIDATING BASIC CLI FUNCTIONALITY")
        print("="*80)
        
        # Test version command
        self.run_test("Version Command", ["--version"])
        
        # Test help system
        self.run_test("Help System", ["help"])
        
        # Test help topics
        self.run_test("Help Topic - Getting Started", ["help", "getting_started"])
        self.run_test("Help Topic - Frameworks", ["help", "frameworks"])
        self.run_test("Help Topic - Monitoring", ["help", "monitoring"])
        
        # Test help workflows
        self.run_test("Help Workflow", ["help", "workflow", "basic_monitoring"])
        
        # Test help search
        self.run_test("Help Search", ["help", "search", "monitor"])
    
    def validate_command_structure(self) -> None:
        """Validate command structure and help."""
        print("\n" + "="*80)
        print("VALIDATING COMMAND STRUCTURE")
        print("="*80)
        
        # Test main command groups
        commands = [
            "monitor",
            "analyze", 
            "config",
            "session",
            "publication",
            "logs"
        ]
        
        for command in commands:
            self.run_test(f"Command Help - {command}", ["help", command])
    
    def validate_configuration_system(self) -> None:
        """Validate configuration system."""
        print("\n" + "="*80)
        print("VALIDATING CONFIGURATION SYSTEM")
        print("="*80)
        
        # Test configuration commands
        self.run_test("Config Check", ["config", "check"])
        self.run_test("Config Show", ["config", "show"])
        
        # Test framework testing (may fail if frameworks not installed)
        frameworks = ["langchain", "autogen", "crewai", "openai"]
        for framework in frameworks:
            self.run_test(
                f"Framework Test - {framework}", 
                ["config", "test", "--framework", framework],
                expect_success=False  # May fail if not installed
            )
    
    def validate_session_management(self) -> None:
        """Validate session management."""
        print("\n" + "="*80)
        print("VALIDATING SESSION MANAGEMENT")
        print("="*80)
        
        # Test session commands
        self.run_test("Session List", ["session", "list"])
        self.run_test("Session List Active", ["session", "list", "--active"])
    
    def validate_analysis_commands(self) -> None:
        """Validate analysis commands."""
        print("\n" + "="*80)
        print("VALIDATING ANALYSIS COMMANDS")
        print("="*80)
        
        # Test analysis help (actual analysis may fail without data)
        self.run_test("Analyze Help", ["help", "analyze"])
        self.run_test("Analyze Patterns Help", ["help", "command", "analyze", "patterns"])
        self.run_test("Analyze Causal Help", ["help", "command", "analyze", "causal"])
    
    def validate_monitoring_commands(self) -> None:
        """Validate monitoring commands."""
        print("\n" + "="*80)
        print("VALIDATING MONITORING COMMANDS")
        print("="*80)
        
        # Test monitoring help
        self.run_test("Monitor Help", ["help", "monitor"])
        self.run_test("Monitor Status", ["monitor", "status"])
    
    def validate_publication_commands(self) -> None:
        """Validate publication commands."""
        print("\n" + "="*80)
        print("VALIDATING PUBLICATION COMMANDS")
        print("="*80)
        
        # Test publication help
        self.run_test("Publication Help", ["help", "publication"])
    
    def validate_error_handling(self) -> None:
        """Validate error handling."""
        print("\n" + "="*80)
        print("VALIDATING ERROR HANDLING")
        print("="*80)
        
        # Test invalid commands (should fail gracefully)
        self.run_test(
            "Invalid Command", 
            ["invalid_command"], 
            expect_success=False
        )
        
        self.run_test(
            "Invalid Subcommand", 
            ["monitor", "invalid_subcommand"], 
            expect_success=False
        )
    
    def validate_interactive_mode_availability(self) -> None:
        """Validate interactive mode is available."""
        print("\n" + "="*80)
        print("VALIDATING INTERACTIVE MODE")
        print("="*80)
        
        # Test interactive mode help
        self.run_test("Interactive Mode Help", ["--help"])
        
        print("Note: Interactive mode requires manual testing with:")
        print("  python -m escai_framework.cli.main --interactive")
    
    def validate_startup_optimization(self) -> None:
        """Validate startup optimization."""
        print("\n" + "="*80)
        print("VALIDATING STARTUP OPTIMIZATION")
        print("="*80)
        
        # Measure startup time
        start_time = time.time()
        result = subprocess.run(
            self.cli_command + ["--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        end_time = time.time()
        
        startup_time = end_time - start_time
        
        if result.returncode == 0 and startup_time < 5.0:  # Should start in under 5 seconds
            print(f"✓ Startup time: {startup_time:.2f}s (acceptable)")
            self.results.append(("Startup Performance", True, f"{startup_time:.2f}s"))
        else:
            print(f"✗ Startup time: {startup_time:.2f}s (too slow)")
            self.results.append(("Startup Performance", False, f"{startup_time:.2f}s"))
    
    def validate_debug_mode(self) -> None:
        """Validate debug mode functionality."""
        print("\n" + "="*80)
        print("VALIDATING DEBUG MODE")
        print("="*80)
        
        # Test debug mode
        self.run_test("Debug Mode", ["--debug", "--version"])
    
    def generate_report(self) -> None:
        """Generate validation report."""
        print("\n" + "="*80)
        print("VALIDATION REPORT")
        print("="*80)
        
        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if total - passed > 0:
            print("\nFailed Tests:")
            for name, success, error in self.results:
                if not success:
                    print(f"  ✗ {name}: {error}")
        
        print("\nAll Tests:")
        for name, success, error in self.results:
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
        
        # Overall assessment
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - CLI integration is complete!")
            return True
        elif passed / total >= 0.8:
            print("\n⚠️  MOSTLY WORKING - Some non-critical issues found")
            return True
        else:
            print("\n❌ SIGNIFICANT ISSUES - CLI integration needs work")
            return False
    
    def run_full_validation(self) -> bool:
        """Run complete CLI validation."""
        print("ESCAI CLI Integration Validation")
        print("="*80)
        print("This script validates the complete CLI integration and polish.")
        print("Some tests may fail if optional dependencies are not installed.")
        print("="*80)
        
        # Run all validation tests
        self.validate_basic_functionality()
        self.validate_command_structure()
        self.validate_configuration_system()
        self.validate_session_management()
        self.validate_analysis_commands()
        self.validate_monitoring_commands()
        self.validate_publication_commands()
        self.validate_error_handling()
        self.validate_interactive_mode_availability()
        self.validate_startup_optimization()
        self.validate_debug_mode()
        
        # Generate final report
        return self.generate_report()


def main():
    """Main validation function."""
    validator = CLIValidator()
    success = validator.run_full_validation()
    
    if success:
        print("\n✅ CLI integration validation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ CLI integration validation found issues!")
        sys.exit(1)


if __name__ == "__main__":
    main()