"""
Comprehensive CLI test runner for all test categories
"""

import pytest
import sys
import os
from pathlib import Path
import argparse
import time
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CLITestRunner:
    """Comprehensive test runner for CLI testing suite"""
    
    def __init__(self):
        self.test_categories = {
            'unit': {
                'path': 'tests/unit/test_cli_commands.py',
                'description': 'Unit tests for CLI command implementations'
            },
            'integration': {
                'path': 'tests/integration/test_cli_framework_integration.py',
                'description': 'Integration tests for framework interactions'
            },
            'e2e': {
                'path': 'tests/e2e/test_cli_workflows.py',
                'description': 'End-to-end workflow tests'
            },
            'performance': {
                'path': 'tests/performance/test_cli_performance.py',
                'description': 'Performance tests for large dataset handling'
            },
            'ux': {
                'path': 'tests/ux/test_cli_user_experience.py',
                'description': 'User experience tests with simulated interactions'
            },
            'documentation': {
                'path': 'tests/documentation/test_cli_documentation_quality.py',
                'description': 'Documentation quality tests'
            },
            'accessibility': {
                'path': 'tests/accessibility/test_cli_accessibility.py',
                'description': 'Accessibility tests for screen reader compatibility'
            }
        }
    
    def run_category(self, category: str, verbose: bool = False) -> Dict[str, Any]:
        """Run tests for a specific category"""
        if category not in self.test_categories:
            raise ValueError(f"Unknown test category: {category}")
        
        test_info = self.test_categories[category]
        print(f"\n{'='*60}")
        print(f"Running {category.upper()} tests")
        print(f"Description: {test_info['description']}")
        print(f"Path: {test_info['path']}")
        print(f"{'='*60}")
        
        # Prepare pytest arguments
        pytest_args = [test_info['path']]
        
        if verbose:
            pytest_args.extend(['-v', '-s'])
        
        # Add coverage if available
        pytest_args.extend(['--tb=short'])
        
        # Run the tests
        start_time = time.time()
        exit_code = pytest.main(pytest_args)
        end_time = time.time()
        
        duration = end_time - start_time
        
        result = {
            'category': category,
            'exit_code': exit_code,
            'duration': duration,
            'success': exit_code == 0
        }
        
        print(f"\n{category.upper()} tests completed in {duration:.2f}s")
        print(f"Result: {'PASSED' if result['success'] else 'FAILED'}")
        
        return result
    
    def run_all(self, verbose: bool = False, stop_on_failure: bool = False) -> List[Dict[str, Any]]:
        """Run all test categories"""
        print("Starting comprehensive CLI test suite")
        print(f"Test categories: {', '.join(self.test_categories.keys())}")
        
        results = []
        total_start_time = time.time()
        
        for category in self.test_categories.keys():
            try:
                result = self.run_category(category, verbose)
                results.append(result)
                
                if not result['success'] and stop_on_failure:
                    print(f"\nStopping due to failure in {category} tests")
                    break
                    
            except Exception as e:
                print(f"\nError running {category} tests: {e}")
                results.append({
                    'category': category,
                    'exit_code': 1,
                    'duration': 0,
                    'success': False,
                    'error': str(e)
                })
                
                if stop_on_failure:
                    break
        
        total_duration = time.time() - total_start_time
        
        # Print summary
        self.print_summary(results, total_duration)
        
        return results
    
    def run_specific_tests(self, test_patterns: List[str], verbose: bool = False) -> Dict[str, Any]:
        """Run specific test files or patterns"""
        print(f"\nRunning specific tests: {', '.join(test_patterns)}")
        
        pytest_args = test_patterns
        
        if verbose:
            pytest_args.extend(['-v', '-s'])
        
        pytest_args.extend(['--tb=short'])
        
        start_time = time.time()
        exit_code = pytest.main(pytest_args)
        end_time = time.time()
        
        duration = end_time - start_time
        
        result = {
            'category': 'specific',
            'exit_code': exit_code,
            'duration': duration,
            'success': exit_code == 0,
            'patterns': test_patterns
        }
        
        print(f"\nSpecific tests completed in {duration:.2f}s")
        print(f"Result: {'PASSED' if result['success'] else 'FAILED'}")
        
        return result
    
    def print_summary(self, results: List[Dict[str, Any]], total_duration: float):
        """Print test summary"""
        print(f"\n{'='*80}")
        print("CLI TEST SUITE SUMMARY")
        print(f"{'='*80}")
        
        passed = sum(1 for r in results if r['success'])
        failed = len(results) - passed
        
        print(f"Total test categories: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total duration: {total_duration:.2f}s")
        
        print(f"\n{'Category':<15} {'Status':<10} {'Duration':<10} {'Details'}")
        print(f"{'-'*60}")
        
        for result in results:
            status = 'PASSED' if result['success'] else 'FAILED'
            duration = f"{result['duration']:.2f}s"
            details = result.get('error', '')
            
            print(f"{result['category']:<15} {status:<10} {duration:<10} {details}")
        
        print(f"\n{'='*80}")
        
        if failed == 0:
            print("🎉 All CLI tests passed!")
        else:
            print(f"❌ {failed} test categories failed")
        
        print(f"{'='*80}")
    
    def list_categories(self):
        """List available test categories"""
        print("Available test categories:")
        print(f"{'Category':<15} {'Description'}")
        print(f"{'-'*60}")
        
        for category, info in self.test_categories.items():
            print(f"{category:<15} {info['description']}")
    
    def validate_environment(self):
        """Validate test environment"""
        print("Validating test environment...")
        
        # Check if pytest is available
        try:
            import pytest
            print(f"✓ pytest available (version: {pytest.__version__})")
        except ImportError:
            print("❌ pytest not available")
            return False
        
        # Check if CLI module is importable
        try:
            from escai_framework.cli.main import cli
            print("✓ CLI module importable")
        except ImportError as e:
            print(f"❌ CLI module not importable: {e}")
            return False
        
        # Check test directories exist
        missing_dirs = []
        for category, info in self.test_categories.items():
            test_dir = Path(info['path']).parent
            if not test_dir.exists():
                missing_dirs.append(str(test_dir))
        
        if missing_dirs:
            print(f"❌ Missing test directories: {', '.join(missing_dirs)}")
            return False
        else:
            print("✓ All test directories exist")
        
        print("✓ Environment validation passed")
        return True


def main():
    """Main entry point for CLI test runner"""
    parser = argparse.ArgumentParser(
        description="Comprehensive CLI test runner for ESCAI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/cli_test_runner.py --all                    # Run all test categories
  python tests/cli_test_runner.py --category unit          # Run unit tests only
  python tests/cli_test_runner.py --category integration   # Run integration tests
  python tests/cli_test_runner.py --list                   # List available categories
  python tests/cli_test_runner.py --validate               # Validate environment
  python tests/cli_test_runner.py --specific tests/unit/test_cli_commands.py
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all test categories'
    )
    
    parser.add_argument(
        '--category',
        choices=['unit', 'integration', 'e2e', 'performance', 'ux', 'documentation', 'accessibility'],
        help='Run specific test category'
    )
    
    parser.add_argument(
        '--specific',
        nargs='+',
        help='Run specific test files or patterns'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available test categories'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate test environment'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--stop-on-failure',
        action='store_true',
        help='Stop on first test category failure'
    )
    
    args = parser.parse_args()
    
    runner = CLITestRunner()
    
    # Handle different command options
    if args.list:
        runner.list_categories()
        return 0
    
    if args.validate:
        if runner.validate_environment():
            return 0
        else:
            return 1
    
    if args.all:
        results = runner.run_all(verbose=args.verbose, stop_on_failure=args.stop_on_failure)
        return 0 if all(r['success'] for r in results) else 1
    
    if args.category:
        result = runner.run_category(args.category, verbose=args.verbose)
        return 0 if result['success'] else 1
    
    if args.specific:
        result = runner.run_specific_tests(args.specific, verbose=args.verbose)
        return 0 if result['success'] else 1
    
    # Default: show help
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())