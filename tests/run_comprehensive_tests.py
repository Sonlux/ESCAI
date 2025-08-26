"""
Comprehensive test runner for ESCAI Framework.
Orchestrates unit, integration, performance, load, accuracy, and end-to-end tests.
"""

import asyncio
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.utils.coverage_analyzer import CoverageAnalyzer


class ComprehensiveTestRunner:
    """Orchestrates all types of tests for the ESCAI Framework."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.test_results = {}
        self.start_time = None
        self.coverage_analyzer = CoverageAnalyzer(project_root)
    
    def run_all_tests(
        self, 
        include_performance: bool = True,
        include_load: bool = True,
        include_accuracy: bool = True,
        include_e2e: bool = True,
        coverage_threshold: float = 95.0
    ) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        
        print("🚀 Starting ESCAI Framework Comprehensive Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Test execution plan
        test_plan = [
            ("Unit Tests", self._run_unit_tests, True),
            ("Integration Tests", self._run_integration_tests, True),
            ("Performance Tests", self._run_performance_tests, include_performance),
            ("Load Tests", self._run_load_tests, include_load),
            ("Accuracy Tests", self._run_accuracy_tests, include_accuracy),
            ("End-to-End Tests", self._run_e2e_tests, include_e2e),
            ("Coverage Analysis", self._run_coverage_analysis, True)
        ]
        
        # Execute tests
        for test_name, test_func, should_run in test_plan:
            if should_run:
                print(f"\n📋 Running {test_name}...")
                try:
                    result = test_func()
                    self.test_results[test_name] = result
                    
                    if result.get("success", False):
                        print(f"✅ {test_name} completed successfully")
                    else:
                        print(f"❌ {test_name} failed: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    print(f"💥 {test_name} crashed: {str(e)}")
                    self.test_results[test_name] = {
                        "success": False,
                        "error": str(e),
                        "crashed": True
                    }
            else:
                print(f"⏭️  Skipping {test_name}")
        
        # Generate final report
        final_report = self._generate_final_report(coverage_threshold)
        
        total_time = time.time() - self.start_time
        print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
        
        return final_report
    
    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "unit"),
            "-v", "--tb=short",
            "--durations=10"
        ]
        
        return self._execute_pytest_command(cmd, "unit")
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "integration"),
            "-v", "--tb=short",
            "--durations=10"
        ]
        
        return self._execute_pytest_command(cmd, "integration")
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "performance" / "test_basic_performance.py"),
            "-v", "--tb=short", "-s",
            "--durations=10"
        ]
        
        return self._execute_pytest_command(cmd, "performance")
    
    def _run_load_tests(self) -> Dict[str, Any]:
        """Run load tests."""
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "load" / "test_basic_load.py"),
            "-v", "--tb=short", "-s",
            "--durations=10",
            "--timeout=300"  # 5 minute timeout for load tests
        ]
        
        return self._execute_pytest_command(cmd, "load")
    
    def _run_accuracy_tests(self) -> Dict[str, Any]:
        """Run accuracy validation tests."""
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "accuracy" / "test_basic_accuracy.py"),
            "-v", "--tb=short", "-s",
            "--durations=10"
        ]
        
        return self._execute_pytest_command(cmd, "accuracy")
    
    def _run_e2e_tests(self) -> Dict[str, Any]:
        """Run end-to-end tests."""
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "e2e" / "test_basic_workflow.py"),
            "-v", "--tb=short", "-s",
            "--durations=10"
        ]
        
        return self._execute_pytest_command(cmd, "e2e")
    
    def _run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis."""
        
        try:
            # Run coverage analysis
            coverage_data = self.coverage_analyzer.run_coverage_analysis(include_integration=True)
            
            if not coverage_data:
                return {
                    "success": False,
                    "error": "No coverage data generated",
                    "coverage_percentage": 0
                }
            
            # Calculate overall coverage
            total_coverage = 0
            component_count = 0
            
            for component_coverage in coverage_data.values():
                total_coverage += component_coverage.overall_coverage
                component_count += 1
            
            overall_coverage = total_coverage / component_count if component_count > 0 else 0
            
            # Generate coverage report
            report = self.coverage_analyzer.generate_coverage_report(coverage_data)
            
            # Save coverage report
            coverage_file = self.project_root / "test_coverage_report.txt"
            with open(coverage_file, 'w') as f:
                f.write(report)
            
            return {
                "success": True,
                "coverage_percentage": overall_coverage,
                "components": len(coverage_data),
                "report_file": str(coverage_file),
                "details": coverage_data
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "coverage_percentage": 0
            }
    
    def _execute_pytest_command(self, cmd: List[str], test_type: str) -> Dict[str, Any]:
        """Execute a pytest command and return results."""
        
        try:
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse pytest output for test counts
            output_lines = result.stdout.split('\n')
            
            # Look for pytest summary line
            summary_info = self._parse_pytest_summary(output_lines)
            
            return {
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "tests_run": summary_info.get("total", 0),
                "tests_passed": summary_info.get("passed", 0),
                "tests_failed": summary_info.get("failed", 0),
                "tests_skipped": summary_info.get("skipped", 0),
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"{test_type} tests timed out",
                "execution_time": 600,
                "timeout": True
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "crashed": True
            }
    
    def _parse_pytest_summary(self, output_lines: List[str]) -> Dict[str, int]:
        """Parse pytest output to extract test summary information."""
        
        summary = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        
        for line in output_lines:
            # Look for summary lines like "5 passed, 2 failed, 1 skipped in 10.5s"
            if " passed" in line or " failed" in line or " skipped" in line:
                # Extract numbers from the line
                import re
                
                passed_match = re.search(r'(\d+) passed', line)
                if passed_match:
                    summary["passed"] = int(passed_match.group(1))
                
                failed_match = re.search(r'(\d+) failed', line)
                if failed_match:
                    summary["failed"] = int(failed_match.group(1))
                
                skipped_match = re.search(r'(\d+) skipped', line)
                if skipped_match:
                    summary["skipped"] = int(skipped_match.group(1))
                
                summary["total"] = summary["passed"] + summary["failed"] + summary["skipped"]
                break
        
        return summary
    
    def _generate_final_report(self, coverage_threshold: float) -> Dict[str, Any]:
        """Generate final comprehensive test report."""
        
        total_time = time.time() - self.start_time
        
        # Calculate overall statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        successful_test_types = 0
        total_test_types = 0
        
        for test_name, result in self.test_results.items():
            if test_name == "Coverage Analysis":
                continue
            
            total_test_types += 1
            
            if result.get("success", False):
                successful_test_types += 1
            
            total_tests += result.get("tests_run", 0)
            total_passed += result.get("tests_passed", 0)
            total_failed += result.get("tests_failed", 0)
            total_skipped += result.get("tests_skipped", 0)
        
        # Get coverage information
        coverage_result = self.test_results.get("Coverage Analysis", {})
        coverage_percentage = coverage_result.get("coverage_percentage", 0)
        
        # Determine overall success
        coverage_meets_threshold = coverage_percentage >= coverage_threshold
        all_tests_passed = total_failed == 0 and successful_test_types == total_test_types
        
        overall_success = all_tests_passed and coverage_meets_threshold
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": total_time,
            "overall_success": overall_success,
            "summary": {
                "total_test_types": total_test_types,
                "successful_test_types": successful_test_types,
                "total_tests": total_tests,
                "tests_passed": total_passed,
                "tests_failed": total_failed,
                "tests_skipped": total_skipped,
                "coverage_percentage": coverage_percentage,
                "coverage_threshold": coverage_threshold,
                "coverage_meets_threshold": coverage_meets_threshold
            },
            "test_results": self.test_results,
            "requirements_validation": self._validate_requirements()
        }
        
        # Print summary
        self._print_final_summary(report)
        
        # Save detailed report
        report_file = self.project_root / "comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report
    
    def _validate_requirements(self) -> Dict[str, bool]:
        """Validate that all testing requirements are met."""
        
        requirements = {
            "unit_tests_pass": False,
            "integration_tests_pass": False,
            "performance_requirements_met": False,
            "load_requirements_met": False,
            "accuracy_requirements_met": False,
            "e2e_tests_pass": False,
            "coverage_above_95_percent": False
        }
        
        # Check unit tests
        unit_result = self.test_results.get("Unit Tests", {})
        requirements["unit_tests_pass"] = unit_result.get("success", False) and unit_result.get("tests_failed", 1) == 0
        
        # Check integration tests
        integration_result = self.test_results.get("Integration Tests", {})
        requirements["integration_tests_pass"] = integration_result.get("success", False) and integration_result.get("tests_failed", 1) == 0
        
        # Check performance tests
        performance_result = self.test_results.get("Performance Tests", {})
        requirements["performance_requirements_met"] = performance_result.get("success", False)
        
        # Check load tests
        load_result = self.test_results.get("Load Tests", {})
        requirements["load_requirements_met"] = load_result.get("success", False)
        
        # Check accuracy tests
        accuracy_result = self.test_results.get("Accuracy Tests", {})
        requirements["accuracy_requirements_met"] = accuracy_result.get("success", False)
        
        # Check e2e tests
        e2e_result = self.test_results.get("End-to-End Tests", {})
        requirements["e2e_tests_pass"] = e2e_result.get("success", False)
        
        # Check coverage
        coverage_result = self.test_results.get("Coverage Analysis", {})
        requirements["coverage_above_95_percent"] = coverage_result.get("coverage_percentage", 0) >= 95.0
        
        return requirements
    
    def _print_final_summary(self, report: Dict[str, Any]):
        """Print final test summary."""
        
        print("\n" + "=" * 60)
        print("🏁 COMPREHENSIVE TEST SUITE SUMMARY")
        print("=" * 60)
        
        summary = report["summary"]
        
        print(f"⏱️  Total Execution Time: {report['execution_time_seconds']:.1f} seconds")
        print(f"🧪 Total Tests Run: {summary['total_tests']}")
        print(f"✅ Tests Passed: {summary['tests_passed']}")
        print(f"❌ Tests Failed: {summary['tests_failed']}")
        print(f"⏭️  Tests Skipped: {summary['tests_skipped']}")
        print(f"📊 Code Coverage: {summary['coverage_percentage']:.1f}%")
        
        print("\nTest Type Results:")
        print("-" * 30)
        
        for test_name, result in self.test_results.items():
            if test_name == "Coverage Analysis":
                continue
            
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            tests_info = ""
            
            if "tests_run" in result:
                tests_info = f" ({result['tests_run']} tests)"
            
            execution_time = result.get("execution_time", 0)
            print(f"{status} {test_name}{tests_info} - {execution_time:.1f}s")
        
        # Requirements validation
        requirements = report["requirements_validation"]
        
        print("\nRequirements Validation:")
        print("-" * 30)
        
        for req_name, met in requirements.items():
            status = "✅" if met else "❌"
            readable_name = req_name.replace("_", " ").title()
            print(f"{status} {readable_name}")
        
        # Overall result
        print("\n" + "=" * 60)
        
        if report["overall_success"]:
            print("🎉 ALL REQUIREMENTS MET - COMPREHENSIVE TEST SUITE PASSED!")
        else:
            print("⚠️  SOME REQUIREMENTS NOT MET - ADDITIONAL WORK NEEDED")
            
            # Show what needs to be fixed
            failed_requirements = [name for name, met in requirements.items() if not met]
            if failed_requirements:
                print("\nFailed Requirements:")
                for req in failed_requirements:
                    readable_name = req.replace("_", " ").title()
                    print(f"  - {readable_name}")
        
        print("=" * 60)


def main():
    """Main function to run comprehensive tests."""
    
    parser = argparse.ArgumentParser(description="Run ESCAI Framework comprehensive test suite")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--skip-load", action="store_true", help="Skip load tests")
    parser.add_argument("--skip-accuracy", action="store_true", help="Skip accuracy tests")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip end-to-end tests")
    parser.add_argument("--coverage-threshold", type=float, default=95.0, help="Coverage threshold percentage")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = ComprehensiveTestRunner()
    
    # Run comprehensive test suite
    report = runner.run_all_tests(
        include_performance=not args.skip_performance,
        include_load=not args.skip_load,
        include_accuracy=not args.skip_accuracy,
        include_e2e=not args.skip_e2e,
        coverage_threshold=args.coverage_threshold
    )
    
    # Exit with appropriate code
    sys.exit(0 if report["overall_success"] else 1)


if __name__ == "__main__":
    main()