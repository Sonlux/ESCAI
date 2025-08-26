#!/usr/bin/env python3
"""
Simple test execution script for ESCAI Framework.
Provides easy commands to run different types of tests.
"""

import sys
import subprocess
from pathlib import Path


def run_unit_tests():
    """Run unit tests only."""
    cmd = [sys.executable, "-m", "pytest", "tests/unit", "-v"]
    return subprocess.run(cmd).returncode


def run_integration_tests():
    """Run integration tests only."""
    cmd = [sys.executable, "-m", "pytest", "tests/integration", "-v"]
    return subprocess.run(cmd).returncode


def run_performance_tests():
    """Run performance tests only."""
    cmd = [sys.executable, "-m", "pytest", "tests/performance", "-v", "-s"]
    return subprocess.run(cmd).returncode


def run_all_tests():
    """Run all tests."""
    cmd = [sys.executable, "tests/run_comprehensive_tests.py"]
    return subprocess.run(cmd).returncode


def run_coverage():
    """Run tests with coverage."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/unit", "tests/integration",
        "--cov=escai_framework",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ]
    return subprocess.run(cmd).returncode


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [unit|integration|performance|all|coverage]")
        sys.exit(1)
    
    test_type = sys.argv[1].lower()
    
    if test_type == "unit":
        exit_code = run_unit_tests()
    elif test_type == "integration":
        exit_code = run_integration_tests()
    elif test_type == "performance":
        exit_code = run_performance_tests()
    elif test_type == "all":
        exit_code = run_all_tests()
    elif test_type == "coverage":
        exit_code = run_coverage()
    else:
        print(f"Unknown test type: {test_type}")
        print("Available types: unit, integration, performance, all, coverage")
        sys.exit(1)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()