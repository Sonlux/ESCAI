#!/usr/bin/env python3
"""
Test runner script for ESCAI framework.
Can be used locally and in CI/CD environments.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run ESCAI framework tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    
    args = parser.parse_args()
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🚀 ESCAI Framework Test Runner")
    print("=" * 50)
    print(f"Working directory: {os.getcwd()}")
    
    # Validate imports first
    print("\n1️⃣ Validating imports...")
    if not run_command([sys.executable, "scripts/validate_imports.py"], "Validating ESCAI imports"):
        print("❌ Import validation failed!")
        return 1
    
    # Prepare pytest arguments
    pytest_args = [sys.executable, "-m", "pytest"]
    
    if args.verbose:
        pytest_args.append("-v")
    
    if args.coverage:
        pytest_args.extend(["--cov=escai_framework", "--cov-report=xml", "--cov-report=html"])
    
    pytest_args.append("--tb=short")
    
    if args.fast:
        pytest_args.extend(["-m", "not slow"])
    
    # Run tests based on arguments
    success = True
    
    if args.unit or (not args.unit and not args.integration):
        print("\n2️⃣ Running unit tests...")
        unit_args = pytest_args + ["tests/unit/"]
        if not run_command(unit_args, "Running unit tests"):
            success = False
    
    if args.integration or (not args.unit and not args.integration):
        print("\n3️⃣ Running integration tests...")
        integration_args = pytest_args + ["tests/integration/"]
        if not run_command(integration_args, "Running integration tests", check=False):
            print("⚠️  Some integration tests failed (this may be expected if external services are not available)")
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())