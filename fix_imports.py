#!/usr/bin/env python3
"""
Quick fix script to resolve import issues.
This script ensures the package is properly discoverable.
"""

import sys
import os
import subprocess
from pathlib import Path

def fix_python_path():
    """Add current directory to Python path."""
    current_dir = Path.cwd()
    project_root = Path(__file__).parent
    
    # Add both current directory and project root to Python path
    paths_to_add = [str(current_dir), str(project_root)]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"[OK] Added paths to PYTHONPATH:")
    for path in paths_to_add:
        print(f"    {path}")

def install_package_editable():
    """Install the package in editable mode."""
    print("[INFO] Installing package in editable mode...")
    
    try:
        # Try to install in editable mode
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("[OK] Package installed in editable mode successfully")
            return True
        else:
            print(f"[WARN] Editable install failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[WARN] Could not install in editable mode: {e}")
        return False

def test_imports():
    """Test that imports work."""
    print("[INFO] Testing imports...")
    
    try:
        # Test main package
        import escai_framework
        print(f"[OK] Main package imported (v{getattr(escai_framework, '__version__', 'unknown')})")
        
        # Test models
        from escai_framework.models import EpistemicState, BeliefState
        print("[OK] Models imported successfully")
        
        # Test a basic functionality
        from datetime import datetime
        belief = BeliefState(
            content="Test",
            confidence=0.8,
            timestamp=datetime.now(),
            evidence=["test"]
        )
        
        if belief.validate():
            print("[OK] Basic functionality works")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main fix function."""
    print("🔧 ESCAI Framework Import Fix")
    print("=" * 40)
    
    # Step 1: Fix Python path
    fix_python_path()
    
    # Step 2: Try editable install (optional)
    install_package_editable()
    
    # Step 3: Test imports
    if test_imports():
        print("\n✅ SUCCESS: All imports are working!")
        print("\n🚀 ESCAI Framework is ready to use.")
        return 0
    else:
        print("\n❌ FAILURE: Import issues persist.")
        print("\n💡 Try running: python -m pip install -e .")
        return 1

if __name__ == "__main__":
    sys.exit(main())