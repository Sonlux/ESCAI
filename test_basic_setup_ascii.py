#!/usr/bin/env python3
"""Basic setup test to verify ESCAI framework installation and functionality (ASCII only)."""

import sys
import traceback
from datetime import datetime

def test_basic_imports():
    """Test that all basic imports work correctly."""
    print("Testing basic imports...")
    
    try:
        # Test main package import
        import escai_framework
        print(f"[OK] ESCAI Framework imported (version: {getattr(escai_framework, '__version__', 'unknown')})")
        
        # Test model imports
        from escai_framework.models import (
            EpistemicState,
            BeliefState,
            KnowledgeState,
            GoalState,
            BehavioralPattern,
            CausalRelationship,
            PredictionResult,
        )
        print("[OK] All model classes imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")
    
    try:
        from escai_framework.models.epistemic_state import BeliefState
        
        # Test creating a belief state
        belief = BeliefState(
            content="Test belief for setup verification",
            confidence=0.85,
            timestamp=datetime.now(),
            evidence=["setup test evidence"]
        )
        
        # Test validation
        if belief.validate():
            print("[OK] BeliefState creation and validation works")
        else:
            print("[ERROR] BeliefState validation failed")
            return False
        
        # Test serialization
        belief_dict = belief.to_dict()
        if isinstance(belief_dict, dict) and 'content' in belief_dict:
            print("[OK] BeliefState serialization works")
        else:
            print("[ERROR] BeliefState serialization failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_package_structure():
    """Test that the package structure is correct."""
    print("\nTesting package structure...")
    
    try:
        import escai_framework
        import escai_framework.models
        import escai_framework.core
        import escai_framework.instrumentation
        
        print("[OK] All main packages are accessible")
        
        # Test that __all__ exports work
        from escai_framework import EpistemicState, BeliefState
        print("[OK] Main package exports work correctly")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Package structure test failed: {e}")
        return False

def main():
    """Run all basic setup tests."""
    print("ESCAI Framework Basic Setup Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_basic_functionality,
        test_package_structure,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n[ERROR] Test {test.__name__} failed!")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All basic setup tests passed! ESCAI Framework is ready to use.")
        return 0
    else:
        print("FAILURE: Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())