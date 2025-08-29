#!/usr/bin/env python3
"""
Quick validation script to test if our mypy fixes are working.
This script imports key modules to check for basic syntax and import errors.
"""

import sys
import traceback

def test_imports():
    """Test importing key modules that had mypy issues."""
    modules_to_test = [
        'escai_framework.analytics.model_evaluation',
        'escai_framework.core.epistemic_extractor', 
        'escai_framework.api.monitoring',
        'escai_framework.utils.circuit_breaker',
        'escai_framework.utils.error_tracking',
        'escai_framework.utils.fallback',
        'escai_framework.utils.load_shedding',
        'escai_framework.security.input_validator',
        'escai_framework.instrumentation.crewai_instrumentor',
        'escai_framework.instrumentation.autogen_instrumentor',
        'escai_framework.instrumentation.openai_instrumentor',
    ]
    
    results = []
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            results.append(f"✅ {module_name}: Import successful")
        except ImportError as e:
            # Expected for optional dependencies
            if "No module named" in str(e) and any(dep in str(e) for dep in ['transformers', 'torch', 'networkx', 'sklearn']):
                results.append(f"⚠️  {module_name}: Import failed due to optional dependency: {e}")
            else:
                results.append(f"❌ {module_name}: Import failed: {e}")
        except SyntaxError as e:
            results.append(f"❌ {module_name}: Syntax error: {e}")
        except Exception as e:
            results.append(f"❌ {module_name}: Unexpected error: {e}")
    
    return results

def test_basic_functionality():
    """Test basic functionality of key classes."""
    try:
        # Test basic class instantiation
        from escai_framework.utils.exceptions import ESCAIBaseException
        from escai_framework.models.epistemic_state import EpistemicState
        
        # Test exception
        try:
            raise ESCAIBaseException("Test error")
        except ESCAIBaseException:
            pass
        
        # Test basic model creation
        from datetime import datetime
        state = EpistemicState(
            agent_id="test",
            timestamp=datetime.utcnow()
        )
        
        return ["✅ Basic functionality test passed"]
        
    except Exception as e:
        return [f"❌ Basic functionality test failed: {e}"]

if __name__ == "__main__":
    print("🔍 Testing mypy fixes...")
    print("=" * 50)
    
    # Test imports
    print("\n📦 Testing module imports:")
    import_results = test_imports()
    for result in import_results:
        print(result)
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality:")
    func_results = test_basic_functionality()
    for result in func_results:
        print(result)
    
    # Summary
    print("\n" + "=" * 50)
    failed_imports = [r for r in import_results if r.startswith("❌")]
    failed_funcs = [r for r in func_results if r.startswith("❌")]
    
    if failed_imports or failed_funcs:
        print(f"❌ {len(failed_imports + failed_funcs)} issues found")
        sys.exit(1)
    else:
        print("✅ All tests passed! Mypy fixes appear to be working.")
        sys.exit(0)