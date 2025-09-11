#!/usr/bin/env python3
"""
Validation script to ensure all ESCAI imports work correctly.
This script is used in CI/CD to verify the package is properly installed.
"""

import sys
import traceback

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing ESCAI framework imports...")
    
    try:
        # Test core models
        from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState
        print("✅ Epistemic state models imported successfully")
        
        from escai_framework.models.behavioral_pattern import BehavioralPattern, ExecutionSequence, ExecutionStep
        print("✅ Behavioral pattern models imported successfully")
        
        from escai_framework.models.causal_relationship import CausalRelationship, CausalEvent, CausalEvidence
        print("✅ Causal relationship models imported successfully")
        
        from escai_framework.models.prediction_result import PredictionResult
        print("✅ Prediction result model imported successfully")
        
        # Test core components
        from escai_framework.core.epistemic_extractor import EpistemicExtractor
        print("✅ Epistemic extractor imported successfully")
        
        from escai_framework.core.pattern_analyzer import PatternAnalyzer
        print("✅ Pattern analyzer imported successfully")
        
        from escai_framework.core.causal_engine import CausalEngine
        print("✅ Causal engine imported successfully")
        
        from escai_framework.core.performance_predictor import PerformancePredictor
        print("✅ Performance predictor imported successfully")
        
        # Test instrumentation
        from escai_framework.instrumentation.base_instrumentor import BaseInstrumentor
        print("✅ Base instrumentor imported successfully")
        
        from escai_framework.instrumentation.events import AgentEvent, EventType
        print("✅ Event system imported successfully")
        
        # Test utilities
        from escai_framework.utils.validation import ValidationError
        print("✅ Validation utilities imported successfully")
        
        from escai_framework.utils.serialization import serialize_datetime, deserialize_datetime
        print("✅ Serialization utilities imported successfully")
        
        print("\n🎉 All imports successful! ESCAI framework is properly installed.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\n📋 Traceback:")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n📋 Traceback:")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of imported components."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        from escai_framework.models.epistemic_state import EpistemicState, BeliefState
        from datetime import datetime
        
        # Test creating a basic belief state
        belief = BeliefState(
            content="Test belief",
            confidence=0.8,
            timestamp=datetime.now(),
            evidence=["test evidence"]
        )
        
        # Test validation
        assert belief.validate(), "Belief state validation failed"
        print("✅ Belief state creation and validation works")
        
        # Test serialization
        belief_dict = belief.to_dict()
        assert isinstance(belief_dict, dict), "Belief state serialization failed"
        print("✅ Belief state serialization works")
        
        print("\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test error: {e}")
        print("\n📋 Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 ESCAI Framework Import Validation")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n✅ All validation tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Functionality tests failed!")
            sys.exit(1)
    else:
        print("\n❌ Import tests failed!")
        sys.exit(1)