#!/usr/bin/env python3
"""Validate that all ESCAI framework modules can be imported correctly."""

import sys
import os
import traceback
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def validate_imports():
    """Test importing all main modules."""
    errors = []

    try:
        # Test main package import
        import escai_framework
        print("[OK] ESCAI Framework package imported successfully")

        # Test models import
        from escai_framework.models import (
            EpistemicState,
            BeliefState,
            KnowledgeState,
            GoalState,
            BehavioralPattern,
            ExecutionSequence,
            CausalRelationship,
            PredictionResult,
        )
        print("[OK] All model classes imported successfully")

        # Test individual module imports
        modules_to_test = [
            "escai_framework.models.epistemic_state",
            "escai_framework.models.behavioral_pattern",
            "escai_framework.models.causal_relationship",
            "escai_framework.models.prediction_result",
        ]

        for module_name in modules_to_test:
            try:
                __import__(module_name)
                print(f"[OK] {module_name} imported successfully")
            except ImportError as e:
                errors.append(f"[ERROR] Failed to import {module_name}: {e}")

        # Test core components if they exist
        try:
            from escai_framework.core.epistemic_extractor import EpistemicExtractor
            print("[OK] Epistemic extractor imported successfully")
        except ImportError:
            print("[WARN] Epistemic extractor not available (optional)")

        try:
            from escai_framework.core.pattern_analyzer import PatternAnalyzer
            print("[OK] Pattern analyzer imported successfully")
        except ImportError:
            print("[WARN] Pattern analyzer not available (optional)")

        try:
            from escai_framework.core.causal_engine import CausalEngine
            print("[OK] Causal engine imported successfully")
        except ImportError:
            print("[WARN] Causal engine not available (optional)")

        try:
            from escai_framework.core.performance_predictor import PerformancePredictor
            print("[OK] Performance predictor imported successfully")
        except ImportError:
            print("[WARN] Performance predictor not available (optional)")

        # Test instrumentation if available
        try:
            from escai_framework.instrumentation.base_instrumentor import BaseInstrumentor
            print("[OK] Base instrumentor imported successfully")
        except ImportError:
            print("[WARN] Base instrumentor not available (optional)")

        try:
            from escai_framework.instrumentation.events import AgentEvent, EventType
            print("[OK] Event system imported successfully")
        except ImportError:
            print("[WARN] Event system not available (optional)")

        # Test utilities if available
        try:
            from escai_framework.utils.validation import ValidationError
            print("[OK] Validation utilities imported successfully")
        except ImportError:
            print("[WARN] Validation utilities not available (optional)")

        try:
            from escai_framework.utils.serialization import serialize_datetime, deserialize_datetime
            print("[OK] Serialization utilities imported successfully")
        except ImportError:
            print("[WARN] Serialization utilities not available (optional)")

    except ImportError as e:
        errors.append(f"[ERROR] Main import failed: {e}")

    if errors:
        print("\nImport validation failed:")
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print("\n[SUCCESS] All imports validated successfully!")
        sys.exit(0)

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
    validate_imports()