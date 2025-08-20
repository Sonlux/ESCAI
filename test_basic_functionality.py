#!/usr/bin/env python3
"""
Basic functionality test for ESCAI Framework core data models.
This script tests the basic functionality without requiring pytest.
"""

import sys
import traceback
from datetime import datetime, timedelta

def test_epistemic_state():
    """Test EpistemicState functionality."""
    print("Testing EpistemicState...")
    
    try:
        from escai_framework.models.epistemic_state import (
            EpistemicState, BeliefState, KnowledgeState, GoalState,
            BeliefType, GoalStatus
        )
        
        # Test BeliefState
        belief = BeliefState(
            content="Test belief",
            belief_type=BeliefType.FACTUAL,
            confidence=0.9,
            evidence=["test evidence"],
            source="test_source"
        )
        assert belief.validate(), "BeliefState validation failed"
        
        # Test KnowledgeState
        knowledge = KnowledgeState(
            facts=["fact1", "fact2"],
            confidence_score=0.8
        )
        assert knowledge.validate(), "KnowledgeState validation failed"
        
        # Test GoalState
        goal = GoalState(
            description="Test goal",
            status=GoalStatus.ACTIVE,
            priority=5,
            progress=0.5
        )
        assert goal.validate(), "GoalState validation failed"
        
        # Test EpistemicState
        epistemic_state = EpistemicState(
            agent_id="test_agent",
            timestamp=datetime.utcnow(),
            belief_states=[belief],
            knowledge_state=knowledge,
            goal_states=[goal],
            confidence_level=0.8,
            uncertainty_score=0.2
        )
        assert epistemic_state.validate(), "EpistemicState validation failed"
        
        # Test serialization
        json_str = epistemic_state.to_json()
        restored = EpistemicState.from_json(json_str)
        assert restored.agent_id == epistemic_state.agent_id, "JSON serialization failed"
        
        print("✓ EpistemicState tests passed")
        return True
        
    except Exception as e:
        print(f"✗ EpistemicState tests failed: {e}")
        traceback.print_exc()
        return False


def test_behavioral_pattern():
    """Test BehavioralPattern functionality."""
    print("Testing BehavioralPattern...")
    
    try:
        from escai_framework.models.behavioral_pattern import (
            BehavioralPattern, ExecutionSequence, ExecutionStep,
            PatternType, ExecutionStatus
        )
        
        # Test ExecutionStep
        step = ExecutionStep(
            step_id="step_001",
            action="test_action",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
        assert step.validate(), "ExecutionStep validation failed"
        
        # Test ExecutionSequence
        sequence = ExecutionSequence(
            sequence_id="seq_001",
            agent_id="test_agent",
            task_description="Test task",
            steps=[step]
        )
        assert sequence.validate(), "ExecutionSequence validation failed"
        
        # Test BehavioralPattern
        pattern = BehavioralPattern(
            pattern_id="pattern_001",
            pattern_name="Test Pattern",
            pattern_type=PatternType.SEQUENTIAL,
            description="Test pattern description",
            execution_sequences=[sequence]
        )
        assert pattern.validate(), "BehavioralPattern validation failed"
        
        # Test statistics calculation
        pattern.calculate_statistics()
        assert pattern.frequency == 1, "Statistics calculation failed"
        
        # Test serialization
        json_str = pattern.to_json()
        restored = BehavioralPattern.from_json(json_str)
        assert restored.pattern_id == pattern.pattern_id, "JSON serialization failed"
        
        print("✓ BehavioralPattern tests passed")
        return True
        
    except Exception as e:
        print(f"✗ BehavioralPattern tests failed: {e}")
        traceback.print_exc()
        return False


def test_causal_relationship():
    """Test CausalRelationship functionality."""
    print("Testing CausalRelationship...")
    
    try:
        from escai_framework.models.causal_relationship import (
            CausalRelationship, CausalEvent, CausalEvidence,
            CausalType, EvidenceType
        )
        
        # Test CausalEvent
        cause_event = CausalEvent(
            event_id="cause_001",
            event_type="action",
            description="Test cause event",
            timestamp=datetime.utcnow(),
            agent_id="test_agent"
        )
        assert cause_event.validate(), "CausalEvent validation failed"
        
        effect_event = CausalEvent(
            event_id="effect_001",
            event_type="outcome",
            description="Test effect event",
            timestamp=datetime.utcnow() + timedelta(seconds=1),
            agent_id="test_agent"
        )
        assert effect_event.validate(), "CausalEvent validation failed"
        
        # Test CausalEvidence
        evidence = CausalEvidence(
            evidence_type=EvidenceType.STATISTICAL,
            description="Test evidence",
            strength=0.8,
            confidence=0.9,
            source="test_source"
        )
        assert evidence.validate(), "CausalEvidence validation failed"
        
        # Test CausalRelationship
        relationship = CausalRelationship(
            relationship_id="rel_001",
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.DIRECT,
            strength=0.8,
            confidence=0.9,
            delay_ms=1000,
            evidence=[evidence]
        )
        assert relationship.validate(), "CausalRelationship validation failed"
        
        # Test temporal order
        assert relationship.get_temporal_order(), "Temporal order check failed"
        
        # Test serialization
        json_str = relationship.to_json()
        restored = CausalRelationship.from_json(json_str)
        assert restored.relationship_id == relationship.relationship_id, "JSON serialization failed"
        
        print("✓ CausalRelationship tests passed")
        return True
        
    except Exception as e:
        print(f"✗ CausalRelationship tests failed: {e}")
        traceback.print_exc()
        return False


def test_prediction_result():
    """Test PredictionResult functionality."""
    print("Testing PredictionResult...")
    
    try:
        from escai_framework.models.prediction_result import (
            PredictionResult, RiskFactor, Intervention, ConfidenceInterval,
            PredictionType, RiskLevel, InterventionType
        )
        
        # Test RiskFactor
        risk_factor = RiskFactor(
            factor_id="risk_001",
            name="Test Risk",
            description="Test risk factor",
            impact_score=0.7,
            probability=0.6,
            category="test"
        )
        assert risk_factor.validate(), "RiskFactor validation failed"
        
        # Test Intervention
        intervention = Intervention(
            intervention_id="int_001",
            intervention_type=InterventionType.PARAMETER_ADJUSTMENT,
            name="Test Intervention",
            description="Test intervention",
            expected_impact=0.8,
            implementation_cost=0.3,
            urgency=RiskLevel.MEDIUM
        )
        assert intervention.validate(), "Intervention validation failed"
        
        # Test ConfidenceInterval
        confidence_interval = ConfidenceInterval(
            lower_bound=0.6,
            upper_bound=0.9,
            confidence_level=0.95
        )
        assert confidence_interval.validate(), "ConfidenceInterval validation failed"
        
        # Test PredictionResult
        prediction = PredictionResult(
            prediction_id="pred_001",
            agent_id="test_agent",
            prediction_type=PredictionType.SUCCESS_PROBABILITY,
            predicted_value=0.75,
            confidence_score=0.85,
            confidence_interval=confidence_interval,
            risk_factors=[risk_factor],
            recommended_interventions=[intervention],
            created_at=datetime.utcnow()
        )
        assert prediction.validate(), "PredictionResult validation failed"
        
        # Test risk score calculation
        risk_score = prediction.calculate_overall_risk_score()
        assert 0.0 <= risk_score <= 1.0, "Risk score calculation failed"
        
        # Test serialization
        json_str = prediction.to_json()
        restored = PredictionResult.from_json(json_str)
        assert restored.prediction_id == prediction.prediction_id, "JSON serialization failed"
        
        print("✓ PredictionResult tests passed")
        return True
        
    except Exception as e:
        print(f"✗ PredictionResult tests failed: {e}")
        traceback.print_exc()
        return False


def test_validation_utilities():
    """Test validation utilities."""
    print("Testing validation utilities...")
    
    try:
        from escai_framework.utils.validation import (
            validate_string, validate_number, validate_probability,
            validate_datetime, ValidationError
        )
        
        # Test string validation
        result = validate_string("test string", "test_field")
        assert result == "test string", "String validation failed"
        
        # Test number validation
        result = validate_number(42, "test_field")
        assert result == 42, "Number validation failed"
        
        # Test probability validation
        result = validate_probability(0.5, "test_field")
        assert result == 0.5, "Probability validation failed"
        
        # Test datetime validation
        now = datetime.utcnow()
        result = validate_datetime(now, "test_field")
        assert result == now, "Datetime validation failed"
        
        # Test validation error
        try:
            validate_string("", "test_field")  # Should fail
            assert False, "Validation error not raised"
        except ValidationError:
            pass  # Expected
        
        print("✓ Validation utilities tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Validation utilities tests failed: {e}")
        traceback.print_exc()
        return False


def test_serialization_utilities():
    """Test serialization utilities."""
    print("Testing serialization utilities...")
    
    try:
        from escai_framework.utils.serialization import (
            to_json, from_json, to_dict, SerializationError
        )
        
        # Test basic JSON serialization
        data = {"key": "value", "number": 42}
        json_str = to_json(data)
        restored = from_json(json_str)
        assert restored == data, "Basic JSON serialization failed"
        
        # Test to_dict with datetime
        dt = datetime.utcnow()
        result = to_dict(dt)
        assert isinstance(result, str), "Datetime to_dict failed"
        
        # Test complex object serialization
        complex_data = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        json_str = to_json(complex_data)
        restored = from_json(json_str)
        assert restored == complex_data, "Complex JSON serialization failed"
        
        print("✓ Serialization utilities tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Serialization utilities tests failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Running ESCAI Framework Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_epistemic_state,
        test_behavioral_pattern,
        test_causal_relationship,
        test_prediction_result,
        test_validation_utilities,
        test_serialization_utilities
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The ESCAI Framework core functionality is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())