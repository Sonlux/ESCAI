"""
Basic accuracy tests for ESCAI Framework.
Tests basic accuracy of existing components.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta

from escai_framework.core.epistemic_extractor import EpistemicExtractor
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState


@pytest.mark.asyncio
class TestBasicAccuracy:
    """Test basic accuracy of existing components."""
    
    async def test_epistemic_extraction_accuracy(self):
        """Test that epistemic extraction can process events without errors."""
        extractor = EpistemicExtractor()
        
        # Create test events with known patterns
        from escai_framework.instrumentation.events import AgentEvent, EventType
        
        logs = [
            AgentEvent(
                event_id="test_event_1",
                agent_id="test_agent",
                event_type=EventType.DECISION_START,
                timestamp=datetime.now(),
                session_id="test_session",
                message="Starting data analysis task",
                data={"confidence": 0.8, "task": "analysis"}
            ),
            AgentEvent(
                event_id="test_event_2",
                agent_id="test_agent",
                event_type=EventType.DECISION_COMPLETE,
                timestamp=datetime.now(),
                session_id="test_session",
                message="Data loaded successfully",
                data={"confidence": 0.9, "success": True}
            ),
            AgentEvent(
                event_id="test_event_3",
                agent_id="test_agent",
                event_type=EventType.BELIEF_UPDATE,
                timestamp=datetime.now(),
                session_id="test_session",
                message="Missing values detected",
                data={"confidence": 0.6, "issue": "data_quality"}
            )
        ]
        
        # Extract beliefs (this tests that the extractor can process events without errors)
        beliefs = await extractor.extract_beliefs(logs)
        
        # Basic validation - the extractor should at least process the events without crashing
        assert isinstance(beliefs, list), "Beliefs should be returned as a list"
        
        # If beliefs are extracted, they should have valid structure
        for belief in beliefs:
            assert hasattr(belief, 'confidence'), "Belief should have confidence attribute"
            assert hasattr(belief, 'content'), "Belief should have content attribute"
            assert 0 <= belief.confidence <= 1, f"Invalid confidence value: {belief.confidence}"
        
        print(f"Epistemic extraction test: processed {len(logs)} events, extracted {len(beliefs)} beliefs")
    
    async def test_belief_state_validation_accuracy(self):
        """Test belief state validation accuracy."""
        
        # Test valid belief states
        valid_beliefs = [
            BeliefState(
                belief_id="test_1",
                content="Valid belief content",
                confidence=0.8,
                timestamp=datetime.now(),
                evidence=["evidence1"]
            ),
            BeliefState(
                belief_id="test_2", 
                content="Another valid belief",
                confidence=0.9,
                timestamp=datetime.now(),
                evidence=["evidence2", "evidence3"]
            )
        ]
        
        # All valid beliefs should pass validation
        for belief in valid_beliefs:
            try:
                belief.validate()
                valid_count = 1
            except Exception:
                valid_count = 0
            
            assert valid_count == 1, f"Valid belief failed validation: {belief.content}"
        
        # Test invalid belief states
        invalid_beliefs = [
            # Invalid confidence (> 1.0)
            {
                "belief_id": "invalid_1",
                "content": "Invalid confidence",
                "confidence": 1.5,
                "timestamp": datetime.now(),
                "evidence": ["evidence"]
            },
            # Invalid confidence (< 0.0)
            {
                "belief_id": "invalid_2", 
                "content": "Negative confidence",
                "confidence": -0.1,
                "timestamp": datetime.now(),
                "evidence": ["evidence"]
            },
            # Empty content
            {
                "belief_id": "invalid_3",
                "content": "",
                "confidence": 0.8,
                "timestamp": datetime.now(),
                "evidence": ["evidence"]
            }
        ]
        
        invalid_count = 0
        for belief_data in invalid_beliefs:
            try:
                belief = BeliefState(**belief_data)
                belief.validate()
                # Should not reach here
                invalid_count += 1
            except Exception:
                # Expected to fail validation
                pass
        
        # All invalid beliefs should fail validation
        assert invalid_count == 0, f"{invalid_count} invalid beliefs passed validation"
        
        print("Belief state validation accuracy: 100%")
    
    async def test_epistemic_state_consistency(self):
        """Test epistemic state consistency."""
        
        # Create consistent epistemic state
        belief_state = BeliefState(
            belief_id="test_belief",
            content="Test belief",
            confidence=0.8,
            timestamp=datetime.now(),
            evidence=["test evidence"]
        )
        
        knowledge_state = KnowledgeState(
            facts=["fact1", "fact2"],
            concepts=["concept1"],
            relationships={"rel1": ["item1", "item2"]},
            timestamp=datetime.now()
        )
        
        goal_state = GoalState(
            primary_goal="Test goal",
            sub_goals=["sub1", "sub2"],
            progress=0.5,
            timestamp=datetime.now()
        )
        
        epistemic_state = EpistemicState(
            agent_id="test_agent",
            timestamp=datetime.now(),
            belief_states=[belief_state],
            knowledge_state=knowledge_state,
            goal_state=goal_state,
            confidence_level=0.8,
            uncertainty_score=0.2,
            decision_context={"test": "context"}
        )
        
        # Validate consistency
        try:
            epistemic_state.validate()
            consistency_check = True
        except Exception as e:
            consistency_check = False
            print(f"Consistency validation failed: {e}")
        
        assert consistency_check, "Epistemic state consistency check failed"
        
        # Check that confidence and uncertainty are complementary
        total_certainty = epistemic_state.confidence_level + epistemic_state.uncertainty_score
        assert abs(total_certainty - 1.0) < 0.1, f"Confidence and uncertainty don't sum to ~1.0: {total_certainty}"
        
        print("Epistemic state consistency: 100%")
    
    async def test_data_serialization_accuracy(self):
        """Test data serialization/deserialization accuracy."""
        
        # Create original epistemic state
        original_belief = BeliefState(
            belief_id="serialize_test",
            content="Serialization test belief",
            confidence=0.75,
            timestamp=datetime.now(),
            evidence=["evidence1", "evidence2"]
        )
        
        original_knowledge = KnowledgeState(
            facts=["serialization fact"],
            concepts=["serialization"],
            relationships={"test": ["serialize", "deserialize"]},
            timestamp=datetime.now()
        )
        
        original_goal = GoalState(
            primary_goal="Test serialization",
            sub_goals=["serialize", "deserialize", "validate"],
            progress=0.33,
            timestamp=datetime.now()
        )
        
        original_state = EpistemicState(
            agent_id="serialize_agent",
            timestamp=datetime.now(),
            belief_states=[original_belief],
            knowledge_state=original_knowledge,
            goal_state=original_goal,
            confidence_level=0.75,
            uncertainty_score=0.25,
            decision_context={"serialization": "test"}
        )
        
        # Serialize to dict
        serialized = original_state.to_dict()
        
        # Deserialize from dict
        deserialized = EpistemicState.from_dict(serialized)
        
        # Check accuracy of serialization/deserialization
        accuracy_checks = [
            deserialized.agent_id == original_state.agent_id,
            len(deserialized.belief_states) == len(original_state.belief_states),
            deserialized.confidence_level == original_state.confidence_level,
            deserialized.uncertainty_score == original_state.uncertainty_score,
            deserialized.belief_states[0].content == original_state.belief_states[0].content,
            deserialized.knowledge_state.facts == original_state.knowledge_state.facts,
            deserialized.goal_state.primary_goal == original_state.goal_state.primary_goal
        ]
        
        accuracy = sum(accuracy_checks) / len(accuracy_checks)
        
        assert accuracy >= 0.9, f"Serialization accuracy {accuracy:.2f} below 90%"
        
        print(f"Data serialization accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    # Run basic accuracy tests
    pytest.main([__file__, "-v", "--tb=short"])