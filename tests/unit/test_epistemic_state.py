"""
Unit tests for EpistemicState data model.
"""

import pytest
from datetime import datetime, timedelta
import json

from escai_framework.models.epistemic_state import (
    EpistemicState, BeliefState, KnowledgeState, GoalState,
    BeliefType, GoalStatus
)


class TestBeliefState:
    """Test cases for BeliefState model."""
    
    def test_belief_state_creation(self):
        """Test creating a valid BeliefState."""
        belief = BeliefState(
            content="The weather is sunny",
            belief_type=BeliefType.FACTUAL,
            confidence=0.9,
            evidence=["weather report", "visual observation"],
            source="weather_sensor"
        )
        
        assert belief.content == "The weather is sunny"
        assert belief.belief_type == BeliefType.FACTUAL
        assert belief.confidence == 0.9
        assert belief.evidence == ["weather report", "visual observation"]
        assert belief.source == "weather_sensor"
        assert isinstance(belief.timestamp, datetime)
    
    def test_belief_state_validation_valid(self):
        """Test validation of valid BeliefState."""
        belief = BeliefState(
            content="Valid belief",
            belief_type=BeliefType.PREDICTIVE,
            confidence=0.7
        )
        assert belief.validate() is True
    
    def test_belief_state_validation_invalid_content(self):
        """Test validation with invalid content."""
        belief = BeliefState(
            content="",
            belief_type=BeliefType.FACTUAL,
            confidence=0.8
        )
        assert belief.validate() is False
        
        belief.content = 123
        assert belief.validate() is False
    
    def test_belief_state_validation_invalid_confidence(self):
        """Test validation with invalid confidence."""
        belief = BeliefState(
            content="Valid content",
            belief_type=BeliefType.FACTUAL,
            confidence=1.5
        )
        assert belief.validate() is False
        
        belief.confidence = -0.1
        assert belief.validate() is False
    
    def test_belief_state_serialization(self):
        """Test BeliefState serialization and deserialization."""
        original = BeliefState(
            content="Test belief",
            belief_type=BeliefType.CONDITIONAL,
            confidence=0.6,
            evidence=["test evidence"],
            source="test_source"
        )
        
        # Test to_dict
        data = original.to_dict()
        assert data["content"] == "Test belief"
        assert data["belief_type"] == "conditional"
        assert data["confidence"] == 0.6
        
        # Test from_dict
        restored = BeliefState.from_dict(data)
        assert restored.content == original.content
        assert restored.belief_type == original.belief_type
        assert restored.confidence == original.confidence
        assert restored.evidence == original.evidence
        assert restored.source == original.source


class TestKnowledgeState:
    """Test cases for KnowledgeState model."""
    
    def test_knowledge_state_creation(self):
        """Test creating a valid KnowledgeState."""
        knowledge = KnowledgeState(
            facts=["fact1", "fact2"],
            rules=["rule1", "rule2"],
            concepts={"concept1": "value1"},
            relationships=[{"from": "A", "to": "B", "type": "causes"}],
            confidence_score=0.8
        )
        
        assert knowledge.facts == ["fact1", "fact2"]
        assert knowledge.rules == ["rule1", "rule2"]
        assert knowledge.concepts == {"concept1": "value1"}
        assert knowledge.relationships == [{"from": "A", "to": "B", "type": "causes"}]
        assert knowledge.confidence_score == 0.8
    
    def test_knowledge_state_validation(self):
        """Test KnowledgeState validation."""
        knowledge = KnowledgeState()
        assert knowledge.validate() is True
        
        knowledge.confidence_score = 1.5
        assert knowledge.validate() is False
        
        knowledge.confidence_score = 0.5
        knowledge.facts = "not a list"
        assert knowledge.validate() is False
    
    def test_knowledge_state_serialization(self):
        """Test KnowledgeState serialization."""
        original = KnowledgeState(
            facts=["test fact"],
            rules=["test rule"],
            concepts={"test": "concept"},
            confidence_score=0.7
        )
        
        data = original.to_dict()
        restored = KnowledgeState.from_dict(data)
        
        assert restored.facts == original.facts
        assert restored.rules == original.rules
        assert restored.concepts == original.concepts
        assert restored.confidence_score == original.confidence_score


class TestGoalState:
    """Test cases for GoalState model."""
    
    def test_goal_state_creation(self):
        """Test creating a valid GoalState."""
        deadline = datetime.utcnow() + timedelta(days=1)
        goal = GoalState(
            description="Complete task",
            status=GoalStatus.ACTIVE,
            priority=5,
            progress=0.3,
            sub_goals=["subtask1", "subtask2"],
            deadline=deadline
        )
        
        assert goal.description == "Complete task"
        assert goal.status == GoalStatus.ACTIVE
        assert goal.priority == 5
        assert goal.progress == 0.3
        assert goal.sub_goals == ["subtask1", "subtask2"]
        assert goal.deadline == deadline
    
    def test_goal_state_validation(self):
        """Test GoalState validation."""
        goal = GoalState(
            description="Valid goal",
            status=GoalStatus.ACTIVE,
            priority=5,
            progress=0.5
        )
        assert goal.validate() is True
        
        # Test invalid priority
        goal.priority = 15
        assert goal.validate() is False
        
        goal.priority = 5
        goal.progress = 1.5
        assert goal.validate() is False
    
    def test_goal_state_serialization(self):
        """Test GoalState serialization."""
        deadline = datetime.utcnow() + timedelta(hours=2)
        original = GoalState(
            description="Test goal",
            status=GoalStatus.COMPLETED,
            priority=8,
            progress=1.0,
            deadline=deadline
        )
        
        data = original.to_dict()
        restored = GoalState.from_dict(data)
        
        assert restored.description == original.description
        assert restored.status == original.status
        assert restored.priority == original.priority
        assert restored.progress == original.progress
        assert restored.deadline == original.deadline


class TestEpistemicState:
    """Test cases for EpistemicState model."""
    
    def test_epistemic_state_creation(self):
        """Test creating a valid EpistemicState."""
        belief = BeliefState(
            content="Test belief",
            belief_type=BeliefType.FACTUAL,
            confidence=0.8
        )
        
        knowledge = KnowledgeState(
            facts=["fact1"],
            confidence_score=0.7
        )
        
        goal = GoalState(
            description="Test goal",
            status=GoalStatus.ACTIVE,
            priority=5,
            progress=0.4
        )
        
        timestamp = datetime.utcnow()
        epistemic_state = EpistemicState(
            agent_id="test_agent",
            timestamp=timestamp,
            belief_states=[belief],
            knowledge_state=knowledge,
            goal_states=[goal],
            confidence_level=0.75,
            uncertainty_score=0.25,
            decision_context={"context_key": "context_value"}
        )
        
        assert epistemic_state.agent_id == "test_agent"
        assert epistemic_state.timestamp == timestamp
        assert len(epistemic_state.belief_states) == 1
        assert epistemic_state.knowledge_state == knowledge
        assert len(epistemic_state.goal_states) == 1
        assert epistemic_state.confidence_level == 0.75
        assert epistemic_state.uncertainty_score == 0.25
        assert epistemic_state.decision_context == {"context_key": "context_value"}
    
    def test_epistemic_state_validation(self):
        """Test EpistemicState validation."""
        epistemic_state = EpistemicState(
            agent_id="test_agent",
            timestamp=datetime.utcnow()
        )
        assert epistemic_state.validate() is True
        
        # Test invalid agent_id
        epistemic_state.agent_id = ""
        assert epistemic_state.validate() is False
        
        epistemic_state.agent_id = "test_agent"
        epistemic_state.confidence_level = 1.5
        assert epistemic_state.validate() is False
    
    def test_epistemic_state_serialization(self):
        """Test EpistemicState serialization."""
        belief = BeliefState(
            content="Serialization test",
            belief_type=BeliefType.PROBABILISTIC,
            confidence=0.9
        )
        
        original = EpistemicState(
            agent_id="serialization_agent",
            timestamp=datetime.utcnow(),
            belief_states=[belief],
            confidence_level=0.8,
            uncertainty_score=0.2
        )
        
        # Test to_dict
        data = original.to_dict()
        assert data["agent_id"] == "serialization_agent"
        assert len(data["belief_states"]) == 1
        
        # Test from_dict
        restored = EpistemicState.from_dict(data)
        assert restored.agent_id == original.agent_id
        assert len(restored.belief_states) == 1
        assert restored.belief_states[0].content == belief.content
        assert restored.confidence_level == original.confidence_level
        assert restored.uncertainty_score == original.uncertainty_score
    
    def test_epistemic_state_json_serialization(self):
        """Test EpistemicState JSON serialization."""
        epistemic_state = EpistemicState(
            agent_id="json_test_agent",
            timestamp=datetime.utcnow(),
            confidence_level=0.6,
            uncertainty_score=0.4
        )
        
        # Test to_json
        json_str = epistemic_state.to_json()
        assert isinstance(json_str, str)
        
        # Test from_json
        restored = EpistemicState.from_json(json_str)
        assert restored.agent_id == epistemic_state.agent_id
        assert restored.confidence_level == epistemic_state.confidence_level
        assert restored.uncertainty_score == epistemic_state.uncertainty_score
    
    def test_epistemic_state_with_invalid_components(self):
        """Test EpistemicState validation with invalid components."""
        invalid_belief = BeliefState(
            content="",  # Invalid empty content
            belief_type=BeliefType.FACTUAL,
            confidence=0.8
        )
        
        epistemic_state = EpistemicState(
            agent_id="test_agent",
            timestamp=datetime.utcnow(),
            belief_states=[invalid_belief]
        )
        
        assert epistemic_state.validate() is False
    
    def test_epistemic_state_defaults(self):
        """Test EpistemicState with default values."""
        epistemic_state = EpistemicState(
            agent_id="default_test",
            timestamp=datetime.utcnow()
        )
        
        assert epistemic_state.belief_states == []
        assert isinstance(epistemic_state.knowledge_state, KnowledgeState)
        assert epistemic_state.goal_states == []
        assert epistemic_state.confidence_level == 0.0
        assert epistemic_state.uncertainty_score == 0.0
        assert epistemic_state.decision_context == {}
        assert epistemic_state.validate() is True


if __name__ == "__main__":
    pytest.main([__file__])