"""
Unit tests for CausalRelationship data model.
"""

import pytest
from datetime import datetime, timedelta

from escai_framework.models.causal_relationship import (
    CausalRelationship, CausalEvent, CausalEvidence,
    CausalType, EvidenceType
)


class TestCausalEvent:
    """Test cases for CausalEvent model."""
    
    def test_causal_event_creation(self):
        """Test creating a valid CausalEvent."""
        timestamp = datetime.utcnow()
        event = CausalEvent(
            event_id="event_001",
            event_type="decision",
            description="Agent made a decision",
            timestamp=timestamp,
            agent_id="agent_001",
            context={"task": "classification"},
            attributes={"confidence": 0.8}
        )
        
        assert event.event_id == "event_001"
        assert event.event_type == "decision"
        assert event.description == "Agent made a decision"
        assert event.timestamp == timestamp
        assert event.agent_id == "agent_001"
        assert event.context == {"task": "classification"}
        assert event.attributes == {"confidence": 0.8}
    
    def test_causal_event_validation_valid(self):
        """Test validation of valid CausalEvent."""
        event = CausalEvent(
            event_id="valid_event",
            event_type="action",
            description="Valid event description",
            timestamp=datetime.utcnow(),
            agent_id="valid_agent"
        )
        assert event.validate() is True
    
    def test_causal_event_validation_invalid(self):
        """Test validation with invalid data."""
        event = CausalEvent(
            event_id="",  # Invalid empty ID
            event_type="action",
            description="Description",
            timestamp=datetime.utcnow(),
            agent_id="agent"
        )
        assert event.validate() is False
        
        event.event_id = "valid_id"
        event.agent_id = ""  # Invalid empty agent ID
        assert event.validate() is False
    
    def test_causal_event_serialization(self):
        """Test CausalEvent serialization."""
        timestamp = datetime.utcnow()
        original = CausalEvent(
            event_id="serialize_event",
            event_type="test_type",
            description="Serialization test",
            timestamp=timestamp,
            agent_id="test_agent",
            context={"test": "context"},
            attributes={"test": "attribute"}
        )
        
        data = original.to_dict()
        restored = CausalEvent.from_dict(data)
        
        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type
        assert restored.description == original.description
        assert restored.timestamp == original.timestamp
        assert restored.agent_id == original.agent_id
        assert restored.context == original.context
        assert restored.attributes == original.attributes


class TestCausalEvidence:
    """Test cases for CausalEvidence model."""
    
    def test_causal_evidence_creation(self):
        """Test creating a valid CausalEvidence."""
        evidence = CausalEvidence(
            evidence_type=EvidenceType.STATISTICAL,
            description="Statistical correlation found",
            strength=0.8,
            confidence=0.9,
            source="correlation_analysis",
            data_points=[{"x": 1, "y": 2}],
            statistical_measures={"p_value": 0.01, "r_squared": 0.75}
        )
        
        assert evidence.evidence_type == EvidenceType.STATISTICAL
        assert evidence.description == "Statistical correlation found"
        assert evidence.strength == 0.8
        assert evidence.confidence == 0.9
        assert evidence.source == "correlation_analysis"
        assert evidence.data_points == [{"x": 1, "y": 2}]
        assert evidence.statistical_measures == {"p_value": 0.01, "r_squared": 0.75}
    
    def test_causal_evidence_validation_valid(self):
        """Test validation of valid CausalEvidence."""
        evidence = CausalEvidence(
            evidence_type=EvidenceType.EXPERIMENTAL,
            description="Valid evidence",
            strength=0.7,
            confidence=0.8,
            source="experiment"
        )
        assert evidence.validate() is True
    
    def test_causal_evidence_validation_invalid(self):
        """Test validation with invalid data."""
        evidence = CausalEvidence(
            evidence_type=EvidenceType.TEMPORAL,
            description="",  # Invalid empty description
            strength=0.5,
            confidence=0.6,
            source="test"
        )
        assert evidence.validate() is False
        
        evidence.description = "Valid description"
        evidence.strength = 1.5  # Invalid strength > 1.0
        assert evidence.validate() is False
    
    def test_causal_evidence_serialization(self):
        """Test CausalEvidence serialization."""
        original = CausalEvidence(
            evidence_type=EvidenceType.OBSERVATIONAL,
            description="Observational evidence",
            strength=0.6,
            confidence=0.7,
            source="observation_study",
            data_points=[{"observation": "test"}],
            statistical_measures={"correlation": 0.5}
        )
        
        data = original.to_dict()
        restored = CausalEvidence.from_dict(data)
        
        assert restored.evidence_type == original.evidence_type
        assert restored.description == original.description
        assert restored.strength == original.strength
        assert restored.confidence == original.confidence
        assert restored.source == original.source
        assert restored.data_points == original.data_points
        assert restored.statistical_measures == original.statistical_measures


class TestCausalRelationship:
    """Test cases for CausalRelationship model."""
    
    def test_causal_relationship_creation(self):
        """Test creating a valid CausalRelationship."""
        cause_time = datetime.utcnow()
        effect_time = cause_time + timedelta(seconds=2)
        
        cause_event = CausalEvent(
            event_id="cause_001",
            event_type="action",
            description="Agent performed action",
            timestamp=cause_time,
            agent_id="agent_001"
        )
        
        effect_event = CausalEvent(
            event_id="effect_001",
            event_type="outcome",
            description="Outcome occurred",
            timestamp=effect_time,
            agent_id="agent_001"
        )
        
        evidence = CausalEvidence(
            evidence_type=EvidenceType.TEMPORAL,
            description="Temporal precedence",
            strength=0.9,
            confidence=0.8,
            source="temporal_analysis"
        )
        
        relationship = CausalRelationship(
            relationship_id="rel_001",
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.DIRECT,
            strength=0.8,
            confidence=0.7,
            delay_ms=2000,
            evidence=[evidence],
            statistical_significance=0.05,
            effect_size=0.6,
            causal_mechanism="Direct causation through action execution"
        )
        
        assert relationship.relationship_id == "rel_001"
        assert relationship.cause_event == cause_event
        assert relationship.effect_event == effect_event
        assert relationship.causal_type == CausalType.DIRECT
        assert relationship.strength == 0.8
        assert relationship.confidence == 0.7
        assert relationship.delay_ms == 2000
        assert len(relationship.evidence) == 1
        assert relationship.statistical_significance == 0.05
        assert relationship.effect_size == 0.6
        assert relationship.causal_mechanism == "Direct causation through action execution"
    
    def test_causal_relationship_validation_valid(self):
        """Test validation of valid CausalRelationship."""
        cause_event = CausalEvent(
            event_id="cause",
            event_type="action",
            description="Cause event",
            timestamp=datetime.utcnow(),
            agent_id="agent"
        )
        
        effect_event = CausalEvent(
            event_id="effect",
            event_type="outcome",
            description="Effect event",
            timestamp=datetime.utcnow() + timedelta(seconds=1),
            agent_id="agent"
        )
        
        relationship = CausalRelationship(
            relationship_id="valid_rel",
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.INDIRECT,
            strength=0.5,
            confidence=0.6,
            delay_ms=1000
        )
        
        assert relationship.validate() is True
    
    def test_causal_relationship_validation_invalid(self):
        """Test validation with invalid data."""
        cause_event = CausalEvent(
            event_id="cause",
            event_type="action",
            description="Cause event",
            timestamp=datetime.utcnow(),
            agent_id="agent"
        )
        
        effect_event = CausalEvent(
            event_id="effect",
            event_type="outcome",
            description="Effect event",
            timestamp=datetime.utcnow(),
            agent_id="agent"
        )
        
        relationship = CausalRelationship(
            relationship_id="",  # Invalid empty ID
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.DIRECT,
            strength=0.5,
            confidence=0.6,
            delay_ms=1000
        )
        
        assert relationship.validate() is False
        
        relationship.relationship_id = "valid_id"
        relationship.strength = 1.5  # Invalid strength > 1.0
        assert relationship.validate() is False
    
    def test_causal_relationship_temporal_order(self):
        """Test temporal order checking."""
        cause_time = datetime.utcnow()
        effect_time = cause_time + timedelta(seconds=1)
        
        cause_event = CausalEvent(
            event_id="cause",
            event_type="action",
            description="Cause",
            timestamp=cause_time,
            agent_id="agent"
        )
        
        effect_event = CausalEvent(
            event_id="effect",
            event_type="outcome",
            description="Effect",
            timestamp=effect_time,
            agent_id="agent"
        )
        
        relationship = CausalRelationship(
            relationship_id="temporal_test",
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.DIRECT,
            strength=0.8,
            confidence=0.7,
            delay_ms=0
        )
        
        assert relationship.get_temporal_order() is True
        
        # Test reverse order
        relationship.cause_event.timestamp = effect_time
        relationship.effect_event.timestamp = cause_time
        assert relationship.get_temporal_order() is False
    
    def test_causal_relationship_calculate_delay(self):
        """Test delay calculation."""
        cause_time = datetime.utcnow()
        effect_time = cause_time + timedelta(milliseconds=1500)
        
        cause_event = CausalEvent(
            event_id="cause",
            event_type="action",
            description="Cause",
            timestamp=cause_time,
            agent_id="agent"
        )
        
        effect_event = CausalEvent(
            event_id="effect",
            event_type="outcome",
            description="Effect",
            timestamp=effect_time,
            agent_id="agent"
        )
        
        relationship = CausalRelationship(
            relationship_id="delay_test",
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.DIRECT,
            strength=0.8,
            confidence=0.7,
            delay_ms=0
        )
        
        calculated_delay = relationship.calculate_delay()
        assert calculated_delay == 1500
        assert relationship.delay_ms == 1500
    
    def test_causal_relationship_add_evidence(self):
        """Test adding evidence to relationship."""
        cause_event = CausalEvent(
            event_id="cause",
            event_type="action",
            description="Cause",
            timestamp=datetime.utcnow(),
            agent_id="agent"
        )
        
        effect_event = CausalEvent(
            event_id="effect",
            event_type="outcome",
            description="Effect",
            timestamp=datetime.utcnow(),
            agent_id="agent"
        )
        
        relationship = CausalRelationship(
            relationship_id="evidence_test",
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.DIRECT,
            strength=0.5,
            confidence=0.3,
            delay_ms=1000
        )
        
        evidence = CausalEvidence(
            evidence_type=EvidenceType.EXPERIMENTAL,
            description="Experimental evidence",
            strength=0.9,
            confidence=0.95,
            source="controlled_experiment"
        )
        
        initial_evidence_count = len(relationship.evidence)
        initial_confidence = relationship.confidence
        
        relationship.add_evidence(evidence)
        
        assert len(relationship.evidence) == initial_evidence_count + 1
        assert relationship.confidence > initial_confidence  # Should increase with strong evidence
    
    def test_causal_relationship_serialization(self):
        """Test CausalRelationship serialization."""
        cause_event = CausalEvent(
            event_id="serialize_cause",
            event_type="action",
            description="Serialization cause",
            timestamp=datetime.utcnow(),
            agent_id="test_agent"
        )
        
        effect_event = CausalEvent(
            event_id="serialize_effect",
            event_type="outcome",
            description="Serialization effect",
            timestamp=datetime.utcnow(),
            agent_id="test_agent"
        )
        
        evidence = CausalEvidence(
            evidence_type=EvidenceType.STATISTICAL,
            description="Test evidence",
            strength=0.8,
            confidence=0.9,
            source="test_source"
        )
        
        original = CausalRelationship(
            relationship_id="serialize_rel",
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.MEDIATED,
            strength=0.7,
            confidence=0.8,
            delay_ms=1500,
            evidence=[evidence],
            statistical_significance=0.01,
            effect_size=0.5,
            causal_mechanism="Test mechanism",
            confounding_factors=["factor1", "factor2"],
            validated=True,
            validation_method="test_validation"
        )
        
        # Test to_dict
        data = original.to_dict()
        assert data["relationship_id"] == "serialize_rel"
        assert data["causal_type"] == "mediated"
        
        # Test from_dict
        restored = CausalRelationship.from_dict(data)
        assert restored.relationship_id == original.relationship_id
        assert restored.cause_event.event_id == original.cause_event.event_id
        assert restored.effect_event.event_id == original.effect_event.event_id
        assert restored.causal_type == original.causal_type
        assert restored.strength == original.strength
        assert restored.confidence == original.confidence
        assert restored.delay_ms == original.delay_ms
        assert len(restored.evidence) == 1
        assert restored.statistical_significance == original.statistical_significance
        assert restored.effect_size == original.effect_size
        assert restored.causal_mechanism == original.causal_mechanism
        assert restored.confounding_factors == original.confounding_factors
        assert restored.validated == original.validated
        assert restored.validation_method == original.validation_method
    
    def test_causal_relationship_json_serialization(self):
        """Test CausalRelationship JSON serialization."""
        cause_event = CausalEvent(
            event_id="json_cause",
            event_type="action",
            description="JSON cause",
            timestamp=datetime.utcnow(),
            agent_id="json_agent"
        )
        
        effect_event = CausalEvent(
            event_id="json_effect",
            event_type="outcome",
            description="JSON effect",
            timestamp=datetime.utcnow(),
            agent_id="json_agent"
        )
        
        relationship = CausalRelationship(
            relationship_id="json_rel",
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.SPURIOUS,
            strength=0.4,
            confidence=0.5,
            delay_ms=500
        )
        
        json_str = relationship.to_json()
        assert isinstance(json_str, str)
        
        restored = CausalRelationship.from_json(json_str)
        assert restored.relationship_id == relationship.relationship_id
        assert restored.cause_event.event_id == relationship.cause_event.event_id
        assert restored.effect_event.event_id == relationship.effect_event.event_id
        assert restored.causal_type == relationship.causal_type


if __name__ == "__main__":
    pytest.main([__file__])