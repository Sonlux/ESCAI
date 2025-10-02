"""
Causal Relationship Models for ESCAI Framework.

This module defines data models for tracking causal relationships between events.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class CausalType(Enum):
    """Types of causal relationships."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    SPURIOUS = "spurious"
    BIDIRECTIONAL = "bidirectional"


class EvidenceType(Enum):
    """Types of evidence for causal relationships."""
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    EXPERIMENTAL = "experimental"
    OBSERVATIONAL = "observational"


@dataclass

class CausalEvent:
    """Represents a causal event."""
    event_id: str
    event_type: str = ""
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)


    def validate(self) -> bool:
        """Validate the causal event."""
        return (
            bool(self.event_id) and
            bool(self.event_type) and
            isinstance(self.context, dict) and
            isinstance(self.attributes, dict)
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'context': self.context,
            'attributes': self.attributes
        }


@dataclass

class CausalEvidence:
    """Represents evidence for a causal relationship."""
    evidence_type: EvidenceType
    description: str = ""
    strength: float = 0.0
    confidence: float = 0.0
    source: str = ""
    statistical_measures: Dict[str, Any] = field(default_factory=dict)


    def validate(self) -> bool:
        """Validate the causal evidence."""
        return (
            isinstance(self.evidence_type, EvidenceType) and
            0.0 <= self.strength <= 1.0 and
            0.0 <= self.confidence <= 1.0 and
            isinstance(self.statistical_measures, dict)
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'evidence_type': self.evidence_type.value,
            'description': self.description,
            'strength': self.strength,
            'confidence': self.confidence,
            'source': self.source,
            'statistical_measures': self.statistical_measures
        }


@dataclass

class CausalRelationship:
    """Represents a causal relationship between events."""
    relationship_id: str
    cause_event: CausalEvent
    effect_event: CausalEvent
    causal_type: CausalType = CausalType.DIRECT
    strength: float = 0.0
    confidence: float = 0.0
    delay_ms: int = 0
    evidence: List[CausalEvidence] = field(default_factory=list)
    statistical_significance: float = 0.0
    causal_mechanism: str = ""


    def validate(self) -> bool:
        """Validate the causal relationship."""
        return (
            bool(self.relationship_id) and
            self.cause_event.validate() and
            self.effect_event.validate() and
            0.0 <= self.strength <= 1.0 and
            0.0 <= self.confidence <= 1.0 and
            self.delay_ms >= 0 and
            all(ev.validate() for ev in self.evidence)
        )


    def get_temporal_order(self) -> bool:
        """Check if the temporal order is correct (cause before effect)."""
        return self.cause_event.timestamp <= self.effect_event.timestamp


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'relationship_id': self.relationship_id,
            'cause_event': self.cause_event.to_dict(),
            'effect_event': self.effect_event.to_dict(),
            'causal_type': self.causal_type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'delay_ms': self.delay_ms,
            'evidence': [ev.to_dict() for ev in self.evidence],
            'statistical_significance': self.statistical_significance,
            'causal_mechanism': self.causal_mechanism
        }


    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod

    def from_json(cls, json_str: str) -> 'CausalRelationship':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod

    def from_dict(cls, data: Dict[str, Any]) -> 'CausalRelationship':
        """Create from dictionary."""
        cause_event = CausalEvent(
            event_id=data['cause_event']['event_id'],
            event_type=data['cause_event']['event_type'],
            description=data['cause_event']['description'],
            timestamp=datetime.fromisoformat(data['cause_event']['timestamp']),
            agent_id=data['cause_event']['agent_id'],
            context=data['cause_event']['context'],
            attributes=data['cause_event']['attributes']
        )

        effect_event = CausalEvent(
            event_id=data['effect_event']['event_id'],
            event_type=data['effect_event']['event_type'],
            description=data['effect_event']['description'],
            timestamp=datetime.fromisoformat(data['effect_event']['timestamp']),
            agent_id=data['effect_event']['agent_id'],
            context=data['effect_event']['context'],
            attributes=data['effect_event']['attributes']
        )

        evidence = [
            CausalEvidence(
                evidence_type=EvidenceType(ev['evidence_type']),
                description=ev['description'],
                strength=ev['strength'],
                confidence=ev['confidence'],
                source=ev['source'],
                statistical_measures=ev['statistical_measures']
            )
            for ev in data.get('evidence', [])
        ]

        return cls(
            relationship_id=data['relationship_id'],
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType(data['causal_type']),
            strength=data['strength'],
            confidence=data['confidence'],
            delay_ms=data['delay_ms'],
            evidence=evidence,
            statistical_significance=data['statistical_significance'],
            causal_mechanism=data['causal_mechanism']
        )
