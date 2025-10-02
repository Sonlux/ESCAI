"""
Epistemic State Models for ESCAI Framework.

This module defines data models for tracking agent beliefs, knowledge, and goals.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class BeliefType(Enum):
    """Types of beliefs an agent can hold."""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONDITIONAL = "conditional"
    PREDICTIVE = "predictive"


class GoalStatus(Enum):
    """Status of agent goals."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


@dataclass

class BeliefState:
    """Represents a single belief held by an agent."""
    content: str
    belief_type: BeliefType = BeliefType.FACTUAL
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


    def validate(self) -> bool:
        """Validate the belief state."""
        return (
            bool(self.content) and
            0.0 <= self.confidence <= 1.0 and
            isinstance(self.evidence, list)
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content': self.content,
            'belief_type': self.belief_type.value,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'source': self.source,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass

class KnowledgeState:
    """Represents the knowledge state of an agent."""
    facts: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""
    confidence_score: float = 1.0  # Legacy compatibility


    def validate(self) -> bool:
        """Validate the knowledge state."""
        return (
            isinstance(self.facts, list) and
            0.0 <= self.confidence <= 1.0
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'facts': self.facts,
            'confidence': self.confidence,
            'source': self.source,
            'confidence_score': self.confidence_score
        }


@dataclass

class GoalState:
    """Represents the goal state of an agent."""
    description: str = ""
    status: GoalStatus = GoalStatus.ACTIVE
    priority: int = 5
    primary_goals: List[str] = field(default_factory=list)
    secondary_goals: List[str] = field(default_factory=list)
    completion_status: Dict[str, Any] = field(default_factory=dict)


    def validate(self) -> bool:
        """Validate the goal state."""
        return (
            isinstance(self.primary_goals, list) and
            isinstance(self.secondary_goals, list) and
            isinstance(self.completion_status, dict) and
            1 <= self.priority <= 10
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority,
            'primary_goals': self.primary_goals,
            'secondary_goals': self.secondary_goals,
            'completion_status': self.completion_status
        }


@dataclass

class EpistemicState:
    """Complete epistemic state of an agent."""
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    belief_states: List[BeliefState] = field(default_factory=list)
    knowledge_state: Optional[KnowledgeState] = None
    goal_states: List[GoalState] = field(default_factory=list)
    confidence_level: float = 1.0
    uncertainty_score: float = 0.0


    def validate(self) -> bool:
        """Validate the epistemic state."""
        return (
            bool(self.agent_id) and
            isinstance(self.belief_states, list) and
            isinstance(self.goal_states, list) and
            0.0 <= self.confidence_level <= 1.0 and
            0.0 <= self.uncertainty_score <= 1.0 and
            all(belief.validate() for belief in self.belief_states) and
            all(goal.validate() for goal in self.goal_states)
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'agent_id': self.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'belief_states': [belief.to_dict() for belief in self.belief_states],
            'knowledge_state': self.knowledge_state.to_dict() if self.knowledge_state else None,
            'goal_states': [goal.to_dict() for goal in self.goal_states],
            'confidence_level': self.confidence_level,
            'uncertainty_score': self.uncertainty_score
        }


    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod

    def from_json(cls, json_str: str) -> 'EpistemicState':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod

    def from_dict(cls, data: Dict[str, Any]) -> 'EpistemicState':
        """Create from dictionary."""
        belief_states = [
            BeliefState(
                content=b['content'],
                belief_type=BeliefType(b['belief_type']),
                confidence=b['confidence'],
                evidence=b['evidence'],
                source=b['source'],
                timestamp=datetime.fromisoformat(b['timestamp'])
            )
            for b in data.get('belief_states', [])
        ]

        goal_states = [
            GoalState(
                description=g['description'],
                status=GoalStatus(g['status']),
                priority=g['priority'],
                primary_goals=g['primary_goals'],
                secondary_goals=g['secondary_goals'],
                completion_status=g['completion_status']
            )
            for g in data.get('goal_states', [])
        ]

        knowledge_state = None
        if data.get('knowledge_state'):
            knowledge_state = KnowledgeState(**data['knowledge_state'])

        return cls(
            agent_id=data['agent_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            belief_states=belief_states,
            knowledge_state=knowledge_state,
            goal_states=goal_states,
            confidence_level=data['confidence_level'],
            uncertainty_score=data['uncertainty_score']
        )
