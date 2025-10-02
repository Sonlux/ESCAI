"""
Behavioral Pattern Models for ESCAI Framework.

This module defines data models for tracking agent behavioral patterns and execution sequences.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class PatternType(Enum):
    """Types of behavioral patterns."""
    SEQUENTIAL = "sequential"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    FEEDBACK_LOOP = "feedback_loop"


class ExecutionStatus(Enum):
    """Status of execution steps/sequences."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    FAILURE = "failure"  # Alias for FAILED
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass

class ExecutionStep:
    """Represents a single execution step."""
    step_id: str
    step_type: str = ""
    action: str = ""
    duration: float = 0.0
    success_probability: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: ExecutionStatus = ExecutionStatus.PENDING


    def validate(self) -> bool:
        """Validate the execution step."""
        return (
            bool(self.step_id) and
            self.duration >= 0.0 and
            0.0 <= self.success_probability <= 1.0
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step_id': self.step_id,
            'step_type': self.step_type,
            'action': self.action,
            'duration': self.duration,
            'success_probability': self.success_probability,
            'context': self.context,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass

class ExecutionSequence:
    """Represents a sequence of execution steps."""
    sequence_id: str
    agent_id: str = ""
    steps: List[ExecutionStep] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    success: bool = True
    error_message: Optional[str] = None
    actions: List[str] = field(default_factory=list)  # Legacy compatibility
    success_rate: float = 1.0
    total_duration_ms: int = 0


    def __post_init__(self):
        """Initialize derived fields."""
        if not self.actions and self.steps:
            self.actions = [step.action for step in self.steps]


    def validate(self) -> bool:
        """Validate the execution sequence."""
        return (
            bool(self.sequence_id) and
            isinstance(self.steps, list) and
            all(step.validate() for step in self.steps)
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sequence_id': self.sequence_id,
            'agent_id': self.agent_id,
            'steps': [step.to_dict() for step in self.steps],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'success': self.success,
            'error_message': self.error_message,
            'actions': self.actions
        }


@dataclass

class BehavioralPattern:
    """Represents a behavioral pattern identified in agent execution."""
    pattern_id: str
    pattern_name: str = ""
    pattern_type: PatternType = PatternType.SEQUENTIAL
    description: str = ""
    execution_sequences: List[ExecutionSequence] = field(default_factory=list)
    frequency: float = 0.0
    success_rate: float = 0.0
    average_duration: float = 0.0
    common_triggers: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    statistical_significance: float = 0.0


    def validate(self) -> bool:
        """Validate the behavioral pattern."""
        return (
            bool(self.pattern_id) and
            isinstance(self.execution_sequences, list) and
            0.0 <= self.frequency <= 1.0 and
            0.0 <= self.success_rate <= 1.0 and
            self.average_duration >= 0.0 and
            all(seq.validate() for seq in self.execution_sequences)
        )


    def calculate_statistics(self) -> None:
        """Calculate pattern statistics from execution sequences."""
        if not self.execution_sequences:
            return

        self.frequency = len(self.execution_sequences)
        successful_sequences = sum(1 for seq in self.execution_sequences if seq.success)
        self.success_rate = successful_sequences / len(self.execution_sequences) if self.execution_sequences else 0.0

        durations = []
        for seq in self.execution_sequences:
            if seq.end_time and seq.start_time:
                duration = (seq.end_time - seq.start_time).total_seconds()
                durations.append(duration)

        self.average_duration = sum(durations) / len(durations) if durations else 0.0


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'pattern_type': self.pattern_type.value,
            'description': self.description,
            'execution_sequences': [seq.to_dict() for seq in self.execution_sequences],
            'frequency': self.frequency,
            'success_rate': self.success_rate,
            'average_duration': self.average_duration,
            'common_triggers': self.common_triggers,
            'failure_modes': self.failure_modes,
            'statistical_significance': self.statistical_significance
        }


    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod

    def from_json(cls, json_str: str) -> 'BehavioralPattern':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod

    def from_dict(cls, data: Dict[str, Any]) -> 'BehavioralPattern':
        """Create from dictionary."""
        execution_sequences = [
            ExecutionSequence(
                sequence_id=seq['sequence_id'],
                agent_id=seq['agent_id'],
                steps=[
                    ExecutionStep(
                        step_id=step['step_id'],
                        step_type=step['step_type'],
                        action=step['action'],
                        duration=step['duration'],
                        success_probability=step['success_probability'],
                        context=step['context'],
                        error_message=step['error_message'],
                        timestamp=datetime.fromisoformat(step['timestamp'])
                    )
                    for step in seq['steps']
                ],
                start_time=datetime.fromisoformat(seq['start_time']),
                end_time=datetime.fromisoformat(seq['end_time']) if seq['end_time'] else None,
                success=seq['success'],
                error_message=seq['error_message']
            )
            for seq in data.get('execution_sequences', [])
        ]

        return cls(
            pattern_id=data['pattern_id'],
            pattern_name=data['pattern_name'],
            pattern_type=PatternType(data['pattern_type']),
            description=data['description'],
            execution_sequences=execution_sequences,
            frequency=data['frequency'],
            success_rate=data['success_rate'],
            average_duration=data['average_duration'],
            common_triggers=data['common_triggers'],
            failure_modes=data['failure_modes'],
            statistical_significance=data['statistical_significance']
        )
