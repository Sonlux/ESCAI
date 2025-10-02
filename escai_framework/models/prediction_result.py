"""
Prediction Result Models for ESCAI Framework.

This module defines data models for tracking prediction results and risk assessments.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class PredictionType(Enum):
    """Types of predictions."""
    SUCCESS_PROBABILITY = "success_probability"
    COMPLETION_TIME = "completion_time"
    RESOURCE_USAGE = "resource_usage"
    FAILURE_RISK = "failure_risk"
    PERFORMANCE_SCORE = "performance_score"


class RiskLevel(Enum):
    """Risk levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InterventionType(Enum):
    """Types of interventions."""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    RESOURCE_ALLOCATION = "resource_allocation"
    PROCESS_MODIFICATION = "process_modification"
    MONITORING_ENHANCEMENT = "monitoring_enhancement"


@dataclass

class RiskFactor:
    """Represents a risk factor for prediction outcomes."""
    factor_id: str
    name: str = ""
    description: str = ""
    impact_score: float = 0.0
    probability: float = 0.0
    category: str = ""
    mitigation_strategies: List[str] = field(default_factory=list)


    def validate(self) -> bool:
        """Validate the risk factor."""
        return (
            bool(self.factor_id) and
            bool(self.name) and
            0.0 <= self.impact_score <= 1.0 and
            0.0 <= self.probability <= 1.0 and
            isinstance(self.mitigation_strategies, list)
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'factor_id': self.factor_id,
            'name': self.name,
            'description': self.description,
            'impact_score': self.impact_score,
            'probability': self.probability,
            'category': self.category,
            'mitigation_strategies': self.mitigation_strategies
        }


@dataclass

class Intervention:
    """Represents a recommended intervention."""
    intervention_id: str
    intervention_type: InterventionType
    name: str = ""
    description: str = ""
    expected_impact: float = 0.0
    implementation_cost: float = 0.0
    urgency: RiskLevel = RiskLevel.MEDIUM


    def validate(self) -> bool:
        """Validate the intervention."""
        return (
            bool(self.intervention_id) and
            bool(self.name) and
            isinstance(self.intervention_type, InterventionType) and
            0.0 <= self.expected_impact <= 1.0 and
            self.implementation_cost >= 0.0 and
            isinstance(self.urgency, RiskLevel)
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'intervention_id': self.intervention_id,
            'intervention_type': self.intervention_type.value,
            'name': self.name,
            'description': self.description,
            'expected_impact': self.expected_impact,
            'implementation_cost': self.implementation_cost,
            'urgency': self.urgency.value
        }


@dataclass

class ConfidenceInterval:
    """Represents a confidence interval for predictions."""
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95


    def validate(self) -> bool:
        """Validate the confidence interval."""
        return (
            self.lower_bound <= self.upper_bound and
            0.0 <= self.confidence_level <= 1.0
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'confidence_level': self.confidence_level
        }


@dataclass

class PredictionResult:
    """Represents a prediction result with associated metadata."""
    agent_id: str
    prediction_type: str
    predicted_value: float = 0.0
    confidence: float = 0.0
    uncertainty: float = 0.0
    risk_factors: List[str] = field(default_factory=list)  # Legacy compatibility
    recommended_actions: List[str] = field(default_factory=list)  # Legacy compatibility
    model_used: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Extended attributes
    prediction_type_enum: PredictionType = PredictionType.SUCCESS_PROBABILITY
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_factor_objects: List[RiskFactor] = field(default_factory=list)
    interventions: List[Intervention] = field(default_factory=list)
    confidence_interval: Optional[ConfidenceInterval] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)


    def validate(self) -> bool:
        """Validate the prediction result."""
        return (
            bool(self.agent_id) and
            bool(self.prediction_type) and
            0.0 <= self.confidence <= 1.0 and
            0.0 <= self.uncertainty <= 1.0 and
            isinstance(self.risk_factors, list) and
            isinstance(self.recommended_actions, list) and
            all(rf.validate() for rf in self.risk_factor_objects) and
            all(intervention.validate() for intervention in self.interventions) and
            (self.confidence_interval is None or self.confidence_interval.validate())
        )


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'agent_id': self.agent_id,
            'prediction_type': self.prediction_type,
            'predicted_value': self.predicted_value,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'risk_factors': self.risk_factors,
            'recommended_actions': self.recommended_actions,
            'model_used': self.model_used,
            'timestamp': self.timestamp.isoformat(),
            'prediction_type_enum': self.prediction_type_enum.value,
            'risk_level': self.risk_level.value,
            'risk_factor_objects': [rf.to_dict() for rf in self.risk_factor_objects],
            'interventions': [intervention.to_dict() for intervention in self.interventions],
            'confidence_interval': self.confidence_interval.to_dict() if self.confidence_interval else None,
            'feature_importance': self.feature_importance
        }


    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod

    def from_json(cls, json_str: str) -> 'PredictionResult':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod

    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """Create from dictionary."""
        risk_factor_objects = [
            RiskFactor(
                factor_id=rf['factor_id'],
                name=rf['name'],
                description=rf['description'],
                impact_score=rf['impact_score'],
                probability=rf['probability'],
                category=rf['category'],
                mitigation_strategies=rf['mitigation_strategies']
            )
            for rf in data.get('risk_factor_objects', [])
        ]

        interventions = [
            Intervention(
                intervention_id=intervention['intervention_id'],
                intervention_type=InterventionType(intervention['intervention_type']),
                name=intervention['name'],
                description=intervention['description'],
                expected_impact=intervention['expected_impact'],
                implementation_cost=intervention['implementation_cost'],
                urgency=RiskLevel(intervention['urgency'])
            )
            for intervention in data.get('interventions', [])
        ]

        confidence_interval = None
        if data.get('confidence_interval'):
            ci_data = data['confidence_interval']
            confidence_interval = ConfidenceInterval(
                lower_bound=ci_data['lower_bound'],
                upper_bound=ci_data['upper_bound'],
                confidence_level=ci_data['confidence_level']
            )

        return cls(
            agent_id=data['agent_id'],
            prediction_type=data['prediction_type'],
            predicted_value=data['predicted_value'],
            confidence=data['confidence'],
            uncertainty=data['uncertainty'],
            risk_factors=data['risk_factors'],
            recommended_actions=data['recommended_actions'],
            model_used=data['model_used'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            prediction_type_enum=PredictionType(data.get('prediction_type_enum', 'success_probability')),
            risk_level=RiskLevel(data.get('risk_level', 'medium')),
            risk_factor_objects=risk_factor_objects,
            interventions=interventions,
            confidence_interval=confidence_interval,
            feature_importance=data.get('feature_importance', {})
        )
