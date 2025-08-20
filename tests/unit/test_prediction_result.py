"""
Unit tests for PredictionResult data model.
"""

import pytest
from datetime import datetime, timedelta

from escai_framework.models.prediction_result import (
    PredictionResult, RiskFactor, Intervention, ConfidenceInterval,
    PredictionType, RiskLevel, InterventionType
)


class TestRiskFactor:
    """Test cases for RiskFactor model."""
    
    def test_risk_factor_creation(self):
        """Test creating a valid RiskFactor."""
        risk_factor = RiskFactor(
            factor_id="risk_001",
            name="High Complexity",
            description="Task complexity is higher than usual",
            impact_score=0.8,
            probability=0.6,
            category="task_complexity",
            mitigation_strategies=["simplify task", "add resources"]
        )
        
        assert risk_factor.factor_id == "risk_001"
        assert risk_factor.name == "High Complexity"
        assert risk_factor.description == "Task complexity is higher than usual"
        assert risk_factor.impact_score == 0.8
        assert risk_factor.probability == 0.6
        assert risk_factor.category == "task_complexity"
        assert risk_factor.mitigation_strategies == ["simplify task", "add resources"]
    
    def test_risk_factor_validation_valid(self):
        """Test validation of valid RiskFactor."""
        risk_factor = RiskFactor(
            factor_id="valid_risk",
            name="Valid Risk",
            description="Valid risk description",
            impact_score=0.7,
            probability=0.5,
            category="test_category"
        )
        assert risk_factor.validate() is True
    
    def test_risk_factor_validation_invalid(self):
        """Test validation with invalid data."""
        risk_factor = RiskFactor(
            factor_id="",  # Invalid empty ID
            name="Risk Name",
            description="Description",
            impact_score=0.5,
            probability=0.5,
            category="category"
        )
        assert risk_factor.validate() is False
        
        risk_factor.factor_id = "valid_id"
        risk_factor.impact_score = 1.5  # Invalid score > 1.0
        assert risk_factor.validate() is False
    
    def test_risk_factor_calculate_risk_score(self):
        """Test risk score calculation."""
        risk_factor = RiskFactor(
            factor_id="score_test",
            name="Score Test",
            description="Test risk score calculation",
            impact_score=0.8,
            probability=0.6,
            category="test"
        )
        
        expected_score = 0.8 * 0.6  # impact * probability
        assert risk_factor.calculate_risk_score() == expected_score
    
    def test_risk_factor_serialization(self):
        """Test RiskFactor serialization."""
        original = RiskFactor(
            factor_id="serialize_risk",
            name="Serialization Risk",
            description="Risk for serialization testing",
            impact_score=0.9,
            probability=0.7,
            category="serialization",
            mitigation_strategies=["strategy1", "strategy2"]
        )
        
        data = original.to_dict()
        restored = RiskFactor.from_dict(data)
        
        assert restored.factor_id == original.factor_id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.impact_score == original.impact_score
        assert restored.probability == original.probability
        assert restored.category == original.category
        assert restored.mitigation_strategies == original.mitigation_strategies


class TestIntervention:
    """Test cases for Intervention model."""
    
    def test_intervention_creation(self):
        """Test creating a valid Intervention."""
        intervention = Intervention(
            intervention_id="int_001",
            intervention_type=InterventionType.PARAMETER_ADJUSTMENT,
            name="Adjust Learning Rate",
            description="Reduce learning rate to improve stability",
            expected_impact=0.7,
            implementation_cost=0.3,
            urgency=RiskLevel.MEDIUM,
            parameters={"learning_rate": 0.001},
            prerequisites=["model_checkpoint", "validation_data"]
        )
        
        assert intervention.intervention_id == "int_001"
        assert intervention.intervention_type == InterventionType.PARAMETER_ADJUSTMENT
        assert intervention.name == "Adjust Learning Rate"
        assert intervention.description == "Reduce learning rate to improve stability"
        assert intervention.expected_impact == 0.7
        assert intervention.implementation_cost == 0.3
        assert intervention.urgency == RiskLevel.MEDIUM
        assert intervention.parameters == {"learning_rate": 0.001}
        assert intervention.prerequisites == ["model_checkpoint", "validation_data"]
    
    def test_intervention_validation_valid(self):
        """Test validation of valid Intervention."""
        intervention = Intervention(
            intervention_id="valid_int",
            intervention_type=InterventionType.RESOURCE_ALLOCATION,
            name="Valid Intervention",
            description="Valid intervention description",
            expected_impact=0.6,
            implementation_cost=0.4,
            urgency=RiskLevel.LOW
        )
        assert intervention.validate() is True
    
    def test_intervention_validation_invalid(self):
        """Test validation with invalid data."""
        intervention = Intervention(
            intervention_id="",  # Invalid empty ID
            intervention_type=InterventionType.STRATEGY_CHANGE,
            name="Intervention",
            description="Description",
            expected_impact=0.5,
            implementation_cost=0.3,
            urgency=RiskLevel.HIGH
        )
        assert intervention.validate() is False
        
        intervention.intervention_id = "valid_id"
        intervention.expected_impact = 1.5  # Invalid impact > 1.0
        assert intervention.validate() is False
    
    def test_intervention_benefit_cost_ratio(self):
        """Test benefit-cost ratio calculation."""
        intervention = Intervention(
            intervention_id="ratio_test",
            intervention_type=InterventionType.HUMAN_INTERVENTION,
            name="Ratio Test",
            description="Test benefit-cost ratio",
            expected_impact=0.8,
            implementation_cost=0.4,
            urgency=RiskLevel.HIGH
        )
        
        expected_ratio = 0.8 / 0.4  # impact / cost
        assert intervention.calculate_benefit_cost_ratio() == expected_ratio
        
        # Test zero cost case
        intervention.implementation_cost = 0.0
        assert intervention.calculate_benefit_cost_ratio() == float('inf')
    
    def test_intervention_serialization(self):
        """Test Intervention serialization."""
        original = Intervention(
            intervention_id="serialize_int",
            intervention_type=InterventionType.EARLY_TERMINATION,
            name="Serialization Intervention",
            description="Intervention for serialization testing",
            expected_impact=0.9,
            implementation_cost=0.2,
            urgency=RiskLevel.CRITICAL,
            parameters={"threshold": 0.1},
            prerequisites=["monitoring_active"]
        )
        
        data = original.to_dict()
        restored = Intervention.from_dict(data)
        
        assert restored.intervention_id == original.intervention_id
        assert restored.intervention_type == original.intervention_type
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.expected_impact == original.expected_impact
        assert restored.implementation_cost == original.implementation_cost
        assert restored.urgency == original.urgency
        assert restored.parameters == original.parameters
        assert restored.prerequisites == original.prerequisites


class TestConfidenceInterval:
    """Test cases for ConfidenceInterval model."""
    
    def test_confidence_interval_creation(self):
        """Test creating a valid ConfidenceInterval."""
        interval = ConfidenceInterval(
            lower_bound=0.6,
            upper_bound=0.9,
            confidence_level=0.95
        )
        
        assert interval.lower_bound == 0.6
        assert interval.upper_bound == 0.9
        assert interval.confidence_level == 0.95
    
    def test_confidence_interval_validation_valid(self):
        """Test validation of valid ConfidenceInterval."""
        interval = ConfidenceInterval(
            lower_bound=0.3,
            upper_bound=0.7,
            confidence_level=0.9
        )
        assert interval.validate() is True
    
    def test_confidence_interval_validation_invalid(self):
        """Test validation with invalid data."""
        interval = ConfidenceInterval(
            lower_bound=0.8,
            upper_bound=0.6,  # Invalid: upper < lower
            confidence_level=0.95
        )
        assert interval.validate() is False
        
        interval.lower_bound = 0.5
        interval.upper_bound = 0.7
        interval.confidence_level = 1.5  # Invalid confidence level > 1.0
        assert interval.validate() is False
    
    def test_confidence_interval_width(self):
        """Test width calculation."""
        interval = ConfidenceInterval(
            lower_bound=0.4,
            upper_bound=0.8,
            confidence_level=0.95
        )
        
        expected_width = 0.8 - 0.4
        assert interval.width() == expected_width
    
    def test_confidence_interval_contains(self):
        """Test contains method."""
        interval = ConfidenceInterval(
            lower_bound=0.3,
            upper_bound=0.7,
            confidence_level=0.9
        )
        
        assert interval.contains(0.5) is True
        assert interval.contains(0.3) is True  # Boundary case
        assert interval.contains(0.7) is True  # Boundary case
        assert interval.contains(0.2) is False
        assert interval.contains(0.8) is False
    
    def test_confidence_interval_serialization(self):
        """Test ConfidenceInterval serialization."""
        original = ConfidenceInterval(
            lower_bound=0.2,
            upper_bound=0.8,
            confidence_level=0.99
        )
        
        data = original.to_dict()
        restored = ConfidenceInterval.from_dict(data)
        
        assert restored.lower_bound == original.lower_bound
        assert restored.upper_bound == original.upper_bound
        assert restored.confidence_level == original.confidence_level


class TestPredictionResult:
    """Test cases for PredictionResult model."""
    
    def test_prediction_result_creation(self):
        """Test creating a valid PredictionResult."""
        confidence_interval = ConfidenceInterval(
            lower_bound=0.6,
            upper_bound=0.9,
            confidence_level=0.95
        )
        
        risk_factor = RiskFactor(
            factor_id="risk_001",
            name="Test Risk",
            description="Test risk factor",
            impact_score=0.7,
            probability=0.5,
            category="test"
        )
        
        intervention = Intervention(
            intervention_id="int_001",
            intervention_type=InterventionType.PARAMETER_ADJUSTMENT,
            name="Test Intervention",
            description="Test intervention",
            expected_impact=0.6,
            implementation_cost=0.3,
            urgency=RiskLevel.MEDIUM
        )
        
        created_at = datetime.utcnow()
        prediction = PredictionResult(
            prediction_id="pred_001",
            agent_id="agent_001",
            prediction_type=PredictionType.SUCCESS_PROBABILITY,
            predicted_value=0.75,
            confidence_score=0.85,
            confidence_interval=confidence_interval,
            risk_level=RiskLevel.MEDIUM,
            risk_factors=[risk_factor],
            recommended_interventions=[intervention],
            model_name="test_model",
            model_version="1.0",
            features_used=["feature1", "feature2"],
            prediction_horizon_ms=60000,
            created_at=created_at
        )
        
        assert prediction.prediction_id == "pred_001"
        assert prediction.agent_id == "agent_001"
        assert prediction.prediction_type == PredictionType.SUCCESS_PROBABILITY
        assert prediction.predicted_value == 0.75
        assert prediction.confidence_score == 0.85
        assert prediction.confidence_interval == confidence_interval
        assert prediction.risk_level == RiskLevel.MEDIUM
        assert len(prediction.risk_factors) == 1
        assert len(prediction.recommended_interventions) == 1
        assert prediction.model_name == "test_model"
        assert prediction.model_version == "1.0"
        assert prediction.features_used == ["feature1", "feature2"]
        assert prediction.prediction_horizon_ms == 60000
        assert prediction.created_at == created_at
    
    def test_prediction_result_validation_valid(self):
        """Test validation of valid PredictionResult."""
        prediction = PredictionResult(
            prediction_id="valid_pred",
            agent_id="valid_agent",
            prediction_type=PredictionType.COMPLETION_TIME,
            predicted_value=1000.0,
            confidence_score=0.8
        )
        assert prediction.validate() is True
    
    def test_prediction_result_validation_invalid(self):
        """Test validation with invalid data."""
        prediction = PredictionResult(
            prediction_id="",  # Invalid empty ID
            agent_id="agent",
            prediction_type=PredictionType.FAILURE_RISK,
            predicted_value=0.5,
            confidence_score=0.7
        )
        assert prediction.validate() is False
        
        prediction.prediction_id = "valid_id"
        prediction.confidence_score = 1.5  # Invalid confidence > 1.0
        assert prediction.validate() is False
    
    def test_prediction_result_expiration(self):
        """Test expiration functionality."""
        prediction = PredictionResult(
            prediction_id="exp_test",
            agent_id="agent",
            prediction_type=PredictionType.SUCCESS_PROBABILITY,
            predicted_value=0.8,
            confidence_score=0.9,
            created_at=datetime.utcnow()
        )
        
        # Initially not expired
        assert prediction.is_expired() is False
        
        # Set expiration in the past
        prediction.set_expiration(timedelta(seconds=-1))
        assert prediction.is_expired() is True
        
        # Set expiration in the future
        prediction.set_expiration(timedelta(hours=1))
        assert prediction.is_expired() is False
    
    def test_prediction_result_calculate_overall_risk_score(self):
        """Test overall risk score calculation."""
        risk1 = RiskFactor(
            factor_id="risk1",
            name="Risk 1",
            description="First risk",
            impact_score=0.8,
            probability=0.6,
            category="test"
        )
        
        risk2 = RiskFactor(
            factor_id="risk2",
            name="Risk 2",
            description="Second risk",
            impact_score=0.5,
            probability=0.4,
            category="test"
        )
        
        prediction = PredictionResult(
            prediction_id="risk_test",
            agent_id="agent",
            prediction_type=PredictionType.FAILURE_RISK,
            predicted_value=0.3,
            confidence_score=0.7,
            risk_factors=[risk1, risk2]
        )
        
        overall_risk = prediction.calculate_overall_risk_score()
        assert 0.0 <= overall_risk <= 1.0
        assert overall_risk > 0  # Should be positive with risk factors
    
    def test_prediction_result_add_risk_factor(self):
        """Test adding risk factors."""
        prediction = PredictionResult(
            prediction_id="add_risk_test",
            agent_id="agent",
            prediction_type=PredictionType.PERFORMANCE_SCORE,
            predicted_value=0.7,
            confidence_score=0.8
        )
        
        initial_risk_level = prediction.risk_level
        
        high_risk = RiskFactor(
            factor_id="high_risk",
            name="High Risk",
            description="High impact risk",
            impact_score=0.9,
            probability=0.8,
            category="critical"
        )
        
        prediction.add_risk_factor(high_risk)
        
        assert len(prediction.risk_factors) == 1
        assert prediction.risk_level != initial_risk_level  # Should update risk level
    
    def test_prediction_result_validate_prediction(self):
        """Test prediction validation against actual outcomes."""
        prediction = PredictionResult(
            prediction_id="validate_test",
            agent_id="agent",
            prediction_type=PredictionType.SUCCESS_PROBABILITY,
            predicted_value=0.8,
            confidence_score=0.9
        )
        
        # Test successful prediction
        accuracy = prediction.validate_prediction(1.0)  # Actual success
        assert prediction.actual_outcome == 1.0
        assert prediction.accuracy_score is not None
        assert 0.0 <= accuracy <= 1.0
        
        # Test for completion time prediction
        time_prediction = PredictionResult(
            prediction_id="time_test",
            agent_id="agent",
            prediction_type=PredictionType.COMPLETION_TIME,
            predicted_value=1000.0,
            confidence_score=0.8
        )
        
        accuracy = time_prediction.validate_prediction(1100.0)  # 10% error
        assert time_prediction.actual_outcome == 1100.0
        assert time_prediction.accuracy_score is not None
        assert 0.0 <= accuracy <= 1.0
    
    def test_prediction_result_serialization(self):
        """Test PredictionResult serialization."""
        confidence_interval = ConfidenceInterval(
            lower_bound=0.5,
            upper_bound=0.9,
            confidence_level=0.95
        )
        
        risk_factor = RiskFactor(
            factor_id="serialize_risk",
            name="Serialization Risk",
            description="Risk for testing",
            impact_score=0.6,
            probability=0.7,
            category="test"
        )
        
        intervention = Intervention(
            intervention_id="serialize_int",
            intervention_type=InterventionType.STRATEGY_CHANGE,
            name="Serialization Intervention",
            description="Intervention for testing",
            expected_impact=0.8,
            implementation_cost=0.4,
            urgency=RiskLevel.HIGH
        )
        
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(hours=1)
        
        original = PredictionResult(
            prediction_id="serialize_pred",
            agent_id="serialize_agent",
            prediction_type=PredictionType.RESOURCE_USAGE,
            predicted_value=0.65,
            confidence_score=0.85,
            confidence_interval=confidence_interval,
            risk_level=RiskLevel.HIGH,
            risk_factors=[risk_factor],
            recommended_interventions=[intervention],
            model_name="serialize_model",
            model_version="2.0",
            features_used=["feature_a", "feature_b"],
            prediction_horizon_ms=120000,
            created_at=created_at,
            expires_at=expires_at,
            actual_outcome=0.7,
            accuracy_score=0.92
        )
        
        # Test to_dict
        data = original.to_dict()
        assert data["prediction_id"] == "serialize_pred"
        assert data["prediction_type"] == "resource_usage"
        
        # Test from_dict
        restored = PredictionResult.from_dict(data)
        assert restored.prediction_id == original.prediction_id
        assert restored.agent_id == original.agent_id
        assert restored.prediction_type == original.prediction_type
        assert restored.predicted_value == original.predicted_value
        assert restored.confidence_score == original.confidence_score
        assert restored.confidence_interval.lower_bound == original.confidence_interval.lower_bound
        assert restored.risk_level == original.risk_level
        assert len(restored.risk_factors) == 1
        assert len(restored.recommended_interventions) == 1
        assert restored.model_name == original.model_name
        assert restored.model_version == original.model_version
        assert restored.features_used == original.features_used
        assert restored.prediction_horizon_ms == original.prediction_horizon_ms
        assert restored.created_at == original.created_at
        assert restored.expires_at == original.expires_at
        assert restored.actual_outcome == original.actual_outcome
        assert restored.accuracy_score == original.accuracy_score
    
    def test_prediction_result_json_serialization(self):
        """Test PredictionResult JSON serialization."""
        prediction = PredictionResult(
            prediction_id="json_pred",
            agent_id="json_agent",
            prediction_type=PredictionType.PERFORMANCE_SCORE,
            predicted_value=0.75,
            confidence_score=0.88
        )
        
        json_str = prediction.to_json()
        assert isinstance(json_str, str)
        
        restored = PredictionResult.from_json(json_str)
        assert restored.prediction_id == prediction.prediction_id
        assert restored.agent_id == prediction.agent_id
        assert restored.prediction_type == prediction.prediction_type
        assert restored.predicted_value == prediction.predicted_value
        assert restored.confidence_score == prediction.confidence_score


if __name__ == "__main__":
    pytest.main([__file__])