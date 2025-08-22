"""
Unit tests for the PerformancePredictor class.

Tests cover prediction accuracy, risk factor identification, intervention recommendations,
and model ensemble functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import uuid

# Import the classes to test
from escai_framework.core.performance_predictor import (
    PerformancePredictor, ExecutionStep, AgentState, TimeEstimate, LSTMPredictor
)
from escai_framework.models.epistemic_state import (
    EpistemicState, BeliefState, KnowledgeState, GoalState,
    BeliefType, GoalStatus
)
from escai_framework.models.prediction_result import (
    PredictionResult, PredictionType, RiskLevel, RiskFactor,
    Intervention, InterventionType, ConfidenceInterval
)


class TestPerformancePredictor:
    """Test cases for PerformancePredictor class."""
    
    @pytest.fixture
    def predictor(self):
        """Create a PerformancePredictor instance for testing."""
        return PerformancePredictor()
    
    @pytest.fixture
    def sample_epistemic_state(self):
        """Create a sample epistemic state for testing."""
        belief_states = [
            BeliefState(
                content="The task is achievable",
                belief_type=BeliefType.FACTUAL,
                confidence=0.8,
                evidence=["Previous successful attempts"]
            ),
            BeliefState(
                content="Resources are sufficient",
                belief_type=BeliefType.PROBABILISTIC,
                confidence=0.6,
                evidence=["Resource monitoring data"]
            )
        ]
        
        knowledge_state = KnowledgeState(
            facts=["Task requires API calls", "Database is available"],
            rules=["If API fails, retry with backoff"],
            concepts={"API": {"type": "service"}, "Database": {"type": "storage"}},
            relationships=[{"subject": "API", "predicate": "depends_on", "object": "Database"}],
            confidence_score=0.7
        )
        
        goal_states = [
            GoalState(
                description="Complete data processing task",
                status=GoalStatus.ACTIVE,
                priority=8,
                progress=0.4
            ),
            GoalState(
                description="Validate results",
                status=GoalStatus.ACTIVE,
                priority=6,
                progress=0.1
            )
        ]
        
        return EpistemicState(
            agent_id="test_agent_001",
            timestamp=datetime.utcnow(),
            belief_states=belief_states,
            knowledge_state=knowledge_state,
            goal_states=goal_states,
            confidence_level=0.7,
            uncertainty_score=0.3,
            decision_context={"task_type": "data_processing", "complexity": "medium"}
        )
    
    @pytest.fixture
    def sample_execution_history(self):
        """Create sample execution history for testing."""
        steps = []
        base_time = datetime.utcnow()
        
        for i in range(5):
            step = ExecutionStep(
                step_id=f"step_{i}",
                timestamp=base_time + timedelta(seconds=i*10),
                action=f"action_{i}",
                duration_ms=1000 + i*200,
                success=i < 4,  # Last step fails
                context={"step_number": i, "complexity": "medium"}
            )
            steps.append(step)
        
        return steps
    
    @pytest.fixture
    def sample_agent_state(self, sample_epistemic_state, sample_execution_history):
        """Create a sample agent state for testing."""
        return AgentState(
            agent_id="test_agent_001",
            current_task="data_processing_task",
            execution_history=sample_execution_history,
            epistemic_state=sample_epistemic_state,
            resource_usage={"cpu": 0.6, "memory": 0.4, "network": 0.3},
            performance_metrics={"accuracy": 0.85, "speed": 0.7, "efficiency": 0.8}
        )
    
    @pytest.mark.asyncio
    async def test_predict_success_basic(self, predictor, sample_epistemic_state):
        """Test basic success prediction functionality."""
        result = await predictor.predict_success(sample_epistemic_state)
        
        assert isinstance(result, PredictionResult)
        assert result.agent_id == "test_agent_001"
        assert result.prediction_type == PredictionType.SUCCESS_PROBABILITY
        assert 0.0 <= result.predicted_value <= 1.0
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.model_name in ["EnsembleSuccessPredictor", "DefaultPredictor"]
        assert len(result.features_used) > 0
    
    @pytest.mark.asyncio
    async def test_predict_success_with_high_confidence(self, predictor):
        """Test success prediction with high confidence state."""
        high_confidence_state = EpistemicState(
            agent_id="test_agent_002",
            timestamp=datetime.utcnow(),
            belief_states=[
                BeliefState("Task is easy", BeliefType.FACTUAL, 0.95),
                BeliefState("All resources available", BeliefType.FACTUAL, 0.9)
            ],
            knowledge_state=KnowledgeState(confidence_score=0.9),
            goal_states=[
                GoalState("Complete task", GoalStatus.ACTIVE, 8, 0.8)
            ],
            confidence_level=0.9,
            uncertainty_score=0.1
        )
        
        result = await predictor.predict_success(high_confidence_state)
        
        # High confidence should lead to higher success probability
        assert result.predicted_value > 0.6
        assert result.confidence_score > 0.5
        assert len(result.risk_factors) <= 2  # Should have fewer risk factors
    
    @pytest.mark.asyncio
    async def test_predict_success_with_low_confidence(self, predictor):
        """Test success prediction with low confidence state."""
        low_confidence_state = EpistemicState(
            agent_id="test_agent_003",
            timestamp=datetime.utcnow(),
            belief_states=[
                BeliefState("Task might be difficult", BeliefType.PROBABILISTIC, 0.2),
                BeliefState("Resources uncertain", BeliefType.PROBABILISTIC, 0.3)
            ],
            knowledge_state=KnowledgeState(confidence_score=0.2),
            goal_states=[
                GoalState("Complete task", GoalStatus.ACTIVE, 8, 0.1)
            ],
            confidence_level=0.2,
            uncertainty_score=0.8
        )
        
        result = await predictor.predict_success(low_confidence_state)
        
        # Low confidence should lead to lower success probability
        assert result.predicted_value < 0.7
        assert len(result.risk_factors) > 0  # Should identify risk factors
        assert any(rf.factor_id == 'low_confidence' for rf in result.risk_factors)
        assert any(rf.factor_id == 'high_uncertainty' for rf in result.risk_factors)
    
    @pytest.mark.asyncio
    async def test_estimate_completion_time_basic(self, predictor, sample_execution_history):
        """Test basic completion time estimation."""
        result = await predictor.estimate_completion_time(sample_execution_history)
        
        assert isinstance(result, TimeEstimate)
        assert result.estimated_duration_ms > 0
        assert isinstance(result.confidence_interval, ConfidenceInterval)
        assert result.confidence_interval.validate()
        assert len(result.factors_considered) > 0
    
    @pytest.mark.asyncio
    async def test_estimate_completion_time_empty_history(self, predictor):
        """Test completion time estimation with empty history."""
        result = await predictor.estimate_completion_time([])
        
        assert isinstance(result, TimeEstimate)
        assert result.estimated_duration_ms == 60000  # Default 1 minute
        assert "default_estimate" in result.factors_considered
    
    @pytest.mark.asyncio
    async def test_identify_risk_factors_basic(self, predictor, sample_agent_state):
        """Test basic risk factor identification."""
        risk_factors = await predictor.identify_risk_factors(sample_agent_state)
        
        assert isinstance(risk_factors, list)
        for rf in risk_factors:
            assert isinstance(rf, RiskFactor)
            assert rf.validate()
            assert 0.0 <= rf.impact_score <= 1.0
            assert 0.0 <= rf.probability <= 1.0
    
    @pytest.mark.asyncio
    async def test_identify_risk_factors_low_confidence(self, predictor):
        """Test risk factor identification for low confidence scenario."""
        # Create agent state with low confidence
        low_confidence_epistemic = EpistemicState(
            agent_id="test_agent_004",
            timestamp=datetime.utcnow(),
            belief_states=[],
            knowledge_state=KnowledgeState(),
            goal_states=[],
            confidence_level=0.1,  # Very low confidence
            uncertainty_score=0.9   # Very high uncertainty
        )
        
        agent_state = AgentState(
            agent_id="test_agent_004",
            current_task="test_task",
            execution_history=[],
            epistemic_state=low_confidence_epistemic,
            resource_usage={"cpu": 0.9},  # High resource usage
            performance_metrics={}
        )
        
        risk_factors = await predictor.identify_risk_factors(agent_state)
        
        # Should identify multiple risk factors
        assert len(risk_factors) > 0
        
        # Check for specific risk factors
        risk_ids = [rf.factor_id for rf in risk_factors]
        assert 'low_confidence' in risk_ids
        assert 'high_uncertainty' in risk_ids
    
    @pytest.mark.asyncio
    async def test_recommend_interventions_high_risk(self, predictor, sample_epistemic_state):
        """Test intervention recommendations for high-risk scenarios."""
        # Create high-risk prediction
        high_risk_prediction = PredictionResult(
            prediction_id=str(uuid.uuid4()),
            agent_id="test_agent_005",
            prediction_type=PredictionType.SUCCESS_PROBABILITY,
            predicted_value=0.2,  # Low success probability
            confidence_score=0.8,
            risk_factors=[
                RiskFactor(
                    factor_id='critical_risk',
                    name='Critical Risk',
                    description='High probability of failure',
                    impact_score=0.9,
                    probability=0.8,
                    category='performance'
                )
            ]
        )
        
        interventions = await predictor.recommend_interventions(high_risk_prediction)
        
        assert isinstance(interventions, list)
        assert len(interventions) > 0
        
        # Should recommend critical interventions
        intervention_types = [i.intervention_type for i in interventions]
        assert InterventionType.HUMAN_INTERVENTION in intervention_types
        
        # Check intervention properties
        for intervention in interventions:
            assert isinstance(intervention, Intervention)
            assert intervention.validate()
            assert 0.0 <= intervention.expected_impact <= 1.0
            assert 0.0 <= intervention.implementation_cost <= 1.0
    
    @pytest.mark.asyncio
    async def test_recommend_interventions_medium_risk(self, predictor):
        """Test intervention recommendations for medium-risk scenarios."""
        medium_risk_prediction = PredictionResult(
            prediction_id=str(uuid.uuid4()),
            agent_id="test_agent_006",
            prediction_type=PredictionType.SUCCESS_PROBABILITY,
            predicted_value=0.5,  # Medium success probability
            confidence_score=0.7,
            risk_factors=[
                RiskFactor(
                    factor_id='medium_risk',
                    name='Medium Risk',
                    description='Moderate probability of issues',
                    impact_score=0.5,
                    probability=0.6,
                    category='performance'
                )
            ]
        )
        
        interventions = await predictor.recommend_interventions(medium_risk_prediction)
        
        assert len(interventions) > 0
        
        # Should recommend moderate interventions
        intervention_types = [i.intervention_type for i in interventions]
        assert InterventionType.STRATEGY_CHANGE in intervention_types or \
               InterventionType.RESOURCE_ALLOCATION in intervention_types
    
    @pytest.mark.asyncio
    async def test_extract_features_from_state(self, predictor, sample_epistemic_state):
        """Test feature extraction from epistemic state."""
        features = await predictor._extract_features_from_state(sample_epistemic_state)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check for expected features
        expected_features = [
            'confidence_level', 'uncertainty_score', 'belief_count', 'goal_count',
            'avg_belief_confidence', 'avg_goal_progress', 'knowledge_confidence'
        ]
        
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
    
    @pytest.mark.asyncio
    async def test_predict_with_ensemble_basic(self, predictor):
        """Test ensemble prediction with basic features."""
        features = {
            'confidence_level': 0.7,
            'uncertainty_score': 0.3,
            'avg_goal_progress': 0.5,
            'belief_count': 3,
            'goal_count': 2
        }
        
        prediction = await predictor._predict_with_ensemble(features, 'success')
        
        assert isinstance(prediction, float)
        assert 0.0 <= prediction <= 1.0
    
    @pytest.mark.asyncio
    async def test_predict_with_ensemble_empty_features(self, predictor):
        """Test ensemble prediction with empty features."""
        prediction = await predictor._predict_with_ensemble({}, 'success')
        
        assert isinstance(prediction, float)
        assert prediction == 0.5  # Default prediction
    
    def test_heuristic_prediction_success(self, predictor):
        """Test heuristic prediction for success."""
        features = {
            'confidence_level': 0.8,
            'uncertainty_score': 0.2,
            'avg_goal_progress': 0.7
        }
        
        prediction = predictor._heuristic_prediction(features, 'success')
        
        assert isinstance(prediction, float)
        assert 0.0 <= prediction <= 1.0
        assert prediction > 0.5  # Should be optimistic with good features
    
    def test_heuristic_prediction_completion_time(self, predictor):
        """Test heuristic prediction for completion time."""
        features = {
            'avg_goal_progress': 0.3,
            'confidence_level': 0.6
        }
        
        prediction = predictor._heuristic_prediction(features, 'completion_time')
        
        assert isinstance(prediction, float)
        assert 0.1 <= prediction <= 2.0  # Within expected range
    
    @pytest.mark.asyncio
    async def test_extract_temporal_sequence(self, predictor, sample_execution_history):
        """Test temporal sequence extraction from execution history."""
        temporal_features = await predictor._extract_temporal_sequence(sample_execution_history)
        
        assert isinstance(temporal_features, list)
        assert len(temporal_features) == len(sample_execution_history)
        
        for i, features in enumerate(temporal_features):
            assert isinstance(features, dict)
            assert 'step_index' in features
            assert 'duration_ms' in features
            assert 'success' in features
            assert features['step_index'] == float(i)
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_interval(self, predictor):
        """Test confidence interval calculation."""
        features = {'uncertainty_score': 0.3}
        prediction = 0.7
        
        ci = await predictor._calculate_confidence_interval(prediction, features, 'success')
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.validate()
        assert ci.lower_bound <= prediction <= ci.upper_bound
        assert 0.0 <= ci.confidence_level <= 1.0
    
    def test_calculate_prediction_confidence(self, predictor):
        """Test prediction confidence calculation."""
        features = {
            'confidence_level': 0.8,
            'uncertainty_score': 0.2,
            'belief_count': 5,
            'goal_count': 3
        }
        
        confidence = predictor._calculate_prediction_confidence(features)
        
        assert isinstance(confidence, float)
        assert 0.1 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_comprehensive_feature_extraction(self, predictor, sample_agent_state):
        """Test comprehensive feature extraction from agent state."""
        features = await predictor._extract_comprehensive_features(sample_agent_state)
        
        assert isinstance(features, dict)
        assert len(features) > 10  # Should extract many features
        
        # Check for different feature categories
        feature_keys = list(features.keys())
        
        # Epistemic features
        assert any('confidence' in key for key in feature_keys)
        assert any('belief' in key for key in feature_keys)
        assert any('goal' in key for key in feature_keys)
        
        # Resource features
        assert any('resource' in key for key in feature_keys)
        
        # Performance features
        assert any('performance' in key for key in feature_keys)
    
    @pytest.mark.asyncio
    async def test_calculate_risk_score(self, predictor, sample_agent_state):
        """Test risk score calculation for specific risk factors."""
        features = await predictor._extract_comprehensive_features(sample_agent_state)
        
        # Test low confidence risk
        low_conf_def = predictor.risk_factor_definitions['low_confidence']
        risk_score = await predictor._calculate_risk_score(
            'low_confidence', low_conf_def, features, sample_agent_state
        )
        
        assert isinstance(risk_score, float)
        assert 0.0 <= risk_score <= 1.0
    
    def test_calculate_risk_probability(self, predictor):
        """Test risk probability calculation."""
        risk_score = 0.6
        probability = predictor._calculate_risk_probability(risk_score)
        
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
        assert probability >= risk_score  # Should be at least as high as risk score
    
    @pytest.mark.asyncio
    async def test_error_handling_predict_success(self, predictor):
        """Test error handling in predict_success method."""
        # Create invalid epistemic state
        invalid_state = EpistemicState(
            agent_id="",  # Invalid empty agent_id
            timestamp=datetime.utcnow(),
            belief_states=[],
            knowledge_state=KnowledgeState(),
            goal_states=[]
        )
        
        # Should not raise exception, should return error prediction
        result = await predictor.predict_success(invalid_state)
        
        assert isinstance(result, PredictionResult)
        # Should still return a valid prediction even with invalid input
        assert result.confidence_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_error_handling_estimate_completion_time(self, predictor):
        """Test error handling in estimate_completion_time method."""
        # Create execution history with invalid data
        invalid_history = [
            ExecutionStep(
                step_id="invalid",
                timestamp=datetime.utcnow(),
                action="test",
                duration_ms=-1000,  # Invalid negative duration
                success=True
            )
        ]
        
        # Should not raise exception, should return reasonable estimate
        result = await predictor.estimate_completion_time(invalid_history)
        
        assert isinstance(result, TimeEstimate)
        assert result.estimated_duration_ms > 0
    
    def test_model_initialization(self, predictor):
        """Test that predictor initializes correctly."""
        assert isinstance(predictor.risk_factor_definitions, dict)
        assert isinstance(predictor.intervention_templates, dict)
        assert isinstance(predictor.feature_extractors, dict)
        assert len(predictor.risk_factor_definitions) > 0
        assert len(predictor.intervention_templates) > 0
        assert len(predictor.feature_extractors) > 0
    
    def test_risk_factor_definitions(self, predictor):
        """Test that risk factor definitions are properly structured."""
        for risk_id, risk_def in predictor.risk_factor_definitions.items():
            assert isinstance(risk_id, str)
            assert isinstance(risk_def, dict)
            assert 'name' in risk_def
            assert 'description' in risk_def
            assert 'category' in risk_def
            assert 'threshold' in risk_def
            assert 'mitigation_strategies' in risk_def
            assert isinstance(risk_def['mitigation_strategies'], list)
    
    def test_intervention_templates(self, predictor):
        """Test that intervention templates are properly structured."""
        for template_id, template in predictor.intervention_templates.items():
            assert isinstance(template_id, str)
            assert isinstance(template, dict)
            assert 'type' in template
            assert 'name' in template
            assert 'description' in template
            assert 'expected_impact' in template
            assert 'implementation_cost' in template
            assert 'urgency' in template


class TestLSTMPredictor:
    """Test cases for LSTMPredictor neural network."""
    
    @pytest.fixture
    def lstm_model(self):
        """Create an LSTM model for testing."""
        try:
            import torch
            return LSTMPredictor(input_size=10, hidden_size=16, num_layers=1)
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_lstm_initialization(self, lstm_model):
        """Test LSTM model initialization."""
        assert lstm_model.hidden_size == 16
        assert lstm_model.num_layers == 1
        assert hasattr(lstm_model, 'lstm')
        assert hasattr(lstm_model, 'fc')
        assert hasattr(lstm_model, 'dropout')
    
    def test_lstm_forward_pass(self, lstm_model):
        """Test LSTM forward pass."""
        try:
            import torch
            
            # Create sample input: batch_size=2, sequence_length=5, input_size=10
            x = torch.randn(2, 5, 10)
            
            output = lstm_model(x)
            
            assert output.shape == (2, 1)  # batch_size, output_size
            assert not torch.isnan(output).any()
            
        except ImportError:
            pytest.skip("PyTorch not available")


class TestExecutionStep:
    """Test cases for ExecutionStep data class."""
    
    def test_execution_step_creation(self):
        """Test ExecutionStep creation and validation."""
        step = ExecutionStep(
            step_id="test_step",
            timestamp=datetime.utcnow(),
            action="test_action",
            duration_ms=1500,
            success=True,
            context={"key": "value"}
        )
        
        assert step.step_id == "test_step"
        assert step.action == "test_action"
        assert step.duration_ms == 1500
        assert step.success is True
        assert step.context == {"key": "value"}
    
    def test_execution_step_to_dict(self):
        """Test ExecutionStep serialization to dictionary."""
        timestamp = datetime.utcnow()
        step = ExecutionStep(
            step_id="test_step",
            timestamp=timestamp,
            action="test_action",
            duration_ms=1500,
            success=True
        )
        
        step_dict = step.to_dict()
        
        assert isinstance(step_dict, dict)
        assert step_dict['step_id'] == "test_step"
        assert step_dict['timestamp'] == timestamp.isoformat()
        assert step_dict['action'] == "test_action"
        assert step_dict['duration_ms'] == 1500
        assert step_dict['success'] is True


class TestAgentState:
    """Test cases for AgentState data class."""
    
    @pytest.fixture
    def sample_agent_state(self):
        """Create a sample agent state for testing."""
        epistemic_state = EpistemicState(
            agent_id="test_agent",
            timestamp=datetime.utcnow(),
            belief_states=[],
            knowledge_state=KnowledgeState(),
            goal_states=[]
        )
        
        execution_history = [
            ExecutionStep(
                step_id="step_1",
                timestamp=datetime.utcnow(),
                action="action_1",
                duration_ms=1000,
                success=True
            )
        ]
        
        return AgentState(
            agent_id="test_agent",
            current_task="test_task",
            execution_history=execution_history,
            epistemic_state=epistemic_state,
            resource_usage={"cpu": 0.5, "memory": 0.3},
            performance_metrics={"accuracy": 0.9}
        )
    
    def test_agent_state_creation(self, sample_agent_state):
        """Test AgentState creation."""
        assert sample_agent_state.agent_id == "test_agent"
        assert sample_agent_state.current_task == "test_task"
        assert len(sample_agent_state.execution_history) == 1
        assert isinstance(sample_agent_state.epistemic_state, EpistemicState)
        assert sample_agent_state.resource_usage["cpu"] == 0.5
        assert sample_agent_state.performance_metrics["accuracy"] == 0.9
    
    def test_agent_state_to_dict(self, sample_agent_state):
        """Test AgentState serialization to dictionary."""
        state_dict = sample_agent_state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert state_dict['agent_id'] == "test_agent"
        assert state_dict['current_task'] == "test_task"
        assert isinstance(state_dict['execution_history'], list)
        assert isinstance(state_dict['epistemic_state'], dict)
        assert isinstance(state_dict['resource_usage'], dict)
        assert isinstance(state_dict['performance_metrics'], dict)


class TestTimeEstimate:
    """Test cases for TimeEstimate data class."""
    
    def test_time_estimate_creation(self):
        """Test TimeEstimate creation."""
        ci = ConfidenceInterval(1000, 3000, 0.8)
        estimate = TimeEstimate(
            estimated_duration_ms=2000,
            confidence_interval=ci,
            factors_considered=["factor1", "factor2"]
        )
        
        assert estimate.estimated_duration_ms == 2000
        assert isinstance(estimate.confidence_interval, ConfidenceInterval)
        assert estimate.factors_considered == ["factor1", "factor2"]
    
    def test_time_estimate_to_dict(self):
        """Test TimeEstimate serialization to dictionary."""
        ci = ConfidenceInterval(1000, 3000, 0.8)
        estimate = TimeEstimate(
            estimated_duration_ms=2000,
            confidence_interval=ci,
            factors_considered=["factor1", "factor2"]
        )
        
        estimate_dict = estimate.to_dict()
        
        assert isinstance(estimate_dict, dict)
        assert estimate_dict['estimated_duration_ms'] == 2000
        assert isinstance(estimate_dict['confidence_interval'], dict)
        assert estimate_dict['factors_considered'] == ["factor1", "factor2"]


if __name__ == "__main__":
    pytest.main([__file__])