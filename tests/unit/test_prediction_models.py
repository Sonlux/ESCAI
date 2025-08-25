"""
Unit tests for prediction models.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import asyncio
from datetime import datetime

from escai_framework.analytics.prediction_models import (
    EnsemblePredictor, OnlineLearningPredictor, LSTMPredictor,
    HyperparameterTuner, ModelConfig, SequenceDataset
)
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState
from escai_framework.models.behavioral_pattern import ExecutionSequence, ExecutionStep
from escai_framework.models.prediction_result import PredictionResult


@pytest.fixture
def model_config():
    """Create model configuration for testing."""
    return ModelConfig(
        lstm_hidden_size=64,
        lstm_num_layers=1,
        lstm_epochs=5,
        lstm_batch_size=16,
        rf_n_estimators=10,
        xgb_n_estimators=10,
        sequence_length=10,
        cv_folds=3
    )


@pytest.fixture
def sample_sequences():
    """Create sample execution sequences."""
    sequences = []
    for i in range(10):
        steps = [
            ExecutionStep(
                step_id=f"step_{i}_{j}",
                step_type="action",
                action=f"action_{j}",
                duration=1.0 + j * 0.1,
                success_probability=0.8 + j * 0.02,
                context={"inputs": [f"input_{j}"], "outputs": [f"output_{j}"]},
                error_message=None if j < 3 else f"error_{j}"
            )
            for j in range(5)
        ]
        
        sequences.append(ExecutionSequence(
            sequence_id=f"seq_{i}",
            agent_id=f"agent_{i % 3}",
            steps=steps,
            total_duration=5.0 + i * 0.5,
            success_rate=0.8 + i * 0.01
        ))
    
    return sequences


@pytest.fixture
def sample_epistemic_states():
    """Create sample epistemic states."""
    states = []
    for i in range(10):
        belief_states = [
            BeliefState(
                belief_id=f"belief_{i}_{j}",
                content=f"belief_content_{j}",
                confidence=0.7 + j * 0.05,
                source="test",
                timestamp=datetime.now()
            )
            for j in range(3)
        ]
        
        knowledge_state = KnowledgeState(
            facts=[f"fact_{i}_{j}" for j in range(2)],
            rules=[f"rule_{i}_{j}" for j in range(2)],
            concepts=[f"concept_{i}_{j}" for j in range(2)]
        )
        
        goal_state = GoalState(
            active_goals=[f"goal_{i}_{j}" for j in range(2)],
            completed_goals=[f"completed_{i}_{j}" for j in range(1)],
            failed_goals=[]
        )
        
        states.append(EpistemicState(
            agent_id=f"agent_{i % 3}",
            timestamp=datetime.now(),
            belief_states=belief_states,
            knowledge_state=knowledge_state,
            goal_state=goal_state,
            confidence_level=0.7 + i * 0.02,
            uncertainty_score=0.3 - i * 0.01,
            decision_context={"context": f"context_{i}"}
        ))
    
    return states


@pytest.fixture
def sample_outcomes():
    """Create sample outcomes."""
    return [True, False, True, True, False, True, False, True, True, False]


@pytest.fixture
def sample_completion_times():
    """Create sample completion times."""
    return [10.0, 15.0, 8.0, 12.0, 20.0, 9.0, 18.0, 11.0, 7.0, 16.0]


class TestLSTMPredictor:
    """Test LSTM neural network predictor."""
    
    def test_initialization(self):
        """Test LSTM model initialization."""
        model = LSTMPredictor(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            output_size=2,
            dropout=0.2
        )
        
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert isinstance(model.lstm, torch.nn.LSTM)
        assert isinstance(model.fc, torch.nn.Linear)
    
    def test_forward_pass(self):
        """Test LSTM forward pass."""
        model = LSTMPredictor(
            input_size=5,
            hidden_size=32,
            num_layers=1,
            output_size=2
        )
        
        # Create sample input
        batch_size, seq_length, input_size = 4, 10, 5
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (batch_size, 2)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


class TestSequenceDataset:
    """Test PyTorch dataset for sequences."""
    
    def test_initialization(self):
        """Test dataset initialization."""
        sequences = np.random.randn(10, 20, 5)
        targets = np.random.randn(10, 2)
        
        dataset = SequenceDataset(sequences, targets)
        
        assert len(dataset) == 10
        assert dataset.sequences.shape == (10, 20, 5)
        assert dataset.targets.shape == (10, 2)
    
    def test_getitem(self):
        """Test dataset item retrieval."""
        sequences = np.random.randn(5, 10, 3)
        targets = np.random.randn(5, 2)
        
        dataset = SequenceDataset(sequences, targets)
        
        seq, target = dataset[0]
        assert seq.shape == (10, 3)
        assert target.shape == (2,)
        assert isinstance(seq, torch.Tensor)
        assert isinstance(target, torch.Tensor)


class TestEnsemblePredictor:
    """Test ensemble prediction model."""
    
    def test_initialization(self, model_config):
        """Test ensemble predictor initialization."""
        predictor = EnsemblePredictor(model_config)
        
        assert predictor.config == model_config
        assert predictor.lstm_model is None  # Not initialized until training
        assert predictor.rf_classifier is not None
        assert predictor.rf_regressor is not None
        assert predictor.xgb_classifier is not None
        assert predictor.xgb_regressor is not None
        assert predictor.scaler is not None
    
    @pytest.mark.asyncio
    async def test_train(self, model_config, sample_sequences, sample_epistemic_states, 
                        sample_outcomes, sample_completion_times):
        """Test ensemble model training."""
        predictor = EnsemblePredictor(model_config)
        
        # Mock the individual training methods to avoid long training times
        with patch.object(predictor, '_train_lstm') as mock_lstm, \
             patch.object(predictor, '_train_random_forest') as mock_rf, \
             patch.object(predictor, '_train_xgboost') as mock_xgb:
            
            mock_lstm.return_value = {'final_loss': 0.1, 'epochs_trained': 5}
            mock_rf.return_value = {'classification_accuracy': 0.8, 'regression_mse': 0.2}
            mock_xgb.return_value = {'classification_accuracy': 0.85, 'regression_mse': 0.15}
            
            metrics = await predictor.train(
                sample_sequences, sample_epistemic_states, 
                sample_outcomes, sample_completion_times
            )
            
            assert 'lstm' in metrics
            assert 'random_forest' in metrics
            assert 'xgboost' in metrics
            assert 'ensemble_weights' in metrics
            
            # Verify training methods were called
            mock_lstm.assert_called_once()
            mock_rf.assert_called_once()
            mock_xgb.assert_called_once()
    
    def test_prepare_features(self, model_config, sample_sequences, sample_epistemic_states):
        """Test feature preparation."""
        predictor = EnsemblePredictor(model_config)
        
        X_seq, X_static = predictor._prepare_features(sample_sequences, sample_epistemic_states)
        
        assert isinstance(X_seq, np.ndarray)
        assert isinstance(X_static, np.ndarray)
        assert X_seq.shape[0] == len(sample_sequences)
        assert X_static.shape[0] == len(sample_sequences)
        assert X_seq.ndim == 3  # (samples, time_steps, features)
        assert X_static.ndim == 2  # (samples, features)
    
    def test_extract_sequence_features(self, model_config, sample_sequences):
        """Test sequence feature extraction."""
        predictor = EnsemblePredictor(model_config)
        
        features = predictor._extract_sequence_features(sample_sequences[0])
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_sequences[0].steps)
        assert features.shape[1] == 5  # Expected number of features per step
    
    def test_extract_static_features(self, model_config, sample_sequences, sample_epistemic_states):
        """Test static feature extraction."""
        predictor = EnsemblePredictor(model_config)
        
        features = predictor._extract_static_features(sample_sequences[0], sample_epistemic_states[0])
        
        assert isinstance(features, list)
        assert len(features) == 8  # Expected number of static features
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_calculate_confidence(self, model_config):
        """Test confidence calculation."""
        predictor = EnsemblePredictor(model_config)
        
        # High agreement (low std) should give high confidence
        high_agreement = [0.8, 0.82, 0.81]
        confidence = predictor._calculate_confidence(high_agreement)
        assert confidence > 0.8
        
        # Low agreement (high std) should give low confidence
        low_agreement = [0.2, 0.8, 0.9]
        confidence = predictor._calculate_confidence(low_agreement)
        assert confidence < 0.5
    
    def test_identify_risk_factors(self, model_config, sample_sequences, sample_epistemic_states):
        """Test risk factor identification."""
        predictor = EnsemblePredictor(model_config)
        
        # Create a high-risk state
        high_risk_state = sample_epistemic_states[0]
        high_risk_state.uncertainty_score = 0.8
        high_risk_state.confidence_level = 0.2
        
        risk_factors = predictor._identify_risk_factors(sample_sequences[0], high_risk_state)
        
        assert isinstance(risk_factors, list)
        assert "high_uncertainty" in risk_factors
        assert "low_confidence" in risk_factors


class TestOnlineLearningPredictor:
    """Test online learning predictor."""
    
    def test_initialization(self):
        """Test online predictor initialization."""
        predictor = OnlineLearningPredictor()
        
        assert predictor.classifier is not None
        assert predictor.regressor is not None
        assert predictor.scaler is not None
        assert predictor.performance_history is not None
    
    @pytest.mark.asyncio
    async def test_learn_online(self):
        """Test online learning functionality."""
        predictor = OnlineLearningPredictor()
        
        features = {
            'sequence_length': 5,
            'total_duration': 10.0,
            'confidence_level': 0.8,
            'uncertainty_score': 0.2
        }
        
        await predictor.learn_online(features, True, 12.0)
        
        # Check that performance history is updated
        assert len(predictor.performance_history['accuracy']) > 0 or \
               len(predictor.performance_history['mae']) > 0
    
    @pytest.mark.asyncio
    async def test_predict_online(self):
        """Test online prediction."""
        predictor = OnlineLearningPredictor()
        
        # Train with some data first
        features = {
            'sequence_length': 5,
            'total_duration': 10.0,
            'confidence_level': 0.8,
            'uncertainty_score': 0.2
        }
        
        await predictor.learn_online(features, True, 12.0)
        
        # Make prediction
        prediction = await predictor.predict_online(features)
        
        assert 'success_probability' in prediction
        assert 'completion_time' in prediction
        assert 'model_confidence' in prediction
        assert 0 <= prediction['success_probability'] <= 1
        assert prediction['completion_time'] > 0
        assert 0 <= prediction['model_confidence'] <= 1
    
    def test_detect_concept_drift(self):
        """Test concept drift detection."""
        predictor = OnlineLearningPredictor()
        
        # Simulate performance degradation
        predictor.performance_history['accuracy'] = [0.9] * 10 + [0.7] * 10
        
        drift_detected = predictor.detect_concept_drift()
        assert drift_detected
        
        # Simulate stable performance
        predictor.performance_history['accuracy'] = [0.8] * 20
        
        drift_detected = predictor.detect_concept_drift()
        assert not drift_detected
    
    def test_adapt_to_drift(self):
        """Test adaptation to concept drift."""
        predictor = OnlineLearningPredictor()
        
        # Store original models
        original_classifier = predictor.classifier
        original_regressor = predictor.regressor
        
        # Adapt to drift
        predictor.adapt_to_drift()
        
        # Models should be reset (new instances)
        assert predictor.classifier is not original_classifier
        assert predictor.regressor is not original_regressor


class TestHyperparameterTuner:
    """Test hyperparameter tuning functionality."""
    
    def test_initialization(self):
        """Test tuner initialization."""
        tuner = HyperparameterTuner()
        
        assert tuner.best_params == {}
        assert tuner.tuning_history == {}
    
    @pytest.mark.asyncio
    async def test_tune_random_forest(self):
        """Test Random Forest hyperparameter tuning."""
        tuner = HyperparameterTuner()
        
        # Create sample data
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        with patch('sklearn.model_selection.GridSearchCV') as mock_grid:
            mock_grid_instance = Mock()
            mock_grid_instance.fit.return_value = None
            mock_grid_instance.best_params_ = {'n_estimators': 100, 'max_depth': 10}
            mock_grid_instance.best_score_ = 0.85
            mock_grid_instance.cv_results_ = {'mean_test_score': [0.8, 0.85, 0.82]}
            mock_grid.return_value = mock_grid_instance
            
            result = await tuner.tune_random_forest(X, y, 'classification')
            
            assert 'best_params' in result
            assert 'best_score' in result
            assert 'cv_results' in result
            assert result['best_params'] == {'n_estimators': 100, 'max_depth': 10}
            assert result['best_score'] == 0.85
    
    @pytest.mark.asyncio
    async def test_tune_xgboost(self):
        """Test XGBoost hyperparameter tuning."""
        tuner = HyperparameterTuner()
        
        # Create sample data
        X = np.random.randn(50, 5)
        y = np.random.randn(50)  # Regression target
        
        with patch('sklearn.model_selection.GridSearchCV') as mock_grid:
            mock_grid_instance = Mock()
            mock_grid_instance.fit.return_value = None
            mock_grid_instance.best_params_ = {'n_estimators': 200, 'learning_rate': 0.1}
            mock_grid_instance.best_score_ = -0.15  # Negative MSE
            mock_grid_instance.cv_results_ = {'mean_test_score': [-0.2, -0.15, -0.18]}
            mock_grid.return_value = mock_grid_instance
            
            result = await tuner.tune_xgboost(X, y, 'regression')
            
            assert 'best_params' in result
            assert 'best_score' in result
            assert result['best_params'] == {'n_estimators': 200, 'learning_rate': 0.1}
            assert result['best_score'] == -0.15


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.lstm_hidden_size == 128
        assert config.lstm_num_layers == 2
        assert config.lstm_dropout == 0.2
        assert config.rf_n_estimators == 100
        assert config.xgb_n_estimators == 100
        assert config.sequence_length == 50
        assert config.cv_folds == 5
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            lstm_hidden_size=64,
            lstm_num_layers=1,
            rf_n_estimators=50,
            cv_folds=3
        )
        
        assert config.lstm_hidden_size == 64
        assert config.lstm_num_layers == 1
        assert config.rf_n_estimators == 50
        assert config.cv_folds == 3


@pytest.mark.integration
class TestPredictionModelsIntegration:
    """Integration tests for prediction models."""
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction_workflow(self, model_config, sample_sequences, 
                                              sample_epistemic_states, sample_outcomes, 
                                              sample_completion_times):
        """Test complete ensemble prediction workflow."""
        predictor = EnsemblePredictor(model_config)
        
        # Mock training to avoid long execution times
        with patch.object(predictor, '_train_lstm') as mock_lstm, \
             patch.object(predictor, '_train_random_forest') as mock_rf, \
             patch.object(predictor, '_train_xgboost') as mock_xgb:
            
            mock_lstm.return_value = {'final_loss': 0.1}
            mock_rf.return_value = {'classification_accuracy': 0.8}
            mock_xgb.return_value = {'classification_accuracy': 0.85}
            
            # Train the model
            await predictor.train(
                sample_sequences, sample_epistemic_states,
                sample_outcomes, sample_completion_times
            )
            
            # Mock prediction methods
            with patch.object(predictor, '_predict_lstm') as mock_pred_lstm, \
                 patch.object(predictor, '_predict_random_forest') as mock_pred_rf, \
                 patch.object(predictor, '_predict_xgboost') as mock_pred_xgb:
                
                mock_pred_lstm.return_value = {
                    'success': np.array([0.8, 0.6, 0.9]),
                    'time': np.array([10.0, 15.0, 8.0])
                }
                mock_pred_rf.return_value = {
                    'success': np.array([0.75, 0.65, 0.85]),
                    'time': np.array([11.0, 14.0, 9.0])
                }
                mock_pred_xgb.return_value = {
                    'success': np.array([0.82, 0.62, 0.88]),
                    'time': np.array([9.5, 15.5, 8.5])
                }
                
                # Make predictions
                predictions = await predictor.predict(
                    sample_sequences[:3], sample_epistemic_states[:3]
                )
                
                assert len(predictions) == 3
                assert all(isinstance(pred, PredictionResult) for pred in predictions)
                assert all(0 <= pred.predicted_value <= 1 for pred in predictions)
                assert all(pred.prediction_horizon > 0 for pred in predictions)
    
    @pytest.mark.asyncio
    async def test_online_learning_workflow(self, sample_sequences, sample_epistemic_states,
                                          sample_outcomes, sample_completion_times):
        """Test online learning workflow."""
        predictor = OnlineLearningPredictor()
        
        # Simulate online learning process
        for seq, state, outcome, time in zip(
            sample_sequences[:5], sample_epistemic_states[:5],
            sample_outcomes[:5], sample_completion_times[:5]
        ):
            # Extract features
            features = {
                'sequence_length': len(seq.steps),
                'total_duration': seq.total_duration,
                'success_rate': seq.success_rate,
                'confidence_level': state.confidence_level,
                'uncertainty_score': state.uncertainty_score
            }
            
            # Make prediction
            prediction = await predictor.predict_online(features)
            assert 'success_probability' in prediction
            
            # Learn from result
            await predictor.learn_online(features, outcome, time)
        
        # Check that model has learned
        assert len(predictor.performance_history['accuracy']) > 0 or \
               len(predictor.performance_history['mae']) > 0
    
    @pytest.mark.asyncio
    async def test_model_performance_comparison(self, model_config):
        """Test performance comparison between different models."""
        # Create synthetic data for comparison
        n_samples = 100
        X = np.random.randn(n_samples, 10)
        y_class = np.random.randint(0, 2, n_samples)
        y_reg = np.random.randn(n_samples)
        
        predictor = EnsemblePredictor(model_config)
        
        # Test individual model training
        with patch.object(predictor.rf_classifier, 'fit'), \
             patch.object(predictor.rf_regressor, 'fit'), \
             patch.object(predictor.xgb_classifier, 'fit'), \
             patch.object(predictor.xgb_regressor, 'fit'):
            
            # Mock cross-validation scores
            with patch('sklearn.model_selection.cross_val_score') as mock_cv:
                mock_cv.return_value = np.array([0.8, 0.82, 0.78, 0.85, 0.79])
                
                rf_metrics = await predictor._train_random_forest(X, y_class, y_reg)
                xgb_metrics = await predictor._train_xgboost(X, y_class, y_reg)
                
                assert 'classification_accuracy' in rf_metrics
                assert 'regression_mse' in rf_metrics
                assert 'classification_accuracy' in xgb_metrics
                assert 'regression_mse' in xgb_metrics