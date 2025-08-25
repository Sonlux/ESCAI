"""
Unit tests for model evaluation module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import asyncio
from datetime import datetime
import json
import tempfile
import os

from escai_framework.analytics.model_evaluation import (
    ModelEvaluator, ModelPerformance, EvaluationResult
)
from escai_framework.analytics.prediction_models import EnsemblePredictor, OnlineLearningPredictor, ModelConfig
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState
from escai_framework.models.behavioral_pattern import ExecutionSequence, ExecutionStep
from escai_framework.models.prediction_result import PredictionResult


@pytest.fixture
def model_evaluator():
    """Create model evaluator instance."""
    return ModelEvaluator(cv_folds=3, random_state=42)


@pytest.fixture
def sample_model_performance():
    """Create sample model performance."""
    return ModelPerformance(
        model_name="Test Model",
        task_type="classification",
        accuracy=0.85,
        precision=0.82,
        recall=0.80,
        f1_score=0.81,
        roc_auc=0.88,
        cv_scores=[0.83, 0.85, 0.87],
        cv_mean=0.85,
        cv_std=0.02,
        training_time=10.5,
        prediction_time=0.1
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


class TestModelPerformance:
    """Test ModelPerformance data structure."""
    
    def test_creation(self, sample_model_performance):
        """Test model performance creation."""
        perf = sample_model_performance
        
        assert perf.model_name == "Test Model"
        assert perf.task_type == "classification"
        assert perf.accuracy == 0.85
        assert perf.precision == 0.82
        assert perf.recall == 0.80
        assert perf.f1_score == 0.81
        assert perf.roc_auc == 0.88
        assert perf.cv_mean == 0.85
        assert perf.cv_std == 0.02


class TestEvaluationResult:
    """Test EvaluationResult data structure."""
    
    def test_creation(self, sample_model_performance):
        """Test evaluation result creation."""
        result = EvaluationResult(
            model_performances=[sample_model_performance],
            best_model="Test Model",
            comparison_metrics={"accuracy": {"Test Model": 0.85}},
            validation_curves={},
            learning_curves={},
            feature_importance={},
            recommendations=["Test recommendation"]
        )
        
        assert len(result.model_performances) == 1
        assert result.best_model == "Test Model"
        assert len(result.recommendations) == 1


class TestModelEvaluator:
    """Test model evaluator functionality."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(cv_folds=3, random_state=42)
        
        assert evaluator.cv_folds == 3
        assert evaluator.random_state == 42
        assert evaluator.evaluation_history == []
    
    @pytest.mark.asyncio
    async def test_evaluate_ensemble_model(self, model_evaluator, sample_sequences, 
                                         sample_epistemic_states, sample_outcomes, 
                                         sample_completion_times):
        """Test ensemble model evaluation."""
        # Create mock ensemble predictor
        mock_predictor = Mock(spec=EnsemblePredictor)
        mock_predictor.lstm_model = Mock()
        mock_predictor.rf_classifier = Mock()
        mock_predictor.rf_regressor = Mock()
        mock_predictor.xgb_classifier = Mock()
        mock_predictor.xgb_regressor = Mock()
        
        # Mock feature preparation
        mock_predictor._prepare_features.return_value = (
            np.random.randn(10, 20, 5),  # X_seq
            np.random.randn(10, 8)       # X_static
        )
        
        # Mock predictions
        mock_predictions = [
            PredictionResult(
                agent_id=f"agent_{i}",
                prediction_type="task_outcome",
                predicted_value=0.8,
                confidence_score=0.9,
                prediction_horizon=10.0,
                risk_factors=[],
                model_version="test",
                features_used=[],
                timestamp=datetime.now()
            )
            for i in range(10)
        ]
        mock_predictor.predict.return_value = mock_predictions
        
        with patch.object(model_evaluator, '_evaluate_lstm') as mock_lstm, \
             patch.object(model_evaluator, '_evaluate_sklearn_model') as mock_sklearn, \
             patch.object(model_evaluator, '_evaluate_ensemble_predictions') as mock_ensemble:
            
            # Mock evaluation results
            mock_lstm.return_value = ModelPerformance(
                model_name="LSTM", task_type="both", accuracy=0.8, mse=0.2
            )
            mock_sklearn.return_value = ModelPerformance(
                model_name="RF", task_type="classification", accuracy=0.85
            )
            mock_ensemble.return_value = ModelPerformance(
                model_name="Ensemble", task_type="both", accuracy=0.87, mse=0.15
            )
            
            result = await model_evaluator.evaluate_ensemble_model(
                mock_predictor, sample_sequences, sample_epistemic_states,
                sample_outcomes, sample_completion_times
            )
            
            assert isinstance(result, EvaluationResult)
            assert len(result.model_performances) > 0
            assert result.best_model is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_online_model(self, model_evaluator, sample_sequences, 
                                       sample_epistemic_states, sample_outcomes, 
                                       sample_completion_times):
        """Test online model evaluation."""
        mock_predictor = Mock(spec=OnlineLearningPredictor)
        
        # Mock online predictions
        async def mock_predict_online(features):
            return {
                'success_probability': 0.8,
                'completion_time': 12.0,
                'model_confidence': 0.9
            }
        
        async def mock_learn_online(features, outcome, time):
            pass
        
        mock_predictor.predict_online = mock_predict_online
        mock_predictor.learn_online = mock_learn_online
        
        performance = await model_evaluator.evaluate_online_model(
            mock_predictor, sample_sequences[:5], sample_epistemic_states[:5],
            sample_outcomes[:5], sample_completion_times[:5]
        )
        
        assert isinstance(performance, ModelPerformance)
        assert performance.model_name == "Online Learning Model"
        assert performance.task_type == "both"
        assert performance.accuracy is not None
        assert performance.mse is not None
    
    @pytest.mark.asyncio
    async def test_cross_validate_model(self, model_evaluator):
        """Test cross-validation functionality."""
        # Create mock model
        mock_model = Mock()
        
        # Create sample data
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.82, 0.78, 0.85, 0.79])
            
            result = await model_evaluator.cross_validate_model(
                mock_model, X, y, scoring='accuracy'
            )
            
            assert 'scores' in result
            assert 'mean' in result
            assert 'std' in result
            assert 'scoring' in result
            assert len(result['scores']) == 5
    
    @pytest.mark.asyncio
    async def test_compare_models(self, model_evaluator):
        """Test model comparison."""
        # Create mock models
        models = {
            'model1': Mock(),
            'model2': Mock()
        }
        
        # Create sample data
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        with patch.object(model_evaluator, 'cross_validate_model') as mock_cv:
            mock_cv.return_value = {
                'scores': [0.8, 0.82, 0.78],
                'mean': 0.8,
                'std': 0.02,
                'scoring': 'accuracy'
            }
            
            results = await model_evaluator.compare_models(models, X, y)
            
            assert isinstance(results, dict)
            assert 'model1' in results
            assert 'model2' in results
            assert mock_cv.call_count == 2
    
    @pytest.mark.asyncio
    async def test_hyperparameter_tuning(self, model_evaluator):
        """Test hyperparameter tuning."""
        mock_model = Mock()
        param_grid = {'n_estimators': [10, 50, 100]}
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        with patch('sklearn.model_selection.GridSearchCV') as mock_grid:
            mock_grid_instance = Mock()
            mock_grid_instance.fit.return_value = None
            mock_grid_instance.best_params_ = {'n_estimators': 50}
            mock_grid_instance.best_score_ = 0.85
            mock_grid_instance.best_estimator_ = mock_model
            mock_grid_instance.cv_results_ = {'mean_test_score': [0.8, 0.85, 0.82]}
            mock_grid.return_value = mock_grid_instance
            
            result = await model_evaluator.hyperparameter_tuning(
                mock_model, param_grid, X, y
            )
            
            assert 'best_params' in result
            assert 'best_score' in result
            assert 'best_estimator' in result
            assert 'cv_results' in result
    
    def test_determine_best_model(self, model_evaluator):
        """Test best model determination."""
        performances = [
            ModelPerformance(
                model_name="Model1", task_type="classification",
                accuracy=0.8, f1_score=0.78, roc_auc=0.82
            ),
            ModelPerformance(
                model_name="Model2", task_type="classification",
                accuracy=0.85, f1_score=0.83, roc_auc=0.87
            ),
            ModelPerformance(
                model_name="Model3", task_type="regression",
                r2=0.75, mse=0.2
            )
        ]
        
        best_model = model_evaluator._determine_best_model(performances)
        
        assert best_model == "Model2"  # Should have highest combined score
    
    def test_generate_comparison_metrics(self, model_evaluator):
        """Test comparison metrics generation."""
        performances = [
            ModelPerformance(
                model_name="Model1", task_type="classification",
                accuracy=0.8, precision=0.78, recall=0.82, f1_score=0.8,
                prediction_time=0.1
            ),
            ModelPerformance(
                model_name="Model2", task_type="regression",
                mse=0.2, mae=0.15, r2=0.75, rmse=0.45,
                prediction_time=0.2
            )
        ]
        
        comparison = model_evaluator._generate_comparison_metrics(performances)
        
        assert 'classification_metrics' in comparison
        assert 'regression_metrics' in comparison
        assert 'efficiency_metrics' in comparison
        assert 'Model1' in comparison['classification_metrics']
        assert 'Model2' in comparison['regression_metrics']
    
    def test_generate_recommendations(self, model_evaluator):
        """Test recommendation generation."""
        performances = [
            ModelPerformance(
                model_name="Ensemble", task_type="both",
                accuracy=0.75, r2=0.5  # Lower performance
            ),
            ModelPerformance(
                model_name="SlowModel", task_type="classification",
                accuracy=0.9, prediction_time=2.0  # Slow model
            )
        ]
        
        comparison_metrics = {}
        
        recommendations = model_evaluator._generate_recommendations(
            performances, comparison_metrics
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend improvements for low performance and slow models
        rec_text = ' '.join(recommendations).lower()
        assert any(keyword in rec_text for keyword in 
                  ['accuracy', 'data', 'optimization', 'performance'])
    
    def test_save_and_load_evaluation_results(self, model_evaluator, sample_model_performance):
        """Test saving and loading evaluation results."""
        result = EvaluationResult(
            model_performances=[sample_model_performance],
            best_model="Test Model",
            comparison_metrics={"test": "data"},
            validation_curves={},
            learning_curves={},
            feature_importance={},
            recommendations=["Test recommendation"]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Save results
            model_evaluator.save_evaluation_results(result, filepath)
            
            # Check file exists
            assert os.path.exists(filepath)
            
            # Load results
            loaded_result = model_evaluator.load_evaluation_results(filepath)
            
            assert isinstance(loaded_result, dict)
            assert 'model_performances' in loaded_result
            assert 'best_model' in loaded_result
            assert loaded_result['best_model'] == "Test Model"
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)


@pytest.mark.integration
class TestModelEvaluationIntegration:
    """Integration tests for model evaluation."""
    
    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(self, sample_sequences, sample_epistemic_states,
                                              sample_outcomes, sample_completion_times):
        """Test complete evaluation workflow."""
        evaluator = ModelEvaluator(cv_folds=3)
        
        # Create mock ensemble predictor with all required methods
        mock_predictor = Mock(spec=EnsemblePredictor)
        mock_predictor.lstm_model = Mock()
        mock_predictor.rf_classifier = Mock()
        mock_predictor.rf_regressor = Mock()
        mock_predictor.xgb_classifier = Mock()
        mock_predictor.xgb_regressor = Mock()
        
        # Mock all required methods
        mock_predictor._prepare_features.return_value = (
            np.random.randn(10, 20, 5),
            np.random.randn(10, 8)
        )
        
        mock_predictions = [
            PredictionResult(
                agent_id=f"agent_{i}",
                prediction_type="task_outcome",
                predicted_value=0.8,
                confidence_score=0.9,
                prediction_horizon=10.0,
                risk_factors=[],
                model_version="test",
                features_used=[],
                timestamp=datetime.now()
            )
            for i in range(10)
        ]
        mock_predictor.predict.return_value = mock_predictions
        
        # Mock individual model evaluations
        with patch.object(evaluator, '_evaluate_lstm'), \
             patch.object(evaluator, '_evaluate_sklearn_model'), \
             patch.object(evaluator, '_evaluate_ensemble_predictions'), \
             patch.object(evaluator, '_generate_validation_curves'), \
             patch.object(evaluator, '_generate_learning_curves'), \
             patch.object(evaluator, '_extract_feature_importance'):
            
            result = await evaluator.evaluate_ensemble_model(
                mock_predictor, sample_sequences, sample_epistemic_states,
                sample_outcomes, sample_completion_times
            )
            
            assert isinstance(result, EvaluationResult)
            assert len(evaluator.evaluation_history) == 1
    
    @pytest.mark.asyncio
    async def test_model_comparison_workflow(self):
        """Test model comparison workflow."""
        evaluator = ModelEvaluator(cv_folds=3)
        
        # Create sample data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Create mock models
        models = {
            'RandomForest': Mock(),
            'XGBoost': Mock(),
            'LogisticRegression': Mock()
        }
        
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            # Mock different performance for each model
            def side_effect(*args, **kwargs):
                model = args[0]
                if hasattr(model, '_mock_name'):
                    if 'RandomForest' in str(model._mock_name):
                        return np.array([0.85, 0.87, 0.83])
                    elif 'XGBoost' in str(model._mock_name):
                        return np.array([0.88, 0.90, 0.86])
                    else:
                        return np.array([0.80, 0.82, 0.78])
                return np.array([0.80, 0.82, 0.78])
            
            mock_cv.side_effect = side_effect
            
            results = await evaluator.compare_models(models, X, y)
            
            assert isinstance(results, dict)
            assert len(results) == 3
            
            # All models should have been evaluated
            for model_name in models.keys():
                assert model_name in results
                assert 'mean' in results[model_name]
                assert 'std' in results[model_name]
    
    @pytest.mark.asyncio
    async def test_hyperparameter_optimization_workflow(self):
        """Test hyperparameter optimization workflow."""
        evaluator = ModelEvaluator()
        
        # Create sample data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Create mock model
        mock_model = Mock()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        }
        
        with patch('sklearn.model_selection.GridSearchCV') as mock_grid:
            mock_grid_instance = Mock()
            mock_grid_instance.fit.return_value = None
            mock_grid_instance.best_params_ = {'n_estimators': 100, 'max_depth': 10}
            mock_grid_instance.best_score_ = 0.88
            mock_grid_instance.best_estimator_ = mock_model
            mock_grid_instance.cv_results_ = {
                'mean_test_score': [0.85, 0.88, 0.86, 0.87, 0.85, 0.84, 0.86, 0.88, 0.87]
            }
            mock_grid.return_value = mock_grid_instance
            
            result = await evaluator.hyperparameter_tuning(
                mock_model, param_grid, X, y, scoring='accuracy'
            )
            
            assert result['best_params'] == {'n_estimators': 100, 'max_depth': 10}
            assert result['best_score'] == 0.88
            assert 'cv_results' in result