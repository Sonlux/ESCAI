"""
Model training and evaluation pipelines with cross-validation.

This module provides comprehensive model evaluation capabilities including
cross-validation, performance metrics, model comparison, and evaluation pipelines.
"""

import asyncio
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, TimeSeriesSplit,
    GridSearchCV, RandomizedSearchCV, validation_curve, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .prediction_models import EnsemblePredictor, OnlineLearningPredictor
from ..models.epistemic_state import EpistemicState
from ..models.behavioral_pattern import ExecutionSequence
from ..models.prediction_result import PredictionResult


@dataclass
class ModelPerformance:
    """Represents model performance metrics."""
    model_name: str
    task_type: str  # 'classification' or 'regression'
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    
    # Regression metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    rmse: Optional[float] = None
    
    # Cross-validation metrics
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Additional metrics
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    model_size: Optional[int] = None


@dataclass
class EvaluationResult:
    """Complete model evaluation result."""
    model_performances: List[ModelPerformance]
    best_model: str
    comparison_metrics: Dict[str, Any]
    validation_curves: Dict[str, Any]
    learning_curves: Dict[str, Any]
    feature_importance: Dict[str, List[Tuple[str, float]]]
    recommendations: List[str]


class ModelEvaluator:
    """
    Comprehensive model evaluation system.
    
    Provides cross-validation, performance metrics calculation,
    model comparison, and evaluation reporting.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.evaluation_history: List[EvaluationResult] = []
        
    async def evaluate_ensemble_model(self, model: EnsemblePredictor,
                                    sequences: List[ExecutionSequence],
                                    epistemic_states: List[EpistemicState],
                                    outcomes: List[bool],
                                    completion_times: List[float]) -> EvaluationResult:
        """
        Evaluate ensemble model performance with comprehensive metrics.
        
        Args:
            model: Trained ensemble predictor
            sequences: Execution sequences
            epistemic_states: Epistemic states
            outcomes: Task outcomes
            completion_times: Task completion times
            
        Returns:
            Complete evaluation result
        """
        # Prepare features
        X_seq, X_static = model._prepare_features(sequences, epistemic_states)
        y_class = np.array(outcomes, dtype=int)
        y_reg = np.array(completion_times, dtype=float)
        
        # Evaluate individual models
        model_performances = []
        
        # Evaluate LSTM
        if model.lstm_model is not None:
            lstm_perf = await self._evaluate_lstm(model, X_seq, y_class, y_reg)
            model_performances.append(lstm_perf)
        
        # Evaluate Random Forest
        rf_class_perf = await self._evaluate_sklearn_model(
            model.rf_classifier, X_static, y_class, "Random Forest (Classification)"
        )
        rf_reg_perf = await self._evaluate_sklearn_model(
            model.rf_regressor, X_static, y_reg, "Random Forest (Regression)", task_type="regression"
        )
        model_performances.extend([rf_class_perf, rf_reg_perf])
        
        # Evaluate XGBoost
        xgb_class_perf = await self._evaluate_sklearn_model(
            model.xgb_classifier, X_static, y_class, "XGBoost (Classification)"
        )
        xgb_reg_perf = await self._evaluate_sklearn_model(
            model.xgb_regressor, X_static, y_reg, "XGBoost (Regression)", task_type="regression"
        )
        model_performances.extend([xgb_class_perf, xgb_reg_perf])
        
        # Evaluate ensemble predictions
        ensemble_predictions = await model.predict(sequences, epistemic_states)
        ensemble_perf = await self._evaluate_ensemble_predictions(
            ensemble_predictions, outcomes, completion_times
        )
        model_performances.append(ensemble_perf)
        
        # Determine best model
        best_model = self._determine_best_model(model_performances)
        
        # Generate comparison metrics
        comparison_metrics = self._generate_comparison_metrics(model_performances)
        
        # Generate validation curves
        validation_curves = await self._generate_validation_curves(model, X_static, y_class)
        
        # Generate learning curves
        learning_curves = await self._generate_learning_curves(model, X_static, y_class)
        
        # Extract feature importance
        feature_importance = self._extract_feature_importance(model)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(model_performances, comparison_metrics)
        
        result = EvaluationResult(
            model_performances=model_performances,
            best_model=best_model,
            comparison_metrics=comparison_metrics,
            validation_curves=validation_curves,
            learning_curves=learning_curves,
            feature_importance=feature_importance,
            recommendations=recommendations
        )
        
        self.evaluation_history.append(result)
        return result
    
    async def evaluate_online_model(self, model: OnlineLearningPredictor,
                                  test_sequences: List[ExecutionSequence],
                                  test_states: List[EpistemicState],
                                  test_outcomes: List[bool],
                                  test_times: List[float]) -> ModelPerformance:
        """
        Evaluate online learning model performance.
        
        Args:
            model: Online learning predictor
            test_sequences: Test execution sequences
            test_states: Test epistemic states
            test_outcomes: Test outcomes
            test_times: Test completion times
            
        Returns:
            Model performance metrics
        """
        predictions = []
        actual_outcomes = []
        actual_times = []
        
        start_time = datetime.now()
        
        for seq, state, outcome, time in zip(test_sequences, test_states, test_outcomes, test_times):
            # Extract features
            features = self._extract_online_features(seq, state)
            
            # Make prediction
            pred = await model.predict_online(features)
            predictions.append(pred)
            
            actual_outcomes.append(outcome)
            actual_times.append(time)
            
            # Learn from this example (online learning)
            await model.learn_online(features, outcome, time)
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate metrics
        pred_outcomes = [p['success_probability'] > 0.5 for p in predictions]
        pred_times = [p['completion_time'] for p in predictions]
        
        accuracy = accuracy_score(actual_outcomes, pred_outcomes)
        precision = precision_score(actual_outcomes, pred_outcomes, zero_division=0)
        recall = recall_score(actual_outcomes, pred_outcomes, zero_division=0)
        f1 = f1_score(actual_outcomes, pred_outcomes, zero_division=0)
        
        mse = mean_squared_error(actual_times, pred_times)
        mae = mean_absolute_error(actual_times, pred_times)
        r2 = r2_score(actual_times, pred_times)
        
        return ModelPerformance(
            model_name="Online Learning Model",
            task_type="both",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mse=mse,
            mae=mae,
            r2=r2,
            rmse=np.sqrt(mse),
            prediction_time=prediction_time
        )
    
    async def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                                 scoring: Union[str, List[str]] = 'accuracy',
                                 cv_type: str = 'stratified') -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric(s)
            cv_type: Cross-validation type ('stratified', 'time_series')
            
        Returns:
            Cross-validation results
        """
        # Choose cross-validation strategy
        if cv_type == 'stratified':
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        elif cv_type == 'time_series':
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            cv = self.cv_folds
        
        # Perform cross-validation
        if isinstance(scoring, str):
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std(),
                'scoring': scoring
            }
        else:
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
            results = {}
            for metric in scoring:
                test_scores = scores[f'test_{metric}']
                results[metric] = {
                    'scores': test_scores.tolist(),
                    'mean': test_scores.mean(),
                    'std': test_scores.std()
                }
            return results
    
    async def compare_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                           scoring: str = 'accuracy') -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models using cross-validation.
        
        Args:
            models: Dictionary of model name to model object
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name, model in models.items():
            cv_results = await self.cross_validate_model(model, X, y, scoring)
            results[name] = cv_results
        
        return results
    
    async def hyperparameter_tuning(self, model: Any, param_grid: Dict[str, List],
                                  X: np.ndarray, y: np.ndarray,
                                  scoring: str = 'accuracy',
                                  search_type: str = 'grid') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            model: Model to tune
            param_grid: Parameter grid
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric
            search_type: 'grid' or 'random'
            
        Returns:
            Tuning results
        """
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=self.cv_folds, scoring=scoring,
                n_jobs=-1, random_state=self.random_state
            )
        else:
            search = RandomizedSearchCV(
                model, param_grid, cv=self.cv_folds, scoring=scoring,
                n_jobs=-1, random_state=self.random_state, n_iter=50
            )
        
        search.fit(X, y)
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': search.cv_results_
        }
    
    # Private helper methods
    
    async def _evaluate_lstm(self, model: EnsemblePredictor, X_seq: np.ndarray,
                           y_class: np.ndarray, y_reg: np.ndarray) -> ModelPerformance:
        """Evaluate LSTM model performance."""
        import torch
        
        start_time = datetime.now()
        
        # Make predictions
        if model.lstm_model is not None:
            model.lstm_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_seq)
                outputs = model.lstm_model(X_tensor)
                predictions = outputs.numpy()
        else:
            # Fallback if LSTM model is not available
            predictions = np.zeros((X_seq.shape[0], 2))
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Classification metrics
        pred_class = (predictions[:, 0] > 0.5).astype(int)
        accuracy = accuracy_score(y_class, pred_class)
        precision = precision_score(y_class, pred_class, zero_division=0)
        recall = recall_score(y_class, pred_class, zero_division=0)
        f1 = f1_score(y_class, pred_class, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_class, predictions[:, 0])
        except:
            roc_auc = None
        
        # Regression metrics
        pred_reg = predictions[:, 1]
        mse = mean_squared_error(y_reg, pred_reg)
        mae = mean_absolute_error(y_reg, pred_reg)
        r2 = r2_score(y_reg, pred_reg)
        
        return ModelPerformance(
            model_name="LSTM",
            task_type="both",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            mse=mse,
            mae=mae,
            r2=r2,
            rmse=np.sqrt(mse),
            prediction_time=prediction_time
        )
    
    async def _evaluate_sklearn_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                                    model_name: str, task_type: str = "classification") -> ModelPerformance:
        """Evaluate scikit-learn model performance."""
        start_time = datetime.now()
        
        if task_type == "classification":
            # Cross-validation for classification
            cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='accuracy')
            
            # Make predictions for detailed metrics
            predictions = model.predict(X)
            pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions, zero_division=0)
            recall = recall_score(y, predictions, zero_division=0)
            f1 = f1_score(y, predictions, zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y, pred_proba) if pred_proba is not None else None
            except:
                roc_auc = None
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            return ModelPerformance(
                model_name=model_name,
                task_type=task_type,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                cv_scores=cv_scores.tolist(),
                cv_mean=cv_scores.mean(),
                cv_std=cv_scores.std(),
                prediction_time=prediction_time
            )
        
        else:  # regression
            # Cross-validation for regression
            cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='neg_mean_squared_error')
            
            # Make predictions for detailed metrics
            predictions = model.predict(X)
            
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            return ModelPerformance(
                model_name=model_name,
                task_type=task_type,
                mse=mse,
                mae=mae,
                r2=r2,
                rmse=np.sqrt(mse),
                cv_scores=(-cv_scores).tolist(),  # Convert back to positive MSE
                cv_mean=(-cv_scores).mean(),
                cv_std=cv_scores.std(),
                prediction_time=prediction_time
            )
    
    async def _evaluate_ensemble_predictions(self, predictions: List[PredictionResult],
                                           actual_outcomes: List[bool],
                                           actual_times: List[float]) -> ModelPerformance:
        """Evaluate ensemble predictions."""
        start_time = datetime.now()
        
        # Extract predictions
        pred_outcomes = [pred.predicted_value > 0.5 for pred in predictions]
        pred_times = [pred.predicted_value for pred in predictions]  # Use predicted_value for time
        pred_probs = [pred.predicted_value for pred in predictions]
        
        # Classification metrics
        accuracy = accuracy_score(actual_outcomes, pred_outcomes)
        precision = precision_score(actual_outcomes, pred_outcomes, zero_division=0)
        recall = recall_score(actual_outcomes, pred_outcomes, zero_division=0)
        f1 = f1_score(actual_outcomes, pred_outcomes, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(actual_outcomes, pred_probs)
        except:
            roc_auc = None
        
        # Regression metrics
        mse = mean_squared_error(actual_times, pred_times)
        mae = mean_absolute_error(actual_times, pred_times)
        r2 = r2_score(actual_times, pred_times)
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        return ModelPerformance(
            model_name="Ensemble",
            task_type="both",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            mse=mse,
            mae=mae,
            r2=r2,
            rmse=np.sqrt(mse),
            prediction_time=prediction_time
        )
    
    def _extract_online_features(self, sequence: ExecutionSequence, 
                               state: EpistemicState) -> Dict[str, float]:
        """Extract features for online learning model."""
        return {
            'sequence_length': len(sequence.steps),
            'total_duration': float(sequence.total_duration_ms),
            'success_rate': sequence.success_rate,
            'confidence_level': state.confidence_level,
            'uncertainty_score': state.uncertainty_score,
            'num_beliefs': len(state.belief_states),
            'num_goals': len(state.goal_states[0].primary_goals) if state.goal_states else 0,
            'avg_step_duration': float(np.mean([step.duration for step in sequence.steps])) if sequence.steps else 0.0,
            'error_count': sum(1 for step in sequence.steps if step.error_message)
        }
    
    def _determine_best_model(self, performances: List[ModelPerformance]) -> str:
        """Determine the best performing model."""
        # Score models based on multiple criteria
        scores = {}
        
        for perf in performances:
            score: float = 0.0
            
            # Classification performance
            if perf.accuracy is not None:
                score += perf.accuracy * 0.3
            if perf.f1_score is not None:
                score += perf.f1_score * 0.3
            if perf.roc_auc is not None:
                score += perf.roc_auc * 0.2
            
            # Regression performance
            if perf.r2 is not None:
                score += max(0, perf.r2) * 0.2  # R² can be negative
            
            scores[perf.model_name] = score
        
        return max(scores, key=scores.get) if scores else "Unknown"
    
    def _generate_comparison_metrics(self, performances: List[ModelPerformance]) -> Dict[str, Any]:
        """Generate comparison metrics across models."""
        comparison: Dict[str, Any] = {
            'classification_metrics': {},
            'regression_metrics': {},
            'efficiency_metrics': {}
        }
        
        for perf in performances:
            name = perf.model_name
            
            # Classification metrics
            if perf.accuracy is not None:
                comparison['classification_metrics'][name] = {
                    'accuracy': perf.accuracy,
                    'precision': perf.precision,
                    'recall': perf.recall,
                    'f1_score': perf.f1_score,
                    'roc_auc': perf.roc_auc
                }
            
            # Regression metrics
            if perf.mse is not None:
                comparison['regression_metrics'][name] = {
                    'mse': perf.mse,
                    'mae': perf.mae,
                    'r2': perf.r2,
                    'rmse': perf.rmse
                }
            
            # Efficiency metrics
            comparison['efficiency_metrics'][name] = {
                'prediction_time': perf.prediction_time,
                'training_time': perf.training_time
            }
        
        return comparison
    
    async def _generate_validation_curves(self, model: EnsemblePredictor, 
                                        X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Generate validation curves for hyperparameters."""
        curves: Dict[str, Any] = {}
        
        # Random Forest validation curve
        try:
            param_range = [10, 50, 100, 200, 500]
            train_scores, val_scores = validation_curve(
                model.rf_classifier, X, y, param_name='n_estimators',
                param_range=param_range, cv=3, scoring='accuracy'
            )
            
            curves['random_forest_n_estimators'] = {
                'param_range': param_range,
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
        except Exception as e:
            curves['random_forest_error'] = str(e)
        
        return curves
    
    async def _generate_learning_curves(self, model: EnsemblePredictor,
                                      X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Generate learning curves."""
        curves: Dict[str, Any] = {}
        
        try:
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model.rf_classifier, X, y, train_sizes=train_sizes,
                cv=3, scoring='accuracy'
            )
            
            curves['random_forest_learning'] = {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
        except Exception as e:
            curves['learning_curve_error'] = str(e)
        
        return curves
    
    def _extract_feature_importance(self, model: EnsemblePredictor) -> Dict[str, List[Tuple[str, float]]]:
        """Extract feature importance from models."""
        importance = {}
        
        # Random Forest feature importance
        if hasattr(model.rf_classifier, 'feature_importances_'):
            rf_importance = model.rf_classifier.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(rf_importance))]
            importance['random_forest'] = list(zip(feature_names, rf_importance.tolist()))
        
        # XGBoost feature importance
        if hasattr(model.xgb_classifier, 'feature_importances_'):
            xgb_importance = model.xgb_classifier.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(xgb_importance))]
            importance['xgboost'] = list(zip(feature_names, xgb_importance.tolist()))
        
        return importance
    
    def _generate_recommendations(self, performances: List[ModelPerformance],
                                comparison_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Find best performing models
        best_classification = max(
            [p for p in performances if p.accuracy is not None],
            key=lambda x: x.accuracy or 0,
            default=None
        )
        
        best_regression = max(
            [p for p in performances if p.r2 is not None],
            key=lambda x: x.r2 or -float('inf'),
            default=None
        )
        
        if best_classification:
            recommendations.append(f"Best classification model: {best_classification.model_name} (accuracy: {best_classification.accuracy:.3f})")
        
        if best_regression:
            recommendations.append(f"Best regression model: {best_regression.model_name} (R²: {best_regression.r2:.3f})")
        
        # Performance-based recommendations
        ensemble_perf = next((p for p in performances if p.model_name == "Ensemble"), None)
        if ensemble_perf:
            if ensemble_perf.accuracy and ensemble_perf.accuracy < 0.8:
                recommendations.append("Consider collecting more training data to improve ensemble accuracy")
            
            if ensemble_perf.r2 and ensemble_perf.r2 < 0.6:
                recommendations.append("Consider feature engineering to improve regression performance")
        
        # Efficiency recommendations
        slow_models = [p for p in performances if p.prediction_time and p.prediction_time > 1.0]
        if slow_models:
            recommendations.append("Consider model optimization for faster predictions")
        
        return recommendations
    
    def save_evaluation_results(self, result: EvaluationResult, filepath: str):
        """Save evaluation results to file."""
        # Convert to serializable format
        serializable_result = {
            'model_performances': [
                {
                    'model_name': p.model_name,
                    'task_type': p.task_type,
                    'accuracy': p.accuracy,
                    'precision': p.precision,
                    'recall': p.recall,
                    'f1_score': p.f1_score,
                    'roc_auc': p.roc_auc,
                    'mse': p.mse,
                    'mae': p.mae,
                    'r2': p.r2,
                    'rmse': p.rmse,
                    'cv_scores': p.cv_scores,
                    'cv_mean': p.cv_mean,
                    'cv_std': p.cv_std,
                    'training_time': p.training_time,
                    'prediction_time': p.prediction_time
                }
                for p in result.model_performances
            ],
            'best_model': result.best_model,
            'comparison_metrics': result.comparison_metrics,
            'validation_curves': result.validation_curves,
            'learning_curves': result.learning_curves,
            'feature_importance': result.feature_importance,
            'recommendations': result.recommendations,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2)
    
    def load_evaluation_results(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation results from file."""
        with open(filepath, 'r') as f:
            return json.load(f)