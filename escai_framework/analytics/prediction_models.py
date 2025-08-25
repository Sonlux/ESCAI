"""
Machine learning models for performance prediction and behavioral analysis.

This module implements LSTM, Random Forest, and XGBoost models with
hyperparameter tuning and online learning capabilities.
"""

import asyncio
import pickle
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import xgboost as xgb
from river import ensemble, tree, metrics, preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from ..models.epistemic_state import EpistemicState
from ..models.behavioral_pattern import ExecutionSequence
from ..models.prediction_result import PredictionResult


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    # LSTM Configuration
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    
    # Random Forest Configuration
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    
    # XGBoost Configuration
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    
    # General Configuration
    sequence_length: int = 50
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42


class SequenceDataset(Dataset):
    """PyTorch dataset for sequence data."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMPredictor(nn.Module):
    """
    LSTM neural network for sequence prediction.
    
    Predicts task success probability and completion time based on
    agent execution sequences and epistemic states.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        # Apply sigmoid for probability outputs
        if output.size(1) == 1:  # Binary classification
            output = self.sigmoid(output)
        
        return output


class EnsemblePredictor:
    """
    Ensemble predictor combining LSTM, Random Forest, and XGBoost models.
    
    Uses weighted voting for final predictions with confidence estimation.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        
        # Initialize models
        self.lstm_model = None
        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=self.config.random_state
        )
        self.rf_regressor = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=self.config.random_state
        )
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            random_state=self.config.random_state
        )
        self.xgb_regressor = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            random_state=self.config.random_state
        )
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model weights (learned during training)
        self.model_weights = {
            'lstm': 0.4,
            'rf': 0.3,
            'xgb': 0.3
        }
        
        # Training history
        self.training_history = {
            'lstm_losses': [],
            'rf_scores': [],
            'xgb_scores': []
        }
    
    async def train(self, sequences: List[ExecutionSequence], 
                   epistemic_states: List[EpistemicState],
                   outcomes: List[bool], completion_times: List[float]) -> Dict[str, float]:
        """
        Train all models in the ensemble.
        
        Args:
            sequences: Agent execution sequences
            epistemic_states: Corresponding epistemic states
            outcomes: Task success outcomes (True/False)
            completion_times: Task completion times
            
        Returns:
            Training metrics for all models
        """
        # Prepare features
        X_seq, X_static = self._prepare_features(sequences, epistemic_states)
        y_class = np.array(outcomes, dtype=int)
        y_reg = np.array(completion_times, dtype=float)
        
        # Train models concurrently
        lstm_metrics, rf_metrics, xgb_metrics = await asyncio.gather(
            self._train_lstm(X_seq, y_class, y_reg),
            self._train_random_forest(X_static, y_class, y_reg),
            self._train_xgboost(X_static, y_class, y_reg)
        )
        
        # Update model weights based on performance
        self._update_model_weights(lstm_metrics, rf_metrics, xgb_metrics)
        
        return {
            'lstm': lstm_metrics,
            'random_forest': rf_metrics,
            'xgboost': xgb_metrics,
            'ensemble_weights': self.model_weights
        }
    
    async def predict(self, sequences: List[ExecutionSequence],
                     epistemic_states: List[EpistemicState]) -> List[PredictionResult]:
        """
        Make ensemble predictions for given sequences and states.
        
        Args:
            sequences: Agent execution sequences
            epistemic_states: Corresponding epistemic states
            
        Returns:
            List of prediction results with confidence scores
        """
        # Prepare features
        X_seq, X_static = self._prepare_features(sequences, epistemic_states)
        
        # Get predictions from all models
        lstm_preds = await self._predict_lstm(X_seq)
        rf_preds = await self._predict_random_forest(X_static)
        xgb_preds = await self._predict_xgboost(X_static)
        
        # Combine predictions
        results = []
        for i in range(len(sequences)):
            # Weighted ensemble for success probability
            success_prob = (
                self.model_weights['lstm'] * lstm_preds['success'][i] +
                self.model_weights['rf'] * rf_preds['success'][i] +
                self.model_weights['xgb'] * xgb_preds['success'][i]
            )
            
            # Weighted ensemble for completion time
            completion_time = (
                self.model_weights['lstm'] * lstm_preds['time'][i] +
                self.model_weights['rf'] * rf_preds['time'][i] +
                self.model_weights['xgb'] * xgb_preds['time'][i]
            )
            
            # Calculate confidence based on model agreement
            confidence = self._calculate_confidence(
                [lstm_preds['success'][i], rf_preds['success'][i], xgb_preds['success'][i]]
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(sequences[i], epistemic_states[i])
            
            result = PredictionResult(
                agent_id=epistemic_states[i].agent_id,
                prediction_type="task_outcome",
                predicted_value=success_prob,
                confidence_score=confidence,
                prediction_horizon=completion_time,
                risk_factors=risk_factors,
                model_version="ensemble_v1.0",
                features_used=list(range(X_static.shape[1])),
                timestamp=epistemic_states[i].timestamp
            )
            results.append(result)
        
        return results
    
    def _prepare_features(self, sequences: List[ExecutionSequence], 
                         epistemic_states: List[EpistemicState]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training and prediction."""
        # Sequence features for LSTM
        seq_features = []
        static_features = []
        
        for seq, state in zip(sequences, epistemic_states):
            # Sequence features (time series)
            seq_feat = self._extract_sequence_features(seq)
            seq_features.append(seq_feat)
            
            # Static features
            static_feat = self._extract_static_features(seq, state)
            static_features.append(static_feat)
        
        # Pad sequences to same length
        max_len = max(len(sf) for sf in seq_features)
        padded_sequences = []
        for sf in seq_features:
            if len(sf) < max_len:
                padding = np.zeros((max_len - len(sf), sf.shape[1]))
                sf = np.vstack([sf, padding])
            padded_sequences.append(sf)
        
        X_seq = np.array(padded_sequences)
        X_static = np.array(static_features)
        
        # Scale static features
        X_static = self.scaler.fit_transform(X_static)
        
        return X_seq, X_static
    
    def _extract_sequence_features(self, sequence: ExecutionSequence) -> np.ndarray:
        """Extract time-series features from execution sequence."""
        features = []
        
        for step in sequence.steps:
            step_features = [
                step.duration,
                step.success_probability,
                len(step.context.get('inputs', [])),
                len(step.context.get('outputs', [])),
                1.0 if step.error_message else 0.0
            ]
            features.append(step_features)
        
        return np.array(features) if features else np.array([[0, 0, 0, 0, 0]])
    
    def _extract_static_features(self, sequence: ExecutionSequence, 
                                state: EpistemicState) -> List[float]:
        """Extract static features from sequence and epistemic state."""
        features = [
            len(sequence.steps),
            sequence.total_duration,
            sequence.success_rate,
            state.confidence_level,
            state.uncertainty_score,
            len(state.belief_states),
            len(state.goal_state.active_goals) if state.goal_state else 0,
            len(state.knowledge_state.facts) if state.knowledge_state else 0
        ]
        
        return features
    
    async def _train_lstm(self, X_seq: np.ndarray, y_class: np.ndarray, 
                         y_reg: np.ndarray) -> Dict[str, float]:
        """Train LSTM model."""
        # Initialize model
        input_size = X_seq.shape[2]
        self.lstm_model = LSTMPredictor(
            input_size=input_size,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            output_size=2,  # success probability + completion time
            dropout=self.config.lstm_dropout
        )
        
        # Prepare targets (combine classification and regression)
        y_combined = np.column_stack([y_class, y_reg])
        
        # Create dataset and dataloader
        dataset = SequenceDataset(X_seq, y_combined)
        dataloader = DataLoader(dataset, batch_size=self.config.lstm_batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.config.lstm_learning_rate)
        
        # Training loop
        losses = []
        for epoch in range(self.config.lstm_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"LSTM Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        self.training_history['lstm_losses'] = losses
        
        return {
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'epochs_trained': len(losses)
        }
    
    async def _train_random_forest(self, X_static: np.ndarray, y_class: np.ndarray, 
                                  y_reg: np.ndarray) -> Dict[str, float]:
        """Train Random Forest models."""
        # Train classifier
        rf_class_scores = cross_val_score(
            self.rf_classifier, X_static, y_class, 
            cv=self.config.cv_folds, scoring='accuracy'
        )
        self.rf_classifier.fit(X_static, y_class)
        
        # Train regressor
        rf_reg_scores = cross_val_score(
            self.rf_regressor, X_static, y_reg, 
            cv=self.config.cv_folds, scoring='neg_mean_squared_error'
        )
        self.rf_regressor.fit(X_static, y_reg)
        
        self.training_history['rf_scores'] = rf_class_scores.tolist()
        
        return {
            'classification_accuracy': rf_class_scores.mean(),
            'classification_std': rf_class_scores.std(),
            'regression_mse': -rf_reg_scores.mean(),
            'regression_std': rf_reg_scores.std()
        }
    
    async def _train_xgboost(self, X_static: np.ndarray, y_class: np.ndarray, 
                            y_reg: np.ndarray) -> Dict[str, float]:
        """Train XGBoost models."""
        # Train classifier
        xgb_class_scores = cross_val_score(
            self.xgb_classifier, X_static, y_class, 
            cv=self.config.cv_folds, scoring='accuracy'
        )
        self.xgb_classifier.fit(X_static, y_class)
        
        # Train regressor
        xgb_reg_scores = cross_val_score(
            self.xgb_regressor, X_static, y_reg, 
            cv=self.config.cv_folds, scoring='neg_mean_squared_error'
        )
        self.xgb_regressor.fit(X_static, y_reg)
        
        self.training_history['xgb_scores'] = xgb_class_scores.tolist()
        
        return {
            'classification_accuracy': xgb_class_scores.mean(),
            'classification_std': xgb_class_scores.std(),
            'regression_mse': -xgb_reg_scores.mean(),
            'regression_std': xgb_reg_scores.std()
        }
    
    async def _predict_lstm(self, X_seq: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions using LSTM model."""
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained")
        
        self.lstm_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq)
            outputs = self.lstm_model(X_tensor)
            predictions = outputs.numpy()
        
        return {
            'success': predictions[:, 0],
            'time': predictions[:, 1]
        }
    
    async def _predict_random_forest(self, X_static: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions using Random Forest models."""
        success_probs = self.rf_classifier.predict_proba(X_static)[:, 1]
        completion_times = self.rf_regressor.predict(X_static)
        
        return {
            'success': success_probs,
            'time': completion_times
        }
    
    async def _predict_xgboost(self, X_static: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions using XGBoost models."""
        success_probs = self.xgb_classifier.predict_proba(X_static)[:, 1]
        completion_times = self.xgb_regressor.predict(X_static)
        
        return {
            'success': success_probs,
            'time': completion_times
        }
    
    def _update_model_weights(self, lstm_metrics: Dict, rf_metrics: Dict, xgb_metrics: Dict):
        """Update ensemble weights based on model performance."""
        # Simple performance-based weighting
        lstm_score = 1.0 - lstm_metrics.get('final_loss', 1.0)
        rf_score = rf_metrics.get('classification_accuracy', 0.0)
        xgb_score = xgb_metrics.get('classification_accuracy', 0.0)
        
        total_score = lstm_score + rf_score + xgb_score
        
        if total_score > 0:
            self.model_weights = {
                'lstm': lstm_score / total_score,
                'rf': rf_score / total_score,
                'xgb': xgb_score / total_score
            }
    
    def _calculate_confidence(self, predictions: List[float]) -> float:
        """Calculate confidence based on model agreement."""
        # Measure of agreement between models
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Higher agreement (lower std) = higher confidence
        confidence = max(0.0, 1.0 - (std_pred * 2))  # Scale factor of 2
        return min(1.0, confidence)
    
    def _identify_risk_factors(self, sequence: ExecutionSequence, 
                              state: EpistemicState) -> List[str]:
        """Identify risk factors from sequence and state."""
        risk_factors = []
        
        # Check for common risk indicators
        if state.uncertainty_score > 0.7:
            risk_factors.append("high_uncertainty")
        
        if state.confidence_level < 0.3:
            risk_factors.append("low_confidence")
        
        if sequence.success_rate < 0.5:
            risk_factors.append("low_historical_success")
        
        if len(sequence.steps) > 20:
            risk_factors.append("complex_execution")
        
        # Check for error patterns
        error_count = sum(1 for step in sequence.steps if step.error_message)
        if error_count > len(sequence.steps) * 0.2:
            risk_factors.append("high_error_rate")
        
        return risk_factors


class OnlineLearningPredictor:
    """
    Online learning predictor with concept drift detection.
    
    Uses River library for incremental learning and adaptation
    to changing agent behavior patterns.
    """
    
    def __init__(self):
        # Initialize online models
        self.classifier = ensemble.AdaptiveRandomForestClassifier(
            n_models=10,
            max_features=0.6,
            lambda_value=6,
            performance_metric=metrics.Accuracy()
        )
        
        self.regressor = ensemble.AdaptiveRandomForestRegressor(
            n_models=10,
            max_features=0.6,
            lambda_value=6,
            performance_metric=metrics.MAE()
        )
        
        # Preprocessing
        self.scaler = preprocessing.StandardScaler()
        
        # Drift detection
        self.drift_detector = None  # Will implement custom drift detection
        
        # Performance tracking
        self.performance_history = {
            'accuracy': [],
            'mae': [],
            'drift_points': []
        }
    
    async def learn_online(self, features: Dict[str, float], 
                          success: bool, completion_time: float):
        """
        Learn from a single example online.
        
        Args:
            features: Feature dictionary
            success: Task success outcome
            completion_time: Task completion time
        """
        # Scale features
        scaled_features = self.scaler.learn_one(features).transform_one(features)
        
        # Update models
        self.classifier.learn_one(scaled_features, success)
        self.regressor.learn_one(scaled_features, completion_time)
        
        # Update performance metrics
        if hasattr(self.classifier, 'performance_metric'):
            current_accuracy = self.classifier.performance_metric.get()
            self.performance_history['accuracy'].append(current_accuracy)
        
        if hasattr(self.regressor, 'performance_metric'):
            current_mae = self.regressor.performance_metric.get()
            self.performance_history['mae'].append(current_mae)
    
    async def predict_online(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Make online prediction.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Prediction dictionary with success probability and completion time
        """
        # Scale features
        scaled_features = self.scaler.transform_one(features)
        
        # Make predictions
        success_prob = self.classifier.predict_proba_one(scaled_features).get(True, 0.0)
        completion_time = self.regressor.predict_one(scaled_features)
        
        return {
            'success_probability': success_prob,
            'completion_time': completion_time,
            'model_confidence': self._calculate_online_confidence()
        }
    
    def _calculate_online_confidence(self) -> float:
        """Calculate confidence for online predictions."""
        # Simple confidence based on recent performance
        if len(self.performance_history['accuracy']) < 10:
            return 0.5  # Low confidence with insufficient data
        
        recent_accuracy = np.mean(self.performance_history['accuracy'][-10:])
        return min(1.0, max(0.0, recent_accuracy))
    
    def detect_concept_drift(self) -> bool:
        """
        Detect concept drift in the data stream.
        
        Returns:
            True if drift is detected, False otherwise
        """
        # Simple drift detection based on performance degradation
        if len(self.performance_history['accuracy']) < 20:
            return False
        
        recent_performance = np.mean(self.performance_history['accuracy'][-10:])
        historical_performance = np.mean(self.performance_history['accuracy'][-20:-10])
        
        # Detect significant performance drop
        if historical_performance - recent_performance > 0.1:
            self.performance_history['drift_points'].append(len(self.performance_history['accuracy']))
            return True
        
        return False
    
    def adapt_to_drift(self):
        """Adapt model to detected concept drift."""
        # Reset models to adapt to new concept
        self.classifier = ensemble.AdaptiveRandomForestClassifier(
            n_models=10,
            max_features=0.6,
            lambda_value=6,
            performance_metric=metrics.Accuracy()
        )
        
        self.regressor = ensemble.AdaptiveRandomForestRegressor(
            n_models=10,
            max_features=0.6,
            lambda_value=6,
            performance_metric=metrics.MAE()
        )
        
        print("Models adapted to concept drift")


class HyperparameterTuner:
    """
    Hyperparameter tuning for machine learning models.
    
    Uses grid search and random search for optimal parameter selection.
    """
    
    def __init__(self):
        self.best_params = {}
        self.tuning_history = {}
    
    async def tune_random_forest(self, X: np.ndarray, y: np.ndarray, 
                                task_type: str = 'classification') -> Dict[str, Any]:
        """
        Tune Random Forest hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'classification' or 'regression'
            
        Returns:
            Best parameters and performance metrics
        """
        if task_type == 'classification':
            model = RandomForestClassifier(random_state=42)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(random_state=42)
            scoring = 'neg_mean_squared_error'
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring=scoring, n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        self.best_params[f'rf_{task_type}'] = grid_search.best_params_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    async def tune_xgboost(self, X: np.ndarray, y: np.ndarray, 
                          task_type: str = 'classification') -> Dict[str, Any]:
        """
        Tune XGBoost hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'classification' or 'regression'
            
        Returns:
            Best parameters and performance metrics
        """
        if task_type == 'classification':
            model = xgb.XGBClassifier(random_state=42)
            scoring = 'accuracy'
        else:
            model = xgb.XGBRegressor(random_state=42)
            scoring = 'neg_mean_squared_error'
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring=scoring, n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        self.best_params[f'xgb_{task_type}'] = grid_search.best_params_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }