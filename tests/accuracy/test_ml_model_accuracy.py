"""
Accuracy validation tests for ML models and prediction systems.
Tests ensure that machine learning components meet accuracy requirements.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import pytest
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

from escai_framework.core.performance_predictor import PerformancePredictor
from escai_framework.core.pattern_analyzer import BehavioralAnalyzer
from escai_framework.core.causal_engine import CausalEngine
from escai_framework.analytics.prediction_models import PredictionModelManager
from escai_framework.analytics.pattern_mining import PatternMiner
from escai_framework.models.epistemic_state import EpistemicState
from escai_framework.models.behavioral_pattern import BehavioralPattern, ExecutionSequence
from escai_framework.models.prediction_result import PredictionResult


class AccuracyTestDataGenerator:
    """Generate test data with known ground truth for accuracy testing."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def generate_prediction_dataset(self, size: int = 1000) -> Tuple[List[Dict], List[bool]]:
        """Generate dataset for prediction accuracy testing."""
        features = []
        labels = []
        
        for i in range(size):
            # Generate features that correlate with success/failure
            confidence = np.random.uniform(0.3, 1.0)
            complexity = np.random.uniform(0.1, 1.0)
            experience = np.random.uniform(0.0, 1.0)
            time_pressure = np.random.uniform(0.0, 1.0)
            
            # Create realistic correlation with success
            success_probability = (
                0.4 * confidence +
                0.2 * experience +
                0.2 * (1 - complexity) +
                0.2 * (1 - time_pressure)
            )
            
            # Add some noise
            success_probability += np.random.normal(0, 0.1)
            success_probability = np.clip(success_probability, 0, 1)
            
            success = np.random.random() < success_probability
            
            features.append({
                "confidence": confidence,
                "complexity": complexity,
                "experience": experience,
                "time_pressure": time_pressure,
                "historical_success_rate": np.random.uniform(0.5, 0.9),
                "resource_availability": np.random.uniform(0.3, 1.0)
            })
            
            labels.append(success)
        
        return features, labels
    
    def generate_pattern_dataset(self, size: int = 500) -> Tuple[List[ExecutionSequence], List[str]]:
        """Generate dataset for pattern recognition accuracy testing."""
        sequences = []
        pattern_labels = []
        
        # Define known patterns
        patterns = {
            "data_analysis": ["load_data", "validate", "analyze", "report"],
            "web_scraping": ["connect", "authenticate", "scrape", "parse", "store"],
            "file_processing": ["open_file", "read", "process", "validate", "save"],
            "api_integration": ["authenticate", "request", "parse_response", "handle_errors"],
            "machine_learning": ["load_data", "preprocess", "train", "validate", "deploy"]
        }
        
        pattern_names = list(patterns.keys())
        
        for i in range(size):
            # Choose a pattern
            pattern_name = np.random.choice(pattern_names)
            base_actions = patterns[pattern_name]
            
            # Add some variation and noise
            actions = base_actions.copy()
            
            # Sometimes add extra actions
            if np.random.random() < 0.3:
                extra_actions = ["debug", "retry", "optimize", "log"]
                actions.extend(np.random.choice(extra_actions, np.random.randint(1, 3)))
            
            # Sometimes remove actions
            if np.random.random() < 0.2 and len(actions) > 2:
                actions = actions[:-1]
            
            # Create execution sequence
            steps = []
            current_time = pd.Timestamp.now()
            
            for j, action in enumerate(actions):
                success = np.random.random() < 0.9  # 90% success rate
                steps.append({
                    "action": action,
                    "timestamp": current_time + pd.Timedelta(seconds=j*30),
                    "success": success,
                    "duration": np.random.exponential(20)
                })
            
            sequence = ExecutionSequence(
                sequence_id=f"test_seq_{i}",
                agent_id=f"test_agent_{i % 10}",
                steps=steps,
                start_time=current_time,
                end_time=current_time + pd.Timedelta(seconds=len(actions)*30),
                success=all(step["success"] for step in steps),
                error_message=None
            )
            
            sequences.append(sequence)
            pattern_labels.append(pattern_name)
        
        return sequences, pattern_labels
    
    def generate_causal_dataset(self, size: int = 200) -> Tuple[List[Dict], List[bool]]:
        """Generate dataset for causal inference accuracy testing."""
        events = []
        causal_labels = []
        
        # Define known causal relationships
        causal_rules = [
            ("data_validation_failure", "analysis_failure", 0.8),
            ("network_timeout", "api_failure", 0.9),
            ("memory_pressure", "performance_degradation", 0.7),
            ("authentication_error", "access_denied", 0.95),
            ("invalid_input", "processing_error", 0.85)
        ]
        
        for i in range(size):
            if np.random.random() < 0.6:  # 60% have causal relationships
                cause, effect, strength = np.random.choice(causal_rules)
                
                # Create temporal sequence with cause before effect
                cause_time = pd.Timestamp.now() + pd.Timedelta(seconds=i*60)
                effect_time = cause_time + pd.Timedelta(seconds=np.random.randint(10, 300))
                
                events.append({
                    "cause_event": cause,
                    "effect_event": effect,
                    "cause_time": cause_time,
                    "effect_time": effect_time,
                    "delay_seconds": (effect_time - cause_time).total_seconds(),
                    "context": {"test_case": i, "strength": strength}
                })
                
                causal_labels.append(True)
            else:
                # Create non-causal relationship
                event1 = np.random.choice(["random_event_1", "unrelated_action", "background_task"])
                event2 = np.random.choice(["random_event_2", "independent_action", "scheduled_task"])
                
                time1 = pd.Timestamp.now() + pd.Timedelta(seconds=i*60)
                time2 = time1 + pd.Timedelta(seconds=np.random.randint(-300, 300))
                
                events.append({
                    "cause_event": event1,
                    "effect_event": event2,
                    "cause_time": min(time1, time2),
                    "effect_time": max(time1, time2),
                    "delay_seconds": abs((time2 - time1).total_seconds()),
                    "context": {"test_case": i, "strength": 0.1}
                })
                
                causal_labels.append(False)
        
        return events, causal_labels


@pytest.mark.asyncio
class TestPredictionAccuracy:
    """Test prediction model accuracy."""
    
    async def test_performance_predictor_accuracy(self, accuracy_test_config):
        """Test performance predictor accuracy against known outcomes."""
        predictor = PerformancePredictor()
        generator = AccuracyTestDataGenerator()
        
        # Generate test dataset
        features, labels = generator.generate_prediction_dataset(
            accuracy_test_config["test_data_size"]
        )
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Train predictor (simulate training with historical data)
        await predictor.train_model(X_train, y_train)
        
        # Make predictions
        predictions = []
        prediction_probabilities = []
        
        for feature_set in X_test:
            # Create mock epistemic state from features
            epistemic_state = self._create_mock_epistemic_state(feature_set)
            
            prediction_result = await predictor.predict_success(epistemic_state)
            predictions.append(prediction_result.predicted_outcome == "success")
            
            # Get probability of success
            prob_success = prediction_result.probability_distribution.get("success", 0.5)
            prediction_probabilities.append(prob_success)
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        # Test probability calibration
        prob_predictions = [p > 0.5 for p in prediction_probabilities]
        prob_accuracy = accuracy_score(y_test, prob_predictions)
        
        # Assertions
        min_accuracy = accuracy_test_config["min_prediction_accuracy"]
        assert accuracy >= min_accuracy, f"Accuracy {accuracy:.3f} below minimum {min_accuracy}"
        assert precision >= 0.8, f"Precision {precision:.3f} below 0.8"
        assert recall >= 0.7, f"Recall {recall:.3f} below 0.7"
        assert f1 >= 0.75, f"F1 score {f1:.3f} below 0.75"
        
        print(f"Prediction accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
              f"Recall: {recall:.3f}, F1: {f1:.3f}")
    
    def _create_mock_epistemic_state(self, features: Dict) -> EpistemicState:
        """Create mock epistemic state from feature dictionary."""
        from escai_framework.models.epistemic_state import BeliefState, KnowledgeState, GoalState
        
        belief_state = BeliefState(
            belief_id="test_belief",
            content="Test belief content",
            confidence=features["confidence"],
            timestamp=pd.Timestamp.now(),
            evidence=["test evidence"]
        )
        
        knowledge_state = KnowledgeState(
            facts=["test fact"],
            concepts=["test concept"],
            relationships={"test": ["relationship"]},
            timestamp=pd.Timestamp.now()
        )
        
        goal_state = GoalState(
            primary_goal="test goal",
            sub_goals=["sub goal 1"],
            progress=features.get("experience", 0.5),
            timestamp=pd.Timestamp.now()
        )
        
        return EpistemicState(
            agent_id="test_agent",
            timestamp=pd.Timestamp.now(),
            belief_states=[belief_state],
            knowledge_state=knowledge_state,
            goal_state=goal_state,
            confidence_level=features["confidence"],
            uncertainty_score=1 - features["confidence"],
            decision_context=features
        )
    
    async def test_cross_validation_accuracy(self, accuracy_test_config):
        """Test prediction accuracy using cross-validation."""
        predictor = PerformancePredictor()
        generator = AccuracyTestDataGenerator()
        
        # Generate larger dataset for cross-validation
        features, labels = generator.generate_prediction_dataset(
            accuracy_test_config["test_data_size"] * 2
        )
        
        # Perform cross-validation
        cv_folds = accuracy_test_config["cross_validation_folds"]
        cv_scores = []
        
        # Manual cross-validation since we need async operations
        fold_size = len(features) // cv_folds
        
        for fold in range(cv_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < cv_folds - 1 else len(features)
            
            # Split data
            test_features = features[start_idx:end_idx]
            test_labels = labels[start_idx:end_idx]
            train_features = features[:start_idx] + features[end_idx:]
            train_labels = labels[:start_idx] + labels[end_idx:]
            
            # Train and test
            await predictor.train_model(train_features, train_labels)
            
            fold_predictions = []
            for feature_set in test_features:
                epistemic_state = self._create_mock_epistemic_state(feature_set)
                prediction_result = await predictor.predict_success(epistemic_state)
                fold_predictions.append(prediction_result.predicted_outcome == "success")
            
            fold_accuracy = accuracy_score(test_labels, fold_predictions)
            cv_scores.append(fold_accuracy)
        
        # Calculate cross-validation statistics
        mean_cv_accuracy = np.mean(cv_scores)
        std_cv_accuracy = np.std(cv_scores)
        
        # Assertions
        min_accuracy = accuracy_test_config["min_prediction_accuracy"]
        assert mean_cv_accuracy >= min_accuracy, \
            f"CV accuracy {mean_cv_accuracy:.3f} below minimum {min_accuracy}"
        assert std_cv_accuracy <= 0.1, \
            f"CV accuracy std {std_cv_accuracy:.3f} too high (unstable model)"
        
        print(f"Cross-validation accuracy: {mean_cv_accuracy:.3f} ± {std_cv_accuracy:.3f}")


@pytest.mark.asyncio
class TestPatternRecognitionAccuracy:
    """Test behavioral pattern recognition accuracy."""
    
    async def test_pattern_mining_accuracy(self, accuracy_test_config):
        """Test pattern mining accuracy against known patterns."""
        analyzer = BehavioralAnalyzer()
        generator = AccuracyTestDataGenerator()
        
        # Generate test dataset with known patterns
        sequences, true_labels = generator.generate_pattern_dataset(
            accuracy_test_config["test_data_size"]
        )
        
        # Mine patterns
        discovered_patterns = await analyzer.mine_patterns(sequences)
        
        # Classify sequences using discovered patterns
        predicted_labels = []
        
        for sequence in sequences:
            # Find best matching pattern
            best_match = None
            best_score = 0
            
            for pattern in discovered_patterns:
                score = await self._calculate_pattern_match_score(sequence, pattern)
                if score > best_score:
                    best_score = score
                    best_match = pattern
            
            if best_match and best_score > 0.5:
                predicted_labels.append(best_match.pattern_name)
            else:
                predicted_labels.append("unknown")
        
        # Calculate accuracy
        # Map pattern names to simplified categories for comparison
        true_categories = [self._simplify_pattern_name(label) for label in true_labels]
        pred_categories = [self._simplify_pattern_name(label) for label in predicted_labels]
        
        accuracy = accuracy_score(true_categories, pred_categories)
        
        # Calculate per-pattern accuracy
        unique_patterns = set(true_categories)
        pattern_accuracies = {}
        
        for pattern in unique_patterns:
            pattern_indices = [i for i, label in enumerate(true_categories) if label == pattern]
            if pattern_indices:
                pattern_true = [true_categories[i] for i in pattern_indices]
                pattern_pred = [pred_categories[i] for i in pattern_indices]
                pattern_accuracies[pattern] = accuracy_score(pattern_true, pattern_pred)
        
        # Assertions
        min_accuracy = accuracy_test_config["min_pattern_detection_accuracy"]
        assert accuracy >= min_accuracy, f"Pattern accuracy {accuracy:.3f} below minimum {min_accuracy}"
        
        # Check that at least 80% of patterns have good accuracy
        good_patterns = sum(1 for acc in pattern_accuracies.values() if acc >= 0.7)
        pattern_ratio = good_patterns / len(pattern_accuracies) if pattern_accuracies else 0
        assert pattern_ratio >= 0.8, f"Only {pattern_ratio:.1%} of patterns have good accuracy"
        
        print(f"Pattern recognition accuracy: {accuracy:.3f}")
        print(f"Per-pattern accuracies: {pattern_accuracies}")
    
    async def _calculate_pattern_match_score(self, sequence: ExecutionSequence, pattern: BehavioralPattern) -> float:
        """Calculate how well a sequence matches a pattern."""
        sequence_actions = [step["action"] for step in sequence.steps]
        
        # Find the most similar sequence in the pattern
        best_similarity = 0
        
        for pattern_sequence in pattern.execution_sequences:
            pattern_actions = [step["action"] for step in pattern_sequence.steps]
            
            # Calculate sequence similarity (simplified)
            common_actions = set(sequence_actions) & set(pattern_actions)
            total_actions = set(sequence_actions) | set(pattern_actions)
            
            if total_actions:
                similarity = len(common_actions) / len(total_actions)
                best_similarity = max(best_similarity, similarity)
        
        return best_similarity
    
    def _simplify_pattern_name(self, pattern_name: str) -> str:
        """Simplify pattern names for comparison."""
        if "data" in pattern_name.lower():
            return "data_processing"
        elif "web" in pattern_name.lower() or "scraping" in pattern_name.lower():
            return "web_operations"
        elif "file" in pattern_name.lower():
            return "file_operations"
        elif "api" in pattern_name.lower():
            return "api_operations"
        elif "machine" in pattern_name.lower() or "ml" in pattern_name.lower():
            return "ml_operations"
        else:
            return "other"
    
    async def test_anomaly_detection_accuracy(self, accuracy_test_config):
        """Test anomaly detection accuracy."""
        analyzer = BehavioralAnalyzer()
        generator = AccuracyTestDataGenerator()
        
        # Generate normal sequences
        normal_sequences, _ = generator.generate_pattern_dataset(200)
        
        # Generate anomalous sequences
        anomalous_sequences = []
        for i in range(50):
            # Create clearly anomalous sequences
            steps = []
            anomalous_actions = ["corrupt_data", "infinite_loop", "memory_leak", "crash", "deadlock"]
            
            for j, action in enumerate(np.random.choice(anomalous_actions, 3)):
                steps.append({
                    "action": action,
                    "timestamp": pd.Timestamp.now() + pd.Timedelta(seconds=j*10),
                    "success": False,
                    "duration": np.random.uniform(1000, 5000)  # Very long duration
                })
            
            anomalous_sequence = ExecutionSequence(
                sequence_id=f"anomaly_{i}",
                agent_id=f"test_agent_{i}",
                steps=steps,
                start_time=pd.Timestamp.now(),
                end_time=pd.Timestamp.now() + pd.Timedelta(seconds=len(steps)*10),
                success=False,
                error_message="Anomalous behavior detected"
            )
            anomalous_sequences.append(anomalous_sequence)
        
        # Train on normal sequences
        await analyzer.train_anomaly_detector(normal_sequences)
        
        # Test anomaly detection
        all_sequences = normal_sequences + anomalous_sequences
        true_labels = [False] * len(normal_sequences) + [True] * len(anomalous_sequences)
        
        predicted_labels = []
        for sequence in all_sequences:
            anomaly_score = await analyzer.detect_anomalies(sequence)
            is_anomaly = anomaly_score.score > 0.5  # Threshold for anomaly
            predicted_labels.append(is_anomaly)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        
        # Assertions
        assert accuracy >= 0.85, f"Anomaly detection accuracy {accuracy:.3f} below 0.85"
        assert precision >= 0.8, f"Anomaly detection precision {precision:.3f} below 0.8"
        assert recall >= 0.7, f"Anomaly detection recall {recall:.3f} below 0.7"
        
        print(f"Anomaly detection - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")


@pytest.mark.asyncio
class TestCausalInferenceAccuracy:
    """Test causal inference accuracy."""
    
    async def test_causal_discovery_accuracy(self, accuracy_test_config):
        """Test causal relationship discovery accuracy."""
        causal_engine = CausalEngine()
        generator = AccuracyTestDataGenerator()
        
        # Generate test dataset with known causal relationships
        events, true_causal_labels = generator.generate_causal_dataset(
            accuracy_test_config["test_data_size"] // 5  # Smaller dataset for causal inference
        )
        
        # Discover causal relationships
        discovered_relationships = []
        
        for event_data in events:
            # Convert to temporal events format expected by causal engine
            temporal_events = [
                {
                    "event_type": event_data["cause_event"],
                    "timestamp": event_data["cause_time"],
                    "agent_id": "test_agent"
                },
                {
                    "event_type": event_data["effect_event"],
                    "timestamp": event_data["effect_time"],
                    "agent_id": "test_agent"
                }
            ]
            
            relationships = await causal_engine.discover_relationships(temporal_events)
            discovered_relationships.extend(relationships)
        
        # Evaluate discovered relationships
        predicted_causal_labels = []
        
        for i, event_data in enumerate(events):
            # Check if a causal relationship was discovered for this event pair
            found_causal = False
            
            for relationship in discovered_relationships:
                if (relationship.cause_event == event_data["cause_event"] and
                    relationship.effect_event == event_data["effect_event"]):
                    # Check if confidence meets minimum threshold
                    min_confidence = accuracy_test_config["min_causal_inference_confidence"]
                    if relationship.confidence >= min_confidence:
                        found_causal = True
                        break
            
            predicted_causal_labels.append(found_causal)
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(true_causal_labels, predicted_causal_labels)
        precision = precision_score(true_causal_labels, predicted_causal_labels)
        recall = recall_score(true_causal_labels, predicted_causal_labels)
        f1 = f1_score(true_causal_labels, predicted_causal_labels)
        
        # Assertions
        min_accuracy = 0.75  # Causal inference is inherently more difficult
        assert accuracy >= min_accuracy, f"Causal inference accuracy {accuracy:.3f} below {min_accuracy}"
        assert precision >= 0.7, f"Causal inference precision {precision:.3f} below 0.7"
        assert recall >= 0.6, f"Causal inference recall {recall:.3f} below 0.6"
        
        print(f"Causal inference - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
              f"Recall: {recall:.3f}, F1: {f1:.3f}")
    
    async def test_granger_causality_accuracy(self, accuracy_test_config):
        """Test Granger causality testing accuracy."""
        causal_engine = CausalEngine()
        
        # Generate time series with known causal relationships
        n_points = 200
        time_index = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
        
        # Create causally related time series
        np.random.seed(42)
        
        # X causes Y with 2-hour delay
        x_series = np.random.normal(0, 1, n_points)
        y_series = np.zeros(n_points)
        
        for i in range(2, n_points):
            # Y depends on X from 2 periods ago plus noise
            y_series[i] = 0.7 * x_series[i-2] + 0.3 * y_series[i-1] + np.random.normal(0, 0.5)
        
        # Create non-causal time series
        z_series = np.random.normal(0, 1, n_points)
        
        # Test Granger causality
        xy_causality = await causal_engine.test_granger_causality(
            pd.DataFrame({'X': x_series, 'Y': y_series}, index=time_index)
        )
        
        xz_causality = await causal_engine.test_granger_causality(
            pd.DataFrame({'X': x_series, 'Z': z_series}, index=time_index)
        )
        
        # Assertions
        assert xy_causality.p_value < 0.05, f"Failed to detect known causal relationship (p={xy_causality.p_value:.4f})"
        assert xz_causality.p_value >= 0.05, f"False positive causal relationship detected (p={xz_causality.p_value:.4f})"
        
        print(f"Granger causality - Known causal p-value: {xy_causality.p_value:.4f}, "
              f"Non-causal p-value: {xz_causality.p_value:.4f}")


@pytest.mark.asyncio
class TestModelRobustness:
    """Test model robustness and reliability."""
    
    async def test_prediction_consistency(self, accuracy_test_config):
        """Test that predictions are consistent across multiple runs."""
        predictor = PerformancePredictor()
        generator = AccuracyTestDataGenerator()
        
        # Generate test data
        features, labels = generator.generate_prediction_dataset(100)
        
        # Train model
        train_features, test_features = features[:70], features[70:]
        train_labels, test_labels = labels[:70], labels[70:]
        
        await predictor.train_model(train_features, train_labels)
        
        # Make predictions multiple times
        prediction_runs = []
        
        for run in range(5):
            run_predictions = []
            for feature_set in test_features:
                epistemic_state = self._create_mock_epistemic_state(feature_set)
                prediction_result = await predictor.predict_success(epistemic_state)
                run_predictions.append(prediction_result.confidence)
            
            prediction_runs.append(run_predictions)
        
        # Calculate consistency
        prediction_array = np.array(prediction_runs)
        consistency_scores = []
        
        for i in range(len(test_features)):
            predictions_for_sample = prediction_array[:, i]
            std_dev = np.std(predictions_for_sample)
            consistency_scores.append(std_dev)
        
        avg_consistency = np.mean(consistency_scores)
        
        # Assertions
        assert avg_consistency <= 0.05, f"Prediction consistency {avg_consistency:.4f} too low (high variance)"
        
        print(f"Prediction consistency (std dev): {avg_consistency:.4f}")
    
    def _create_mock_epistemic_state(self, features: Dict) -> EpistemicState:
        """Create mock epistemic state from feature dictionary."""
        from escai_framework.models.epistemic_state import BeliefState, KnowledgeState, GoalState
        
        belief_state = BeliefState(
            belief_id="test_belief",
            content="Test belief content",
            confidence=features["confidence"],
            timestamp=pd.Timestamp.now(),
            evidence=["test evidence"]
        )
        
        knowledge_state = KnowledgeState(
            facts=["test fact"],
            concepts=["test concept"],
            relationships={"test": ["relationship"]},
            timestamp=pd.Timestamp.now()
        )
        
        goal_state = GoalState(
            primary_goal="test goal",
            sub_goals=["sub goal 1"],
            progress=features.get("experience", 0.5),
            timestamp=pd.Timestamp.now()
        )
        
        return EpistemicState(
            agent_id="test_agent",
            timestamp=pd.Timestamp.now(),
            belief_states=[belief_state],
            knowledge_state=knowledge_state,
            goal_state=goal_state,
            confidence_level=features["confidence"],
            uncertainty_score=1 - features["confidence"],
            decision_context=features
        )


if __name__ == "__main__":
    # Run accuracy tests
    pytest.main([__file__, "-v", "--tb=short"])