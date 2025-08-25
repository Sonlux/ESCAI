"""
Unit tests for failure analysis module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import asyncio
from datetime import datetime
from collections import Counter

from escai_framework.analytics.failure_analysis import (
    FailureAnalysisEngine, FailurePatternDetector, RootCauseAnalyzer,
    FailureMode, RootCause, FailureAnalysisResult
)
from escai_framework.models.behavioral_pattern import ExecutionSequence, ExecutionStep
from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState


@pytest.fixture
def failed_sequences():
    """Create sample failed execution sequences."""
    sequences = []
    
    # Timeout failure pattern
    for i in range(5):
        steps = [
            ExecutionStep(
                step_id=f"timeout_{i}_1",
                step_type="planning",
                action="analyze",
                duration=2.0,
                success_probability=0.8,
                context={},
                error_message=None
            ),
            ExecutionStep(
                step_id=f"timeout_{i}_2",
                step_type="execution",
                action="execute",
                duration=10.0,  # Long duration
                success_probability=0.3,
                context={},
                error_message="Timeout error: Operation exceeded time limit"
            )
        ]
        
        sequences.append(ExecutionSequence(
            sequence_id=f"timeout_seq_{i}",
            agent_id=f"agent_{i % 2}",
            steps=steps,
            total_duration=12.0,
            success_rate=0.0  # Failed
        ))
    
    # Memory failure pattern
    for i in range(3):
        steps = [
            ExecutionStep(
                step_id=f"memory_{i}_1",
                step_type="reasoning",
                action="think",
                duration=1.0,
                success_probability=0.9,
                context={},
                error_message=None
            ),
            ExecutionStep(
                step_id=f"memory_{i}_2",
                step_type="action",
                action="process",
                duration=3.0,
                success_probability=0.2,
                context={},
                error_message="Memory error: Out of memory"
            )
        ]
        
        sequences.append(ExecutionSequence(
            sequence_id=f"memory_seq_{i}",
            agent_id=f"agent_{i}",
            steps=steps,
            total_duration=4.0,
            success_rate=0.0  # Failed
        ))
    
    return sequences


@pytest.fixture
def successful_sequences():
    """Create sample successful execution sequences."""
    sequences = []
    
    for i in range(10):
        steps = [
            ExecutionStep(
                step_id=f"success_{i}_1",
                step_type="planning",
                action="analyze",
                duration=1.0,
                success_probability=0.9,
                context={},
                error_message=None
            ),
            ExecutionStep(
                step_id=f"success_{i}_2",
                step_type="execution",
                action="execute",
                duration=2.0,
                success_probability=0.95,
                context={},
                error_message=None
            ),
            ExecutionStep(
                step_id=f"success_{i}_3",
                step_type="observation",
                action="verify",
                duration=0.5,
                success_probability=0.98,
                context={},
                error_message=None
            )
        ]
        
        sequences.append(ExecutionSequence(
            sequence_id=f"success_seq_{i}",
            agent_id=f"agent_{i % 3}",
            steps=steps,
            total_duration=3.5,
            success_rate=1.0  # Successful
        ))
    
    return sequences


@pytest.fixture
def sample_epistemic_states():
    """Create sample epistemic states."""
    states = []
    
    # States for failed sequences (8 total)
    for i in range(8):
        belief_states = [
            BeliefState(
                belief_id=f"belief_{i}",
                content=f"belief_content_{i}",
                confidence=0.3 + i * 0.05,  # Lower confidence for failures
                source="test",
                timestamp=datetime.now()
            )
        ]
        
        knowledge_state = KnowledgeState(
            facts=[f"fact_{i}"],
            rules=[f"rule_{i}"],
            concepts=[f"concept_{i}"]
        )
        
        goal_state = GoalState(
            active_goals=[f"goal_{i}"],
            completed_goals=[],
            failed_goals=[f"failed_goal_{i}"]
        )
        
        states.append(EpistemicState(
            agent_id=f"agent_{i % 3}",
            timestamp=datetime.now(),
            belief_states=belief_states,
            knowledge_state=knowledge_state,
            goal_state=goal_state,
            confidence_level=0.3 + i * 0.05,  # Low confidence
            uncertainty_score=0.8 - i * 0.05,  # High uncertainty
            decision_context={}
        ))
    
    # States for successful sequences (10 total)
    for i in range(10):
        belief_states = [
            BeliefState(
                belief_id=f"success_belief_{i}",
                content=f"success_belief_content_{i}",
                confidence=0.8 + i * 0.01,  # Higher confidence for success
                source="test",
                timestamp=datetime.now()
            )
        ]
        
        knowledge_state = KnowledgeState(
            facts=[f"success_fact_{i}"],
            rules=[f"success_rule_{i}"],
            concepts=[f"success_concept_{i}"]
        )
        
        goal_state = GoalState(
            active_goals=[f"success_goal_{i}"],
            completed_goals=[f"completed_goal_{i}"],
            failed_goals=[]
        )
        
        states.append(EpistemicState(
            agent_id=f"agent_{i % 3}",
            timestamp=datetime.now(),
            belief_states=belief_states,
            knowledge_state=knowledge_state,
            goal_state=goal_state,
            confidence_level=0.8 + i * 0.01,  # High confidence
            uncertainty_score=0.2 - i * 0.01,  # Low uncertainty
            decision_context={}
        ))
    
    return states


class TestFailureMode:
    """Test FailureMode data structure."""
    
    def test_creation(self):
        """Test failure mode creation."""
        failure_mode = FailureMode(
            failure_id="timeout_failure",
            failure_name="Timeout Failure Mode",
            description="Operations that exceed time limits",
            frequency=10,
            severity=0.8,
            common_triggers=["long_operation", "network_delay"],
            failure_patterns=["plan -> execute(timeout)"],
            recovery_strategies=["increase_timeout", "retry_mechanism"],
            prevention_measures=["timeout_monitoring", "early_detection"],
            statistical_significance=0.95
        )
        
        assert failure_mode.failure_id == "timeout_failure"
        assert failure_mode.failure_name == "Timeout Failure Mode"
        assert failure_mode.frequency == 10
        assert failure_mode.severity == 0.8
        assert len(failure_mode.common_triggers) == 2
        assert len(failure_mode.recovery_strategies) == 2


class TestRootCause:
    """Test RootCause data structure."""
    
    def test_creation(self):
        """Test root cause creation."""
        root_cause = RootCause(
            cause_id="low_confidence",
            cause_description="Agent has low confidence in decisions",
            confidence=0.85,
            evidence=["confidence < 0.3", "high uncertainty"],
            contributing_factors=["insufficient_data", "complex_task"],
            causal_chain=["low_confidence", "poor_decisions", "task_failure"]
        )
        
        assert root_cause.cause_id == "low_confidence"
        assert root_cause.confidence == 0.85
        assert len(root_cause.evidence) == 2
        assert len(root_cause.contributing_factors) == 2
        assert len(root_cause.causal_chain) == 3


class TestFailurePatternDetector:
    """Test failure pattern detection functionality."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = FailurePatternDetector(min_cluster_size=2, eps=0.3)
        
        assert detector.min_cluster_size == 2
        assert detector.eps == 0.3
        assert detector.scaler is not None
        assert detector.clusterer is not None
        assert detector.anomaly_detector is not None
    
    @pytest.mark.asyncio
    async def test_detect_failure_patterns(self, failed_sequences):
        """Test failure pattern detection."""
        detector = FailurePatternDetector(min_cluster_size=2)
        
        with patch.object(detector.clusterer, 'fit_predict') as mock_cluster:
            # Mock clustering results - two clusters plus noise
            mock_cluster.return_value = np.array([0, 0, 0, 1, 1, 1, -1, -1])
            
            patterns = await detector.detect_failure_patterns(failed_sequences)
            
            assert isinstance(patterns, dict)
            assert len(patterns) > 0
            
            # Should have clusters and possibly anomalous failures
            cluster_names = list(patterns.keys())
            assert any("failure_pattern" in name for name in cluster_names)
    
    @pytest.mark.asyncio
    async def test_identify_failure_signatures(self, failed_sequences):
        """Test failure signature identification."""
        detector = FailurePatternDetector()
        
        # Create mock failure clusters
        failure_clusters = {
            "timeout_failures": failed_sequences[:5],
            "memory_failures": failed_sequences[5:]
        }
        
        signatures = await detector.identify_failure_signatures(failure_clusters)
        
        assert isinstance(signatures, dict)
        assert "timeout_failures" in signatures
        assert "memory_failures" in signatures
        
        # Check signature structure
        for cluster_name, signature in signatures.items():
            assert 'cluster_size' in signature
            assert 'common_errors' in signature
            assert 'common_patterns' in signature
            assert 'timing_stats' in signature
            assert 'failure_characteristics' in signature
    
    def test_extract_failure_features(self, failed_sequences):
        """Test failure feature extraction."""
        detector = FailurePatternDetector()
        
        features = detector._extract_failure_features(failed_sequences)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(failed_sequences)
        assert features.shape[1] > 0  # Should have multiple features
        
        # Check that features are numerical
        assert np.all(np.isfinite(features))
    
    @pytest.mark.asyncio
    async def test_extract_cluster_signature(self, failed_sequences):
        """Test cluster signature extraction."""
        detector = FailurePatternDetector()
        
        # Test with timeout failures
        timeout_sequences = failed_sequences[:5]
        signature = await detector._extract_cluster_signature(timeout_sequences)
        
        assert 'cluster_size' in signature
        assert signature['cluster_size'] == 5
        assert 'common_errors' in signature
        assert 'timing_stats' in signature
        
        # Should find timeout errors
        common_errors = [error['error'] for error in signature['common_errors']]
        assert any('timeout' in error.lower() for error in common_errors)
    
    def test_analyze_failure_characteristics(self, failed_sequences):
        """Test failure characteristics analysis."""
        detector = FailurePatternDetector()
        
        characteristics = detector._analyze_failure_characteristics(failed_sequences)
        
        assert isinstance(characteristics, dict)
        expected_keys = ['early_failures', 'late_failures', 'timeout_failures', 
                        'resource_failures', 'logic_failures']
        
        for key in expected_keys:
            assert key in characteristics
            assert isinstance(characteristics[key], int)
            assert characteristics[key] >= 0


class TestRootCauseAnalyzer:
    """Test root cause analysis functionality."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = RootCauseAnalyzer()
        
        assert analyzer.decision_tree is not None
        assert analyzer.causal_graph is not None
    
    @pytest.mark.asyncio
    async def test_analyze_root_causes(self, failed_sequences, successful_sequences, sample_epistemic_states):
        """Test root cause analysis."""
        analyzer = RootCauseAnalyzer()
        
        with patch.object(analyzer.decision_tree, 'fit') as mock_fit, \
             patch.object(analyzer, '_extract_decision_rules') as mock_rules, \
             patch.object(analyzer, '_build_causal_graph') as mock_graph, \
             patch.object(analyzer, '_identify_root_causes') as mock_identify:
            
            # Mock decision rules
            mock_rules.return_value = [
                {
                    'conditions': ['feature_4 <= 0.3', 'feature_5 > 0.7'],
                    'prediction': 0,  # Failure
                    'confidence': 0.85,
                    'samples': 15
                }
            ]
            
            # Mock root causes
            mock_identify.return_value = [
                RootCause(
                    cause_id="low_confidence",
                    cause_description="Low agent confidence",
                    confidence=0.85,
                    evidence=['feature_4 <= 0.3'],
                    contributing_factors=['confidence_level'],
                    causal_chain=['low_confidence', 'poor_decisions', 'failure']
                )
            ]
            
            root_causes = await analyzer.analyze_root_causes(
                failed_sequences, successful_sequences, sample_epistemic_states
            )
            
            assert isinstance(root_causes, list)
            assert len(root_causes) > 0
            assert all(isinstance(cause, RootCause) for cause in root_causes)
            
            # Verify mocks were called
            mock_fit.assert_called_once()
            mock_rules.assert_called_once()
            mock_graph.assert_called_once()
            mock_identify.assert_called_once()
    
    def test_prepare_causal_data(self, failed_sequences, successful_sequences, sample_epistemic_states):
        """Test causal data preparation."""
        analyzer = RootCauseAnalyzer()
        
        X, y = analyzer._prepare_causal_data(
            failed_sequences, successful_sequences, sample_epistemic_states
        )
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 0  # Should have features
        
        # Check labels
        assert np.all((y == 0) | (y == 1))  # Binary labels
        
        # Should have both failure (0) and success (1) labels
        assert 0 in y
        assert 1 in y
    
    def test_extract_causal_features(self, failed_sequences, sample_epistemic_states):
        """Test causal feature extraction."""
        analyzer = RootCauseAnalyzer()
        
        features = analyzer._extract_causal_features(
            failed_sequences[0], sample_epistemic_states[0]
        )
        
        assert isinstance(features, list)
        assert len(features) == 12  # Expected number of features
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_extract_decision_rules(self):
        """Test decision rule extraction."""
        analyzer = RootCauseAnalyzer()
        
        # Mock decision tree structure
        mock_tree = Mock()
        mock_tree.tree_.feature = np.array([0, 1, -2, -2, -2])  # -2 indicates leaf
        mock_tree.tree_.threshold = np.array([0.5, 0.3, 0, 0, 0])
        mock_tree.tree_.children_left = np.array([1, 2, -1, -1, -1])
        mock_tree.tree_.children_right = np.array([3, 4, -1, -1, -1])
        mock_tree.tree_.value = np.array([
            [[10, 5]],   # Root
            [[8, 2]],    # Left child
            [[6, 1]],    # Leaf
            [[2, 3]],    # Right child
            [[2, 1]]     # Leaf
        ])
        mock_tree.tree_.n_node_samples = np.array([15, 10, 7, 5, 3])
        mock_tree.tree_.n_features = 2
        
        analyzer.decision_tree = mock_tree
        
        rules = analyzer._extract_decision_rules()
        
        assert isinstance(rules, list)
        assert len(rules) > 0
        
        # Check rule structure
        for rule in rules:
            assert 'conditions' in rule
            assert 'prediction' in rule
            assert 'confidence' in rule
            assert 'samples' in rule
    
    def test_interpret_rule_conditions(self):
        """Test rule condition interpretation."""
        analyzer = RootCauseAnalyzer()
        
        conditions = ['feature_4 <= 0.3', 'feature_5 > 0.7', 'feature_0 > 20']
        interpretation = analyzer._interpret_rule_conditions(conditions)
        
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
        
        # Should contain interpretable descriptions
        assert any(keyword in interpretation.lower() for keyword in 
                  ['confidence', 'uncertainty', 'complex'])
    
    def test_extract_contributing_factors(self):
        """Test contributing factor extraction."""
        analyzer = RootCauseAnalyzer()
        
        conditions = ['feature_4 <= 0.3', 'feature_5 > 0.7', 'feature_0 > 20']
        factors = analyzer._extract_contributing_factors(conditions)
        
        assert isinstance(factors, list)
        assert len(factors) > 0
        
        # Should map to meaningful factor names
        expected_factors = ['confidence_level', 'uncertainty_score', 'sequence_complexity']
        assert any(factor in expected_factors for factor in factors)
    
    def test_trace_causal_chain(self):
        """Test causal chain tracing."""
        analyzer = RootCauseAnalyzer()
        
        # Test different cause descriptions
        low_confidence_chain = analyzer._trace_causal_chain("Low agent confidence")
        assert isinstance(low_confidence_chain, list)
        assert len(low_confidence_chain) > 0
        assert "Low confidence" in low_confidence_chain[0]
        
        high_uncertainty_chain = analyzer._trace_causal_chain("High uncertainty")
        assert isinstance(high_uncertainty_chain, list)
        assert "High uncertainty" in high_uncertainty_chain[0]


class TestFailureAnalysisEngine:
    """Test the main failure analysis engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = FailureAnalysisEngine()
        
        assert isinstance(engine.pattern_detector, FailurePatternDetector)
        assert isinstance(engine.root_cause_analyzer, RootCauseAnalyzer)
    
    @pytest.mark.asyncio
    async def test_analyze_failures(self, failed_sequences, successful_sequences, sample_epistemic_states):
        """Test comprehensive failure analysis."""
        engine = FailureAnalysisEngine()
        
        with patch.object(engine.pattern_detector, 'detect_failure_patterns') as mock_detect, \
             patch.object(engine.pattern_detector, 'identify_failure_signatures') as mock_signatures, \
             patch.object(engine.root_cause_analyzer, 'analyze_root_causes') as mock_root_causes:
            
            # Mock pattern detection
            mock_detect.return_value = {
                "timeout_failures": failed_sequences[:5],
                "memory_failures": failed_sequences[5:]
            }
            
            # Mock signatures
            mock_signatures.return_value = {
                "timeout_failures": {
                    'cluster_size': 5,
                    'common_errors': [{'error': 'Timeout error', 'frequency': 5}],
                    'timing_stats': {'mean_duration': 12.0}
                },
                "memory_failures": {
                    'cluster_size': 3,
                    'common_errors': [{'error': 'Memory error', 'frequency': 3}],
                    'timing_stats': {'mean_duration': 4.0}
                }
            }
            
            # Mock root causes
            mock_root_causes.return_value = [
                RootCause(
                    cause_id="low_confidence",
                    cause_description="Low agent confidence",
                    confidence=0.85,
                    evidence=['confidence < 0.3'],
                    contributing_factors=['confidence_level'],
                    causal_chain=['low_confidence', 'failure']
                )
            ]
            
            result = await engine.analyze_failures(
                failed_sequences, successful_sequences, sample_epistemic_states
            )
            
            assert isinstance(result, FailureAnalysisResult)
            assert len(result.failure_modes) > 0
            assert len(result.root_causes) > 0
            assert isinstance(result.failure_clusters, dict)
            assert isinstance(result.risk_factors, list)
            assert isinstance(result.recommendations, list)
            assert isinstance(result.prevention_strategies, list)
    
    def test_calculate_failure_severity(self, failed_sequences):
        """Test failure severity calculation."""
        engine = FailureAnalysisEngine()
        
        severity = engine._calculate_failure_severity(failed_sequences[:3])
        
        assert isinstance(severity, float)
        assert 0.0 <= severity <= 1.0
    
    def test_extract_common_triggers(self, failed_sequences):
        """Test common trigger extraction."""
        engine = FailureAnalysisEngine()
        
        triggers = engine._extract_common_triggers(failed_sequences)
        
        assert isinstance(triggers, list)
        assert len(triggers) <= 5  # Limited to top 5
        
        # Should contain step_type:action format
        for trigger in triggers:
            assert ':' in trigger
    
    def test_generate_failure_name(self):
        """Test failure name generation."""
        engine = FailureAnalysisEngine()
        
        # Test timeout failure
        timeout_signature = {
            'common_errors': [{'error': 'Timeout error: Operation exceeded limit', 'frequency': 5}]
        }
        name = engine._generate_failure_name("cluster_0", timeout_signature)
        assert "timeout" in name.lower()
        
        # Test memory failure
        memory_signature = {
            'common_errors': [{'error': 'Memory error: Out of memory', 'frequency': 3}]
        }
        name = engine._generate_failure_name("cluster_1", memory_signature)
        assert "memory" in name.lower() or "resource" in name.lower()
    
    def test_generate_failure_description(self):
        """Test failure description generation."""
        engine = FailureAnalysisEngine()
        
        signature = {
            'timing_stats': {'mean_duration': 12.5},
            'common_errors': [{'error': 'Timeout error', 'frequency': 5}],
            'failure_characteristics': {'early_failures': 2, 'late_failures': 8}
        }
        
        description = engine._generate_failure_description(signature)
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "12.5" in description  # Duration should be mentioned
        assert "timeout" in description.lower()  # Error should be mentioned
    
    def test_suggest_recovery_strategies(self):
        """Test recovery strategy suggestions."""
        engine = FailureAnalysisEngine()
        
        # Test timeout failure signature
        timeout_signature = {
            'common_errors': [{'error': 'Timeout error: Operation exceeded limit', 'frequency': 5}]
        }
        
        strategies = engine._suggest_recovery_strategies(timeout_signature)
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        
        # Should suggest timeout-related strategies
        strategy_text = ' '.join(strategies).lower()
        assert any(keyword in strategy_text for keyword in ['timeout', 'retry', 'checkpoint'])
    
    def test_suggest_prevention_measures(self):
        """Test prevention measure suggestions."""
        engine = FailureAnalysisEngine()
        
        signature = {
            'failure_characteristics': {
                'timeout_failures': 5,
                'resource_failures': 2,
                'early_failures': 7,
                'late_failures': 1
            }
        }
        
        measures = engine._suggest_prevention_measures(signature)
        
        assert isinstance(measures, list)
        assert len(measures) > 0
        
        # Should suggest relevant prevention measures
        measures_text = ' '.join(measures).lower()
        assert any(keyword in measures_text for keyword in 
                  ['timeout', 'resource', 'monitoring', 'validation'])
    
    def test_identify_risk_factors(self):
        """Test risk factor identification."""
        engine = FailureAnalysisEngine()
        
        failure_modes = [
            FailureMode(
                failure_id="mode1",
                failure_name="Test Mode",
                description="Test",
                frequency=5,
                severity=0.8,
                common_triggers=["trigger1", "trigger2"],
                failure_patterns=[],
                recovery_strategies=[],
                prevention_measures=[],
                statistical_significance=0.9
            )
        ]
        
        root_causes = [
            RootCause(
                cause_id="cause1",
                cause_description="Test cause",
                confidence=0.8,
                evidence=[],
                contributing_factors=["factor1", "factor2"],
                causal_chain=[]
            )
        ]
        
        risk_factors = engine._identify_risk_factors(failure_modes, root_causes)
        
        assert isinstance(risk_factors, list)
        assert "trigger1" in risk_factors
        assert "trigger2" in risk_factors
        assert "factor1" in risk_factors
        assert "factor2" in risk_factors
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        engine = FailureAnalysisEngine()
        
        # High frequency failure mode
        failure_modes = [
            FailureMode(
                failure_id="high_freq",
                failure_name="High Frequency Failure",
                description="Test",
                frequency=10,  # High frequency
                severity=0.9,  # High severity
                common_triggers=[],
                failure_patterns=[],
                recovery_strategies=[],
                prevention_measures=[],
                statistical_significance=0.9
            )
        ]
        
        root_causes = [
            RootCause(
                cause_id="confidence_issue",
                cause_description="Low agent confidence",
                confidence=0.9,
                evidence=[],
                contributing_factors=[],
                causal_chain=[]
            )
        ]
        
        recommendations = engine._generate_recommendations(failure_modes, root_causes)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend addressing high-frequency and high-severity failures
        rec_text = ' '.join(recommendations).lower()
        assert any(keyword in rec_text for keyword in 
                  ['high-frequency', 'high-severity', 'confidence'])


@pytest.mark.integration
class TestFailureAnalysisIntegration:
    """Integration tests for failure analysis."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_failure_analysis(self, failed_sequences, successful_sequences, 
                                              sample_epistemic_states):
        """Test complete failure analysis workflow."""
        engine = FailureAnalysisEngine()
        
        # This should work with real analysis (though may be slow)
        result = await engine.analyze_failures(
            failed_sequences, successful_sequences, sample_epistemic_states
        )
        
        assert isinstance(result, FailureAnalysisResult)
        assert len(result.failure_modes) >= 0  # May be 0 with small dataset
        assert len(result.root_causes) >= 0
        assert isinstance(result.failure_clusters, dict)
        assert isinstance(result.risk_factors, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.prevention_strategies, list)
    
    @pytest.mark.asyncio
    async def test_pattern_detection_with_real_clustering(self, failed_sequences):
        """Test pattern detection with real clustering algorithms."""
        detector = FailurePatternDetector(min_cluster_size=2, eps=0.5)
        
        patterns = await detector.detect_failure_patterns(failed_sequences)
        
        assert isinstance(patterns, dict)
        # With real clustering, results may vary based on data
        
        if patterns:  # If patterns were found
            for cluster_name, sequences in patterns.items():
                assert isinstance(sequences, list)
                assert all(isinstance(seq, ExecutionSequence) for seq in sequences)
    
    @pytest.mark.asyncio
    async def test_failure_signature_analysis(self, failed_sequences):
        """Test failure signature analysis with real data."""
        detector = FailurePatternDetector()
        
        # Group sequences by error type manually for testing
        timeout_sequences = [seq for seq in failed_sequences 
                           if any('timeout' in step.error_message.lower() 
                                 for step in seq.steps if step.error_message)]
        memory_sequences = [seq for seq in failed_sequences 
                          if any('memory' in step.error_message.lower() 
                                for step in seq.steps if step.error_message)]
        
        failure_clusters = {}
        if timeout_sequences:
            failure_clusters['timeout_failures'] = timeout_sequences
        if memory_sequences:
            failure_clusters['memory_failures'] = memory_sequences
        
        if failure_clusters:
            signatures = await detector.identify_failure_signatures(failure_clusters)
            
            assert isinstance(signatures, dict)
            
            for cluster_name, signature in signatures.items():
                assert 'cluster_size' in signature
                assert 'common_errors' in signature
                assert 'timing_stats' in signature
                assert signature['cluster_size'] > 0
    
    @pytest.mark.asyncio
    async def test_root_cause_analysis_with_real_data(self, failed_sequences, successful_sequences, 
                                                     sample_epistemic_states):
        """Test root cause analysis with real decision tree."""
        analyzer = RootCauseAnalyzer()
        
        # Ensure we have enough data for analysis
        if len(failed_sequences) >= 3 and len(successful_sequences) >= 3:
            root_causes = await analyzer.analyze_root_causes(
                failed_sequences, successful_sequences, sample_epistemic_states
            )
            
            assert isinstance(root_causes, list)
            # May be empty if decision tree doesn't find clear patterns
            
            for cause in root_causes:
                assert isinstance(cause, RootCause)
                assert cause.confidence >= 0.0
                assert cause.confidence <= 1.0
                assert isinstance(cause.evidence, list)
                assert isinstance(cause.contributing_factors, list)