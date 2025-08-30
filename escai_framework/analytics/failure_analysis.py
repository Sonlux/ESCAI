"""
Failure analysis module for identifying and analyzing agent failure patterns.

This module provides comprehensive failure analysis capabilities including
failure mode identification, root cause analysis, and failure prediction.
"""

import asyncio
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
import networkx as nx

from ..models.behavioral_pattern import BehavioralPattern, ExecutionSequence, ExecutionStep
from ..models.epistemic_state import EpistemicState
from ..models.causal_relationship import CausalRelationship


@dataclass
class FailureMode:
    """Represents an identified failure mode."""
    failure_id: str
    failure_name: str
    description: str
    frequency: int
    severity: float
    common_triggers: List[str]
    failure_patterns: List[str]
    recovery_strategies: List[str]
    prevention_measures: List[str]
    statistical_significance: float


@dataclass
class RootCause:
    """Represents a root cause of failure."""
    cause_id: str
    cause_description: str
    confidence: float
    evidence: List[str]
    contributing_factors: List[str]
    causal_chain: List[str]


@dataclass
class FailureAnalysisResult:
    """Complete failure analysis result."""
    failure_modes: List[FailureMode]
    root_causes: List[RootCause]
    failure_clusters: Dict[str, List[ExecutionSequence]]
    risk_factors: List[str]
    recommendations: List[str]
    prevention_strategies: List[str]


class FailurePatternDetector:
    """
    Detects patterns in agent failures using machine learning techniques.
    
    Identifies common failure modes, clusters similar failures,
    and extracts failure signatures.
    """
    
    def __init__(self, min_cluster_size: int = 3, eps: float = 0.5):
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        self.scaler = StandardScaler()
        self.clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    async def detect_failure_patterns(self, failed_sequences: List[ExecutionSequence]) -> Dict[str, List[ExecutionSequence]]:
        """
        Detect patterns in failed execution sequences.
        
        Args:
            failed_sequences: List of failed execution sequences
            
        Returns:
            Dictionary mapping pattern names to sequences
        """
        if not failed_sequences:
            return {}
        
        # Extract features from failed sequences
        features = self._extract_failure_features(failed_sequences)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Cluster failures
        cluster_labels = self.clusterer.fit_predict(features_scaled)
        
        # Group sequences by cluster
        failure_clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # -1 indicates noise in DBSCAN
                failure_clusters[f"failure_pattern_{label}"].append(failed_sequences[i])
            else:
                failure_clusters["anomalous_failures"].append(failed_sequences[i])
        
        return dict(failure_clusters)
    
    async def identify_failure_signatures(self, failure_clusters: Dict[str, List[ExecutionSequence]]) -> Dict[str, Dict[str, Any]]:
        """
        Identify characteristic signatures of each failure pattern.
        
        Args:
            failure_clusters: Clustered failure sequences
            
        Returns:
            Dictionary mapping cluster names to their signatures
        """
        signatures = {}
        
        for cluster_name, sequences in failure_clusters.items():
            signature = await self._extract_cluster_signature(sequences)
            signatures[cluster_name] = signature
        
        return signatures
    
    def _extract_failure_features(self, sequences: List[ExecutionSequence]) -> np.ndarray:
        """Extract numerical features from failed sequences."""
        features = []
        
        for seq in sequences:
            # Sequence-level features
            seq_features = [
                len(seq.steps),
                seq.total_duration_ms,
                seq.success_rate,
                sum(1 for step in seq.steps if step.error_message),  # Error count
                np.mean([step.duration for step in seq.steps]) if seq.steps else 0,
                np.std([step.duration for step in seq.steps]) if len(seq.steps) > 1 else 0,
            ]
            
            # Step-type distribution
            step_types = [step.step_type for step in seq.steps]
            type_counts = Counter(step_types)
            common_types = ['reasoning', 'action', 'observation', 'planning', 'execution']
            for step_type in common_types:
                seq_features.append(type_counts.get(step_type, 0))
            
            # Error pattern features
            error_positions = [i for i, step in enumerate(seq.steps) if step.error_message]
            if error_positions:
                seq_features.extend([
                    min(error_positions),  # First error position
                    max(error_positions),  # Last error position
                    len(error_positions),  # Total errors
                    np.mean(error_positions) if error_positions else 0  # Average error position
                ])
            else:
                seq_features.extend([0, 0, 0, 0])
            
            features.append(seq_features)
        
        return np.array(features)
    
    async def _extract_cluster_signature(self, sequences: List[ExecutionSequence]) -> Dict[str, Any]:
        """Extract characteristic signature of a failure cluster."""
        if not sequences:
            return {}
        
        # Common error messages
        error_messages = []
        for seq in sequences:
            for step in seq.steps:
                if step.error_message:
                    error_messages.append(step.error_message)
        
        common_errors = Counter(error_messages).most_common(5)
        
        # Common step patterns
        step_patterns = []
        for seq in sequences:
            pattern = [step.step_type for step in seq.steps]
            step_patterns.append(tuple(pattern))
        
        common_patterns = Counter(step_patterns).most_common(3)
        
        # Timing characteristics
        durations = [seq.total_duration_ms for seq in sequences]
        step_counts = [len(seq.steps) for seq in sequences]
        
        signature = {
            'cluster_size': len(sequences),
            'common_errors': [{'error': error, 'frequency': freq} for error, freq in common_errors],
            'common_patterns': [{'pattern': list(pattern), 'frequency': freq} for pattern, freq in common_patterns],
            'timing_stats': {
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'mean_steps': np.mean(step_counts),
                'std_steps': np.std(step_counts)
            },
            'failure_characteristics': self._analyze_failure_characteristics(sequences)
        }
        
        return signature
    
    def _analyze_failure_characteristics(self, sequences: List[ExecutionSequence]) -> Dict[str, Any]:
        """Analyze specific characteristics of failures in the cluster."""
        characteristics = {
            'early_failures': 0,
            'late_failures': 0,
            'timeout_failures': 0,
            'resource_failures': 0,
            'logic_failures': 0
        }
        
        for seq in sequences:
            # Analyze failure timing
            total_steps = len(seq.steps)
            first_error_step = next((i for i, step in enumerate(seq.steps) if step.error_message), total_steps)
            
            if first_error_step < total_steps * 0.3:
                characteristics['early_failures'] += 1
            elif first_error_step > total_steps * 0.7:
                characteristics['late_failures'] += 1
            
            # Analyze failure types based on error messages
            for step in seq.steps:
                if step.error_message:
                    error_lower = step.error_message.lower()
                    if 'timeout' in error_lower or 'time' in error_lower:
                        characteristics['timeout_failures'] += 1
                    elif 'memory' in error_lower or 'resource' in error_lower:
                        characteristics['resource_failures'] += 1
                    elif 'logic' in error_lower or 'assertion' in error_lower:
                        characteristics['logic_failures'] += 1
        
        return characteristics


class RootCauseAnalyzer:
    """
    Analyzes root causes of agent failures using causal inference
    and decision tree analysis.
    """
    
    def __init__(self):
        self.decision_tree = DecisionTreeClassifier(
            max_depth=10, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.causal_graph = nx.DiGraph()
    
    async def analyze_root_causes(self, failed_sequences: List[ExecutionSequence],
                                successful_sequences: List[ExecutionSequence],
                                epistemic_states: List[EpistemicState]) -> List[RootCause]:
        """
        Analyze root causes of failures by comparing with successful sequences.
        
        Args:
            failed_sequences: Failed execution sequences
            successful_sequences: Successful execution sequences
            epistemic_states: Corresponding epistemic states
            
        Returns:
            List of identified root causes
        """
        # Prepare data for analysis
        X, y = self._prepare_causal_data(failed_sequences, successful_sequences, epistemic_states)
        
        # Train decision tree to identify discriminating factors
        self.decision_tree.fit(X, y)
        
        # Extract decision rules
        decision_rules = self._extract_decision_rules()
        
        # Build causal graph
        await self._build_causal_graph(failed_sequences, epistemic_states)
        
        # Identify root causes
        root_causes = await self._identify_root_causes(decision_rules)
        
        return root_causes
    
    def _prepare_causal_data(self, failed_sequences: List[ExecutionSequence],
                           successful_sequences: List[ExecutionSequence],
                           epistemic_states: List[EpistemicState]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for causal analysis."""
        features = []
        labels = []
        
        # Process failed sequences
        for i, seq in enumerate(failed_sequences):
            if i < len(epistemic_states):
                feature_vector = self._extract_causal_features(seq, epistemic_states[i])
                features.append(feature_vector)
                labels.append(0)  # Failed
        
        # Process successful sequences
        success_states = epistemic_states[len(failed_sequences):]
        for i, seq in enumerate(successful_sequences):
            if i < len(success_states):
                feature_vector = self._extract_causal_features(seq, success_states[i])
                features.append(feature_vector)
                labels.append(1)  # Successful
        
        return np.array(features), np.array(labels)
    
    def _extract_causal_features(self, sequence: ExecutionSequence, 
                               epistemic_state: EpistemicState) -> List[float]:
        """Extract features for causal analysis."""
        features = [
            # Sequence features
            float(len(sequence.steps)),
            float(sequence.total_duration_ms),
            float(sequence.success_rate),
            float(sum(1 for step in sequence.steps if step.error_message)),
            
            # Epistemic state features
            float(epistemic_state.confidence_level),
            float(epistemic_state.uncertainty_score),
            float(len(epistemic_state.belief_states)),
            float(len(epistemic_state.goal_states) if epistemic_state.goal_states else 0),
            
            # Timing features
            float(np.mean([step.duration for step in sequence.steps]) if sequence.steps else 0),
            float(np.max([step.duration for step in sequence.steps]) if sequence.steps else 0),
            
            # Complexity features
            float(len(set(step.step_type for step in sequence.steps))),
            float(np.mean([step.success_probability for step in sequence.steps]) if sequence.steps else 0)
        ]
        
        return features
    
    def _extract_decision_rules(self) -> List[Dict[str, Any]]:
        """Extract interpretable rules from decision tree."""
        tree = self.decision_tree.tree_
        feature_names = [f"feature_{i}" for i in range(tree.n_features)]
        
        def recurse(node, depth=0):
            rules = []
            if tree.feature[node] != -2:  # Not a leaf
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                # Left child (condition is true)
                left_rules = recurse(tree.children_left[node], depth + 1)
                for rule in left_rules:
                    rule['conditions'].append(f"{feature} <= {threshold:.3f}")
                
                # Right child (condition is false)
                right_rules = recurse(tree.children_right[node], depth + 1)
                for rule in right_rules:
                    rule['conditions'].append(f"{feature} > {threshold:.3f}")
                
                rules.extend(left_rules)
                rules.extend(right_rules)
            else:
                # Leaf node
                value = tree.value[node][0]
                prediction = 1 if value[1] > value[0] else 0
                confidence = max(value) / sum(value)
                
                rules.append({
                    'conditions': [],
                    'prediction': prediction,
                    'confidence': confidence,
                    'samples': tree.n_node_samples[node]
                })
            
            return rules
        
        return recurse(0)
    
    async def _build_causal_graph(self, failed_sequences: List[ExecutionSequence],
                                epistemic_states: List[EpistemicState]):
        """Build causal graph from failure data."""
        self.causal_graph.clear()
        
        # Add nodes for different types of factors
        factor_types = [
            'low_confidence', 'high_uncertainty', 'complex_execution',
            'resource_constraints', 'timing_issues', 'logic_errors'
        ]
        
        for factor in factor_types:
            self.causal_graph.add_node(factor)
        
        # Add edges based on observed patterns
        for seq, state in zip(failed_sequences, epistemic_states):
            # Analyze causal relationships
            if state.confidence_level < 0.3:
                self.causal_graph.add_edge('low_confidence', 'failure')
            
            if state.uncertainty_score > 0.7:
                self.causal_graph.add_edge('high_uncertainty', 'failure')
            
            if len(seq.steps) > 20:
                self.causal_graph.add_edge('complex_execution', 'failure')
            
            # Add more causal relationships based on analysis
    
    async def _identify_root_causes(self, decision_rules: List[Dict[str, Any]]) -> List[RootCause]:
        """Identify root causes from decision rules and causal graph."""
        root_causes = []
        
        # Analyze decision rules for failure patterns
        failure_rules = [rule for rule in decision_rules if rule['prediction'] == 0 and rule['confidence'] > 0.7]
        
        for i, rule in enumerate(failure_rules):
            # Extract root cause from rule conditions
            cause_description = self._interpret_rule_conditions(rule['conditions'])
            
            root_cause = RootCause(
                cause_id=f"root_cause_{i}",
                cause_description=cause_description,
                confidence=rule['confidence'],
                evidence=rule['conditions'],
                contributing_factors=self._extract_contributing_factors(rule['conditions']),
                causal_chain=self._trace_causal_chain(cause_description)
            )
            
            root_causes.append(root_cause)
        
        return root_causes
    
    def _interpret_rule_conditions(self, conditions: List[str]) -> str:
        """Interpret decision rule conditions into human-readable description."""
        interpretations = []
        
        for condition in conditions:
            if 'feature_4' in condition and '<=' in condition:  # Confidence level
                interpretations.append("Low agent confidence")
            elif 'feature_5' in condition and '>' in condition:  # Uncertainty score
                interpretations.append("High uncertainty")
            elif 'feature_0' in condition and '>' in condition:  # Number of steps
                interpretations.append("Complex execution sequence")
            elif 'feature_3' in condition and '>' in condition:  # Error count
                interpretations.append("Multiple errors during execution")
        
        if interpretations:
            return "; ".join(interpretations)
        else:
            return "Complex interaction of factors"
    
    def _extract_contributing_factors(self, conditions: List[str]) -> List[str]:
        """Extract contributing factors from rule conditions."""
        factors = []
        
        for condition in conditions:
            if 'feature_4' in condition:
                factors.append("confidence_level")
            elif 'feature_5' in condition:
                factors.append("uncertainty_score")
            elif 'feature_0' in condition:
                factors.append("sequence_complexity")
            elif 'feature_3' in condition:
                factors.append("error_frequency")
        
        return factors
    
    def _trace_causal_chain(self, cause_description: str) -> List[str]:
        """Trace causal chain from root cause to failure."""
        # Simplified causal chain tracing
        if "Low agent confidence" in cause_description:
            return ["Low confidence", "Poor decision making", "Execution errors", "Task failure"]
        elif "High uncertainty" in cause_description:
            return ["High uncertainty", "Hesitant actions", "Suboptimal choices", "Task failure"]
        elif "Complex execution" in cause_description:
            return ["Complex sequence", "Increased error probability", "Cascading failures", "Task failure"]
        else:
            return ["Unknown root cause", "Intermediate factors", "Task failure"]


class FailureAnalysisEngine:
    """
    Main engine for comprehensive failure analysis.
    
    Coordinates failure pattern detection, root cause analysis,
    and generates actionable recommendations.
    """
    
    def __init__(self):
        self.pattern_detector = FailurePatternDetector()
        self.root_cause_analyzer = RootCauseAnalyzer()
        
    async def analyze_failures(self, failed_sequences: List[ExecutionSequence],
                             successful_sequences: List[ExecutionSequence],
                             epistemic_states: List[EpistemicState]) -> FailureAnalysisResult:
        """
        Perform comprehensive failure analysis.
        
        Args:
            failed_sequences: Failed execution sequences
            successful_sequences: Successful execution sequences
            epistemic_states: Corresponding epistemic states
            
        Returns:
            Complete failure analysis result
        """
        # Detect failure patterns
        failure_clusters = await self.pattern_detector.detect_failure_patterns(failed_sequences)
        
        # Identify failure signatures
        failure_signatures = await self.pattern_detector.identify_failure_signatures(failure_clusters)
        
        # Analyze root causes
        root_causes = await self.root_cause_analyzer.analyze_root_causes(
            failed_sequences, successful_sequences, epistemic_states
        )
        
        # Create failure modes from clusters
        failure_modes = await self._create_failure_modes(failure_clusters, failure_signatures)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(failure_modes, root_causes)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(failure_modes, root_causes)
        
        # Generate prevention strategies
        prevention_strategies = self._generate_prevention_strategies(root_causes)
        
        return FailureAnalysisResult(
            failure_modes=failure_modes,
            root_causes=root_causes,
            failure_clusters=failure_clusters,
            risk_factors=risk_factors,
            recommendations=recommendations,
            prevention_strategies=prevention_strategies
        )
    
    async def _create_failure_modes(self, failure_clusters: Dict[str, List[ExecutionSequence]],
                                  failure_signatures: Dict[str, Dict[str, Any]]) -> List[FailureMode]:
        """Create failure modes from detected clusters."""
        failure_modes = []
        
        for cluster_name, sequences in failure_clusters.items():
            if cluster_name in failure_signatures:
                signature = failure_signatures[cluster_name]
                
                # Calculate severity based on impact
                severity = self._calculate_failure_severity(sequences)
                
                # Extract common triggers
                common_triggers = self._extract_common_triggers(sequences)
                
                failure_mode = FailureMode(
                    failure_id=cluster_name,
                    failure_name=self._generate_failure_name(cluster_name, signature),
                    description=self._generate_failure_description(signature),
                    frequency=len(sequences),
                    severity=severity,
                    common_triggers=common_triggers,
                    failure_patterns=self._extract_failure_patterns(signature),
                    recovery_strategies=self._suggest_recovery_strategies(signature),
                    prevention_measures=self._suggest_prevention_measures(signature),
                    statistical_significance=self._calculate_statistical_significance(sequences)
                )
                
                failure_modes.append(failure_mode)
        
        return failure_modes
    
    def _calculate_failure_severity(self, sequences: List[ExecutionSequence]) -> float:
        """Calculate severity score for failure mode."""
        # Consider factors like duration, complexity, and impact
        avg_duration = np.mean([seq.total_duration_ms for seq in sequences])
        avg_steps = np.mean([len(seq.steps) for seq in sequences])
        
        # Normalize to 0-1 scale
        duration_score = min(1.0, float(avg_duration) / 3600)  # Assume 1 hour is high
        complexity_score = min(1.0, float(avg_steps) / 50)     # Assume 50 steps is high
        
        return (duration_score + complexity_score) / 2
    
    def _extract_common_triggers(self, sequences: List[ExecutionSequence]) -> List[str]:
        """Extract common triggers that lead to this failure mode."""
        triggers = []
        
        for seq in sequences:
            if seq.steps:
                first_step = seq.steps[0]
                triggers.append(f"{first_step.step_type}:{first_step.action}")
        
        # Return most common triggers
        trigger_counts = Counter(triggers)
        return [trigger for trigger, _ in trigger_counts.most_common(5)]
    
    def _generate_failure_name(self, cluster_name: str, signature: Dict[str, Any]) -> str:
        """Generate human-readable name for failure mode."""
        if 'common_errors' in signature and signature['common_errors']:
            main_error = signature['common_errors'][0]['error']
            if 'timeout' in main_error.lower():
                return "Timeout Failure Mode"
            elif 'memory' in main_error.lower():
                return "Resource Exhaustion Failure Mode"
            elif 'logic' in main_error.lower():
                return "Logic Error Failure Mode"
        
        return f"Failure Mode {cluster_name.split('_')[-1]}"
    
    def _generate_failure_description(self, signature: Dict[str, Any]) -> str:
        """Generate description for failure mode."""
        description_parts = []
        
        if 'timing_stats' in signature:
            avg_duration = signature['timing_stats']['mean_duration']
            description_parts.append(f"Average failure time: {avg_duration:.1f}s")
        
        if 'common_errors' in signature and signature['common_errors']:
            main_error = signature['common_errors'][0]['error']
            description_parts.append(f"Primary error: {main_error}")
        
        if 'failure_characteristics' in signature:
            chars = signature['failure_characteristics']
            if chars['early_failures'] > chars['late_failures']:
                description_parts.append("Typically fails early in execution")
            else:
                description_parts.append("Typically fails late in execution")
        
        return "; ".join(description_parts)
    
    def _extract_failure_patterns(self, signature: Dict[str, Any]) -> List[str]:
        """Extract failure patterns from signature."""
        patterns = []
        
        if 'common_patterns' in signature:
            for pattern_info in signature['common_patterns']:
                pattern_str = " -> ".join(pattern_info['pattern'])
                patterns.append(f"{pattern_str} (freq: {pattern_info['frequency']})")
        
        return patterns
    
    def _suggest_recovery_strategies(self, signature: Dict[str, Any]) -> List[str]:
        """Suggest recovery strategies based on failure signature."""
        strategies = []
        
        if 'common_errors' in signature:
            for error_info in signature['common_errors']:
                error = error_info['error'].lower()
                if 'timeout' in error:
                    strategies.append("Increase timeout limits")
                    strategies.append("Implement checkpoint/resume mechanism")
                elif 'memory' in error:
                    strategies.append("Implement memory cleanup")
                    strategies.append("Use streaming processing")
                elif 'connection' in error:
                    strategies.append("Implement retry with exponential backoff")
                    strategies.append("Use circuit breaker pattern")
        
        if not strategies:
            strategies.append("Implement generic error recovery")
            strategies.append("Add fallback mechanisms")
        
        return strategies
    
    def _suggest_prevention_measures(self, signature: Dict[str, Any]) -> List[str]:
        """Suggest prevention measures based on failure signature."""
        measures = []
        
        if 'failure_characteristics' in signature:
            chars = signature['failure_characteristics']
            
            if chars['timeout_failures'] > 0:
                measures.append("Implement proactive timeout monitoring")
                measures.append("Optimize slow operations")
            
            if chars['resource_failures'] > 0:
                measures.append("Implement resource monitoring")
                measures.append("Add resource usage limits")
            
            if chars['early_failures'] > chars['late_failures']:
                measures.append("Improve input validation")
                measures.append("Add pre-execution checks")
        
        return measures
    
    def _calculate_statistical_significance(self, sequences: List[ExecutionSequence]) -> float:
        """Calculate statistical significance of failure mode."""
        # Simple significance based on frequency and consistency
        frequency = len(sequences)
        
        # Check consistency of failure patterns
        error_messages = []
        for seq in sequences:
            for step in seq.steps:
                if step.error_message:
                    error_messages.append(step.error_message)
        
        if error_messages:
            most_common_error_freq = Counter(error_messages).most_common(1)[0][1]
            consistency = most_common_error_freq / len(error_messages)
        else:
            consistency = 0.0
        
        # Combine frequency and consistency
        significance = min(1.0, (frequency / 10) * consistency)
        return significance
    
    def _identify_risk_factors(self, failure_modes: List[FailureMode], 
                             root_causes: List[RootCause]) -> List[str]:
        """Identify overall risk factors from failure analysis."""
        risk_factors = set()
        
        # Extract from failure modes
        for mode in failure_modes:
            risk_factors.update(mode.common_triggers)
        
        # Extract from root causes
        for cause in root_causes:
            risk_factors.update(cause.contributing_factors)
        
        return list(risk_factors)
    
    def _generate_recommendations(self, failure_modes: List[FailureMode], 
                                root_causes: List[RootCause]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # High-frequency failure modes
        high_freq_modes = [mode for mode in failure_modes if mode.frequency > 5]
        if high_freq_modes:
            recommendations.append("Focus on addressing high-frequency failure modes first")
        
        # High-severity failure modes
        high_sev_modes = [mode for mode in failure_modes if mode.severity > 0.7]
        if high_sev_modes:
            recommendations.append("Implement immediate mitigation for high-severity failures")
        
        # Root cause patterns
        confidence_issues = [cause for cause in root_causes if 'confidence' in cause.cause_description.lower()]
        if confidence_issues:
            recommendations.append("Improve agent confidence calibration")
        
        uncertainty_issues = [cause for cause in root_causes if 'uncertainty' in cause.cause_description.lower()]
        if uncertainty_issues:
            recommendations.append("Implement uncertainty reduction strategies")
        
        return recommendations
    
    def _generate_prevention_strategies(self, root_causes: List[RootCause]) -> List[str]:
        """Generate prevention strategies based on root causes."""
        strategies = []
        
        for cause in root_causes:
            if cause.confidence > 0.8:  # High confidence root causes
                if 'confidence' in cause.cause_description.lower():
                    strategies.append("Implement confidence monitoring and calibration")
                elif 'uncertainty' in cause.cause_description.lower():
                    strategies.append("Add uncertainty quantification and management")
                elif 'complex' in cause.cause_description.lower():
                    strategies.append("Implement complexity reduction techniques")
        
        # Generic strategies
        strategies.extend([
            "Implement comprehensive monitoring and alerting",
            "Add automated failure detection and recovery",
            "Establish failure analysis feedback loops"
        ])
        
        return list(set(strategies))  # Remove duplicates