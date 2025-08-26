"""
Automated test data generation utilities for consistent testing.
Provides synthetic data generators for all ESCAI components.
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import uuid

from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState
from escai_framework.models.behavioral_pattern import BehavioralPattern, ExecutionSequence
from escai_framework.models.causal_relationship import CausalRelationship
from escai_framework.models.prediction_result import PredictionResult
from escai_framework.instrumentation.events import AgentEvent, EventType


class TestDataGenerator:
    """Comprehensive test data generator for ESCAI Framework."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Predefined vocabularies for realistic data generation
        self.agent_types = ["data_analyst", "web_scraper", "api_integrator", "ml_engineer", "researcher"]
        self.task_types = ["analysis", "extraction", "integration", "modeling", "research"]
        self.error_types = ["NetworkError", "ValidationError", "TimeoutError", "AuthError", "DataError"]
        self.decision_types = ["retry", "fallback", "optimize", "escalate", "abort"]
        
        self.belief_templates = [
            "The user wants to {action} {object}",
            "The data quality is {quality}",
            "The current approach is {effectiveness}",
            "The system performance is {performance}",
            "The task complexity is {complexity}"
        ]
        
        self.knowledge_facts = [
            "Data is stored in {format} format",
            "API rate limit is {limit} requests per minute",
            "Processing time scales with {factor}",
            "Error rate increases with {condition}",
            "Success depends on {dependency}"
        ]
        
        self.goal_templates = [
            "Complete {task} within {timeframe}",
            "Achieve {metric} accuracy",
            "Process {quantity} items",
            "Optimize {parameter} performance",
            "Reduce {issue} occurrence"
        ]
    
    def generate_agent_events(
        self, 
        agent_id: str, 
        count: int = 100,
        scenario: str = "mixed",
        time_span_hours: int = 24
    ) -> List[AgentEvent]:
        """Generate a sequence of realistic agent events."""
        events = []
        start_time = datetime.now() - timedelta(hours=time_span_hours)
        
        # Define scenario-specific event patterns
        if scenario == "data_analysis":
            event_pattern = [
                (EventType.TASK_START, 0.1),
                (EventType.DECISION_MADE, 0.3),
                (EventType.DATA_LOADED, 0.1),
                (EventType.ANALYSIS_STARTED, 0.1),
                (EventType.INSIGHT_GENERATED, 0.15),
                (EventType.VISUALIZATION_CREATED, 0.1),
                (EventType.ERROR_OCCURRED, 0.05),
                (EventType.TASK_COMPLETE, 0.1)
            ]
        elif scenario == "web_scraping":
            event_pattern = [
                (EventType.TASK_START, 0.1),
                (EventType.CONNECTION_ESTABLISHED, 0.1),
                (EventType.DATA_EXTRACTED, 0.3),
                (EventType.DATA_VALIDATION, 0.2),
                (EventType.ERROR_OCCURRED, 0.1),
                (EventType.DECISION_MADE, 0.1),
                (EventType.TASK_COMPLETE, 0.1)
            ]
        else:  # mixed scenario
            event_pattern = [(event_type, 1.0/len(EventType)) for event_type in EventType]
        
        for i in range(count):
            # Select event type based on scenario pattern
            event_type = np.random.choice(
                [et for et, _ in event_pattern],
                p=[prob for _, prob in event_pattern]
            )
            
            # Generate timestamp with some clustering
            if i == 0:
                timestamp = start_time
            else:
                # Events tend to cluster in time
                time_delta = np.random.exponential(time_span_hours * 3600 / count)
                timestamp = events[-1].timestamp + timedelta(seconds=time_delta)
            
            # Generate event-specific data
            event_data = self._generate_event_data(event_type, i)
            
            event = AgentEvent(
                event_id=f"{agent_id}_event_{i:04d}",
                agent_id=agent_id,
                event_type=event_type,
                timestamp=timestamp,
                data=event_data
            )
            
            events.append(event)
        
        return events
    
    def _generate_event_data(self, event_type: EventType, sequence_num: int) -> Dict[str, Any]:
        """Generate realistic data for specific event types."""
        
        if event_type == EventType.TASK_START:
            return {
                "task_id": f"task_{sequence_num}",
                "task_type": np.random.choice(self.task_types),
                "priority": np.random.choice(["low", "medium", "high"], p=[0.3, 0.5, 0.2]),
                "estimated_duration": np.random.randint(60, 3600),
                "complexity": np.random.uniform(0.1, 1.0)
            }
        
        elif event_type == EventType.DECISION_MADE:
            return {
                "decision": np.random.choice(self.decision_types),
                "confidence": np.random.uniform(0.5, 1.0),
                "reasoning": f"Based on {np.random.choice(['performance', 'accuracy', 'efficiency', 'reliability'])} considerations",
                "alternatives_considered": np.random.randint(1, 5),
                "decision_time_ms": np.random.randint(10, 1000)
            }
        
        elif event_type == EventType.ERROR_OCCURRED:
            return {
                "error_type": np.random.choice(self.error_types),
                "error_message": f"Error in {np.random.choice(['processing', 'validation', 'connection', 'authentication'])}",
                "severity": np.random.choice(["warning", "error", "critical"], p=[0.6, 0.3, 0.1]),
                "retry_count": np.random.randint(0, 3),
                "error_code": np.random.randint(400, 599)
            }
        
        elif event_type == EventType.DATA_LOADED:
            return {
                "data_source": np.random.choice(["file", "database", "api", "stream"]),
                "records_count": np.random.randint(100, 100000),
                "data_size_mb": np.random.uniform(0.1, 1000),
                "load_time_ms": np.random.randint(100, 10000),
                "data_quality_score": np.random.uniform(0.7, 1.0)
            }
        
        elif event_type == EventType.ANALYSIS_STARTED:
            return {
                "analysis_type": np.random.choice(["descriptive", "predictive", "prescriptive", "diagnostic"]),
                "variables": [f"var_{i}" for i in range(np.random.randint(2, 10))],
                "method": np.random.choice(["regression", "classification", "clustering", "correlation"]),
                "expected_duration": np.random.randint(60, 1800)
            }
        
        elif event_type == EventType.INSIGHT_GENERATED:
            return {
                "insight_type": np.random.choice(["trend", "anomaly", "correlation", "prediction"]),
                "confidence": np.random.uniform(0.6, 0.95),
                "statistical_significance": np.random.uniform(0.8, 0.99),
                "insight_text": f"Discovered {np.random.choice(['positive', 'negative', 'neutral'])} correlation",
                "actionable": np.random.choice([True, False], p=[0.7, 0.3])
            }
        
        else:
            # Generic data for other event types
            return {
                "sequence_number": sequence_num,
                "timestamp_ms": int(datetime.now().timestamp() * 1000),
                "random_value": np.random.uniform(0, 1),
                "category": np.random.choice(["A", "B", "C"])
            }
    
    def generate_epistemic_states(
        self, 
        agent_id: str, 
        count: int = 50,
        evolution_pattern: str = "learning"
    ) -> List[EpistemicState]:
        """Generate a sequence of evolving epistemic states."""
        states = []
        base_time = datetime.now() - timedelta(hours=count)
        
        # Initialize base knowledge and goals
        base_knowledge = self._generate_knowledge_state()
        base_goals = self._generate_goal_state()
        
        for i in range(count):
            timestamp = base_time + timedelta(hours=i)
            
            # Generate beliefs that evolve over time
            beliefs = self._generate_belief_states(i, evolution_pattern)
            
            # Evolve knowledge (accumulates over time)
            knowledge = self._evolve_knowledge_state(base_knowledge, i, evolution_pattern)
            
            # Evolve goals (progress increases)
            goals = self._evolve_goal_state(base_goals, i / count)
            
            # Calculate confidence and uncertainty based on evolution
            if evolution_pattern == "learning":
                confidence = min(0.5 + (i / count) * 0.4, 0.9)
                uncertainty = max(0.5 - (i / count) * 0.3, 0.1)
            elif evolution_pattern == "struggling":
                confidence = max(0.8 - (i / count) * 0.3, 0.4)
                uncertainty = min(0.2 + (i / count) * 0.4, 0.6)
            else:  # stable
                confidence = 0.7 + np.random.uniform(-0.1, 0.1)
                uncertainty = 0.3 + np.random.uniform(-0.1, 0.1)
            
            state = EpistemicState(
                agent_id=agent_id,
                timestamp=timestamp,
                belief_states=beliefs,
                knowledge_state=knowledge,
                goal_state=goals,
                confidence_level=confidence,
                uncertainty_score=uncertainty,
                decision_context={
                    "phase": f"phase_{i // 10}",
                    "complexity": np.random.uniform(0.3, 0.9),
                    "time_pressure": np.random.uniform(0.1, 0.8)
                }
            )
            
            states.append(state)
        
        return states
    
    def _generate_belief_states(self, sequence: int, pattern: str) -> List[BeliefState]:
        """Generate belief states for a given sequence."""
        beliefs = []
        num_beliefs = np.random.randint(2, 6)
        
        for i in range(num_beliefs):
            # Generate belief content from templates
            template = np.random.choice(self.belief_templates)
            content = template.format(
                action=np.random.choice(["analyze", "process", "extract", "validate"]),
                object=np.random.choice(["data", "information", "results", "patterns"]),
                quality=np.random.choice(["high", "medium", "low", "excellent"]),
                effectiveness=np.random.choice(["working", "suboptimal", "failing", "excellent"]),
                performance=np.random.choice(["good", "poor", "acceptable", "outstanding"]),
                complexity=np.random.choice(["low", "medium", "high", "very high"])
            )
            
            # Confidence evolves based on pattern
            if pattern == "learning":
                confidence = min(0.4 + (sequence / 50) * 0.5, 0.9)
            elif pattern == "struggling":
                confidence = max(0.8 - (sequence / 50) * 0.4, 0.3)
            else:
                confidence = np.random.uniform(0.5, 0.9)
            
            belief = BeliefState(
                belief_id=f"belief_{sequence}_{i}",
                content=content,
                confidence=confidence,
                timestamp=datetime.now(),
                evidence=[f"evidence_{j}" for j in range(np.random.randint(1, 4))]
            )
            
            beliefs.append(belief)
        
        return beliefs
    
    def _generate_knowledge_state(self) -> KnowledgeState:
        """Generate initial knowledge state."""
        facts = []
        for _ in range(np.random.randint(3, 8)):
            template = np.random.choice(self.knowledge_facts)
            fact = template.format(
                format=np.random.choice(["CSV", "JSON", "XML", "Parquet"]),
                limit=np.random.randint(100, 10000),
                factor=np.random.choice(["data size", "complexity", "network latency"]),
                condition=np.random.choice(["high load", "poor connectivity", "invalid data"]),
                dependency=np.random.choice(["data quality", "system resources", "network stability"])
            )
            facts.append(fact)
        
        concepts = [
            "data processing", "machine learning", "statistical analysis",
            "pattern recognition", "anomaly detection", "performance optimization"
        ]
        
        relationships = {
            "data": ["quality", "volume", "velocity"],
            "performance": ["latency", "throughput", "accuracy"],
            "errors": ["network", "validation", "processing"]
        }
        
        return KnowledgeState(
            facts=facts,
            concepts=np.random.choice(concepts, np.random.randint(2, 5)).tolist(),
            relationships=relationships,
            timestamp=datetime.now()
        )
    
    def _evolve_knowledge_state(self, base_knowledge: KnowledgeState, sequence: int, pattern: str) -> KnowledgeState:
        """Evolve knowledge state over time."""
        # Add new facts occasionally
        new_facts = base_knowledge.facts.copy()
        if sequence > 0 and np.random.random() < 0.3:
            new_fact = f"Learned fact at sequence {sequence}: {np.random.choice(['optimization works', 'pattern identified', 'correlation found'])}"
            new_facts.append(new_fact)
        
        # Expand concepts
        new_concepts = base_knowledge.concepts.copy()
        if sequence > 0 and np.random.random() < 0.2:
            additional_concepts = ["deep learning", "reinforcement learning", "natural language processing"]
            new_concept = np.random.choice(additional_concepts)
            if new_concept not in new_concepts:
                new_concepts.append(new_concept)
        
        return KnowledgeState(
            facts=new_facts,
            concepts=new_concepts,
            relationships=base_knowledge.relationships,
            timestamp=datetime.now()
        )
    
    def _generate_goal_state(self) -> GoalState:
        """Generate initial goal state."""
        template = np.random.choice(self.goal_templates)
        primary_goal = template.format(
            task=np.random.choice(["analysis", "processing", "extraction", "modeling"]),
            timeframe=np.random.choice(["1 hour", "2 hours", "end of day"]),
            metric=np.random.choice(["95%", "90%", "85%"]),
            quantity=np.random.choice(["1000", "5000", "10000"]),
            parameter=np.random.choice(["speed", "accuracy", "efficiency"]),
            issue=np.random.choice(["error", "delay", "failure"])
        )
        
        sub_goals = [
            f"Sub-goal {i}: {np.random.choice(['validate', 'process', 'analyze', 'report'])} data"
            for i in range(np.random.randint(2, 5))
        ]
        
        return GoalState(
            primary_goal=primary_goal,
            sub_goals=sub_goals,
            progress=0.0,
            timestamp=datetime.now()
        )
    
    def _evolve_goal_state(self, base_goals: GoalState, progress_ratio: float) -> GoalState:
        """Evolve goal state with progress."""
        return GoalState(
            primary_goal=base_goals.primary_goal,
            sub_goals=base_goals.sub_goals,
            progress=min(progress_ratio + np.random.uniform(-0.1, 0.1), 1.0),
            timestamp=datetime.now()
        )
    
    def generate_execution_sequences(
        self, 
        count: int = 20,
        pattern_type: str = "mixed"
    ) -> List[ExecutionSequence]:
        """Generate realistic execution sequences."""
        sequences = []
        
        # Define action patterns for different types
        if pattern_type == "data_processing":
            action_patterns = [
                ["load_data", "validate", "clean", "transform", "analyze", "save"],
                ["load_data", "validate", "process", "aggregate", "report"],
                ["load_data", "clean", "analyze", "visualize", "export"]
            ]
        elif pattern_type == "web_scraping":
            action_patterns = [
                ["connect", "authenticate", "navigate", "extract", "validate", "store"],
                ["connect", "scrape", "parse", "clean", "save"],
                ["setup", "crawl", "extract", "process", "export"]
            ]
        elif pattern_type == "api_integration":
            action_patterns = [
                ["authenticate", "request", "parse", "validate", "store"],
                ["connect", "sync", "transform", "update"],
                ["auth", "fetch", "process", "respond"]
            ]
        else:  # mixed
            action_patterns = [
                ["initialize", "process", "validate", "complete"],
                ["start", "analyze", "decide", "execute", "finish"],
                ["setup", "run", "check", "cleanup"]
            ]
        
        for i in range(count):
            pattern = np.random.choice(action_patterns)
            
            # Add some variation to the pattern
            if np.random.random() < 0.3:  # 30% chance to add extra steps
                extra_actions = ["debug", "retry", "optimize", "log", "backup"]
                pattern.extend(np.random.choice(extra_actions, np.random.randint(1, 3)))
            
            # Create steps
            steps = []
            start_time = datetime.now() - timedelta(minutes=np.random.randint(10, 120))
            current_time = start_time
            
            success_probability = 0.9  # Base success probability
            overall_success = True
            
            for j, action in enumerate(pattern):
                # Determine step success
                step_success = np.random.random() < success_probability
                if not step_success:
                    overall_success = False
                    success_probability = 0.7  # Reduce future success probability after failure
                
                duration = np.random.exponential(30)  # Average 30 seconds per step
                
                steps.append({
                    "action": action,
                    "timestamp": current_time,
                    "success": step_success,
                    "duration": duration,
                    "metadata": {
                        "step_number": j + 1,
                        "complexity": np.random.uniform(0.1, 1.0),
                        "resource_usage": np.random.uniform(0.2, 0.8)
                    }
                })
                
                current_time += timedelta(seconds=duration)
            
            sequence = ExecutionSequence(
                sequence_id=f"seq_{pattern_type}_{i:04d}",
                agent_id=f"agent_{i % 10}",  # Distribute across 10 agents
                steps=steps,
                start_time=start_time,
                end_time=current_time,
                success=overall_success,
                error_message=f"Failed at step {np.random.randint(1, len(steps))}" if not overall_success else None
            )
            
            sequences.append(sequence)
        
        return sequences
    
    def generate_causal_relationships(
        self, 
        count: int = 30,
        include_spurious: bool = True
    ) -> List[CausalRelationship]:
        """Generate realistic causal relationships."""
        relationships = []
        
        # Define known causal patterns
        causal_patterns = [
            ("data_validation_failure", "analysis_error", 0.85, 50),
            ("network_timeout", "api_failure", 0.92, 100),
            ("memory_pressure", "performance_degradation", 0.78, 200),
            ("authentication_error", "access_denied", 0.95, 10),
            ("invalid_input", "processing_error", 0.88, 30),
            ("high_load", "response_delay", 0.82, 150),
            ("data_corruption", "validation_failure", 0.90, 25),
            ("resource_exhaustion", "system_failure", 0.87, 300)
        ]
        
        # Generate true causal relationships
        true_count = int(count * 0.7) if include_spurious else count
        
        for i in range(true_count):
            cause, effect, base_strength, base_delay = np.random.choice(causal_patterns)
            
            # Add some variation
            strength = np.clip(base_strength + np.random.normal(0, 0.05), 0.5, 0.99)
            delay = max(base_delay + np.random.normal(0, base_delay * 0.2), 5)
            confidence = np.clip(strength + np.random.normal(0, 0.03), 0.6, 0.99)
            
            relationship = CausalRelationship(
                cause_event=cause,
                effect_event=effect,
                strength=strength,
                confidence=confidence,
                delay_ms=int(delay),
                evidence=[
                    f"Temporal correlation observed in {np.random.randint(50, 200)} cases",
                    f"Statistical significance: p < {np.random.uniform(0.001, 0.05):.3f}",
                    "Mechanism validated through controlled experiments"
                ],
                statistical_significance=np.random.uniform(0.95, 0.99),
                causal_mechanism=f"Causal mechanism: {cause} directly impacts {effect} through system dependencies"
            )
            
            relationships.append(relationship)
        
        # Generate spurious relationships if requested
        if include_spurious:
            spurious_count = count - true_count
            
            for i in range(spurious_count):
                # Create non-causal relationships
                random_events = [
                    "background_task", "scheduled_maintenance", "user_login", "cache_refresh",
                    "log_rotation", "backup_process", "monitoring_check", "health_probe"
                ]
                
                cause = np.random.choice(random_events)
                effect = np.random.choice(random_events)
                
                # Spurious relationships have low strength and confidence
                strength = np.random.uniform(0.1, 0.4)
                confidence = np.random.uniform(0.3, 0.6)
                delay = np.random.randint(1000, 10000)  # Random delay
                
                relationship = CausalRelationship(
                    cause_event=cause,
                    effect_event=effect,
                    strength=strength,
                    confidence=confidence,
                    delay_ms=delay,
                    evidence=[
                        f"Weak correlation observed in {np.random.randint(10, 50)} cases",
                        "No clear causal mechanism identified"
                    ],
                    statistical_significance=np.random.uniform(0.5, 0.8),
                    causal_mechanism=None
                )
                
                relationships.append(relationship)
        
        return relationships
    
    def generate_prediction_results(
        self, 
        agent_id: str, 
        count: int = 25,
        accuracy_level: str = "high"
    ) -> List[PredictionResult]:
        """Generate realistic prediction results."""
        results = []
        
        # Define accuracy levels
        if accuracy_level == "high":
            base_confidence = 0.85
            confidence_variance = 0.1
        elif accuracy_level == "medium":
            base_confidence = 0.70
            confidence_variance = 0.15
        else:  # low
            base_confidence = 0.55
            confidence_variance = 0.20
        
        outcomes = ["success", "failure", "partial_success"]
        
        for i in range(count):
            outcome = np.random.choice(outcomes, p=[0.6, 0.25, 0.15])
            
            confidence = np.clip(
                base_confidence + np.random.normal(0, confidence_variance),
                0.3, 0.95
            )
            
            # Generate probability distribution
            if outcome == "success":
                success_prob = confidence
                failure_prob = (1 - confidence) * 0.7
                partial_prob = (1 - confidence) * 0.3
            elif outcome == "failure":
                failure_prob = confidence
                success_prob = (1 - confidence) * 0.4
                partial_prob = (1 - confidence) * 0.6
            else:  # partial_success
                partial_prob = confidence
                success_prob = (1 - confidence) * 0.6
                failure_prob = (1 - confidence) * 0.4
            
            # Normalize probabilities
            total = success_prob + failure_prob + partial_prob
            probability_distribution = {
                "success": success_prob / total,
                "failure": failure_prob / total,
                "partial_success": partial_prob / total
            }
            
            # Generate risk factors and recommendations
            risk_factors = np.random.choice([
                "data_quality", "time_constraint", "resource_limitation", 
                "complexity", "external_dependency", "network_reliability"
            ], np.random.randint(1, 4)).tolist()
            
            recommendations = np.random.choice([
                "increase_timeout", "validate_data_first", "add_retry_logic",
                "allocate_more_resources", "simplify_approach", "add_monitoring"
            ], np.random.randint(1, 3)).tolist()
            
            result = PredictionResult(
                prediction_id=f"pred_{agent_id}_{i:04d}",
                agent_id=agent_id,
                predicted_outcome=outcome,
                confidence=confidence,
                probability_distribution=probability_distribution,
                risk_factors=risk_factors,
                recommended_actions=recommendations,
                timestamp=datetime.now() - timedelta(minutes=np.random.randint(0, 1440)),
                model_version=f"v{np.random.randint(1, 5)}.{np.random.randint(0, 10)}.{np.random.randint(0, 10)}"
            )
            
            results.append(result)
        
        return results
    
    def generate_time_series_data(
        self, 
        metric_name: str,
        duration_hours: int = 24,
        frequency_minutes: int = 5,
        trend: str = "stable",
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """Generate realistic time series data for performance metrics."""
        
        # Calculate number of data points
        points = (duration_hours * 60) // frequency_minutes
        
        # Generate time index
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=duration_hours)
        time_index = pd.date_range(start=start_time, end=end_time, periods=points)
        
        # Generate base values based on metric type
        if "response_time" in metric_name.lower():
            base_value = 200  # milliseconds
            seasonal_amplitude = 50
        elif "throughput" in metric_name.lower():
            base_value = 1000  # requests per second
            seasonal_amplitude = 200
        elif "error_rate" in metric_name.lower():
            base_value = 0.02  # 2% error rate
            seasonal_amplitude = 0.01
        elif "cpu" in metric_name.lower() or "memory" in metric_name.lower():
            base_value = 0.6  # 60% utilization
            seasonal_amplitude = 0.2
        else:
            base_value = 100
            seasonal_amplitude = 20
        
        # Generate trend component
        if trend == "increasing":
            trend_component = np.linspace(0, base_value * 0.3, points)
        elif trend == "decreasing":
            trend_component = np.linspace(0, -base_value * 0.3, points)
        else:  # stable
            trend_component = np.zeros(points)
        
        # Generate seasonal component (daily pattern)
        hours = np.array([(start_time + timedelta(minutes=i * frequency_minutes)).hour for i in range(points)])
        seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * hours / 24)
        
        # Generate noise
        noise = np.random.normal(0, base_value * noise_level, points)
        
        # Combine components
        values = base_value + trend_component + seasonal_component + noise
        
        # Ensure non-negative values for certain metrics
        if metric_name.lower() in ["response_time", "throughput", "cpu_usage", "memory_usage"]:
            values = np.maximum(values, 0)
        
        # Ensure percentage metrics stay within bounds
        if "rate" in metric_name.lower() or "usage" in metric_name.lower():
            values = np.clip(values, 0, 1)
        
        return pd.DataFrame({
            "timestamp": time_index,
            metric_name: values
        })
    
    def generate_test_scenario_data(self, scenario_name: str) -> Dict[str, Any]:
        """Generate comprehensive test data for a specific scenario."""
        
        agent_id = f"test_agent_{scenario_name}"
        
        # Generate all related data types
        events = self.generate_agent_events(agent_id, count=50, scenario=scenario_name)
        epistemic_states = self.generate_epistemic_states(agent_id, count=20)
        execution_sequences = self.generate_execution_sequences(count=10, pattern_type=scenario_name)
        causal_relationships = self.generate_causal_relationships(count=15)
        predictions = self.generate_prediction_results(agent_id, count=10)
        
        # Generate performance metrics
        metrics = {}
        for metric in ["response_time", "throughput", "error_rate", "cpu_usage"]:
            metrics[metric] = self.generate_time_series_data(metric)
        
        return {
            "scenario": scenario_name,
            "agent_id": agent_id,
            "events": events,
            "epistemic_states": epistemic_states,
            "execution_sequences": execution_sequences,
            "causal_relationships": causal_relationships,
            "predictions": predictions,
            "metrics": metrics,
            "metadata": {
                "generated_at": datetime.now(),
                "data_points": {
                    "events": len(events),
                    "states": len(epistemic_states),
                    "sequences": len(execution_sequences),
                    "causal_links": len(causal_relationships),
                    "predictions": len(predictions)
                }
            }
        }
    
    def save_test_data(self, data: Dict[str, Any], filepath: str):
        """Save generated test data to file."""
        # Convert datetime objects to strings for JSON serialization
        serializable_data = self._make_json_serializable(data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (EpistemicState, BehavioralPattern, CausalRelationship, PredictionResult, AgentEvent)):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


# Convenience functions for common test data generation
def generate_test_agent_data(agent_id: str, scenario: str = "mixed") -> Dict[str, Any]:
    """Generate comprehensive test data for a single agent."""
    generator = TestDataGenerator(seed=42)
    return generator.generate_test_scenario_data(scenario)


def generate_multi_agent_data(num_agents: int = 5, scenarios: List[str] = None) -> Dict[str, Any]:
    """Generate test data for multiple agents."""
    if scenarios is None:
        scenarios = ["data_analysis", "web_scraping", "api_integration", "machine_learning"]
    
    generator = TestDataGenerator(seed=42)
    
    all_data = {
        "agents": {},
        "summary": {
            "num_agents": num_agents,
            "scenarios": scenarios,
            "generated_at": datetime.now()
        }
    }
    
    for i in range(num_agents):
        scenario = scenarios[i % len(scenarios)]
        agent_id = f"multi_agent_{i:02d}"
        
        all_data["agents"][agent_id] = generator.generate_test_scenario_data(scenario)
    
    return all_data


if __name__ == "__main__":
    # Example usage
    generator = TestDataGenerator(seed=42)
    
    # Generate sample data
    test_data = generator.generate_test_scenario_data("data_analysis")
    
    print(f"Generated test data for scenario: {test_data['scenario']}")
    print(f"Events: {len(test_data['events'])}")
    print(f"Epistemic states: {len(test_data['epistemic_states'])}")
    print(f"Execution sequences: {len(test_data['execution_sequences'])}")
    print(f"Causal relationships: {len(test_data['causal_relationships'])}")
    print(f"Predictions: {len(test_data['predictions'])}")