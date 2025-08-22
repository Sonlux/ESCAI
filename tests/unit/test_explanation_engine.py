"""
Unit tests for the ExplanationEngine class.

Tests explanation generation for behavior summaries, decision pathways,
causal relationships, predictions, and comparative analysis.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from escai_framework.core.explanation_engine import (
    ExplanationEngine, ExplanationType, ExplanationStyle, ExplanationResult
)
from escai_framework.models.epistemic_state import (
    EpistemicState, BeliefState, BeliefType, KnowledgeState, GoalState, GoalStatus
)
from escai_framework.models.behavioral_pattern import (
    BehavioralPattern, PatternType, ExecutionSequence, ExecutionStep, ExecutionStatus
)
from escai_framework.models.causal_relationship import (
    CausalRelationship, CausalEvent, CausalEvidence, CausalType, EvidenceType
)
from escai_framework.models.prediction_result import (
    PredictionResult, PredictionType, RiskLevel, RiskFactor, Intervention, InterventionType
)


# Module-level fixtures
@pytest.fixture
def engine():
    """Create ExplanationEngine instance for testing."""
    return ExplanationEngine()

@pytest.fixture
def sample_epistemic_state():
    """Create sample epistemic state for testing."""
    beliefs = [
        BeliefState(
            content="The task requires careful analysis",
            belief_type=BeliefType.FACTUAL,
            confidence=0.8,
            evidence=["previous experience", "documentation"]
        ),
        BeliefState(
            content="Time is limited",
            belief_type=BeliefType.TEMPORAL,
            confidence=0.9,
            evidence=["deadline information"]
        )
    ]
    
    knowledge = KnowledgeState(
        facts=["Task complexity is high", "Resources are available"],
        rules=["Always validate inputs", "Check for edge cases"],
        confidence_score=0.7
    )
    
    goals = [
        GoalState(
            description="Complete task successfully",
            status=GoalStatus.ACTIVE,
            priority=8,
            progress=0.3
        )
    ]
    
    return EpistemicState(
        agent_id="test_agent",
        timestamp=datetime.utcnow(),
        belief_states=beliefs,
        knowledge_state=knowledge,
        goal_states=goals,
        confidence_level=0.75,
        uncertainty_score=0.25
    )

@pytest.fixture
def sample_execution_sequence():
    """Create sample execution sequence for testing."""
    steps = [
        ExecutionStep(
            step_id="step_1",
            action="analyze_requirements",
            timestamp=datetime.utcnow(),
            duration_ms=1500,
            status=ExecutionStatus.SUCCESS
        ),
        ExecutionStep(
            step_id="step_2",
            action="design_solution",
            timestamp=datetime.utcnow(),
            duration_ms=3000,
            status=ExecutionStatus.SUCCESS
        ),
        ExecutionStep(
            step_id="step_3",
            action="implement_solution",
            timestamp=datetime.utcnow(),
            duration_ms=2000,
            status=ExecutionStatus.FAILURE,
            error_message="Resource not available"
        )
    ]
    
    sequence = ExecutionSequence(
        sequence_id="seq_1",
        agent_id="test_agent",
        task_description="Complete analysis task",
        steps=steps,
        start_time=datetime.utcnow() - timedelta(seconds=10),
        end_time=datetime.utcnow()
    )
    sequence.calculate_metrics()
    return sequence

@pytest.fixture
def sample_behavioral_pattern(sample_execution_sequence):
    """Create sample behavioral pattern for testing."""
    pattern = BehavioralPattern(
        pattern_id="pattern_1",
        pattern_name="Analysis-First Approach",
        pattern_type=PatternType.SEQUENTIAL,
        description="Agent always starts with thorough analysis",
        execution_sequences=[sample_execution_sequence],
        common_triggers=["complex task", "high priority"],
        failure_modes=["resource unavailability", "time constraints"]
    )
    pattern.calculate_statistics()
    return pattern

@pytest.fixture
def sample_causal_relationship():
    """Create sample causal relationship for testing."""
    cause_event = CausalEvent(
        event_id="cause_1",
        event_type="decision",
        description="Agent decided to analyze requirements first",
        timestamp=datetime.utcnow() - timedelta(seconds=5),
        agent_id="test_agent"
    )
    
    effect_event = CausalEvent(
        event_id="effect_1",
        event_type="outcome",
        description="Solution design was more accurate",
        timestamp=datetime.utcnow(),
        agent_id="test_agent"
    )
    
    evidence = [
        CausalEvidence(
            evidence_type=EvidenceType.STATISTICAL,
            description="Strong correlation observed",
            strength=0.8,
            confidence=0.9,
            source="pattern analysis"
        )
    ]
    
    return CausalRelationship(
        relationship_id="causal_1",
        cause_event=cause_event,
        effect_event=effect_event,
        causal_type=CausalType.DIRECT,
        strength=0.8,
        confidence=0.85,
        delay_ms=5000,
        evidence=evidence,
        causal_mechanism="Thorough analysis leads to better understanding"
    )

@pytest.fixture
def sample_prediction_result():
    """Create sample prediction result for testing."""
    risk_factors = [
        RiskFactor(
            factor_id="risk_1",
            name="Resource Availability",
            description="Limited computational resources",
            impact_score=0.7,
            probability=0.6,
            category="infrastructure"
        )
    ]
    
    interventions = [
        Intervention(
            intervention_id="int_1",
            intervention_type=InterventionType.RESOURCE_ALLOCATION,
            name="Allocate Additional Resources",
            description="Increase available computational resources",
            expected_impact=0.8,
            implementation_cost=0.5,
            urgency=RiskLevel.MEDIUM
        )
    ]
    
    return PredictionResult(
        prediction_id="pred_1",
        agent_id="test_agent",
        prediction_type=PredictionType.SUCCESS_PROBABILITY,
        predicted_value=0.75,
        confidence_score=0.8,
        risk_level=RiskLevel.MEDIUM,
        risk_factors=risk_factors,
        recommended_interventions=interventions,
        model_name="RandomForest",
        model_version="1.0",
        features_used=["execution_history", "resource_usage", "complexity_score"]
    )


class TestExplanationEngine:
    """Test cases for ExplanationEngine."""
    pass


class TestBehaviorExplanation:
    """Test behavior explanation generation."""
    
    @pytest.mark.asyncio
    async def test_explain_behavior_simple_style(self, engine, sample_behavioral_pattern, sample_execution_sequence):
        """Test simple behavior explanation generation."""
        patterns = [sample_behavioral_pattern]
        sequences = [sample_execution_sequence]
        
        result = await engine.explain_behavior(patterns, sequences, ExplanationStyle.SIMPLE)
        
        assert result.validate()
        assert result.explanation_type == ExplanationType.BEHAVIOR_SUMMARY
        assert result.style == ExplanationStyle.SIMPLE
        assert "executed" in result.content.lower()
        assert "success rate" in result.content.lower()
        assert result.confidence_score > 0
        assert result.coverage_score > 0
        assert len(result.supporting_evidence) > 0
    
    @pytest.mark.asyncio
    async def test_explain_behavior_detailed_style(self, engine, sample_behavioral_pattern, sample_execution_sequence):
        """Test detailed behavior explanation generation."""
        patterns = [sample_behavioral_pattern]
        sequences = [sample_execution_sequence]
        
        result = await engine.explain_behavior(patterns, sequences, ExplanationStyle.DETAILED)
        
        assert result.validate()
        assert result.explanation_type == ExplanationType.BEHAVIOR_SUMMARY
        assert result.style == ExplanationStyle.DETAILED
        assert "test_agent" in result.content
        assert "patterns" in result.content.lower()
        assert "failure modes" in result.content.lower()
        assert result.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_explain_behavior_empty_data(self, engine):
        """Test behavior explanation with no data."""
        result = await engine.explain_behavior([], [], ExplanationStyle.SIMPLE)
        
        assert result.validate()
        assert result.confidence_score == 0.0
        assert result.coverage_score == 0.0
        assert "insufficient data" in result.content.lower()
        assert len(result.limitations) > 0


class TestDecisionPathwayExplanation:
    """Test decision pathway explanation generation."""
    
    @pytest.mark.asyncio
    async def test_explain_decision_pathway_simple(self, engine, sample_epistemic_state, sample_execution_sequence):
        """Test simple decision pathway explanation."""
        epistemic_states = [sample_epistemic_state]
        
        result = await engine.explain_decision_pathway(epistemic_states, sample_execution_sequence, ExplanationStyle.SIMPLE)
        
        assert result.validate()
        assert result.explanation_type == ExplanationType.DECISION_PATHWAY
        assert result.style == ExplanationStyle.SIMPLE
        assert "decided" in result.content.lower()
        assert "because" in result.content.lower()
        assert result.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_explain_decision_pathway_detailed(self, engine, sample_epistemic_state, sample_execution_sequence):
        """Test detailed decision pathway explanation."""
        epistemic_states = [sample_epistemic_state]
        
        result = await engine.explain_decision_pathway(epistemic_states, sample_execution_sequence, ExplanationStyle.DETAILED)
        
        assert result.validate()
        assert result.explanation_type == ExplanationType.DECISION_PATHWAY
        assert result.style == ExplanationStyle.DETAILED
        assert "test_agent" in result.content
        assert "initial state" in result.content.lower()
        assert "confidence level" in result.content.lower()
        assert result.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_explain_decision_pathway_empty_data(self, engine):
        """Test decision pathway explanation with no data."""
        result = await engine.explain_decision_pathway([], None, ExplanationStyle.SIMPLE)
        
        assert result.validate()
        assert result.confidence_score == 0.0
        assert "insufficient data" in result.content.lower()


class TestCausalExplanation:
    """Test causal relationship explanation generation."""
    
    @pytest.mark.asyncio
    async def test_explain_causal_relationship_simple(self, engine, sample_causal_relationship):
        """Test simple causal explanation generation."""
        result = await engine.explain_causal_relationship(sample_causal_relationship, ExplanationStyle.SIMPLE)
        
        assert result.validate()
        assert result.explanation_type == ExplanationType.CAUSAL_EXPLANATION
        assert result.style == ExplanationStyle.SIMPLE
        assert "caused" in result.content.lower()
        assert "confidence" in result.content.lower()
        assert result.confidence_score == sample_causal_relationship.confidence
    
    @pytest.mark.asyncio
    async def test_explain_causal_relationship_detailed(self, engine, sample_causal_relationship):
        """Test detailed causal explanation generation."""
        result = await engine.explain_causal_relationship(sample_causal_relationship, ExplanationStyle.DETAILED)
        
        assert result.validate()
        assert result.explanation_type == ExplanationType.CAUSAL_EXPLANATION
        assert result.style == ExplanationStyle.DETAILED
        assert "causal analysis" in result.content.lower()
        assert "evidence" in result.content.lower()
        assert "mechanism" in result.content.lower()
        assert result.confidence_score == sample_causal_relationship.confidence
    
    @pytest.mark.asyncio
    async def test_explain_causal_relationship_empty_data(self, engine):
        """Test causal explanation with no data."""
        result = await engine.explain_causal_relationship(None, ExplanationStyle.SIMPLE)
        
        assert result.validate()
        assert result.confidence_score == 0.0
        assert "insufficient data" in result.content.lower()


class TestPredictionExplanation:
    """Test prediction explanation generation."""
    
    @pytest.mark.asyncio
    async def test_explain_prediction_simple(self, engine, sample_prediction_result):
        """Test simple prediction explanation generation."""
        result = await engine.explain_prediction(sample_prediction_result, ExplanationStyle.SIMPLE)
        
        assert result.validate()
        assert result.explanation_type == ExplanationType.PREDICTION_EXPLANATION
        assert result.style == ExplanationStyle.SIMPLE
        assert "chance" in result.content.lower()
        assert "risk level" in result.content.lower()
        assert result.confidence_score == sample_prediction_result.confidence_score
    
    @pytest.mark.asyncio
    async def test_explain_prediction_detailed(self, engine, sample_prediction_result):
        """Test detailed prediction explanation generation."""
        result = await engine.explain_prediction(sample_prediction_result, ExplanationStyle.DETAILED)
        
        assert result.validate()
        assert result.explanation_type == ExplanationType.PREDICTION_EXPLANATION
        assert result.style == ExplanationStyle.DETAILED
        assert "prediction analysis" in result.content.lower()
        assert "risk factors" in result.content.lower()
        assert "recommended actions" in result.content.lower()
        assert result.confidence_score == sample_prediction_result.confidence_score
    
    @pytest.mark.asyncio
    async def test_explain_prediction_empty_data(self, engine):
        """Test prediction explanation with no data."""
        result = await engine.explain_prediction(None, ExplanationStyle.SIMPLE)
        
        assert result.validate()
        assert result.confidence_score == 0.0
        assert "insufficient data" in result.content.lower()


class TestComparativeAnalysis:
    """Test comparative analysis between success and failure."""
    
    @pytest.mark.asyncio
    async def test_compare_success_failure_simple(self, engine, sample_execution_sequence):
        """Test simple comparative analysis."""
        # Create successful and failed sequences
        successful_seq = sample_execution_sequence
        successful_seq.success_rate = 0.9
        
        failed_seq = ExecutionSequence(
            sequence_id="seq_2",
            agent_id="test_agent",
            task_description="Failed task",
            steps=[
                ExecutionStep(
                    step_id="step_1",
                    action="rush_implementation",
                    timestamp=datetime.utcnow(),
                    duration_ms=500,
                    status=ExecutionStatus.FAILURE
                )
            ]
        )
        failed_seq.calculate_metrics()
        
        result = await engine.compare_success_failure([successful_seq], [failed_seq], ExplanationStyle.SIMPLE)
        
        assert result.validate()
        assert result.explanation_type == ExplanationType.COMPARATIVE_ANALYSIS
        assert result.style == ExplanationStyle.SIMPLE
        assert "successful attempts" in result.content.lower()
        assert "failed attempts" in result.content.lower()
        assert result.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_compare_success_failure_detailed(self, engine, sample_execution_sequence):
        """Test detailed comparative analysis."""
        successful_seq = sample_execution_sequence
        successful_seq.success_rate = 0.9
        
        failed_seq = ExecutionSequence(
            sequence_id="seq_2",
            agent_id="test_agent",
            task_description="Failed task",
            steps=[
                ExecutionStep(
                    step_id="step_1",
                    action="rush_implementation",
                    timestamp=datetime.utcnow(),
                    duration_ms=500,
                    status=ExecutionStatus.FAILURE
                )
            ]
        )
        failed_seq.calculate_metrics()
        
        result = await engine.compare_success_failure([successful_seq], [failed_seq], ExplanationStyle.DETAILED)
        
        assert result.validate()
        assert result.explanation_type == ExplanationType.COMPARATIVE_ANALYSIS
        assert result.style == ExplanationStyle.DETAILED
        assert "comparative analysis" in result.content.lower()
        assert "success patterns" in result.content.lower()
        assert "failure patterns" in result.content.lower()
        assert "recommendations" in result.content.lower()
    
    @pytest.mark.asyncio
    async def test_compare_success_failure_empty_data(self, engine):
        """Test comparative analysis with no data."""
        result = await engine.compare_success_failure([], [], ExplanationStyle.SIMPLE)
        
        assert result.validate()
        assert result.confidence_score == 0.0
        assert "insufficient data" in result.content.lower()


class TestExplanationQuality:
    """Test explanation quality metrics and validation."""
    
    @pytest.mark.asyncio
    async def test_explanation_quality_metrics(self, engine, sample_behavioral_pattern, sample_execution_sequence):
        """Test calculation of explanation quality metrics."""
        patterns = [sample_behavioral_pattern]
        sequences = [sample_execution_sequence]
        
        explanation = await engine.explain_behavior(patterns, sequences, ExplanationStyle.DETAILED)
        metrics = await engine.get_explanation_quality_metrics(explanation)
        
        assert "confidence" in metrics
        assert "coverage" in metrics
        assert "completeness" in metrics
        assert "clarity" in metrics
        assert "actionability" in metrics
        assert "overall_quality" in metrics
        
        # All metrics should be between 0 and 1
        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"Metric {metric_name} out of range: {value}"
    
    def test_clarity_score_calculation(self, engine):
        """Test clarity score calculation."""
        # Test with clear, moderate-length content
        clear_content = "The agent performed well. It completed the task successfully. The approach was effective."
        clarity_score = engine._calculate_clarity_score(clear_content)
        assert 0.0 <= clarity_score <= 1.0
        
        # Test with overly technical content
        technical_content = "The algorithm utilized heuristic optimization for regression-based correlation analysis."
        tech_clarity_score = engine._calculate_clarity_score(technical_content)
        assert tech_clarity_score < clarity_score  # Should be lower due to jargon
    
    def test_actionability_score_calculation(self, engine):
        """Test actionability score calculation."""
        # Test with actionable content
        actionable_content = "I recommend focusing on the analysis phase. You should consider optimizing the approach."
        actionability_score = engine._calculate_actionability_score(actionable_content)
        assert actionability_score > 0.5
        
        # Test with non-actionable content
        descriptive_content = "The system processed data and generated results."
        desc_actionability_score = engine._calculate_actionability_score(descriptive_content)
        assert desc_actionability_score < actionability_score


class TestHelperMethods:
    """Test helper methods used in explanation generation."""
    
    def test_describe_strength(self, engine):
        """Test strength description helper."""
        assert engine._describe_strength(0.9) == "very strong"
        assert engine._describe_strength(0.7) == "strong"
        assert engine._describe_strength(0.5) == "moderate"
        assert engine._describe_strength(0.3) == "weak"
        assert engine._describe_strength(0.1) == "very weak"
    
    def test_describe_delay(self, engine):
        """Test delay description helper."""
        assert "immediately" in engine._describe_delay(50)
        assert "ms later" in engine._describe_delay(500)
        assert "s later" in engine._describe_delay(5000)
        assert "min later" in engine._describe_delay(120000)
    
    def test_format_prediction_value(self, engine, sample_prediction_result):
        """Test prediction value formatting."""
        # Test probability prediction
        prob_prediction = sample_prediction_result
        prob_prediction.prediction_type = PredictionType.SUCCESS_PROBABILITY
        prob_prediction.predicted_value = 0.75
        
        formatted = engine._format_prediction_value(prob_prediction)
        assert formatted == "75"
        
        # Test time prediction
        time_prediction = sample_prediction_result
        time_prediction.prediction_type = PredictionType.COMPLETION_TIME
        time_prediction.predicted_value = 12.5
        
        formatted = engine._format_prediction_value(time_prediction)
        assert formatted == "12.5"
    
    def test_extract_primary_goal(self, engine, sample_epistemic_state):
        """Test primary goal extraction."""
        goal = engine._extract_primary_goal(sample_epistemic_state)
        assert isinstance(goal, str)
        assert len(goal) > 0
        
        # Test with empty goals
        empty_state = EpistemicState(
            agent_id="test",
            timestamp=datetime.utcnow(),
            goal_states=[]
        )
        goal = engine._extract_primary_goal(empty_state)
        assert "no clear goal" in goal.lower()


class TestTemplateSystem:
    """Test explanation template system."""
    
    def test_template_initialization(self, engine):
        """Test that templates are properly initialized."""
        assert len(engine.templates) > 0
        
        # Check that all explanation types have templates
        template_types = {template.explanation_type for template in engine.templates.values()}
        assert ExplanationType.BEHAVIOR_SUMMARY in template_types
        assert ExplanationType.DECISION_PATHWAY in template_types
        assert ExplanationType.CAUSAL_EXPLANATION in template_types
        assert ExplanationType.PREDICTION_EXPLANATION in template_types
    
    def test_template_structure(self, engine):
        """Test template structure and required fields."""
        for template in engine.templates.values():
            assert hasattr(template, 'template_id')
            assert hasattr(template, 'explanation_type')
            assert hasattr(template, 'style')
            assert hasattr(template, 'template_text')
            assert hasattr(template, 'required_fields')
            
            # Template text should contain placeholders for required fields
            for field in template.required_fields:
                assert f"{{{field}}}" in template.template_text


if __name__ == "__main__":
    pytest.main([__file__])