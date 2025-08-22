"""
Example demonstrating the ExplanationEngine functionality.

This example shows how to use the ExplanationEngine to generate
human-readable explanations for agent behavior, decisions, causal
relationships, and predictions.
"""

import asyncio
from datetime import datetime, timedelta

from escai_framework.core.explanation_engine import ExplanationEngine, ExplanationStyle
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


async def demonstrate_behavior_explanation():
    """Demonstrate behavior explanation generation."""
    print("=== Behavior Explanation Demo ===")
    
    # Create sample execution sequence
    steps = [
        ExecutionStep(
            step_id="step_1",
            action="analyze_problem",
            timestamp=datetime.utcnow(),
            duration_ms=2000,
            status=ExecutionStatus.SUCCESS
        ),
        ExecutionStep(
            step_id="step_2",
            action="research_solutions",
            timestamp=datetime.utcnow(),
            duration_ms=5000,
            status=ExecutionStatus.SUCCESS
        ),
        ExecutionStep(
            step_id="step_3",
            action="implement_solution",
            timestamp=datetime.utcnow(),
            duration_ms=3000,
            status=ExecutionStatus.SUCCESS
        )
    ]
    
    sequence = ExecutionSequence(
        sequence_id="demo_seq_1",
        agent_id="demo_agent",
        task_description="Solve complex problem",
        steps=steps
    )
    sequence.calculate_metrics()
    
    # Create behavioral pattern
    pattern = BehavioralPattern(
        pattern_id="demo_pattern_1",
        pattern_name="Methodical Problem Solving",
        pattern_type=PatternType.SEQUENTIAL,
        description="Agent follows systematic approach to problem solving",
        execution_sequences=[sequence],
        common_triggers=["complex problems", "high stakes"],
        failure_modes=["time pressure", "incomplete information"]
    )
    pattern.calculate_statistics()
    
    # Generate explanations
    engine = ExplanationEngine()
    
    # Simple explanation
    simple_explanation = await engine.explain_behavior([pattern], [sequence], ExplanationStyle.SIMPLE)
    print(f"Simple Explanation:\n{simple_explanation.content}\n")
    
    # Detailed explanation
    detailed_explanation = await engine.explain_behavior([pattern], [sequence], ExplanationStyle.DETAILED)
    print(f"Detailed Explanation:\n{detailed_explanation.content}\n")
    
    # Quality metrics
    metrics = await engine.get_explanation_quality_metrics(detailed_explanation)
    print(f"Quality Metrics: {metrics}\n")


async def demonstrate_decision_pathway_explanation():
    """Demonstrate decision pathway explanation generation."""
    print("=== Decision Pathway Explanation Demo ===")
    
    # Create epistemic state
    beliefs = [
        BeliefState(
            content="The problem is complex and requires careful analysis",
            belief_type=BeliefType.FACTUAL,
            confidence=0.9
        ),
        BeliefState(
            content="I have sufficient resources to solve this",
            belief_type=BeliefType.PROBABILISTIC,
            confidence=0.7
        )
    ]
    
    knowledge = KnowledgeState(
        facts=["Problem involves multiple variables", "Similar problems solved before"],
        rules=["Always analyze before acting", "Use proven methodologies"],
        confidence_score=0.8
    )
    
    goals = [
        GoalState(
            description="Solve the problem efficiently",
            status=GoalStatus.ACTIVE,
            priority=9,
            progress=0.6
        )
    ]
    
    epistemic_state = EpistemicState(
        agent_id="demo_agent",
        timestamp=datetime.utcnow(),
        belief_states=beliefs,
        knowledge_state=knowledge,
        goal_states=goals,
        confidence_level=0.8,
        uncertainty_score=0.2
    )
    
    # Create execution sequence
    steps = [
        ExecutionStep(
            step_id="decision_1",
            action="decide_to_analyze_first",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.SUCCESS
        )
    ]
    
    sequence = ExecutionSequence(
        sequence_id="decision_seq",
        agent_id="demo_agent",
        task_description="Make strategic decision",
        steps=steps
    )
    sequence.calculate_metrics()
    
    # Generate explanation
    engine = ExplanationEngine()
    explanation = await engine.explain_decision_pathway([epistemic_state], sequence, ExplanationStyle.DETAILED)
    print(f"Decision Pathway Explanation:\n{explanation.content}\n")


async def demonstrate_causal_explanation():
    """Demonstrate causal relationship explanation generation."""
    print("=== Causal Explanation Demo ===")
    
    # Create causal relationship
    cause_event = CausalEvent(
        event_id="cause_1",
        event_type="action",
        description="Agent performed thorough analysis",
        timestamp=datetime.utcnow() - timedelta(seconds=10),
        agent_id="demo_agent"
    )
    
    effect_event = CausalEvent(
        event_id="effect_1",
        event_type="outcome",
        description="Solution was highly accurate",
        timestamp=datetime.utcnow(),
        agent_id="demo_agent"
    )
    
    evidence = [
        CausalEvidence(
            evidence_type=EvidenceType.STATISTICAL,
            description="Strong correlation between analysis time and accuracy",
            strength=0.85,
            confidence=0.9,
            source="historical data analysis"
        )
    ]
    
    causal_relationship = CausalRelationship(
        relationship_id="demo_causal_1",
        cause_event=cause_event,
        effect_event=effect_event,
        causal_type=CausalType.DIRECT,
        strength=0.85,
        confidence=0.9,
        delay_ms=10000,
        evidence=evidence,
        causal_mechanism="Thorough analysis provides better understanding of problem space"
    )
    
    # Generate explanation
    engine = ExplanationEngine()
    explanation = await engine.explain_causal_relationship(causal_relationship, ExplanationStyle.DETAILED)
    print(f"Causal Explanation:\n{explanation.content}\n")


async def demonstrate_prediction_explanation():
    """Demonstrate prediction explanation generation."""
    print("=== Prediction Explanation Demo ===")
    
    # Create prediction result
    risk_factors = [
        RiskFactor(
            factor_id="risk_1",
            name="Time Constraints",
            description="Limited time available for task completion",
            impact_score=0.6,
            probability=0.8,
            category="temporal"
        ),
        RiskFactor(
            factor_id="risk_2",
            name="Resource Availability",
            description="Computational resources may be limited",
            impact_score=0.4,
            probability=0.5,
            category="infrastructure"
        )
    ]
    
    interventions = [
        Intervention(
            intervention_id="int_1",
            intervention_type=InterventionType.STRATEGY_CHANGE,
            name="Simplify Approach",
            description="Use a simpler, faster solution approach",
            expected_impact=0.7,
            implementation_cost=0.3,
            urgency=RiskLevel.MEDIUM
        )
    ]
    
    prediction = PredictionResult(
        prediction_id="demo_pred_1",
        agent_id="demo_agent",
        prediction_type=PredictionType.SUCCESS_PROBABILITY,
        predicted_value=0.75,
        confidence_score=0.8,
        risk_level=RiskLevel.MEDIUM,
        risk_factors=risk_factors,
        recommended_interventions=interventions,
        model_name="GradientBoosting",
        model_version="2.1",
        features_used=["complexity_score", "time_available", "resource_usage"]
    )
    
    # Generate explanation
    engine = ExplanationEngine()
    explanation = await engine.explain_prediction(prediction, ExplanationStyle.DETAILED)
    print(f"Prediction Explanation:\n{explanation.content}\n")


async def demonstrate_comparative_analysis():
    """Demonstrate comparative analysis between success and failure."""
    print("=== Comparative Analysis Demo ===")
    
    # Create successful sequence
    successful_steps = [
        ExecutionStep(
            step_id="s1",
            action="careful_analysis",
            timestamp=datetime.utcnow(),
            duration_ms=3000,
            status=ExecutionStatus.SUCCESS
        ),
        ExecutionStep(
            step_id="s2",
            action="methodical_implementation",
            timestamp=datetime.utcnow(),
            duration_ms=4000,
            status=ExecutionStatus.SUCCESS
        )
    ]
    
    successful_seq = ExecutionSequence(
        sequence_id="success_seq",
        agent_id="demo_agent",
        task_description="Successful task",
        steps=successful_steps
    )
    successful_seq.calculate_metrics()
    
    # Create failed sequence
    failed_steps = [
        ExecutionStep(
            step_id="f1",
            action="rushed_implementation",
            timestamp=datetime.utcnow(),
            duration_ms=1000,
            status=ExecutionStatus.FAILURE,
            error_message="Insufficient analysis"
        )
    ]
    
    failed_seq = ExecutionSequence(
        sequence_id="failed_seq",
        agent_id="demo_agent",
        task_description="Failed task",
        steps=failed_steps
    )
    failed_seq.calculate_metrics()
    
    # Generate comparative analysis
    engine = ExplanationEngine()
    comparison = await engine.compare_success_failure([successful_seq], [failed_seq], ExplanationStyle.DETAILED)
    print(f"Comparative Analysis:\n{comparison.content}\n")


async def main():
    """Run all demonstration examples."""
    print("ExplanationEngine Demonstration\n" + "="*50 + "\n")
    
    await demonstrate_behavior_explanation()
    await demonstrate_decision_pathway_explanation()
    await demonstrate_causal_explanation()
    await demonstrate_prediction_explanation()
    await demonstrate_comparative_analysis()
    
    print("="*50)
    print("Demonstration completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())